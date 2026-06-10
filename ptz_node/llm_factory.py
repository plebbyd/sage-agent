"""Construct LangChain ChatModels aligned with Jetson Thor local + cloud setups."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel


def chat_model_from_config(section: dict[str, Any]) -> BaseChatModel:
    provider = (section.get("provider") or "ollama").strip().lower()
    model_name = section.get("model") or "llama3.2"

    temperature = float(section.get("temperature", 0.2))

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError(
                "Provider ollama requires langchain-ollama: pip install langchain-ollama"
            ) from e
        base = section.get("base_url") or "http://127.0.0.1:11434"
        kw: dict[str, Any] = {
            "model": model_name,
            "base_url": base,
            "temperature": temperature,
        }
        nt = section.get("num_predict")
        if nt is not None:
            kw["num_predict"] = int(nt)
        return ChatOllama(**kw)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError(
                "Provider anthropic requires langchain-anthropic"
            ) from e
        max_tokens = int(section.get("max_tokens", 4096))
        api_key = section.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError(
                "Provider openrouter requires langchain-openai: pip install langchain-openai"
            ) from e
        api_key = (
            section.get("api_key")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Provider openrouter needs an API key: set OPENROUTER_API_KEY "
                "(get one at https://openrouter.ai/keys)"
            )
        # Use a dedicated key/env so an inherited Ollama default base_url can't
        # hijack the OpenRouter endpoint.
        cfg_base = section.get("openrouter_base_url") or section.get("base_url")
        if cfg_base and "11434" in str(cfg_base):  # leaked ollama default
            cfg_base = None
        base_url = (
            cfg_base
            or os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        # OpenRouter expects fully-qualified model slugs, e.g.
        # "anthropic/claude-3.7-sonnet", "openai/gpt-4o", "google/gemini-2.0-flash-001".
        max_tokens = int(section.get("max_tokens", 4096))
        # Optional attribution headers OpenRouter recommends.
        default_headers = {
            "HTTP-Referer": section.get("referer")
            or os.environ.get("OPENROUTER_REFERER", "https://github.com/jetson-ptz-agent-graph"),
            "X-Title": section.get("title")
            or os.environ.get("OPENROUTER_TITLE", "jetson-ptz-agent-graph"),
        }
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )

    if provider in ("openai", "vllm", "openai_compat", "argo_proxy"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError(
                "Provider openai requires langchain-openai"
            ) from e
        base_url = section.get("openai_base_url") or section.get("base_url")
        if provider == "argo_proxy":
            api_key = (
                section.get("api_key")
                or os.environ.get("ARGO_PROXY_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "not-needed")
            )
            env_base = os.environ.get("ARGO_PROXY_BASE_URL") or os.environ.get(
                "OPENAI_BASE_URL"
            )
            if env_base and not base_url:
                base_url = env_base if env_base.rstrip("/").endswith("/v1") else f"{env_base.rstrip('/')}/v1"
        else:
            api_key = section.get("api_key") or os.environ.get("OPENAI_API_KEY", "not-needed")
        max_tokens = int(section.get("max_tokens", 2048))
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError(f"Unknown model.provider: {provider!r}")
