"""msa/model.py — Multi-backend model client.

Supports:
    * Anthropic (claude-*)
    * OpenAI / vLLM compatible chat completions
    * argo-proxy (OpenAI ``/v1/chat/completions`` gateway to ANL ARGO)
    * Ollama /api/chat (local models, with optional vision)

Returns a typed ``ModelResponse`` from every call so callers can record
token usage in the store. Backends that don't expose token counts return
``-1`` for the missing field; callers should treat that as "unknown" not
zero.

Two call shapes:

    client.complete(system, user, images=...)           # one-shot
    client.chat(system, messages, images=...)           # multi-turn

The chat path is what powers the REPL and the worker's ReAct loop; the
one-shot path stays for legacy code that hasn't been ported yet.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    content: str
    prompt_tokens: int = -1
    completion_tokens: int = -1
    backend: str = ""
    model: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            return -1
        return self.prompt_tokens + self.completion_tokens


Message = dict  # {"role": "system|user|assistant|tool", "content": "..."}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ollama_url(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        raw = "http://127.0.0.1:11434"
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw


def _normalize_openai_base_url(raw: str) -> str:
    """Ensure scheme and ``.../v1`` suffix for OpenAI-compatible clients."""
    u = (raw or "").strip().rstrip("/")
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def _images_to_b64_png(images: Iterable) -> list[str]:
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Pillow required for images=…") from exc

    out: list[str] = []
    for img in images:
        if img is None:
            continue
        if isinstance(img, (bytes, bytearray)):
            out.append(base64.b64encode(bytes(img)).decode("ascii"))
            continue
        if isinstance(img, str):
            out.append(img)
            continue
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ModelClient:
    def __init__(self, config: dict):
        self.backend = config.get("backend", "anthropic")
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = int(config.get("max_tokens", 1024))
        cfg_base = config.get("base_url")
        self.base_url = cfg_base
        cfg_key = config.get("api_key")
        if cfg_key:
            self.api_key = cfg_key
        elif self.backend == "argo_proxy":
            self.api_key = (
                os.environ.get("ARGO_PROXY_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", "")
            )
        elif self.backend in ("openai", "vllm"):
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if self.backend == "argo_proxy":
            env_base = os.environ.get("ARGO_PROXY_BASE_URL", "") or os.environ.get(
                "OPENAI_BASE_URL", ""
            )
            if env_base:
                self.base_url = _normalize_openai_base_url(env_base)
            elif cfg_base:
                self.base_url = _normalize_openai_base_url(str(cfg_base))
            else:
                self.base_url = "http://127.0.0.1:44497/v1"

        self.temperature = config.get("temperature")
        self.top_p = config.get("top_p")
        self.top_k = config.get("top_k")
        self.timeout = float(config.get("timeout", 600))

    # ---- one-shot (legacy) -----------------------------------------------

    def complete(
        self, system: str, user: str, images: list | None = None,
    ) -> str:
        """Single-turn helper. Returns the content string only.

        Kept for backwards compat with code paths that don't care about
        token counts. New callers should use ``chat`` and read
        ``ModelResponse.content`` / ``.prompt_tokens`` / ``.completion_tokens``.
        """
        msgs: list[Message] = [{"role": "user", "content": user}]
        return self.chat(system, msgs, images=images).content

    # ---- multi-turn (the new default) -----------------------------------

    def chat(
        self,
        system: str,
        messages: list[Message],
        images: list | None = None,
    ) -> ModelResponse:
        """Multi-turn chat completion.

        ``messages`` is the full alternating history of user/assistant
        (and ``tool``-role notes if present). ``system`` is sent as the
        system message; do not duplicate it inside ``messages``.

        Vision: ``images`` attaches to the LAST user message. Backends
        without vision support log a warning and ignore it.
        """
        if self.backend == "anthropic":
            return self._anthropic(system, messages, images)
        if self.backend in ("vllm", "openai", "argo_proxy"):
            return self._openai_compat(system, messages, images)
        if self.backend == "ollama":
            return self._ollama(system, messages, images)
        raise ValueError(f"Unknown backend: {self.backend}")

    # ---- backend implementations ----------------------------------------

    def _anthropic(
        self, system: str, messages: list[Message], images: list | None,
    ) -> ModelResponse:
        if images:
            logger.warning("Anthropic backend: ignoring %d images "
                           "(vision needs the messages-API content blocks)",
                           len(images))
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Anthropic backend requires `anthropic`") from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        # Anthropic only allows user/assistant roles; map any 'tool' to 'user'.
        ant_msgs = [
            {"role": "user" if m["role"] == "tool" else m["role"],
             "content": m["content"]}
            for m in messages
        ]
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=ant_msgs,
        )
        content = resp.content[0].text if resp.content else ""
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            content=content,
            prompt_tokens=getattr(usage, "input_tokens", -1) if usage else -1,
            completion_tokens=getattr(usage, "output_tokens", -1) if usage else -1,
            backend="anthropic",
            model=self.model,
        )

    def _openai_compat(
        self, system: str, messages: list[Message], images: list | None,
    ) -> ModelResponse:
        if images:
            logger.warning("OpenAI-compat backend: ignoring %d images", len(images))
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI-compat backend requires `openai`") from exc

        client = OpenAI(
            base_url=self.base_url or "http://localhost:8000/v1",
            api_key=self.api_key or "EMPTY",
            timeout=self.timeout,
        )
        oai_msgs = [{"role": "system", "content": system}]
        for m in messages:
            role = m["role"]
            if role == "tool":
                # vLLM/OpenAI tool-result rendering; treat as user-side context.
                oai_msgs.append({"role": "user", "content": f"[tool] {m['content']}"})
            else:
                oai_msgs.append({"role": role, "content": m["content"]})

        create_kw: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": oai_msgs,
        }
        if self.temperature is not None:
            create_kw["temperature"] = float(self.temperature)
        if self.top_p is not None:
            create_kw["top_p"] = float(self.top_p)

        resp = client.chat.completions.create(**create_kw)
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", -1) if usage else -1,
            completion_tokens=getattr(usage, "completion_tokens", -1) if usage else -1,
            backend=self.backend,
            model=self.model,
        )

    def _ollama(
        self, system: str, messages: list[Message], images: list | None,
    ) -> ModelResponse:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Ollama backend requires `requests`") from exc

        url = self._ollama_url("/api/chat")

        ol_msgs: list[dict] = [{"role": "system", "content": system}]
        for i, m in enumerate(messages):
            role = m["role"]
            if role == "tool":
                # Ollama's chat API doesn't yet support a 'tool' role
                # universally; fold tool output into the user channel.
                ol_msgs.append({"role": "user",
                                 "content": f"[tool result]\n{m['content']}"})
            else:
                ol_msgs.append({"role": role, "content": m["content"]})

        if images and ol_msgs:
            # Attach to the last user message (or append a new one if the
            # caller forgot).
            for msg in reversed(ol_msgs):
                if msg["role"] == "user":
                    msg["images"] = _images_to_b64_png(images)
                    break
            else:
                ol_msgs.append({"role": "user", "content": "",
                                "images": _images_to_b64_png(images)})

        options: dict = {"num_predict": self.max_tokens}
        if self.temperature is not None:
            options["temperature"] = float(self.temperature)
        if self.top_p is not None:
            options["top_p"] = float(self.top_p)
        if self.top_k is not None:
            options["top_k"] = int(self.top_k)

        payload = {
            "model": self.model,
            "messages": ol_msgs,
            "stream": False,
            "options": options,
            # Hold the model in VRAM between cycles on edge devices.
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
        }

        last: dict = {}
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                raise RuntimeError(f"Ollama request to {url} failed: {exc}") from exc
            try:
                data = resp.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Ollama returned non-JSON (HTTP {resp.status_code}): "
                    f"{resp.text[:500]}"
                ) from exc
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama HTTP {resp.status_code}: "
                    f"{(data.get('error') if isinstance(data, dict) else resp.text[:500])}"
                )
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"Ollama error: {data['error']}")

            content = ""
            msg = data.get("message") if isinstance(data, dict) else None
            if isinstance(msg, dict):
                content = str(msg.get("content") or "")
            elif isinstance(data, dict) and "response" in data:
                content = str(data["response"] or "")
            last = data

            if content.strip():
                return ModelResponse(
                    content=content,
                    prompt_tokens=int(data.get("prompt_eval_count", -1) or -1),
                    completion_tokens=int(data.get("eval_count", -1) or -1),
                    backend="ollama",
                    model=self.model,
                    raw={
                        "total_duration_ns": data.get("total_duration"),
                        "load_duration_ns": data.get("load_duration"),
                    },
                )
            logger.warning("Ollama returned empty content (attempt %d/2). "
                           "Retrying…", attempt + 1)

        return ModelResponse(
            content="",
            prompt_tokens=int(last.get("prompt_eval_count", -1) or -1),
            completion_tokens=int(last.get("eval_count", -1) or -1),
            backend="ollama",
            model=self.model,
        )

    # ---- url plumbing ----------------------------------------------------

    def _ollama_url(self, suffix: str) -> str:
        if self.base_url:
            base = self.base_url.rstrip("/")
            if base.endswith(("/api/generate", "/api/chat")):
                base = base.rsplit("/api/", 1)[0]
            return f"{_normalize_ollama_url(base)}{suffix}"
        return f"{_normalize_ollama_url(os.environ.get('OLLAMA_HOST', ''))}{suffix}"
