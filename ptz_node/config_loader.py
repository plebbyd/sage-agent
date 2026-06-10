"""Lightweight YAML config for gateway + LangChain ChatModel creation."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pip install pyyaml") from e


def _here_config() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(
        path
        if path
        else os.environ.get("PTZ_GRAPH_CONFIG")
        or _here_config(),
    ).expanduser()

    defaults: dict[str, Any] = {
        "ptz_agent_root": None,
        "gateway": {"default_ptz_id": "ptz_primary", "enable_msa_sensors": True},
        "model": {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://127.0.0.1:11434",
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        "agent": {"max_iterations_hint": None},
        "skills": {},
        "self_healing": {
            "enabled": False,
            "mode": "approval",          # off | notify_only | approval | auto
            "swallow_errors": False,
            "model": {},                 # strong API LLM; empty → reuse agent model
            "snapshot": {
                "interval_hours": 12,
                "keep": 24,
                "max_total_mb": 50,
                "include_pip_freeze": True,
                "include_file_inventory": True,
            },
            "notify": {
                "channels": [],          # ["slack"], ["email"], or both
                "slack_channel": "",
                "email_to": [],
            },
        },
        "argo_proxy": {
            "enabled": False,
            "username": "",
            "host": "127.0.0.1",
            "port": 44497,
            "model": "gpt-4o",
            "auto_start": True,
            "ssh_jump_host": "",
            "ssh_local_port": 44497,
            "argo_upstream": "https://apps.inside.anl.gov/argoapi",
        },
    }
    cfg = deepcopy(defaults)
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        if not isinstance(overlay, dict):
            raise ValueError("config root must be a mapping")
        def deep_merge(dst: dict, src: dict) -> dict:
            for k, v in src.items():
                if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v
            return dst
        cfg = deep_merge(cfg, overlay)
    local_path = p.parent / "argo_proxy.local.yaml"
    if local_path.is_file():
        with open(local_path, encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        if isinstance(local_cfg, dict):
            cfg = deep_merge(cfg, local_cfg)
    return cfg


_OLLAMA_DEFAULT_BASES = {"http://127.0.0.1:11434", "http://localhost:11434"}


def resolve_model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply ``argo_proxy`` settings to the ``model`` section when enabled."""
    model = deepcopy(cfg.get("model") or {})
    # The merged default base_url is Ollama-specific; don't let it leak into
    # cloud providers (openrouter/openai/anthropic) that have their own endpoints.
    if (model.get("provider") or "").lower() not in ("ollama", "") and \
            model.get("base_url") in _OLLAMA_DEFAULT_BASES:
        model.pop("base_url", None)
    ap = cfg.get("argo_proxy") or {}
    if not ap.get("enabled") and model.get("provider") != "argo_proxy":
        return model

    model["provider"] = "argo_proxy"
    if ap.get("model"):
        model["model"] = ap["model"]

    jump = (ap.get("ssh_jump_host") or "").strip()
    if jump:
        local_port = int(ap.get("ssh_local_port") or ap.get("port") or 44497)
        model["base_url"] = f"http://127.0.0.1:{local_port}/v1"
    else:
        host = ap.get("host") or "127.0.0.1"
        port = int(ap.get("port") or 44497)
        model["base_url"] = f"http://{host}:{port}/v1"

    username = (ap.get("username") or "").strip()
    if username:
        model["api_key"] = username
    model.setdefault("max_tokens", 2048)
    model.setdefault("temperature", 0.2)
    return model
