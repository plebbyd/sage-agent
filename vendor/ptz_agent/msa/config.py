"""
msa/config.py — Configuration loader with deep-merge defaults.
"""

import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    "model": {
        "backend": "ollama",
        "model": "gemma4:e2b",
        "max_tokens": 1024,
        "base_url": "http://127.0.0.1:11434",
        "timeout": 600,
    },
    "max_iterations": 12,
    "tools": {},
    "sensors": {},
    "supervisor": {
        "concurrency": 2,
        "poll_seconds": 1.5,
    },
    "webui": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "sim_ptz": {
        "watch_along": False,
        "move_delay_seconds": 0.45,
        "inference_delay_seconds": 0.35,
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


def load_config(path: str = "config/config.yaml") -> dict:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        _deep_merge(config, user_config)
    # Local overrides from setup_argo_proxy.sh (gitignored; holds username etc.)
    local_path = config_path.parent / "argo_proxy.local.yaml"
    if local_path.exists():
        with open(local_path) as f:
            local_cfg = yaml.safe_load(f) or {}
        _deep_merge(config, local_cfg)
    return config


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def resolve_model_config(cfg: dict) -> dict:
    """Return ``model`` dict with ``argo_proxy`` settings applied when enabled."""
    from copy import deepcopy

    model = deepcopy(cfg.get("model") or {})
    ap = cfg.get("argo_proxy") or {}
    if not ap.get("enabled") and model.get("backend") != "argo_proxy":
        return model

    model["backend"] = "argo_proxy"
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
    model.setdefault("temperature", 0.4)
    model.setdefault("timeout", 600)
    return model
