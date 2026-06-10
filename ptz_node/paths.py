"""Filesystem locations for local runs and Cursor-friendly debug artifacts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def local_data_root() -> Path:
    override = os.environ.get("PTZ_GRAPH_LOCAL_DIR", "").strip()
    base = Path(override).expanduser() if override else repo_root() / ".local"
    base.mkdir(parents=True, exist_ok=True)
    return base


def runs_dir() -> Path:
    d = local_data_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tests_dir() -> Path:
    d = local_data_root() / "tests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def debug_dir() -> Path:
    d = local_data_root() / "debug"
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshots_dir() -> Path:
    d = local_data_root() / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def heal_dir() -> Path:
    d = local_data_root() / "heal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def heal_pending_dir() -> Path:
    d = heal_dir() / "pending"
    d.mkdir(parents=True, exist_ok=True)
    return d


def heal_backups_dir() -> Path:
    d = heal_dir() / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
