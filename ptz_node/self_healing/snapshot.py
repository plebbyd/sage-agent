"""Periodic, memory-bounded system/OS state snapshots.

Each snapshot is a gzipped JSON file capturing what's useful for later
"what changed over time?" debugging — *without* hoarding bytes:

  included: platform/python, sanitized env keys, pip freeze, disk/mem/load,
            git HEAD + status, resolved config, doctor summary, device catalog,
            shallow repo file inventory (path + size + mtime, no contents),
            recent run/heal pointers.
  excluded: file contents, images, .venv, the snapshots dir itself, secrets.

Retention is enforced two ways after every write: keep at most ``keep`` files
AND stay under ``max_total_mb``; oldest go first. Gzip keeps each snapshot at
a few KB, so the default budget holds weeks of history in a few MB.
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ptz_node.paths import repo_root, snapshots_dir

# env keys whose *names* hint at secrets — store "<set:N chars>" not the value
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "CREDENTIAL")
# only keep env vars relevant to this project to avoid dumping the whole shell
_ENV_PREFIXES = ("PTZ_", "MSA_", "ARGO_", "OLLAMA", "ANTHROPIC", "OPENAI",
                 "SLACK_", "REOLINK_", "VIRTUAL_ENV", "PATH", "HOME", "SHELL")
_SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".local", "node_modules"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 8.0) -> str:
    try:
        out = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True,
                             text=True, timeout=timeout, check=False)
        return (out.stdout or out.stderr or "").strip()
    except Exception as exc:
        return f"<error: {exc}>"


def _sanitized_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if not k.startswith(_ENV_PREFIXES):
            continue
        if any(h in k.upper() for h in _SECRET_HINTS):
            out[k] = f"<set:{len(v)} chars>" if v else "<empty>"
        else:
            out[k] = v[:512]
    return out


def _pip_freeze(limit: int = 400) -> list[str]:
    raw = _run([sys.executable, "-m", "pip", "freeze"], timeout=20.0)
    if raw.startswith("<error"):
        return [raw]
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    return lines[:limit]


def _git_state(root: Path) -> dict[str, Any]:
    return {
        "head": _run(["git", "rev-parse", "HEAD"], cwd=root),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
        "status": _run(["git", "status", "--porcelain"], cwd=root)[:4000],
        "last_commit": _run(["git", "log", "-1", "--oneline"], cwd=root),
    }


def _file_inventory(root: Path, max_files: int = 600) -> list[dict[str, Any]]:
    """Path + size + mtime only — never contents. Bounded by ``max_files``."""
    inv: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS
                       and not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            inv.append({
                "path": str(p.relative_to(root)),
                "size": st.st_size,
                "mtime": round(st.st_mtime, 1),
            })
            if len(inv) >= max_files:
                inv.append({"path": "<truncated>", "size": 0, "mtime": 0})
                return inv
    return inv


def _safe(fn, default):
    try:
        return fn()
    except Exception as exc:  # snapshots must never crash the caller
        return {"_error": str(exc)} if isinstance(default, dict) else default


class SystemSnapshotter:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        sh = (self.config.get("self_healing") or {}).get("snapshot") or {}
        self.keep = int(sh.get("keep", 24))
        self.max_total_mb = float(sh.get("max_total_mb", 50))
        self.interval_hours = float(sh.get("interval_hours", 12))
        self.include_pip = bool(sh.get("include_pip_freeze", True))
        self.include_inventory = bool(sh.get("include_file_inventory", True))

    # -- capture -----------------------------------------------------------

    def capture(self, *, reason: str = "scheduled") -> dict[str, Any]:
        root = repo_root()
        snap: dict[str, Any] = {
            "captured_at": _utc(),
            "epoch": time.time(),
            "reason": reason,
            "host": platform.node(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            "env": _safe(_sanitized_env, {}),
            "git": _safe(lambda: _git_state(root), {}),
        }

        # System resource readings via the gateway sensor (best-effort).
        snap["system_stats"] = _safe(self._system_stats, {})
        snap["devices"] = _safe(self._devices, [])
        snap["doctor"] = _safe(self._doctor_summary, {})
        if self.include_pip:
            snap["pip_freeze"] = _safe(self._pip_freeze, [])
        if self.include_inventory:
            snap["file_inventory"] = _safe(lambda: _file_inventory(root), [])
        snap["pointers"] = _safe(self._pointers, {})
        return snap

    def _system_stats(self) -> dict[str, Any]:
        from ptz_node.sensor_gateway import SensorGateway

        gw = SensorGateway(self.config)
        for dev in gw.list_devices():
            if dev.get("id") == "sensor:system_stats" or dev.get("kind") == "sensor":
                return json.loads(gw.read_sensor(dev["id"]))
        return {}

    def _devices(self) -> list[dict[str, Any]]:
        from ptz_node.sensor_gateway import SensorGateway

        return SensorGateway(self.config).list_devices()

    def _doctor_summary(self) -> dict[str, Any]:
        from ptz_node.debug_report import run_doctor

        rep = run_doctor(self.config)
        return {
            "ok": rep.get("ok"),
            "gateway_ok": rep.get("gateway_ok"),
            "agent_ok": rep.get("agent_ok"),
            "vision_ok": rep.get("vision_ok"),
            "detectors": rep.get("detectors"),
            "issues": rep.get("issues"),
            "warnings": rep.get("warnings"),
            "llm": rep.get("llm"),
        }

    def _pip_freeze(self) -> list[str]:
        return _pip_freeze()

    def _pointers(self) -> dict[str, Any]:
        from ptz_node.paths import debug_dir

        out: dict[str, Any] = {}
        for label, path in (("latest_run", debug_dir() / "latest_run.json"),):
            if path.is_file():
                try:
                    out[label] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return out

    # -- persistence + retention ------------------------------------------

    def write(self, *, reason: str = "scheduled") -> Path:
        snap = self.capture(reason=reason)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = snapshots_dir() / f"snapshot_{ts}.json.gz"
        data = json.dumps(snap, default=str).encode("utf-8")
        with gzip.open(path, "wb") as f:
            f.write(data)
        self._prune()
        self._write_index()
        return path

    def _snapshot_files(self) -> list[Path]:
        return sorted(snapshots_dir().glob("snapshot_*.json.gz"))

    def _prune(self) -> None:
        files = self._snapshot_files()
        # by count (keep newest)
        while len(files) > self.keep:
            files[0].unlink(missing_ok=True)
            files.pop(0)
        # by total size budget
        budget = self.max_total_mb * 1024 * 1024
        total = sum(f.stat().st_size for f in files if f.exists())
        while files and total > budget and len(files) > 1:
            victim = files.pop(0)
            try:
                total -= victim.stat().st_size
            except OSError:
                pass
            victim.unlink(missing_ok=True)

    def _write_index(self) -> None:
        files = self._snapshot_files()
        index = {
            "updated_at": _utc(),
            "count": len(files),
            "total_bytes": sum(f.stat().st_size for f in files if f.exists()),
            "keep": self.keep,
            "max_total_mb": self.max_total_mb,
            "snapshots": [
                {"file": f.name, "bytes": f.stat().st_size,
                 "mtime": round(f.stat().st_mtime, 1)}
                for f in files
            ],
        }
        (snapshots_dir() / "index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8")

    # -- read / schedule helpers ------------------------------------------

    def latest(self) -> dict[str, Any] | None:
        files = self._snapshot_files()
        if not files:
            return None
        with gzip.open(files[-1], "rb") as f:
            return json.loads(f.read().decode("utf-8"))

    def load(self, path: Path) -> dict[str, Any]:
        with gzip.open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))

    def seconds_since_last(self) -> float | None:
        files = self._snapshot_files()
        if not files:
            return None
        return time.time() - files[-1].stat().st_mtime

    def is_due(self) -> bool:
        since = self.seconds_since_last()
        if since is None:
            return True
        return since >= self.interval_hours * 3600
