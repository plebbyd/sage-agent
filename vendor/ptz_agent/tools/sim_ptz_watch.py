"""
Watch-along mode for the simulated PTZ — optional pauses after moves and inference
so a human can follow the camera and detections in the web UI.

Resolution order (later overrides earlier):
  1. defaults
  2. config/config.yaml  ``sim_ptz:``
  3. scratchpads/sim_ptz_state.json  (viewer / live toggle)
  4. environment: MSA_PTZ_WATCH_ALONG, MSA_PTZ_MOVE_DELAY, MSA_PTZ_INFERENCE_DELAY
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_STATE_FILE = _PROJECT_ROOT / "scratchpads" / "sim_ptz_state.json"
_CONFIG_FILE = _PROJECT_ROOT / "config" / "config.yaml"


def _read_state_dict() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _yaml_sim_ptz() -> dict:
    try:
        import yaml

        if _CONFIG_FILE.exists():
            with open(_CONFIG_FILE) as f:
                cfg = yaml.safe_load(f) or {}
            sp = cfg.get("sim_ptz")
            return sp if isinstance(sp, dict) else {}
    except Exception:
        pass
    return {}


def get_watch_settings() -> dict:
    """Return ``{enabled, move_delay_seconds, inference_delay_seconds}``."""
    y = _yaml_sim_ptz()
    st = _read_state_dict()

    enabled = bool(y.get("watch_along", False))
    move_s = float(y.get("move_delay_seconds", 0.45))
    inf_s = float(y.get("inference_delay_seconds", 0.35))

    if "watch_along" in st:
        enabled = bool(st["watch_along"])
    if st.get("move_delay_seconds") is not None:
        move_s = float(st["move_delay_seconds"])
    if st.get("inference_delay_seconds") is not None:
        inf_s = float(st["inference_delay_seconds"])

    ev = os.environ.get("MSA_PTZ_WATCH_ALONG", "").strip()
    if ev:
        enabled = ev.lower() in ("1", "true", "yes", "on")
    md = os.environ.get("MSA_PTZ_MOVE_DELAY", "").strip()
    if md:
        move_s = float(md)
    inf = os.environ.get("MSA_PTZ_INFERENCE_DELAY", "").strip()
    if inf:
        inf_s = float(inf)

    return {
        "enabled": enabled,
        "move_delay_seconds": max(0.0, move_s),
        "inference_delay_seconds": max(0.0, inf_s),
    }


def save_position_state(
    pan: float,
    tilt: float,
    fov_h: float | None = None,
) -> None:
    """Write pan/tilt/fov to state file, preserving ``watch_*`` keys."""
    base = _read_state_dict()
    base["pan"] = round(float(pan), 2)
    base["tilt"] = round(float(tilt), 2)
    if fov_h is not None:
        base["fov_h"] = round(float(fov_h), 1)
    elif "fov_h" not in base:
        base["fov_h"] = 60.0
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(base))


def sleep_after_move() -> None:
    s = get_watch_settings()
    if s["enabled"] and s["move_delay_seconds"] > 0:
        time.sleep(s["move_delay_seconds"])


def sleep_after_inference() -> None:
    s = get_watch_settings()
    if s["enabled"] and s["inference_delay_seconds"] > 0:
        time.sleep(s["inference_delay_seconds"])


def merge_watch_from_payload(data: dict) -> None:
    """Persist watch keys from a client payload (e.g. web UI) into state file."""
    keys = ("watch_along", "move_delay_seconds", "inference_delay_seconds")
    if not data or not any(k in data for k in keys):
        return
    base = _read_state_dict()
    if "watch_along" in data:
        base["watch_along"] = bool(data["watch_along"])
    if "move_delay_seconds" in data and data["move_delay_seconds"] is not None:
        base["move_delay_seconds"] = max(0.0, float(data["move_delay_seconds"]))
    if "inference_delay_seconds" in data and data["inference_delay_seconds"] is not None:
        base["inference_delay_seconds"] = max(
            0.0, float(data["inference_delay_seconds"])
        )
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(base))
