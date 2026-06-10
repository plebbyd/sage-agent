"""tools/startup_checks.py — startup orchestration for the PTZ viewer.

On launch we want two things to happen:

1. **Warm detection models** — load YOLO and BioCLIP into memory, ensure
   Ollama is up and the Gemma 4 vision tag is pulled and primed. This is
   slow (model weights, GPU init) so it runs in a background thread and
   prints progress as each backend comes up.

2. **PTZ self-test** — for the Reolink backend, verify that
   ``tools/calibration.json`` exists and matches the live camera. We read
   position, perform a tiny ±1° pan jog, and re-read. If any step fails
   the calibration script is re-run automatically.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

_CAL_PATH = Path(__file__).parent / "calibration.json"


# ---------------------------------------------------------------------------
# Detection model warm-up
# ---------------------------------------------------------------------------

def _ensure_ollama_running(log) -> bool:
    """Probe Ollama; if it isn't up, try to start ``ollama serve`` in the background."""
    from tools.detectors import _ollama_probe

    if _ollama_probe():
        return True

    if shutil.which("ollama") is None:
        log("[startup] Ollama: 'ollama' binary not found on PATH; skipping Gemma 4 warm-up")
        return False

    log("[startup] Ollama: not running — launching `ollama serve` in background")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        log(f"[startup] Ollama: failed to launch — {exc}")
        return False

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if _ollama_probe():
            log("[startup] Ollama: up")
            return True
        time.sleep(0.5)
    log("[startup] Ollama: did not come up within 15s")
    return False


def _ensure_gemma_model(log) -> bool:
    """Make sure the Gemma 4 tag is pulled, then prime it with a tiny request."""
    import requests

    from tools.detectors import _normalize_ollama_host

    base = _normalize_ollama_host(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = os.environ.get("GEMMA4_OLLAMA_MODEL", "gemma4:31b")

    try:
        tags = requests.get(f"{base}/api/tags", timeout=5).json().get("models", [])
        names = {m.get("name", "") for m in tags}
    except Exception as exc:
        log(f"[startup] Ollama: could not list tags — {exc}")
        return False

    have = any(name == model or name.startswith(model + ":") or name.startswith(model) for name in names)
    if not have:
        if shutil.which("ollama") is None:
            log(f"[startup] Gemma 4: '{model}' not pulled and ollama CLI not on PATH")
            return False
        log(f"[startup] Gemma 4: pulling {model} (this can take a while)…")
        try:
            r = subprocess.run(["ollama", "pull", model], check=False)
            if r.returncode != 0:
                log(f"[startup] Gemma 4: 'ollama pull {model}' exited {r.returncode}")
                return False
        except Exception as exc:
            log(f"[startup] Gemma 4: pull failed — {exc}")
            return False

    log(f"[startup] Gemma 4: priming {model}…")
    try:
        # /api/generate with an empty prompt loads weights into VRAM without inference cost.
        requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "5m"},
            timeout=120,
        )
        log("[startup] Gemma 4: ready")
        return True
    except Exception as exc:
        log(f"[startup] Gemma 4: prime request failed — {exc}")
        return False


def _warm_yolo(log) -> None:
    try:
        from tools.detectors import _HAS_YOLO, get_detector

        if not _HAS_YOLO:
            log("[startup] YOLO: package not installed — skipping")
            return
        log("[startup] YOLO: loading…")
        get_detector("yolo")
        log("[startup] YOLO: ready")
    except Exception as exc:
        log(f"[startup] YOLO: load failed — {exc}")


def _warm_bioclip(log) -> None:
    try:
        from tools.detectors import _HAS_BIOCLIP, get_detector

        if not _HAS_BIOCLIP:
            log("[startup] BioCLIP: package not installed — skipping")
            return
        log("[startup] BioCLIP: loading…")
        get_detector("bioclip")
        log("[startup] BioCLIP: ready")
    except Exception as exc:
        log(f"[startup] BioCLIP: load failed — {exc}")


def _warm_gemma(log) -> None:
    if not _ensure_ollama_running(log):
        return
    _ensure_gemma_model(log)


def warm_detection_models(log=print) -> threading.Thread:
    """Kick off model warm-up in a daemon thread; returns the thread handle."""
    def _run():
        # YOLO + BioCLIP both touch torch; serialize them to avoid double torchvision init.
        _warm_yolo(log)
        _warm_bioclip(log)
        _warm_gemma(log)
        log("[startup] Detection models: warm-up complete")

    t = threading.Thread(target=_run, name="msa-warmup", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# PTZ self-test + auto-recalibration (Reolink only)
# ---------------------------------------------------------------------------

class _PTZHealthError(RuntimeError):
    """PTZ self-test couldn't complete; calibration should be re-run."""


async def _ptz_self_test(log) -> None:
    """Connect, read position, do a tiny ±1° pan jog, read again.

    Uses its own short-lived ReolinkCamera so this can run **before** the
    long-lived viewer worker starts. Any exception bubbles up as a
    ``_PTZHealthError``.
    """
    from tools.reolink_camera import CalibrationMismatchError, ReolinkCamera

    try:
        async with ReolinkCamera(calibration_path=_CAL_PATH) as cam:
            pos0 = await cam.get_position()
            log(f"[startup] PTZ: connected — pan={pos0['pan_deg']:.1f}°, tilt={pos0['tilt_deg']:.1f}°")

            # Pick a jog direction that is guaranteed to stay inside the range.
            pan_range = cam._pan_range_deg  # type: ignore[attr-defined]
            sign = 1.0 if pos0["pan_deg"] < pan_range / 2 else -1.0

            log("[startup] PTZ: test jog…")
            await cam.pan(sign * 1.0)
            await cam.pan(-sign * 1.0)
            pos1 = await cam.get_position()
            log(
                f"[startup] PTZ: jog OK — pan={pos1['pan_deg']:.1f}°, "
                f"tilt={pos1['tilt_deg']:.1f}°"
            )
    except CalibrationMismatchError as exc:
        raise _PTZHealthError(f"calibration mismatch: {exc}") from exc
    except Exception as exc:
        raise _PTZHealthError(str(exc)) from exc


def _run_calibration(log) -> bool:
    log("[startup] PTZ: running calibration (camera will sweep its full range)…")
    try:
        from tools.calibrate_ptz import calibrate

        asyncio.run(calibrate())
        return _CAL_PATH.exists()
    except SystemExit as exc:
        log(f"[startup] PTZ: calibration aborted — {exc}")
        return False
    except Exception as exc:
        log(f"[startup] PTZ: calibration failed — {exc}")
        return False


def verify_ptz_health(log=print) -> bool:
    """Verify Reolink calibration + responsiveness; recalibrate on failure.

    Returns True if the camera is usable after this check, False if
    re-calibration was attempted but did not produce a working state.
    """
    if not _CAL_PATH.exists():
        log("[startup] PTZ: no calibration.json — calibrating…")
        if not _run_calibration(log):
            return False

    try:
        asyncio.run(_ptz_self_test(log))
        return True
    except _PTZHealthError as exc:
        log(f"[startup] PTZ: self-test failed ({exc})")

    if not _run_calibration(log):
        return False

    try:
        asyncio.run(_ptz_self_test(log))
        return True
    except _PTZHealthError as exc:
        log(f"[startup] PTZ: still failing after recalibration — {exc}")
        return False


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_startup(*, reolink: bool, log=print) -> None:
    """Run all startup checks. Detection warm-up is async; PTZ check is sync."""
    warm_detection_models(log)
    if reolink:
        ok = verify_ptz_health(log)
        if not ok:
            log("[startup] PTZ: continuing without verified hardware (camera control may fail)")
