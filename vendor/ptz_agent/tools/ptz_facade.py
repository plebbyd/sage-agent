"""
PTZ backend selection: simulated panorama (default) or Reolink hardware.

Environment:
    MSA_PTZ_BACKEND=reolink   (or PTZ_BACKEND=reolink)
    REOLINK_IP or REOLINK_HOST, REOLINK_USER, REOLINK_PASSWORD

State file ``scratchpads/sim_ptz_state.json`` may set ``ptz_backend`` to
``sim`` or ``reolink`` when the viewer is started with ``--reolink``.
Precedence: environment variable, then state file, then ``sim``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_STATE_FILE = _PROJECT_ROOT / "scratchpads" / "sim_ptz_state.json"
_CAL_PATH = Path(__file__).parent / "calibration.json"

_preview_lock = threading.Lock()
_preview_jpeg: bytes | None = None
_reolink_cmd_queue: queue.Queue = queue.Queue()
_reolink_worker_thread: threading.Thread | None = None
_reolink_worker_start_lock = threading.Lock()


def ptz_backend_name() -> str:
    for key in ("MSA_PTZ_BACKEND", "PTZ_BACKEND"):
        v = os.environ.get(key, "").strip().lower()
        if v in ("reolink", "sim"):
            return v
    if _STATE_FILE.exists():
        try:
            st = json.loads(_STATE_FILE.read_text())
            b = str(st.get("ptz_backend", "sim")).strip().lower()
            if b in ("reolink", "sim"):
                return b
        except Exception:
            pass
    return "sim"


def is_reolink_backend() -> bool:
    return ptz_backend_name() == "reolink"


def reolink_op_timeout_s() -> float:
    try:
        return max(5.0, float(os.environ.get("REOLINK_OP_TIMEOUT", "90")))
    except ValueError:
        return 90.0


def warm_reolink_worker() -> None:
    """Start the background Reolink worker (one TCP session + live JPEG ring)."""
    if is_reolink_backend():
        _ensure_reolink_worker()


def get_reolink_preview_jpeg() -> bytes | None:
    """Latest live JPEG from the worker, or None before the first frame."""
    if not is_reolink_backend():
        return None
    _ensure_reolink_worker()
    with _preview_lock:
        return _preview_jpeg


def _ensure_reolink_worker() -> None:
    global _reolink_worker_thread
    with _reolink_worker_start_lock:
        if _reolink_worker_thread is not None and _reolink_worker_thread.is_alive():
            return
        t = threading.Thread(
            target=_reolink_worker_thread_main,
            name="reolink-ptz-worker",
            daemon=True,
        )
        _reolink_worker_thread = t
        t.start()


def _reolink_worker_thread_main() -> None:
    try:
        asyncio.run(_reolink_worker_async_loop())
    except Exception:
        logger.exception("Reolink PTZ worker thread exited")


def _drain_queue() -> list[dict]:
    items: list[dict] = []
    while True:
        try:
            items.append(_reolink_cmd_queue.get_nowait())
        except queue.Empty:
            return items


def _coalesce_items(items: list[dict]) -> list[dict]:
    """Collapse bursts of user input into one motion.

    - Consecutive ``move_to`` requests: only the **last** target matters; earlier
      ones are cancelled (their waiter is released with the same result).
    - Consecutive ``jog`` requests: deltas are summed so the camera makes one
      combined move instead of multiple overshoot-prone tiny ones.
    - ``snapshot_pil`` / ``get_position`` pass through untouched.

    Cancelled items still get their event set so HTTP waiters do not hang — they
    are linked to the surviving op via ``_linked`` and share its box on success.
    """
    out: list[dict] = []
    last_move: dict | None = None
    last_jog: dict | None = None

    def flush_motion():
        nonlocal last_move, last_jog
        if last_move is not None:
            out.append(last_move)
            last_move = None
        if last_jog is not None:
            out.append(last_jog)
            last_jog = None

    for item in items:
        op = item["op"]
        if op == "move_to":
            if last_jog is not None:
                out.append(last_jog)
                last_jog = None
            if last_move is not None:
                last_move.setdefault("_linked", []).append(item)
            last_move = item
        elif op == "jog":
            if last_move is not None:
                out.append(last_move)
                last_move = None
            if last_jog is None:
                last_jog = item
            else:
                last_jog["dp"] = float(last_jog.get("dp") or 0.0) + float(item.get("dp") or 0.0)
                last_jog["dt"] = float(last_jog.get("dt") or 0.0) + float(item.get("dt") or 0.0)
                last_jog.setdefault("_linked", []).append(item)
        else:
            flush_motion()
            out.append(item)
    flush_motion()
    return out


async def _reolink_worker_async_loop() -> None:
    global _preview_jpeg
    from tools.reolink_camera import ReolinkCamera

    async with ReolinkCamera(calibration_path=_CAL_PATH) as cam:
        while True:
            # Gentle debounce so a burst of slider POSTs arrives as one batch.
            first_item: dict | None = None
            try:
                first_item = _reolink_cmd_queue.get(timeout=0.08)
            except queue.Empty:
                first_item = None

            had_cmd = False
            if first_item is not None:
                await asyncio.sleep(0.05)  # collect followups
                items = [first_item, *_drain_queue()]
                had_cmd = True
                for item in _coalesce_items(items):
                    await _reolink_dispatch_cam(cam, item)

            # Main-stream snapshot right after PTZ can yield bogus GetPtzCurPos on some firmware.
            if had_cmd:
                await asyncio.sleep(0.2)
            try:
                pil = await cam.snapshot_pil()
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=88)
                data = buf.getvalue()
                with _preview_lock:
                    _preview_jpeg = data
            except Exception as exc:
                logger.debug("reolink preview snapshot: %s", exc)
            if not had_cmd:
                await asyncio.sleep(0.04)


async def _reolink_dispatch_cam(cam, item: dict) -> None:
    box: dict[str, Any] = item["_box"]
    op = item["op"]
    try:
        if op == "move_to":
            box["value"] = await cam.move_to(float(item["pan"]), float(item["tilt"]))
        elif op == "get_position":
            box["value"] = await cam.get_position()
        elif op == "snapshot_pil":
            box["value"] = await cam.snapshot_pil()
        elif op == "jog":
            dp = float(item.get("dp") or 0.0)
            dt = float(item.get("dt") or 0.0)
            if dp:
                await cam.pan(dp)
            if dt:
                await cam.tilt(dt)
            box["value"] = await cam.get_position()
        else:
            raise RuntimeError(f"unknown reolink op {op!r}")
    except Exception as exc:
        box["error"] = exc
        logger.warning("reolink worker op %s failed: %s", op, exc)
    finally:
        item["_ev"].set()
        # Any requests that were coalesced into this one wake up with the same result.
        for linked in item.get("_linked", ()):
            if "value" in box:
                linked["_box"]["value"] = box["value"]
            if "error" in box:
                linked["_box"]["error"] = box["error"]
            linked["_ev"].set()


def _reolink_submit_op(op: str, timeout: float, **fields: Any) -> Any:
    if not is_reolink_backend():
        raise RuntimeError("Reolink backend not active")
    _ensure_reolink_worker()
    ev = threading.Event()
    box: dict[str, Any] = {}
    _reolink_cmd_queue.put({"op": op, "_ev": ev, "_box": box, **fields})
    if not ev.wait(timeout=timeout):
        raise RuntimeError(
            f"Reolink operation {op!r} timed out after {timeout:.0f}s "
            f"(camera at {os.environ.get('REOLINK_IP') or os.environ.get('REOLINK_HOST', '')!r})"
        )
    err = box.get("error")
    if err is not None:
        raise RuntimeError(str(err)) from err
    return box.get("value")


class ReolinkPTZSync:
    """Sync facade matching SimulatedPTZ patterns for missions, viewer, and tools."""

    def __init__(self):
        try:
            from PIL import Image

            _ = Image
        except ImportError as exc:
            raise RuntimeError("Pillow required for Reolink PTZ") from exc

        with open(_CAL_PATH) as f:
            cal = json.load(f)
        self.PAN_RANGE = float(cal["pan_degrees"])
        self.tilt_range = float(cal["tilt_degrees"])

        self.pan, self.tilt, self.fov_h = self._load_state_tuple()
        self.fov_v = round(self.fov_h * 9.0 / 16.0, 1)

        host = (
            os.environ.get("REOLINK_IP", "").strip()
            or os.environ.get("REOLINK_HOST", "").strip()
        )
        if not host:
            raise RuntimeError(
                "Reolink backend requires REOLINK_IP (or REOLINK_HOST) in the environment"
            )
        if not os.environ.get("REOLINK_PASSWORD", "").strip():
            raise RuntimeError(
                "Reolink backend requires REOLINK_PASSWORD in the environment"
            )

    def _load_state_tuple(self) -> tuple[float, float, float]:
        pan_d = self.PAN_RANGE / 2.0
        tilt_d = self.tilt_range / 2.0
        fov = 60.0
        if _STATE_FILE.exists():
            try:
                d = json.loads(_STATE_FILE.read_text())
                pan_d = float(d.get("pan", pan_d))
                tilt_d = float(d.get("tilt", tilt_d))
                fov = float(d.get("fov_h", fov))
            except Exception:
                pass
        pan_d = max(0.0, min(self.PAN_RANGE, pan_d))
        tilt_d = max(0.0, min(self.tilt_range, tilt_d))
        return pan_d, tilt_d, max(10.0, min(120.0, fov))

    def _persist(self) -> None:
        try:
            from tools.sim_ptz_watch import save_position_state

            save_position_state(self.pan, self.tilt, self.fov_h)
        except ImportError:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            base: dict = {}
            if _STATE_FILE.exists():
                try:
                    base = json.loads(_STATE_FILE.read_text())
                except Exception:
                    pass
            base["pan"] = round(float(self.pan), 2)
            base["tilt"] = round(float(self.tilt), 2)
            base["fov_h"] = round(float(self.fov_h), 1)
            base["ptz_backend"] = "reolink"
            _STATE_FILE.write_text(json.dumps(base))

    def refresh_from_device(self) -> dict:
        r = _reolink_submit_op("get_position", 30.0)
        self.pan = float(r["pan_deg"])
        self.tilt = float(r["tilt_deg"])
        self._persist()
        return self.get_position()

    def move_to(self, pan: float, tilt: float) -> dict:
        try:
            from tools.sim_ptz_watch import sleep_after_move
        except ImportError:

            def sleep_after_move():
                pass

        pan = max(0.0, min(self.PAN_RANGE, float(pan)))
        tilt = max(0.0, min(self.tilt_range, float(tilt)))

        r = _reolink_submit_op("move_to", reolink_op_timeout_s(), pan=pan, tilt=tilt)
        self.pan = float(r["pan_deg"])
        self.tilt = float(r["tilt_deg"])
        self._persist()
        sleep_after_move()
        return self.get_position()

    def jog(self, dp: float, dt: float) -> dict:
        """Move relative to the **camera's** current position (best for arrow nudges)."""
        try:
            from tools.sim_ptz_watch import sleep_after_move
        except ImportError:

            def sleep_after_move():
                pass

        dp = float(dp)
        dt = float(dt)
        r = _reolink_submit_op("jog", reolink_op_timeout_s(), dp=dp, dt=dt)
        self.pan = float(r["pan_deg"])
        self.tilt = float(r["tilt_deg"])
        self._persist()
        sleep_after_move()
        return self.get_position()

    def pan_by(self, degrees: float) -> dict:
        return self.jog(float(degrees), 0.0)

    def tilt_by(self, degrees: float) -> dict:
        return self.jog(0.0, float(degrees))

    def set_fov_h(self, fov_h: float) -> dict:
        self.fov_h = max(10.0, min(120.0, float(fov_h)))
        self.fov_v = round(self.fov_h * 9.0 / 16.0, 1)
        self.tilt = max(0.0, min(self.tilt_range, self.tilt))
        self._persist()
        try:
            from tools.sim_ptz_watch import sleep_after_move

            sleep_after_move()
        except ImportError:
            pass
        return self.get_position()

    def get_position(self) -> dict:
        return {
            "pan_deg": round(self.pan, 1),
            "tilt_deg": round(self.tilt, 1),
            "fov_h": self.fov_h,
            "fov_v": self.fov_v,
            "pan_range": self.PAN_RANGE,
            "tilt_range": self.tilt_range,
            "backend": "reolink",
        }

    def _crop_viewport(self):
        return _reolink_submit_op(
            "snapshot_pil", min(60.0, reolink_op_timeout_s())
        )

    def snapshot(self, filename: str | None = None) -> str:
        vp = self._crop_viewport()
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reolink_snapshot_{ts}.jpg"
        out = (_PROJECT_ROOT / filename).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        vp.save(str(out), quality=92)
        return str(out)

    def overview(self, filename: str | None, scale: float = 0.3) -> str:
        from PIL import Image, ImageDraw, ImageFont

        vp = self._crop_viewport()
        tw = max(320, int(vp.width * scale))
        th = max(180, int(vp.height * scale))
        thumb = vp.resize((tw, th), Image.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        label = (
            f"Reolink live  pan {self.pan:.1f}  tilt {self.tilt:.1f}  "
            f"FOV {self.fov_h:.0f}x{self.fov_v:.0f}"
        )
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except OSError:
                font = ImageFont.load_default()
        draw.text((8, th - 28), label, fill="white", font=font)
        draw.text((6, th - 30), label, fill="black", font=font)
        if filename is None:
            filename = "reolink_overview.jpg"
        out = (_PROJECT_ROOT / filename).resolve()
        thumb.save(str(out), quality=90)
        return str(out)

    def composite(self, filename: str | None = None) -> str:
        from PIL import Image

        vp = self._crop_viewport()
        if filename is None:
            filename = "reolink_composite.jpg"
        out = (_PROJECT_ROOT / filename).resolve()
        vp.save(str(out), quality=92)
        return str(out)


def get_ptz_camera():
    """Return SimulatedPTZ or ReolinkPTZSync based on backend configuration."""
    if is_reolink_backend():
        return ReolinkPTZSync()
    from tools.sim_ptz_tool import SimulatedPTZ

    return SimulatedPTZ()
