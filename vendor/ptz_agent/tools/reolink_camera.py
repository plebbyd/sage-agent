"""
reolink_camera.py — MSA tool for controlling a Reolink PTZ camera.

Loads calibration from calibration.json (project root).
Credentials via env: REOLINK_IP or REOLINK_HOST, REOLINK_USER, REOLINK_PASSWORD.
Never commit camera passwords; use environment or a secrets manager.

Designed to be called from the MSA agent as an async context manager:

    async with ReolinkCamera() as cam:
        await cam.pan(15)
        path = await cam.snapshot()

Or from the command line:

    python -m tools.reolink_camera pan 10
    python -m tools.reolink_camera tilt -5
    python -m tools.reolink_camera move_to 90 25
    python -m tools.reolink_camera get_position
    python -m tools.reolink_camera snapshot [filename]
    python -m tools.reolink_camera look_at_preset 1
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from reolink_aio.api import Host

logger = logging.getLogger(__name__)


class CalibrationMismatchError(RuntimeError):
    """Raised when Baichuan PTZ position reads fall outside the calibrated range."""

# ---------------------------------------------------------------------------
# Calibration / credential defaults
# ---------------------------------------------------------------------------

_CAL_PATH = Path(__file__).parent / "calibration.json"

def _read_camera_env() -> tuple[str, str, str]:
    """Read camera credentials from os.environ. Called at connect time so
    a tool that updates the IP (e.g. ``ptz_find_camera``) can change the
    target host without a process restart."""
    host = (
        os.environ.get("REOLINK_IP", "").strip()
        or os.environ.get("REOLINK_HOST", "").strip()
    )
    user = os.environ.get("REOLINK_USER", "admin")
    pwd  = os.environ.get("REOLINK_PASSWORD", "")
    return host, user, pwd


# Module-level constants are kept for backward compat (calibrate_ptz reads
# them) but the live values used by ReolinkCamera come from _read_camera_env
# at connect time.
REOLINK_HOST, REOLINK_USER, REOLINK_PASS = _read_camera_env()
# When the camera is mounted right-side-up (i.e. opposite its intended
# upside-down install), the live image arrives rotated 180° and the pan/tilt
# axes appear mirrored to the operator. REOLINK_FLIPPED=1 rotates the preview
# and mirrors both axes so user-facing "left" / "up" match what they see.
REOLINK_FLIPPED = os.environ.get("REOLINK_FLIPPED", "").strip().lower() in (
    "1", "true", "yes", "on"
)

POLL_INTERVAL = 0.05   # seconds between position polls
MOVE_TIMEOUT  = 30.0   # seconds before giving up on a move (covers wide-unit encoders)
NO_MOTION_TIMEOUT = 5.0  # if motors don't budge within N s of sending the command,
                          # fail fast — the camera is ignoring the request.
SETTLE_DEG    = 1.0    # consider settled within this many degrees of target
COAST_DEG     = 4.0    # send Stop this many degrees early to account for motor coast
MOVE_SPEED    = 16     # Reolink PTZ speed (1=slowest, 64=fastest); slower = less overshoot
MOVE_SLOW_SPEED = 4    # speed used for corrective nudges (small moves overshoot
                       # less when slower).
# After the main move at MOVE_SPEED, the camera typically lands within 5-15° of
# the target due to motor coast that depends on direction, distance, and lubrication
# state. A single tolerance band ends the agent's "is it close enough?" loop:
# the driver retries at MOVE_SLOW_SPEED until error <= MOVE_TOLERANCE_DEG (or it
# gives up after MAX_CORRECTION_PASSES). The user-facing degrees-API is precise
# to ~MOVE_TOLERANCE_DEG, which is plenty for the agent's "go to pan X" semantics.
MOVE_TOLERANCE_DEG    = 2.0
MAX_CORRECTION_PASSES = 3

# Some Reolink models (e.g. E1-Zoom-style PTZs, the "Porch" dome here) refuse
# the ``speed=`` kwarg with NotSupportedError. Probe once, then remember, so
# we only pay the failed call a single time per process.
_SPEED_SUPPORTED: bool | None = None


async def _ptz_command(host: Host, cmd: str, speed: int = MOVE_SPEED) -> None:
    """Drive a PTZ command, using ``speed=`` if the firmware accepts it."""
    global _SPEED_SUPPORTED
    if _SPEED_SUPPORTED is False:
        await host.set_ptz_command(0, command=cmd)
        return
    try:
        await host.set_ptz_command(0, command=cmd, speed=int(speed))
        if _SPEED_SUPPORTED is None:
            _SPEED_SUPPORTED = True
    except TypeError:
        _SPEED_SUPPORTED = False
        await host.set_ptz_command(0, command=cmd)
    except Exception as exc:
        # reolink_aio raises ``NotSupportedError`` (a plain Exception subclass
        # in older versions); fall back on any error that looks speed-related.
        if "speed" in str(exc).lower():
            _SPEED_SUPPORTED = False
            logger.info("camera rejects speed kwarg; falling back to default speed")
            await host.set_ptz_command(0, command=cmd)
        else:
            raise


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class ReolinkCamera:
    """
    Async context manager that wraps a reolink_aio Host connection and exposes
    a degree-based PTZ API calibrated from calibration.json.

    Pan:  0° = hard-left limit, 355° = hard-right limit
          positive degrees = right, negative = left
    Tilt: 0° = hard-down limit, 50° = hard-up limit
          positive degrees = up, negative = down
    """

    def __init__(self, calibration_path: Path = _CAL_PATH):
        with open(calibration_path) as f:
            cal = json.load(f)

        # Pan/tilt calibration. ``*_units_per_degree`` is signed — negative means
        # encoder units DECREASE as the axis moves in the API's positive direction
        # (right for pan, up for tilt). That flag also selects the motor command
        # to send when units need to increase vs decrease, because different camera
        # firmwares wire the axes in opposite directions.
        self._pan_min_units  = cal["pan_min"]
        self._pan_max_units  = cal["pan_max"]
        self._pan_upd        = abs(cal["pan_units_per_degree"])
        self._pan_inverted   = cal["pan_units_per_degree"] < 0

        self._tilt_min_units = cal["tilt_min"]
        self._tilt_max_units = cal["tilt_max"]
        self._tilt_upd       = abs(cal["tilt_units_per_degree"])
        self._tilt_inverted  = cal["tilt_units_per_degree"] < 0

        self._pan_range_deg  = cal["pan_degrees"]
        self._tilt_range_deg = cal["tilt_degrees"]

        # Physical-mount flip: when True, the public API speaks "user/display"
        # coordinates that are mirrored on both axes relative to the camera's
        # native frame, and snapshots are rotated 180° before being returned.
        self._flipped = REOLINK_FLIPPED

        # Unit-domain tolerances derived from the (now possibly much larger)
        # encoder scale so overshoot / settle logic stays in sensible degrees.
        self._pan_settle_units  = max(2, round(SETTLE_DEG * self._pan_upd))
        self._pan_coast_units   = max(self._pan_settle_units + 2, round(COAST_DEG * self._pan_upd))
        self._tilt_settle_units = max(2, round(SETTLE_DEG * self._tilt_upd))
        self._tilt_coast_units  = max(self._tilt_settle_units + 2, round(COAST_DEG * self._tilt_upd))

        self._host: Host | None = None

    def _clamp_pan_units(self, u: int) -> int:
        lo = min(self._pan_min_units, self._pan_max_units)
        hi = max(self._pan_min_units, self._pan_max_units)
        return max(lo, min(hi, int(u)))

    def _clamp_tilt_units(self, u: int) -> int:
        lo = min(self._tilt_min_units, self._tilt_max_units)
        hi = max(self._tilt_min_units, self._tilt_max_units)
        return max(lo, min(hi, int(u)))

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        if self._host is None:
            host, user, pwd = _read_camera_env()
            if not host:
                raise RuntimeError(
                    "REOLINK_IP is not set. Run ptz_find_camera to scan the "
                    "network for the camera, or set REOLINK_IP in ~/.msa.env."
                )
            self._host = Host(host=host, username=user, password=pwd)
            await self._host.get_host_data()

    async def disconnect(self):
        if self._host is not None:
            await self._host.logout()
            self._host = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.disconnect()

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------

    def _pan_units_to_deg(self, units: int) -> float:
        """Convert raw pan position units to degrees (0° = hard-left)."""
        u = self._clamp_pan_units(int(units))
        if self._pan_inverted:
            return (self._pan_min_units - u) / self._pan_upd
        return (u - self._pan_min_units) / self._pan_upd

    def _pan_deg_to_units(self, deg: float) -> int:
        """Convert pan degrees to raw position units."""
        deg = max(0.0, min(self._pan_range_deg, deg))
        if self._pan_inverted:
            return round(self._pan_min_units - deg * self._pan_upd)
        return round(self._pan_min_units + deg * self._pan_upd)

    def _tilt_units_to_deg(self, units: int) -> float:
        """Convert raw tilt position units to degrees (0° = hard-down)."""
        u = self._clamp_tilt_units(int(units))
        if self._tilt_inverted:
            return (self._tilt_min_units - u) / self._tilt_upd
        return (u - self._tilt_min_units) / self._tilt_upd

    def _tilt_deg_to_units(self, deg: float) -> int:
        """Convert tilt degrees to raw position units."""
        deg = max(0.0, min(self._tilt_range_deg, deg))
        if self._tilt_inverted:
            return round(self._tilt_min_units - deg * self._tilt_upd)
        return round(self._tilt_min_units + deg * self._tilt_upd)

    # ------------------------------------------------------------------
    # Low-level position fetch
    # ------------------------------------------------------------------

    async def _fetch_position(self) -> tuple[int, int]:
        """
        Read pan/tilt encoder units via Baichuan.

        Raises ``CalibrationMismatchError`` if reads consistently fall outside
        the calibrated range — that means ``tools/calibration.json`` no longer
        matches what the camera reports, and open-loop moves based on bogus
        degrees would lie about position. Rerun ``python3 -m tools.calibrate_ptz``.
        """
        last_pu, last_tu = None, None
        bad = 0
        for attempt in range(3):
            await self._host.baichuan.get_ptz_position(0)
            pu = int(self._host.ptz_pan_position(0))
            tu = int(self._host.ptz_tilt_position(0))
            last_pu, last_tu = pu, tu
            pu_c = self._clamp_pan_units(pu)
            tu_c = self._clamp_tilt_units(tu)
            if pu_c == pu and tu_c == tu:
                return pu, tu
            bad += 1
            await asyncio.sleep(0.08)
        pan_lo = min(self._pan_min_units, self._pan_max_units)
        pan_hi = max(self._pan_min_units, self._pan_max_units)
        tilt_lo = min(self._tilt_min_units, self._tilt_max_units)
        tilt_hi = max(self._tilt_min_units, self._tilt_max_units)
        raise CalibrationMismatchError(
            f"Reolink position reads out of calibrated range after {bad} tries "
            f"(pan_raw={last_pu} allowed [{pan_lo},{pan_hi}], "
            f"tilt_raw={last_tu} allowed [{tilt_lo},{tilt_hi}]). "
            "Rerun: python3 -m tools.calibrate_ptz"
        )

    # ------------------------------------------------------------------
    # Low-level move primitive
    # ------------------------------------------------------------------

    async def _move_axis(
        self,
        command_fwd: str,
        command_rev: str,
        axis: str,            # "pan" or "tilt"
        target_units: int,
        target_deg: float,
        speed: int = MOVE_SPEED,
    ) -> int:
        """
        Move one axis toward target_units.

        Sends command_fwd when we need to increase units, command_rev to
        decrease.  Polls at POLL_INTERVAL, sends Stop COAST_UNITS before the
        target, then waits for settle.  Returns final position in units.
        """
        if axis == "pan":
            settle_units = self._pan_settle_units
            coast_units = self._pan_coast_units
        else:
            settle_units = self._tilt_settle_units
            coast_units = self._tilt_coast_units

        # Coast distance scales roughly with speed (and quadratically in real
        # mechanics, but linear is a good-enough first approximation here).
        # COAST_DEG was tuned at MOVE_SPEED; at MOVE_SLOW_SPEED=4 the camera
        # coasts ~1/4 as far, so stopping 4° early leaves us 3° short. Scale
        # the coast budget by speed/MOVE_SPEED so corrective passes actually
        # reach their target.
        speed_factor = max(0.2, min(1.0, speed / max(1, MOVE_SPEED)))
        coast_units = max(settle_units + 2, int(round(coast_units * speed_factor)))

        pu, tu = await self._fetch_position()
        current_units = pu if axis == "pan" else tu
        delta = target_units - current_units

        if abs(delta) <= settle_units:
            return current_units

        # Positive delta → need to increase units → use command_fwd
        cmd = command_fwd if delta > 0 else command_rev

        # Are we aiming for a hard limit? If so, skip the coast-early logic —
        # the mechanical endstop will halt the motor and we want the full range.
        if axis == "pan":
            lo = min(self._pan_min_units, self._pan_max_units)
            hi = max(self._pan_min_units, self._pan_max_units)
        else:
            lo = min(self._tilt_min_units, self._tilt_max_units)
            hi = max(self._tilt_min_units, self._tilt_max_units)
        at_hard_limit = target_units <= lo + settle_units or target_units >= hi - settle_units

        # Stop early enough to account for motor coast, but never more than half the move.
        if at_hard_limit:
            stop_threshold = 0  # drive all the way into the endstop
        else:
            stop_threshold = max(settle_units, min(coast_units, abs(delta) // 2))

        await _ptz_command(self._host, cmd, speed=speed)
        stopped = False
        stable_count = 0        # consecutive polls with no position change after stop
        last_units = current_units
        start_units = current_units
        cmd_sent_at = time.monotonic()
        deadline = cmd_sent_at + MOVE_TIMEOUT
        no_motion_logged = False

        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            pu, tu = await self._fetch_position()
            current_units = pu if axis == "pan" else tu
            current_deg = (self._pan_units_to_deg(current_units) if axis == "pan"
                           else self._tilt_units_to_deg(current_units))

            remaining = target_units - current_units
            print(f"  {axis}: {current_deg:.1f}°  (target {target_deg:.1f}°, "
                  f"Δ={remaining:+d} units)   ", end="\r", flush=True)

            # Fail-fast: if the motor hasn't moved at all within
            # NO_MOTION_TIMEOUT seconds of sending the command, the
            # camera is ignoring us. Stop and raise so the agent gets
            # an actionable error instead of waiting 30s for a
            # silent timeout. The most common causes are a stale or
            # inverted ``calibration.json`` and a privacy/sleep mask
            # on the camera that disables PTZ.
            elapsed = time.monotonic() - cmd_sent_at
            if (not stopped and current_units == start_units
                    and elapsed >= NO_MOTION_TIMEOUT):
                print()  # finalise the carriage-return line
                try:
                    await self._host.set_ptz_command(0, command="Stop")
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"PTZ {axis} command '{cmd}' sent but the camera did "
                    f"not move (still at {start_units} units after "
                    f"{elapsed:.1f}s). Likely causes: stale calibration "
                    f"(re-run `python -m tools.calibrate_ptz`), camera in "
                    f"privacy / sleep mode, or PTZ disabled in the camera "
                    f"web UI."
                )
            if not no_motion_logged and current_units != start_units:
                no_motion_logged = True  # camera is alive; no need to fail-fast

            if not stopped:
                # When driving into a hard limit, wait for the motor to stall
                # (position stops changing) instead of stopping early.
                if at_hard_limit:
                    if current_units == last_units:
                        stable_count += 1
                    else:
                        stable_count = 0
                    if stable_count >= 3:
                        await self._host.set_ptz_command(0, command="Stop")
                        stopped = True
                        stable_count = 0
                elif abs(remaining) <= stop_threshold:
                    await self._host.set_ptz_command(0, command="Stop")
                    stopped = True
                    stable_count = 0
                elif (delta > 0 and current_units > target_units + settle_units) or \
                     (delta < 0 and current_units < target_units - settle_units):
                    await self._host.set_ptz_command(0, command="Stop")
                    stopped = True
                    stable_count = 0
            else:
                if current_units == last_units:
                    stable_count += 1
                else:
                    stable_count = 0
                if abs(remaining) <= settle_units or stable_count >= 3:
                    break

            last_units = current_units

        if not stopped:
            await self._host.set_ptz_command(0, command="Stop")

        # Brief settle wait after stop
        await asyncio.sleep(0.4)
        pu, tu = await self._fetch_position()
        return pu if axis == "pan" else tu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Coordinate transforms for upside-down mount support
    # ------------------------------------------------------------------

    def _user_to_cam_pan(self, deg: float) -> float:
        return self._pan_range_deg - deg if self._flipped else deg

    def _user_to_cam_tilt(self, deg: float) -> float:
        return self._tilt_range_deg - deg if self._flipped else deg

    def _cam_to_user_pan(self, deg: float) -> float:
        return self._pan_range_deg - deg if self._flipped else deg

    def _cam_to_user_tilt(self, deg: float) -> float:
        return self._tilt_range_deg - deg if self._flipped else deg

    async def get_position(self) -> dict:
        """
        Return the current camera position in degrees.

        Returns a dict with:
            pan_deg:  0 = hard-left limit, 355 = hard-right limit (in display coords)
            tilt_deg: 0 = hard-down limit,  50 = hard-up limit
        """
        await self.connect()
        pu, tu = await self._fetch_position()
        cam_pan  = self._pan_units_to_deg(pu)
        cam_tilt = self._tilt_units_to_deg(tu)
        return {
            "pan_deg":  round(self._cam_to_user_pan(cam_pan),  1),
            "tilt_deg": round(self._cam_to_user_tilt(cam_tilt), 1),
        }

    async def pan(self, degrees: float) -> dict:
        """Public pan (positive = right in displayed image, negative = left)."""
        # 180° flip mirrors the horizontal axis, so the user's "right" is the
        # camera's "left". Negate the relative delta.
        if self._flipped:
            degrees = -degrees
        res = await self._pan_camera(degrees)
        return {"pan_deg": round(self._cam_to_user_pan(res["pan_deg"]), 1)}

    async def tilt(self, degrees: float) -> dict:
        """Public tilt (positive = up in displayed image, negative = down)."""
        if self._flipped:
            degrees = -degrees
        res = await self._tilt_camera(degrees)
        return {"tilt_deg": round(self._cam_to_user_tilt(res["tilt_deg"]), 1)}

    async def move_to(self, pan_degrees: float, tilt_degrees: float) -> dict:
        """Absolute move using user/display coordinates."""
        cam_pan  = self._user_to_cam_pan(pan_degrees)
        cam_tilt = self._user_to_cam_tilt(tilt_degrees)
        res = await self._move_to_camera(cam_pan, cam_tilt)
        return {
            "pan_deg":  round(self._cam_to_user_pan(res["pan_deg"]), 1),
            "tilt_deg": round(self._cam_to_user_tilt(res["tilt_deg"]), 1),
        }

    async def _pan_camera(self, degrees: float,
                            speed: int = MOVE_SPEED) -> dict:
        """
        Pan the camera by the given number of degrees relative to current position.

        Positive = right, negative = left.
        Clamped so the result stays within [0°, 355°].

        Returns a dict with:
            pan_deg: new absolute pan position in degrees
        """
        await self.connect()
        pu, _tu = await self._fetch_position()
        current_deg = self._pan_units_to_deg(pu)
        target_deg  = max(0.0, min(self._pan_range_deg, current_deg + degrees))
        target_units = self._pan_deg_to_units(target_deg)

        direction = "right" if degrees >= 0 else "left"
        print(f"Panning {direction} {abs(degrees):.1f}°  "
              f"({current_deg:.1f}° → {target_deg:.1f}°, speed={speed})")

        # Motor-command direction depends on how this camera wires the pan axis:
        # when _pan_inverted is True, "Left" increases encoder units; otherwise
        # "Right" does. _move_axis always uses command_fwd when delta>0.
        if self._pan_inverted:
            fwd, rev = "Left", "Right"
        else:
            fwd, rev = "Right", "Left"
        final_units = await self._move_axis(
            command_fwd=fwd,
            command_rev=rev,
            axis="pan",
            target_units=target_units,
            target_deg=target_deg,
            speed=speed,
        )

        final_deg = self._pan_units_to_deg(final_units)
        print(f"\nPan done. Position: {final_deg:.1f}°")
        return {"pan_deg": round(final_deg, 1)}

    async def _tilt_camera(self, degrees: float,
                             speed: int = MOVE_SPEED) -> dict:
        """
        Tilt the camera by the given number of degrees relative to current position.

        Positive = up, negative = down.
        Clamped so the result stays within [0°, 50°].

        Returns a dict with:
            tilt_deg: new absolute tilt position in degrees
        """
        await self.connect()
        _pu, tu = await self._fetch_position()
        current_deg  = self._tilt_units_to_deg(tu)
        target_deg   = max(0.0, min(self._tilt_range_deg, current_deg + degrees))
        target_units = self._tilt_deg_to_units(target_deg)

        direction = "up" if degrees >= 0 else "down"
        print(f"Tilting {direction} {abs(degrees):.1f}°  "
              f"({current_deg:.1f}° → {target_deg:.1f}°, speed={speed})")

        # If _tilt_inverted, "Down" is the command that increases units; otherwise "Up".
        if self._tilt_inverted:
            fwd, rev = "Down", "Up"
        else:
            fwd, rev = "Up", "Down"
        final_units = await self._move_axis(
            command_fwd=fwd,
            command_rev=rev,
            axis="tilt",
            target_units=target_units,
            target_deg=target_deg,
            speed=speed,
        )

        final_deg = self._tilt_units_to_deg(final_units)
        print(f"\nTilt done. Position: {final_deg:.1f}°")
        return {"tilt_deg": round(final_deg, 1)}

    async def _move_to_camera(self, pan_degrees: float, tilt_degrees: float) -> dict:
        """Absolute move in raw camera coordinates (no flip transform).

        Iteratively converges to within MOVE_TOLERANCE_DEG of the target.
        Pass 1 runs at MOVE_SPEED for the bulk of the distance; subsequent
        passes use MOVE_SLOW_SPEED so corrective nudges don't overshoot
        further. Without this loop a single ``ptz_move(pan=90)`` typically
        lands 5-15° off (motor coast varies by direction and lubrication),
        which the agent reads as failure and retries forever.
        """
        await self.connect()
        pan_degrees  = max(0.0, min(self._pan_range_deg,  pan_degrees))
        tilt_degrees = max(0.0, min(self._tilt_range_deg, tilt_degrees))

        pu, tu = await self._fetch_position()
        current_pan_deg  = self._pan_units_to_deg(pu)
        current_tilt_deg = self._tilt_units_to_deg(tu)
        # The "[converge v2]" tag is a deployment marker: if you don't see
        # it in the worker log when ptz_move runs, the new code isn't on
        # the device and you need to (re)scp the file.
        print(f"[converge v2] Moving to pan={pan_degrees:.1f}°, "
              f"tilt={tilt_degrees:.1f}°  "
              f"(from pan={current_pan_deg:.1f}°, tilt={current_tilt_deg:.1f}°, "
              f"tol={MOVE_TOLERANCE_DEG}°, max_passes={MAX_CORRECTION_PASSES})")

        last_pan_deg  = current_pan_deg
        last_tilt_deg = current_tilt_deg

        for attempt in range(MAX_CORRECTION_PASSES):
            pan_err  = pan_degrees - last_pan_deg
            tilt_err = tilt_degrees - last_tilt_deg

            pan_done  = abs(pan_err)  <= MOVE_TOLERANCE_DEG
            tilt_done = abs(tilt_err) <= MOVE_TOLERANCE_DEG
            if pan_done and tilt_done:
                break

            # Slow down for corrective passes — small moves overshoot less
            # at lower speed. Also slow down for short moves on the very
            # first pass (anything under ~5° doesn't benefit from speed).
            speed = MOVE_SPEED
            if attempt > 0:
                speed = MOVE_SLOW_SPEED
            elif max(abs(pan_err), abs(tilt_err)) <= 5.0:
                speed = MOVE_SLOW_SPEED

            if not pan_done:
                pr = await self._pan_camera(pan_err, speed=speed)
                last_pan_deg = pr["pan_deg"]
            if not tilt_done:
                tr = await self._tilt_camera(tilt_err, speed=speed)
                last_tilt_deg = tr["tilt_deg"]

            print(f"  pass {attempt + 1}: at pan={last_pan_deg:.1f}°, "
                  f"tilt={last_tilt_deg:.1f}° "
                  f"(err pan={pan_degrees - last_pan_deg:+.1f}°, "
                  f"tilt={tilt_degrees - last_tilt_deg:+.1f}°)")

        # Authoritative position read after the last move.
        pu, tu = await self._fetch_position()
        return {
            "pan_deg":  self._pan_units_to_deg(pu),
            "tilt_deg": self._tilt_units_to_deg(tu),
        }

    async def snapshot_pil(self):
        """Return the current main-stream frame as an RGB PIL Image.

        When the camera is mounted right-side-up (REOLINK_FLIPPED=1) the raw
        image arrives upside down; we rotate 180° so callers always see a
        correctly-oriented frame.
        """
        from io import BytesIO

        await self.connect()
        raw = await self._host.get_snapshot(0)
        if raw is None:
            raise RuntimeError("Camera returned no image data")
        # Brief settle: main-stream grab can leave the next GetPtzCurPos stale/garbled on some firmware.
        await asyncio.sleep(0.1)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow required for snapshot_pil") from exc
        img = Image.open(BytesIO(raw)).convert("RGB")
        if self._flipped:
            img = img.transpose(Image.ROTATE_180)
        return img

    async def snapshot(self, filename: str | None = None) -> str:
        """
        Capture a JPEG snapshot from the camera and save it to disk.

        Args:
            filename: Optional output path. If omitted, saves as
                      snapshot_YYYYMMDD_HHMMSS.jpg in the current directory.

        Returns:
            Absolute path to the saved JPEG file.
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{ts}.jpg"
        path = Path(filename).resolve()
        print(f"Capturing snapshot → {path}")
        pil = await self.snapshot_pil()
        path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(str(path), quality=92)
        img = path.read_bytes()
        print(f"Snapshot saved ({len(img):,} bytes)")
        return str(path)

    async def look_at_preset(self, preset_id: int) -> dict:
        """
        Move the camera to a predefined PTZ preset by ID.

        Preset IDs are configured in the camera's firmware/web interface.
        The move completes asynchronously in the camera; this method waits
        3 seconds then reads back the final position.

        Args:
            preset_id: Integer preset ID.

        Returns a dict with:
            pan_deg: final pan position in degrees
            tilt_deg: final tilt position in degrees
        """
        await self.connect()
        presets = self._host.ptz_presets(0)
        print(f"Available presets: {presets}")
        print(f"Moving to preset {preset_id}...")
        await self._host.set_ptz_command(0, preset=preset_id)
        await asyncio.sleep(3)
        pu, tu = await self._fetch_position()
        pan_deg  = self._pan_units_to_deg(pu)
        tilt_deg = self._tilt_units_to_deg(tu)
        print(f"Preset reached. Position: pan={pan_deg:.1f}°, tilt={tilt_deg:.1f}°")
        return {"pan_deg": round(pan_deg, 1), "tilt_deg": round(tilt_deg, 1)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd, *rest = args

    async with ReolinkCamera() as cam:
        if cmd == "pan":
            print(await cam.pan(float(rest[0])))

        elif cmd == "tilt":
            print(await cam.tilt(float(rest[0])))

        elif cmd == "move_to":
            print(await cam.move_to(float(rest[0]), float(rest[1])))

        elif cmd == "get_position":
            print(await cam.get_position())

        elif cmd == "snapshot":
            print(await cam.snapshot(rest[0] if rest else None))

        elif cmd == "look_at_preset":
            print(await cam.look_at_preset(int(rest[0])))

        else:
            print(f"Unknown command: {cmd}")
            print("Commands: pan, tilt, move_to, get_position, snapshot, look_at_preset")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
