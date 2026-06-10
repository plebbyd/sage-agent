# calibrate_ptz.py — measure PTZ range and units-per-degree for Reolink E1 Outdoor SE PoE
#
# Must match the runtime position source (Baichuan GetPtzCurPos) so the
# viewer/agent math matches what the camera reports during operation. Saves to
# tools/calibration.json (the path ReolinkCamera reads on startup).
#
# Two ways to use it:
#   * CLI:    python3 -m tools.calibrate_ptz
#   * Agent:  the `ptz_calibrate` tool wraps `calibrate()` so the master agent
#             can recalibrate when the camera ignores moves (stale calibration
#             after a swap, etc.) without operator intervention.

import asyncio
import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Callable

from reolink_aio.api import Host

REOLINK_HOST = (
    os.environ.get("REOLINK_IP", "").strip()
    or os.environ.get("REOLINK_HOST", "").strip()
)
REOLINK_USER = os.environ.get("REOLINK_USER", "admin")
REOLINK_PASS = os.environ.get("REOLINK_PASSWORD", "")

PAN_DEGREES = 355.0
TILT_DEGREES = 50.0

_CAL_PATH = Path(__file__).parent / "calibration.json"


async def _read_pos(host: Host) -> tuple[int, int]:
    """Read pan/tilt via the same Baichuan channel the runtime uses."""
    await host.baichuan.get_ptz_position(0)
    return int(host.ptz_pan_position(0)), int(host.ptz_tilt_position(0))


async def _move_and_read(host, command, duration, label, log):
    log(f"  Moving {command} for {duration}s…")
    await host.set_ptz_command(0, command=command)
    await asyncio.sleep(duration)
    await host.set_ptz_command(0, command="Stop")
    await asyncio.sleep(1)  # let motor settle
    pan, tilt = await _read_pos(host)
    log(f"  {label}: pan={pan}, tilt={tilt}")
    return pan, tilt


async def calibrate(
    *,
    on_log: Callable[[str], None] | None = None,
    timeout_seconds: float = 180.0,
) -> dict:
    """Run the calibration sweep and write ``calibration.json``.

    Returns the result dict on success. Raises ``RuntimeError`` (with a
    clear message) on env-var problems, connection failures, or if the
    camera doesn't physically move when commanded — the same conditions
    the agent's ``ptz_calibrate`` tool needs to surface to the user.

    ``on_log`` is invoked with each progress line so callers (the CLI
    and the tool wrapper) can route output however they like. If
    omitted, lines go to stdout via ``print``.
    """
    log = on_log or (lambda s: print(s))

    # Re-read env at call time so a freshly-discovered IP from
    # ptz_find_camera takes effect without restarting the process.
    host_addr = (
        os.environ.get("REOLINK_IP", "").strip()
        or os.environ.get("REOLINK_HOST", "").strip()
    )
    user = os.environ.get("REOLINK_USER", "admin")
    pwd  = os.environ.get("REOLINK_PASSWORD", "")

    if not host_addr:
        raise RuntimeError(
            "REOLINK_IP (or REOLINK_HOST) must be set in the environment. "
            "Run ptz_find_camera to scan the LAN."
        )
    if not pwd:
        raise RuntimeError("REOLINK_PASSWORD must be set in the environment")

    log(f"Connecting to camera at {host_addr}…")
    host = Host(host=host_addr, username=user, password=pwd)
    started = time.monotonic()
    try:
        await host.get_host_data()

        pan0, tilt0 = await _read_pos(host)
        log(f"Start position (Baichuan): pan={pan0}, tilt={tilt0}")

        log("=== Pan symmetry test (2 s each) ===")
        pan_r, _ = await _move_and_read(host, "Right", 2, "After Right 2s", log)
        log(f"  Pan delta Right: {pan_r - pan0:+d} units")

        pan_l, _ = await _move_and_read(host, "Left", 2, "After Left 2s", log)
        log(f"  Pan delta Left (from Right pos): {pan_l - pan_r:+d} units")

        # If neither symmetry move budged the encoder, the camera is
        # ignoring PTZ commands — fail fast rather than burn a full
        # minute driving "to the limits" that we never reach.
        if pan_r == pan0 and pan_l == pan_r:
            raise RuntimeError(
                "Camera ignored both Right and Left PTZ commands "
                "(encoder did not change). Likely causes: PTZ disabled "
                "in camera web UI, an active privacy mask, auto-tracking "
                "fighting the move, or a non-PTZ camera model."
            )

        log("=== Tilt symmetry test (2 s each) ===")
        _, tilt_u = await _move_and_read(host, "Up", 2, "After Up 2s", log)
        log(f"  Tilt delta Up: {tilt_u - tilt0:+d} units")

        _, tilt_d = await _move_and_read(host, "Down", 2, "After Down 2s", log)
        log(f"  Tilt delta Down (from Up pos): {tilt_d - tilt_u:+d} units")

        if time.monotonic() - started > timeout_seconds:
            raise RuntimeError(f"Calibration exceeded {timeout_seconds}s budget")

        log("=== Pan full range ===")
        log("  Driving all the way Left (15 s)…")
        pan_min, _ = await _move_and_read(host, "Left", 15,
                                            "Hard-left position", log)

        log("  Driving all the way Right (15 s)…")
        pan_max, _ = await _move_and_read(host, "Right", 15,
                                            "Hard-right position", log)

        pan_range = pan_max - pan_min
        if pan_range == 0:
            raise RuntimeError("Pan range collapsed to 0 — encoder reads are stuck.")
        pan_upd = pan_range / PAN_DEGREES
        log(f"  Pan range: {pan_min} → {pan_max}  "
            f"({pan_range} units over {PAN_DEGREES}°)")
        log(f"  Pan units/degree: {pan_upd:.2f}")

        log("=== Tilt full range ===")
        log("  Driving all the way Down (15 s)…")
        _, tilt_min = await _move_and_read(host, "Down", 15,
                                             "Hard-down position", log)

        log("  Driving all the way Up (15 s)…")
        _, tilt_max = await _move_and_read(host, "Up", 15,
                                             "Hard-up position", log)

        tilt_range = tilt_max - tilt_min
        if tilt_range == 0:
            raise RuntimeError("Tilt range collapsed to 0 — encoder reads are stuck.")
        tilt_upd = tilt_range / TILT_DEGREES
        log(f"  Tilt range: {tilt_min} → {tilt_max}  "
            f"({tilt_range} units over {TILT_DEGREES}°)")
        log(f"  Tilt units/degree: {tilt_upd:.2f}")

        results = {
            "pan_min": pan_min,
            "pan_max": pan_max,
            "pan_range": pan_range,
            "pan_degrees": PAN_DEGREES,
            "pan_units_per_degree": pan_upd,
            "tilt_min": tilt_min,
            "tilt_max": tilt_max,
            "tilt_range": tilt_range,
            "tilt_degrees": TILT_DEGREES,
            "tilt_units_per_degree": tilt_upd,
        }

        _CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Back up any prior calibration so the operator can compare /
        # revert. Only one .bak per run is kept (good enough; it's a
        # rarely-used file and the new one is the source of truth).
        if _CAL_PATH.exists():
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = _CAL_PATH.with_suffix(f".json.{stamp}.bak")
            backup.write_bytes(_CAL_PATH.read_bytes())
            log(f"Backed up previous calibration to {backup.name}")
        _CAL_PATH.write_text(json.dumps(results, indent=2))
        log(f"Calibration saved to {_CAL_PATH}")

        return results
    finally:
        try:
            await host.logout()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    asyncio.run(calibrate())
