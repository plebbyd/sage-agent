"""
tools/ptz_tool.py — PTZ camera control as an MSA tool plugin.

Auto-discovered by the MSA plugin loader.  Wraps the ReolinkCamera
class from the same directory.

Requires: reolink_aio  (pip install reolink_aio)
Environment: REOLINK_IP, REOLINK_USER, REOLINK_PASSWORD
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from msa.tools import BaseTool

try:
    from reolink_camera import ReolinkCamera
    HAS_REOLINK = True
except ImportError:
    HAS_REOLINK = False

_MISSING_MSG = "ERROR: reolink_aio not installed. pip install reolink_aio"

# Per-process loop guards. Each worker is its own subprocess, so these are
# naturally scoped to a single agent run.
#
# Policy:
# 1. The first ptz_move call in a run runs normally.
# 2. Any duplicate (pan, tilt) target afterwards is refused outright. Reolink
#    motor coast means a re-issued move only makes things worse.
# 3. After PTZ_MOVE_HARD_CAP attempts in this run -- counting BOTH driven and
#    skipped invocations -- every further ptz_move is refused regardless of
#    args. This catches the case where the model jiggles the target a bit
#    (e.g. 325 -> 355 -> 325) to slip past the duplicate check.
# 4. ptz_observe / ptz_move callers also share the "already at this target"
#    short-circuit so we don't re-drive the motor for tiny corrections.
_LAST_MOVE: dict = {
    "pan": None, "tilt": None,
    "result_pan": None, "result_tilt": None,
    "repeat_count": 0,
}
# 3 attempts (driven OR skipped) per worker run. After this every ptz_move
# returns a hard-stop response. Lower than feels comfortable on purpose.
PTZ_MOVE_HARD_CAP_PER_WORKER = 3
# When ptz_observe / ptz_move sees the camera is already within this many
# degrees of the requested target, the move is skipped entirely (no motor
# command). Larger than the driver's internal MOVE_TOLERANCE_DEG=2.0° so a
# legit 1-2° drift after the previous move doesn't trigger a fresh coast cycle.
ALREADY_THERE_DEG = 3.0
_PTZ_MOVE_ATTEMPT_COUNT = 0  # all invocations, not just successful ones

# Loop guard for ptz_scan. A scan with describe=true takes ~60-90 seconds; a
# small model finishing one and immediately calling it again is the single
# biggest source of multi-minute wasted runs. Same per-process scoping as
# _LAST_MOVE: each worker is its own subprocess so this resets per run.
_LAST_SCAN_RESULT: str | None = None
_PTZ_SCAN_ATTEMPT_COUNT = 0
PTZ_SCAN_HARD_CAP_PER_WORKER = 1  # one scan per worker run, full stop

# Default tilt used when ptz_move is called with only `pan`. The user prefers
# the camera to sit at ~20° tilt during normal operation rather than at the
# raw current tilt (which after a calibration sweep is 0°). Override per-call
# by passing `tilt` explicitly, or globally via PTZ_DEFAULT_TILT env var.
DEFAULT_TILT_DEG = float(os.environ.get("PTZ_DEFAULT_TILT", "20").strip() or 20.0)

# Path to the env file `setup.sh` writes camera credentials into. Persisted
# updates from ptz_find_camera get merged into this file so the new IP
# survives process restarts.
ENV_FILE_PATH = Path(os.environ.get(
    "MSA_ENV_FILE", str(Path.home() / ".msa.env")
))


def _update_env_file(updates: dict, path: Path = ENV_FILE_PATH) -> bool:
    """Merge ``updates`` ({"REOLINK_IP": "10.31.81.43"}) into the shell env
    file at ``path``. Existing lines for the same key are replaced;
    everything else is preserved. Returns True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text() if path.exists() else ""
    except OSError:
        existing = ""

    lines = existing.splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        m = re.match(r'^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=', line)
        if m and m.group(1) in updates:
            key = m.group(1)
            seen.add(key)
            out.append(f'export {key}="{updates[key]}"')
        else:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f'export {key}="{val}"')

    try:
        path.write_text("\n".join(out) + "\n")
        return True
    except OSError:
        return False


def _arp_scan_for_reolink(
    interface: str | None = None,
    subnet: str | None = None,
    timeout: float = 15.0,
) -> tuple[str | None, str]:
    """Run arp-scan and return (ip, log) where ip is the first Reolink-vendor
    address found (or None) and log is the raw stdout/stderr captured for
    diagnostics."""
    if not shutil.which("arp-scan"):
        return None, ("arp-scan not installed. Install with "
                       "`apt-get install arp-scan` (Debian/Ubuntu).")

    cmd = ["sudo", "-n", "arp-scan"]
    if interface:
        cmd.append(f"--interface={interface}")
    if subnet:
        cmd.append(subnet)
    else:
        cmd.append("--localnet")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return None, f"arp-scan timed out after {timeout}s: {exc}"
    except FileNotFoundError as exc:
        return None, f"arp-scan invocation failed: {exc}"

    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "Reolink" not in log:
        # Common cause: passwordless sudo not configured for this user.
        return None, (
            f"arp-scan exit {proc.returncode}: {log[-400:]}\n"
            "Likely fix: add a sudoers rule allowing `sudo -n arp-scan` "
            "without a password, or run the agent as root."
        )

    for line in log.splitlines():
        if "Reolink" in line:
            m = re.match(r"^\s*(\d+\.\d+\.\d+\.\d+)\s", line)
            if m:
                return m.group(1), log
    return None, log


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class PTZMoveTool(BaseTool):
    name = "ptz_move"
    description = (
        "Move the PTZ camera to an absolute position. "
        "Args: pan (float, 0-355 degrees), tilt (float, 0-50 degrees). "
        f"If `tilt` is omitted, defaults to {DEFAULT_TILT_DEG}° (the "
        "operating-height default). To pan without changing tilt, also "
        "pass the current tilt explicitly. The driver converges internally "
        "to within ~2° of the target (motor coast makes exact placement "
        "physically impossible), so the returned pan_deg/tilt_deg may differ "
        "from the requested value by a couple of degrees -- THIS IS SUCCESS, "
        "NOT FAILURE. Do NOT call ptz_move again to 'correct' a 1-3° offset; "
        "treat the move as done."
    )

    def run(self, pan: float = None, tilt: float = None, **kwargs) -> str:
        global _PTZ_MOVE_ATTEMPT_COUNT
        if not HAS_REOLINK:
            return _MISSING_MSG

        # Count EVERY invocation (including the ones we skip below) so the
        # model can't bypass the cap by jiggling its target.
        _PTZ_MOVE_ATTEMPT_COUNT += 1
        attempt_n = _PTZ_MOVE_ATTEMPT_COUNT

        req_pan  = float(pan)  if pan  is not None else None
        req_tilt = float(tilt) if tilt is not None else None
        last_pan_target  = _LAST_MOVE["pan"]
        last_tilt_target = _LAST_MOVE["tilt"]
        last_result_pan  = _LAST_MOVE["result_pan"]
        last_result_tilt = _LAST_MOVE["result_tilt"]

        # Hard cap: too many ptz_move attempts in this run, regardless of args.
        if attempt_n > PTZ_MOVE_HARD_CAP_PER_WORKER:
            return json.dumps({
                "ok": True,
                "skipped": True,
                "reason": (
                    f"ptz_move HARD CAP HIT ({PTZ_MOVE_HARD_CAP_PER_WORKER} "
                    f"attempts already in this run). The camera is where it "
                    "is. Reolink motor coast means further ptz_move calls "
                    "will NOT improve accuracy. STOP calling ptz_move. Use "
                    "ptz_observe to capture the current view, or respond to "
                    "the user with the position below."
                ),
                "pan_deg":  last_result_pan,
                "tilt_deg": last_result_tilt,
                "attempt_n": attempt_n,
            })

        # Already-there short-circuit: if the camera is currently within
        # ALREADY_THERE_DEG of the requested target, skip the move. Avoids
        # re-coasting for a 1-2° "correction" that just lands somewhere else.
        if last_result_pan is not None:
            pan_close = (req_pan is None
                         or abs(last_result_pan - req_pan) <= ALREADY_THERE_DEG)
            tilt_close = (req_tilt is None or last_result_tilt is None
                          or abs(last_result_tilt - req_tilt) <= ALREADY_THERE_DEG)
            if pan_close and tilt_close:
                return json.dumps({
                    "ok": True,
                    "skipped": True,
                    "reason": (
                        f"Camera already within {ALREADY_THERE_DEG}° of this "
                        "target from a previous move. Skipped (re-driving the "
                        "motor for sub-degree corrections only adds coast "
                        "error). STOP calling ptz_move."
                    ),
                    "pan_deg":  last_result_pan,
                    "tilt_deg": last_result_tilt,
                    "requested_pan":  req_pan,
                    "requested_tilt": req_tilt,
                    "attempt_n": attempt_n,
                })

        # Duplicate-target guard (exact match).
        same_target = (
            last_pan_target is not None
            and last_result_pan is not None
            and req_pan  == last_pan_target
            and req_tilt == last_tilt_target
        )
        if same_target:
            _LAST_MOVE["repeat_count"] += 1
            return json.dumps({
                "ok": True,
                "skipped": True,
                "reason": (
                    "DUPLICATE ptz_move call refused. The camera was already "
                    "driven to this exact target on the previous tool call. "
                    "Reolink motors coast unpredictably -- re-issuing the same "
                    "move WILL NOT make the position more accurate. STOP. Use "
                    "ptz_pan/ptz_tilt for relative nudges, or call ptz_observe "
                    "/ respond."
                ),
                "pan_deg":  last_result_pan,
                "tilt_deg": last_result_tilt,
                "requested_pan":  req_pan,
                "requested_tilt": req_tilt,
                "repeat_count": _LAST_MOVE["repeat_count"],
                "attempt_n": attempt_n,
            })

        async def _move():
            async with ReolinkCamera() as cam:
                if pan is not None and tilt is not None:
                    result = await cam.move_to(float(pan), float(tilt))
                elif pan is not None:
                    # Pan without tilt: default to DEFAULT_TILT_DEG so the
                    # camera consistently sits at the operating-height tilt
                    # rather than wherever the previous move left it.
                    result = await cam.move_to(float(pan), DEFAULT_TILT_DEG)
                elif tilt is not None:
                    pos = await cam.get_position()
                    result = await cam.move_to(pos["pan_deg"], float(tilt))
                else:
                    return json.dumps({"error": "provide pan and/or tilt in degrees"})
                return json.dumps(result)

        out = _run_async(_move())
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict) and "pan_deg" in parsed:
                _LAST_MOVE["pan"]          = req_pan
                _LAST_MOVE["tilt"]         = req_tilt
                _LAST_MOVE["result_pan"]   = parsed.get("pan_deg")
                _LAST_MOVE["result_tilt"]  = parsed.get("tilt_deg")
                _LAST_MOVE["repeat_count"] = 0
        except (ValueError, TypeError):
            pass
        return out


class PTZPanTool(BaseTool):
    """Relative pan — preferred when the user says 'turn slightly left/right'.

    Reads the current pan from the camera, adds ``delta_deg``, clamps to
    the calibrated range, and issues an absolute ``move_to`` for the
    new position. Tilt is preserved.
    """

    name = "ptz_pan"
    description = (
        "Pan the PTZ camera by a relative amount. Args: "
        "delta_deg (float, required). Negative = left, positive = right. "
        "Examples: turn slightly left → delta_deg=-15. Slightly right → 15."
    )

    def run(self, delta_deg: float = None, **kwargs) -> str:
        if not HAS_REOLINK:
            return _MISSING_MSG
        if delta_deg is None:
            return ('ERROR: delta_deg is required (e.g. {"delta_deg": -15} '
                    'for "slightly left")')

        async def _go():
            async with ReolinkCamera() as cam:
                pos = await cam.get_position()
                target_pan = float(pos["pan_deg"]) + float(delta_deg)
                result = await cam.move_to(target_pan, pos["tilt_deg"])
                return json.dumps(result)

        return _run_async(_go())


class PTZTiltTool(BaseTool):
    """Relative tilt — for 'look up' / 'look down' style requests."""

    name = "ptz_tilt"
    description = (
        "Tilt the PTZ camera by a relative amount. Args: "
        "delta_deg (float, required). Negative = down, positive = up."
    )

    def run(self, delta_deg: float = None, **kwargs) -> str:
        if not HAS_REOLINK:
            return _MISSING_MSG
        if delta_deg is None:
            return ('ERROR: delta_deg is required (e.g. {"delta_deg": 10} '
                    'for "look up a bit")')

        async def _go():
            async with ReolinkCamera() as cam:
                pos = await cam.get_position()
                target_tilt = float(pos["tilt_deg"]) + float(delta_deg)
                result = await cam.move_to(pos["pan_deg"], target_tilt)
                return json.dumps(result)

        return _run_async(_go())


class PTZPositionTool(BaseTool):
    name = "ptz_position"
    description = "Get the current PTZ camera position in degrees."

    def run(self, **kwargs) -> str:
        if not HAS_REOLINK:
            return _MISSING_MSG

        async def _pos():
            async with ReolinkCamera() as cam:
                return json.dumps(await cam.get_position())

        return _run_async(_pos())


class PTZSnapshotTool(BaseTool):
    name = "ptz_snapshot"
    description = (
        "Capture a JPEG snapshot from the PTZ camera. "
        "Args: filename (str, optional — defaults to snapshot_TIMESTAMP.jpg)."
    )

    def run(self, filename: str = None, **kwargs) -> str:
        if not HAS_REOLINK:
            return _MISSING_MSG

        async def _snap():
            async with ReolinkCamera() as cam:
                path = await cam.snapshot(filename)
                return f"Snapshot saved to {path}"

        return _run_async(_snap())


class PTZScanTool(BaseTool):
    """One-shot panorama sweep — covers the whole pan range in a single call.

    Designed so an LLM agent doesn't have to choreograph each individual move.
    The agent calls `ptz_scan` once and gets back a JSON summary of every stop.
    Optional `describe=true` runs Gemma 4 (via Ollama) on each snapshot so the
    agent receives natural-language descriptions in addition to the file paths.
    """

    name = "ptz_scan"
    description = (
        "Sweep the PTZ camera across its pan range, snapshot at each stop, "
        "and (optionally) describe each viewport with Gemma 4. Returns a JSON "
        "summary. Args: stops (int, default 8 — number of evenly-spaced pan "
        "positions), pan_min (float, default 0), pan_max (float, default 355), "
        "tilt (float, optional — fix tilt for the whole sweep; defaults to "
        "current tilt), describe (bool, default false — caption each frame "
        "with Gemma 4), prompt (str, optional — Gemma 4 caption prompt), "
        "out_dir (str, default 'snapshots' — directory for saved JPEGs)."
    )

    def run(
        self,
        stops: int = 8,
        pan_min: float = 0.0,
        pan_max: float = 355.0,
        tilt: float = None,
        describe: bool = False,
        prompt: str = None,
        out_dir: str = "snapshots",
        **kwargs,
    ) -> str:
        global _PTZ_SCAN_ATTEMPT_COUNT, _LAST_SCAN_RESULT
        if not HAS_REOLINK:
            return _MISSING_MSG

        # Hard cap: one full scan per worker run. ptz_scan with describe=true
        # takes ~60-90 s; a small model finishing one and immediately
        # re-running it is by far the most expensive failure mode. Return
        # the cached result so the model can re-read it instead.
        _PTZ_SCAN_ATTEMPT_COUNT += 1
        if _PTZ_SCAN_ATTEMPT_COUNT > PTZ_SCAN_HARD_CAP_PER_WORKER:
            return json.dumps({
                "ok": True,
                "skipped": True,
                "reason": (
                    f"ptz_scan HARD CAP HIT ({PTZ_SCAN_HARD_CAP_PER_WORKER} "
                    "scan per worker run). A scan with descriptions takes "
                    "~60-90 s and produces a complete view of the room -- "
                    "running it again will give substantially the same "
                    "captions. STOP. Read the captions from the previous "
                    "scan (echoed below) and respond to the user."
                ),
                "attempt_n": _PTZ_SCAN_ATTEMPT_COUNT,
                "previous_result": _LAST_SCAN_RESULT,
            }, indent=2)

        try:
            stops = max(1, int(stops))
        except (TypeError, ValueError):
            stops = 8
        pan_min = float(pan_min)
        pan_max = float(pan_max)
        # Accept single-stop / zero-width range as a degenerate "look at
        # one specific pan and describe" — the master agent reaches for
        # this when the user says "go to X and describe". Failing in
        # that case forced the model to fabricate a 1°-wide range.
        if pan_max < pan_min:
            return ("ERROR: pan_max must be >= pan_min (use ptz_observe "
                    "for a single-pan snapshot+describe).")
        if pan_max == pan_min:
            stops = 1

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        describer = None
        describe_err = None
        if describe:
            try:
                from tools.detectors import get_detector
                describer = get_detector("gemma4")
            except Exception as exc:
                describer = None
                describe_err = f"describe disabled: {exc}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if stops <= 1:
            targets = [round(pan_min, 2)]
        else:
            step = (pan_max - pan_min) / (stops - 1)
            targets = [round(pan_min + i * step, 2) for i in range(stops)]

        results: list[dict] = []

        async def _sweep():
            async with ReolinkCamera() as cam:
                if tilt is None:
                    pos = await cam.get_position()
                    tilt_deg = pos["tilt_deg"]
                else:
                    tilt_deg = float(tilt)

                for i, pan_deg in enumerate(targets):
                    t0 = time.time()
                    try:
                        move = await cam.move_to(pan_deg, tilt_deg)
                    except Exception as exc:
                        results.append({
                            "stop": i + 1,
                            "pan_deg": pan_deg,
                            "tilt_deg": tilt_deg,
                            "error": f"move failed: {exc}",
                        })
                        continue

                    fname = f"scan_{ts}_p{pan_deg:06.2f}.jpg".replace(" ", "0")
                    snap_path = await cam.snapshot(str(out_path / fname))

                    description = None
                    if describer is not None:
                        try:
                            from PIL import Image
                            img = Image.open(snap_path).convert("RGB")
                            description = describer.describe(img, prompt=prompt)
                        except Exception as exc:
                            description = f"describe error: {exc}"

                    entry = {
                        "stop": i + 1,
                        "pan_deg": pan_deg,
                        "tilt_deg": tilt_deg,
                        "snapshot": snap_path,
                        "elapsed_s": round(time.time() - t0, 2),
                        "actual_pan_deg": move.get("pan_deg") if isinstance(move, dict) else None,
                        "actual_tilt_deg": move.get("tilt_deg") if isinstance(move, dict) else None,
                    }
                    if description is not None:
                        entry["description"] = description[:600]
                    results.append(entry)

        try:
            _run_async(_sweep())
        except Exception as exc:
            return f"ERROR: ptz_scan failed: {exc}"

        # Compact view: pan_deg → caption / error. Short and JSON-truncation
        # friendly so the agent sees the actual content even if its tool-result
        # buffer is small. Full per-stop detail still ships in `results` below.
        captions: list[dict] = []
        for r in results:
            row = {"pan_deg": r.get("pan_deg")}
            if "error" in r:
                row["error"] = r["error"]
            elif "description" in r:
                row["caption"] = r["description"]
            else:
                row["snapshot"] = r.get("snapshot")
            captions.append(row)

        summary = {
            "stops_planned": stops,
            "stops_completed": len([r for r in results if "error" not in r]),
            "pan_range": [pan_min, pan_max],
            "out_dir": str(out_path),
            "described": describer is not None,
            "describe_unavailable_reason": describe_err,
            "captions": captions,
            "results": results,
        }

        # Persist the full report next to the snapshots so it's recoverable
        # regardless of how aggressively the agent's notes get truncated.
        try:
            report_json = out_path / f"scan_{ts}_report.json"
            report_json.write_text(json.dumps(summary, indent=2))
            report_md = out_path / f"scan_{ts}_report.md"
            with report_md.open("w") as f:
                f.write(f"# PTZ scan report — {ts}\n\n")
                f.write(f"- pan_range: [{pan_min}, {pan_max}]  stops: {stops}\n")
                f.write(f"- out_dir: {out_path}\n")
                f.write(f"- described: {describer is not None}\n")
                if describe_err:
                    f.write(f"- describe_unavailable_reason: {describe_err}\n")
                f.write("\n## Stops\n\n")
                for r in results:
                    f.write(f"### Stop {r.get('stop')} — pan {r.get('pan_deg')}°\n\n")
                    if "error" in r:
                        f.write(f"**ERROR:** {r['error']}\n\n")
                        continue
                    if r.get("snapshot"):
                        f.write(f"![]({r['snapshot']})\n\n")
                    if r.get("description"):
                        f.write(f"{r['description']}\n\n")
            summary["report_json"] = str(report_json)
            summary["report_md"] = str(report_md)
        except Exception as exc:
            summary["report_error"] = str(exc)

        out = json.dumps(summary, indent=2)
        # Cache for the loop-guard branch so a re-run can echo the
        # captions back instead of re-driving the camera.
        _LAST_SCAN_RESULT = out[:6000]
        return out


class PTZObserveTool(BaseTool):
    """Move + snapshot + describe at a single pan/tilt — the right
    primitive for "go to X and tell me what you see".

    The agent kept reaching for ``ptz_scan(stops=1, pan_min=X, pan_max=X)``
    which fails the range check, then fudging a 1°-wide range to work
    around it. ``ptz_observe`` does the obvious one-shot path: optional
    move, capture, caption, return the description in a single call.
    """

    name = "ptz_observe"
    description = (
        "Look at ONE position with the PTZ camera: snapshot + Gemma 4 "
        "caption, no panorama sweep. Args: pan (float, optional — "
        "degrees, defaults to CURRENT pan if omitted), tilt (float, "
        "optional — defaults to CURRENT tilt), prompt (str, optional — "
        "Gemma 4 caption prompt; e.g. 'describe the person in detail'), "
        "describe (bool, default true), out_dir (str, default "
        "'snapshots'). Returns JSON with pan_deg, tilt_deg, snapshot "
        "path, and description. "
        ""
        "USE THIS when the user wants the CURRENT camera view "
        "described — phrases like 'describe what you see', 'describe "
        "the person', 'caption this view', 'what's the camera looking "
        "at right now'. Call with NO pan/tilt args to keep the camera "
        "where it is and just snap+describe. "
        ""
        "Also use it for 'go to X and describe' (pass `pan` / `tilt`). "
        "Do NOT use ptz_scan for single-frame describes — ptz_scan does "
        "a multi-stop sweep and is overkill."
    )

    def run(
        self,
        pan: float = None,
        tilt: float = None,
        describe: bool = True,
        prompt: str = None,
        out_dir: str = "snapshots",
        **kwargs,
    ) -> str:
        if not HAS_REOLINK:
            return _MISSING_MSG

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        describer = None
        describe_err = None
        if describe:
            try:
                from tools.detectors import get_detector
                describer = get_detector("gemma4")
            except Exception as exc:
                describer = None
                describe_err = f"describe disabled: {exc}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        async def _go():
            async with ReolinkCamera() as cam:
                pos = await cam.get_position()
                target_pan = float(pan) if pan is not None else pos["pan_deg"]
                target_tilt = float(tilt) if tilt is not None else pos["tilt_deg"]
                # Only drive the motors if we're actually meaningfully far
                # from the target. Avoids re-coasting the camera 5-10° when
                # the model says "observe at pan=325" while we're already at
                # pan=326.2 from the previous ptz_move.
                pan_far  = pan  is not None and abs(target_pan  - pos["pan_deg"])  > ALREADY_THERE_DEG
                tilt_far = tilt is not None and abs(target_tilt - pos["tilt_deg"]) > ALREADY_THERE_DEG
                if pan_far or tilt_far:
                    move = await cam.move_to(target_pan, target_tilt)
                else:
                    move = pos
                fname = f"observe_{ts}_p{target_pan:06.2f}.jpg".replace(" ", "0")
                snap = await cam.snapshot(str(out_path / fname))
                return move, snap

        try:
            move, snap_path = _run_async(_go())
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"move/snapshot failed: {exc}"})

        result = {
            "ok": True,
            "pan_deg": move.get("pan_deg") if isinstance(move, dict) else None,
            "tilt_deg": move.get("tilt_deg") if isinstance(move, dict) else None,
            "snapshot": snap_path,
        }

        if describer is not None:
            try:
                from PIL import Image
                img = Image.open(snap_path).convert("RGB")
                result["description"] = describer.describe(img, prompt=prompt)
            except Exception as exc:
                result["description_error"] = str(exc)
        elif describe_err:
            result["describe_unavailable_reason"] = describe_err

        return json.dumps(result, indent=2)


class PTZCalibrateTool(BaseTool):
    """Recalibrate the PTZ camera and rewrite ``tools/calibration.json``.

    Use sparingly — the sweep drives the camera into both pan and tilt
    hard-stops and takes ~70-90 s. The agent should call this when:
      * a previous PTZ command failed with a "camera did not move" /
        no-motion error, or
      * the user explicitly asks to (re)calibrate the camera, or
      * pan/tilt readings look suspicious (e.g. ``pan_min > pan_max``
        in the existing calibration file).
    """

    name = "ptz_calibrate"
    description = (
        "Recalibrate the PTZ camera by driving it through its full pan and "
        "tilt range and measuring units-per-degree. Overwrites "
        "tools/calibration.json (previous file backed up automatically). "
        "Slow: ~70-90 seconds. Returns a JSON summary with the new ranges. "
        "Args: confirm (bool, default true — set to false to abort without "
        "moving the camera). Use when ptz_move/ptz_pan/ptz_tilt fail with a "
        "no-motion error, when the user asks to (re)calibrate, or when the "
        "existing calibration file looks corrupt."
    )

    def run(self, confirm: bool = True, **kwargs) -> str:
        if not HAS_REOLINK:
            return _MISSING_MSG
        if not confirm:
            return ('ERROR: ptz_calibrate aborted (confirm=false). Pass '
                    'confirm=true to actually run the ~90s sweep.')

        # Defer import until call time: the module imports reolink_aio
        # at top level, so the plugin loader's import-timeout guard
        # otherwise has to wait on it.
        from tools.calibrate_ptz import calibrate

        log_lines: list[str] = []

        def _capture(line: str) -> None:
            log_lines.append(line)

        started = time.time()
        try:
            results = _run_async(calibrate(on_log=_capture))
        except Exception as exc:
            payload = {
                "ok": False,
                "error": str(exc),
                "elapsed_s": round(time.time() - started, 2),
                "log": log_lines[-40:],
            }
            return json.dumps(payload, indent=2)

        payload = {
            "ok": True,
            "elapsed_s": round(time.time() - started, 2),
            "calibration": results,
            "calibration_path": "tools/calibration.json",
            "log_tail": log_lines[-12:],
        }
        return json.dumps(payload, indent=2)


class PTZFindCameraTool(BaseTool):
    """Locate the Reolink PTZ camera on the LAN via arp-scan and update
    REOLINK_IP in-process and in ~/.msa.env.

    Use when:
      * ptz_* tools fail with connection / authentication errors that
        suggest REOLINK_IP is wrong (DHCP gave the camera a new address),
      * the user explicitly asks to find / locate / refresh the camera,
      * REOLINK_IP is unset.

    Requires `arp-scan` to be installed and (typically) passwordless
    sudo for the agent's user. The scan looks for any device whose
    vendor string contains "Reolink".
    """

    name = "ptz_find_camera"
    description = (
        "Scan the local network for the Reolink PTZ camera and update "
        "REOLINK_IP. Args: interface (str, optional -- network interface "
        "to scan, e.g. 'lan0'; defaults to env REOLINK_SCAN_INTERFACE or "
        "'lan0'), subnet (str, optional -- CIDR like '10.31.81.0/24'; "
        "default uses arp-scan --localnet for auto-detection), "
        "update_env (bool, default true -- if true, also persists the new "
        "IP to ~/.msa.env so it survives restarts). Returns the IP found "
        "(or null) plus diagnostic log. Use this when ptz_* tools fail "
        "with connection errors, when REOLINK_IP is unset, or when the "
        "user asks to (re)find / fix the camera."
    )

    def run(
        self,
        interface: str = None,
        subnet: str = None,
        update_env: bool = True,
        **kwargs,
    ) -> str:
        iface = (interface
                  or os.environ.get("REOLINK_SCAN_INTERFACE", "").strip()
                  or "lan0")

        prev_ip = os.environ.get("REOLINK_IP", "").strip() or None
        ip, log = _arp_scan_for_reolink(interface=iface, subnet=subnet)

        if ip is None:
            return json.dumps({
                "ok": False,
                "ip": None,
                "previous_ip": prev_ip,
                "interface": iface,
                "subnet": subnet,
                "log_tail": log[-800:],
                "hint": (
                    "No Reolink-vendor device found. Check that the camera "
                    "is powered, on the same LAN, and that arp-scan has "
                    "permission to run. Try a different `interface`."
                ),
            }, indent=2)

        os.environ["REOLINK_IP"] = ip
        env_written = False
        env_error = None
        if update_env:
            try:
                env_written = _update_env_file({"REOLINK_IP": ip})
            except Exception as exc:  # noqa: BLE001
                env_error = str(exc)

        return json.dumps({
            "ok": True,
            "ip": ip,
            "previous_ip": prev_ip,
            "changed": prev_ip != ip,
            "interface": iface,
            "env_file": str(ENV_FILE_PATH),
            "env_file_updated": env_written,
            "env_file_error": env_error,
            "next_step": (
                "Retry the original ptz_* tool call. The camera client "
                "reads REOLINK_IP afresh on every connect, so the new "
                "address is already in effect for this run."
            ),
        }, indent=2)
