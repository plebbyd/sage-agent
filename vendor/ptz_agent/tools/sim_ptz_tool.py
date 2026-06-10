"""
tools/sim_ptz_tool.py — Simulated PTZ camera using a panoramic image.

Uses stitched.png as a 360-degree panorama and simulates PTZ movement
by sliding a viewport window across it.  Pan wraps horizontally;
tilt clamps at image edges.

State persists in scratchpads/sim_ptz_state.json so position survives
across tool calls and agent cycles.

Requires: Pillow  (pip install Pillow)
"""

import json
from datetime import datetime
from pathlib import Path

from msa.tools import BaseTool

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_DEFAULT_PANORAMA = _PROJECT_ROOT / "stitched.png"
_STATE_FILE = _PROJECT_ROOT / "scratchpads" / "sim_ptz_state.json"
_MISSING_MSG = "ERROR: Pillow not installed. pip install Pillow"


def _cam():
    from tools.ptz_facade import get_ptz_camera

    return get_ptz_camera()

try:
    from tools.sim_ptz_watch import (
        save_position_state,
        sleep_after_inference,
        sleep_after_move,
    )
except ImportError:
    def save_position_state(pan, tilt, fov_h=None):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        d = {"pan": round(float(pan), 2), "tilt": round(float(tilt), 2)}
        if fov_h is not None:
            d["fov_h"] = round(float(fov_h), 1)
        _STATE_FILE.write_text(json.dumps(d))

    def sleep_after_move():
        pass

    def sleep_after_inference():
        pass


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class SimulatedPTZ:
    """PTZ camera simulator backed by a panoramic image.

    Coordinate system:
        Pan:  0-360 degrees, wraps around.  0° = left edge of image.
        Tilt: 0° = bottom of image, tilt_range° = top of image.
              tilt_range is derived from the image aspect ratio so that
              angular pixel density is uniform in both axes.
        FOV:  Horizontal field of view in degrees (viewport width).
              Vertical FOV is derived for a 16:9 viewport aspect.
    """

    PAN_RANGE = 360.0

    def __init__(self, image_path=None, fov_h=60.0):
        path = Path(image_path) if image_path else _DEFAULT_PANORAMA
        self.img = Image.open(path)
        self.img_w, self.img_h = self.img.size

        self.ppd = self.img_w / self.PAN_RANGE
        self.tilt_range = round(self.img_h / self.ppd, 1)

        self.fov_h = fov_h
        self.fov_v = round(fov_h * 9.0 / 16.0, 1)

        self.pan, self.tilt = self._load_state()

        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                if "fov_h" in data:
                    self.fov_h = max(10.0, min(120.0, float(data["fov_h"])))
                    self.fov_v = round(self.fov_h * 9.0 / 16.0, 1)
                    half_v = self.fov_v / 2
                    self.tilt = max(half_v, min(self.tilt_range - half_v, self.tilt))
            except Exception:
                pass

    # -- state persistence --

    def _load_state(self) -> tuple[float, float]:
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                return float(data.get("pan", 180.0)), float(data.get("tilt", self.tilt_range / 2))
            except Exception:
                pass
        return 180.0, self.tilt_range / 2

    def _save_state(self):
        save_position_state(self.pan, self.tilt, self.fov_h)

    def set_fov_h(self, fov_h: float) -> dict:
        """Set horizontal field of view (10–120°). Clamps tilt if needed."""
        self.fov_h = max(10.0, min(120.0, float(fov_h)))
        self.fov_v = round(self.fov_h * 9.0 / 16.0, 1)
        half_v = self.fov_v / 2
        self.tilt = max(half_v, min(self.tilt_range - half_v, self.tilt))
        self._save_state()
        sleep_after_move()
        return self.get_position()

    # -- movement --

    def move_to(self, pan: float, tilt: float) -> dict:
        self.pan = pan % self.PAN_RANGE
        half_v = self.fov_v / 2
        self.tilt = max(half_v, min(self.tilt_range - half_v, tilt))
        self._save_state()
        sleep_after_move()
        return self.get_position()

    def pan_by(self, degrees: float) -> dict:
        return self.move_to(self.pan + degrees, self.tilt)

    def tilt_by(self, degrees: float) -> dict:
        return self.move_to(self.pan, self.tilt + degrees)

    def get_position(self) -> dict:
        return {
            "pan_deg": round(self.pan, 1),
            "tilt_deg": round(self.tilt, 1),
            "fov_h": self.fov_h,
            "fov_v": self.fov_v,
            "pan_range": self.PAN_RANGE,
            "tilt_range": self.tilt_range,
        }

    # -- viewport geometry --

    def _viewport_px(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom) in pixel coords.
        left/right may exceed [0, img_w] to signal wrapping."""
        cx = self.pan * self.ppd
        cy = (self.tilt_range - self.tilt) * self.ppd
        hw = self.fov_h * self.ppd / 2
        hh = self.fov_v * self.ppd / 2
        top = max(0, cy - hh)
        bot = min(self.img_h, cy + hh)
        return cx - hw, top, cx + hw, bot

    def _crop_viewport(self) -> Image.Image:
        """Extract the viewport region, handling panoramic wrapping."""
        left, top, right, bot = self._viewport_px()
        vp_w = int(right - left)
        vp_h = int(bot - top)

        if left < 0:
            lp = self.img.crop((int(left + self.img_w), int(top), self.img_w, int(bot)))
            rp = self.img.crop((0, int(top), int(right), int(bot)))
            out = Image.new("RGB", (vp_w, vp_h))
            out.paste(lp, (0, 0))
            out.paste(rp, (lp.width, 0))
            return out

        if right > self.img_w:
            lp = self.img.crop((int(left), int(top), self.img_w, int(bot)))
            rp = self.img.crop((0, int(top), int(right - self.img_w), int(bot)))
            out = Image.new("RGB", (vp_w, vp_h))
            out.paste(lp, (0, 0))
            out.paste(rp, (lp.width, 0))
            return out

        return self.img.crop((int(left), int(top), int(right), int(bot)))

    # -- outputs --

    def snapshot(self, filename: str = None) -> str:
        """Save the current viewport as a JPEG.  Returns the file path."""
        viewport = self._crop_viewport()
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_snapshot_{ts}.jpg"
        out = (_PROJECT_ROOT / filename).resolve()
        viewport.save(str(out), quality=92)
        return str(out)

    def overview(self, filename: str = None, scale: float = 0.3) -> str:
        """Save a panorama thumbnail with the viewport rectangle highlighted."""
        tw = int(self.img_w * scale)
        th = int(self.img_h * scale)
        thumb = self.img.resize((tw, th), Image.LANCZOS)
        draw = ImageDraw.Draw(thumb)

        left, top, right, bot = self._viewport_px()
        sl, st, sr, sb = left * scale, top * scale, right * scale, bot * scale
        lw = max(2, int(4 * scale * 4))

        if sl < 0:
            draw.rectangle([sl + tw, st, tw - 1, sb], outline="red", width=lw)
            draw.rectangle([0, st, sr, sb], outline="red", width=lw)
        elif sr > tw:
            draw.rectangle([sl, st, tw - 1, sb], outline="red", width=lw)
            draw.rectangle([0, st, sr - tw, sb], outline="red", width=lw)
        else:
            draw.rectangle([sl, st, sr, sb], outline="red", width=lw)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            except OSError:
                font = ImageFont.load_default()

        label = (f"Pan {self.pan:.1f}\u00b0  Tilt {self.tilt:.1f}\u00b0  "
                 f"FOV {self.fov_h:.0f}\u00b0\u00d7{self.fov_v:.0f}\u00b0")
        draw.text((12, th - 30), label, fill="black", font=font)
        draw.text((10, th - 32), label, fill="white", font=font)

        if filename is None:
            filename = "sim_ptz_overview.jpg"
        out = (_PROJECT_ROOT / filename).resolve()
        thumb.save(str(out), quality=90)
        return str(out)

    def composite(self, filename: str = None) -> str:
        """Side-by-side: overview (panorama + rect) on top, viewport on bottom."""
        scale = 0.3
        tw = int(self.img_w * scale)
        th = int(self.img_h * scale)

        overview_img = Image.open(self.overview("_tmp_overview.jpg", scale))
        viewport_img = self._crop_viewport()

        vp_display_w = tw
        vp_display_h = int(viewport_img.height * (vp_display_w / viewport_img.width))
        viewport_resized = viewport_img.resize((vp_display_w, vp_display_h), Image.LANCZOS)

        gap = 4
        composite = Image.new("RGB", (tw, th + gap + vp_display_h), (30, 30, 30))
        composite.paste(overview_img, (0, 0))
        composite.paste(viewport_resized, (0, th + gap))

        if filename is None:
            filename = "sim_ptz_composite.jpg"
        out = (_PROJECT_ROOT / filename).resolve()
        composite.save(str(out), quality=92)

        tmp = _PROJECT_ROOT / "_tmp_overview.jpg"
        if tmp.exists():
            tmp.unlink()

        return str(out)


# ---------------------------------------------------------------------------
# MSA Tool Plugins (auto-discovered by plugin loader)
# ---------------------------------------------------------------------------

class SimPTZMoveTool(BaseTool):
    name = "sim_ptz_move"
    description = (
        "Move the simulated PTZ camera to an absolute position. "
        "Args: pan (float, 0-360 degrees), tilt (float, 0-~153 degrees). "
        "Provide one or both.  Pan wraps; tilt clamps."
    )

    def run(self, pan=None, tilt=None, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        cam = _cam()
        if pan is not None and tilt is not None:
            r = cam.move_to(float(pan), float(tilt))
        elif pan is not None:
            r = cam.move_to(float(pan), cam.tilt)
        elif tilt is not None:
            r = cam.move_to(cam.pan, float(tilt))
        else:
            return "ERROR: provide pan and/or tilt in degrees"
        return json.dumps(r, indent=2)


class SimPTZPanTool(BaseTool):
    name = "sim_ptz_pan"
    description = (
        "Pan the simulated PTZ camera by a relative amount. "
        "Args: degrees (float — positive=right, negative=left)."
    )

    def run(self, degrees=0, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        return json.dumps(_cam().pan_by(float(degrees)), indent=2)


class SimPTZTiltTool(BaseTool):
    name = "sim_ptz_tilt"
    description = (
        "Tilt the simulated PTZ camera by a relative amount. "
        "Args: degrees (float — positive=up, negative=down)."
    )

    def run(self, degrees=0, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        return json.dumps(_cam().tilt_by(float(degrees)), indent=2)


class SimPTZPositionTool(BaseTool):
    name = "sim_ptz_position"
    description = "Get the current simulated PTZ camera position and FOV info."

    def run(self, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        return json.dumps(_cam().get_position(), indent=2)


class SimPTZSnapshotTool(BaseTool):
    name = "sim_ptz_snapshot"
    description = (
        "Capture a snapshot from the simulated PTZ camera's current viewport. "
        "Args: filename (str, optional). Returns path to saved JPEG."
    )

    def run(self, filename=None, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        return f"Snapshot saved to {_cam().snapshot(filename)}"


class SimPTZOverviewTool(BaseTool):
    name = "sim_ptz_overview"
    description = (
        "Generate a panorama overview showing the full 360-degree scene "
        "with the current viewport highlighted as a red rectangle. "
        "Args: filename (str, optional)."
    )

    def run(self, filename=None, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        return f"Overview saved to {_cam().overview(filename)}"


class SimPTZCompositeTool(BaseTool):
    name = "sim_ptz_composite"
    description = (
        "Generate a composite image: panorama overview on top with viewport "
        "rectangle, camera viewport on bottom. Full picture of what the "
        "camera sees and where it's pointing. Args: filename (str, optional)."
    )

    def run(self, filename=None, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        return f"Composite saved to {_cam().composite(filename)}"


class SimPTZDetectTool(BaseTool):
    name = "sim_ptz_detect"
    description = (
        "Run object detection on the simulated PTZ camera's current viewport. "
        "Args: model (str, default 'yolo' — also 'bioclip' or 'gemma4'), "
        "targets (str, default '*' — comma-separated class names for YOLO; "
        "for Gemma4, categories to find or '*' for all prominent objects), "
        "target_taxon (str, default '' — BioCLIP lineage filter, e.g. Mammalia or Animalia Chordata Mammalia), "
        "target (str, optional — alias for Gemma4 detection hint), "
        "max_soft_tokens (int, optional — Gemma4 visual token budget: 70|140|280|560|1120). "
        "Returns JSON list of detections with bbox, label, confidence."
    )

    def run(
        self,
        model="yolo",
        targets="*",
        target_taxon="",
        target="",
        max_soft_tokens=None,
        **kwargs,
    ) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        try:
            from tools.detectors import detect
        except ImportError:
            return "ERROR: detectors module not available"
        cam = _cam()
        viewport = cam._crop_viewport()
        if str(model).lower() == "gemma4":
            hint = (target or "").strip() or (
                targets if str(targets).strip() not in ("", "*") else ""
            )
            gkw: dict = {"target": hint}
            if max_soft_tokens is not None:
                gkw["max_soft_tokens"] = int(max_soft_tokens)
            result = detect(viewport, model="gemma4", **gkw)
        else:
            result = detect(
                viewport,
                model=model,
                targets=targets,
                target_taxon=target_taxon,
            )
        sleep_after_inference()
        return json.dumps(result, indent=2)


class SimPTZCaptionTool(BaseTool):
    name = "sim_ptz_caption"
    description = (
        "Generate a text caption describing the simulated PTZ camera's "
        "current viewport. "
        "Args: model (str, default 'bioclip' — also 'gemma4'), "
        "prompt (str, optional — Gemma4 instruction; ignored for BioCLIP), "
        "max_soft_tokens (int, optional — Gemma4 visual token budget). "
        "Returns caption text and timing."
    )

    def run(self, model="bioclip", prompt="", max_soft_tokens=None, **kwargs) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        try:
            from tools.detectors import caption
        except ImportError:
            return "ERROR: detectors module not available"
        cam = _cam()
        viewport = cam._crop_viewport()
        ckw = {}
        if str(model).lower() == "gemma4":
            if prompt:
                ckw["prompt"] = str(prompt)
            if max_soft_tokens is not None:
                ckw["max_soft_tokens"] = int(max_soft_tokens)
        result = caption(viewport, model=model, **ckw)
        sleep_after_inference()
        return json.dumps(result, indent=2)


class SimPTZMissionTool(BaseTool):
    name = "sim_ptz_mission"
    description = (
        "Run an agentic panorama mission: sweep or sample viewpoints, detect "
        "objects, dedupe across views, and return counts. "
        "Pass a natural-language mission in ``mission``. Examples: "
        "'scan for all animals', 'find cows', 'count animals', "
        "'random things', 'find Aves' (non-COCO targets use BioCLIP when available), "
        "'find forests near lakes' (semantic scene goals use Gemma4 when available). "
        "Args: mission (str, required), model (str, optional: yolo | bioclip | gemma_scene), "
        "random_views (int, default 10 — for random/explore mode), "
        "pan_step_ratio (float, default 0.82 — scan overlap), "
        "tilt_step_ratio (float, default 0.82 — grid mode vertical step), "
        "tilt (float, optional — fixed tilt for single-row scan), "
        "max_pan_stops (int, default 48 — cap pan positions per row), "
        "max_tilt_rows (int, default 64 — grid cap), "
        "max_total_stops (int, default 512 — grid cap). "
        "Returns JSON with summary.counts_by_label, unique_instances_estimated, "
        "unique_detections, and per-frame details."
    )

    def run(
        self,
        mission="",
        model=None,
        random_views=10,
        pan_step_ratio=0.82,
        tilt_step_ratio=0.82,
        tilt=None,
        max_pan_stops=48,
        max_tilt_rows=64,
        max_total_stops=512,
        **kwargs,
    ) -> str:
        if not HAS_PIL:
            return _MISSING_MSG
        if not str(mission).strip():
            return "ERROR: mission (str) is required, e.g. 'find all animals'"
        try:
            from tools.ptz_mission import run_mission
        except ImportError:
            return "ERROR: ptz_mission module not available"
        tilt_f = float(tilt) if tilt is not None else None
        out = run_mission(
            str(mission),
            model=model,
            random_views=int(random_views),
            pan_step_ratio=float(pan_step_ratio),
            tilt_step_ratio=float(tilt_step_ratio),
            tilt=tilt_f,
            max_pan_stops=int(max_pan_stops),
            max_tilt_rows=int(max_tilt_rows),
            max_total_stops=int(max_total_stops),
        )
        return json.dumps(out, indent=2, default=str)
