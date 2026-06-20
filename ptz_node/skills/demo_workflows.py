"""Canned Sage demo workflows — deterministic gateway sequences, no LLM looping.

Each demo is the scripted twin of an agentic test case in
``config/agentic_test_cases.yaml``: the reasoning LLM (or a human at the CLI)
triggers ONE skill call and the skill drives the :class:`SensorGateway`
step-by-step. Big sweeps (e.g. a full stitched-panorama scan with every vision
backend) therefore cannot wander, stall mid-plan, or blow the recursion limit.

CLI:    python -m ptz_node skill run demo --args '{"name":"panorama_scan"}'
        python -m ptz_node skill run demo --args '{"action":"list"}'
Agent:  run_skill("demo", '{"name":"wildfire_smoke_patrol"}')

Progress is printed to stderr per step so long demos are visibly alive.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from ptz_node.skills.base import BaseSkill, SkillContext, SkillResult

_FIRE_PROMPT = ("Describe fire or smoke risk cues in this scene: smoke, haze, "
                "flames, burn scars, dry fuel load.")
_AGRI_PROMPT = ("Describe crop vs bare soil, irrigation cues, vegetation "
                "stress, and nearby water or infrastructure.")


def _j(payload: str) -> dict[str, Any]:
    try:
        out = json.loads(payload)
        return out if isinstance(out, dict) else {"ok": False, "raw": out}
    except Exception:
        return {"ok": False, "error": "unparseable gateway reply",
                "raw": str(payload)[:300]}


def _progress(msg: str) -> None:
    print(f"[demo] {msg}", file=sys.stderr, flush=True)


def _det_summary(res: dict[str, Any], top: int = 12) -> dict[str, Any]:
    """Collapse a ptz_detect reply to counts + top-confidence labels."""
    r = res.get("result") or {}
    dets = r.get("detections") or []
    counts: dict[str, int] = {}
    best: dict[str, float] = {}
    for d in dets:
        lab = str(d.get("label", "?"))
        counts[lab] = counts.get(lab, 0) + 1
        conf = float(d.get("confidence") or 0.0)
        if conf > best.get(lab, 0.0):
            best[lab] = conf
    out: dict[str, Any] = {
        "n": len(dets),
        "counts": counts,
        "top_conf": [{"label": lab, "conf": round(c, 3)}
                     for lab, c in sorted(best.items(), key=lambda kv: -kv[1])[:top]],
    }
    err = r.get("error") or res.get("error")
    if err or not res.get("ok", True):
        out["error"] = str(err or "gateway call failed")
    return out


def _cap_summary(res: dict[str, Any], max_chars: int = 700) -> dict[str, Any]:
    r = res.get("result") or {}
    caption = str(r.get("caption", ""))[:max_chars]
    out: dict[str, Any] = {"caption": caption}
    err = r.get("error") or res.get("error")
    if err or not res.get("ok", True):
        out["error"] = str(err or "gateway call failed")
    elif not caption.strip():
        # Empty caption with no error = the vision model returned no text (e.g. a
        # tiny gemma4 tag). Flag it so it doesn't read as a silent blank success.
        out["note"] = "model returned an empty caption (check the gemma4 vision tag)"
    return out


def _pan_of(gw) -> float:
    pos = (_j(gw.ptz_get_position()).get("result") or {})
    return float(pos.get("pan_deg") or 0.0)


class DemoWorkflowsSkill(BaseSkill):
    name = "demo"
    description = (
        "Run a canned Sage demo workflow as ONE deterministic call (scripted gateway "
        "sequence — no multi-cycle tool planning). Demos: edge_gateway_preflight, "
        "ptz_multimodel_scientific_survey, wildfire_smoke_patrol, "
        "aves_biodiversity_scan, land_cover_agriculture_scene, panorama_scan. "
        "panorama_scan sweeps the FULL panorama heading-by-heading with every "
        "available vision backend — ALWAYS use it for 'scan the whole panorama / "
        "every subsection with every backend' requests instead of looping PTZ tools. "
        "Args: {'name': <demo_id>}; panorama_scan options: backends (list), step_deg, "
        "tilt, target_taxon, time_budget_s (default 900). action=list enumerates."
    )
    agent_callable = True

    _DEMOS = {
        "edge_gateway_preflight": "device catalog, sensor census, detector status, position, snapshot",
        "ptz_multimodel_scientific_survey": "YOLO + BioCLIP + Gemma4 across three headings",
        "wildfire_smoke_patrol": "YOLO smoke/fire classes on two headings + Gemma4 fire-risk read",
        "aves_biodiversity_scan": "BioCLIP target_taxon=Aves at two pan stops",
        "land_cover_agriculture_scene": "Gemma4 agriculture captions at two viewpoints",
        "panorama_scan": "full panorama sweep, every backend on every heading (tiled YOLO)",
    }

    def run(self, ctx: SkillContext) -> SkillResult:
        action = str(ctx.args.get("action", "run")).lower()
        if action == "list":
            return SkillResult(ok=True, skill=self.name,
                               summary=f"{len(self._DEMOS)} demos available",
                               data={"demos": self._DEMOS})

        name = str(ctx.args.get("name", "")).strip()
        if name not in self._DEMOS:
            return SkillResult(ok=False, skill=self.name,
                               summary=f"unknown demo {name!r}",
                               data={"demos": sorted(self._DEMOS)})

        gw = ctx.gateway()
        t0 = time.time()
        try:
            handler = getattr(self, f"_demo_{name}")
            summary, data, artifacts = handler(gw, ctx.args)
        except Exception as exc:  # demos must never crash the loop
            return SkillResult(ok=False, skill=self.name,
                               summary=f"{name} failed: {type(exc).__name__}: {exc}",
                               data={"demo": name})

        data["demo"] = name
        data["elapsed_s"] = round(time.time() - t0, 1)
        artifacts.append(self._write_artifact(name, data))
        return SkillResult(ok=not data.get("degraded", False) or True,
                           skill=self.name, summary=summary, data=data,
                           artifacts=artifacts)

    # ------------------------------------------------------------------ #
    # demos
    # ------------------------------------------------------------------ #

    def _demo_edge_gateway_preflight(self, gw, a) -> tuple[str, dict, list]:
        devices = gw.list_devices()
        readings = {d["id"]: _j(gw.read_sensor(d["id"]))
                    for d in devices if d.get("kind") == "sensor"}
        detectors = _j(gw.detector_status())
        pos = _j(gw.ptz_get_position())
        snap = _j(gw.ptz_snapshot(filename="demo_preflight.jpg"))
        snap_path = (snap.get("result") or {}).get("path")
        data = {
            "gateway_ok": bool(devices) and pos.get("ok", False),
            "devices": [{k: d.get(k) for k in ("id", "kind", "interface", "backend")}
                        for d in devices],
            "sensor_readings": readings,
            "detectors": detectors.get("models", detectors),
            "position": pos.get("result"),
            "snapshot_path": snap_path,
        }
        ok = data["gateway_ok"]
        return (f"preflight {'OK' if ok else 'DEGRADED'}: {len(devices)} device(s), "
                f"snapshot={'yes' if snap_path else 'NO'}",
                data, [snap_path] if snap_path else [])

    def _demo_ptz_multimodel_scientific_survey(self, gw, a) -> tuple[str, dict, list]:
        steps: list[dict] = []
        _progress("survey 1/3: tiled YOLO at current heading")
        steps.append({"backend": "yolo", "pan_deg": _pan_of(gw),
                      **_det_summary(_j(gw.ptz_detect(model="yolo", targets="*",
                                                      tile=True)))})
        _j(gw.ptz_pan_by(45))
        taxon = str(a.get("target_taxon", "Mammalia"))
        _progress(f"survey 2/3: BioCLIP {taxon} at +45°")
        steps.append({"backend": "bioclip", "pan_deg": _pan_of(gw), "taxon": taxon,
                      **_det_summary(_j(gw.ptz_detect(model="bioclip",
                                                      target_taxon=taxon)))})
        _j(gw.ptz_pan_by(45))
        _progress("survey 3/3: Gemma4 scene caption at +90°")
        steps.append({"backend": "gemma4", "pan_deg": _pan_of(gw),
                      **_cap_summary(_j(gw.ptz_caption(
                          model="gemma4",
                          prompt="Describe habitat structure and any animals.")))})
        n = sum(s.get("n", 0) for s in steps)
        return (f"survey complete: 3 headings, {n} detection(s) across "
                f"yolo/bioclip/gemma4", {"steps": steps}, [])

    def _demo_wildfire_smoke_patrol(self, gw, a) -> tuple[str, dict, list]:
        headings: list[dict] = []
        _progress("patrol 1/2: YOLO smoke/fire at current heading")
        first = _det_summary(_j(gw.ptz_detect(model="yolo", targets="smoke,fire")))
        if first.get("n", 0) == 0:
            first = {"note": "no smoke/fire classes; widened to all targets",
                     **_det_summary(_j(gw.ptz_detect(model="yolo", targets="*")))}
        headings.append({"pan_deg": _pan_of(gw), "yolo": first})
        _j(gw.ptz_pan_by(90))
        _progress("patrol 2/2: YOLO all-targets at +90°")
        headings.append({"pan_deg": _pan_of(gw),
                         "yolo": _det_summary(_j(gw.ptz_detect(model="yolo",
                                                               targets="*")))})
        _progress("patrol: Gemma4 fire-risk read")
        risk = _cap_summary(_j(gw.ptz_caption(model="gemma4", prompt=_FIRE_PROMPT)))
        smoke_hits = sum(
            c for h in headings
            for lab, c in (h["yolo"].get("counts") or {}).items()
            if lab.lower() in ("smoke", "fire")
        )
        verdict = ("ALERT: smoke/fire signatures detected — notify operator"
                   if smoke_hits else "no smoke/fire indicators; routine rescan advised")
        return (f"wildfire patrol: 2 headings checked, {smoke_hits} smoke/fire hit(s) — "
                f"{verdict}",
                {"headings": headings, "fire_risk_caption": risk,
                 "smoke_fire_hits": smoke_hits, "recommendation": verdict}, [])

    def _demo_aves_biodiversity_scan(self, gw, a) -> tuple[str, dict, list]:
        taxon = str(a.get("target_taxon", "Aves"))
        views: list[dict] = []
        _progress(f"aves 1/2: BioCLIP {taxon} at current heading")
        views.append({"pan_deg": _pan_of(gw),
                      **_det_summary(_j(gw.ptz_detect(model="bioclip",
                                                      target_taxon=taxon)))})
        _j(gw.ptz_pan_by(60))
        _progress(f"aves 2/2: BioCLIP {taxon} at +60°")
        views.append({"pan_deg": _pan_of(gw),
                      **_det_summary(_j(gw.ptz_detect(model="bioclip",
                                                      target_taxon=taxon)))})
        all_labels = sorted({lab for v in views for lab in (v.get("counts") or {})})
        return (f"{taxon} scan: 2 views, {sum(v.get('n', 0) for v in views)} "
                f"candidate(s), {len(all_labels)} unique label(s)",
                {"taxon": taxon, "views": views, "unique_labels": all_labels,
                 "next_step": "pan +60 again for a fuller transect"}, [])

    def _demo_land_cover_agriculture_scene(self, gw, a) -> tuple[str, dict, list]:
        views: list[dict] = []
        snap = _j(gw.ptz_snapshot(filename="demo_agri_view1.jpg"))
        _progress("agri 1/2: Gemma4 land-cover caption at current heading")
        views.append({"pan_deg": _pan_of(gw),
                      "snapshot": (snap.get("result") or {}).get("path"),
                      **_cap_summary(_j(gw.ptz_caption(model="gemma4",
                                                       prompt=_AGRI_PROMPT)))})
        _j(gw.ptz_pan_by(45))
        _progress("agri 2/2: Gemma4 land-cover caption at +45°")
        views.append({"pan_deg": _pan_of(gw),
                      **_cap_summary(_j(gw.ptz_caption(model="gemma4",
                                                       prompt=_AGRI_PROMPT)))})
        arts = [v["snapshot"] for v in views if v.get("snapshot")]
        return ("land-cover assessment: 2 viewpoints captioned; validate with soil "
                "moisture + weather sensors",
                {"views": views,
                 "ground_truth_suggestion": "soil moisture, weather station"}, arts)

    def _demo_panorama_scan(self, gw, a) -> tuple[str, dict, list]:
        budget = float(a.get("time_budget_s", 900))
        t0 = time.time()

        det = _j(gw.detector_status())
        # detector_status is either flat ({"yolo": true, ...}) or nested under "models"
        models = det.get("models") if isinstance(det.get("models"), dict) else det
        requested = list(a.get("backends") or ["yolo", "bioclip", "gemma4"])
        backends = []
        for b in requested:
            info = models.get(b)
            if info is False or (isinstance(info, dict)
                                 and info.get("available") is False):
                continue
            backends.append(b)
        if not backends:
            return ("panorama scan aborted: no requested vision backend available",
                    {"detectors": models, "requested": requested}, [])

        pos = _j(gw.ptz_get_position()).get("result") or {}
        fov = float(pos.get("fov_h") or 60.0)
        pan_range = float(pos.get("pan_range") or 360.0)
        step = float(a.get("step_deg") or fov)
        tilt = a.get("tilt", pos.get("tilt_deg", 0.0))
        taxon = str(a.get("target_taxon", "Animalia"))
        headings = [i * step for i in range(max(1, int(pan_range // step)))]

        results: dict[str, dict[str, Any]] = {b: {} for b in backends}
        skipped: list[str] = []
        for i, pan in enumerate(headings, 1):
            if time.time() - t0 > budget:
                skipped.append(f"headings {i}..{len(headings)} (time budget)")
                break
            _j(gw.ptz_move_to(pan, float(tilt)))
            key = f"pan_{pan:.0f}"
            for b in backends:
                if time.time() - t0 > budget:
                    skipped.append(f"{key}:{b} onward (time budget)")
                    break
                _progress(f"panorama heading {i}/{len(headings)} (pan={pan:.0f}°) "
                          f"backend={b} [{time.time() - t0:.0f}s]")
                if b == "yolo":
                    results[b][key] = _det_summary(
                        _j(gw.ptz_detect(model="yolo", targets="*", tile=True)))
                elif b in ("bioclip", "bioclip2"):
                    results[b][key] = _det_summary(
                        _j(gw.ptz_detect(model=b, target_taxon=taxon)))
                else:  # gemma4 or other captioners
                    results[b][key] = _cap_summary(
                        _j(gw.ptz_caption(model=b,
                                          prompt="List animals and notable objects "
                                                 "or events in this view.")))

        by_backend = {}
        for b, views in results.items():
            labels: dict[str, int] = {}
            for v in views.values():
                for lab, c in (v.get("counts") or {}).items():
                    labels[lab] = labels.get(lab, 0) + c
            by_backend[b] = {"headings_scanned": len(views),
                             "total_detections": sum(labels.values()),
                             "labels": labels}
        data = {"backends": backends, "headings_deg": headings, "tilt_deg": tilt,
                "step_deg": step, "target_taxon": taxon, "results": results,
                "rollup": by_backend, "skipped": skipped, "detectors": models}
        return (f"panorama scan: {len(headings)} heading(s) × {len(backends)} "
                f"backend(s); " +
                "; ".join(f"{b}: {by_backend[b]['total_detections']} det"
                          for b in backends if b in by_backend) +
                (f"; SKIPPED {len(skipped)} (budget)" if skipped else ""),
                data, [])

    # ------------------------------------------------------------------ #

    def _write_artifact(self, name: str, data: dict[str, Any]) -> str:
        from ptz_node.paths import local_data_root, stamp, write_json

        path = write_json(local_data_root() / "demos" / f"{name}_{stamp()}.json", data)
        return str(path)
