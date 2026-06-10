"""HTTP façade on top of :class:`~ptz_node.sensor_gateway.SensorGateway`.

Run::

    PYTHONPATH=. uvicorn ptz_node.api_server:app --host 0.0.0.0 --port 8848
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException

from ptz_node.config_loader import load_config
from ptz_node.sensor_gateway import SensorGateway

app = FastAPI(title="Jetson Sensor Gateway API", version="0.1.0")


def _gw() -> SensorGateway:
    return SensorGateway(load_config())


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/v1/devices")
def devices() -> list[dict[str, Any]]:
    return _gw().list_devices()


@app.get("/v1/devices/{device_id}/capabilities")
def device_capabilities(device_id: str) -> dict[str, Any]:
    out = _gw().capabilities(device_id)
    if not out.get("ok"):
        raise HTTPException(status_code=404, detail=out.get("error", "unknown device"))
    return out


@app.get("/v1/devices/{device_id}/read")
def device_read(device_id: str) -> dict[str, Any]:
    out = json.loads(_gw().read_sensor(device_id))
    if not out.get("ok"):
        raise HTTPException(status_code=400, detail=out.get("error", "read failed"))
    return out


@app.post("/v1/devices/{device_id}/invoke")
def device_invoke(device_id: str, body: dict[str, Any]) -> dict[str, Any]:
    capability = body.get("capability")
    if not capability:
        raise HTTPException(status_code=422,
                            detail='need JSON {"capability": str, "params": {...}}')
    params = body.get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(status_code=422, detail='"params" must be an object')
    out = _gw().invoke_obj(device_id, str(capability), params)
    if not out.get("ok"):
        raise HTTPException(status_code=400, detail=out.get("error", "invoke failed"))
    return out


@app.get("/v1/ptz/{device_id}/position")
def ptz_position(device_id: str) -> dict[str, Any]:
    raw = _gw().ptz_get_position(device_id=device_id)
    try:
        return dict(json.loads(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=raw) from exc


@app.post("/v1/ptz/{device_id}/move")
def ptz_move(device_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        pan = float(body["pan"])
        tilt = float(body["tilt"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail="need JSON {pan: float, tilt: float}") from e
    raw = _gw().ptz_move_to(pan, tilt, device_id=device_id)
    return json.loads(raw)


@app.post("/v1/ptz/{device_id}/pan-by")
def ptz_pan_by(device_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        deg = float(body["degrees"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422,
                            detail='need JSON {"degrees": number}') from e
    raw = _gw().ptz_pan_by(deg, device_id=device_id)
    return json.loads(raw)


@app.post("/v1/ptz/{device_id}/tilt-by")
def ptz_tilt_by(device_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        deg = float(body["degrees"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422,
                            detail='need JSON {"degrees": number}') from e
    raw = _gw().ptz_tilt_by(deg, device_id=device_id)
    return json.loads(raw)


@app.post("/v1/ptz/{device_id}/fov")
def ptz_fov(device_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        fov = float(body["fov_h"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422,
                            detail='need JSON {"fov_h": number}') from e
    raw = _gw().ptz_set_fov_h(fov, device_id=device_id)
    return json.loads(raw)


@app.post("/v1/ptz/{device_id}/snapshot")
def ptz_snapshot(device_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    fname = body.get("filename")
    raw = _gw().ptz_snapshot(filename=fname, device_id=device_id)
    return json.loads(raw)


@app.get("/v1/debug/doctor")
def debug_doctor() -> dict[str, Any]:
    from ptz_node.debug_report import run_doctor

    return run_doctor(load_config())


@app.get("/v1/debug/status")
def debug_status() -> dict[str, Any]:
    from ptz_node.paths import local_data_root

    path = local_data_root() / "status.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    from ptz_node.debug_report import run_doctor

    run_doctor(load_config())
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/v1/debug/latest-run")
def debug_latest_run() -> dict[str, Any]:
    from ptz_node.paths import debug_dir

    path = debug_dir() / "latest_run.json"
    if not path.is_file():
        return {"error": "no runs yet", "hint": "python -m ptz_node run '...'"}
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/v1/ptz/{device_id}/detect")
def ptz_detect_route(device_id: str, body: dict[str, Any]) -> dict[str, Any]:
    raw = _gw().ptz_detect(
        model=str(body.get("model", "yolo")),
        targets=str(body.get("targets", "*")),
        target_taxon=str(body.get("target_taxon", "")),
        target=str(body.get("target", "")),
        max_soft_tokens=body.get("max_soft_tokens"),
        tile=bool(body.get("tile", False)),
        tile_size=body.get("tile_size"),
        tile_overlap=int(body.get("tile_overlap", 0) or 0),
        tile_iou=float(body.get("tile_iou", 0.45)),
        device_id=device_id,
    )
    return json.loads(raw)


@app.get("/v1/detectors/status")
def detectors_status() -> dict[str, Any]:
    return json.loads(_gw().detector_status())


# --- skills & self-healing -------------------------------------------------

@app.get("/v1/skills")
def skills_list() -> list[dict[str, Any]]:
    from ptz_node.skills import SkillRegistry

    return SkillRegistry(load_config()).list_skills()


@app.post("/v1/skills/{name}/run")
def skills_run(name: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    from ptz_node.skills import SkillRegistry

    reg = SkillRegistry(load_config())
    if not reg.has(name):
        raise HTTPException(status_code=404, detail=f"unknown skill {name!r}")
    args = (body or {}).get("args") or {}
    return reg.run(name, args).as_dict()


@app.post("/v1/debug/snapshot")
def debug_snapshot(body: dict[str, Any] | None = None) -> dict[str, Any]:
    from ptz_node.skills import SkillRegistry

    force = bool((body or {}).get("force", False))
    return SkillRegistry(load_config()).run(
        "self_diagnosis", {"action": "snapshot", "force": force}).as_dict()


@app.get("/v1/heal/pending")
def heal_pending() -> list[dict[str, Any]]:
    from ptz_node.self_healing.healer import Healer

    return Healer(load_config()).list_pending()


@app.post("/v1/heal/{heal_id}/approve")
def heal_approve(heal_id: str) -> dict[str, Any]:
    from ptz_node.self_healing.healer import Healer

    out = Healer(load_config()).approve(heal_id)
    if not out.get("ok"):
        raise HTTPException(status_code=400, detail=out.get("error", "approve failed"))
    return out


@app.post("/v1/heal/{heal_id}/reject")
def heal_reject(heal_id: str) -> dict[str, Any]:
    from ptz_node.self_healing.healer import Healer

    out = Healer(load_config()).reject(heal_id)
    if not out.get("ok"):
        raise HTTPException(status_code=404, detail=out.get("error", "reject failed"))
    return out
