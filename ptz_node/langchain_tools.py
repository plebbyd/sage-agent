"""LangChain tool bindings that only call :class:`ptz_node.sensor_gateway.SensorGateway`.

Two layers of tools are exposed to the agent:

  * **Generic** — ``sensor_list_devices`` / ``sensor_capabilities`` /
    ``sensor_read`` / ``sensor_invoke`` work for *any* managed device, including
    sensors added later (weather, serial, GPIO, …) with no code changes here.
  * **Typed PTZ** — ergonomic, schema-rich tools (``ptz_move_to`` …) that models
    use far more reliably than a generic ``invoke``. Registered only when a PTZ
    camera is actually present.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from ptz_node.sensor_gateway import SensorGateway


def _has_ptz(gateway: SensorGateway) -> bool:
    try:
        return any(d.get("kind") == "ptz_camera" for d in gateway.list_devices())
    except Exception:
        return True  # assume yes; PTZ tools self-report errors if missing


def build_gateway_tools(gateway: SensorGateway) -> list:
    # ----- generic device tools (any sensor / actuator) -----------------

    @tool
    def sensor_list_devices() -> str:
        """List every managed device: id, kind, interface, backend, capabilities."""
        return json.dumps(gateway.list_devices(), indent=2, default=str)

    @tool
    def sensor_capabilities(device_id: str) -> str:
        """List the capabilities (callable verbs) of one device by id."""
        return json.dumps(gateway.capabilities(device_id), indent=2, default=str)

    @tool
    def sensor_read(device_id: str) -> str:
        """Take a reading from a read-only sensor device (e.g. sensor:system_stats)."""
        return gateway.read_sensor(device_id)

    @tool
    def sensor_invoke(device_id: str, capability: str,
                      params_json: str = "{}") -> str:
        """Invoke any device capability.

        device_id: from sensor_list_devices
        capability: from sensor_capabilities
        params_json: JSON object of arguments, e.g. '{"pan": 30, "tilt": 0}'
        """
        try:
            params = json.loads(params_json) if params_json.strip() else {}
            if not isinstance(params, dict):
                raise ValueError("params_json must be a JSON object")
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"bad params_json: {exc}"})
        return gateway.invoke(device_id, capability, params)

    # ----- skill tools (agent-callable skills only) ---------------------

    @tool
    def list_skills() -> str:
        """List modular skills available on this node (name, description)."""
        try:
            from ptz_node.skills import SkillRegistry

            skills = SkillRegistry(gateway._cfg).list_skills()
            return json.dumps([s for s in skills if s.get("agent_callable")],
                              indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    def run_skill(name: str, args_json: str = "{}") -> str:
        """Run an agent-callable skill once. args_json is a JSON object of arguments.

        Example: run_skill("self_diagnosis", '{"action": "status"}')
        """
        try:
            from ptz_node.skills import SkillRegistry

            reg = SkillRegistry(gateway._cfg)
            if not reg.has(name):
                return json.dumps({"ok": False, "error": f"unknown skill {name!r}",
                                   "known": reg.names()})
            if not reg.get(name).agent_callable:
                return json.dumps({"ok": False,
                                   "error": f"skill {name!r} is not agent-callable"})
            params = json.loads(args_json) if args_json.strip() else {}
            return json.dumps(reg.run(name, params).as_dict(), indent=2, default=str)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})

    generic = [sensor_list_devices, sensor_capabilities, sensor_read, sensor_invoke,
               list_skills, run_skill]

    if not _has_ptz(gateway):
        return generic

    # ----- typed PTZ tools ----------------------------------------------

    @tool
    def detector_status() -> str:
        """Report which vision models are available: yolo, bioclip/bioclip2, gemma4."""
        return gateway.detector_status()

    @tool
    def ptz_get_position(device_id: str | None = None) -> str:
        """Read pan, tilt, FOV, and limits for the managed PTZ (sim or Reolink)."""
        return gateway.ptz_get_position(device_id=device_id)

    @tool
    def ptz_move_to(pan: float, tilt: float, device_id: str | None = None) -> str:
        """Move PTZ to absolute pan (degrees) and tilt."""
        return gateway.ptz_move_to(pan, tilt, device_id=device_id)

    @tool
    def ptz_pan_by(degrees: float, device_id: str | None = None) -> str:
        """Pan relative by degrees (positive right)."""
        return gateway.ptz_pan_by(degrees, device_id=device_id)

    @tool
    def ptz_tilt_by(degrees: float, device_id: str | None = None) -> str:
        """Tilt relative by degrees."""
        return gateway.ptz_tilt_by(degrees, device_id=device_id)

    @tool
    def ptz_set_fov_h(fov_h_degrees: float, device_id: str | None = None) -> str:
        """Set horizontal field of view in degrees."""
        return gateway.ptz_set_fov_h(fov_h_degrees, device_id=device_id)

    @tool
    def ptz_take_snapshot(filename: str | None = None,
                          device_id: str | None = None) -> str:
        """Save a JPEG from the current PTZ viewport; returns filesystem path."""
        return gateway.ptz_snapshot(filename=filename, device_id=device_id)

    @tool
    def ptz_detect(model: str = "yolo", targets: str = "*",
                   target_taxon: str = "", target: str = "",
                   max_soft_tokens: int | None = None,
                   tile: bool = False, tile_size: int | None = None,
                   tile_overlap: int = 0, tile_iou: float = 0.45,
                   device_id: str | None = None) -> str:
        """Detect objects/species/scenes on the PTZ viewport.

        model: yolo | bioclip (BioCLIP 2) | bioclip2 | gemma4
        targets: YOLO class filter or Gemma4 category hint (* = all)
        target_taxon: BioCLIP lineage e.g. Mammalia, Aves, Animalia Chordata Mammalia
        tile: slice a large viewport into model-sized tiles for batch inference
              (preserves small-object resolution; YOLO tiles run batched).
        tile_size: override tile edge in px (default: model input size, e.g. 640).
        tile_overlap: tile overlap in px (default 0).
        tile_iou: IoU threshold for merging duplicate boxes across tiles.
        """
        return gateway.ptz_detect(model=model, targets=targets,
                                  target_taxon=target_taxon, target=target,
                                  max_soft_tokens=max_soft_tokens,
                                  tile=tile, tile_size=tile_size,
                                  tile_overlap=tile_overlap, tile_iou=tile_iou,
                                  device_id=device_id)

    @tool
    def ptz_caption(model: str = "bioclip", prompt: str = "",
                    max_soft_tokens: int | None = None,
                    device_id: str | None = None) -> str:
        """Generate a scientific caption for the current PTZ viewport."""
        return gateway.ptz_caption(model=model, prompt=prompt,
                                   max_soft_tokens=max_soft_tokens,
                                   device_id=device_id)

    return generic + [
        detector_status,
        ptz_get_position,
        ptz_move_to,
        ptz_pan_by,
        ptz_tilt_by,
        ptz_set_fov_h,
        ptz_take_snapshot,
        ptz_detect,
        ptz_caption,
    ]
