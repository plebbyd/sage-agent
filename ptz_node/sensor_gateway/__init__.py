"""Sensor / actuator management façade — the single choke point for hardware.

LangGraph tools, the REST API, and the MCP server import only :class:`SensorGateway`.
It owns a :class:`~ptz_node.sensor_gateway.registry.DeviceRegistry` of pluggable
drivers (PTZ camera + every auto-discovered ptz-agent sensor) and exposes both:

  * a **generic** surface — ``list_devices`` / ``capabilities`` / ``invoke`` /
    ``read_sensor`` — that works for any device, including ones added later; and
  * **typed PTZ helpers** (``ptz_get_position`` …) kept for ergonomic tools and
    backward compatibility.

All methods return JSON strings so callers (tools / MCP) can pass results to a
model unchanged; structured callers use :meth:`invoke_obj` / :meth:`list_devices`.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ptz_node.sensor_gateway.base import DriverError
from ptz_node.sensor_gateway.registry import DeviceRegistry


class SensorGateway:
    def __init__(self, cfg: dict[str, Any] | None = None, *,
                 ptz_agent_root: str | None = None) -> None:
        self._cfg = cfg or {}
        # Publish the configured local vision model to the vendored gemma4 detector
        # (reads GEMMA4_OLLAMA_MODEL). setdefault → an explicit env var still wins.
        gemma4_model = (self._cfg.get("vision") or {}).get("gemma4_model")
        if gemma4_model:
            os.environ.setdefault("GEMMA4_OLLAMA_MODEL", str(gemma4_model))
        self._registry = DeviceRegistry(self._cfg, ptz_agent_root=ptz_agent_root)

    @property
    def registry(self) -> DeviceRegistry:
        return self._registry

    # ------------------------------------------------------------------ #
    # Generic device surface (works for arbitrary sensors)
    # ------------------------------------------------------------------ #

    def list_devices(self) -> list[dict[str, Any]]:
        return [d.as_dict() for d in self._registry.devices()]

    def capabilities(self, device_id: str) -> dict[str, Any]:
        try:
            drv = self._registry.get(device_id)
        except DriverError as exc:
            return {"ok": False, "error": str(exc)}
        info = drv.describe()
        return {"ok": True, "device_id": device_id, "kind": info.kind,
                "capabilities": info.capabilities, "read_only": info.read_only}

    def invoke_obj(self, device_id: str | None, capability: str,
                   params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Structured invoke (raises nothing; returns ok/error envelope)."""
        try:
            result = self._registry.invoke(device_id, capability, params)
            return {"ok": True, "device_id": device_id or self._registry.default_ptz_id,
                    "capability": capability, "result": result}
        except DriverError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # hardware / model faults
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def invoke(self, device_id: str | None, capability: str,
               params: dict[str, Any] | None = None) -> str:
        return json.dumps(self.invoke_obj(device_id, capability, params),
                          indent=2, default=str)

    def read_sensor(self, device_id: str) -> str:
        return self.invoke(device_id, "read")

    def device_status(self, device_id: str) -> str:
        try:
            return json.dumps(self._registry.get(device_id).status(),
                              indent=2, default=str)
        except DriverError as exc:
            return json.dumps({"ok": False, "error": str(exc)})

    # ------------------------------------------------------------------ #
    # Detector availability (node-level)
    # ------------------------------------------------------------------ #

    def detector_status(self) -> str:
        try:
            return json.dumps(self._registry.ptz_driver().detector_status(),
                              indent=2, default=str)
        except Exception as exc:
            return json.dumps({"models": {}, "error": str(exc)})

    # ------------------------------------------------------------------ #
    # Typed PTZ helpers (delegate to the registry's PTZ driver)
    # ------------------------------------------------------------------ #

    def _ptz(self, device_id: str | None, capability: str,
             **params: Any) -> str:
        did = device_id or self._registry.default_ptz_id
        # Validate it really is the PTZ device for friendlier errors.
        try:
            drv = self._registry.get(did)
        except DriverError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        if drv.kind != "ptz_camera":
            return json.dumps({"ok": False,
                               "error": f"{did!r} is not a PTZ camera"})
        return self.invoke(did, capability, params)

    def ptz_get_position(self, device_id: str | None = None) -> str:
        return self._ptz(device_id, "get_position")

    def ptz_move_to(self, pan: float, tilt: float,
                    device_id: str | None = None) -> str:
        return self._ptz(device_id, "move_to", pan=pan, tilt=tilt)

    def ptz_pan_by(self, degrees: float, device_id: str | None = None) -> str:
        return self._ptz(device_id, "pan_by", degrees=degrees)

    def ptz_tilt_by(self, degrees: float, device_id: str | None = None) -> str:
        return self._ptz(device_id, "tilt_by", degrees=degrees)

    def ptz_set_fov_h(self, degrees: float, device_id: str | None = None) -> str:
        return self._ptz(device_id, "set_fov_h", fov_h=degrees)

    def ptz_snapshot(self, filename: str | None = None,
                     device_id: str | None = None) -> str:
        return self._ptz(device_id, "snapshot", filename=filename)

    def ptz_detect(self, model: str = "yolo", targets: str = "*",
                   target_taxon: str = "", target: str = "",
                   max_soft_tokens: int | None = None,
                   tile: bool = False, tile_size: int | None = None,
                   tile_overlap: int = 0, tile_iou: float = 0.45,
                   device_id: str | None = None) -> str:
        return self._ptz(device_id, "detect", model=model, targets=targets,
                         target_taxon=target_taxon, target=target,
                         max_soft_tokens=max_soft_tokens,
                         tile=tile, tile_size=tile_size,
                         tile_overlap=tile_overlap, tile_iou=tile_iou)

    def ptz_caption(self, model: str = "bioclip", prompt: str = "",
                    max_soft_tokens: int | None = None,
                    device_id: str | None = None) -> str:
        return self._ptz(device_id, "caption", model=model, prompt=prompt,
                         max_soft_tokens=max_soft_tokens)


_GATEWAY_SINGLETON: SensorGateway | None = None


def get_gateway(cfg: dict[str, Any] | None = None) -> SensorGateway:
    global _GATEWAY_SINGLETON
    if cfg is None:
        if _GATEWAY_SINGLETON is None:
            from ptz_node.config_loader import load_config

            _GATEWAY_SINGLETON = SensorGateway(load_config())
        return _GATEWAY_SINGLETON
    return SensorGateway(cfg)
