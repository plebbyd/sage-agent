"""Device registry — discovers and routes to all drivers.

This is the heart of the management layer. It assembles the set of devices the
node exposes:

  * One (or more) PTZ cameras (sim or Reolink, via ptz-agent).
  * Every auto-discovered ``ptz-agent/sensors/*`` plugin (system stats today;
    weather / serial / USB / I2C / GPIO / ethernet as users add them).

Adding a sensor type is therefore a *drop-in* operation on the ptz-agent side —
no edits here — which is exactly the extensibility the node was meant to have.
"""

from __future__ import annotations

import logging
from typing import Any

from ptz_node.bootstrap import bootstrap_ptz_agent_runtime
from ptz_node.sensor_gateway.base import BaseDriver, DeviceInfo, DriverError
from ptz_node.sensor_gateway.drivers.ptz_camera import PTZCameraDriver
from ptz_node.sensor_gateway.drivers.msa_sensor import MsaSensorDriver

logger = logging.getLogger(__name__)


class DeviceRegistry:
    def __init__(self, cfg: dict[str, Any] | None = None, *,
                 ptz_agent_root: str | None = None) -> None:
        self._cfg = cfg or {}
        self._ptz_agent_root = (
            ptz_agent_root
            if ptz_agent_root is not None
            else (self._cfg.get("ptz_agent_root") or None)
        )
        gw = self._cfg.get("gateway") or {}
        self._default_ptz_id = str(gw.get("default_ptz_id", "ptz_primary"))
        self._enable_sensors = bool(gw.get("enable_msa_sensors", True))
        self._drivers: dict[str, BaseDriver] | None = None

    # -- discovery ---------------------------------------------------------

    def _build(self) -> dict[str, BaseDriver]:
        drivers: dict[str, BaseDriver] = {}

        ptz = PTZCameraDriver(self._default_ptz_id, ptz_agent_root=self._ptz_agent_root)
        drivers[ptz.device_id] = ptz

        if self._enable_sensors:
            for drv in self._discover_msa_sensors():
                if drv.device_id in drivers:
                    logger.warning("device id clash, skipping sensor %s", drv.device_id)
                    continue
                drivers[drv.device_id] = drv

        return drivers

    def _discover_msa_sensors(self) -> list[MsaSensorDriver]:
        out: list[MsaSensorDriver] = []
        try:
            bootstrap_ptz_agent_runtime(self._ptz_agent_root)
            from msa.sensors import SensorRegistry

            sreg = SensorRegistry(self._cfg.get("sensors_config") or {})
            for schema in sreg.list_sensors():
                name = schema.get("name")
                if not name:
                    continue
                out.append(MsaSensorDriver(
                    device_id=f"sensor:{name}",
                    registry=sreg,
                    sensor_name=name,
                    schema=schema,
                ))
        except Exception as exc:  # discovery is best-effort
            logger.warning("sensor discovery failed: %s", exc)
        return out

    @property
    def drivers(self) -> dict[str, BaseDriver]:
        if self._drivers is None:
            self._drivers = self._build()
        return self._drivers

    def reload(self) -> None:
        self._drivers = None

    # -- access ------------------------------------------------------------

    def get(self, device_id: str | None) -> BaseDriver:
        did = device_id or self._default_ptz_id
        drv = self.drivers.get(did)
        if drv is None:
            raise DriverError(
                f"unknown device_id {did!r}; known: {sorted(self.drivers)}"
            )
        return drv

    @property
    def default_ptz_id(self) -> str:
        return self._default_ptz_id

    def ptz_driver(self) -> PTZCameraDriver:
        drv = self.get(self._default_ptz_id)
        if not isinstance(drv, PTZCameraDriver):
            raise DriverError(f"{self._default_ptz_id!r} is not a PTZ camera")
        return drv

    def devices(self) -> list[DeviceInfo]:
        return [d.describe() for d in self.drivers.values()]

    def invoke(self, device_id: str | None, capability: str,
               params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.get(device_id).invoke(capability, **(params or {}))
