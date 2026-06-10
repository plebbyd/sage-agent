"""Read-only driver wrapping one auto-discovered ptz-agent sensor plugin.

Users extend the node with *arbitrary* sensors (weather, serial/UART, USB,
I2C, GPIO, ethernet, …) simply by dropping a ``BaseSensor`` plugin into
``ptz-agent/sensors/``. The :class:`~ptz_node.sensor_gateway.registry.DeviceRegistry`
discovers each one and wraps it here, so it shows up in the gateway — and thus
the agent, REST API, and MCP — with no new code.
"""

from __future__ import annotations

from typing import Any

from ptz_node.sensor_gateway.base import BaseDriver, DeviceInfo, DriverError

_SENSOR_CAPS = ["read", "status"]


class MsaSensorDriver(BaseDriver):
    kind = "sensor"
    read_only = True

    def __init__(self, device_id: str, registry: Any, sensor_name: str,
                 schema: dict[str, Any]) -> None:
        super().__init__(device_id)
        self._registry = registry          # ptz-agent msa.sensors.SensorRegistry
        self._sensor_name = sensor_name
        self._schema = schema or {}
        self.interface = str(self._schema.get("interface", "unknown"))

    def describe(self) -> DeviceInfo:
        return DeviceInfo(
            id=self.device_id,
            kind=self.kind,
            interface=self.interface,
            backend="msa_sensor",
            description=str(self._schema.get("description", "")),
            units=str(self._schema.get("units", "")),
            read_only=True,
            capabilities=list(_SENSOR_CAPS),
            paths={"sensor_name": self._sensor_name},
        )

    def invoke(self, capability: str, **params: Any) -> dict[str, Any]:
        if capability == "read":
            return dict(self._registry.read(self._sensor_name))
        if capability == "status":
            return dict(self._registry.status(self._sensor_name))
        raise DriverError(
            f"unknown capability {capability!r} for sensor {self.device_id}; "
            f"choose from {_SENSOR_CAPS}"
        )

    def read(self) -> dict[str, Any]:
        return self.invoke("read")
