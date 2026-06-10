"""Driver contract for the sensor-management layer.

Every physical or simulated device the agent can touch is represented by a
:class:`BaseDriver`. Drivers expose a flat list of *capabilities* (named verbs)
plus an optional ``read()`` for pure sensors. The agent, REST API, and MCP
server only ever speak to drivers through :class:`~ptz_node.sensor_gateway.SensorGateway`
— never to hardware directly. This keeps the management layer the single
choke point the original design called for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


class DriverError(Exception):
    """Raised for unknown capabilities, bad params, or device faults."""


@dataclass
class DeviceInfo:
    """Public, serializable description of one managed device."""

    id: str
    kind: str                      # ptz_camera | sensor | actuator | ...
    interface: str = "unknown"     # network | serial | usb | i2c | gpio | system
    backend: str = ""              # sim | reolink | system | <driver-specific>
    description: str = ""
    units: str = ""
    read_only: bool = False
    capabilities: list[str] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseDriver(ABC):
    """Base class for all device drivers in the management layer."""

    kind: str = "device"
    interface: str = "unknown"
    read_only: bool = False

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    @abstractmethod
    def describe(self) -> DeviceInfo:
        """Return a :class:`DeviceInfo` (cheap; avoid heavy live I/O)."""

    def capabilities(self) -> list[str]:
        return list(self.describe().capabilities)

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities()

    @abstractmethod
    def invoke(self, capability: str, **params: Any) -> dict[str, Any]:
        """Run one capability and return a JSON-serializable dict.

        Raise :class:`DriverError` for unknown capabilities or bad params.
        """

    def read(self) -> dict[str, Any]:
        """Pure-sensor convenience. Default: invoke a ``read`` capability."""
        if self.has_capability("read"):
            return self.invoke("read")
        raise DriverError(f"{self.device_id} has no 'read' capability")

    def status(self) -> dict[str, Any]:
        info = self.describe()
        return {"id": self.device_id, "kind": info.kind,
                "interface": info.interface, "backend": info.backend,
                "status": "ok"}
