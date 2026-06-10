"""
msa/sensors.py — Sensor management for the MSA.

High-level API for managing diverse hardware interfaces (serial, USB,
I2C, SPI, GPIO, network, system) through a unified plugin architecture.

Users drop sensor plugins into the sensors/ directory and optionally
configure them via config.yaml under the 'sensors:' key.

Interface types (convention, not enforced):
    system  — OS-level readings (CPU, memory, disk)
    serial  — UART / RS-232 / RS-485 devices
    usb     — USB-connected devices
    i2c     — I2C bus sensors
    spi     — SPI bus sensors
    gpio    — Digital I/O, PWM
    network — Ethernet / Wi-Fi / MQTT / ONVIF devices
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from . import plugins

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Sensor
# ---------------------------------------------------------------------------

class BaseSensor(ABC):
    """Base class for all sensor plugins.

    Subclass this, set the class attributes, implement read(), and drop
    the file into sensors/.  The MSA will discover and register it
    automatically.
    """

    name: str = ""
    description: str = ""
    interface: str = "unknown"
    units: str = ""

    @abstractmethod
    def read(self) -> dict:
        """Take a reading.  Return a dict of named values."""

    def status(self) -> dict:
        """Return sensor health / connectivity info."""
        return {"name": self.name, "interface": self.interface, "status": "ok"}

    def configure(self, config: dict):
        """Apply runtime configuration from config.yaml."""

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "interface": self.interface,
            "units": self.units,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SensorRegistry:
    """Discovers, configures, and manages sensor plugins."""

    def __init__(self, config: dict = None):
        self._sensors: dict[str, BaseSensor] = {}
        self._config = config or {}
        self._last_readings: dict[str, dict] = {}

        project_root = Path(__file__).parent.parent.resolve()
        sensors_dir = project_root / "sensors"
        for sensor in plugins.discover(str(sensors_dir), BaseSensor, self._config):
            self.register(sensor)

        sensor_configs = self._config.get("sensors", {})
        for sensor_name, sensor_cfg in sensor_configs.items():
            if sensor_name in self._sensors and isinstance(sensor_cfg, dict):
                try:
                    self._sensors[sensor_name].configure(sensor_cfg)
                except Exception as e:
                    logger.warning("Failed to configure sensor %s: %s", sensor_name, e)

    def register(self, sensor: BaseSensor):
        self._sensors[sensor.name] = sensor
        logger.debug("Registered sensor: %s (%s)", sensor.name, sensor.interface)

    def has(self, name: str) -> bool:
        return name in self._sensors

    def list_sensors(self) -> list[dict]:
        return [s.schema() for s in self._sensors.values()]

    def read(self, name: str) -> dict:
        """Read a single sensor by name."""
        if name not in self._sensors:
            raise ValueError(f"Unknown sensor: {name}")
        try:
            reading = self._sensors[name].read()
            reading["_timestamp"] = time.time()
            reading["_sensor"] = name
            self._last_readings[name] = reading
            return reading
        except Exception as e:
            logger.error("Sensor %s read failed: %s", name, e)
            return {"_sensor": name, "_error": str(e), "_timestamp": time.time()}

    def read_all(self) -> dict[str, dict]:
        """Read every registered sensor.  Returns {name: reading}."""
        return {name: self.read(name) for name in self._sensors}

    def status(self, name: str = None):
        """Get status of one sensor (dict) or all sensors (list)."""
        if name:
            if name not in self._sensors:
                raise ValueError(f"Unknown sensor: {name}")
            return self._sensors[name].status()
        return [s.status() for s in self._sensors.values()]

    def describe(self) -> str:
        if not self._sensors:
            return "(no sensors registered)"
        lines = []
        for s in self._sensors.values():
            lines.append(f"- {s.name} [{s.interface}]: {s.description}")
        return "\n".join(lines)
