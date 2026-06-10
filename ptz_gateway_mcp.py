#!/usr/bin/env python3
"""MCP server — routed through :class:`~ptz_node.sensor_gateway.SensorGateway`.

Run (from this repository root so ``ptz_node`` is importable, or pip install -e .)::

    python3 ptz_gateway_mcp.py

Cursor MCP config: ``command`` = python3; ``args`` = [ ``ptz_gateway_mcp.py`` ]; ``cwd`` = repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: MCP SDK missing. pip install mcp", file=sys.stderr)
    sys.exit(1)


mcp = FastMCP(
    "Jetson PTZ sensor-gateway",
    instructions=(
        "All hardware access is mediated by the SensorGateway. "
        "Use jetson_gateway_list_devices, then PTZ helpers (sim or Reolink per ptz-agent env). "
        "PTZ_AGENT_ROOT env may point at the original ptz-agent checkout."
    ),
)


def _gw():
    from ptz_node.sensor_gateway import get_gateway

    return get_gateway()


def _jd(s: str) -> str:
    try:
        return json.dumps(json.loads(s), indent=2)
    except Exception:
        return s


@mcp.tool()
def jetson_gateway_list_devices() -> str:
    """Return managed sensors/actuators (id, kind, interface, capabilities)."""
    try:
        return json.dumps(_gw().list_devices(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_sensor_capabilities(device_id: str) -> str:
    """List the callable capabilities of one managed device."""
    try:
        return json.dumps(_gw().capabilities(device_id), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_sensor_read(device_id: str) -> str:
    """Take a reading from a read-only sensor device (e.g. sensor:system_stats)."""
    try:
        return _jd(_gw().read_sensor(device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_sensor_invoke(device_id: str, capability: str,
                         params_json: str = "{}") -> str:
    """Invoke any device capability. params_json is a JSON object of arguments."""
    try:
        params = json.loads(params_json) if params_json.strip() else {}
        if not isinstance(params, dict):
            return json.dumps({"ok": False, "error": "params_json must be an object"})
        return _jd(_gw().invoke(device_id, capability, params))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_get_position(device_id: str | None = None) -> str:
    """Pan, tilt, FOV for the gateway PTZ device."""
    try:
        return _jd(_gw().ptz_get_position(device_id=device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_move_to(pan: float, tilt: float, device_id: str | None = None) -> str:
    """Absolute PTZ aim."""
    try:
        return _jd(_gw().ptz_move_to(pan, tilt, device_id=device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_pan_by(degrees: float, device_id: str | None = None) -> str:
    """Relative pan."""
    try:
        return _jd(_gw().ptz_pan_by(degrees, device_id=device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_tilt_by(degrees: float, device_id: str | None = None) -> str:
    """Relative tilt."""
    try:
        return _jd(_gw().ptz_tilt_by(degrees, device_id=device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_set_fov_h(fov_h_degrees: float, device_id: str | None = None) -> str:
    """Horizontal FOV adjustment."""
    try:
        return _jd(_gw().ptz_set_fov_h(fov_h_degrees, device_id=device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_snapshot(filename: str | None = None,
                        device_id: str | None = None) -> str:
    """Write viewport JPEG; returns filesystem path."""
    try:
        return _jd(_gw().ptz_snapshot(filename=filename, device_id=device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_detector_status() -> str:
    """Which vision backends are available (yolo, bioclip, gemma4)."""
    try:
        return _jd(_gw().detector_status())
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_detect(
    model: str = "yolo",
    targets: str = "*",
    target_taxon: str = "",
    target: str = "",
    device_id: str | None = None,
) -> str:
    """Run YOLO, BioCLIP, or Gemma4 on the current PTZ viewport."""
    try:
        return _jd(_gw().ptz_detect(
            model=model,
            targets=targets,
            target_taxon=target_taxon,
            target=target,
            device_id=device_id,
        ))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_ptz_caption(
    model: str = "bioclip",
    prompt: str = "",
    device_id: str | None = None,
) -> str:
    """Caption the PTZ viewport for scientific scene understanding."""
    try:
        return _jd(_gw().ptz_caption(model=model, prompt=prompt, device_id=device_id))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_list_skills() -> str:
    """List modular skills on this node (incl. self_diagnosis self-healing)."""
    try:
        from ptz_node.config_loader import load_config
        from ptz_node.skills import SkillRegistry

        return json.dumps(SkillRegistry(load_config()).list_skills(),
                          indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_run_skill(name: str, args_json: str = "{}") -> str:
    """Run a skill once. e.g. jetson_run_skill("self_diagnosis", '{"action":"status"}')."""
    try:
        from ptz_node.config_loader import load_config
        from ptz_node.skills import SkillRegistry

        reg = SkillRegistry(load_config())
        if not reg.has(name):
            return json.dumps({"ok": False, "error": f"unknown skill {name!r}",
                               "known": reg.names()})
        params = json.loads(args_json) if args_json.strip() else {}
        return json.dumps(reg.run(name, params).as_dict(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def jetson_heal_pending() -> str:
    """List self-healing fix proposals awaiting human approval."""
    try:
        from ptz_node.config_loader import load_config
        from ptz_node.self_healing.healer import Healer

        return json.dumps(Healer(load_config()).list_pending(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
