"""
msa/tools.py — Tool registry and built-in tool implementations.

Tools are the agent's "embodiment" — what it can actually DO in the world.

Built-in tools are registered automatically.  Additional tools are
discovered from the tools/ directory via the plugin loader — drop a .py
file that subclasses BaseTool and it will be picked up on next start.

Security model
--------------
  ShellTool    — shell=False + shlex.split blocks metacharacter injection.
  ReadFileTool / WriteFileTool — _validate_path() constrains to project root.
  HttpGetTool  — scheme + private-IP checks block SSRF.
"""

import ipaddress
import logging
import shlex
import socket
import subprocess
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from . import plugins

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent.resolve()


# ---------------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> str:
        pass

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description}


# ---------------------------------------------------------------------------
# Built-in Tools
# ---------------------------------------------------------------------------

class EchoTool(BaseTool):
    name = "echo"
    description = "Echo a message back. Useful for testing."

    def run(self, message: str = "", **kwargs) -> str:
        return f"ECHO: {message}"


class ShellTool(BaseTool):
    name = "shell"
    description = "Run a shell command and return stdout. Use carefully."

    def run(self, command: str = "", **kwargs) -> str:
        try:
            args = shlex.split(command)
        except ValueError as e:
            return f"ERROR: Could not parse command: {e}"

        if not args:
            return "ERROR: Empty command"

        try:
            result = subprocess.run(
                args, shell=False, capture_output=True, text=True, timeout=30
            )
            return result.stdout or result.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out"
        except FileNotFoundError:
            return f"ERROR: Command not found: {args[0]}"
        except Exception as e:
            return f"ERROR: {e}"


def _validate_path(path_str: str) -> Path:
    """Resolve *path_str* and verify it lives inside _BASE_DIR."""
    resolved = Path(path_str).resolve()
    try:
        resolved.relative_to(_BASE_DIR)
    except ValueError:
        raise ValueError(
            f"Access denied: '{path_str}' resolves to '{resolved}', which is "
            f"outside the project directory ({_BASE_DIR}). "
            "Only paths within the project tree are permitted."
        )
    return resolved


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a file from disk and return its contents."

    def run(self, path: str = "", filename: str = "", **kwargs) -> str:
        path = path or filename
        try:
            validated = _validate_path(path)
            with open(validated) as f:
                return f.read()
        except ValueError as e:
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: {e}"


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file on disk."

    def run(self, path: str = "", content: str = "", filename: str = "", **kwargs) -> str:
        path = path or filename
        try:
            validated = _validate_path(path)
            with open(validated, "w") as f:
                f.write(content)
            return f"Written to {validated}"
        except ValueError as e:
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: {e}"


_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_private_host(hostname: str) -> bool:
    if hostname.lower() in ("localhost", "metadata.google.internal"):
        return True
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(hostname))
        return any(addr in net for net in _BLOCKED_NETWORKS)
    except (socket.gaierror, ValueError):
        return False


class HttpGetTool(BaseTool):
    name = "http_get"
    description = (
        "Make an HTTP GET request and return the response body. "
        "Only http/https URLs are accepted; internal/private addresses are blocked."
    )

    def run(self, url: str = "", **kwargs) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return (
                f"ERROR: Only http and https URLs are allowed "
                f"(got scheme '{parsed.scheme}')"
            )
        if not parsed.hostname:
            return "ERROR: URL must include a hostname"
        if _is_private_host(parsed.hostname):
            return (
                f"ERROR: Requests to private or internal addresses are not "
                f"allowed ({parsed.hostname})"
            )

        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return r.read().decode()[:2000]
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self, config: dict = None):
        self._tools: dict[str, BaseTool] = {}

        for tool_cls in [EchoTool, ShellTool, ReadFileTool, WriteFileTool, HttpGetTool]:
            self.register(tool_cls())

        project_root = Path(__file__).parent.parent.resolve()
        tools_dir = project_root / "tools"
        for tool in plugins.discover(str(tools_dir), BaseTool, config or {}):
            self.register(tool)

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, args: dict) -> str:
        if not self.has(name):
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].run(**args)

    def describe(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())
