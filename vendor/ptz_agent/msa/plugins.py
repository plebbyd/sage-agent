"""
msa/plugins.py — Plugin discovery for tools and sensors.

Scans a directory for Python files, imports them, and finds subclasses
of a given base class.  Handles import errors gracefully so a missing
dependency in one plugin doesn't bring down the whole agent.

Skip rules (so the loader can't be hung by non-plugin scripts in the
same directory):
  - Files starting with '_' or 'test_'
  - Files matching `MSA_SKIP_PLUGINS` (comma-separated list of stems
    or fnmatch globs, env override). Defaults skip known long-running
    services (ptz_viewer, sim_ptz_watch, startup_checks, calibrate*).
  - Each module import is run with a soft timeout (`MSA_PLUGIN_IMPORT_TIMEOUT`
    seconds, default 8) so a single hanging plugin can't wedge agent startup.
"""

import fnmatch
import importlib.util
import inspect
import logging
import os
import signal
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Files that live next to plugins but are NOT plugins themselves — they
# are CLIs, long-running services, or test scripts. Importing them can
# start servers, open hardware, or warm models, which will deadlock the
# agent's startup.
_DEFAULT_SKIP = {
    "ptz_viewer",        # web UI HTTP server
    "ptz_facade",        # imported transitively by ptz_tool when needed
    "sim_ptz_watch",     # interactive watcher CLI
    "sim_ptz_tool",      # simulated PTZ over stitched.png — opt-in only
                         #   set MSA_INCLUDE_SIM_PTZ=1 to enable.
                         #   When the real ptz_* tools are present, the
                         #   simulated ones confuse the agent (it picks
                         #   the more verbose description).
    "startup_checks",    # warm-up routine; called explicitly elsewhere
    "calibrate_ptz",     # CLI script — invoked by ptz_calibrate tool
    "reolink_camera",    # bare driver, imported transitively as needed
    "detectors",         # heavy imports; loaded lazily by ptz_tool when needed
}


def _effective_skip() -> set[str]:
    """Default skip set, minus anything the operator opted back in to."""
    skip = set(_DEFAULT_SKIP)
    if os.environ.get("MSA_INCLUDE_SIM_PTZ") == "1":
        skip.discard("sim_ptz_tool")
        skip.discard("sim_ptz_watch")
    return skip


def _parse_skip_env() -> set[str]:
    raw = os.environ.get("MSA_SKIP_PLUGINS", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def _should_skip(stem: str, extra_patterns: set[str]) -> bool:
    if stem in _effective_skip():
        return True
    for pat in extra_patterns:
        if fnmatch.fnmatch(stem, pat) or pat == stem:
            return True
    return False


def _import_with_timeout(spec, mod, timeout: float):
    """Run spec.loader.exec_module(mod) with a wall-clock timeout.

    Uses a daemon thread so a stuck import (network call, blocking server
    start) doesn't wedge the agent forever. Raises TimeoutError on miss.
    """
    err: list = []

    def _target():
        try:
            spec.loader.exec_module(mod)
        except BaseException as e:
            err.append(e)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"import exceeded {timeout:.1f}s")
    if err:
        raise err[0]


def discover(directory: str, base_class: type, config: dict = None) -> list:
    """
    Scan *directory* for .py files and return instances of every class
    that is a direct or indirect subclass of *base_class*.

    Files starting with '_' or 'test_' are skipped, as are entries in
    `_DEFAULT_SKIP` and the `MSA_SKIP_PLUGINS` env var.  Files that fail
    to import (e.g. missing dependency) or that exceed
    `MSA_PLUGIN_IMPORT_TIMEOUT` seconds are logged and skipped.

    Each discovered class is instantiated with (config=config) if its
    __init__ accepts a 'config' keyword, otherwise with no arguments.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.debug("Plugin directory does not exist: %s", directory)
        return []

    dir_str = str(directory.resolve())
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    extra_skip = _parse_skip_env()
    try:
        timeout = float(os.environ.get("MSA_PLUGIN_IMPORT_TIMEOUT", "8"))
    except ValueError:
        timeout = 8.0

    instances = []
    for py_file in sorted(directory.glob("*.py")):
        stem = py_file.stem
        if stem.startswith("_") or stem.startswith("test_"):
            continue
        if _should_skip(stem, extra_skip):
            logger.debug("Skipping non-plugin file: %s", py_file.name)
            continue

        module_name = f"msa_plugin_{directory.name}_{stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            mod = importlib.util.module_from_spec(spec)
            _import_with_timeout(spec, mod, timeout)
        except TimeoutError:
            logger.warning(
                "Plugin %s exceeded %.1fs at import — skipped. "
                "If it's a real plugin, make its imports lazy or add it to "
                "MSA_SKIP_PLUGINS.", py_file.name, timeout,
            )
            continue
        except Exception as e:
            logger.debug("Skipping plugin %s: %s", py_file.name, e)
            continue

        for _attr_name, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, base_class) or obj is base_class:
                continue
            if obj.__module__ != module_name:
                continue

            try:
                sig = inspect.signature(obj.__init__)
                if "config" in sig.parameters:
                    instance = obj(config=config or {})
                else:
                    instance = obj()
                instances.append(instance)
                logger.info("Discovered plugin: %s from %s",
                            getattr(instance, "name", obj.__name__), py_file.name)
            except Exception as e:
                logger.warning("Failed to instantiate %s from %s: %s",
                               obj.__name__, py_file.name, e)

    # avoid "unused import" lint when signal stays optional
    _ = signal
    return instances
