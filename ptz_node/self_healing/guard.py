"""Error capture that routes incidents to the :class:`Healer`.

Three ways to wire it up:
  * ``with guard("component", cfg): ...`` — context manager around risky work.
  * ``@guarded("component")`` — decorator for a function.
  * ``install_excepthook(cfg)`` — last-resort hook for uncaught exceptions.

By default the original exception is re-raised after being recorded, so program
behavior is unchanged unless ``self_healing.swallow_errors`` is true. Healing is
opportunistic and never itself raises.
"""

from __future__ import annotations

import functools
import logging
import sys
from contextlib import ContextDecorator
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _load_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is not None:
        return cfg
    try:
        from ptz_node.config_loader import load_config

        return load_config()
    except Exception:
        return {}


def trigger_heal(exc: BaseException, *, component: str,
                 cfg: dict[str, Any] | None = None,
                 context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Best-effort: hand an exception to the healer. Never raises."""
    cfg = _load_cfg(cfg)
    sh = cfg.get("self_healing") or {}
    if not sh.get("enabled", False):
        return None
    try:
        from ptz_node.self_healing.healer import Healer

        healer = Healer(cfg)
        if not healer.enabled():
            return None
        return healer.handle_exception(exc, component=component, context=context)
    except Exception as heal_exc:  # healing must never mask the real error
        logger.warning("self-heal failed for %s: %s", component, heal_exc)
        return None


def report_anomaly(component: str, message: str, *,
                   cfg: dict[str, Any] | None = None,
                   context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Route an unexpected/undefined output (not an exception) to the healer."""
    cfg = _load_cfg(cfg)
    sh = cfg.get("self_healing") or {}
    if not sh.get("enabled", False):
        return None
    try:
        from ptz_node.self_healing.healer import Healer

        healer = Healer(cfg)
        if not healer.enabled():
            return None
        return healer.report_anomaly(component, message, context=context)
    except Exception as exc:
        logger.warning("anomaly report failed for %s: %s", component, exc)
        return None


class guard(ContextDecorator):
    """Context manager / decorator that records + heals exceptions."""

    def __init__(self, component: str, cfg: dict[str, Any] | None = None,
                 *, context: dict[str, Any] | None = None,
                 swallow: bool | None = None) -> None:
        self.component = component
        self.cfg = cfg
        self.context = context
        self.swallow = swallow
        self.heal_result: dict[str, Any] | None = None

    def __enter__(self) -> "guard":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        self.heal_result = trigger_heal(exc, component=self.component,
                                        cfg=self.cfg, context=self.context)
        cfg = _load_cfg(self.cfg)
        swallow = (self.swallow if self.swallow is not None
                   else bool((cfg.get("self_healing") or {}).get("swallow_errors", False)))
        return bool(swallow)  # True suppresses the exception


def guarded(component: str, cfg: dict[str, Any] | None = None) -> Callable:
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with guard(component, cfg):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def install_excepthook(cfg: dict[str, Any] | None = None) -> None:
    """Route uncaught exceptions through the healer before the default hook."""
    prev = sys.excepthook

    def hook(exc_type, exc, tb):
        try:
            trigger_heal(exc, component="uncaught", cfg=cfg)
        finally:
            prev(exc_type, exc, tb)

    sys.excepthook = hook
