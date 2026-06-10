"""Auto-discovery registry for skills.

Discovers every :class:`~ptz_node.skills.base.BaseSkill` subclass defined in modules
under ``ptz_node/skills/`` plus an optional external directory
(``PTZ_GRAPH_SKILLS_DIR`` or ``config['skills_dir']``). Import failures are logged
and skipped so one broken skill never takes down the rest.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
from pathlib import Path
from typing import Any

from ptz_node.skills.base import BaseSkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

_SKILLS_PKG = "ptz_node.skills"
_INTERNAL_SKIP = {"base", "registry", "__init__"}


class SkillRegistry:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._skills: dict[str, BaseSkill] = {}
        self._discover()

    # -- discovery ---------------------------------------------------------

    def _discover(self) -> None:
        pkg_dir = Path(__file__).resolve().parent
        for mod in pkgutil.iter_modules([str(pkg_dir)]):
            if mod.name in _INTERNAL_SKIP:
                continue
            self._load_module(f"{_SKILLS_PKG}.{mod.name}")

        ext = (
            os.environ.get("PTZ_GRAPH_SKILLS_DIR", "").strip()
            or str(self.config.get("skills_dir") or "").strip()
        )
        if ext:
            self._load_external_dir(Path(ext).expanduser())

    def _register_from_module(self, module) -> None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseSkill)
                and obj is not BaseSkill
                and obj.__module__ == module.__name__
                and getattr(obj, "name", "")
            ):
                try:
                    inst = obj(self.config)
                except Exception as exc:
                    logger.warning("skill %s failed to init: %s", obj.__name__, exc)
                    continue
                self._skills[inst.name] = inst

    def _load_module(self, dotted: str) -> None:
        try:
            module = importlib.import_module(dotted)
            self._register_from_module(module)
        except Exception as exc:
            logger.warning("skill module %s failed to import: %s", dotted, exc)

    def _load_external_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for py in sorted(directory.glob("*.py")):
            if py.stem.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"ext_skill_{py.stem}", py)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._register_from_module(module)
            except Exception as exc:
                logger.warning("external skill %s failed: %s", py, exc)

    # -- access ------------------------------------------------------------

    def list_skills(self) -> list[dict[str, Any]]:
        return [s.schema() for s in self._skills.values()]

    def names(self) -> list[str]:
        return sorted(self._skills)

    def has(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> BaseSkill:
        if name not in self._skills:
            raise KeyError(f"unknown skill {name!r}; known: {self.names()}")
        return self._skills[name]

    def run(self, name: str, args: dict[str, Any] | None = None) -> SkillResult:
        skill = self.get(name)
        ctx = SkillContext(config=self.config, args=args or {})
        return skill.run(ctx)
