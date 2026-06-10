"""Skill contract for the modular skill system.

A *skill* is a higher-level, self-contained capability that the node can run on a
schedule or on demand (vs. a *tool*, which is one atomic gateway call). Drop a
``BaseSkill`` subclass into ``ptz_node/skills/`` (or an external skills dir set by
``PTZ_GRAPH_SKILLS_DIR``) and it is auto-discovered and registered — no wiring.

Skills receive a :class:`SkillContext` (config + gateway access) and return a
:class:`SkillResult`. They may optionally be exposed to the LLM agent as tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillContext:
    """Everything a skill needs, without importing heavy modules itself."""

    config: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)

    def gateway(self):
        from ptz_node.sensor_gateway import SensorGateway

        return SensorGateway(self.config)


@dataclass
class SkillResult:
    ok: bool
    skill: str
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skill": self.skill,
            "summary": self.summary,
            "data": self.data,
            "artifacts": self.artifacts,
        }


class BaseSkill(ABC):
    """Base class for all skills.

    Subclass, set ``name``/``description``, implement :meth:`run`, drop the file
    into ``ptz_node/skills/``. Set ``agent_callable = True`` to also expose the
    skill to the LLM agent as a ``skill_<name>`` tool.
    """

    name: str = ""
    description: str = ""
    # Default cadence for scheduled skills; None means "manual/triggered only".
    default_interval_hours: float | None = None
    agent_callable: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def skill_config(self) -> dict[str, Any]:
        """This skill's slice of ``config['skills'][name]`` (if any)."""
        skills = self.config.get("skills") or {}
        return dict(skills.get(self.name) or {})

    def enabled(self) -> bool:
        return bool(self.skill_config().get("enabled", True))

    @abstractmethod
    def run(self, ctx: SkillContext) -> SkillResult:
        """Execute the skill once."""

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "default_interval_hours": self.default_interval_hours,
            "agent_callable": self.agent_callable,
            "enabled": self.enabled(),
        }
