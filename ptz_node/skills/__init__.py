"""Modular skill system — auto-discovered, schedulable, agent-callable skills."""

from ptz_node.skills.base import BaseSkill, SkillContext, SkillResult
from ptz_node.skills.registry import SkillRegistry

__all__ = ["BaseSkill", "SkillContext", "SkillResult", "SkillRegistry"]
