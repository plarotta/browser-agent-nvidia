"""Structured goal spec: agent defines in step 0. Termination = required_outputs_found."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GoalSpec:
    """
    Agent-defined success criteria (parsed from step 0 output).
    Goal checking is just: if required_outputs_found then done = True.
    """

    goal_outputs_required: List[str] = field(default_factory=list)
    """Output names that must be present for task done (e.g. ['price'], ['code'])."""

    success_patterns: List[str] = field(default_factory=list)
    """Optional page/success patterns (e.g. ['confirmation']). Logged, not used for done."""

    stop_action_enabled: bool = True
    """Whether agent may emit FINISH (informational)."""

    def to_dict(self) -> dict:
        return {
            "goal_outputs_required": self.goal_outputs_required,
            "success_patterns": self.success_patterns,
            "stop_action_enabled": self.stop_action_enabled,
        }

    @classmethod
    def from_parsed(cls, d: dict) -> "GoalSpec":
        """Build GoalSpec from agent output (step 0)."""
        required = d.get("goal_outputs_required") or d.get("required_outputs") or []
        if not isinstance(required, list):
            required = [str(required)] if required else []
        patterns = d.get("success_patterns") or []
        if not isinstance(patterns, list):
            patterns = [str(patterns)] if patterns else []
        stop = d.get("stop_action_enabled", True)
        if isinstance(stop, str):
            stop = stop.strip().lower() in ("true", "1", "yes")
        return cls(
            goal_outputs_required=required,
            success_patterns=patterns,
            stop_action_enabled=bool(stop),
        )
