"""
Rule-based interpreter: natural language goal → structured GoalSpec.
Keeps the demo stable; can be replaced or augmented with a small LLM later.
"""

import re
from src.evaluation.goal_spec import GoalSpec


# Keywords/phrases that map to goal types and expected artifacts.
GOAL_PATTERNS = [
    # (pattern, goal_type, success_signals, required_outputs, expected_artifact)
    (r"cheapest\s+(flight|price)|find\s+(the\s+)?(cheapest|lowest)\s+price|return\s+the\s+price", "extract_information", ["results_page_visible", "price_visible"], ["price"], "price"),
    (r"price\s+of|get\s+the\s+price|what\s+is\s+the\s+price", "extract_information", ["price_visible"], ["price"], "price"),
    (r"(\d+)[\s\-]?character\s+code|(\d+)\s*char\s+code|code\s+is\s+\d+\s+char", "extract_information", ["code_visible"], ["code"], "code"),
    (r"confirmation|confirm\s+(order|booking|payment)|order\s+confirmed|thank\s+you", "form_submit", ["confirmation_visible", "success_banner"], [], "confirmation message"),
    (r"log\s+in|sign\s+in|login", "navigate", ["logged_in_visible", "dashboard_or_profile"], [], ""),
    (r"search\s+for|find\s+(information\s+about|details\s+on)", "extract_information", ["results_page_visible"], [], "relevant result"),
    (r"book\s+(a\s+)?flight|reserve|submit\s+form", "form_submit", ["confirmation_visible", "success_banner"], [], "confirmation"),
    (r"extract|scrape|get\s+(the\s+)?(data|value|text)", "extract_information", ["content_visible"], ["extracted_value"], "extracted value"),
]


def _match_goal(goal_lower: str) -> tuple:
    for pattern, goal_type, signals, outputs, artifact in GOAL_PATTERNS:
        if re.search(pattern, goal_lower, re.IGNORECASE):
            return goal_type, signals, outputs, artifact
    return "general", ["task_related_content"], [], ""


class GoalInterpreter:
    """Converts a user's natural-language goal into a structured GoalSpec."""

    def interpret(self, user_goal: str, max_idle_steps: int = 3) -> GoalSpec:
        """
        Derive observable success criteria from the user goal.
        Rule-based; no LLM call.
        """
        goal_lower = (user_goal or "").strip().lower()
        if not goal_lower:
            return GoalSpec(
                goal_type="general",
                success_signals=[],
                required_outputs=[],
                expected_artifact="",
                max_idle_steps=max_idle_steps,
                stop_action_enabled=True,
            )

        goal_type, success_signals, required_outputs, expected_artifact = _match_goal(goal_lower)
        return GoalSpec(
            goal_type=goal_type,
            success_signals=success_signals,
            required_outputs=required_outputs,
            expected_artifact=expected_artifact or "task outcome",
            max_idle_steps=max_idle_steps,
            stop_action_enabled=True,
        )
