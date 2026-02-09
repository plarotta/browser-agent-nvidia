"""
Goal checking: if required_outputs_found then done = True.
Agent defines goal_outputs_required (and optionally success_patterns, stop_action_enabled) in step 0.
"""

import re
from typing import Optional

from src.evaluation.goal_spec import GoalSpec


def _extract_price(text: str) -> bool:
    return bool(re.search(r"\$[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP)", text, re.IGNORECASE))


def _extract_code(text: str) -> bool:
    return bool(re.search(r"\b[A-Z0-9]{4,8}\b", text))


def _has_confirmation(text: str) -> bool:
    return bool(re.search(r"confirm(ed|ation)|thank\s+you|order\s+(placed|confirmed)|success(fully)?", text, re.IGNORECASE))


def required_outputs_found(goal_spec: GoalSpec, dom_text: str, final_answer: Optional[str] = None) -> bool:
    """
    True iff every required output is present in DOM or final_answer.
    If goal_outputs_required is empty, returns False (no automatic done).
    """
    if not goal_spec.goal_outputs_required:
        return False
    combined = (dom_text or "") + "\n" + (final_answer or "")
    for name in goal_spec.goal_outputs_required:
        key = name.lower()
        if key == "price" and not _extract_price(combined):
            return False
        if key == "code" and not _extract_code(combined):
            return False
        if key == "confirmation" and not _has_confirmation(combined):
            return False
        if key in ("value", "result", "answer") and not (final_answer and final_answer.strip()):
            return False
    return True
