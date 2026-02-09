import json
import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ActionParser:
    
    """
    Parses JSON action output from the model into the dictionary format
    required by ActionExecutor.

    Expected model output (JSON):
    {
        "page_intent": "login page",
        "reasoning": "The email field is empty. I should click it to begin typing.",
        "action": {
            "type": "CLICK",
            "target_element_id": "e24",
            "value": null
        },
        "confidence": 0.85
    }

    For task completion the model may output action.type "FINISH" and optionally
    "FINAL_ANSWER: <value>" in the text (or action.value) for evaluation.
    """

    _FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
    # Fallback: only match action keywords at the START of a line or after "ACTION:"
    # This prevents matching TYPE/CLICK mentions in echoed prompt text or DOM hints.
    _ACTION_TYPE_RE = re.compile(
        r"^(?:ACTION\s*:\s*)?(CLICK|TYPE|PRESS_ENTER|SCROLL|WAIT|NAVIGATE|FINISH)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    # Detect "goal achieved" language when model doesn't output a proper FINISH action
    _DONE_LANGUAGE_RE = re.compile(
        r"goal\s+(?:is\s+)?(?:achieved|complete[d]?|accomplished|done)|task\s+(?:is\s+)?(?:complete[d]?|done|finished)",
        re.IGNORECASE,
    )
    _ELEMENT_ID_RE = re.compile(r"target_element_id\s*[:=]\s*([eE]?\d+)", re.IGNORECASE)
    _VALUE_RE = re.compile(
        r'value\s*[:=]\s*"([^"]*)"|value\s*[:=]\s*[\']([^\']*)[\']|value\s*[:=]\s*(.+?)(?=\s*(?:\n|target_|$))',
        re.IGNORECASE | re.DOTALL,
    )
    _GOAL_SPEC_RE = re.compile(r"GOAL_SPEC\s*[:\s]*(\{[\s\S]*?\})(?=\s*(?:```|\n\n|$))", re.IGNORECASE)
    # Where the action block starts in fallback output (prose is before this)
    _ACTION_START_RE = re.compile(
        r"\b(?:ACTION\s*:\s*|TYPE\s*:|CLICK\s*:|PRESS_ENTER|SCROLL|WAIT|NAVIGATE|FINISH|TARGET_ELEMENT_ID\s*:|```\s*tool_code)",
        re.IGNORECASE,
    )

    @staticmethod
    def extract_goal_spec(text: str) -> Optional[Dict[str, Any]]:
        """
        Parse goal_outputs_required, success_patterns, stop_action_enabled from step 0 output.
        Looks for GOAL_SPEC: {...} or a JSON object with goal_outputs_required / required_outputs.
        """
        if not text or not text.strip():
            return None
        # Try GOAL_SPEC: { ... } first
        m = ActionParser._GOAL_SPEC_RE.search(text.strip())
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        # Try any JSON object that has goal_outputs_required or required_outputs
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i : j + 1])
                        if isinstance(obj, dict) and (
                            "goal_outputs_required" in obj or "required_outputs" in obj
                        ):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        return None

    @staticmethod
    def extract_final_answer(text: str) -> Optional[str]:
        """Extract FINAL_ANSWER: <value> from raw model output."""
        if not text:
            return None
        m = ActionParser._FINAL_ANSWER_RE.search(text.strip())
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Find the best JSON object in text by trying each '{' as a start."""
        candidates = []
        for i, ch in enumerate(text):
            if ch != '{':
                continue
            # Walk forward to find matching closing brace
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i:j+1])
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    break
        if not candidates:
            return None
        # Prefer the one that has an "action" key; otherwise take the largest
        for c in candidates:
            if "action" in c:
                return c
        return max(candidates, key=lambda c: len(c))

    @staticmethod
    def _extract_fallback_metadata(raw_text: str) -> Dict[str, Any]:
        """From prose before ACTION/TYPE/CLICK/etc., extract intent and reasoning for fallback parse."""
        if not raw_text or not raw_text.strip():
            return {"page_intent": "", "reasoning": "", "confidence": 0.75}
        text = raw_text.strip()
        m = ActionParser._ACTION_START_RE.search(text)
        prose = text[: m.start()].strip() if m else text
        # Drop GOAL_SPEC line so it's not used as reasoning
        prose = re.sub(r"GOAL_SPEC\s*:\s*\{[^}]*\}", "", prose, flags=re.IGNORECASE).strip()
        prose = re.sub(r"\n\s*\n", " ", prose)
        if not prose:
            return {"page_intent": "", "reasoning": "", "confidence": 0.75}
        # First 1–2 sentences, max ~180 chars for reasoning
        sentences = re.split(r"\.\s+", prose)
        reasoning = sentences[0].strip()
        if reasoning and not reasoning.endswith("."):
            reasoning += "."
        if len(sentences) > 1 and len(reasoning) < 120:
            reasoning += " " + sentences[1].strip()
            if not reasoning.endswith("."):
                reasoning += "."
        reasoning = reasoning[:180].strip()
        # Short page_intent from first few words or keywords
        words = prose.replace("\n", " ").split()
        if len(words) >= 3:
            page_intent = " ".join(words[:5])
            if len(page_intent) > 50:
                page_intent = page_intent[:47] + "..."
        else:
            page_intent = prose[:50] if prose else ""
        return {
            "page_intent": page_intent,
            "reasoning": reasoning,
            "confidence": 0.75,
        }

    @staticmethod
    def _parse_fallback(text: str) -> Optional[Dict[str, Any]]:
        """Parse non-JSON formats: TYPE: target_element_id=e0, value="...", or ACTION: TYPE with key: value lines."""
        # Strip markdown code blocks so we match inside them
        cleaned = re.sub(r"^```\w*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        action_m = ActionParser._ACTION_TYPE_RE.search(cleaned)
        if not action_m:
            return None
        action_type = action_m.group(1).lower()

        element_m = ActionParser._ELEMENT_ID_RE.search(cleaned)
        target = element_m.group(1) if element_m else None

        value_m = ActionParser._VALUE_RE.search(cleaned)
        value = None
        if value_m:
            value = (value_m.group(1) or value_m.group(2) or value_m.group(3) or "").strip()
            if value:
                value = value.split("\n")[0].strip()  # single line

        params = {}
        if target:
            params["element_id"] = target
        if value is not None and value:
            params["value"] = value

        metadata = ActionParser._extract_fallback_metadata(text)
        return {
            "type": action_type,
            "params": params,
            "metadata": metadata,
        }

    @staticmethod
    def parse(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()

        data = ActionParser._extract_json(text)
        if data:
            action = data.get("action")
            if action and "type" in action:
                action_type = action["type"].lower()
                target = action.get("target_element_id")
                value = action.get("value")
                final_answer = ActionParser.extract_final_answer(text)
                if action_type == "finish" and value and not final_answer:
                    final_answer = str(value).strip() if value else final_answer
                params = {}
                if target:
                    params["element_id"] = target
                if value is not None:
                    params["value"] = value
                result = {
                    "type": action_type,
                    "params": params,
                    "metadata": {
                        "page_intent": data.get("page_intent", ""),
                        "reasoning": data.get("reasoning", ""),
                        "confidence": data.get("confidence", 0.0),
                    },
                }
                if final_answer is not None:
                    result["final_answer"] = final_answer
                return result

        # Fallback: parse TYPE/ACTION style output (e.g. Gemma with truncated prompt)
        fallback = ActionParser._parse_fallback(text)
        if fallback:
            logger.info("Parsed action from non-JSON format (fallback).")
            return fallback

        # Last resort: detect "goal is achieved/complete" language → treat as FINISH
        if ActionParser._DONE_LANGUAGE_RE.search(text):
            logger.info("Model indicated goal done (no formal action). Parsing as FINISH.")
            return {
                "type": "finish",
                "params": {},
                "metadata": {"reasoning": "Model indicated goal is complete.", "confidence": 0.6},
            }

        logger.warning(f"No JSON or parseable action in model output: {text[:200]}")
        return None
