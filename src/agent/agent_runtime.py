import time
from typing import Optional, Dict, Any, List, Tuple
from src.browser_interaction.session_manager import SessionManager
from src.browser_interaction.action_executor import ActionExecutor
from src.observation.visual_capture import VisualCapture
from src.observation.dom_snapshotter import DOMSnapshotter
from src.utils.trajectory_logger import TrajectoryLogger
from src.policy.multimodal_policy import MultimodalPolicy
from src.sdft.sdft_module import SDFTModule
from src.agent.action_parser import ActionParser
from src.utils.config import AgentConfig
from src.utils.file_download import (
    check_url_is_file, download_via_context,
    page_looks_empty, url_has_file_extension,
)
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a browser agent. Complete the goal by choosing ONE action.
GOAL: {goal}

Respond with ONLY this JSON:
{{"reasoning": "<1 sentence>", "action": {{"type": "<ACTION>", "target_element_id": "<id or null>", "value": "<string or null>"}}}}

Actions: CLICK (element_id required), TYPE (element_id + value), PRESS_ENTER, SCROLL (value: up/down), WAIT, NAVIGATE (value: URL), FINISH (when done; ALWAYS set value to the answer or key result you found on the page).

Rules:
- IDs must come from DOM_SUMMARY. Never invent IDs.
- NAVIGATE URLs must come from links in DOM_SUMMARY. NEVER invent or guess URLs.
- Check ACTION HISTORY — do not repeat failed or completed actions. If stuck, try FINISH.
- After TYPE, you MUST press PRESS_ENTER to submit before using FINISH.
- Do NOT use FINISH until the page shows actual results. FINISH value must be real data from PAGE TEXT below, never your own search query.
{error_section}
ACTION HISTORY:
{action_history}

DOM_SUMMARY:
{dom_summary}
"""

MAX_HISTORY = 5


class AgentRuntime:
    def __init__(self, config: AgentConfig, policy: Optional[MultimodalPolicy] = None, sdft: Optional[SDFTModule] = None, trajectory_uploader=None):
        self.config = config
        self.session_manager = SessionManager(headless=config.headless)
        self.action_executor = None
        self.visual_capture = None
        self.dom_snapshotter = None
        self.logger = TrajectoryLogger(config.log_dir)
        self.step_count = 0

        # Policy & Learning
        self.policy = policy
        self.sdft = sdft
        self.action_parser = ActionParser()
        self.last_error: Optional[str] = None
        self.action_history: List[str] = []
        self.trajectory_uploader = trajectory_uploader

    def start(self, start_url: str) -> Optional[str]:
        """Initializes the browser session.

        Returns a filepath if the start URL was a file download (agent loop
        should be skipped), or None for normal operation.
        """
        page = self.session_manager.start()
        page.set_viewport_size({"width": self.config.viewport_width, "height": self.config.viewport_height})

        self.action_executor = ActionExecutor(page)
        self.visual_capture = VisualCapture(page)
        self.dom_snapshotter = DOMSnapshotter(page)
        self.session_manager.navigate(start_url)

        # Pre-navigation file check
        is_file, content_type = check_url_is_file(start_url)
        if is_file:
            logger.info(f"Start URL is a file ({content_type}). Downloading directly.")
            filepath = download_via_context(self.session_manager.context.request, start_url)
            if filepath:
                return filepath
            logger.warning("Direct download failed, continuing with agent loop.")

        if self.policy:
            if not self.policy.model:
                 self.policy.load_model()

        return None

    def _format_history(self) -> str:
        if not self.action_history:
            return "(none yet — this is the first step)"
        return "\n".join(self.action_history[-MAX_HISTORY:])

    def _build_prompt(self, dom_summary: str) -> str:
        """Constructs the full prompt from the system template + current observation."""
        # Keep DOM within budget.  The compact prompt template (~400 chars) +
        # history (~300 chars) leaves ~2300 chars inside the 3000-char MLX limit.
        if getattr(self.config, "backend", None) == "mlx":
            max_dom_chars = 2000
            if len(dom_summary) > max_dom_chars:
                dom_summary = dom_summary[:max_dom_chars] + "\n... [truncated]"
        error_section = ""
        if self.last_error:
            error_section = f"\nLAST ACTION FAILED: {self.last_error}\nYou must try a different action.\n"
        return SYSTEM_PROMPT.format(
            goal=self.config.task_goal,
            action_history=self._format_history(),
            dom_summary=dom_summary,
            error_section=error_section,
        )

    def _record_action(self, action_dict: Dict[str, Any], success: bool):
        """Add a human-readable summary of the action to history."""
        action_type = action_dict.get("type", "?").upper()
        target = action_dict.get("params", {}).get("element_id", "")
        value = action_dict.get("params", {}).get("value", "")
        status = "OK" if success else "FAILED"

        parts = [f"Step {self.step_count + 1}: {action_type}"]
        if target:
            parts.append(f"on {target}")
        if value:
            parts.append(f'"{value}"')
        parts.append(f"-> {status}")
        self.action_history.append(" ".join(parts))

    def step(self, action: Optional[Dict[str, Any]] = None) -> Tuple[bool, bool]:
        """
        Executes one step: Observe -> Predict -> Act -> Log -> Check success.
        Returns (success, done). done=True when the agent outputs FINISH.
        """
        if not self.action_executor:
            raise RuntimeError("Agent not started")

        # 1. Observe
        dom = self.dom_snapshotter.capture()

        # Runtime guard: detect file/binary pages
        current_url = self.action_executor.page.url
        auto_download = page_looks_empty(dom) and url_has_file_extension(current_url)

        # Second check: page text missing (e.g. PDF rendered in browser viewer).
        # The viewer has interactive elements so page_looks_empty returns False,
        # and URLs like arxiv.org/pdf/ID have no file extension.
        if not auto_download:
            has_page_text = "PAGE TEXT" in dom and len(dom.split("PAGE TEXT:", 1)[1].strip()) > 20
            if not has_page_text:
                is_file, ct = check_url_is_file(current_url)
                if is_file:
                    logger.info(f"File detected via content-type ({ct}): {current_url}")
                    auto_download = True

        if auto_download:
            logger.info(f"File page detected: {current_url}")
            filepath = download_via_context(self.session_manager.context.request, current_url)
            if filepath:
                logger.info(f"Auto-downloaded: {filepath}")
                screenshot = self.visual_capture.capture()
                finish_action = {
                    "type": "finish",
                    "params": {"value": f"Downloaded: {filepath}"},
                    "metadata": {"reasoning": "File page detected, downloaded automatically."},
                    "final_answer": filepath,
                }
                self._record_action(finish_action, True)
                self.logger.log_step(
                    step_id=self.step_count, dom=dom, screenshot=screenshot,
                    action=finish_action, reward=1.0, done=True,
                )
                self.step_count += 1
                return True, True

        screenshot = self.visual_capture.capture()

        # 2. Predict Action (if not provided manually)
        action_dict = action
        raw_output = None
        if not action_dict and self.policy:
            prompt = self._build_prompt(dom)
            logger.debug(f"Prompt length: {len(prompt)} chars")
            # Log PAGE TEXT section so we can verify the model sees real page content
            if "PAGE TEXT" in dom:
                pt_start = dom.index("PAGE TEXT")
                logger.info(f"PAGE TEXT snippet: {dom[pt_start:pt_start+200]}")
            else:
                logger.info("PAGE TEXT: (none extracted)")
            try:
                raw_output = self.policy.forward(screenshot, prompt)
                logger.info(f"Model output: {raw_output}")
                action_dict = self.action_parser.parse(raw_output)
                if action_dict and action_dict.get("metadata"):
                    meta = action_dict["metadata"]
                    logger.info(f"  reasoning={meta.get('reasoning')}")
            except Exception as e:
                logger.error(f"Prediction failed: {e}")

        # Fallback
        if not action_dict:
            logger.warning("No valid action predicted. Waiting...")
            action_dict = {"type": "wait", "params": {"duration": 1000}}

        # Guard: if model says FINISH but no submit has happened since the last TYPE, force PRESS_ENTER.
        # This prevents 12B models from skipping the submit step.
        if (action_dict.get("type") == "finish"
                and self.action_history
                and "PRESS_ENTER" not in " ".join(self.action_history)):
            has_type = any("TYPE" in h for h in self.action_history)
            if has_type:
                logger.info("Guard: overriding premature FINISH → PRESS_ENTER (search not submitted yet)")
                action_dict = {"type": "press_enter", "params": {}}

        # Guard: after a failed CLICK with a pending TYPE (no PRESS_ENTER yet), force PRESS_ENTER.
        # First CLICK attempt is always allowed; this only triggers on retry after failure.
        if (action_dict.get("type") == "click"
                and self.last_error
                and "CLICK" in self.last_error.upper()
                and any("TYPE" in h for h in self.action_history)
                and not any("PRESS_ENTER" in h for h in self.action_history)):
            logger.info("Guard: CLICK failed with pending TYPE — forcing PRESS_ENTER")
            action_dict = {"type": "press_enter", "params": {}}

        # Guard: loop detection — if the same action repeated 3+ times AND the current
        # action also matches the loop pattern, force FINISH.  If the model chose a
        # *different* action (e.g. NAVIGATE after 3x CLICK), let it through.
        if len(self.action_history) >= 3:
            last_three = self.action_history[-3:]
            # Extract action signature: strip "Step N: " prefix, compare type+target before "->"
            def _sig(h):
                after_colon = h.split(": ", 1)[1] if ": " in h else h
                return after_colon.split("->")[0].strip()
            signatures = [_sig(h) for h in last_three]

            # Build current action signature
            cur_type = action_dict.get("type", "?").upper()
            cur_target = action_dict.get("params", {}).get("element_id", "")
            cur_value = action_dict.get("params", {}).get("value", "")
            cur_parts = [cur_type]
            if cur_target:
                cur_parts.append(f"on {cur_target}")
            if cur_value:
                cur_parts.append(f'"{cur_value}"')
            current_sig = " ".join(cur_parts)

            if signatures[0] == signatures[1] == signatures[2] == current_sig:
                logger.info("Guard: loop detected (same action 3x + current matches). Forcing FINISH.")
                action_dict = {
                    "type": "finish",
                    "params": {"value": "Agent stuck in loop — could not complete task."},
                    "metadata": {"reasoning": "Loop detected: same action repeated 3 times."},
                }

        # 3. Act
        success, error_reason = self.action_executor.execute(action_dict.get("type"), action_dict.get("params", {}))

        # Track error for next prompt
        if success:
            self.last_error = None
        else:
            action_type = action_dict.get("type", "unknown")
            target = action_dict.get("params", {}).get("element_id", "none")
            self.last_error = f"{action_type.upper()} on {target}: {error_reason}"
            # Hint: if CLICK failed and there's a pending TYPE, suggest PRESS_ENTER
            if action_type == "click":
                has_type = any("TYPE" in h for h in self.action_history)
                has_enter = any("PRESS_ENTER" in h for h in self.action_history)
                if has_type and not has_enter:
                    self.last_error += " Hint: use PRESS_ENTER to submit your search query."

        # Record to history
        self._record_action(action_dict, success)

        # 4. Done when the agent outputs FINISH
        done = action_dict.get("type", "").lower() == "finish"
        if done:
            final_answer = action_dict.get("final_answer") or action_dict.get("params", {}).get("value")
            logger.info("Task done: agent issued FINISH. answer=%s", final_answer)

        # 5. Log
        reward = 1.0 if success else 0.0
        self.logger.log_step(
            step_id=self.step_count,
            dom=dom,
            screenshot=screenshot,
            action=action_dict,
            reward=reward,
            done=done,
        )

        # 6. Learn (SDFT)
        if self.sdft and self.policy:
             if self.sdft.should_update(entropy=0.0, success=success):
                 self.sdft.update_teacher()
                 logger.info("SDFT: Updated Teacher Model")

        self.step_count += 1
        return success, done

    def stop(self):
        """Cleanup."""
        path = self.logger.save_trajectory()
        logger.info(f"Trajectory saved to {path}")
        if self.trajectory_uploader:
            try:
                traj_id = self.trajectory_uploader.upload(self.config.log_dir)
                logger.info(f"Trajectory uploaded to server: {traj_id}")
            except Exception as e:
                logger.warning(f"Trajectory upload failed: {e}")
        self.session_manager.close()
