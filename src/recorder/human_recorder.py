import json
import queue
import logging
from typing import Optional, Tuple

from src.browser_interaction.session_manager import SessionManager
from src.observation.dom_snapshotter import DOMSnapshotter
from src.observation.visual_capture import VisualCapture
from src.utils.trajectory_logger import TrajectoryLogger

logger = logging.getLogger(__name__)

# JS injected via add_init_script — survives full-page navigations.
# All listeners push events into a queue via the exposed __recorderEvent function.
# The main Python loop drains the queue outside the callback (avoids reentrancy).
_INIT_SCRIPT = """
(() => {
    if (window.__recorderInstalled) return;
    window.__recorderInstalled = true;

    // --- helpers ---
    function agentId(el) {
        let cur = el;
        while (cur && cur !== document.body) {
            const id = cur.getAttribute && cur.getAttribute('data-agent-id');
            if (id) return id;
            cur = cur.parentElement;
        }
        return null;
    }

    function isTypeable(el) {
        const tag = (el.tagName || '').toLowerCase();
        const type = (el.getAttribute('type') || '').toLowerCase();
        const noType = ['submit', 'button', 'reset', 'checkbox', 'radio', 'image', 'file'];
        const role = el.getAttribute('role') || '';
        return (tag === 'input' && !noType.includes(type)) || tag === 'textarea'
               || role === 'textbox' || role === 'combobox';
    }

    // Dedup flag: when Enter fires in an input, we emit TYPE+PRESS_ENTER from
    // keydown and set this flag so the change listener skips the redundant TYPE.
    let _enterHandledChange = false;

    // --- click (capture phase) ---
    document.addEventListener('click', (e) => {
        const el = e.target;
        // Skip clicks on typeable inputs — those are handled by change/keydown
        if (isTypeable(el)) return;
        const eid = agentId(el);
        if (!eid) return;
        window.__recorderEvent(JSON.stringify({
            type: 'click', element_id: eid
        }));
    }, true);

    // --- change (input committed) ---
    document.addEventListener('change', (e) => {
        if (_enterHandledChange) {
            _enterHandledChange = false;
            return;
        }
        const el = e.target;
        if (!isTypeable(el)) return;
        const eid = agentId(el);
        if (!eid) return;
        window.__recorderEvent(JSON.stringify({
            type: 'type', element_id: eid, value: el.value || ''
        }));
    }, true);

    // --- keydown (Enter) ---
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter') return;
        const el = e.target;
        const eid = agentId(el);

        if (isTypeable(el) && eid) {
            // Emit TYPE for the current value, then PRESS_ENTER
            _enterHandledChange = true;
            window.__recorderEvent(JSON.stringify({
                type: 'type', element_id: eid, value: el.value || ''
            }));
        }
        // Always emit PRESS_ENTER
        window.__recorderEvent(JSON.stringify({ type: 'press_enter' }));
    }, true);

    // --- scroll (debounced) ---
    let _scrollTimer = null;
    let _lastScrollY = window.scrollY;
    window.addEventListener('scroll', () => {
        if (_scrollTimer) clearTimeout(_scrollTimer);
        _scrollTimer = setTimeout(() => {
            const dy = window.scrollY - _lastScrollY;
            if (Math.abs(dy) < 50) return;  // ignore tiny scrolls
            _lastScrollY = window.scrollY;
            window.__recorderEvent(JSON.stringify({
                type: 'scroll', direction: dy > 0 ? 'down' : 'up'
            }));
        }, 400);
    }, true);
})();
"""


class HumanRecorder:
    """Records human browser demonstrations as trajectories.

    Uses Playwright expose_function + JS event listeners to capture user
    actions in the same TimeStep format the agent produces.
    """

    def __init__(self, log_dir: str, goal: str, viewport: Tuple[int, int] = (1280, 720)):
        self.log_dir = log_dir
        self.goal = goal
        self.viewport = viewport

        self.session = SessionManager(headless=False)
        self.dom_snapshotter: Optional[DOMSnapshotter] = None
        self.visual_capture: Optional[VisualCapture] = None
        self.traj_logger = TrajectoryLogger(log_dir)
        self.step_count = 0

        # Queue for events from JS → Python (avoids reentrancy)
        self._event_queue: queue.Queue = queue.Queue()

        # Current observation (refreshed after each action)
        self._current_dom: Optional[str] = None
        self._current_screenshot = None  # PIL Image

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, url: str):
        """Launch browser, inject listeners, navigate, take initial observation."""
        page = self.session.start()
        page.set_viewport_size({"width": self.viewport[0], "height": self.viewport[1]})

        # Expose the bridge function BEFORE adding init script
        page.expose_function("__recorderEvent", self._on_browser_event)

        # Inject JS listeners (survives navigations via add_init_script)
        page.context.add_init_script(_INIT_SCRIPT)
        # Also run it on the current page immediately
        page.evaluate(_INIT_SCRIPT)

        self.dom_snapshotter = DOMSnapshotter(page)
        self.visual_capture = VisualCapture(page)

        self.session.navigate(url)
        # Wait for page to settle before first observation
        page.wait_for_timeout(1500)
        self._refresh_observation()
        logger.info(f"[REC] Recording started at {url}")
        logger.info("[REC] Browse normally. Press Ctrl+C in terminal to finish.")

    def run(self) -> str:
        """Main event loop. Blocks until KeyboardInterrupt (Ctrl+C).

        Returns the path to the saved trajectory JSON.
        """
        page = self.session.page
        try:
            while True:
                # Yield to Playwright so JS callbacks can fire
                page.wait_for_timeout(200)
                self._drain_queue()
        except KeyboardInterrupt:
            logger.info("[REC] Ctrl+C received — finishing recording.")

        # Prompt for final answer before closing browser (browser still visible)
        try:
            final_answer = input("\n[REC] Enter the final answer/result (or press Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            final_answer = ""

        return self.stop(final_answer=final_answer or None)

    def stop(self, final_answer: Optional[str] = None) -> str:
        """Log FINISH step, save trajectory + metadata, close browser. Returns trajectory path."""
        # Drain any remaining events
        self._drain_queue()

        # Log a FINISH step with current observation
        if self._current_dom is not None and self._current_screenshot is not None:
            finish_value = final_answer or "Human recording ended"
            finish_action = {
                "type": "finish",
                "params": {"value": finish_value},
                "metadata": {"reasoning": "User pressed Ctrl+C to end recording."},
                "final_answer": final_answer,
            }
            self.traj_logger.log_step(
                step_id=self.step_count,
                dom=self._current_dom,
                screenshot=self._current_screenshot,
                action=finish_action,
                reward=1.0,
                done=True,
            )
            self.step_count += 1

        traj_path = self.traj_logger.save_trajectory()

        # Save metadata
        meta_path = traj_path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"goal": self.goal, "source": "human"}, f, indent=2)

        logger.info(f"[REC] Trajectory saved: {traj_path} ({self.step_count} steps)")
        logger.info(f"[REC] Metadata saved: {meta_path}")

        # Close browser — suppress asyncio noise from Playwright teardown
        try:
            self.session.close()
        except Exception:
            pass
        return traj_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_browser_event(self, raw_json: str):
        """Called from JS via expose_function. Just enqueues — no Playwright calls here."""
        self._event_queue.put(raw_json)

    def _drain_queue(self):
        """Process all queued events."""
        while not self._event_queue.empty():
            try:
                raw = self._event_queue.get_nowait()
                event = json.loads(raw)
                self._process_event(event)
            except queue.Empty:
                break
            except json.JSONDecodeError:
                logger.warning(f"[REC] Bad event JSON: {raw}")

    def _process_event(self, event: dict):
        """Map a browser event to an agent-format action and log a TimeStep."""
        event_type = event.get("type")

        if event_type == "click":
            action = {
                "type": "click",
                "params": {"element_id": event["element_id"]},
            }
            label = f"CLICK on {event['element_id']}"

        elif event_type == "type":
            value = event.get("value", "")
            if not value:
                return  # skip empty TYPE (matches executor behavior)
            action = {
                "type": "type",
                "params": {"element_id": event["element_id"], "value": value},
            }
            label = f'TYPE on {event["element_id"]} "{value}"'

        elif event_type == "press_enter":
            action = {"type": "press_enter", "params": {}}
            label = "PRESS_ENTER"

        elif event_type == "scroll":
            direction = event.get("direction", "down")
            action = {
                "type": "scroll",
                "params": {"value": direction},
            }
            label = f"SCROLL {direction}"

        else:
            logger.debug(f"[REC] Unknown event type: {event_type}")
            return

        # Log step with current observation (captured BEFORE the user acted)
        if self._current_dom is None or self._current_screenshot is None:
            logger.warning("[REC] No observation yet — skipping event")
            return

        self.traj_logger.log_step(
            step_id=self.step_count,
            dom=self._current_dom,
            screenshot=self._current_screenshot,
            action=action,
            reward=1.0,  # human demos are expert demonstrations
            done=False,
        )
        logger.info(f"[REC] Step {self.step_count}: {label}")
        self.step_count += 1

        # Refresh observation for next step (page may have changed)
        self._settle_and_refresh()

    def _settle_and_refresh(self):
        """Wait for page to settle after an action, then refresh observation."""
        try:
            self.session.page.wait_for_timeout(1500)
        except Exception:
            pass
        self._refresh_observation()

    def _refresh_observation(self):
        """Take a fresh DOM snapshot + screenshot and store as current observation."""
        try:
            self._current_dom = self.dom_snapshotter.capture()
            self._current_screenshot = self.visual_capture.capture()
        except Exception as e:
            logger.warning(f"[REC] Observation capture failed: {e}")
