from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from typing import Optional, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)


class ActionExecutor:
    def __init__(self, page: Page):
        self.page = page
        self.logger = logging.getLogger(__name__)

    def _resolve_element(self, params: Dict[str, Any]):
        """Resolve an element from element_id (data-agent-id) or fallback to selector."""
        element_id = params.get("element_id")
        if element_id:
            return self.page.locator(f'[data-agent-id="{element_id}"]')
        selector = params.get("selector")
        if selector:
            return self.page.locator(selector)
        return None

    def execute(self, action_type: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Executes a browser action. Returns (success, error_reason)."""
        try:
            if action_type == "click":
                locator = self._resolve_element(params)
                if not locator:
                    return False, "Click requires 'element_id' — none provided."
                locator.first.click(timeout=5000)

            elif action_type == "type":
                text = params.get("value") or params.get("text", "")
                if not text:
                    return False, "TYPE requires a value — got empty or missing value."
                locator = self._resolve_element(params)
                if locator:
                    try:
                        locator.first.fill(text, timeout=5000)
                    except Exception as fill_err:
                        err_msg = str(fill_err).lower()
                        if "not an <input>" in err_msg or "contenteditable" in err_msg or "role" in err_msg:
                            locator = self.page.locator(
                                "input:not([type='submit']):not([type='button']):not([type='hidden']), "
                                "textarea, [role='textbox']"
                            ).first
                            if locator.count() > 0:
                                locator.fill(text, timeout=5000)
                            else:
                                raise fill_err
                        else:
                            raise
                else:
                    locator = self.page.locator(
                        "input:not([type='submit']):not([type='button']):not([type='hidden']), "
                        "textarea, [role='textbox']"
                    ).first
                    if locator.count() > 0:
                        locator.fill(text, timeout=5000)
                    else:
                        return False, "Type requires 'element_id' (or a visible text input on the page)."

            elif action_type == "press_enter":
                self.page.keyboard.press("Enter")
                # Wait for navigation/content load after submit
                try:
                    self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                except PlaywrightTimeoutError:
                    pass  # Page may not navigate; that's fine
                self.page.wait_for_timeout(1500)

            elif action_type == "select":
                locator = self._resolve_element(params)
                if not locator:
                    return False, "Select requires 'element_id' — none provided."
                locator.first.select_option(params.get("value", ""), timeout=5000)

            elif action_type == "scroll":
                direction = params.get("value", "down")
                amount = 500
                self.page.mouse.move(
                    self.page.viewport_size["width"] / 2,
                    self.page.viewport_size["height"] / 2,
                )
                if direction == "up":
                    self.page.mouse.wheel(0, -amount)
                else:
                    self.page.mouse.wheel(0, amount)
                self.page.wait_for_timeout(100)

            elif action_type == "navigate":
                url = params.get("value", "")
                if not url or not url.startswith(("http://", "https://")):
                    return False, "Navigate requires a full URL starting with http:// or https://."
                self.page.goto(url, wait_until="domcontentloaded", timeout=10000)

            elif action_type == "wait":
                duration = params.get("duration", 1000)
                self.page.wait_for_timeout(duration)

            elif action_type == "finish":
                # No-op: agent proposes task done; SuccessChecker validates.
                pass

            else:
                return False, f"Unknown action type: {action_type}"

            return True, ""

        except PlaywrightTimeoutError:
            reason = f"Timeout — element may not be visible or clickable."
            self.logger.error(f"Timeout executing {action_type} with params {params}")
            return False, reason
        except Exception as e:
            reason = str(e).split("\n")[0]  # First line of the error
            self.logger.error(f"Error executing {action_type}: {reason}")
            return False, reason
