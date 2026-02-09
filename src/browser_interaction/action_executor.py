from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)

class ActionExecutor:
    def __init__(self, page: Page):
        self.page = page
        self.logger = logging.getLogger(__name__)

    def execute(self, action_type: str, params: Dict[str, Any]) -> bool:
        """Executes a browser action."""
        try:
            if action_type == "click":
                selector = params.get("selector")
                if not selector:
                    raise ValueError("Click action requires 'selector'")
                self.page.click(selector)
            
            elif action_type == "type":
                selector = params.get("selector")
                text = params.get("text")
                if not selector or text is None:
                    raise ValueError("Type action requires 'selector' and 'text'")
                self.page.fill(selector, text)
            
            elif action_type == "scroll":
                direction = params.get("direction", "down")
                amount = params.get("amount", 500)
                # Ensure mouse is over the viewport
                self.page.mouse.move(self.page.viewport_size["width"] / 2, self.page.viewport_size["height"] / 2)
                if direction == "down":
                    self.page.mouse.wheel(0, amount)
                elif direction == "up":
                    self.page.mouse.wheel(0, -amount)
                # specific wait for scroll to likely happen
                self.page.wait_for_timeout(100)
            
            elif action_type == "wait":
                duration = params.get("duration", 1000) # ms
                self.page.wait_for_timeout(duration)
                
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                return False

            return True

        except PlaywrightTimeoutError:
            self.logger.error(f"Timeout executing {action_type} with params {params}")
            return False
        except Exception as e:
            self.logger.error(f"Error executing {action_type}: {e}")
            return False
