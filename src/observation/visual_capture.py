from playwright.sync_api import Page
from PIL import Image
import io

class VisualCapture:
    def __init__(self, page: Page):
        self.page = page

    def capture(self) -> Image.Image:
        """Captures a screenshot and returns it as a PIL Image."""
        screenshot_bytes = self.page.screenshot()
        return Image.open(io.BytesIO(screenshot_bytes))
