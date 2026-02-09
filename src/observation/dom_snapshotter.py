from playwright.sync_api import Page

class DOMSnapshotter:
    def __init__(self, page: Page):
        self.page = page

    def capture(self) -> str:
        """Captures a simplified DOM snapshot."""
        # return basic outerHTML of the body as a starting point.
        # We can enhance this with pruning later.
        try:
            return self.page.evaluate("document.body ? document.body.outerHTML : ''")
        except Exception:
            return ""
