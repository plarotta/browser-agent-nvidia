from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, Playwright
from typing import Optional

class SessionManager:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def start(self) -> Page:
        """Starts a new browser session and returns a page."""
        if self.page:
            return self.page

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self.context = self.browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        # Remove the webdriver flag that sites use to detect automation
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        self.page = self.context.new_page()
        return self.page

    def navigate(self, url: str):
        """Navigates the current page to the specified URL."""
        if not self.page:
            self.start()
        if self.page:
            self.page.goto(url)

    def get_screenshot(self, path: Optional[str] = None) -> bytes:
        """Captures a screenshot of the current page."""
        if not self.page:
            raise RuntimeError("Session not started")
        return self.page.screenshot(path=path)

    def get_dom(self) -> str:
        """Returns the full HTML content of the page."""
        if not self.page:
            raise RuntimeError("Session not started")
        return self.page.content()

    def close(self):
        """Closes the browser session and releases resources."""
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        self.page = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
