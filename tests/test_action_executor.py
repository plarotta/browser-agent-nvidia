import pytest
from src.browser_interaction.session_manager import SessionManager
from src.browser_interaction.action_executor import ActionExecutor

def test_action_execution():
    """Test basic actions like click and type."""
    with SessionManager(headless=True) as session:
        # Load a simple test page (using data:text/html for speed/reliability)
        html_content = """
        <html>
            <body>
                <input id="test-input" type="text" />
                <button id="test-button" onclick="document.body.style.backgroundColor = 'red'">Click Me</button>
            </body>
        </html>
        """
        session.navigate(f"data:text/html,{html_content}")
        
        executor = ActionExecutor(session.page)
        
        # Test Type
        success = executor.execute("type", {"selector": "#test-input", "text": "Hello World"})
        assert success
        value = session.page.input_value("#test-input")
        assert value == "Hello World"
        
        # Test Click
        success = executor.execute("click", {"selector": "#test-button"})
        assert success
        # Verify effect (bg color change)
        bg_color = session.page.evaluate("document.body.style.backgroundColor")
        assert bg_color == "red"

def test_scroll_action():
    """Test scroll action."""
    with SessionManager(headless=True) as session:
        # Page with content to scroll
        html_content = """
        <html>
            <body style="height: 2000px">
                <div id="top">Top</div>
                <div id="bottom" style="position: absolute; top: 1500px">Bottom</div>
            </body>
        </html>
        """
        session.navigate(f"data:text/html,{html_content}")
        executor = ActionExecutor(session.page)
        
        # Scroll down
        initial_scroll = session.page.evaluate("window.scrollY")
        assert initial_scroll == 0
        
        success = executor.execute("scroll", {"direction": "down", "amount": 500})
        assert success
        
        # Wait for scroll to happen
        session.page.wait_for_function("window.scrollY > 0")
        
        new_scroll = session.page.evaluate("window.scrollY")
        assert new_scroll > 0 # Precise 500 is hard to guarantee with wheel event physics

