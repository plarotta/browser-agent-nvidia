import pytest
from src.browser_interaction.session_manager import SessionManager

def test_session_lifecycle():
    """Test that the session manager can start and stop a browser."""
    manager = SessionManager(headless=True)
    try:
        page = manager.start()
        assert page is not None
        assert manager.browser is not None
        assert manager.context is not None
    finally:
        manager.close()
        assert manager.page is None
        assert manager.browser is None

def test_navigation():
    """Test page navigation."""
    with SessionManager(headless=True) as session:
        session.navigate("https://example.com")
        assert "Example Domain" in session.page.title()
        
        content = session.get_dom()
        assert "<h1>Example Domain</h1>" in content
