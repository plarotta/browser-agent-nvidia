import pytest
from src.browser_interaction.session_manager import SessionManager
from src.observation.visual_capture import VisualCapture
from src.observation.dom_snapshotter import DOMSnapshotter
from PIL import Image

def test_visual_capture():
    with SessionManager(headless=True) as session:
        session.navigate("data:text/html,<h1>Screenshot me</h1>")
        capture = VisualCapture(session.page)
        image = capture.capture()
        assert isinstance(image, Image.Image)
        assert image.width > 0
        assert image.height > 0

def test_dom_snapshotter():
    with SessionManager(headless=True) as session:
        html = "<body><div id='test'>Hello</div></body>"
        session.navigate(f"data:text/html,{html}")
        snapshotter = DOMSnapshotter(session.page)
        dom = snapshotter.capture()
        assert "<div id=\"test\">Hello</div>" in dom
        assert "<body>" in dom
