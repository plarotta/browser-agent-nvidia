from src.agent.action_parser import ActionParser

def test_action_parser():
    # Test valid click
    text = 'click(selector="#submit-btn")'
    res = ActionParser.parse(text)
    assert res == {"type": "click", "params": {"selector": "#submit-btn"}}
    
    # Test valid type
    text = 'type(selector="#input", text="hello world")'
    res = ActionParser.parse(text)
    assert res == {"type": "type", "params": {"selector": "#input", "text": "hello world"}}
    
    # Test valid scroll with integer
    text = 'scroll(direction="down", amount=500)'
    res = ActionParser.parse(text)
    assert res == {"type": "scroll", "params": {"direction": "down", "amount": 500}}
    
    # Test invalid format
    text = "just clicking around"
    res = ActionParser.parse(text)
    assert res is None
