import pytest
import os
import glob
from src.agent.agent_runtime import AgentRuntime

from src.utils.config import AgentConfig

def test_agent_lifecycle_and_logging(tmp_path):
    log_dir = str(tmp_path / "logs")
    config = AgentConfig(log_dir=log_dir, headless=True)
    
    # Use a dummy page
    agent = AgentRuntime(config=config)
    
    try:
        agent.start(start_url="data:text/html,<h1>Hello</h1>")
        
        # Execute a dummy action
        action = {"type": "wait", "params": {"duration": 100}}
        success = agent.step(action)
        assert success
        
        agent.stop()
        
        # Verify logs
        json_files = glob.glob(os.path.join(log_dir, "*.json"))
        assert len(json_files) == 1
        
        png_files = glob.glob(os.path.join(log_dir, "*.png"))
        assert len(png_files) >= 1
        
    except Exception as e:
        agent.stop()
        raise e
