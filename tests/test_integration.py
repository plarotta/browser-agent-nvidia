import pytest
from unittest.mock import MagicMock, patch
from src.agent.agent_runtime import AgentRuntime
from src.utils.config import AgentConfig
from src.policy.multimodal_policy import MultimodalPolicy
from src.sdft.sdft_module import SDFTModule

@patch("src.policy.multimodal_policy.AutoModelForCausalLM")
@patch("src.policy.multimodal_policy.AutoProcessor")
def test_full_agent_loop(mock_processor, mock_model_class, tmp_path):
    # Setup Mocks
    mock_model_instance = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model_instance
    
    # Mock generation output
    # The model returns token IDs, processor decodes them.
    # We mock the forward method of our wrapper for simplicity in this integration test
    # or we can mock the processor decode. Let's mock the wrapper method to avoid complex token mocking.
    
    config = AgentConfig(log_dir=str(tmp_path), headless=True)
    
    with patch("src.policy.multimodal_policy.MultimodalPolicy.forward", return_value='click(selector="#login")') as mock_forward:
        policy = MultimodalPolicy(device="cpu")
        sdft = SDFTModule(policy)
        
        agent = AgentRuntime(config=config, policy=policy, sdft=sdft)
        
        # Start Agent (Use dummy URL)
        agent.start(start_url="data:text/html,<button id='login'>Login</button>")
        
        # Step 1: Should call forward -> parse -> act -> log -> update
        success = agent.step()
        
        assert success
        mock_forward.assert_called_once()
        
        # Verify SDFT update called (since success=True)
        assert sdft.update_count == 1
        
        agent.stop()
