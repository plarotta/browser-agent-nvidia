import pytest
import torch
from unittest.mock import MagicMock, patch
from src.policy.multimodal_policy import MultimodalPolicy
from src.policy.adapter_layer import AdapterLayer, ActionHead

def test_adapter_layer_shape():
    batch_size = 2
    in_features = 64
    rank = 4
    
    layer = AdapterLayer(in_features, rank=rank)
    x = torch.randn(batch_size, in_features)
    out = layer(x)
    
    layer = AdapterLayer(in_features, rank=rank)
    x = torch.randn(batch_size, in_features)
    out = layer(x)
    
    assert out.shape == (batch_size, in_features)
    # LoRA B is zero-initialized, so output should be zero initially
    assert torch.allclose(out, torch.zeros_like(out)) 
    
    # Now set weights to non-zero to test flow
    torch.nn.init.ones_(layer.lora_B.weight)
    out_active = layer(x)
    assert not torch.allclose(out_active, torch.zeros_like(out_active))

def test_action_head_shape():
    batch_size = 2
    input_dim = 128
    action_space = 10
    
    head = ActionHead(input_dim, action_space)
    x = torch.randn(batch_size, input_dim)
    out = head(x)
    
    assert out.shape == (batch_size, action_space)

@patch("src.policy.multimodal_policy.AutoModelForCausalLM")
@patch("src.policy.multimodal_policy.AutoProcessor")
def test_multimodal_policy_structure(mock_processor, mock_model_class):
    # Mock return values to avoid network calls
    mock_processor.from_pretrained.return_value = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model_instance
    
    policy = MultimodalPolicy(device="cpu") # Force CPU for test
    
    # Test load_model calls
    policy.load_model()
    mock_model_class.from_pretrained.assert_called_once()
    
    # Test forward pass logic (mocked)
    mock_image = MagicMock()
    policy.forward(mock_image, "Run action")
    mock_model_instance.generate.assert_called_once()
