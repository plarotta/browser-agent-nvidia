import pytest
import torch
import torch.nn as nn
from src.sdft.sdft_module import SDFTModule

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        # Initialize with known weights
        nn.init.ones_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

def test_ema_update():
    student = SimpleModel()
    sdft = SDFTModule(student, ema_decay=0.5)
    
    # Verify teacher clone
    assert torch.allclose(sdft.teacher.linear.weight, sdft.student.linear.weight)
    
    # Modify student manually (simulate gradient update)
    with torch.no_grad():
        sdft.student.linear.weight.add_(1.0) # Student weight becomes 2.0
    
    # Update teacher
    sdft.update_teacher()
    
    # Teacher = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    expected = torch.tensor([[1.5, 1.5], [1.5, 1.5]])
    assert torch.allclose(sdft.teacher.linear.weight, expected)

def test_loss_computation():
    student = SimpleModel()
    sdft = SDFTModule(student)
    
    batch_size = 1
    logits_dim = 4
    
    # Dummy logits
    s_logits = torch.randn(batch_size, logits_dim)
    t_logits = torch.randn(batch_size, logits_dim)
    
    loss = sdft.compute_loss(s_logits, t_logits)
    assert loss >= 0
    assert loss.shape == () # Scalar

def test_gating_logic():
    student = SimpleModel()
    sdft = SDFTModule(student)
    
    # Success, High Confidence (Low Entropy) -> Update
    assert sdft.should_update(entropy=0.1, success=True, confidence_threshold=0.8)
    
    # Failure -> No Update
    assert not sdft.should_update(entropy=0.1, success=False, confidence_threshold=0.8)
    
    # Success, Low Confidence (High Entropy) -> No Update
    assert not sdft.should_update(entropy=0.8, success=True, confidence_threshold=0.8)
