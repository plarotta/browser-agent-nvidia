import pytest
import os
import torch
from src.safety.checkpoint_manager import CheckpointManager

def test_checkpoint_lifecycle(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    manager = CheckpointManager(ckpt_dir)
    
    # Simple model
    model = torch.nn.Linear(1, 1)
    torch.nn.init.constant_(model.weight, 1.0)
    
    # Save initial (Step 0)
    manager.save(model, step=0, is_best=True)
    assert os.path.exists(os.path.join(ckpt_dir, "checkpoint_step_0.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "best_checkpoint.pt"))
    
    # Mutate model (Bad update)
    torch.nn.init.constant_(model.weight, 99.0)
    manager.save(model, step=1, is_best=False)
    
    # Rollback
    manager.rollback(model)
    
    # Validation
    assert model.weight.item() == 1.0
