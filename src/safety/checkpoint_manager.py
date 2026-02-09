import os
import torch
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages saving, loading, and rolling back model checkpoints.
    Critical for the 'Safe Self-Distillation' feature.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_checkpoint: Optional[str] = None
        self.latest_checkpoint: Optional[str] = None

    def save(self, model: torch.nn.Module, step: int, is_best: bool = False):
        """Saves current model state (adapters only in real usage)."""
        filename = f"checkpoint_step_{step}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        
        # In a real LoRA setup, we'd use model.save_pretrained(path)
        # Here we mock-save the state dict
        torch.save(model.state_dict(), path)
        self.latest_checkpoint = path
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            shutil.copyfile(path, best_path)
            self.best_checkpoint = best_path
            logger.info(f"Saved best checkpoint to {best_path}")

    def load(self, model: torch.nn.Module, path: Optional[str] = None):
        """Loads weights into the model."""
        target_path = path or self.latest_checkpoint
        if not target_path or not os.path.exists(target_path):
            logger.warning(f"No checkpoint found at {target_path}")
            return
            
        logger.info(f"Loading checkpoint from {target_path}...")
        state_dict = torch.load(target_path)
        model.load_state_dict(state_dict)

    def rollback(self, model: torch.nn.Module):
        """Reverts to the best known checkpoint."""
        if not self.best_checkpoint:
            logger.warning("No best checkpoint to rollback to!")
            return
            
        logger.warning(f"Rolling back to {self.best_checkpoint}...")
        self.load(model, self.best_checkpoint)
