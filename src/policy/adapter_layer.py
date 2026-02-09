import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterLayer(nn.Module):
    """
    A simple Low-Rank Adapter (LoRA) layer.
    In a full implementation, we would use peft library to inject this into the transformer.
    For this prototype/skeleton, we define the conceptual block.
    """
    def __init__(self, in_features: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA A and B matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, in_features, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # standard LoRA forward: x + (x @ A @ B) * scaling
        return (self.lora_B(self.lora_A(x))) * self.scaling

class ActionHead(nn.Module):
    """
    Trainable head that sits on top of the frozen backbone if we extract embeddings.
    Alternatively, we use LoRA on the backbone itself.
    """
    def __init__(self, input_dim: int, action_space_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
