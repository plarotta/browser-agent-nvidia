import torch
import torch.nn as nn
import copy
from typing import Optional, Dict

class SDFTModule(nn.Module):
    """
    Self-Distillation Fine-Tuning Module.
    Manages the Student (Active) and Teacher (EMA) policies.
    """
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        super().__init__()
        self.student = student_model
        # Create a deep copy for the teacher. 
        # In a real heavy model scenario, we might only copy the adapters 
        # if the backbone is truly frozen and shared.
        self.teacher = copy.deepcopy(student_model)
        self.teacher.eval() # Teacher is always in eval mode
        
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.ema_decay = ema_decay
        self.update_count = 0

    def update_teacher(self):
        """
        Updates the teacher weights using Exponential Moving Average (EMA).
        teacher = decay * teacher + (1 - decay) * student
        """
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                # Only update if the parameter requires grad in student (i.e. adapters)
                # Or if we want to track everything. Since backbone is frozen, 
                # strictly speaking we only need to update the trainable parts.
                if s_param.requires_grad:
                    t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=(1 - self.ema_decay))
        
        self.update_count += 1

    def compute_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Computes KL Divergence loss between Student and Teacher.
        Loss = KL(softmax(teacher/T) || softmax(student/T)) * T^2
        """
        s_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)
        t_prob = nn.functional.softmax(teacher_logits / temperature, dim=-1)
        
        loss = nn.functional.kl_div(s_prob, t_prob, reduction='batchmean') * (temperature ** 2)
        return loss

    def should_update(self, entropy: float, success: bool, confidence_threshold: float = 0.8) -> bool:
        """
        Gating logic for safety.
        Only update if:
        1. The action was successful
        2. The model was reasonably confident (low entropy) OR sufficiently confident
        """
        # This is high-level logic. Entropy formatting depends on specific outputs.
        # Simple version:
        return success and (entropy < (1.0 - confidence_threshold))
