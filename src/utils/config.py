import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    # Model
    # User requested smaller Nemotron.
    # Using Nemotron Nano VL 8B (optimized for docs/vision on consumer hardware).
    model_id: str = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
    device: str = "cuda"

    # Inference Backend
    backend: str = "transformers" # "transformers", "tensorrt", "mlx", or "nim"
    engine_dir: Optional[str] = None # Path to TRT engine if backend is tensorrt

    # Browser
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720

    # Learning
    learning_rate: float = 1e-4
    ema_decay: float = 0.99
    update_budget: int = 100

    # Agent
    task_goal: str = "Complete the task on the current web page."

    # Paths
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AgentConfig":
        if not path:
            return cls()
            
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
