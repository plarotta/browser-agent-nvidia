from pydantic import BaseModel
from typing import Optional, List


class TrainRequest(BaseModel):
    trajectory_ids: List[str]
    num_epochs: int = 1
    learning_rate: float = 1e-4
    lora_rank: int = 16
    adapter_name: str = "browser_agent"
    ema_alpha: float = 0.02
    enrich: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


class TrainResponse(BaseModel):
    status: str  # "queued" | "running" | "completed" | "failed"
    job_id: str
    adapter_name: Optional[str] = None
    message: str = ""


class AdapterInfo(BaseModel):
    name: str
    path: str
    created_at: str
    training_steps: int


class HealthResponse(BaseModel):
    status: str
    model_id: str
    training_status: str = "idle"
