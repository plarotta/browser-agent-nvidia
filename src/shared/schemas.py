from pydantic import BaseModel
from typing import Optional, List


class ActRequest(BaseModel):
    prompt: str
    screenshot_base64: str
    adapter_name: Optional[str] = None


class ActResponse(BaseModel):
    raw_output: str


class TrainRequest(BaseModel):
    trajectory_ids: List[str]
    num_epochs: int = 1
    learning_rate: float = 1e-4
    lora_rank: int = 16
    adapter_name: str = "browser_agent"


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
    is_active: bool


class HealthResponse(BaseModel):
    status: str
    vllm_ready: bool
    model_id: str
    active_adapter: Optional[str] = None
