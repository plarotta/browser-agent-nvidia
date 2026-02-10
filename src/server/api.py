import base64
import io
import os
import tarfile
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from openai import OpenAI

from src.shared.schemas import (
    ActRequest, ActResponse,
    TrainRequest, TrainResponse,
    HealthResponse, AdapterInfo,
)
from src.server.adapter_manager import AdapterManager
from src.server.trainer_worker import run_training

logger = logging.getLogger(__name__)


def create_app(
    model_id: str,
    vllm_url: str = "http://localhost:8000",
    adapters_dir: str = "./adapters",
    trajectories_dir: str = "./trajectories",
) -> FastAPI:
    app = FastAPI(title="Browser Agent vLLM Server")

    adapter_mgr = AdapterManager(adapters_dir, vllm_url, model_id)
    vllm_client = OpenAI(base_url=f"{vllm_url}/v1", api_key="unused")
    os.makedirs(trajectories_dir, exist_ok=True)

    # Training job state
    training_state = {"status": "idle", "job_id": None, "message": ""}

    # ---- Health ----
    @app.get("/health", response_model=HealthResponse)
    def health():
        vllm_ready = adapter_mgr.is_vllm_ready()
        return HealthResponse(
            status="ok" if vllm_ready else "vllm_unavailable",
            vllm_ready=vllm_ready,
            model_id=model_id,
            active_adapter=adapter_mgr.active_adapter,
        )

    # ---- Inference ----
    @app.post("/act", response_model=ActResponse)
    def act(req: ActRequest):
        messages = [
            {"role": "system", "content": "/no_think"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{req.screenshot_base64}",
                        },
                    },
                    {"type": "text", "text": req.prompt},
                ],
            },
        ]

        extra_body = {}
        if req.adapter_name:
            extra_body["model"] = req.adapter_name

        completion = vllm_client.chat.completions.create(
            model=req.adapter_name or model_id,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            extra_body=extra_body if extra_body else None,
        )

        raw = completion.choices[0].message.content
        return ActResponse(raw_output=raw)

    # ---- Trajectory Upload ----
    @app.post("/upload_trajectory")
    async def upload_trajectory(file: UploadFile = File(...)):
        traj_id = str(uuid.uuid4())[:8]
        traj_dir = os.path.join(trajectories_dir, traj_id)
        os.makedirs(traj_dir, exist_ok=True)

        # Save and extract tar.gz
        tar_path = os.path.join(traj_dir, "upload.tar.gz")
        content = await file.read()
        with open(tar_path, "wb") as f:
            f.write(content)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=traj_dir, filter="data")

        os.remove(tar_path)
        logger.info(f"Trajectory uploaded: {traj_id}")
        return {"trajectory_id": traj_id}

    # ---- Training ----
    def _run_training_job(req: TrainRequest, job_id: str):
        training_state["status"] = "running"
        training_state["job_id"] = job_id
        try:
            # Pause vLLM to free GPU
            adapter_mgr.stop_vllm()

            traj_dirs = [
                os.path.join(trajectories_dir, tid) for tid in req.trajectory_ids
            ]
            adapter_path = os.path.join(adapters_dir, req.adapter_name)

            # Check for existing adapter to resume from
            resume_from = adapter_path if os.path.exists(adapter_path) else None

            result = run_training(
                model_id=model_id,
                trajectory_dirs=traj_dirs,
                adapter_save_path=adapter_path,
                num_epochs=req.num_epochs,
                learning_rate=req.learning_rate,
                lora_rank=req.lora_rank,
                resume_from=resume_from,
            )

            if result["status"] == "completed":
                adapter_mgr.register_adapter(req.adapter_name, result["total_steps"])
                training_state["status"] = "completed"
                training_state["message"] = f"Trained {result['total_steps']} steps, avg loss {result['avg_loss']:.4f}"
            else:
                training_state["status"] = "failed"
                training_state["message"] = result.get("message", "Unknown error")

            # Restart vLLM with new adapter
            adapter_mgr.start_vllm(model_id, req.adapter_name)

        except Exception as e:
            logger.exception("Training failed")
            training_state["status"] = "failed"
            training_state["message"] = str(e)
            # Try to restart vLLM even on failure
            try:
                adapter_mgr.start_vllm(model_id)
            except Exception:
                logger.exception("Failed to restart vLLM after training error")

    @app.post("/train", response_model=TrainResponse)
    def train(req: TrainRequest, background_tasks: BackgroundTasks):
        if training_state["status"] == "running":
            return TrainResponse(
                status="running",
                job_id=training_state["job_id"],
                message="Training already in progress",
            )

        job_id = str(uuid.uuid4())[:8]
        training_state["status"] = "queued"
        training_state["job_id"] = job_id
        background_tasks.add_task(_run_training_job, req, job_id)

        return TrainResponse(
            status="queued",
            job_id=job_id,
            adapter_name=req.adapter_name,
            message="Training job queued",
        )

    @app.get("/train/status", response_model=TrainResponse)
    def train_status():
        return TrainResponse(
            status=training_state["status"],
            job_id=training_state.get("job_id", ""),
            message=training_state.get("message", ""),
        )

    # ---- Adapter Management ----
    @app.get("/adapters", response_model=list[AdapterInfo])
    def list_adapters():
        return adapter_mgr.list_adapters()

    @app.post("/adapters/{name}/load")
    def load_adapter(name: str):
        adapter_mgr.load_adapter(name)
        return {"status": "loaded", "name": name}

    @app.post("/adapters/{name}/unload")
    def unload_adapter(name: str):
        adapter_mgr.unload_adapter(name)
        return {"status": "unloaded", "name": name}

    return app
