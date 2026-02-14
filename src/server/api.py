"""Training server control plane (FastAPI).

Receives trajectory uploads, runs SDFT training, and serves trained adapters
for download. No inference — all inference happens locally on the Mac via MLX.
"""

import json
import os
import tarfile
import tempfile
import uuid
import logging

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from src.shared.schemas import TrainRequest, TrainResponse, HealthResponse, AdapterInfo
from src.server.trainer_worker import run_training

logger = logging.getLogger(__name__)


def create_app(
    model_id: str,
    adapters_dir: str = "./adapters",
    trajectories_dir: str = "./trajectories",
) -> FastAPI:
    app = FastAPI(title="Browser Agent Training Server")

    os.makedirs(adapters_dir, exist_ok=True)
    os.makedirs(trajectories_dir, exist_ok=True)

    # Adapter metadata (simple JSON file)
    meta_path = os.path.join(adapters_dir, "adapters_meta.json")

    def _load_meta() -> dict:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return {}

    def _save_meta(meta: dict):
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Training job state
    training_state = {"status": "idle", "job_id": None, "message": ""}

    # ---- Health ----
    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_id=model_id,
            training_status=training_state["status"],
        )

    # ---- Trajectory Upload ----
    @app.post("/upload_trajectory")
    async def upload_trajectory(file: UploadFile = File(...)):
        traj_id = str(uuid.uuid4())[:8]
        traj_dir = os.path.join(trajectories_dir, traj_id)
        os.makedirs(traj_dir, exist_ok=True)

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
            traj_dirs = [
                os.path.join(trajectories_dir, tid) for tid in req.trajectory_ids
            ]
            adapter_path = os.path.join(adapters_dir, req.adapter_name)
            resume_from = adapter_path if os.path.exists(adapter_path) else None

            result = run_training(
                model_id=model_id,
                trajectory_dirs=traj_dirs,
                adapter_save_path=adapter_path,
                num_epochs=req.num_epochs,
                learning_rate=req.learning_rate,
                lora_rank=req.lora_rank,
                resume_from=resume_from,
                ema_alpha=req.ema_alpha,
                enrich=req.enrich,
                wandb_project=req.wandb_project,
                wandb_run_name=req.wandb_run_name,
            )

            if result["status"] == "completed":
                from datetime import datetime, timezone
                meta = _load_meta()
                meta[req.adapter_name] = {
                    "path": adapter_path,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "training_steps": result["total_steps"],
                }
                _save_meta(meta)
                training_state["status"] = "completed"
                training_state["message"] = (
                    f"Trained {result['total_steps']} steps, "
                    f"avg loss {result['avg_loss']:.4f}"
                )
            else:
                training_state["status"] = "failed"
                training_state["message"] = result.get("message", "Unknown error")

        except Exception as e:
            logger.exception("Training failed")
            training_state["status"] = "failed"
            training_state["message"] = str(e)

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
        meta = _load_meta()
        return [
            AdapterInfo(
                name=name,
                path=info["path"],
                created_at=info["created_at"],
                training_steps=info["training_steps"],
            )
            for name, info in meta.items()
        ]

    @app.get("/adapters/{name}/download")
    def download_adapter(name: str):
        adapter_path = os.path.join(adapters_dir, name)
        if not os.path.isdir(adapter_path):
            raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")

        tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        tmp_path = tmp.name
        tmp.close()

        with tarfile.open(tmp_path, "w:gz") as tar:
            for fname in os.listdir(adapter_path):
                fpath = os.path.join(adapter_path, fname)
                if os.path.isfile(fpath):
                    tar.add(fpath, arcname=fname)

        def iterfile():
            try:
                with open(tmp_path, "rb") as f:
                    yield from iter(lambda: f.read(64 * 1024), b"")
            finally:
                os.remove(tmp_path)

        return StreamingResponse(
            iterfile(),
            media_type="application/gzip",
            headers={"Content-Disposition": f"attachment; filename={name}.tar.gz"},
        )

    return app
