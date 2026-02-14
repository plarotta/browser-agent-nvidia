import os
import click
from src.utils.config import AgentConfig
from src.agent.agent_runtime import AgentRuntime
from src.policy.multimodal_policy import MultimodalPolicy
from src.sdft.sdft_module import SDFTModule
from src.policy.adapter_layer import AdapterLayer
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cli")

@click.group()
def cli():
    """Browser Agent CLI - Record, Run, and Train."""
    pass

@cli.command()
@click.option("--url", required=True, help="URL to start recording from")
@click.option("--task", required=True, help="Name of the task")
@click.option("--goal", required=True, help="Natural language goal description")
def record(url, task, goal):
    """Record a human demonstration trajectory."""
    import warnings
    from src.recorder.human_recorder import HumanRecorder

    logger.info(f"Starting recording for task '{task}' at {url}")
    logger.info(f"Goal: {goal}")

    recorder = HumanRecorder(log_dir=f"logs/{task}", goal=goal)
    recorder.start(url)
    path = recorder.run()  # blocks until Ctrl+C
    logger.info(f"Trajectory saved to {path}")

    # Suppress noisy asyncio errors from Playwright subprocess teardown
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine.*was never awaited")
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

@cli.command()
@click.option("--task", required=True, help="Task name to run")
@click.option("--goal", required=True, help="Natural language goal for the agent (e.g. 'Log in to the website')")
@click.option("--headless/--no-headless", default=True, help="Run in headless mode")
@click.option("--train/--no-train", default=False, help="Enable online training (SDFT)")
@click.option("--backend", default="transformers", type=click.Choice(["transformers", "tensorrt", "mlx", "nim"]), help="Inference backend")
@click.option("--engine-dir", default=None, help="Path to TensorRT engine (required for tensorrt backend)")
@click.option("--url", default="https://google.com", help="Starting URL for the task")
@click.option("--max-steps", default=15, help="Maximum number of agent steps")
@click.option("--model-id", default=None, help="Override the model ID (e.g. mlx-community/gemma-3-12b-it-qat-4bit)")
@click.option("--adapter-path", default=None, help="Path to LoRA adapter directory (for MLX backend)")
def run(task, goal, headless, train, backend, engine_dir, url, max_steps, model_id, adapter_path):
    """Run the agent on a specific task."""
    logger.info(f"Running task '{task}' (Training: {train}, Backend: {backend})")
    logger.info(f"Goal: {goal}")

    config = AgentConfig(
        log_dir=f"logs/{task}_run",
        headless=headless,
        backend=backend,
        engine_dir=engine_dir,
        task_goal=goal,
    )
    
    # If user passed --model-id, use it directly
    if model_id:
        config.model_id = model_id
        logger.info(f"Using user-specified model: {config.model_id}")
    # Otherwise, override for MLX if using default Nemotron (which doesn't run on MLX)
    elif backend == "mlx" and config.model_id == "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1":
        config.model_id = "mlx-community/gemma-3-12b-it-qat-4bit"
        logger.info(f"Using MLX default model: {config.model_id}")
    elif backend == "nim" and config.model_id == "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1":
        config.model_id = "nvidia/llama-4-maverick-17b-128e-instruct"
        logger.info(f"Using NIM default model: {config.model_id}")
    
    # Initialize Policy
    # Note: In real usage, we would check for GPU availability here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing Policy on {device}...")
    
    # Instantiate Policy
    # We use a mocked model ID for dev if needed, or the real Llama 3.2
    if adapter_path:
        logger.info(f"Using adapter: {adapter_path}")

    policy = MultimodalPolicy(
        model_id=config.model_id,
        device=device,
        backend=config.backend,
        engine_dir=config.engine_dir,
        adapter_path=adapter_path,
    )

    # Initialize SDFT if training enabled
    sdft = None
    if train:
        logger.info("Initializing SDFT Module...")
        # Inject adapter logic here conceptually
        # In a full implementation, we'd wrap policy.model with PEFT/LoRA here
        sdft = SDFTModule(policy, ema_decay=config.ema_decay)

    # Initialize Agent
    agent = AgentRuntime(config=config, policy=policy, sdft=sdft)
    
    try:
        download_path = agent.start(start_url=url)

        if download_path:
            logger.info(f"File downloaded: {download_path}")
        else:
            # Run loop (stop on task done or max_steps)
            for _ in range(max_steps):
                success, done = agent.step()
                if not success:
                    logger.warning("Step failed.")
                if done:
                    logger.info("Task complete: stopping.")
                    break
               
    except KeyboardInterrupt:
        logger.info("Stopping agent...")
    finally:
        agent.stop()

@cli.command()
@click.option("--task", required=True, help="Task name (used for log dir path)")
@click.option("--trajectory-dir", default=None, help="Path to trajectory dir (defaults to logs/{task}_run)")
@click.option("--model-id", default=None, help="Model to fine-tune")
@click.option("--adapter-path", default="./adapters/local", help="Where to save LoRA adapter")
@click.option("--epochs", default=2, help="Training epochs (paper recommends 2)")
@click.option("--lr", default=1e-5, type=float, help="Learning rate")
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--ema-alpha", default=0.02, type=float, help="EMA update rate for teacher")
@click.option("--enrich/--no-enrich", default=True, help="Enrich teacher demos via NIM API (requires NVIDIA_API_KEY)")
@click.option("--wandb-project", default=None, help="W&B project name (enables logging)")
@click.option("--wandb-run-name", default=None, help="W&B run name (defaults to task name)")
def train(task, trajectory_dir, model_id, adapter_path, epochs, lr, lora_rank, ema_alpha, enrich, wandb_project, wandb_run_name):
    """Run SDFT training on collected trajectories (MLX, Apple Silicon)."""
    from src.sdft.sdft_trainer_mlx import run_sdft_training

    if trajectory_dir is None:
        trajectory_dir = f"logs/{task}_run"

    if model_id is None:
        model_id = "mlx-community/gemma-3-12b-it-qat-4bit"

    logger.info(f"SDFT Training: task={task}, model={model_id}")
    logger.info(f"Trajectory dir: {trajectory_dir}, Adapter path: {adapter_path}")

    result = run_sdft_training(
        model_id=model_id,
        trajectory_dirs=[trajectory_dir],
        adapter_save_path=adapter_path,
        num_epochs=epochs,
        learning_rate=lr,
        lora_rank=lora_rank,
        ema_alpha=ema_alpha,
        enrich=enrich,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name or task,
    )

    logger.info(f"Training result: {result}")

@cli.command()
@click.option("--task", required=True, help="Task name (used to find trajectory in logs/{task})")
@click.option("--trajectory-dir", default=None, help="Path to trajectory dir (defaults to logs/{task})")
@click.option("--server-url", required=True, help="Remote server URL (e.g. http://<runpod-ip>:8080)")
@click.option("--adapter-name", required=True, help="Name for the adapter on the server")
@click.option("--adapter-path", default=None, help="Local path to save downloaded adapter (defaults to ./adapters/{adapter_name})")
@click.option("--ema-alpha", default=0.1, type=float, help="EMA update rate for teacher (try 0.1-0.3 for few-shot)")
@click.option("--enrich/--no-enrich", default=True, help="Enrich teacher demos via NIM API")
@click.option("--epochs", default=2, help="Training epochs")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--poll-interval", default=10, type=int, help="Seconds between training status polls")
@click.option("--wandb-project", default=None, help="W&B project name (enables server-side logging)")
def deploy(task, trajectory_dir, server_url, adapter_name, adapter_path, ema_alpha, enrich, epochs, lr, lora_rank, poll_interval, wandb_project):
    """Deploy: upload trajectory, train on remote GPU, download adapter.

    Orchestrates the full remote round-trip:
    1. Upload trajectory to server
    2. Trigger SDFT training
    3. Poll until training completes
    4. Download the trained adapter locally
    """
    import time
    import requests
    from src.utils.trajectory_uploader import TrajectoryUploader

    if trajectory_dir is None:
        trajectory_dir = f"logs/{task}"

    if adapter_path is None:
        adapter_path = f"./adapters/{adapter_name}"

    if not os.path.isdir(trajectory_dir):
        logger.error(f"Trajectory dir not found: {trajectory_dir}")
        raise click.Abort()

    uploader = TrajectoryUploader(server_url)

    # Step 1: Upload trajectory
    logger.info(f"Uploading trajectory from {trajectory_dir}...")
    traj_id = uploader.upload(trajectory_dir)
    logger.info(f"Trajectory uploaded: {traj_id}")

    # Step 2: Trigger training
    logger.info("Triggering SDFT training on server...")
    train_resp = requests.post(
        f"{server_url}/train",
        json={
            "trajectory_ids": [traj_id],
            "num_epochs": epochs,
            "learning_rate": lr,
            "lora_rank": lora_rank,
            "adapter_name": adapter_name,
            "ema_alpha": ema_alpha,
            "enrich": enrich,
            "wandb_project": wandb_project,
            "wandb_run_name": f"{adapter_name}_deploy",
        },
        timeout=30,
    )
    train_resp.raise_for_status()
    train_data = train_resp.json()
    logger.info(f"Training queued: job_id={train_data.get('job_id', 'unknown')}")

    # Step 3: Poll until complete
    logger.info(f"Polling training status every {poll_interval}s...")
    while True:
        time.sleep(poll_interval)
        status_resp = requests.get(f"{server_url}/train/status", timeout=10)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get("status", "unknown")
        message = status_data.get("message", "")

        if status == "completed":
            logger.info(f"Training completed: {message}")
            break
        elif status == "failed":
            logger.error(f"Training failed: {message}")
            raise click.Abort()
        else:
            logger.info(f"Training status: {status} — {message}")

    # Step 4: Download adapter
    logger.info(f"Downloading adapter '{adapter_name}' to {adapter_path}...")
    uploader.download_adapter(adapter_name, adapter_path)
    logger.info(f"Adapter saved to {adapter_path}")

    # Step 5: Print run command
    click.echo("\nAdapter ready! Run inference with:")
    click.echo(
        f"  uv run python -m src.main run "
        f"--task {task} --goal '<your goal>' "
        f"--backend mlx --adapter-path {adapter_path} "
        f"--url '<start_url>' --no-headless"
    )


@cli.command()
def status():
    """Show agent status."""
    click.echo("Agent Status: Online")
    click.echo("Base Model: Llama 3.2 Vision")
    click.echo(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

@cli.command()
@click.option("--model-id", default="nvidia/Nemotron-Nano-12B-v2-VL-BF16", help="Model to fine-tune")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8080, help="Server port")
@click.option("--adapters-dir", default="./adapters", help="Directory for LoRA adapters")
@click.option("--trajectories-dir", default="./trajectories", help="Directory for uploaded trajectories")
def serve(model_id, host, port, adapters_dir, trajectories_dir):
    """Start the training control plane server (GPU box)."""
    import uvicorn
    from src.server.api import create_app

    logger.info(f"Starting training server: model={model_id}")
    app = create_app(
        model_id=model_id,
        adapters_dir=adapters_dir,
        trajectories_dir=trajectories_dir,
    )
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    cli()
