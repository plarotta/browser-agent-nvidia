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
def record(url, task):
    """Record a new trajectory for a task."""
    logger.info(f"Starting recording for task '{task}' at {url}...")
    # Config loading could go here
    config = AgentConfig(log_dir=f"logs/{task}", headless=False)
    
    agent = AgentRuntime(config=config)
    agent.start(start_url=url)
    input("Press Enter to stop recording via CLI (Concept)...")
    agent.stop()

@cli.command()
@click.option("--task", required=True, help="Task name to run")
@click.option("--goal", required=True, help="Natural language goal for the agent (e.g. 'Log in to the website')")
@click.option("--headless/--no-headless", default=True, help="Run in headless mode")
@click.option("--train/--no-train", default=False, help="Enable online training (SDFT)")
@click.option("--backend", default="transformers", type=click.Choice(["transformers", "tensorrt", "mlx"]), help="Inference backend")
@click.option("--engine-dir", default=None, help="Path to TensorRT engine (required for tensorrt backend)")
@click.option("--url", default="https://google.com", help="Starting URL for the task")
@click.option("--max-steps", default=15, help="Maximum number of agent steps")
@click.option("--model-id", default=None, help="Override the model ID (e.g. mlx-community/gemma-3-12b-it-qat-4bit)")
def run(task, goal, headless, train, backend, engine_dir, url, max_steps, model_id):
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
    
    # Initialize Policy
    # Note: In real usage, we would check for GPU availability here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing Policy on {device}...")
    
    # Instantiate Policy
    # We use a mocked model ID for dev if needed, or the real Llama 3.2
    policy = MultimodalPolicy(
        model_id=config.model_id,
        device=device,
        backend=config.backend,
        engine_dir=config.engine_dir
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
        agent.start(start_url=url)
        
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
def status():
    """Show agent status."""
    click.echo("Agent Status: Online")
    click.echo("Base Model: Llama 3.2 Vision")
    click.echo(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

if __name__ == "__main__":
    cli()
