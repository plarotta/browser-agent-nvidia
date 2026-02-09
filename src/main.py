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
@click.option("--headless/--no-headless", default=True, help="Run in headless mode")
@click.option("--train/--no-train", default=False, help="Enable online training (SDFT)")
def run(task, headless, train):
    """Run the agent on a specific task."""
    logger.info(f"Running task '{task}' (Training: {train})...")
    
    # Load Config
    config = AgentConfig(log_dir=f"logs/{task}_run", headless=headless)
    
    # Initialize Policy
    # Note: In real usage, we would check for GPU availability here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing Policy on {device}...")
    
    # Instantiate Policy
    # We use a mocked model ID for dev if needed, or the real Llama 3.2
    policy = MultimodalPolicy(device=device)
    
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
        agent.start(start_url="https://example.com") # Should come from task config
        
        # Run loop
        max_steps = 5 
        for _ in range(max_steps):
            success = agent.step()
            if not success:
               logger.warning("Step failed.")
               
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
