# Self-Learning Browser Agent (Nemotron Edition)

A browser automation agent that learns from user demonstrations and adapts to UI changes during deployment using **Self-Distillation Fine-Tuning (SDFT)**.

Powered by **NVIDIA Nemotron** (Default: `Llama-3.1-Nemotron-Nano-VL-8B-V1`) and **Playwright**.

![Architecture](https://placehold.co/600x200?text=Architecture+Placeholder)

## 🚀 Features

- **One-Shot Learning**: Learn a task from a single user demonstration.
- **Robust Execution**: Uses multimodal inputs (DOM + Screenshots) to survive UI changes.
- **Deployment-Time Adaptation**: The agent fine-tunes itself on successful trajectories without human labeling.
- **Safety First**: Automatic rollback if performance degrades.
- **Consumer Hardware Friendly**: Optimized for NVIDIA RTX 3090/4090.

## 📦 Installation

This project uses `uv` for dependency management.

```bash
# 1. Clone the repo
git clone https://github.com/your-repo/browser-agent-nvidia.git
cd browser-agent-nvidia

# 2. Install dependencies
uv sync
uv run playwright install
```

## 🛠️ Usage

### 1. Record a Demonstration
Capture your workflow to teach the agent.

```bash
uv run python -m src.main record --url "https://example.com" --task "login_demo"
```

### 2. Run the Agent (Inference)
Execute the learned task.

```bash
uv run python -m src.main run --task "login_demo" --no-headless
```

### 3. Run with Online Learning (SDFT)
Enable self-distillation to adapt to changes.

```bash
uv run python -m src.main run --task "login_demo" --train
```

## 🧠 Configuration

Edit `src/utils/config.py` to change models or hyperparameters.

```python
# Default Model
model_id = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
device = "cuda" # or "cpu"
```

## 📂 Project Structure

- `src/agent`: Core runtime and orchestration.
- `src/browser_interaction`: Playwright wrappers.
- `src/observation`: Screenshot and DOM capture.
- `src/policy`: Llama 3.2 / Nemotron policy wrappers.
- `src/sdft`: Self-Distillation logic.
- `src/safety`: Checkpoint and rollback.

## 🛡️ Safety Mechanisms

- **Confidence Gating**: The agent only learns from high-confidence, successful steps.
- **Rollback**: If the success rate drops, the system automatically reverts to the last stable checkpoint (`src/safety/checkpoint_manager.py`).

## License
MIT
