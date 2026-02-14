# Self-Learning Browser Agent (Nemotron Edition)

A browser automation agent that uses a **Vision Language Model (VLM)** to observe web pages and take actions toward a natural-language goal. Optionally adapts at deployment time using **Self-Distillation Fine-Tuning (SDFT)**.

Powered by **NVIDIA Nemotron** (default: `Llama-3.1-Nemotron-Nano-VL-8B-V1`) on CUDA, or **Gemma-3-12B-QAT** on Apple Silicon via **MLX**. All inference runs locally — a remote GPU instance is used only for LoRA adapter training. Uses **Playwright** for browser control.

For system design and architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For a prioritized roadmap, see [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Features

- **Multimodal control**: DOM snapshots (interactive elements + visible page text) + screenshots for robust action selection.
- **Multiple backends**: Transformers (CUDA/CPU), TensorRT, MLX (Apple Silicon), NIM (NVIDIA API).
- **Remote training server**: Upload trajectories to a GPU box (e.g. RunPod), train LoRA adapters via SDFT, download and run locally.
- **Compact, truncation-safe prompt**: Output format instructions come first so the model always sees them, even when DOM is long.
- **Programmatic guards**: Prevents common small-model mistakes (e.g. skipping submit after typing).
- **Deployment-time adaptation**: SDFT with PEFT LoRA adapters, trainable on uploaded trajectories via the server or locally on Apple Silicon via MLX.
- **NIM-enriched teacher demos**: Optionally calls the NIM VLM API during training to generate rich explanations of expert actions, improving teacher logit quality.
- **Consumer hardware**: Targets NVIDIA RTX 3090/4090; MLX for Mac (tested on M4/24GB with 12B-4bit).

## Installation

This project uses **uv** for dependency management.

```bash
git clone https://github.com/your-repo/browser-agent-nvidia.git
cd browser-agent-nvidia

uv sync
uv run playwright install
```

**Server (GPU box only):** Install with server extras for FastAPI and PEFT:

```bash
pip install -e ".[server]"
```

## Usage

### Record a demonstration

Capture a trajectory (browser opens; press Enter to stop).

```bash
uv run python -m src.main record --url "https://example.com" --task "login_demo"
```

### Run the agent (inference)

Execute the agent toward a natural-language goal. Requires `--task` and `--goal`.

```bash
uv run python -m src.main run --task "demo" --goal "Search for flights to NYC" --url "https://google.com" --no-headless
```

**Options:**

| Option | Description | Default |
| -------- | ------------- | --------- |
| `--backend` | `transformers`, `tensorrt`, `mlx`, or `nim` | `transformers` |
| `--engine-dir` | Path to TensorRT engine (for `tensorrt`) | -- |
| `--model-id` | Override model (e.g. `mlx-community/gemma-3-12b-it-qat-4bit`) | backend-dependent |
| `--url` | Starting URL | `https://google.com` |
| `--max-steps` | Max agent steps | `15` |
| `--adapter-path` | Path to LoRA adapter directory (for MLX) | -- |
| `--train` | Enable online SDFT | off |

**Examples:**

```bash
# MLX (Apple Silicon) — uses Gemma-3-12B-QAT by default
uv run python -m src.main run --task "demo" --goal "Tell me the temperature in NYC" --backend mlx --no-headless

# Override model
uv run python -m src.main run --task "demo" --goal "Search for X" --backend mlx --model-id "mlx-community/some-other-model" --no-headless

# With a trained LoRA adapter (MLX)
uv run python -m src.main run --task "demo" --goal "Search for flights" --backend mlx --adapter-path ./adapters/my_adapter --no-headless

# With online learning (SDFT)
uv run python -m src.main run --task "demo" --goal "Complete the task" --train --no-headless
```

### Train an adapter (MLX, Apple Silicon)

Run SDFT training on collected trajectories to produce a LoRA adapter.

```bash
# Basic (raw expert actions as teacher demos)
uv run python -m src.main train --task "demo" --no-enrich

# With NIM-enriched teacher demos (recommended, requires NVIDIA_API_KEY)
NVIDIA_API_KEY=nvapi-... uv run python -m src.main train --task "demo" --enrich
```

**Options:**

| Option | Description | Default |
| -------- | ------------- | --------- |
| `--task` | Task name (trajectory loaded from `logs/{task}_run`) | required |
| `--trajectory-dir` | Override trajectory directory | `logs/{task}_run` |
| `--model-id` | Model to fine-tune | `mlx-community/gemma-3-12b-it-qat-4bit` |
| `--adapter-path` | Where to save LoRA adapter | `./adapters/local` |
| `--epochs` | Training epochs | `2` |
| `--lr` | Learning rate | `1e-5` |
| `--lora-rank` | LoRA rank | `16` |
| `--ema-alpha` | EMA update rate for teacher | `0.02` |
| `--enrich/--no-enrich` | Enrich teacher demos via NIM API | `--enrich` |

When `--enrich` is enabled and `NVIDIA_API_KEY` is set, each training sample is sent to the NIM VLM (default: `meta/llama-3.2-90b-vision-instruct`) to generate a rich explanation of *why* the expert action is correct (page context, element rationale, expected outcome). This produces more informative teacher logits. If the API key is missing or a call fails, training falls back to raw action JSON seamlessly. Enrichment results are cached in `.enrichment_cache/` to avoid redundant API calls across runs.

See [TRAINING_PLAN.md](TRAINING_PLAN.md) for a step-by-step walkthrough.

### Deploy: remote round-trip (record → train → download)

The `deploy` command orchestrates the full remote training pipeline: upload a trajectory to the server, trigger SDFT training, poll until complete, and download the trained adapter locally.

```bash
# Record a demo first
uv run python -m src.main record --url "https://google.com" --task my_task --goal "Search for hello world"

# Deploy to remote GPU server
uv run python -m src.main deploy \
  --task my_task \
  --server-url http://<runpod-ip>:8080 \
  --adapter-name my_task \
  --ema-alpha 0.1
```

**Options:**

| Option | Description | Default |
| -------- | ------------- | --------- |
| `--task` | Task name (trajectory loaded from `logs/{task}`) | required |
| `--trajectory-dir` | Override trajectory directory | `logs/{task}` |
| `--server-url` | Remote server URL | required |
| `--adapter-name` | Name for the adapter on the server | required |
| `--adapter-path` | Local path to save downloaded adapter | `./adapters/{adapter_name}` |
| `--ema-alpha` | EMA update rate for teacher (try 0.1-0.3 for few-shot) | `0.1` |
| `--enrich/--no-enrich` | Enrich teacher demos via NIM API | `--enrich` |
| `--epochs` | Training epochs | `2` |
| `--lr` | Learning rate | `1e-4` |
| `--lora-rank` | LoRA rank | `16` |
| `--poll-interval` | Seconds between training status polls | `10` |

After completion, the command prints the `run` invocation to use the downloaded adapter.

### Start the training server (GPU box)

On a GPU machine (e.g. RunPod), start the training control plane:

```bash
pip install -e ".[server]"
uv run python -m src.main serve \
  --model-id nvidia/Nemotron-Nano-12B-v2-VL-BF16 \
  --host 0.0.0.0 --port 8080
```

**Server endpoints:**

| Endpoint | Method | Purpose |
| ---------- | -------- | --------- |
| `/health` | GET | Health check + training status |
| `/upload_trajectory` | POST | Upload trajectory tar.gz for training |
| `/train` | POST | Trigger SDFT training with PEFT LoRA |
| `/train/status` | GET | Check training job progress |
| `/adapters` | GET | List available LoRA adapters |
| `/adapters/{name}/download` | GET | Download adapter as tar.gz |

### Status

```bash
uv run python -m src.main status
```

### Demo script

```bash
./run_demo.sh
```

## How it works

Each step the agent:

1. **Observes** the page: extracts interactive DOM elements (with stable `data-agent-id`s) and visible page text via `innerText`, plus a screenshot.
2. **Builds a prompt** with goal, output format, rules, action history, and DOM summary. The prompt is structured so format instructions come first (truncation-safe for small models).
3. **Generates** a JSON action via the VLM (TYPE, CLICK, PRESS_ENTER, SCROLL, WAIT, NAVIGATE, or FINISH).
4. **Parses** the output: primary JSON parser, fallback regex parser (line-start anchored to avoid phantom matches), and done-language detection.
5. **Applies guards**: e.g. if the model tries to FINISH without submitting a search, the runtime overrides to PRESS_ENTER.
6. **Executes** the action via Playwright. PRESS_ENTER waits for page load. TYPE rejects empty values.
7. **Terminates** when the model outputs FINISH (with the answer in `value`), or at `max_steps`.

## Configuration

Edit `src/utils/config.py` for defaults: `model_id`, `device`, `backend`, `engine_dir`, viewport, learning rate, EMA decay, update budget, and paths.

## Project structure

| Path | Purpose |
| ------ | --------- |
| `src/agent/` | Agent runtime (step loop, prompt, guards), action parser |
| `src/browser_interaction/` | Playwright session manager, action executor |
| `src/observation/` | DOM snapshotter (elements + page text), visual capture |
| `src/policy/` | Multimodal policy dispatch, backend implementations (Transformers, TensorRT, MLX, NIM) |
| `src/server/` | FastAPI training control plane (`api.py`), SDFT trainer worker |
| `src/shared/` | Pydantic request/response schemas shared between client and server |
| `src/sdft/` | Self-distillation: EMA teacher/gating (`sdft_module.py`), MLX SDFT trainer (`sdft_trainer_mlx.py`), shared enrichment module with caching (`enrichment.py`) |
| `src/safety/` | Checkpoint save/load/rollback |
| `src/evaluation/` | Goal spec, success checker (not used in runtime; available for offline evaluation) |
| `src/utils/` | Config, trajectory logger, trajectory uploader (upload + adapter download) |
| `scripts/` | Server launch scripts (RunPod) |

## License

MIT
