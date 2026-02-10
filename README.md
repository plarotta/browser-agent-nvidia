# Self-Learning Browser Agent (Nemotron Edition)

A browser automation agent that uses a **Vision Language Model (VLM)** to observe web pages and take actions toward a natural-language goal. Optionally adapts at deployment time using **Self-Distillation Fine-Tuning (SDFT)**.

Powered by **NVIDIA Nemotron** (default: `Llama-3.1-Nemotron-Nano-VL-8B-V1`) on CUDA, or **Gemma-3-12B-QAT** on Apple Silicon via **MLX**. Supports a **remote vLLM server** for GPU-offloaded inference and LoRA-based deployment-time learning. Uses **Playwright** for browser control.

For system design and architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For a prioritized roadmap, see [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Features

- **Multimodal control**: DOM snapshots (interactive elements + visible page text) + screenshots for robust action selection.
- **Multiple backends**: Transformers (CUDA/CPU), TensorRT, MLX (Apple Silicon), NIM (NVIDIA API), or Remote vLLM (self-hosted GPU server).
- **Remote vLLM server**: Host Nemotron on a GPU box (e.g. RunPod) with LoRA hot-swapping, trajectory upload, and SDFT training — the Mac runs only the browser agent.
- **Compact, truncation-safe prompt**: Output format instructions come first so the model always sees them, even when DOM is long.
- **Programmatic guards**: Prevents common small-model mistakes (e.g. skipping submit after typing).
- **Deployment-time adaptation**: SDFT with PEFT LoRA adapters, trainable on uploaded trajectories via the server.
- **Consumer hardware**: Targets NVIDIA RTX 3090/4090; MLX for Mac (tested on M4/24GB with 12B-4bit).

## Installation

This project uses **uv** for dependency management.

```bash
git clone https://github.com/your-repo/browser-agent-nvidia.git
cd browser-agent-nvidia

uv sync
uv run playwright install
```

**Server (GPU box only):** Install with server extras for FastAPI, vLLM, and PEFT:

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
|--------|-------------|---------|
| `--backend` | `transformers`, `tensorrt`, `mlx`, `nim`, or `remote_vllm` | `transformers` |
| `--engine-dir` | Path to TensorRT engine (for `tensorrt`) | -- |
| `--model-id` | Override model (e.g. `mlx-community/gemma-3-12b-it-qat-4bit`) | backend-dependent |
| `--server-url` | Remote vLLM server URL (for `remote_vllm`) | `http://localhost:8080` |
| `--url` | Starting URL | `https://google.com` |
| `--max-steps` | Max agent steps | `15` |
| `--train` | Enable online SDFT | off |

**Examples:**

```bash
# MLX (Apple Silicon) — uses Gemma-3-12B-QAT by default
uv run python -m src.main run --task "demo" --goal "Tell me the temperature in NYC" --backend mlx --no-headless

# Remote vLLM (browser on Mac, model on GPU box)
export VLLM_SERVER_URL="http://<gpu-box-ip>:8080"
uv run python -m src.main run --task "demo" --goal "Search for weather in NYC" \
  --backend remote_vllm --url "https://google.com" --no-headless --max-steps 5

# Or pass server URL directly
uv run python -m src.main run --task "demo" --goal "Search for weather" \
  --backend remote_vllm --server-url "http://192.168.1.100:8080" --no-headless

# Override model
uv run python -m src.main run --task "demo" --goal "Search for X" --backend mlx --model-id "mlx-community/some-other-model" --no-headless

# With online learning (SDFT)
uv run python -m src.main run --task "demo" --goal "Complete the task" --train --no-headless
```

### Start the server (GPU box)

On a GPU machine (e.g. RunPod), start the vLLM inference server + FastAPI control plane:

```bash
pip install -e ".[server]"
./scripts/start_server.sh nvidia/Nemotron-Nano-12B-v2-VL-BF16
```

Or start manually:

```bash
uv run python -m src.main serve \
  --model-id nvidia/Nemotron-Nano-12B-v2-VL-BF16 \
  --host 0.0.0.0 --port 8080 \
  --vllm-url http://localhost:8000
```

**Server endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, vLLM status, active adapter |
| `/act` | POST | Inference (prompt + screenshot) |
| `/upload_trajectory` | POST | Upload trajectory tar.gz for training |
| `/train` | POST | Trigger SDFT training with PEFT LoRA |
| `/train/status` | GET | Check training job progress |
| `/adapters` | GET | List available LoRA adapters |
| `/adapters/{name}/load` | POST | Hot-load adapter into vLLM |
| `/adapters/{name}/unload` | POST | Unload adapter from vLLM |

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

Edit `src/utils/config.py` for defaults: `model_id`, `device`, `backend`, `engine_dir`, `server_url`, viewport, learning rate, EMA decay, update budget, and paths.

## Project structure

| Path | Purpose |
|------|---------|
| `src/agent/` | Agent runtime (step loop, prompt, guards), action parser |
| `src/browser_interaction/` | Playwright session manager, action executor |
| `src/observation/` | DOM snapshotter (elements + page text), visual capture |
| `src/policy/` | Multimodal policy dispatch, backend implementations (Transformers, TensorRT, MLX, NIM, Remote vLLM) |
| `src/server/` | FastAPI control plane (`api.py`), LoRA adapter manager, SDFT trainer worker |
| `src/shared/` | Pydantic request/response schemas shared between client and server |
| `src/sdft/` | Self-distillation (EMA teacher, gating, loss) |
| `src/safety/` | Checkpoint save/load/rollback |
| `src/evaluation/` | Goal spec, success checker (not used in runtime; available for offline evaluation) |
| `src/utils/` | Config, trajectory logger, trajectory uploader |
| `scripts/` | Server launch scripts (RunPod) |

## License

MIT
