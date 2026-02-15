# Self-Learning Browser Agent

A browser agent that learns from a single human demonstration. Show it how to do a task once, and it trains a LoRA adapter that makes it better at that task — all on consumer hardware.

## Why

Most "agentic" browser use today is either fragile (API/MCP wrappers), expensive (cloud-hosted vision models), or both. And conventional fine-tuning on raw demonstrations is a dead end for small models — they either can't learn the behavior, or they forget their general knowledge in the process.

This project takes a different approach:

- **Local-first.** All inference runs on your machine. No API calls per action, no cloud costs, no data leaving your laptop. A 12B quantized model on a Mac is free to run forever.
- **Learn from one demo, not thousands.** Record a single demonstration of a task. A large teacher model (Nemotron Ultra 253B) enriches each step with reasoning about *why* the action is correct, then Self-Distillation Fine-Tuning (SDFT) transfers that understanding into a lightweight LoRA adapter. The small model doesn't learn from raw actions alone — it learns from a much stronger model's interpretation of those actions.
- **Adapters, not fine-tunes.** [Self-Distillation Fine-Tuning (SDFT)](https://arxiv.org/abs/2601.19897) trains a LoRA adapter (~10-50MB), not a new model. The base model's general knowledge stays intact. You can stack adapters for different tasks or drop them entirely to get back to baseline.
- **Lightweight models become useful.** This pipeline turns a 12B quantized model (Gemma 3) into a capable browser agent for specific tasks — something that would otherwise require a much larger model or expensive API access.

## How It Works (overview)

1. **Record** a demonstration on your Mac (Playwright opens a browser, you perform the task)
2. **Deploy** the trajectory to a remote GPU, where it's enriched by **Nemotron Ultra 253B** and used for **SDFT**
3. **Run** the agent locally with the trained adapter — inference stays on your Mac via MLX

```text
Mac (M-series, 24GB)                  Remote GPU (RunPod / Lambda)
────────────────────                  ──────────────────────────────
Browser (Playwright)                  Training server (FastAPI)
Demo recording                        Receives trajectory uploads
Local inference (MLX + LoRA)          SDFT with PEFT LoRA
                                      Nemotron Ultra 253B enrichment
      ──── upload trajectory ────>
      <──── download adapter ─────
```

The pipeline is **backbone-agnostic** — the same workflow works whether the base model runs on NVIDIA GPUs (Transformers, TensorRT, NIM) or Apple Silicon (MLX). Training happens once on a GPU; the resulting adapter slots into any backend.

## Quickstart

### Prerequisites

- **Mac** (M-series, 24GB+ RAM) — or any machine with a supported backend
- **GPU instance** (RunPod, Lambda, etc. — 1x A100/H100/4090, 24GB+ VRAM) for training only
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on both machines
- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) for NIM enrichment (optional but recommended)

### 1. Mac Setup

```bash
git clone <repo-url> && cd browser-agent-nvidia
uv sync
uv run playwright install chromium
```

### 2. GPU Server Setup

The GPU instance only runs the training control plane — no inference.

```bash
git clone <repo-url> && cd browser-agent-nvidia
uv sync
uv pip install -e ".[server]"

# Gemma 3 is gated — accept the license on HuggingFace, then login
huggingface-cli login

# NIM enrichment (calls Nemotron Ultra 253B during training)
export NVIDIA_API_KEY="nvapi-..."

# Start the training server (model loads in 4-bit via bitsandbytes)
uv run python -m src.main serve \
  --model-id google/gemma-3-12b-it-qat-q4_0-unquantized \
  --port 8080
```

### 3. Record a Demo (Mac)

A browser opens. Perform the task yourself, then press Ctrl+C.

```bash
uv run python -m src.main record \
  --url "https://google.com" \
  --task google_search \
  --goal "Search for 'NVIDIA Nemotron'"
```

Trajectory saves to `logs/google_search/` (screenshots + DOM + actions).

### 4. Deploy: Train Remotely, Download Adapter (Mac)

One command uploads the demo, trains an adapter on the GPU, and downloads the result.

```bash
uv run python -m src.main deploy \
  --task google_search \
  --server-url http://<GPU_IP>:8080 \
  --adapter-name google_search \
  --ema-alpha 0.1 \
  --epochs 2
```

Takes a few minutes. The adapter downloads to `./adapters/google_search/`.

### 5. Run with Adapter (Mac)

All inference runs locally — no GPU needed.

```bash
uv run python -m src.main run \
  --task google_search \
  --goal "Search for 'NVIDIA Nemotron'" \
  --backend mlx \
  --adapter-path ./adapters/google_search \
  --url "https://google.com" \
  --no-headless
```

### 6. Verify: Base vs. Adapted

Run the same task with and without the adapter to confirm improvement.

```bash
# Baseline (no adapter)
uv run python -m src.main run \
  --task eval_base \
  --goal "Search for 'NVIDIA Nemotron'" \
  --backend mlx --url "https://google.com" --no-headless

# With adapter
uv run python -m src.main run \
  --task eval_adapted \
  --goal "Search for 'NVIDIA Nemotron'" \
  --backend mlx --adapter-path ./adapters/google_search \
  --url "https://google.com" --no-headless
```

---

## How It Works

### Agent Loop (each step)

1. **Observe** the page: extract interactive DOM elements (with stable IDs) and visible page text via `innerText`, plus a screenshot.
2. **Build a prompt** with goal, output format, rules, action history, and DOM summary. Format instructions come first so small models always see them, even when DOM is long.
3. **Generate** a JSON action via the VLM: TYPE, CLICK, PRESS_ENTER, SCROLL, WAIT, NAVIGATE, or FINISH.
4. **Parse** the output: JSON extraction, line-anchored fallback regex, done-language detection.
5. **Guard**: programmatic overrides for common small-model mistakes (e.g. FINISH without submitting a search → override to PRESS_ENTER).
6. **Execute** via Playwright. PRESS_ENTER waits for page load. TYPE rejects empty values.
7. **Terminate** when the model outputs FINISH (with the answer in `value`), or at `max_steps`.

### Training (SDFT)

This project adapts [Self-Distillation Fine-Tuning (SDFT)](https://arxiv.org/abs/2601.19897) for browser automation. The original method updates the model's full weights; we train LoRA adapters instead, keeping the base model frozen and the adapters small enough to swap per-task. The core training loop remains the same:

1. **Enrichment** (optional): Each demo step's DOM observation and expert action are sent to **Nemotron Ultra 253B** via the NIM API. The large model generates a rich explanation of *why* the action is correct — page context, element rationale, expected outcome. This produces more informative teacher signals than raw action JSON. Results are cached in `.enrichment_cache/`.
2. **On-policy rollout**: The student model generates its own continuation from the demo state (with sampling).
3. **Teacher forward**: EMA teacher weights process the enriched demo with the rollout tokens.
4. **Reverse KL loss**: `D_KL(student || teacher)` — no SFT term. The student learns to match the teacher's distribution.
5. **EMA update**: Teacher weights are updated as a moving average of student weights.
6. **Adapter save**: Only LoRA parameters (~10-50MB) are saved. PEFT adapters auto-remap to MLX format for cross-platform compatibility.

---

## CLI Reference

| Command | Purpose |
|---------|---------|
| `record` | Record a human demonstration |
| `run` | Run the agent toward a goal |
| `train` | Train a LoRA adapter locally (MLX) |
| `deploy` | Full remote round-trip: upload → train → poll → download |
| `serve` | Start the training server on a GPU box |
| `status` | Show agent status |

### Key Options

| Flag | Commands | Purpose |
|------|----------|---------|
| `--backend` | run | `transformers`, `tensorrt`, `mlx`, or `nim` (default: `transformers`) |
| `--model-id` | run, train, serve | Override model ID |
| `--adapter-path` | run, train | Load/save LoRA adapter (MLX) |
| `--ema-alpha` | train, deploy | EMA teacher update rate. 0.1-0.3 for few-shot (default 0.02). |
| `--enrich/--no-enrich` | train, deploy | Enrich teacher demos via Nemotron Ultra 253B. Needs `NVIDIA_API_KEY`. |
| `--epochs` | train, deploy | Training epochs (default 2) |
| `--lora-rank` | train, deploy | LoRA rank (default 16, use 8 if OOM) |
| `--url` | record, run | Starting URL |
| `--max-steps` | run | Max agent steps (default 15) |
| `--no-headless` | run | Show the browser window |

### Server Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check + training status |
| `/upload_trajectory` | POST | Upload trajectory tar.gz |
| `/train` | POST | Trigger SDFT training |
| `/train/status` | GET | Check training progress |
| `/adapters` | GET | List available adapters |
| `/adapters/{name}/download` | GET | Download adapter as tar.gz |

---

## Project Structure

| Path | Purpose |
|------|---------|
| `src/agent/` | Agent runtime (step loop, prompt, guards), action parser |
| `src/browser_interaction/` | Playwright session manager, action executor |
| `src/observation/` | DOM snapshotter (elements + page text), visual capture |
| `src/policy/` | Multimodal policy dispatch + backends (Transformers, TensorRT, MLX, NIM) |
| `src/server/` | FastAPI training control plane, SDFT trainer worker |
| `src/sdft/` | SDFT module (EMA teacher, KL loss), MLX SDFT trainer, enrichment with caching |
| `src/shared/` | Pydantic schemas shared between client and server |
| `src/safety/` | Checkpoint save/load/rollback |
| `src/evaluation/` | Goal spec, success checker (offline eval) |
| `src/utils/` | Config, trajectory logger, trajectory uploader |
| `docs/` | Architecture deep-dive, roadmap |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No training samples found" | Trajectory has no positive-reward steps. Re-record with successful actions. |
| Step-0 KL near zero | Increase `--ema-alpha` (try 0.2-0.3) or check enrichment is working. |
| OOM on Mac during training | Use `--lora-rank 8`. |
| OOM on GPU server | Model loads in 4-bit automatically. If still OOM, use `--lora-rank 8`. |
| HuggingFace 401 | Gemma 3 is gated. Run `huggingface-cli login` and accept the license. |
| Server 404 on adapter download | Training may have failed. Check `GET <server>/train/status`. |
| No enrichment during training | Ensure `NVIDIA_API_KEY` is exported on the server. |

## License

MIT
