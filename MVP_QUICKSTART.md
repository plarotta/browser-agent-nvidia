# MVP Quickstart: End-to-End Workflow

Record a demo on your Mac, train a LoRA adapter on a remote GPU, download it, run inference locally. All inference happens on your Mac via MLX — the GPU instance is only used for training.

```
Mac M4 (24GB)                         RunPod GPU
─────────────                         ──────────
Browser (Playwright)                  Training server (FastAPI)
Demo recording                        Receives trajectory uploads
Local inference (MLX + LoRA)          Runs SDFT LoRA training
                                      Nemotron Ultra 253B enrichment
      ──── upload trajectory ────>
      <──── download adapter ─────
```

## Prerequisites

- **Mac** (M-series, 24GB+ RAM)
- **GPU instance** (RunPod, Lambda, etc. — 1x A100/H100/4090, 24GB+ VRAM) for training only
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on both machines
- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) (for NIM enrichment — optional but recommended)
- `WANDB_API_KEY` (for W&B logging — optional)

---

## 1. Mac Setup

```bash
git clone <repo-url> && cd browser-agent-nvidia

uv sync
uv run playwright install chromium

# (Optional) W&B logging
uv pip install wandb
wandb login
```

## 2. GPU Instance Setup (training server)

The GPU instance only runs the training control plane — it does not serve inference.

```bash
git clone <repo-url> && cd browser-agent-nvidia

uv sync
uv pip install -e ".[server]"

# NIM enrichment (calls Nemotron Ultra 253B on build.nvidia.com during training)
export NVIDIA_API_KEY="nvapi-..."

# (Optional) W&B
export WANDB_API_KEY="..."

# Start the training server
uv run python -m src.main serve \
  --model-id gemma-3-12b-it-qat-4bit \
  --port 8080
```

Note the instance's public IP address.

## 3. Record a Demo (Mac)

A Playwright browser opens. Perform the task yourself, then press Ctrl+C.

```bash
uv run python -m src.main record \
  --url "https://google.com" \
  --task google_search \
  --goal "Search for 'NVIDIA Nemotron'"
```

Trajectory saves to `logs/google_search/` (screenshots + DOM + actions).

## 4. Train on GPU, Download Adapter (Mac)

One command uploads the demo, trains an adapter remotely, and downloads the result.

```bash
uv run python -m src.main deploy \
  --task google_search \
  --server-url http://<GPU_IP>:8080 \
  --adapter-name google_search \
  --ema-alpha 0.1 \
  --epochs 2
```

Add `--wandb-project browser-agent` for W&B logging.

Takes a few minutes. Adapter downloads to `./adapters/google_search/`.

## 5. Run with Adapter (Mac, local inference)

All inference runs locally on your Mac via MLX — no GPU needed.

```bash
uv run python -m src.main run \
  --task google_search \
  --goal "Search for 'NVIDIA Nemotron'" \
  --backend mlx \
  --adapter-path ./adapters/google_search \
  --url "https://google.com" \
  --no-headless
```

## 6. Verify: Base vs. Adapted

Run the same task with and without the adapter to confirm the training had an effect.

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

## Local-Only Alternative (no GPU instance)

Skip the remote server entirely and train on the Mac via MLX. Slower, but no GPU needed.

```bash
# 1. Record
uv run python -m src.main record \
  --url "https://google.com" --task my_task --goal "Search for hello world"

# 2. Train locally (with NIM enrichment)
NVIDIA_API_KEY=nvapi-... uv run python -m src.main train \
  --task my_task --trajectory-dir logs/my_task \
  --ema-alpha 0.1 --wandb-project browser-agent

# 3. Run with adapter
uv run python -m src.main run \
  --task my_task --goal "Search for hello world" \
  --backend mlx --adapter-path ./adapters/local \
  --url "https://google.com" --no-headless
```

---

## Key Options

| Flag | Where | Purpose |
|------|-------|---------|
| `--ema-alpha` | train, deploy | EMA teacher update rate. Try 0.1-0.3 for few-shot (default 0.02 is very slow). |
| `--enrich/--no-enrich` | train, deploy | Use Nemotron Ultra 253B to enrich teacher demos. Needs `NVIDIA_API_KEY`. |
| `--wandb-project` | train, deploy | Enables W&B logging. Install with `uv pip install wandb`. |
| `--adapter-path` | run | Load a trained LoRA adapter for local inference (MLX). |
| `--epochs` | train, deploy | Training epochs (default 2). |
| `--lora-rank` | train, deploy | LoRA rank (default 16). Use 8 if OOM. |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No training samples found" | Trajectory has no positive-reward steps. Re-record with successful actions. |
| Step-0 KL near zero | Increase `--ema-alpha` (try 0.2-0.3) or check enrichment is working. |
| OOM on Mac during local training | Use `--lora-rank 8`. |
| Server 404 on adapter download | Training may have failed. Check `GET <server>/train/status`. |
| wandb not logging | `uv pip install wandb && wandb login` |
