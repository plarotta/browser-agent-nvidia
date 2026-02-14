# First Trajectory + First Adapter Training Plan

## Option A: Use Existing Data (fastest)

Existing positive samples across log dirs:

| Directory | Trajectories | Positive Samples |
| ----------- | ------------- | ----------------- |
| test_llama_run | 19 | 60 |
| paper_download_run | 6 | 16 |
| test_nim_run | 4 | 15 |
| find_flights_run | 2 | 9 |
| test_url_run | 1 | 5 |

Train immediately on existing data:

```bash
# Without NIM enrichment (no API key needed)
uv run python -m src.main train --task test_llama --trajectory-dir logs/test_llama_run --no-enrich

# With NIM enrichment
NVIDIA_API_KEY=nvapi-... uv run python -m src.main train --task test_llama --trajectory-dir logs/test_llama_run --enrich
```

**Recommendation:** Start with `test_nim_run` (15 samples, ~2 min) to validate the pipeline end-to-end before a longer run.

---

## Option B: Collect a Fresh Trajectory

### Step 1: Pick a task + goal

Something simple with 3-5 steps, e.g. a Google search.

### Step 2: Collect the trajectory

```bash
uv run python -m src.main run \
  --task my_first_task \
  --goal "Search Google for 'NVIDIA Nemotron' and click the first result" \
  --url "https://google.com" \
  --backend mlx \
  --no-headless \
  --max-steps 10
```

Trajectory saves to `logs/my_first_task_run/` automatically when the agent finishes or you Ctrl-C.

### Step 3: Verify the trajectory has positive samples

```bash
python3 -c "
import json, os
path = 'logs/my_first_task_run'
for jf in sorted(f for f in os.listdir(path) if f.endswith('.json')):
    steps = json.load(open(os.path.join(path, jf)))
    pos = sum(1 for s in steps if s.get('reward',0) > 0)
    print(f'{jf}: {len(steps)} steps, {pos} positive')
"
```

### Step 4: Train the adapter

```bash
# Enriched (recommended, ~1-2s extra per sample)
NVIDIA_API_KEY=nvapi-... uv run python -m src.main train \
  --task my_first_task \
  --enrich

# Or without enrichment
uv run python -m src.main train \
  --task my_first_task \
  --no-enrich
```

### Step 5: Check the training output

Adapter saves to `./adapters/local/adapters.safetensors` alongside `adapter_config.json` (stores rank, alpha, model_id). Logs show per-step KL loss — should decrease over the 2 epochs. A step-0 KL diagnostic warns if the initial KL is near zero (enrichment too weak or `ema_alpha` too low).

### Step 6: Test the trained adapter

The `run` command supports `--adapter-path` to load a trained LoRA adapter at inference.

#### Side-by-side comparison (base vs. adapted)

Run the same task twice — once without the adapter (baseline) and once with — and compare behavior:

```bash
# Baseline (no adapter)
uv run python -m src.main run \
  --task eval_baseline \
  --goal "Search Google for 'NVIDIA Nemotron' and click the first result" \
  --url "https://google.com" \
  --backend mlx \
  --no-headless \
  --max-steps 10

# Adapted
uv run python -m src.main run \
  --task eval_adapted \
  --goal "Search Google for 'NVIDIA Nemotron' and click the first result" \
  --url "https://google.com" \
  --backend mlx \
  --no-headless \
  --max-steps 10 \
  --adapter-path ./adapters/local
```

#### What to look for

- **Adapter loads cleanly** — no shape mismatches or missing keys
- **Output is valid action JSON** — not garbled text or prompt echoing
- **Qualitative improvement** — the adapted model should be more decisive and make fewer hallucinated actions compared to baseline

---

---

## Option C: Remote Training via `deploy` Command

If you have a RunPod GPU server running, use the `deploy` command for the full round-trip:

```bash
# 1. Record a demo
uv run python -m src.main record --url "https://google.com" --task my_search --goal "Search for hello world"

# 2. Deploy (uploads trajectory, trains on server, downloads adapter)
uv run python -m src.main deploy \
  --task my_search \
  --server-url http://<runpod-ip>:8080 \
  --adapter-name my_search \
  --ema-alpha 0.1

# 3. Run with the downloaded adapter
uv run python -m src.main run \
  --task my_search \
  --goal "Search for hello world" \
  --backend mlx \
  --adapter-path ./adapters/my_search \
  --url "https://google.com" \
  --no-headless
```

Check the server logs for the step-0 KL diagnostic — if KL < 0.01, try increasing `--ema-alpha` (e.g. 0.2-0.3).

---

## Troubleshooting

- **"No training samples found"** — Trajectory had no positive-reward steps (agent failed every action). Re-run or use a different trajectory dir.
- **NIM enrichment warnings** — `NVIDIA_API_KEY` not set or API errors. Training continues with raw demos, no harm done.
- **OOM on M4 24GB** — 12B model + LoRA + teacher/student forward passes is tight. Try `--lora-rank 8` to reduce memory.
