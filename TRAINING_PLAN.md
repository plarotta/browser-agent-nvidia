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

Adapter saves to `./adapters/local/adapters.safetensors`. Logs show per-step KL loss — should decrease over the 2 epochs.

### Step 6: Test the trained adapter

The `run` command doesn't currently support `--adapter-path`, but `mlx_vlm.load()` accepts one natively. Two ways to test:

#### Quick smoke test (Python one-liner)

Verify the adapter loads without errors and produces output:

```bash
uv run python -c "
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_config
from PIL import Image

model_id = 'mlx-community/gemma-3-12b-it-qat-4bit'
adapter_path = './adapters/local'

model, processor = load(model_id, adapter_path=adapter_path)
config = load_config(model_id)

image = Image.open('logs/test_nim_run/traj_1770685362_step_0.png').convert('RGB')
prompt = 'What action should I take on this webpage?'
formatted = apply_chat_template(processor, config, prompt, num_images=1)
output = generate(model, processor, formatted, image=[image], max_tokens=128, verbose=False)
print(output)
"
```

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

# Adapted (requires adding --adapter-path to the run command, see note below)
uv run python -m src.main run \
  --task eval_adapted \
  --goal "Search Google for 'NVIDIA Nemotron' and click the first result" \
  --url "https://google.com" \
  --backend mlx \
  --no-headless \
  --max-steps 10 \
  --adapter-path ./adapters/local
```

> **Note:** The `run` command doesn't have `--adapter-path` yet. To enable this,
> add the flag to `src/main.py` and pass it through to `MLXPolicy`, which would
> call `load(model_id, adapter_path=adapter_path)` instead of `load(model_id)`.
> This is a small follow-up change (~10 lines across `main.py` and `mlx_policy.py`).

#### What to look for

- **Adapter loads cleanly** — no shape mismatches or missing keys
- **Output is valid action JSON** — not garbled text or prompt echoing
- **Qualitative improvement** — the adapted model should be more decisive and make fewer hallucinated actions compared to baseline

---

## Troubleshooting

- **"No training samples found"** — Trajectory had no positive-reward steps (agent failed every action). Re-run or use a different trajectory dir.
- **NIM enrichment warnings** — `NVIDIA_API_KEY` not set or API errors. Training continues with raw demos, no harm done.
- **OOM on M4 24GB** — 12B model + LoRA + teacher/student forward passes is tight. Try `--lora-rank 8` to reduce memory.
