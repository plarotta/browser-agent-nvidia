# System Architecture

High-level architecture of the Self-Learning Browser Agent.

---

## 1. Overview

The system is a **browser automation agent** that:

1. **Observes** the page (DOM elements + visible page text + screenshot) and a natural-language goal.
2. **Predicts** the next browser action via a multimodal VLM policy.
3. **Applies guards** to prevent common small-model errors (e.g. premature FINISH).
4. **Executes** the action via Playwright.
5. **Terminates** when the model outputs FINISH (with an answer) or `max_steps` is reached.
6. **Optionally learns** at deployment time using Self-Distillation Fine-Tuning (SDFT).

**Design principle:** *Frozen general intelligence + bounded personalization at the edges.*

---

## 2. Architecture Diagram

### Local Mode (all-in-one)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI (src/main.py)                              │
│  record | run --task --goal [--backend] [--train] [--url] [--max-steps]  │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent Runtime (agent_runtime.py)                     │
│  Observe → Build Prompt → Policy → Parse → Guard → Execute → Log        │
│  Done when action.type == FINISH                                         │
└─────────────────────────────────────────────────────────────────────────┘
     │                │                │                │
     ▼                ▼                ▼                ▼
┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────────┐
│Observation│   │ Policy       │   │ Action      │   │ Learning / Safety│
│           │   │              │   │ Execution   │   │                  │
│ DOM       │   │ Multimodal   │   │ Playwright  │   │ SDFT (EMA teacher│
│ Snapshot  │   │ (Transformers│   │ ActionExec  │   │ + gating)        │
│ + Page    │   │ TensorRT    │   │             │   │ CheckpointManager│
│ Text      │   │ MLX / NIM)   │   │             │   │ (rollback)       │
│ + Visual  │   │              │   │             │   │                  │
│ Capture   │   │              │   │             │   │                  │
└──────────┘   └──────────────┘   └─────────────┘   └──────────────────┘
     │                │                   │
     └────────────────┴───────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Trajectory    │
              │ Logger        │
              │ (JSON + PNG)  │
              └───────────────┘
```

### Remote Training Mode (Mac + GPU server)

```
Mac (consumer)                          RunPod GPU box (training server)
──────────────                          ────────────────────────────────
Browser + Playwright                    FastAPI training control plane
Observation pipeline                      /upload_trajectory
Agent loop (local inference via MLX)      /train → SDFT with PEFT LoRA
TrajectoryLogger                          /train/status
  └─ TrajectoryUploader ── HTTP ──→       /adapters/* (list + download)
                                        Trainer worker (SDFT on GPU)
                                        Adapter storage (/adapters/)
```

All inference runs locally on the Mac (MLX, Transformers, TensorRT, or NIM). The remote GPU server is used only for training LoRA adapters. Trajectories are uploaded via `deploy`, training runs server-side, and the trained adapter is downloaded back to the Mac for local inference.

---

## 3. Data Flow (Single Step)

1. **Observe:** `DOMSnapshotter.capture()` extracts interactive elements with stable `data-agent-id`s and visible page text (via `document.body.innerText`). `VisualCapture` takes a screenshot. Both feed into the prompt.

2. **Build Prompt:** `AgentRuntime._build_prompt()` fills the system template. The prompt is structured for truncation safety:
   - **Top:** Goal + JSON output format + action descriptions + rules (always visible)
   - **Middle:** Error feedback (if last action failed) + action history
   - **Bottom:** DOM summary with interactive elements + page text (may be truncated)

   For MLX backend, DOM is capped at 2000 chars; total prompt at 3000 chars.

3. **Predict:** `MultimodalPolicy.forward(screenshot, prompt)` dispatches to the active backend (Transformers/TensorRT/MLX/NIM) and returns raw model text.

4. **Parse:** `ActionParser.parse()` extracts the action in priority order:
   - **JSON extraction:** Brace-balanced parser finds JSON objects with an `action` key.
   - **Fallback regex:** Line-start-anchored regex (`^ACTION_TYPE`) matches non-JSON output without false-matching action words in echoed prompts or DOM hints.
   - **Done-language detection:** If the model says "goal is achieved/complete" without a formal action, it's parsed as FINISH.

5. **Guard:** Programmatic safety checks override the model when needed:
   - If FINISH is requested but no PRESS_ENTER has occurred since the last TYPE, the runtime overrides to PRESS_ENTER (prevents skipping search submission).

6. **Execute:** `ActionExecutor.execute(type, params)` runs the action via Playwright:
   - TYPE rejects empty values to prevent phantom actions.
   - PRESS_ENTER waits for `domcontentloaded` + 1.5s settle time so the next step sees loaded results.
   - FINISH is a no-op (termination is handled by the runtime).

7. **Terminate:** `done = (action_type == "finish")`. The FINISH action's `value` field contains the final answer.

8. **Log:** `TrajectoryLogger.log_step()` writes DOM, screenshot, action, reward, done per step.

9. **Learn (optional):** If `--train`, `SDFTModule.should_update()` gates EMA teacher updates based on success.

---

## 4. Component Summary

| Component | Location | Status | Notes |
| ----------- | ---------- | -------- | -------- |
| **CLI** | `src/main.py` | Done | `record`, `run`, `train`, `deploy`, `status`, `serve`; backends: transformers, tensorrt, mlx, nim; `--model-id`, `--adapter-path`, `--enrich/--no-enrich` |
| **Agent runtime** | `src/agent/agent_runtime.py` | Done | Step loop, truncation-safe prompt, guards, FINISH-based termination, optional trajectory upload |
| **Action parser** | `src/agent/action_parser.py` | Done | JSON extraction, line-anchored fallback regex, done-language detection |
| **Session manager** | `src/browser_interaction/session_manager.py` | Done | Playwright launch with stealth settings, navigate, close |
| **Action executor** | `src/browser_interaction/action_executor.py` | Done | click, type (rejects empty), press_enter (waits for load), scroll, wait, navigate, finish |
| **DOM snapshotter** | `src/observation/dom_snapshotter.py` | Done | Interactive elements with stable IDs + visible page text via `innerText` |
| **Visual capture** | `src/observation/visual_capture.py` | Done | Screenshot for policy |
| **Multimodal policy** | `src/policy/multimodal_policy.py` | Done | Dispatches to Transformers / TensorRT / MLX / NIM |
| **Transformers policy** | `src/policy/transformers_policy.py` | Done | Hugging Face VLM inference |
| **TensorRT policy** | `src/policy/tensorrt_policy.py` | Done | TRT engine inference |
| **MLX policy** | `src/policy/mlx_policy.py` | Done | Apple Silicon VLM; 3000-char prompt limit; BOS-strip fix for mllama; optional LoRA adapter loading via `--adapter-path`; auto-remaps PEFT (PyTorch) adapter keys + transposes weights for cross-platform compatibility |
| **NIM policy** | `src/policy/nim_policy.py` | Done | NVIDIA NIM cloud API inference |
| **FastAPI server** | `src/server/api.py` | Done | Training control plane: `/train`, `/upload_trajectory`, `/adapters/*` (incl. download), `/health` |
| **Trainer worker** | `src/server/trainer_worker.py` | Done | True on-policy SDFT with PEFT LoRA: 4-bit NF4 quantized loading via bitsandbytes, on-policy rollout (gradient checkpointing toggled off for generate), reverse KL (no SFT term), EMA teacher via weight swap, NIM enrichment, step-0 KL diagnostics |
| **Shared schemas** | `src/shared/schemas.py` | Done | Pydantic models for client-server communication |
| **Adapter layer** | `src/policy/adapter_layer.py` | Present | Not yet wired as learnable head in run path |
| **SDFT module** | `src/sdft/sdft_module.py` | Done | EMA teacher, KL loss, confidence/success gating; no actual gradient step yet |
| **Enrichment module** | `src/sdft/enrichment.py` | Done | Nemotron Ultra 253B enrichment (text-only, DOM + action). SHA-256 caching to `.enrichment_cache/`. Used by both MLX and server trainers |
| **MLX SDFT trainer** | `src/sdft/sdft_trainer_mlx.py` | Done | Full SDFT training loop on Apple Silicon: on-policy rollout, teacher/student KL loss, EMA update, LoRA adapter save. NIM-enriched teacher demos with caching. Step-0 KL diagnostics. Saves `adapter_config.json` |
| **Checkpoint manager** | `src/safety/checkpoint_manager.py` | Done | Save/load/rollback; not yet invoked from runtime |
| **Evaluation** | `src/evaluation/` | Unused | GoalSpec, GoalInterpreter, SuccessChecker exist but are not used in runtime; available for offline eval |
| **Config** | `src/utils/config.py` | Done | Model, backend, device, browser, learning, paths |
| **Trajectory logger** | `src/utils/trajectory_logger.py` | Done | Per-step log, save trajectory to disk |
| **Trajectory uploader** | `src/utils/trajectory_uploader.py` | Done | Packages log_dir as tar.gz, uploads to server `/upload_trajectory`. Also downloads trained adapters via `/adapters/{name}/download` |
| **Launch script** | `scripts/start_server.sh` | Done | Starts training server on RunPod |

---

## 5. Training Architecture

### Remote training flow (GPU server, on-policy SDFT with PEFT LoRA)
1. Client uploads trajectory via `TrajectoryUploader` → `/upload_trajectory` (tar.gz of JSON + PNGs).
2. Client POSTs `/train` with trajectory IDs and hyperparameters (`ema_alpha`, `enrich`, etc.).
3. `trainer_worker.run_training()` loads the base model in **4-bit NF4** (bitsandbytes) + PEFT LoRA, runs true on-policy SDFT:
   - On-policy rollout from student (`model.generate(do_sample=True, temperature=1.0)`) — gradient checkpointing temporarily disabled for generation so KV cache works with Gemma 3's vision masking
   - Optional Nemotron Ultra 253B enrichment of expert demonstrations (via `src/sdft/enrichment.py`, text-only, with caching)
   - Swap to EMA teacher weights, forward with ICL-enriched prompt on rollout tokens (extends `token_type_ids` for Gemma 3)
   - Reverse KL: `D_KL(student || teacher)` — no SFT term
   - EMA teacher update with configurable `ema_alpha`
4. Teacher state is a `dict` of cloned LoRA tensors (~10-50MB), not a full model copy.
5. Step-0 KL diagnostic: warns if KL is near zero (enrichment too weak or `ema_alpha` too low).
6. After training, adapter + `adapter_config.json` are saved.
7. Client downloads the trained adapter via `GET /adapters/{name}/download` (tar.gz).

### Deploy command (E2E round-trip)

The `deploy` CLI command orchestrates the full remote pipeline from the Mac:
1. Upload trajectory → 2. Trigger training → 3. Poll status → 4. Download adapter → 5. Print `run` command with adapter path.

### Local training flow (MLX SDFT)

For Apple Silicon users, SDFT training runs locally without a server:

1. Agent collects a trajectory during `run` (JSON + screenshots saved to `logs/{task}_run/`).
2. `train` command loads positive-reward steps from the trajectory.
3. **(Optional) Nemotron Ultra enrichment:** If `--enrich` and `NVIDIA_API_KEY` is set, each sample's DOM observation and expert action are sent to Nemotron Ultra 253B via the NIM API. The model returns a rich explanation (page context, element rationale, expected outcome) that replaces the raw action JSON as the teacher's ICL demonstration. Falls back to raw actions on failure.
4. For each sample: student rollout (no grad) -> teacher forward with ICL demo (no grad) -> student forward + reverse KL loss (grad) -> optimizer step -> EMA teacher update.
5. LoRA adapter saved to `--adapter-path` (default `./adapters/local/adapters.safetensors`) alongside `adapter_config.json`.
6. Step-0 KL diagnostic: warns if KL < 0.01 (enrichment too weak or `ema_alpha` too low).
7. The trained adapter can be loaded at inference via `run --adapter-path ./adapters/local --backend mlx`.

---

## 6. Key Design Decisions

### Truncation-safe prompt
Small VLMs (12B quantized) fail when the output format instructions are at the end of a long prompt and get truncated. The prompt puts format instructions at the top (GOAL + JSON format + rules) and variable-length content at the bottom (history + DOM). Even if DOM is truncated, the model always sees how to respond.

### Programmatic guards over prompt rules
12B models can't reliably follow multi-step sequencing rules in text (e.g. "you must PRESS_ENTER before FINISH"). The runtime enforces these constraints programmatically, which is 100% reliable regardless of model size.

### FINISH-based termination
Previously used GoalSpec + SuccessChecker (agent defines success criteria in step 0, runtime checks DOM for required outputs). This was too complex for 12B models — they couldn't reliably produce GOAL_SPEC JSON. Replaced with simple FINISH action detection: `done = (action_type == "finish")`.

### Page text extraction
DOM snapshots only capture interactive elements (buttons, links, textboxes). Static content (search results, weather data, prices) is invisible to the model. Added `document.body.innerText` extraction so the model can read and report actual page content in FINISH value.

### Line-anchored fallback parser
The previous fallback regex matched action keywords (TYPE, CLICK) anywhere in text, causing phantom action matches when the model echoed the prompt. The new regex requires keywords at the start of a line (`^` with `re.MULTILINE`), eliminating false matches from DOM hints and echoed prompt text.

---

## 7. Technology Choices

- **Playwright:** Reliable automation, clear actions, good for logging and replay. Stealth settings prevent CAPTCHA triggers.
- **Multimodal (DOM + page text + screenshot):** DOM gives element IDs; page text gives readable content; screenshot gives layout and visual affordances.
- **Backends:** Transformers for flexibility; TensorRT for throughput on NVIDIA; MLX for Apple Silicon; NIM for NVIDIA cloud.
- **Local inference, remote training:** All inference runs locally (Mac or workstation). The remote GPU server is used only for LoRA adapter training, keeping the architecture simple and the same repo runs on both sides.
- **Frozen backbone + PEFT LoRA adapter:** Keeps behavior stable and limits memory. Only LoRA parameters (~10-50MB) are trained and swapped.
- **4-bit quantized training:** Base model loaded in NF4 via bitsandbytes, with `prepare_model_for_kbit_training()` for proper gradient handling. Fits 12B models on 24-48GB GPUs.
- **Cross-platform adapter compatibility:** Adapters trained on GPU (PEFT/PyTorch) auto-remap to MLX format for Mac inference — key name transformation + weight transposing.
- **EMA teacher + gating:** Enables deployment-time self-distillation. Teacher state is cloned LoRA weights only (not full model), enabling SDFT on single-GPU setups.

---

## 8. References

- **Run instructions & options:** `README.md`
- **Roadmap:** `docs/NEXT_STEPS.md`
