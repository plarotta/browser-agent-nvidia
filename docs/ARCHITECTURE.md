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
│ Text      │   │ MLX)         │   │             │   │ (rollback)       │
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

---

## 3. Data Flow (Single Step)

1. **Observe:** `DOMSnapshotter.capture()` extracts interactive elements with stable `data-agent-id`s and visible page text (via `document.body.innerText`). `VisualCapture` takes a screenshot. Both feed into the prompt.

2. **Build Prompt:** `AgentRuntime._build_prompt()` fills the system template. The prompt is structured for truncation safety:
   - **Top:** Goal + JSON output format + action descriptions + rules (always visible)
   - **Middle:** Error feedback (if last action failed) + action history
   - **Bottom:** DOM summary with interactive elements + page text (may be truncated)

   For MLX backend, DOM is capped at 2000 chars; total prompt at 3000 chars.

3. **Predict:** `MultimodalPolicy.forward(screenshot, prompt)` dispatches to the active backend (Transformers/TensorRT/MLX) and returns raw model text.

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
|-----------|----------|--------|--------|
| **CLI** | `src/main.py` | Done | `record`, `run`, `status`; backends: transformers, tensorrt, mlx; `--model-id` override |
| **Agent runtime** | `src/agent/agent_runtime.py` | Done | Step loop, truncation-safe prompt, guards, FINISH-based termination |
| **Action parser** | `src/agent/action_parser.py` | Done | JSON extraction, line-anchored fallback regex, done-language detection |
| **Session manager** | `src/browser_interaction/session_manager.py` | Done | Playwright launch with stealth settings, navigate, close |
| **Action executor** | `src/browser_interaction/action_executor.py` | Done | click, type (rejects empty), press_enter (waits for load), scroll, wait, navigate, finish |
| **DOM snapshotter** | `src/observation/dom_snapshotter.py` | Done | Interactive elements with stable IDs + visible page text via `innerText` |
| **Visual capture** | `src/observation/visual_capture.py` | Done | Screenshot for policy |
| **Multimodal policy** | `src/policy/multimodal_policy.py` | Done | Dispatches to Transformers / TensorRT / MLX |
| **Transformers policy** | `src/policy/transformers_policy.py` | Done | Hugging Face VLM inference |
| **TensorRT policy** | `src/policy/tensorrt_policy.py` | Done | TRT engine inference |
| **MLX policy** | `src/policy/mlx_policy.py` | Done | Apple Silicon VLM; 3000-char prompt limit; BOS-strip fix for mllama |
| **Adapter layer** | `src/policy/adapter_layer.py` | Present | Not yet wired as learnable head in run path |
| **SDFT module** | `src/sdft/sdft_module.py` | Done | EMA teacher, KL loss, confidence/success gating; no actual gradient step yet |
| **Checkpoint manager** | `src/safety/checkpoint_manager.py` | Done | Save/load/rollback; not yet invoked from runtime |
| **Evaluation** | `src/evaluation/` | Unused | GoalSpec, GoalInterpreter, SuccessChecker exist but are not used in runtime; available for offline eval |
| **Config** | `src/utils/config.py` | Done | Model, backend, device, browser, learning, paths |
| **Trajectory logger** | `src/utils/trajectory_logger.py` | Done | Per-step log, save trajectory to disk |

---

## 5. Key Design Decisions

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

## 6. Technology Choices

- **Playwright:** Reliable automation, clear actions, good for logging and replay. Stealth settings prevent CAPTCHA triggers.
- **Multimodal (DOM + page text + screenshot):** DOM gives element IDs; page text gives readable content; screenshot gives layout and visual affordances.
- **Backends:** Transformers for flexibility; TensorRT for throughput on NVIDIA; MLX for Apple Silicon.
- **Frozen backbone + adapter:** Keeps behavior stable and limits memory; only a small surface (future LoRA/head) should adapt.
- **EMA teacher + gating:** Enables deployment-time self-distillation with a safety story (success gating, and future rollback).

---

## 7. References

- **Run instructions & options:** `README.md`
- **Roadmap:** `docs/NEXT_STEPS.md`
