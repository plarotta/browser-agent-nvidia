# System Architecture

High-level architecture of the Self-Learning Browser Agent and its pending work. For full design rationale, see the repo root: `description.txt` and `implementation_plan.txt`.

---

## 1. Overview

The system is a **browser automation agent** that:

1. **Observes** the page (DOM + screenshot) and a natural-language goal.
2. **Predicts** the next browser action via a multimodal policy (vision + language).
3. **Executes** the action via Playwright.
4. **Optionally learns** at deployment time using Self-Distillation Fine-Tuning (SDFT) from successful steps.

**Design principle:** *Frozen general intelligence + bounded personalization at the edges.*

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI (src/main.py)                              │
│  record | run --task --goal [--backend] [--train] [--url] [--max-steps]  │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
          User goal → GoalInterpreter → GoalSpec (success_signals, required_outputs, expected_artifact)
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent Runtime (agent_runtime.py)                     │
│  Observe → Policy → Action → Log → SuccessChecker (done?) → SDFT (opt.)   │
└─────────────────────────────────────────────────────────────────────────┘
     │                │                │                │
     ▼                ▼                ▼                ▼
┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────────┐
│Observation│   │ Policy       │   │ Action      │   │ Learning / Safety│
│           │   │              │   │ Execution   │   │                  │
│ DOM       │   │ Multimodal   │   │ Playwright  │   │ SDFT (EMA teacher│
│ Snapshot  │   │ (Transformers│   │ ActionExec  │   │ + gating)        │
│ + Visual  │   │ TensorRT    │   │             │   │ CheckpointManager│
│ Capture   │   │ MLX)         │   │             │   │ (rollback)       │
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

1. **Observe:** `DOMSnapshotter` captures interactive elements with stable `data-agent-id`s; `VisualCapture` captures a screenshot. Both are used to build a text prompt (DOM summary + “screenshot as image”).
2. **Prompt:** `AgentRuntime._build_prompt()` fills a system template with goal, step number, action history, DOM summary, and optional last-error section.
3. **Predict:** `MultimodalPolicy.forward(screenshot, prompt)` returns raw model text. `ActionParser` extracts a single JSON action (type, target_element_id, value, confidence).
4. **Execute:** `ActionExecutor.execute(type, params)` resolves element by `element_id` (or selector), then runs click/type/scroll/wait/navigate/etc. Returns success/failure and error reason.
5. **Log:** `TrajectoryLogger.log_step()` writes DOM, screenshot, action, reward, done to the current trajectory; `save_trajectory()` persists to `log_dir`.
6. **Learn (if `--train`):** `SDFTModule.should_update(entropy, success)` gates updates; when allowed, `update_teacher()` runs EMA from student to teacher. CheckpointManager is available for save/rollback but not yet wired into the main loop.

---

## 4. Component Summary

| Component | Location | Status | Notes |
|-----------|----------|--------|--------|
| **CLI** | `src/main.py` | ✅ Implemented | `record`, `run`, `status`; backends: transformers, tensorrt, mlx |
| **Agent runtime** | `src/agent/agent_runtime.py` | ✅ Implemented | Step loop, prompt build, policy call, execute, log, SDFT gating |
| **Action parser** | `src/agent/action_parser.py` | ✅ Implemented | JSON extraction, normalizes to executor format (element_id, value) |
| **Session manager** | `src/browser_interaction/session_manager.py` | ✅ Implemented | Playwright launch, navigate, close |
| **Action executor** | `src/browser_interaction/action_executor.py` | ✅ Implemented | click, type, press_enter, select, scroll, wait, navigate |
| **DOM snapshotter** | `src/observation/dom_snapshotter.py` | ✅ Implemented | Interactive elements, stable IDs, formatted summary |
| **Visual capture** | `src/observation/visual_capture.py` | ✅ Implemented | Screenshot for policy |
| **Multimodal policy** | `src/policy/multimodal_policy.py` | ✅ Implemented | Dispatches to Transformers / TensorRT / MLX |
| **Transformers policy** | `src/policy/transformers_policy.py` | ✅ Implemented | Hugging Face VLM inference |
| **TensorRT policy** | `src/policy/tensorrt_policy.py` | ✅ Implemented | TRT engine inference |
| **MLX policy** | `src/policy/mlx_policy.py` | ✅ Implemented | Apple Silicon VLM |
| **Adapter layer** | `src/policy/adapter_layer.py` | ⚠️ Present | Not yet wired as learnable head in run path |
| **SDFT module** | `src/sdft/sdft_module.py` | ✅ Implemented | EMA teacher, KL loss, confidence/success gating; no actual gradient step on student yet |
| **Checkpoint manager** | `src/safety/checkpoint_manager.py` | ✅ Implemented | Save/load/rollback; not yet invoked from runtime on metrics |
| **Goal spec** | `src/evaluation/goal_spec.py` | ✅ Implemented | Structured success criteria (goal_type, success_signals, required_outputs, expected_artifact) |
| **Goal interpreter** | `src/evaluation/goal_interpreter.py` | ✅ Implemented | Rule-based: user goal → GoalSpec (no LLM) |
| **Success checker** | `src/evaluation/success_checker.py` | ✅ Implemented | check_output_present, check_page_signal, check_agent_finish; done = any two of three |
| **Config** | `src/utils/config.py` | ✅ Implemented | Model, backend, device, browser, learning, paths, goal_spec |
| **Trajectory logger** | `src/utils/trajectory_logger.py` | ✅ Implemented | Per-step log, save trajectory to disk |
| **Visualization** | `src/visualization/` | ❌ Placeholder | No dashboard or overlay yet |

---

## 5. Pending Todos (High Level)

- **Policy adaptation in the loop:** The policy backbone is used as-is; there is no LoRA/adapter training step. SDFT updates the EMA teacher from the student, but the “student” is not actually updated by gradient steps. *Todo: Wire a small learnable adapter (e.g. LoRA) and perform bounded online updates using SDFT loss when gating allows.*

- **Checkpoint integration:** `CheckpointManager` exists but is not used in the main run loop. There is no “save on success / rollback on degradation” logic. *Todo: Integrate checkpoint save when metrics are good and rollback when success rate or a chosen metric degrades.*

- **Entropy/confidence from model:** SDFT gating uses a placeholder entropy (e.g. 0.0). The model outputs a `confidence` field in JSON; it is not yet passed into `sdft.should_update()`. *Todo: Pass parsed confidence (or derived entropy) from the policy output into SDFT gating.*

- **Task completion / stopping:** The run loop stops only on `max_steps` or failure; there is no “task done” signal (e.g. success page or user-defined criterion). Done: GoalInterpreter + SuccessChecker; stop when any two of (output present, page signal, validated finish).

- **Replay / one-shot demo usage:** Recorded trajectories are logged but not used as few-shot examples or for direct replay in the current run path. *Todo: Use recorded demos (e.g. as in-context examples or for behavioral cloning) as per the original one-shot learning vision.*

- **Update budget:** Config has `update_budget`; SDFT does not enforce a cap on the number of updates. *Todo: Enforce update budget in SDFT and optionally in checkpoint/rollback policy.*

- **Visualization / demo UX:** No live dashboard, overlay, or simple graphs for “current action”, “confidence”, “learning on/off”, “rollback”. *Todo: Add a minimal live view or console overlay for demos.*

- **Config from file:** `AgentConfig.load(path)` exists but the CLI does not support loading config from a YAML/JSON file. *Todo: Add a `--config` option to the CLI.*

---

## 6. Technology Choices

- **Playwright:** Reliable automation, clear actions, good for logging and replay.
- **Multimodal (DOM + screenshot):** DOM gives semantics and element IDs; screenshot gives layout and visual affordances; together they improve robustness to UI drift.
- **Backends:** Transformers for flexibility; TensorRT for throughput on NVIDIA; MLX for Apple Silicon.
- **Frozen backbone + adapter:** Keeps behavior stable and limits GPU memory; only a small surface (future LoRA/head) should adapt.
- **EMA teacher + gating:** Enables deployment-time self-distillation with a safety story (confidence + success gating, and future rollback).

---

## 7. References

- **Design & lifecycle:** `description.txt` (high-level design, learning phases, safety).
- **Module breakdown & build order:** `implementation_plan.txt` (per-module responsibilities and MVP notes).
- **Run instructions & options:** `README.md` and `docs/NEXT_STEPS.md`.
