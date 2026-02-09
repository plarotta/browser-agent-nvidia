# Next Steps: Prioritized Plan

A concise, prioritized plan for advancing the Self-Learning Browser Agent. Items are ordered by impact and dependency; adjust to match your timeline (e.g. demo-first vs research-first).

---

## Phase 1: Robustness & Correctness (Foundation)

### 1.1 Wire confidence into SDFT gating  
**Goal:** SDFT only updates when the model was actually confident.  
**Current:** `agent_runtime.step()` calls `sdft.should_update(entropy=0.0, success=success)`; entropy is never computed from the model.  
**Action:**  
- In `agent_runtime.step()`, take `confidence` from the parsed action metadata (from `ActionParser`).  
- Map confidence to a simple entropy proxy if needed (e.g. `entropy = 1 - confidence`).  
- Pass that into `sdft.should_update(entropy, success)`.  
**Effort:** Small.

### 1.2 Task completion / stop condition  
**Goal:** Stop the run loop when the task is done, not only on `max_steps` or failure.  
**Action:**  
- Add a simple completion criterion: e.g. user-provided “success URL” or “success DOM snippet” in config/CLI.  
- In `agent_runtime.step()`, after execute, check completion; if true, set `done=True` in the logger and break the loop in `main.run()`.  
- Optionally add a `--success-url` or `--success-text` CLI flag.  
**Effort:** Small–medium.

### 1.3 Enforce SDFT update budget  
**Goal:** Cap how many times the teacher is updated so adaptation is bounded.  
**Action:**  
- In `SDFTModule`, respect `config.update_budget` (or a parameter).  
- In `should_update()` or before `update_teacher()`, check `self.update_count < update_budget` and return/skip when exceeded.  
**Effort:** Small.

---

## Phase 2: Safety & Learning Loop

### 2.1 Integrate CheckpointManager into the run loop  
**Goal:** Save checkpoints when performance is good; rollback when it degrades.  
**Action:**  
- In `AgentRuntime`, instantiate `CheckpointManager(config.checkpoint_dir)`.  
- After a successful step (or every N steps), optionally save a checkpoint (e.g. adapter only when LoRA exists).  
- Maintain a short history of success/failure (e.g. last 10 steps). If success rate drops below a threshold, call `checkpoint_manager.rollback(policy)` and optionally log a “rollback” event.  
- Document the chosen metric (e.g. “fraction of last K steps that succeeded”) in ARCHITECTURE.md.  
**Effort:** Medium.

### 2.2 Add a learnable adapter and real SDFT updates  
**Goal:** Actually update the policy’s “student” with gradient steps so SDFT has an effect.  
**Action:**  
- Introduce a small adapter (e.g. LoRA on the language model head, or an MLP head on top of pooled features). See `implementation_plan.txt` §4.2.  
- In the run path, use the adapter so that “student” = backbone + adapter.  
- When `sdft.should_update()` is True: (1) get teacher logits (no grad), (2) get student logits, (3) compute `sdft.compute_loss(student_logits, teacher_logits)`, (4) backward, (5) optimizer step with very small LR and optional gradient clip.  
- Keep backbone frozen; only adapter parameters in the optimizer.  
**Effort:** Medium–large (depends on where you get logits from the current VLM).

### 2.3 Optional: Shadow evaluation  
**Goal:** Compare adapted vs frozen policy for demos and diagnostics.  
**Action:**  
- In dev/demo mode, optionally run the frozen policy in parallel and log divergence (e.g. action agreement or KL) without executing the frozen action.  
**Effort:** Medium (optional).

---

## Phase 3: One-Shot Learning & Demo

### 3.1 Use recorded trajectories in the run  
**Goal:** Align with “one-shot learning from a single demonstration.”  
**Action:**  
- When running a task, optionally load a recorded trajectory for that task from `logs/<task>/`.  
- Use it as in-context examples (e.g. “Here is a successful run: …”) in the prompt, or as a replay reference for the policy.  
- Alternatively or in addition, implement a simple “replay mode” that replays a recorded trajectory for comparison.  
**Effort:** Medium.

### 3.2 Config from file  
**Goal:** Reproducibility and easier experimentation.  
**Action:**  
- Add `--config path/to/config.yaml` to the `run` (and optionally `record`) command.  
- If present, load `AgentConfig` from file and override with CLI flags.  
**Effort:** Small.

### 3.3 Minimal visualization / demo UX  
**Goal:** Show “current action”, “confidence”, “learning on/off”, “rollback” during demos.  
**Action:**  
- Add a simple console overlay or a minimal web/terminal dashboard that prints or displays: last action, confidence, step index, SDFT update count, and rollback events.  
- Consider using `src/visualization/` for a small module that the runtime can call each step.  
**Effort:** Medium.

---

## Phase 4: Polish & Scale

### 4.1 TensorRT and MLX defaults  
**Goal:** Clear, documented defaults for TensorRT (engine path, build) and MLX (default model when backend is MLX).  
**Action:**  
- Document in README and config: default `engine_dir`, how to build the engine (`scripts/build_engine.py`), and the default MLX model when `--backend mlx` is used.  
- Ensure `run_demo.sh` (or a variant) can run with each backend where applicable.  
**Effort:** Small.

### 4.2 Metric tracker and logging  
**Goal:** Track success rate, retries, time-to-complete for evaluation and tuning.  
**Action:**  
- Add a small `MetricTracker` (or extend the logger) that aggregates per-run and optionally across runs: success count, failure count, steps to completion, SDFT updates.  
- Optionally write a one-line summary to the trajectory file or a separate `metrics.json`.  
**Effort:** Small–medium.

### 4.3 Tests and CI  
**Goal:** Keep refactors safe.  
**Action:**  
- Add or extend tests for: action parser (more JSON shapes), SDFT gating and update budget, checkpoint save/load/rollback when integrated.  
- Run tests in CI (e.g. GitHub Actions) on key branches.  
**Effort:** Medium.

---

## Suggested order (if starting now)

1. **1.1** – Confidence → SDFT gating (quick win, correct behavior).  
2. **1.2** – Task completion (better UX and evaluation).  
3. **1.3** – Update budget (safety).  
4. **2.1** – Checkpoint integration (safety story).  
5. **3.2** – Config from file (quality of life).  
6. **2.2** – Learnable adapter + real SDFT (core learning).  
7. Then **3.1**, **3.3**, **4.x** as needed for demo and scale.

---

## References

- **Architecture and pending todos:** `docs/ARCHITECTURE.md`  
- **Design and modules:** `description.txt`, `implementation_plan.txt`  
- **Usage:** `README.md`
