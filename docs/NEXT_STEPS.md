# Next Steps: Prioritized Plan

A concise, prioritized plan for advancing the Self-Learning Browser Agent. Items are ordered by impact and dependency.

---

## Completed (recent fixes)

- **Truncation-safe prompt:** Output format instructions moved to top of prompt; DOM/history at bottom. Model always sees JSON format even when DOM is long.
- **FINISH-based termination:** Replaced GoalSpec/SuccessChecker with simple `done = (action_type == "finish")`. Removed dependency on 12B model producing GOAL_SPEC JSON.
- **Fallback parser fix:** Tightened regex to line-start-anchored (`^`) to prevent phantom action matches from echoed prompt text.
- **Page text extraction:** Added `document.body.innerText` to DOM snapshots so the model can read static page content (search results, weather, prices).
- **Programmatic guards:** Runtime overrides premature FINISH to PRESS_ENTER when search hasn't been submitted yet.
- **TYPE validation:** Action executor rejects TYPE with empty value to prevent phantom actions.
- **PRESS_ENTER wait:** Added `wait_for_load_state` + 1.5s settle time after Enter so the next step sees loaded results.
- **MLX prompt budget:** Raised from 1500 to 3000 chars; DOM budget from 700 to 2000 chars.
- **Remote training server:** Training-only control plane on GPU box (FastAPI). Mac uploads trajectories, triggers SDFT training, downloads trained adapters. No inference on the server.
- **SDFT training server-side:** PEFT LoRA training on uploaded trajectories via `/train` endpoint.
- **Trajectory upload:** Client packages log_dir as tar.gz, uploads to server. Server stores and uses for training.
- **NIM backend:** NVIDIA cloud API inference via NIMPolicy.
- **MLX SDFT trainer:** Full SDFT training loop on Apple Silicon (`sdft_trainer_mlx.py`): on-policy rollout, teacher/student KL loss, EMA teacher update, LoRA adapter save via `train` CLI command.
- **NIM-enriched teacher demonstrations:** Optional `--enrich` flag calls NIM VLM API to generate rich ICL demonstrations (page context, element rationale, expected outcome) for the teacher. Graceful fallback to raw action JSON when API is unavailable.
- **True on-policy SDFT on server:** Rewrote `trainer_worker.py` from SFT+KL hybrid to true on-policy SDFT (on-policy rollout, reverse KL only, configurable EMA alpha, NIM enrichment integration).
- **Shared enrichment module with caching:** Factored `enrich_demonstration()` and `build_teacher_prompt()` into `src/sdft/enrichment.py`. SHA-256 caching to `.enrichment_cache/` avoids redundant NIM API calls. Both MLX and server trainers use the shared module.
- **Enrichment model upgraded:** Default NIM enrichment model changed from Nemotron-Nano-12B to `meta/llama-3.2-90b-vision-instruct` for higher-quality reasoning.
- **Adapter download endpoint:** `GET /adapters/{name}/download` returns trained adapter as tar.gz. `TrajectoryUploader.download_adapter()` on the client side.
- **MLX adapter loading at inference:** `--adapter-path` flag on `run` command loads LoRA adapter into MLXPolicy. Reads `adapter_config.json` for rank/alpha.
- **Deploy command:** New `deploy` CLI command orchestrates full remote round-trip: upload trajectory → trigger training → poll status → download adapter → print run command.
- **Step-0 KL diagnostics:** Both MLX and server trainers log a prominent diagnostic at step 0 — warns if KL < 0.01 (enrichment too weak or ema_alpha too low).
- **Adapter config saved during training:** Both trainers save `adapter_config.json` alongside adapter weights for reliable loading at inference.
- **TrainRequest schema extended:** Added `ema_alpha` and `enrich` fields, passed through from API to trainer.

---

## Phase 1: Model Quality & Robustness

### 1.1 Improve answer extraction accuracy
**Goal:** FINISH value should contain accurate data from the page, not hallucinated values.
**Current:** The model sometimes ignores PAGE TEXT and hallucinates plausible-sounding values from the screenshot (12B quantized models can't reliably read text from images).
**Action:**
- Experiment with stronger models (e.g. larger quantized models that fit in 24GB).
- Consider adding a post-FINISH verification step that checks the answer against PAGE TEXT.
- Test whether putting PAGE TEXT closer to the JSON format instruction (higher in prompt) improves extraction.
**Effort:** Medium.

### 1.2 Wire confidence into SDFT gating
**Goal:** SDFT only updates when the model was actually confident.
**Current:** `agent_runtime.step()` calls `sdft.should_update(entropy=0.0, success=success)`; entropy is never computed from the model. The simplified prompt no longer asks for confidence.
**Action:**
- Re-add confidence to the JSON output format, or derive it from model token probabilities.
- Map to entropy proxy and pass into `sdft.should_update(entropy, success)`.
**Effort:** Small.

### 1.3 Enforce SDFT update budget
**Goal:** Cap how many times the teacher is updated so adaptation is bounded.
**Action:**
- In `SDFTModule`, respect `config.update_budget`.
- In `should_update()`, check `self.update_count < update_budget`.
**Effort:** Small.

### 1.4 More programmatic guards
**Goal:** Handle more common small-model failure modes.
**Action:**
- Detect repeated identical actions (loop detection) and force a different action.
- Detect when the model outputs the same WAIT action 3+ times and force FINISH or a different strategy.
- Guard against CLICK on non-existent element IDs (validate against current DOM).
**Effort:** Small-medium.

---

## Phase 2: Safety & Learning Loop

### 2.1 Integrate CheckpointManager into the run loop
**Goal:** Save checkpoints when performance is good; rollback when it degrades.
**Action:**
- In `AgentRuntime`, instantiate `CheckpointManager(config.checkpoint_dir)`.
- Maintain a short history of success/failure. If success rate drops, call `checkpoint_manager.rollback(policy)`.
**Effort:** Medium.

### 2.2 ~~Add a learnable adapter and real SDFT updates~~ (Done — E2E)
**Status:** Fully implemented. Server-side trainer now uses true on-policy SDFT (reverse KL, no SFT term). `deploy` CLI command orchestrates the full round-trip (upload → train → download). `run --adapter-path` loads the adapter at inference on MLX.
**Remaining:** None — E2E pipeline complete.

---

## Phase 3: One-Shot Learning & Demo

### 3.1 Use recorded trajectories in the run
**Goal:** Align with "one-shot learning from a single demonstration."
**Action:**
- When running a task, optionally load a recorded trajectory as in-context examples in the prompt.
- Alternatively implement a simple "replay mode" for comparison.
**Effort:** Medium.

### 3.2 Config from file
**Goal:** Reproducibility and easier experimentation.
**Action:**
- Add `--config path/to/config.yaml` to the CLI.
- If present, load `AgentConfig` from file and override with CLI flags.
**Effort:** Small.

### 3.3 Minimal visualization / demo UX
**Goal:** Show current action, step index, SDFT update count during demos.
**Action:**
- Add a simple console overlay or terminal dashboard.
**Effort:** Medium.

---

## Phase 4: Polish & Scale

### 4.1 Offline evaluation harness
**Goal:** Measure agent quality across a set of tasks.
**Action:**
- The `src/evaluation/` module (GoalSpec, SuccessChecker) is available for offline evaluation.
- Build a task suite with expected outcomes and score trajectories post-hoc.
**Effort:** Medium.

### 4.2 Metric tracker and logging
**Goal:** Track success rate, retries, time-to-complete for evaluation and tuning.
**Action:**
- Add a `MetricTracker` that aggregates per-run stats.
- Optionally write a summary to the trajectory file or a separate `metrics.json`.
**Effort:** Small-medium.

### 4.3 Tests and CI
**Goal:** Keep refactors safe.
**Action:**
- Add tests for: action parser (JSON + fallback + done-language), guards, page text extraction.
- Run tests in CI on key branches.
**Effort:** Medium.

---

## Suggested order

1. **1.1** -- Answer extraction accuracy (core quality).
2. **1.4** -- More guards (robustness for small models).
3. **1.2** -- Confidence into SDFT gating (correct learning).
4. **1.3** -- Update budget (safety).
5. **2.1** -- Checkpoint integration (safety story).
6. **3.2** -- Config from file (quality of life).
7. ~~**2.2** -- Learnable adapter + real SDFT~~ (Done — E2E pipeline working: deploy command, adapter download, MLX adapter loading).
8. Then **3.1**, **3.3**, **4.x** as needed for demo and scale.

---

## References

- **Architecture:** `docs/ARCHITECTURE.md`
- **Usage:** `README.md`
