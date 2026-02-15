# Research Proposal: Backbone-Agnostic Self-Distillation for Lightweight Browser Agents

## 1. Introduction & Motivation

Vision-Language Models (VLMs) have shown remarkable capability in web navigation tasks, but deploying them as practical browser agents faces three core challenges: (1) large models are too expensive for real-time interaction, (2) small models (7B–12B) fail at multi-step reasoning, and (3) domain adaptation typically requires retraining the full model. We propose a system that addresses all three by combining **frozen VLM backbones** with **Self-Distillation Fine-Tuning (SDFT)** of lightweight LoRA adapters, enabling deployment-time learning that is both backbone-agnostic and practical on consumer hardware.

Our key insight is that the failure modes of small VLMs in browser tasks are systematic and predictable — skipped steps, hallucinated observations, and malformed outputs — and can be corrected through a combination of **programmatic runtime guards** and **on-policy self-distillation** from an EMA teacher. This produces agents that improve with use while remaining safe and portable across inference backends.

---

## 2. Research Questions

**RQ1: Can SDFT with LoRA adapters improve small VLM browser agents without degrading generalization?**
We hypothesize that training only LoRA parameters (1–5% of model weights) via KL-divergence distillation preserves the frozen backbone's general capabilities while improving task-specific browser navigation.

**RQ2: How do programmatic guards compare to prompt-based rules for small model reliability?**
We hypothesize that runtime-enforced guards (e.g., blocking premature FINISH) achieve near-100% compliance regardless of model size, while prompt-based instructions degrade sharply below ~30B parameters.

**RQ3: Is multimodal observation (DOM + page text + screenshot) necessary, or can any single modality suffice?**
We hypothesize that DOM structure alone is insufficient (missing static content), screenshots alone cause hallucination in small models, and the combination is strictly better than any single source.

**RQ4: Does NIM-enriched teacher distillation outperform raw trajectory distillation?**
We hypothesize that using a stronger API-hosted model to generate rich explanations of expert actions ("why this element, what to expect") produces higher-quality teacher logits than simply replaying the action JSON.

**RQ5: Is the approach truly backbone-agnostic?**
We test whether the same training pipeline and adapter architecture produce consistent improvements across Nemotron-12B, Gemma-3-12B, and Llama-3.2-11B backbones.

---

## 3. System Design

### 3.1 Architecture Overview

The system operates in two configurations:

- **Local mode**: Browser + agent + VLM inference + SDFT training all on a single machine (Apple Silicon with MLX, or NVIDIA GPU with Transformers/TensorRT).
- **Client-server mode**: Browser agent on consumer hardware, VLM inference and training on a remote GPU via a FastAPI control plane with dynamic LoRA hot-swapping.

### 3.2 Observation Pipeline

Each step produces a **trimodal observation**:

| Modality | Source | Purpose |
|----------|--------|---------|
| DOM summary | Playwright JS injection | Interactive elements with stable IDs, semantic `kind` and `hint` labels |
| Page text | `document.body.innerText` | Visible static content (search results, prices, text) |
| Screenshot | Playwright PNG capture | Visual layout and affordances |

### 3.3 Truncation-Safe Prompt Design

The prompt is structured with output format instructions at the **top** and variable-length DOM content at the **bottom**, ensuring the model always sees how to respond even when context is truncated:

```
GOAL → JSON FORMAT → ACTION DESCRIPTIONS → RULES → ERROR FEEDBACK → HISTORY → DOM + PAGE TEXT
```

### 3.4 Programmatic Guard System

Runtime guards enforce behavioral constraints that small models cannot reliably learn from instructions alone:

| Guard | Trigger | Override |
|-------|---------|----------|
| Submit guard | FINISH attempted with no PRESS_ENTER after TYPE | Force PRESS_ENTER |
| Recovery guard | CLICK failed with pending TYPE | Force PRESS_ENTER |
| Loop guard | Same action repeated 3+ times | Force FINISH |
| Empty-input guard | TYPE with empty value | Reject action |

### 3.5 SDFT Training Pipeline

**Algorithm**: On-policy rollout → EMA teacher forward → student KL loss → adapter update

1. **Trajectory collection**: Agent runs produce step-level logs (DOM, screenshot, action, reward).
2. **Data filtering**: Only steps with reward > 0 (successful actions) are used for training.
3. **Optional NIM enrichment**: A stronger cloud-hosted VLM generates rich explanations of each expert action, used as in-context learning demonstrations for the teacher.
4. **On-policy rollout**: Student generates tokens with stochastic sampling (temperature=1.0).
5. **Teacher forward**: EMA-weighted LoRA parameters are swapped into the same backbone (no model duplication). Teacher receives the enriched ICL demonstration.
6. **KL loss**: Reverse KL divergence `KL(student || teacher)` is backpropagated through student LoRA parameters only.
7. **EMA update**: Teacher LoRA weights are updated: `θ_teacher = 0.98 · θ_teacher + 0.02 · θ_student`.

**Memory efficiency**: The teacher is stored as cloned LoRA weights (~10–50MB), not a full model copy. Weight swapping happens in-place, enabling SDFT on 24GB unified memory.

---

## 4. Experimental Plan

### 4.1 Benchmark Tasks

We evaluate on a suite of web navigation tasks spanning three difficulty tiers:

| Tier | Tasks | Example |
|------|-------|---------|
| **Simple** (1–3 steps) | Google search, Wikipedia lookup, weather check | "Search Google for 'NVIDIA stock price' and report the value" |
| **Medium** (4–8 steps) | Form filling, multi-page navigation, comparison shopping | "Find the cheapest flight from SFO to JFK on Google Flights" |
| **Hard** (8+ steps) | Multi-site workflows, login + action sequences | "Log into GitHub, create a new repo, and push a README" |

Each task is evaluated over **N=20 runs** to account for stochastic model behavior.

### 4.2 Experiment 1: SDFT Effectiveness (RQ1)

**Setup**: Compare base model vs. SDFT-adapted model on all benchmark tasks.

| Condition | Description |
|-----------|-------------|
| **Base** | Frozen Nemotron-12B with no adapter |
| **SDFT-5** | SDFT with 5 successful trajectory samples |
| **SDFT-20** | SDFT with 20 successful trajectory samples |
| **SDFT-50** | SDFT with 50 successful trajectory samples |
| **SFT baseline** | Standard supervised fine-tuning on same trajectories |

**Metrics**: Task success rate, steps-to-completion, action accuracy per step.

**Generalization test**: After SDFT on Google Search tasks, evaluate on unseen Bing Search and DuckDuckGo tasks (zero-shot transfer).

### 4.3 Experiment 2: Guard Ablation (RQ2)

**Setup**: Run the full agent with guards enabled vs. disabled, across model sizes.

| Condition | Guards | Model |
|-----------|--------|-------|
| Guards ON | All four guards active | Nemotron-12B |
| Guards OFF | No runtime guards | Nemotron-12B |
| Guards ON | All four guards active | Gemma-3-12B-4bit |
| Guards OFF | No runtime guards | Gemma-3-12B-4bit |

**Metrics**: Task success rate, premature FINISH rate, loop rate, empty-action rate.

### 4.4 Experiment 3: Observation Modality Ablation (RQ3)

**Setup**: Run agent with different observation combinations.

| Condition | DOM | Page Text | Screenshot |
|-----------|-----|-----------|------------|
| Full (ours) | ✓ | ✓ | ✓ |
| DOM + Screenshot | ✓ | ✗ | ✓ |
| DOM only | ✓ | ✗ | ✗ |
| Screenshot only | ✗ | ✗ | ✓ |
| DOM + Page Text | ✓ | ✓ | ✗ |

**Metrics**: Task success rate, answer accuracy (for extraction tasks), hallucination rate (model output vs. ground truth page content).

### 4.5 Experiment 4: Teacher Enrichment (RQ4)

**Setup**: Compare SDFT with and without NIM enrichment.

| Condition | Teacher Signal |
|-----------|----------------|
| **Raw** | Action JSON only (`{"type": "CLICK", "target": "e3"}`) |
| **Enriched** | NIM-generated explanation + action JSON |

**Metrics**: Training loss convergence, task success rate post-training, sample efficiency (samples needed to reach 80% success).

### 4.6 Experiment 5: Backbone Agnosticism (RQ5)

**Setup**: Run the identical SDFT pipeline across three VLM backbones.

| Backbone | Parameters | Quantization | Hardware |
|----------|-----------|--------------|----------|
| NVIDIA Nemotron-Nano-12B-v2-VL | 12B | BF16 | RTX 4090 |
| Gemma-3-12B-IT | 12B | 4-bit (QAT) | M4 Mac (MLX) |
| Llama-3.2-11B-Vision | 11B | 4-bit | M4 Mac (MLX) |

**Metrics**: Pre-SDFT vs. post-SDFT success rates per backbone, adapter size, training time.

---

## 5. Proposed Figures

### Figure 1: System Architecture Diagram
A block diagram showing the two deployment modes (local and client-server), with the observation pipeline, agent runtime, policy dispatch, and SDFT training loop. Arrows show data flow: page → trimodal observation → prompt construction → VLM inference → action parsing → guards → execution → trajectory logging → SDFT training → adapter update.

### Figure 2: SDFT Learning Curves
**Line plot** — X-axis: number of training trajectories. Y-axis: task success rate (%). Lines for SDFT-adapted vs. SFT baseline vs. base model. Expected: SDFT reaches high performance with fewer samples than SFT; base model is a flat line. Include error bars from N=20 runs per condition.

### Figure 3: Guard Ablation Bar Chart
**Grouped bar chart** — X-axis: guard condition (All ON, Submit OFF, Loop OFF, All OFF). Y-axis: task success rate. Grouped by model (Nemotron-12B, Gemma-3-12B). Expected: sharp drops when submit guard is removed; loop guard matters less.

### Figure 4: Observation Modality Ablation
**Stacked bar chart or heatmap** — Rows: observation conditions (Full, DOM+Screenshot, DOM-only, Screenshot-only, DOM+Text). Columns: task types (search, form-fill, extraction). Cell values: success rate. Expected: Full > DOM+Text > DOM+Screenshot > DOM-only >> Screenshot-only.

### Figure 5: Hallucination Rate vs. Observation Modality
**Bar chart** — X-axis: observation condition. Y-axis: hallucination rate (% of answers that don't match actual page content). Expected: screenshot-only has the highest hallucination; adding page text drops it dramatically.

### Figure 6: Backbone Comparison (Pre/Post SDFT)
**Paired bar chart** — X-axis: backbone (Nemotron, Gemma, Llama). Two bars per backbone: pre-SDFT (gray) and post-SDFT (colored). Y-axis: task success rate. Expected: all three backbones improve, demonstrating backbone agnosticism.

### Figure 7: NIM Enrichment Effect on Training Loss
**Line plot** — X-axis: training steps. Y-axis: KL divergence loss. Two lines: enriched teacher vs. raw teacher. Expected: enriched teacher produces lower loss and faster convergence.

### Figure 8: Sample Efficiency Comparison
**Line plot** — X-axis: number of training samples (5, 10, 20, 50). Y-axis: success rate. Lines for: SDFT-enriched, SDFT-raw, SFT. Expected: SDFT-enriched reaches target performance with fewest samples.

### Figure 9: Prompt Truncation Robustness
**Line plot** — X-axis: prompt budget (characters). Y-axis: action parse success rate. Two lines: format-at-top (ours) vs. format-at-bottom (baseline). Expected: format-at-bottom degrades sharply below 2000 chars; format-at-top remains stable.

### Figure 10: Qualitative Trajectory Comparison
**Side-by-side screenshot grid** — Two columns: base model trajectory vs. SDFT-adapted trajectory, for the same task. Each row is a step (screenshot + action taken). Annotate failure points in the base model (wrong element, premature finish) and successful recovery in the adapted model.

### Figure 11: Adapter Portability
**Table/figure** — Show adapter file sizes, training times, and success rates across hardware configurations (M4 Mac, RTX 4090, RunPod). Demonstrate that adapters trained on one backend can be transferred and used on another.

### Figure 12: EMA Teacher Weight Evolution
**Line plot** — X-axis: training steps. Y-axis: L2 distance between student and teacher LoRA weights. Expected: teacher weights converge smoothly; no catastrophic divergence. Include loss on secondary axis.

---

## 6. Expected Contributions

1. **First demonstration of SDFT for VLM-based browser agents** — showing that self-distillation with LoRA adapters improves small model performance on web navigation without degrading generalization.

2. **Backbone-agnostic training pipeline** — the same SDFT procedure works across NVIDIA Nemotron, Google Gemma, and Meta Llama backbones with consistent improvements.

3. **Trimodal observation design for small VLMs** — empirical evidence that DOM + page text + screenshot is necessary for reliable operation below 30B parameters.

4. **Programmatic guard framework** — a systematic approach to runtime safety enforcement that compensates for small model reasoning failures, achieving near-perfect compliance on behavioral constraints.

5. **Consumer-hardware SDFT** — training pipeline that fits in 24GB unified memory by storing only LoRA teacher weights (not full model copies), making deployment-time learning accessible without datacenter GPUs.

6. **NIM-enriched teacher distillation** — a novel approach where a stronger cloud-hosted model generates rich teaching signals for local self-distillation, improving sample efficiency.

---

## 7. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1: Benchmark Setup** | 2 weeks | Define task suite, build evaluation harness, establish baselines |
| **Phase 2: Core Experiments** | 4 weeks | Experiments 1–3 (SDFT effectiveness, guard ablation, modality ablation) |
| **Phase 3: Advanced Experiments** | 3 weeks | Experiments 4–5 (teacher enrichment, backbone agnosticism) |
| **Phase 4: Analysis & Writing** | 3 weeks | Statistical analysis, figure generation, paper draft |

---

## 8. Related Work

- **WebArena / Mind2Web** — benchmark suites for browser agents, but focus on large models (GPT-4V, Gemini). We target small open-weight VLMs.
- **SeeAct / WebVoyager** — VLM-based browser agents using screenshots. We add DOM and page text to reduce hallucination.
- **LoRA / QLoRA** — parameter-efficient fine-tuning. We apply LoRA specifically for deployment-time self-distillation, not offline training.
- **Self-Play / Self-Distillation** — SDFT (Cheng et al.) for language models. We extend to multimodal browser agents with EMA teachers and NIM enrichment.
- **ReAct / Chain-of-Thought Agents** — reasoning frameworks for LLM agents. Our approach supplements reasoning with programmatic guards for reliability.
