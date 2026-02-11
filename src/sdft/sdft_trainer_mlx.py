"""SDFT (Self-Distillation Fine-Tuning) trainer for Apple Silicon via MLX.

Implements Algorithm 1 from arXiv:2601.19897:
  1. On-policy rollout from student
  2. Teacher forward with ICL demonstration conditioning
  3. Student forward + reverse KL loss
  4. Gradient update + EMA teacher update
"""

import json
import os
import io
import base64
import logging
from typing import List, Dict, Any

import requests
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from PIL import Image

logger = logging.getLogger(__name__)

MLX_MAX_PROMPT_CHARS = 3000
NIM_ENRICHMENT_OBS_CHARS = 1500


class TrajectoryDataset:
    """Loads trajectory JSON + screenshots for SDFT training.

    Filters to steps with reward > 0 (successful actions).
    """

    def __init__(self, trajectory_dirs: List[str]):
        self.samples: List[Dict[str, str]] = []
        for traj_dir in trajectory_dirs:
            if not os.path.isdir(traj_dir):
                logger.warning(f"Trajectory dir not found: {traj_dir}")
                continue
            json_files = sorted(
                f
                for f in os.listdir(traj_dir)
                if f.endswith(".json") and f.startswith("traj_")
            )
            for jf in json_files:
                with open(os.path.join(traj_dir, jf), "r") as f:
                    steps = json.load(f)
                for step in steps:
                    if step.get("reward", 0) <= 0:
                        continue
                    screenshot_path = step.get("screenshot_path", "")
                    if not os.path.isabs(screenshot_path):
                        screenshot_path = os.path.join(
                            traj_dir, os.path.basename(screenshot_path)
                        )
                    if not os.path.exists(screenshot_path):
                        continue
                    self.samples.append(
                        {
                            "screenshot_path": screenshot_path,
                            "prompt": step.get("observation_dom", ""),
                            "target": json.dumps(step.get("action", {})),
                        }
                    )
        logger.info(f"TrajectoryDataset: {len(self.samples)} positive samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["screenshot_path"]).convert("RGB")
        return image, s["prompt"], s["target"]


def enrich_demonstration(
    image: Image.Image,
    observation: str,
    expert_action: str,
    api_key: str,
    api_url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
    model_id: str = "nvidia/nemotron-nano-12b-v2-vl",
) -> str:
    """Call NIM VLM to produce a rich ICL demonstration with reasoning.

    On failure, returns the raw expert_action as fallback.
    """
    # Encode screenshot as base64 PNG
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    truncated_obs = observation[:NIM_ENRICHMENT_OBS_CHARS]

    messages = [
        {
            "role": "system",
            "content": (
                "/no_think You are an expert browser agent tutor. Given a webpage "
                "screenshot, the DOM observation, and an expert's chosen action, "
                "produce a concise explanation of WHY this action is correct. Include:\n"
                "1. What the page currently shows (key visual/text context)\n"
                "2. Which element the action targets and why it's the right choice\n"
                "3. What the expected outcome of this action is\n\n"
                "Keep it to 3-5 sentences. End with the action JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"DOM observation:\n{truncated_obs}\n\n"
                        f"Expert action taken:\n{expert_action}\n\n"
                        "Explain why this action is correct:"
                    ),
                },
            ],
        },
    ]

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        enriched = resp.json()["choices"][0]["message"]["content"]
        logger.debug("NIM enriched demonstration (%d chars)", len(enriched))
        return enriched
    except Exception as e:
        logger.warning("NIM enrichment failed (%s), falling back to raw action", e)
        return expert_action


def build_teacher_prompt(student_prompt: str, demonstration: str) -> str:
    """Append ICL demonstration block to the student prompt."""
    return (
        f"{student_prompt}\n\n"
        "DEMONSTRATION:\n"
        f"{demonstration}\n\n"
        "Now provide your own action in the same JSON format:"
    )


def compute_kl_loss(student_logits: mx.array, teacher_logits: mx.array) -> mx.array:
    """Reverse KL divergence: D_KL(student || teacher).

    Args:
        student_logits: (1, T, V) raw logits at rollout positions.
        teacher_logits: (1, T, V) raw logits at rollout positions (constant).
    Returns:
        Scalar KL loss.
    """
    s_p = mx.softmax(student_logits.astype(mx.float32), axis=-1)
    t_p = mx.softmax(teacher_logits.astype(mx.float32), axis=-1)

    s_p = mx.maximum(s_p, 1e-8)
    t_p = mx.maximum(t_p, 1e-8)

    per_token_kl = mx.sum(s_p * (mx.log(s_p) - mx.log(t_p)), axis=-1)
    return mx.mean(per_token_kl)


def run_sdft_training(
    model_id: str,
    trajectory_dirs: List[str],
    adapter_save_path: str,
    num_epochs: int = 2,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    ema_alpha: float = 0.02,
    max_gen_tokens: int = 256,
    enrich: bool = True,
) -> Dict[str, Any]:
    """Run SDFT training on Apple Silicon using MLX.

    Returns dict with training stats.
    """
    from mlx_vlm import load, apply_chat_template, stream_generate, prepare_inputs
    from mlx_vlm.utils import load_config
    from mlx_vlm.trainer.utils import get_peft_model, find_all_linear_names
    from mlx_vlm.trainer.trainer import save_adapter

    # ── Load model ──
    logger.info(f"Loading model: {model_id}")
    model, processor = load(model_id)
    config = load_config(model_id)

    # ── Apply LoRA ──
    linear_names = find_all_linear_names(model)
    logger.info(f"Applying LoRA (rank={lora_rank}) to {len(linear_names)} layers")
    model = get_peft_model(
        model,
        linear_names,
        rank=lora_rank,
        alpha=float(lora_rank * 2),
        dropout=0.0,
        freeze=True,
    )

    # ── Teacher weights: snapshot of initial LoRA params ──
    mx.eval(model.parameters())
    teacher_weights = {k: v for k, v in tree_flatten(model.trainable_parameters())}

    # ── Optimizer ──
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # ── Dataset ──
    dataset = TrajectoryDataset(trajectory_dirs)
    if len(dataset) == 0:
        logger.warning("No training samples found")
        return {"status": "failed", "message": "No positive samples in trajectories"}

    # ── NIM enrichment setup ──
    nim_api_key = os.environ.get("NVIDIA_API_KEY") if enrich else None
    if enrich and not nim_api_key:
        logger.warning(
            "NVIDIA_API_KEY not set — falling back to raw demonstrations "
            "(set the env var or use --no-enrich to suppress this warning)"
        )

    # ── Image token index (model-specific) ──
    image_token_index = None
    if hasattr(model, "config"):
        image_token_index = getattr(model.config, "image_token_index", None)

    # ── Loss function for nn.value_and_grad ──
    def sdft_loss(model, full_ids, pixel_values, mask, prompt_len, num_rollout, t_logits):
        outputs = model(full_ids, pixel_values, mask=mask)
        logits = outputs.logits.astype(mx.float32)
        # logits[:, prompt_len-1] predicts rollout_token[0]
        student_logits = logits[:, prompt_len - 1 : prompt_len + num_rollout - 1, :]
        return compute_kl_loss(student_logits, t_logits)

    loss_and_grad_fn = nn.value_and_grad(model, sdft_loss)

    # ── Training loop ──
    total_steps = 0
    total_loss = 0.0

    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ({len(dataset)} samples) ===")

        for idx in range(len(dataset)):
            image, student_prompt_raw, expert_action = dataset[idx]

            # Truncate prompt (same budget as agent runtime)
            if len(student_prompt_raw) > MLX_MAX_PROMPT_CHARS:
                student_prompt_raw = (
                    student_prompt_raw[: MLX_MAX_PROMPT_CHARS - 20]
                    + "\n\n... [truncated]"
                )

            if enrich and nim_api_key:
                demonstration = enrich_demonstration(
                    image, student_prompt_raw, expert_action,
                    api_key=nim_api_key,
                )
            else:
                demonstration = expert_action

            teacher_prompt_raw = build_teacher_prompt(
                student_prompt_raw, demonstration
            )

            formatted_student = apply_chat_template(
                processor, config, student_prompt_raw, num_images=1
            )
            formatted_teacher = apply_chat_template(
                processor, config, teacher_prompt_raw, num_images=1
            )

            # ── STEP 2: On-policy rollout (student, no grad) ──
            rollout_ids = []
            for result in stream_generate(
                model,
                processor,
                prompt=formatted_student,
                image=[image],
                temperature=1.0,
                max_tokens=max_gen_tokens,
            ):
                if result.token is not None:
                    rollout_ids.append(result.token)

            if not rollout_ids:
                logger.debug(f"Sample {idx}: empty rollout, skipping")
                continue

            rollout_array = mx.array(rollout_ids)[None, :]  # (1, R)
            num_rollout = len(rollout_ids)

            # Prepare tokenized inputs
            teacher_inputs = prepare_inputs(
                processor,
                images=[image],
                prompts=formatted_teacher,
                image_token_index=image_token_index,
            )
            student_inputs = prepare_inputs(
                processor,
                images=[image],
                prompts=formatted_student,
                image_token_index=image_token_index,
            )

            # ── STEP 3: Teacher logits (no grad) ──
            # Save student LoRA weights, swap in teacher
            mx.eval(model.parameters())
            student_weights = tree_flatten(model.trainable_parameters())

            model.load_weights(list(teacher_weights.items()))
            mx.eval(model.parameters())

            t_ids = mx.concatenate(
                [teacher_inputs["input_ids"], rollout_array], axis=1
            )
            t_mask = mx.concatenate(
                [teacher_inputs["attention_mask"], mx.ones((1, num_rollout))],
                axis=1,
            )
            t_pv = teacher_inputs["pixel_values"]
            t_prompt_len = teacher_inputs["input_ids"].shape[1]

            teacher_out = model(t_ids, t_pv, mask=t_mask)
            teacher_rollout_logits = teacher_out.logits.astype(mx.float32)[
                :, t_prompt_len - 1 : t_prompt_len + num_rollout - 1, :
            ]
            mx.eval(teacher_rollout_logits)

            # Restore student LoRA weights
            model.load_weights(student_weights)
            mx.eval(model.parameters())

            # ── STEP 4: Student forward + KL loss (with grad) ──
            s_ids = mx.concatenate(
                [student_inputs["input_ids"], rollout_array], axis=1
            )
            s_mask = mx.concatenate(
                [student_inputs["attention_mask"], mx.ones((1, num_rollout))],
                axis=1,
            )
            s_pv = student_inputs["pixel_values"]
            s_prompt_len = student_inputs["input_ids"].shape[1]

            loss, grads = loss_and_grad_fn(
                model,
                s_ids,
                s_pv,
                s_mask,
                s_prompt_len,
                num_rollout,
                teacher_rollout_logits,
            )

            # ── STEP 5: Update student ──
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            # ── STEP 6: EMA update teacher ──
            for k, v in tree_flatten(model.trainable_parameters()):
                if k in teacher_weights:
                    teacher_weights[k] = (
                        (1 - ema_alpha) * teacher_weights[k] + ema_alpha * v
                    )
            mx.eval(list(teacher_weights.values()))

            total_steps += 1
            loss_val = loss.item()
            total_loss += loss_val

            logger.info(
                f"[E{epoch + 1} S{total_steps}] "
                f"Sample {idx + 1}/{len(dataset)}, "
                f"Loss: {loss_val:.4f}, "
                f"Rollout: {num_rollout} tok"
            )

    # ── Save adapter ──
    os.makedirs(adapter_save_path, exist_ok=True)
    adapter_file = os.path.join(adapter_save_path, "adapters.safetensors")
    save_adapter(model, adapter_file)
    logger.info(f"Adapter saved to {adapter_save_path}")

    return {
        "status": "completed",
        "total_steps": total_steps,
        "avg_loss": total_loss / max(total_steps, 1),
    }
