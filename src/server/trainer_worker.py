"""Server-side SDFT trainer (PyTorch + PEFT LoRA on GPU).

Implements true on-policy SDFT (arXiv:2601.19897):
  1. On-policy rollout from student (sampling, no grad)
  2. Swap to EMA teacher weights, forward with ICL-enriched prompt on rollout tokens
  3. Compute reverse KL: D_KL(student || teacher)
  4. Gradient update + EMA teacher update

No SFT term — pure KL distillation from enriched teacher.
"""

import json
import os
import logging
from typing import List, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.sdft.enrichment import enrich_demonstration, build_teacher_prompt

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 3000


class TrajectoryDataset(Dataset):
    """Loads trajectory JSON + screenshots for training.

    Filters to steps with reward > 0 (successful actions).
    Each item is (image, prompt_text, target_text).
    """

    def __init__(self, trajectory_dirs: List[str]):
        self.samples = []

        for traj_dir in trajectory_dirs:
            if not os.path.isdir(traj_dir):
                logger.warning(f"Trajectory dir not found: {traj_dir}")
                continue
            json_files = sorted(
                f for f in os.listdir(traj_dir)
                if f.endswith(".json") and f.startswith("traj_") and "_meta" not in f
            )
            for jf in json_files:
                with open(os.path.join(traj_dir, jf), "r") as f:
                    steps = json.load(f)
                for step in steps:
                    if step.get("reward", 0) <= 0:
                        continue
                    screenshot_path = step.get("screenshot_path", "")
                    if not os.path.isabs(screenshot_path):
                        screenshot_path = os.path.join(traj_dir, os.path.basename(screenshot_path))
                    if not os.path.exists(screenshot_path):
                        continue
                    self.samples.append({
                        "screenshot_path": screenshot_path,
                        "prompt": step.get("observation_dom", ""),
                        "target": json.dumps(step.get("action", {})),
                    })

        logger.info(f"TrajectoryDataset: {len(self.samples)} positive samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["screenshot_path"]).convert("RGB")
        return {
            "image": image,
            "prompt": sample["prompt"],
            "target": sample["target"],
        }


def run_training(
    model_id: str,
    trajectory_dirs: List[str],
    adapter_save_path: str,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    resume_from: str = None,
    ema_alpha: float = 0.02,
    enrich: bool = True,
    wandb_project: str = None,
    wandb_run_name: str = None,
) -> Dict[str, Any]:
    """Run on-policy SDFT training with PEFT LoRA on a single GPU.

    Returns dict with training stats.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel

    # ── W&B setup ──
    wb_run = None
    if wandb_project:
        try:
            import wandb
            wb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "model_id": model_id,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "lora_rank": lora_rank,
                    "ema_alpha": ema_alpha,
                    "enrich": enrich,
                    "backend": "pytorch_gpu",
                },
            )
            logger.info(f"W&B run started: {wb_run.url}")
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging (pip install wandb)")
        except Exception as e:
            logger.warning(f"wandb init failed: {e} — continuing without W&B")

    logger.info(f"Loading base model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply or resume LoRA
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from adapter: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from)
    else:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Teacher state: clone LoRA weights only (not full model)
    teacher_state = {
        k: v.clone() for k, v in model.named_parameters() if v.requires_grad
    }
    ema_decay = 1.0 - ema_alpha  # ema_alpha is the update rate toward student

    # Dataset
    dataset = TrajectoryDataset(trajectory_dirs)
    if len(dataset) == 0:
        logger.warning("No training samples found")
        return {"status": "failed", "message": "No positive samples in trajectories"}

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # NIM enrichment setup
    nim_api_key = os.environ.get("NVIDIA_API_KEY") if enrich else None
    if enrich and not nim_api_key:
        logger.warning(
            "NVIDIA_API_KEY not set — falling back to raw demonstrations"
        )

    # Enrichment cache dir alongside adapter save path
    cache_dir = os.path.join(os.path.dirname(adapter_save_path), ".enrichment_cache") if enrich else None

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    total_steps = 0
    total_loss = 0.0

    model.train()
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ({len(dataset)} samples) ===")

        for batch in dataloader:
            image = batch["image"][0]  # batch_size=1
            student_prompt = batch["prompt"][0]
            expert_action = batch["target"][0]

            # Truncate prompt
            if len(student_prompt) > MAX_PROMPT_CHARS:
                student_prompt = student_prompt[:MAX_PROMPT_CHARS - 20] + "\n\n... [truncated]"

            # ── Enrichment ──
            if enrich and nim_api_key:
                demonstration = enrich_demonstration(
                    student_prompt, expert_action,
                    api_key=nim_api_key,
                    cache_dir=cache_dir,
                )
            else:
                demonstration = expert_action

            teacher_prompt = build_teacher_prompt(student_prompt, demonstration)

            # ── STEP 1: On-policy rollout from student (no grad) ──
            student_inputs = processor(
                text=student_prompt,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                rollout_output = model.generate(
                    **student_inputs,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=256,
                )
            # Extract only the generated tokens (after the prompt)
            prompt_len = student_inputs["input_ids"].shape[1]
            rollout_ids = rollout_output[:, prompt_len:]  # (1, R)
            num_rollout = rollout_ids.shape[1]

            if num_rollout == 0:
                logger.debug("Empty rollout, skipping")
                continue

            # ── STEP 2: Teacher forward with enriched prompt on rollout tokens ──
            # Save student LoRA weights, swap in teacher
            student_state = {}
            for name, param in model.named_parameters():
                if name in teacher_state:
                    student_state[name] = param.data.clone()
                    param.data.copy_(teacher_state[name])

            teacher_inputs = processor(
                text=teacher_prompt,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            # Concatenate teacher prompt with rollout tokens
            t_prompt_len = teacher_inputs["input_ids"].shape[1]
            t_full_ids = torch.cat([teacher_inputs["input_ids"], rollout_ids], dim=1)
            t_attention = torch.cat([
                teacher_inputs["attention_mask"],
                torch.ones_like(rollout_ids),
            ], dim=1)

            # Build teacher inputs with concatenated ids
            t_forward_inputs = {
                "input_ids": t_full_ids,
                "attention_mask": t_attention,
            }
            # Pass through pixel_values if present
            if "pixel_values" in teacher_inputs:
                t_forward_inputs["pixel_values"] = teacher_inputs["pixel_values"]

            with torch.no_grad():
                teacher_outputs = model(**t_forward_inputs)
                # Extract logits at rollout positions
                teacher_logits = teacher_outputs.logits[:, t_prompt_len - 1:t_prompt_len + num_rollout - 1, :]

            # Restore student weights
            for name, param in model.named_parameters():
                if name in student_state:
                    param.data.copy_(student_state[name])

            # ── STEP 3: Student forward on same rollout tokens (with grad) ──
            s_full_ids = torch.cat([student_inputs["input_ids"], rollout_ids], dim=1)
            s_attention = torch.cat([
                student_inputs["attention_mask"],
                torch.ones_like(rollout_ids),
            ], dim=1)

            s_forward_inputs = {
                "input_ids": s_full_ids,
                "attention_mask": s_attention,
            }
            if "pixel_values" in student_inputs:
                s_forward_inputs["pixel_values"] = student_inputs["pixel_values"]

            student_outputs = model(**s_forward_inputs)
            student_logits = student_outputs.logits[:, prompt_len - 1:prompt_len + num_rollout - 1, :]

            # ── Reverse KL: D_KL(student || teacher) ──
            student_logprobs = torch.nn.functional.log_softmax(student_logits.float(), dim=-1)
            student_probs = torch.nn.functional.softmax(student_logits.float(), dim=-1)
            teacher_logprobs = torch.nn.functional.log_softmax(teacher_logits.float(), dim=-1)

            # D_KL(p || q) = sum p * (log p - log q)
            kl_loss = (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1).mean()

            # ── Step-0 KL diagnostic ──
            total_steps += 1
            if total_steps == 1:
                kl_val = kl_loss.item()
                if kl_val < 0.01:
                    logger.warning(
                        f"DIAGNOSTIC: Step-0 KL = {kl_val:.6f} (near zero). "
                        "Enrichment may be too weak or ema_alpha too low. "
                        "Consider increasing ema_alpha or improving enrichment prompt."
                    )
                else:
                    logger.info(
                        f"DIAGNOSTIC: Step-0 KL = {kl_val:.6f} — signal looks healthy."
                    )

            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()

            # ── EMA update teacher ──
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in teacher_state:
                        teacher_state[name].mul_(ema_decay).add_(
                            param.data, alpha=ema_alpha
                        )

            loss_val = kl_loss.item()
            total_loss += loss_val

            if wb_run:
                wb_run.log({
                    "kl_loss": loss_val,
                    "avg_loss": total_loss / total_steps,
                    "rollout_tokens": num_rollout,
                    "epoch": epoch + 1,
                    "step": total_steps,
                })

            if total_steps % 5 == 0:
                avg_loss = total_loss / total_steps
                logger.info(
                    f"[E{epoch + 1} S{total_steps}] "
                    f"KL Loss: {loss_val:.4f}, Avg: {avg_loss:.4f}, "
                    f"Rollout: {num_rollout} tok"
                )

    # Save adapter
    os.makedirs(adapter_save_path, exist_ok=True)
    model.save_pretrained(adapter_save_path)

    # Save adapter config
    config_path = os.path.join(adapter_save_path, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "lora_rank": lora_rank,
            "lora_alpha": lora_rank * 2,
            "model_id": model_id,
        }, f, indent=2)

    logger.info(f"Adapter saved to {adapter_save_path}")

    if wb_run:
        wb_run.log({"final_avg_loss": total_loss / max(total_steps, 1)})
        wb_run.finish()

    # Cleanup
    del model, optimizer, teacher_state
    torch.cuda.empty_cache()

    return {
        "status": "completed",
        "total_steps": total_steps,
        "avg_loss": total_loss / max(total_steps, 1),
    }
