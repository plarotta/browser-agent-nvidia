import json
import os
import logging
from typing import List, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """Loads trajectory JSON + screenshots for training.

    Filters to steps with reward > 0 (successful actions).
    Each item is (image, prompt_text, target_text).
    """

    def __init__(self, trajectory_dirs: List[str], tokenizer, processor):
        self.samples = []
        self.tokenizer = tokenizer
        self.processor = processor

        for traj_dir in trajectory_dirs:
            json_files = [
                f for f in os.listdir(traj_dir)
                if f.endswith(".json") and f.startswith("traj_")
            ]
            for jf in json_files:
                with open(os.path.join(traj_dir, jf), "r") as f:
                    steps = json.load(f)
                for step in steps:
                    if step.get("reward", 0) <= 0:
                        continue
                    screenshot_path = step.get("screenshot_path", "")
                    # Resolve relative to trajectory dir
                    if not os.path.isabs(screenshot_path):
                        screenshot_path = os.path.join(traj_dir, os.path.basename(screenshot_path))
                    if not os.path.exists(screenshot_path):
                        continue
                    action = step.get("action", {})
                    action_json = json.dumps(action)
                    dom = step.get("observation_dom", "")
                    self.samples.append({
                        "screenshot_path": screenshot_path,
                        "prompt": dom,
                        "target": action_json,
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
) -> Dict[str, Any]:
    """Run SDFT training with PEFT LoRA on a single GPU.

    Returns dict with training stats.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel

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
    ema_decay = 0.99

    # Dataset
    dataset = TrajectoryDataset(trajectory_dirs, processor, processor)
    if len(dataset) == 0:
        logger.warning("No training samples found")
        return {"status": "failed", "message": "No positive samples in trajectories"}

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    total_steps = 0
    total_loss = 0.0

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            image = batch["image"][0]  # batch_size=1
            prompt_text = batch["prompt"][0]
            target_text = batch["target"][0]

            # Build input: prompt + target for causal LM
            full_text = f"{prompt_text}\n{target_text}"
            inputs = processor(
                text=full_text,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            # SFT loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            sft_loss = outputs.loss

            # KL distillation: swap in teacher weights, compute teacher logits
            student_state = {}
            for name, param in model.named_parameters():
                if name in teacher_state:
                    student_state[name] = param.data.clone()
                    param.data.copy_(teacher_state[name])

            with torch.no_grad():
                teacher_outputs = model(**inputs)
                teacher_logits = teacher_outputs.logits

            # Restore student weights
            for name, param in model.named_parameters():
                if name in student_state:
                    param.data.copy_(student_state[name])

            # KL divergence
            student_logprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
            kl_loss = torch.nn.functional.kl_div(
                student_logprobs, teacher_probs, reduction="batchmean"
            )

            loss = sft_loss + 0.1 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update teacher
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in teacher_state:
                        teacher_state[name].mul_(ema_decay).add_(
                            param.data, alpha=1 - ema_decay
                        )

            total_steps += 1
            total_loss += loss.item()

            if total_steps % 10 == 0:
                avg_loss = total_loss / total_steps
                logger.info(f"Step {total_steps}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")

    # Save adapter
    os.makedirs(adapter_save_path, exist_ok=True)
    model.save_pretrained(adapter_save_path)
    logger.info(f"Adapter saved to {adapter_save_path}")

    # Cleanup
    del model, optimizer, teacher_state
    torch.cuda.empty_cache()

    return {
        "status": "completed",
        "total_steps": total_steps,
        "avg_loss": total_loss / max(total_steps, 1),
    }
