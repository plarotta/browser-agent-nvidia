import json
import os
import logging

import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

logger = logging.getLogger(__name__)

# Prompt char budget.  The system prompt is now compact (format instructions first,
# DOM/history last) so 3000 chars gives plenty of room while staying safe for
# Gemma-3 / mlx-vlm image-token limits.
MLX_MAX_PROMPT_CHARS = 3000


def _remap_peft_to_mlx(weights: dict) -> dict:
    """Remap PEFT (PyTorch) LoRA weight keys to MLX format.

    PEFT:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    MLX:   model.layers.0.self_attn.q_proj.lora_a
    """
    remapped = {}
    for key, value in weights.items():
        new_key = key
        # Strip PEFT prefix
        if new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]
        # lora_A.weight -> lora_a, lora_B.weight -> lora_b
        new_key = new_key.replace(".lora_A.weight", ".lora_a")
        new_key = new_key.replace(".lora_B.weight", ".lora_b")
        new_key = new_key.replace(".lora_A.default.weight", ".lora_a")
        new_key = new_key.replace(".lora_B.default.weight", ".lora_b")
        # Transpose: PEFT stores Linear weights as (out, in), MLX LoRA stores raw matrices
        # lora_A: PEFT (rank, in_features) -> MLX (in_features, rank)
        # lora_B: PEFT (out_features, rank) -> MLX (rank, out_features)
        if new_key.endswith(".lora_a") or new_key.endswith(".lora_b"):
            value = mx.transpose(value)
        remapped[new_key] = value
    return remapped


class MLXPolicy:
    def __init__(self, model_id: str = "mlx-community/gemma-3-12b-it-qat-4bit", adapter_path: str = None):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.config = None

    def load_model(self):
        print(f"Loading MLX model: {self.model_id}...")
        self.model, self.processor = load(self.model_id, trust_remote_code=True)
        self.config = load_config(self.model_id)

        if self.adapter_path and os.path.isdir(self.adapter_path):
            self._load_adapter()

        print("MLX Model loaded.")

    def _load_adapter(self):
        """Apply LoRA structure and load saved adapter weights."""
        from mlx_vlm.trainer.utils import get_peft_model, find_all_linear_names

        # Read adapter config if available
        config_path = os.path.join(self.adapter_path, "adapter_config.json")
        rank = 16
        alpha = 32.0
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                adapter_cfg = json.load(f)
            rank = adapter_cfg.get("lora_rank", 16)
            alpha = float(adapter_cfg.get("lora_alpha", rank * 2))
            logger.info(f"Adapter config: rank={rank}, alpha={alpha}")

        # Apply LoRA structure
        linear_names = find_all_linear_names(self.model)
        self.model = get_peft_model(
            self.model,
            linear_names,
            rank=rank,
            alpha=alpha,
            dropout=0.0,
            freeze=True,
        )

        # Load saved weights — support both MLX and PEFT (PyTorch) formats
        mlx_file = os.path.join(self.adapter_path, "adapters.safetensors")
        peft_file = os.path.join(self.adapter_path, "adapter_model.safetensors")

        if os.path.exists(mlx_file):
            weights = mx.load(mlx_file)
            self.model.load_weights(list(weights.items()), strict=False)
            mx.eval(self.model.parameters())
            logger.info(f"Loaded MLX adapter weights from {mlx_file}")
        elif os.path.exists(peft_file):
            weights = mx.load(peft_file)
            remapped = _remap_peft_to_mlx(weights)
            self.model.load_weights(list(remapped.items()), strict=False)
            mx.eval(self.model.parameters())
            logger.info(f"Loaded PEFT adapter weights from {peft_file} ({len(remapped)} keys remapped)")
        else:
            logger.warning(f"No adapter weights found in {self.adapter_path}")

    def _get_model_type(self) -> str:
        if isinstance(self.config, dict):
            return self.config.get("model_type", "")
        return getattr(self.config, "model_type", "")

    def forward(self, image: Image.Image, prompt: str) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Gemma 3 (and some other VLMs in mlx-vlm) fail or output empty when the prompt
        # is too long (~500+ tokens). Truncate so the model actually sees the image.
        if len(prompt) > MLX_MAX_PROMPT_CHARS:
            prompt = prompt[: MLX_MAX_PROMPT_CHARS - 20] + "\n\n... [truncated]"

        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=1,
        )

        # Llama 3.2 Vision (mllama) fix: apply_chat_template prepends
        # <|begin_of_text|>, but generate()'s internal prepare_inputs always
        # tokenises with add_special_tokens=True, which adds a SECOND BOS.
        # The double-BOS corrupts positional encodings and the model outputs
        # random garbage.  Strip the leading BOS so prepare_inputs adds
        # exactly one.
        if self._get_model_type() == "mllama":
            bos = "<|begin_of_text|>"
            if formatted_prompt.startswith(bos):
                formatted_prompt = formatted_prompt[len(bos) :]

        output = generate(
            self.model,
            self.processor,
            prompt=formatted_prompt,
            image=[image],
            verbose=False,
            max_tokens=256,
            temperature=0.0,
        )

        # Library may return str or an object with .text
        text = getattr(output, "text", None)
        if text is not None:
            return text
        if isinstance(output, str):
            return output
        return ""
