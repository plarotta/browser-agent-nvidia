from mlx_vlm import load, generate, apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

# Prompt char budget.  The system prompt is now compact (format instructions first,
# DOM/history last) so 3000 chars gives plenty of room while staying safe for
# Gemma-3 / mlx-vlm image-token limits.
MLX_MAX_PROMPT_CHARS = 3000


class MLXPolicy:
    def __init__(self, model_id: str = "mlx-community/gemma-3-12b-it-qat-4bit"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.config = None

    def load_model(self):
        print(f"Loading MLX model: {self.model_id}...")
        self.model, self.processor = load(self.model_id, trust_remote_code=True)
        self.config = load_config(self.model_id)
        print("MLX Model loaded.")

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
