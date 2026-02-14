import os
import base64
import io
import requests
from PIL import Image


class NIMPolicy:
    def __init__(self, model_id: str = "nvidia/llama-4-maverick-17b-128e-instruct"):
        self.model_id = model_id
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        self.api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = None       # set to True after "load" (protocol compat)
        self.processor = None   # unused, protocol compat

    def load_model(self):
        """Validate API key. No local model to load."""
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY env var not set")
        self.model = True  # signal "loaded" for lazy-load check in agent_runtime
        print(f"NIM backend ready (model: {self.model_id})")

    def forward(self, image: Image.Image, prompt: str) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Encode screenshot as base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        messages = [
            {"role": "system", "content": "/no_think"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                ],
            },
        ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]
