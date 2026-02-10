import os
import base64
import io
import requests
from PIL import Image


class RemoteVLLMPolicy:
    def __init__(self, model_id: str = "nvidia/Nemotron-Nano-12B-v2-VL-BF16"):
        self.model_id = model_id
        self.server_url = os.environ.get("VLLM_SERVER_URL", "http://localhost:8080")
        self.model = None
        self.processor = None

    def load_model(self):
        """Verify remote server is reachable."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            resp.raise_for_status()
            health = resp.json()
            print(f"Remote vLLM server ready (model: {health.get('model_id', 'unknown')})")
            self.model = True
        except Exception as e:
            raise RuntimeError(f"Cannot reach vLLM server at {self.server_url}: {e}")

    def forward(self, image: Image.Image, prompt: str) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "prompt": prompt,
            "screenshot_base64": b64_image,
        }

        resp = requests.post(
            f"{self.server_url}/act",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["raw_output"]
