import torch.nn as nn
from PIL import Image
from typing import Optional
from src.policy.transformers_policy import TransformersPolicy
from src.policy.tensorrt_policy import TensorRTPolicy
# Deferred import for MLXPolicy to avoid hard dependency on non-mac systems, 
# but for now we can import it inside __init__ or just try/except
try:
    from src.policy.mlx_policy import MLXPolicy
except ImportError:
    MLXPolicy = None

try:
    from src.policy.nim_policy import NIMPolicy
except ImportError:
    NIMPolicy = None

class MultimodalPolicy(nn.Module):
    def __init__(self,
                 model_id: str = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
                 device: str = "cuda",
                 backend: str = "transformers",
                 engine_dir: Optional[str] = None,
                 adapter_path: Optional[str] = None):
        super().__init__()
        self.backend = backend
        self.device = device
        self.model_id = model_id
        self.engine_dir = engine_dir
        self.adapter_path = adapter_path

        if backend == "tensorrt":
            if not engine_dir:
                # Default engine dir if not specified
                self.engine_dir = "trt_engine_output"
            print(f"Initializing TensorRTPolicy with engine at {self.engine_dir}")
            self.impl = TensorRTPolicy(model_id, self.engine_dir, device)
        elif backend == "mlx":
            print("Initializing MLXPolicy")
            if MLXPolicy is None:
                raise ImportError("MLXPolicy could not be imported. Ensure mlx and mlx-vlm are installed.")
            self.impl = MLXPolicy(model_id, adapter_path=adapter_path)
        elif backend == "nim":
            print("Initializing NIMPolicy")
            if NIMPolicy is None:
                raise ImportError("NIMPolicy could not be imported. Ensure requests is installed.")
            self.impl = NIMPolicy(model_id)
        else:
            print("Initializing TransformersPolicy")
            self.impl = TransformersPolicy(model_id, device)

    @property
    def model(self):
        return self.impl.model

    @property
    def processor(self):
        if self.backend == "mlx":
             return self.impl.processor
        return self.impl.processor

    def load_model(self):
        self.impl.load_model()

    def forward(self, image: Image.Image, prompt: str) -> str:
        return self.impl.forward(image, prompt)
