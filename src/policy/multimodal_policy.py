import torch.nn as nn
from PIL import Image
from typing import Optional
from src.policy.transformers_policy import TransformersPolicy
from src.policy.tensorrt_policy import TensorRTPolicy

class MultimodalPolicy(nn.Module):
    def __init__(self, 
                 model_id: str = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1", 
                 device: str = "cuda", 
                 backend: str = "transformers",
                 engine_dir: Optional[str] = None):
        super().__init__()
        self.backend = backend
        self.device = device
        self.model_id = model_id
        self.engine_dir = engine_dir
        
        if backend == "tensorrt":
            if not engine_dir:
                # Default engine dir if not specified
                self.engine_dir = "trt_engine_output"
            print(f"Initializing TensorRTPolicy with engine at {self.engine_dir}")
            self.impl = TensorRTPolicy(model_id, self.engine_dir, device)
        else:
            print("Initializing TransformersPolicy")
            self.impl = TransformersPolicy(model_id, device)

    @property
    def model(self):
        return self.impl.model

    @property
    def processor(self):
        return self.impl.processor

    def load_model(self):
        self.impl.load_model()

    def forward(self, image: Image.Image, prompt: str) -> str:
        return self.impl.forward(image, prompt)
