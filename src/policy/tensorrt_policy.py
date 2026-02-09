import torch
import torch.nn as nn
from transformers import LlavaProcessor
from PIL import Image
from typing import Optional, List, Union
import os

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner
except ImportError:
    tensorrt_llm = None
    ModelRunner = None

class TensorRTPolicy(nn.Module):
    def __init__(self, model_id: str, engine_dir: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_id = model_id
        self.engine_dir = engine_dir
        
        if tensorrt_llm is None:
            print("Warning: tensorrt_llm not installed. TensorRTPolicy will not work.")
        
        # Load the processor (we still need this for tokenization/image processing)
        try:
            self.processor = LlavaProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True
            )
             # Patch as in original policy
            self.processor.patch_size = 16
        except Exception as e:
            print(f"Warning: Could not load processor for {model_id}: {e}")
            self.processor = None

        self.model = None # This will hold the runner

    def load_model(self):
        """Loads the TensorRT engine."""
        if tensorrt_llm is None:
            raise ImportError("tensorrt_llm is not installed.")
        
        if not os.path.exists(self.engine_dir):
            raise FileNotFoundError(f"TensorRT engine directory not found at {self.engine_dir}")

        print(f"Loading TensorRT engine from {self.engine_dir}...")
        
        # Use ModelRunner or similar high-level API
        # Note: The exact API might vary by TRT-LLM version, but ModelRunner is standard for recent versions
        try:
            self.runner = ModelRunner.from_dir(self.engine_dir)
            self.model = self.runner # Alias for compatibility
            print("TensorRT engine loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")

    def forward(self, image: Image.Image, prompt: str) -> str:
        """
        Forward pass using TensorRT-LLM.
        """
        if not self.model:
            raise RuntimeError("Model (Engine) not loaded. Call load_model() first.")

        # 1. Process inputs using HF Processor
        # We need to adapt this to what the TRT engine expects.
        # Usually TRT-LLM runners accept input_ids and pixel_values/images directly.
        
        prompt = prompt.replace("<|image|>", "<image>")
        
        # Example using high-level runner generation
        # The runner usually takes text and images
        
        # NOTE: This implementation depends on how the engine was built (inputs expected)
        # Assuming the engine was built to accept "input_ids" and "pixel_values" or similar
        
        # For straightforward generation with ModelRunner in recent TRT-LLM:
        # runner.generate(batch_input_ids, max_new_tokens=...)
        # But multimodal handling is specific.
        
        # Let's perform the tokenization here as we did in TransformersPolicy
        # to ensure we feed the right tokens.
        
        image_outputs = self.processor.image_processor([image], return_tensors="pt")
        # pixel_values = image_outputs["pixel_values"].half().cuda() # TRT usually expects half/float
        
        # Construct input_ids (reuse logic from TransformersPolicy for consistency)
        num_patches = image_outputs["num_patches"][0]
        num_image_token = 256 # Model specific
        
        if "<image>" in prompt:
            text_part = prompt.replace("<image>", "")
            text_ids = self.processor.tokenizer(text_part, return_tensors="pt", add_special_tokens=False).input_ids
            
            # For TRT-LLM multimodal, check if it handles image embeddings internally or expects tokens.
            # If using the standard multimodal example runner, it often takes `input_ids` with placeholders.
            
            # Let's assume we pass the full input_ids (with spacers) and pixel_values.
            # But TensorRT-LLM 0.9+ Multimodal usually handles the image embedding lookup if built correctly.
            # We will approximate the logic here. 
            
            # Simple approach: Delegate to runner.generate if it supports explicit multimodal args
            # If standard ModelRunner, we might need to pass embeddings.
            
            # For now, let's assume we can pass `input_ids` and `images` (or `pixel_values`)
            
            outputs = self.runner.generate(
                prompt, # Some runners take text directly
                images=[image],
                max_new_tokens=128
            )
            
            # If outputs is text:
            if isinstance(outputs, str):
                return outputs
            elif isinstance(outputs, list) and isinstance(outputs[0], str):
                return outputs[0]
            else:
                 # Decode if it returns tokens
                return self.processor.decode(outputs[0], skip_special_tokens=True)

        else:
             # Text only
             outputs = self.runner.generate(prompt, max_new_tokens=128)
             return outputs

