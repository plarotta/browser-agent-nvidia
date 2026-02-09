import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlavaProcessor
from PIL import Image
from typing import Optional, List, Union

class TransformersPolicy(nn.Module):
    def __init__(self, model_id: str = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_id = model_id
        
        # Load the processor
        try:
            self.processor = LlavaProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True, 
                attn_implementation="eager"
            )
            # Patch the processor to include patch_size, which is missing in the image processor config
            # but required by LlavaProcessor's forward pass.
            self.processor.patch_size = 16
        except OSError:
            print(f"Warning: Could not load processor for {model_id}. implementation relies on it.")
            self.processor = None

        self.model = None

    def load_model(self):
        """Loads the actual model weights. Separated to allow lazy loading."""
        print(f"Loading model {self.model_id} to {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.eval() # Backbone is frozen by default in our design
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Set img_context_token_id required by the model
        if hasattr(self.processor, "tokenizer"):
            token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            self.model.img_context_token_id = token_id
            
        print("Model loaded and frozen.")

    def forward(self, image: Image.Image, prompt: str) -> str:
        """
        Forward pass for inference.
        Returns the generated text action.
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 1. Process image manually
        image_outputs = self.processor.image_processor([image], return_tensors="pt")
        pixel_values = image_outputs["pixel_values"].to(self.device)
        num_patches_list = image_outputs["num_patches"]
        
        # 2. Calculate tokens
        # Note: 256 is fixed for this model ref: modeling.py
        num_image_token = 256 
        
        # 3. Expand prompt
        # Ensure we handle <|image|> or <image> correctly
        prompt = prompt.replace("<|image|>", "<image>")
        
        if "<image>" in prompt:
            # We need to construct input_ids manually
            text_part = prompt.replace("<image>", "")
            
            # Tokenize text part
            # Use add_special_tokens=False because the prompt likely contains explicit special tokens
            text_ids = self.processor.tokenizer(text_part, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            
            # Create image tokens tensor
            total_tokens = num_image_token * num_patches_list[0]
            # Get token ID directly from model if set, else from tokenizer
            img_context_token_id = getattr(self.model, "img_context_token_id", None)
            if img_context_token_id is None: 
                img_context_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
                
            image_ids = torch.full((1, total_tokens), img_context_token_id, dtype=torch.long, device=self.device)
            
            # Concatenate: image tokens + text tokens
            # Assuming <image> is at the start based on current usage
            input_ids = torch.cat([image_ids, text_ids], dim=1)
        else:
            inputs = self.processor.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            input_ids = inputs.input_ids

        # Generate action
        # 4. Generate
        attention_mask = torch.ones_like(input_ids)
        
        output = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=128
        )
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded
