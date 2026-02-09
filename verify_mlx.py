from src.policy.mlx_policy import MLXPolicy
from PIL import Image
import numpy as np

def verify():
    print("Testing MLX Policy...")
    try:
        policy = MLXPolicy()
        policy.load_model()
        
        # Create dummy image
        img = Image.new('RGB', (100, 100), color='red')
        prompt = "Describe this image."
        
        print("Running inference...")
        output = policy.forward(img, prompt)
        print(f"Output: {output}")
        print("Verification Successful!")
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
