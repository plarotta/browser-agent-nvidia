from src.policy.multimodal_policy import MultimodalPolicy
from transformers import LlavaProcessor
from PIL import Image
import traceback

print("Verifying MultimodalPolicy forward pass...")
try:
    policy = MultimodalPolicy(device="cpu")
    
    # Create a dummy image
    image = Image.new('RGB', (512, 512), color='red')
    prompt = "Describe this image."
    
    print(f"Processor class: {type(policy.processor)}")
    print(f"Image Processor class: {type(policy.processor.image_processor)}")
    print(f"Patch size: {getattr(policy.processor, 'patch_size', 'Not Found')}")
    print(f"Image Processor Patch size: {getattr(policy.processor.image_processor, 'patch_size', 'Not Found')}")
    print(f"Image Processor Config: {policy.processor.image_processor.to_dict()}")

    # Patch the patch_size if missing to see if it fixes it
    # if not hasattr(policy.processor, 'patch_size') or policy.processor.patch_size is None:
    #     print("Patching processor.patch_size to 16...")
    #     policy.processor.patch_size = 16
    
    print("Loading model...")
    policy.load_model()
    
    print("Running forward pass...")
    # This acts as the full integration test
    # Note: Use <image> tag as expected by the model
    prompt = "<image>Describe this image."
    output = policy.forward(image, prompt)
    print(f"Forward pass successful. Output: {output}")
    
except Exception as e:
    print(f"Caught exception: {e}")
    traceback.print_exc()
    exit(1)





