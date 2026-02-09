from src.policy.multimodal_policy import MultimodalPolicy
from src.utils.config import AgentConfig

print("Testing MultimodalPolicy initialization with fix...")
try:
    # This should now succeed because we hardcoded attn_implementation="eager" in the class
    policy = MultimodalPolicy(device="cpu")
    print("Success! MultimodalPolicy initialized without TypeError.")
    
    if policy.processor:
        print("Processor loaded successfully.")
    else:
        print("Warning: Processor is None (OSError caught?)")

    print("Attempting to load model...")
    policy.load_model()
    print("Model loaded successfully!")

except Exception as e:
    print(f"Failed: {e}")
    raise e

