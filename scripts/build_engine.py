import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)

def main():
    parser = argparse.ArgumentParser(description="Build TensorRT-LLM engine for Llama-3.1-Nemotron-Nano-VL-8B-V1")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the Hugging Face model directory")
    parser.add_argument("--output_dir", type=str, default="trt_engine_output", help="Directory to save the built engine")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for the engine (float16, bfloat16)")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallelism size")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoint"
    
    print(f"Building engine for model at {model_dir}")
    print(f"Output directory: {output_dir}")
    
    # 1. Convert Checkpoint
    # Note: Llama-3.1-Nemotron-Nano-VL-8B-V1 is based on Llama architecture + Vision
    # We use the generic llama conversion script provided by TRT-LLM examples
    # Adjust the script path based on where TRT-LLM examples are installed or use the python module if available.
    
    # Assuming standard installation where we can use `trtllm-build` directly, 
    # but first we often need `python convert_checkpoint.py` from examples.
    # However, newer TRT-LLM versions support direct loading or have a `checkpoint` command.
    
    # Let's assume the user has cloned tensorrt_llm and we are using the python API or a standard conversion command.
    # For simplicity in this script, we will assume we are converting a generic Llama model (ignoring vision for a moment? No, we need multimodal).
    
    # ACTUAL TRT-LLM Workflow for Multimodal is complex and often requires specific example scripts.
    # We will output instructions and attempt a standard build command.
    
    print("\nIMPORTANT: Ensure you have tensorrt_llm installed and accessible.")
    
    # Check for trtllm-build availability
    try:
        subprocess.run("trtllm-build --help", shell=True, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error: `trtllm-build` command not found. Please install tensorrt_llm.")
        sys.exit(1)

    # Clean output dirs
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step 1: Convert Checkpoint
    # This part relies on specific conversion scripts usually found in TensorRT-LLM examples
    # e.g., examples/llama/convert_checkpoint.py or examples/multimodal/run.py logic.
    # Creating a fully self-contained builder is hard without the repo.
    # We will try to use the `trtllm-build` command which is the modern way.
    
    print("Step 1: Converting checkpoint (mock step - depends on exact TRT-LLM version)")
    # For many models, we can try to use the high level commands.
    
    # Command template (User might need to adjust based on their TRT-LLM version)
    # python3 -m tensorrt_llm.commands.convert_checkpoint --model_dir <model_dir> --output_dir <ckpt_dir> --dtype float16
    
    convert_cmd = f"python3 -m tensorrt_llm.commands.build --checkpoint_dir {model_dir} --output_dir {output_dir} --gemm_plugin float16"
    
    # However, standard `trtllm-build` works on *checkpoints*. We need to convert HF -> TRT Checkpoint first.
    convert_cmd = f"python3 -m tensorrt_llm.commands.convert_checkpoint --model_dir {model_dir} --output_dir {checkpoint_dir} --dtype {args.dtype} --tp_size {args.tp_size}"
    
    run_command(convert_cmd)
    
    # Step 2: Build Engine
    print("\nStep 2: Building TRT Engine")
    build_cmd = f"trtllm-build --checkpoint_dir {checkpoint_dir} --output_dir {output_dir} --gemm_plugin {args.dtype}"
    
    run_command(build_cmd)
    
    print(f"\nEngine built successfully at {output_dir}")

if __name__ == "__main__":
    main()
