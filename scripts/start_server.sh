#!/bin/bash
# ============================================
# start_server.sh
# Launch Nemotron-Nano-12B-V2-VL on vLLM with
# automatic VRAM handling for a single 24GB GPU
# ============================================

MODEL_ID=${1:-"nvidia/Nemotron-Nano-12B-v2-VL-BF16"}
VLLM_PORT=8000
API_PORT=8080

# Environment tweaks
export VLLM_NUM_ENGINE_PROCS=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:64,expandable_segments:True"

# Helper function to launch vLLM
launch_vllm() {
    echo "Launching vLLM..."
    uv run python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --dtype bf16 \
        --port "$VLLM_PORT" \
        --offload-dir /tmp \
        --load-in-4bit
}

# Detect GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
echo "Detected GPU memory: ${GPU_MEM} MiB"

# If GPU < 32GB, enforce 4-bit + offload
if [ "$GPU_MEM" -lt 32000 ]; then
    echo "GPU < 32GB: using 4-bit quantization + CPU offload to avoid OOM"
    launch_vllm
else
    echo "GPU >= 32GB: loading full model in bf16"
    uv run
