#!/bin/bash
# RunPod launch script: starts vLLM + FastAPI control plane
# Usage: ./scripts/start_server.sh [MODEL_ID]

set -e

MODEL_ID="${1:-nvidia/Nemotron-Nano-12B-v2-VL-BF16}"
VLLM_PORT=8000
API_PORT=8080
ADAPTERS_DIR="./adapters"
TRAJECTORIES_DIR="./trajectories"

# Enable runtime LoRA updates
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

# Memory management tweaks to avoid OOM
export VLLM_NUM_ENGINE_PROCS=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:64,expandable_segments:True"

echo "=== Browser Agent Server ==="
echo "Model: $MODEL_ID"
echo "vLLM port: $VLLM_PORT"
echo "API port: $API_PORT"

mkdir -p "$ADAPTERS_DIR" "$TRAJECTORIES_DIR"

# Detect GPU VRAM
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
echo "Detected GPU memory: ${GPU_MEM} MiB"

# Decide whether to enforce 4-bit + offload
VLLM_ARGS="--host 0.0.0.0 --port $VLLM_PORT --enable-lora --max-lora-rank 64 --trust-remote-code --dtype bf16"
if [ "$GPU_MEM" -lt 32000 ]; then
    echo "GPU < 32GB: enabling 4-bit quantization + CPU offload to avoid OOM"
    VLLM_ARGS="$VLLM_ARGS --load-in-4bit --offload-dir /tmp"
fi

# Start vLLM in background
echo "Starting vLLM..."
uv run vllm serve "$MODEL_ID" $VLLM_ARGS &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM is ready (took ~${i}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 2
done

# Final check
if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM failed to start within 240s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Start FastAPI control plane
echo "Starting FastAPI control plane on port $API_PORT..."
uv run python -m src.main serve \
    --model-id "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --vllm-url "http://localhost:$VLLM_PORT" \
    --adapters-dir "$ADAPTERS_DIR" \
    --trajectories-dir "$TRAJECTORIES_DIR"
