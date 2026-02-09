#!/bin/bash
echo "Starting Browser Agent Demo..."
echo "1. Checking dependencies..."
uv sync

echo "2. Running Agent Status Check..."
uv run python -m src.main status

echo "3. Launching Agent in Inference Mode (Dummy Task)..."
uv run python -m src.main run --task "demo_task" --no-headless

echo "Done!"
