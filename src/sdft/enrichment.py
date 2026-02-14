"""Shared enrichment utilities for SDFT training.

Provides NIM VLM enrichment of expert demonstrations and teacher prompt
construction, used by both the MLX trainer and the server (PyTorch) trainer.
"""

import io
import base64
import hashlib
import json
import logging
import os

import requests
from PIL import Image

logger = logging.getLogger(__name__)

NIM_ENRICHMENT_OBS_CHARS = 1500


def enrich_demonstration(
    image: Image.Image,
    observation: str,
    expert_action: str,
    api_key: str,
    api_url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
    model_id: str = "meta/llama-3.2-90b-vision-instruct",
    cache_dir: str | None = None,
) -> str:
    """Call NIM VLM to produce a rich ICL demonstration with reasoning.

    If cache_dir is set, results are cached to avoid redundant API calls.
    On failure, returns the raw expert_action as fallback.
    """
    # Encode screenshot as base64 PNG
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    screenshot_bytes = buf.getvalue()
    b64_image = base64.b64encode(screenshot_bytes).decode("utf-8")

    # Check cache
    if cache_dir:
        cache_key = _cache_key(screenshot_bytes, observation, expert_action)
        cached = _cache_read(cache_dir, cache_key)
        if cached is not None:
            logger.debug("Enrichment cache hit: %s", cache_key[:12])
            return cached

    truncated_obs = observation[:NIM_ENRICHMENT_OBS_CHARS]

    messages = [
        {
            "role": "system",
            "content": (
                "/no_think You are an expert browser agent tutor. Given a webpage "
                "screenshot, the DOM observation, and an expert's chosen action, "
                "produce a concise explanation of WHY this action is correct. Include:\n"
                "1. What the page currently shows (key visual/text context)\n"
                "2. Which element the action targets and why it's the right choice\n"
                "3. What the expected outcome of this action is\n\n"
                "Keep it to 3-5 sentences. End with the action JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"DOM observation:\n{truncated_obs}\n\n"
                        f"Expert action taken:\n{expert_action}\n\n"
                        "Explain why this action is correct:"
                    ),
                },
            ],
        },
    ]

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        enriched = resp.json()["choices"][0]["message"]["content"]
        logger.debug("NIM enriched demonstration (%d chars)", len(enriched))

        if cache_dir:
            _cache_write(cache_dir, cache_key, enriched)

        return enriched
    except Exception as e:
        logger.warning("NIM enrichment failed (%s), falling back to raw action", e)
        return expert_action


def build_teacher_prompt(student_prompt: str, demonstration: str) -> str:
    """Append ICL demonstration block to the student prompt."""
    return (
        f"{student_prompt}\n\n"
        "DEMONSTRATION:\n"
        f"{demonstration}\n\n"
        "Now provide your own action in the same JSON format:"
    )


# ── Caching helpers ──

def _cache_key(screenshot_bytes: bytes, observation: str, expert_action: str) -> str:
    h = hashlib.sha256()
    h.update(screenshot_bytes)
    h.update(observation.encode("utf-8"))
    h.update(expert_action.encode("utf-8"))
    return h.hexdigest()


def _cache_read(cache_dir: str, key: str) -> str | None:
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)["enriched"]
    return None


def _cache_write(cache_dir: str, key: str, enriched: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w") as f:
        json.dump({"enriched": enriched}, f)
