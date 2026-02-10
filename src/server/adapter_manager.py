import json
import os
import signal
import subprocess
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List

import requests

from src.shared.schemas import AdapterInfo

logger = logging.getLogger(__name__)


class AdapterManager:
    def __init__(self, adapters_dir: str, vllm_url: str, model_id: str):
        self.adapters_dir = adapters_dir
        self.vllm_url = vllm_url
        self.model_id = model_id
        self.meta_path = os.path.join(adapters_dir, "adapters_meta.json")
        self.active_adapter: Optional[str] = None
        self._vllm_process: Optional[subprocess.Popen] = None
        os.makedirs(adapters_dir, exist_ok=True)
        self._meta = self._load_meta()

    def _load_meta(self) -> dict:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                return json.load(f)
        return {}

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

    def register_adapter(self, name: str, training_steps: int):
        adapter_path = os.path.join(self.adapters_dir, name)
        self._meta[name] = {
            "path": adapter_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_steps": training_steps,
        }
        self._save_meta()
        logger.info(f"Registered adapter '{name}' ({training_steps} steps)")

    def load_adapter(self, name: str):
        if name not in self._meta:
            raise ValueError(f"Adapter '{name}' not found")
        adapter_path = self._meta[name]["path"]
        resp = requests.post(
            f"{self.vllm_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": adapter_path},
            timeout=30,
        )
        resp.raise_for_status()
        self.active_adapter = name
        logger.info(f"Loaded adapter '{name}' into vLLM")

    def unload_adapter(self, name: str):
        resp = requests.post(
            f"{self.vllm_url}/v1/unload_lora_adapter",
            json={"lora_name": name},
            timeout=30,
        )
        resp.raise_for_status()
        if self.active_adapter == name:
            self.active_adapter = None
        logger.info(f"Unloaded adapter '{name}' from vLLM")

    def list_adapters(self) -> List[AdapterInfo]:
        result = []
        for name, meta in self._meta.items():
            result.append(AdapterInfo(
                name=name,
                path=meta["path"],
                created_at=meta["created_at"],
                training_steps=meta["training_steps"],
                is_active=(name == self.active_adapter),
            ))
        return result

    def is_loaded(self, name: str) -> bool:
        return self.active_adapter == name

    def is_vllm_ready(self) -> bool:
        try:
            resp = requests.get(f"{self.vllm_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def stop_vllm(self):
        if self._vllm_process and self._vllm_process.poll() is None:
            logger.info("Stopping vLLM process to free GPU memory...")
            self._vllm_process.send_signal(signal.SIGTERM)
            self._vllm_process.wait(timeout=30)
            self._vllm_process = None
            self.active_adapter = None
            logger.info("vLLM stopped")

    def start_vllm(self, model_id: Optional[str] = None, adapter_name: Optional[str] = None):
        mid = model_id or self.model_id
        cmd = [
            "vllm", "serve", mid,
            "--host", "0.0.0.0",
            "--port", self.vllm_url.split(":")[-1].rstrip("/"),
            "--enable-lora",
            "--max-lora-rank", "64",
        ]
        if adapter_name and adapter_name in self._meta:
            adapter_path = self._meta[adapter_name]["path"]
            cmd.extend(["--lora-modules", f"{adapter_name}={adapter_path}"])

        env = os.environ.copy()
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

        logger.info(f"Starting vLLM: {' '.join(cmd)}")
        self._vllm_process = subprocess.Popen(cmd, env=env)

        # Wait for vLLM to become ready
        for _ in range(120):
            if self.is_vllm_ready():
                logger.info("vLLM is ready")
                if adapter_name:
                    self.active_adapter = adapter_name
                return
            time.sleep(2)
        raise RuntimeError("vLLM failed to start within 240 seconds")
