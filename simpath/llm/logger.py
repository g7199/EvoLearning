"""Log all LLM calls for reproducibility (Section 12)."""

import json
import os
import time
from pathlib import Path


class LLMLogger:
    def __init__(self, log_dir: str = "outputs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "llm_calls.jsonl"

    def log(self, provider: str, model: str, prompt: str, response: str,
            system: str = None, temperature: float = None, seed: int = None,
            elapsed_seconds: float = None):
        entry = {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "system": system,
            "prompt": prompt,
            "response": response,
            "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds else None,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
