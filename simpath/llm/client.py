"""Unified LLM client for OpenAI and Anthropic APIs."""

import os
import json
import time
from typing import Optional
from simpath.llm.logger import LLMLogger


class LLMClient:
    """Unified interface for OpenAI and Anthropic LLMs."""

    def __init__(
        self,
        provider: str = "openai",  # "openai" | "anthropic" | "mock"
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        log_dir: str = "outputs/logs",
    ):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = LLMLogger(log_dir)

        if provider == "openai":
            import openai
            self.model = model or "gpt-5.4-2026-03-05"
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set. Pass api_key= or set env var.")
            self.client = openai.OpenAI(api_key=key)

        elif provider == "anthropic":
            import anthropic
            self.model = model or "claude-sonnet-4-6"
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set. Pass api_key= or set env var.")
            self.client = anthropic.Anthropic(api_key=key)

        elif provider == "mock":
            self.model = "mock"
            self.client = None

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system: str = None, seed: int = None) -> str:
        start = time.time()

        if self.provider == "openai":
            response = self._openai_generate(prompt, system, seed)
        elif self.provider == "anthropic":
            response = self._anthropic_generate(prompt, system)
        elif self.provider == "mock":
            response = self._mock_generate(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        elapsed = time.time() - start
        self.logger.log(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            system=system,
            response=response,
            temperature=self.temperature,
            seed=seed,
            elapsed_seconds=elapsed,
        )
        return response

    def _openai_generate(self, prompt: str, system: str = None, seed: int = None) -> str:
        """
        OpenAI Chat Completions API (SDK v2.29+).
        - system prompt → role: "developer" (modern) instead of "system"
        - max_completion_tokens (not max_tokens) for GPT-5.x models
        - seed is deprecated for chat completions but still accepted
        """
        messages = []
        if system:
            messages.append({"role": "developer", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _anthropic_generate(self, prompt: str, system: str = None) -> str:
        """
        Anthropic Messages API (SDK v0.86+).
        - system is a separate parameter, NOT in messages array
        - max_tokens (not max_completion_tokens)
        - temperature range: 0.0-1.0
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": min(self.temperature, 1.0),  # Anthropic caps at 1.0
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)
        return resp.content[0].text

    def _mock_generate(self, prompt: str) -> str:
        """Fallback mock: extracts question IDs from prompt and returns a JSON array."""
        import re
        ids = re.findall(r'(q_\d+)', prompt)
        unique_ids = list(dict.fromkeys(ids))  # preserve order, deduplicate
        return json.dumps(unique_ids[:8])
