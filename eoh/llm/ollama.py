"""Ollama LLM client via the OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger("eoh")


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b-instruct-q4_K_M",
        num_ctx: int = 8192,
        timeout: int = 180,
        max_retries: int = 5,
        verify_on_init: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.max_retries = max_retries
        if verify_on_init:
            self._verify_connection()

    def get_response(self, prompt: str) -> str | None:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": self.num_ctx},
        }
        url = f"{self.base_url}/v1/chat/completions"

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=body, timeout=self.timeout)
                if resp.status_code == 200:
                    parsed = resp.json()
                    choices = parsed.get("choices") or []
                    if not choices:
                        raise ValueError(f"No choices in response: {parsed!r}")
                    return choices[0]["message"]["content"]
                logger.debug(
                    "Ollama HTTP %d (attempt %d/%d): %s",
                    resp.status_code, attempt + 1, self.max_retries, resp.text[:200],
                )
            except Exception as e:
                logger.debug(
                    "Ollama error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        logger.warning(
            "Ollama call failed after %d attempts (url=%s, model=%s).",
            self.max_retries, url, self.model,
        )
        return None

    def _verify_connection(self) -> None:
        """Probe Ollama; raise RuntimeError if unreachable or model missing."""
        probe = self.get_response("1+1=?")
        if probe is None:
            raise RuntimeError(
                f"Ollama check failed. Is `ollama serve` running at {self.base_url}, "
                f"and is the model '{self.model}' pulled?"
            )
        logger.info("Ollama connection verified: %s @ %s", self.model, self.base_url)
