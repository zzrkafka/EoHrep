"""LLM client Protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def get_response(self, prompt: str) -> str | None:
        """Return text response, or None on failure."""
        ...
