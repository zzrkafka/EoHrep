"""LLM client Protocol.

Mirrors upstream `InterfaceLLM.get_response` — single method, single user
message, returns a string or None.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def get_response(self, prompt: str) -> str | None:
        """Return the LLM's text response, or None after all retries fail."""
        ...
