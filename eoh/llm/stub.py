"""Stub LLM client for --dry-run and tests."""
from __future__ import annotations


DEFAULT_RESPONSE = """{Pick the bin with the smallest leftover capacity (best-fit).}
```python
import numpy as np
def score(item, bins):
    return -bins
```"""


class StubLLMClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else [DEFAULT_RESPONSE]
        self._idx = 0
        self.call_count = 0

    def get_response(self, prompt: str) -> str | None:
        self.call_count += 1
        r = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return r
