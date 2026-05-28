"""Unit tests for code/algorithm extraction. Covers upstream's edge cases."""
from __future__ import annotations

from eoh.util.code_extract import extract, prepend_imports


def test_fenced_python_block() -> None:
    resp = """{This algorithm picks the bin with the smallest leftover capacity.}
```python
import numpy as np
def score(item, bins):
    return -bins
```"""
    algo, code = extract(resp)
    assert len(code) == 1
    assert "def score" in code[0]
    assert algo == ["This algorithm picks the bin with the smallest leftover capacity."]


def test_fenced_plain_block() -> None:
    resp = """{Best-fit variant trying tightest leftover.}
```
def score(item, bins):
    return -(bins - item)
```"""
    algo, code = extract(resp)
    assert len(code) == 1
    assert "def score" in code[0]


def test_no_fence_uses_ast_fallback() -> None:
    resp = """{Smallest leftover wins.}
def score(item, bins):
    return -bins

Some trailing prose that should not break parsing."""
    algo, code = extract(resp)
    assert len(code) == 1
    assert "def score" in code[0]
    assert "trailing prose" not in code[0]


def test_short_braces_ignored_as_algorithm() -> None:
    # `{x}` is too short (< 8 chars), should not be matched as description.
    resp = """The idea is simple.
```python
def score(item, bins):
    return {x: 1}.get('y', bins)
```"""
    algo, code = extract(resp)
    # Algorithm falls back to the pre-code text since no ≥8-char brace match.
    assert algo == ["The idea is simple."]


def test_empty_response() -> None:
    assert extract(None) == ([], [])
    assert extract("") == ([], [])


def test_strips_leading_brace_line_inside_code_block() -> None:
    resp = """```python
{leftover description line}
def score(item, bins):
    return -bins
```"""
    algo, code = extract(resp)
    assert code[0].startswith("def score")


def test_prepend_imports_adds_missing() -> None:
    template = "import numpy as np\ndef score(item, bins):\n    return bins\n"
    body = "def score(item, bins):\n    return -bins\n"
    out = prepend_imports(body, template)
    assert out.startswith("import numpy as np")
    assert "def score" in out


def test_prepend_imports_skip_existing() -> None:
    template = "import numpy as np\ndef score(item, bins):\n    return bins\n"
    body = "import numpy as np\ndef score(item, bins):\n    return -bins\n"
    out = prepend_imports(body, template)
    assert out == body  # nothing added
