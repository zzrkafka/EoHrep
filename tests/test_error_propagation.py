"""Verify exception details (NameError / ZeroDivision / TypeError / syntax)
make it all the way from the sandbox subprocess back into the sample record.
"""
from __future__ import annotations

from eoh.eval.sandbox import EvalResult, eval_with_timeout
from eoh.tasks.obp import BPONLINE


def test_name_error_propagates() -> None:
    task = BPONLINE(capacity=100, timeout=5)
    code = """
import numpy as np
def score(item, bins):
    return undefined_variable + bins
"""
    r = eval_with_timeout(task, code, timeout=5)
    assert not r.ok
    assert r.error_type == "NameError", r.error_type
    assert "undefined_variable" in (r.error_msg or "")
    assert "NameError" in (r.traceback or "")


def test_zero_division_propagates() -> None:
    task = BPONLINE(capacity=100, timeout=5)
    code = """
import numpy as np
def score(item, bins):
    return bins / 0
"""
    r = eval_with_timeout(task, code, timeout=5)
    # numpy treats 1/0 as inf with a warning, NOT an exception. So this
    # should succeed at the eval layer; non-finite is filtered later by
    # Evolution. We just verify the layer below doesn't raise spuriously.
    # If a future numpy version makes this raise ZeroDivisionError, that's
    # also acceptable — adjust the assert.
    assert r.ok or r.error_type in {"ZeroDivisionError", "FloatingPointError"}


def test_syntax_error_during_exec() -> None:
    task = BPONLINE(capacity=100, timeout=5)
    code = "def score(item, bins"   # unclosed paren
    r = eval_with_timeout(task, code, timeout=5)
    assert not r.ok
    assert r.error_type == "SyntaxError", r.error_type


def test_missing_entry_point() -> None:
    task = BPONLINE(capacity=100, timeout=5)
    code = "def not_score(item, bins): return bins"
    r = eval_with_timeout(task, code, timeout=5)
    assert not r.ok
    assert r.error_type == "NameError"
    assert "score" in (r.error_msg or "")


def test_runtime_typeerror() -> None:
    task = BPONLINE(capacity=100, timeout=5)
    code = """
import numpy as np
def score(item, bins):
    return bins + None       # ndarray + None -> TypeError
"""
    r = eval_with_timeout(task, code, timeout=5)
    assert not r.ok
    assert r.error_type == "TypeError", r.error_type
    assert "None" in (r.error_msg or "") or "NoneType" in (r.error_msg or "")


def test_timeout_classified_separately() -> None:
    task = BPONLINE(capacity=100, timeout=2)
    code = """
def score(item, bins):
    while True:
        pass
"""
    r = eval_with_timeout(task, code, timeout=2)
    assert not r.ok
    assert r.error_type == "SandboxTimeout"


def test_success_still_returns_float() -> None:
    task = BPONLINE(capacity=100, timeout=5)
    code = "import numpy as np\ndef score(item, bins): return -bins"
    r = eval_with_timeout(task, code, timeout=5)
    assert r.ok
    assert isinstance(r.value, float)
    assert abs(r.value - 0.03984304255961367) < 1e-9
