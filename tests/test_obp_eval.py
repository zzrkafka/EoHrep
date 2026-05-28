"""Numerical regression tests for the OBP evaluator.

Three canary heuristics with deterministic objectives. If any number drifts,
something in the data, the binpacking loop, or numpy semantics has changed.
"""
from __future__ import annotations

import math

import pytest

from eoh.tasks.obp import BPONLINE


# Numbers captured on first successful evaluator run (Python 3.14, numpy ≥1.26).
# capacity=100, Weibull 5k (5 instances × 5000 items each), lb=1987.8.
GOLDEN: dict[str, float] = {
    "worst_fit_template":  1.5153435959352046,   # def score(item, bins): return bins
    "best_fit":            0.03984304255961367,  # def score(item, bins): return -bins
    "first_fit":           0.04225777241171155,  # def score(item, bins): return np.ones_like(bins)
}

ATOL = 1e-9


@pytest.fixture(scope="module")
def task() -> BPONLINE:
    return BPONLINE(capacity=100, timeout=40, n_processes=1)


def _eval(task: BPONLINE, code: str) -> float:
    obj = task.evaluate(code)
    assert obj is not None, "evaluate returned None — code did not compile/run"
    return obj


def test_worst_fit_template(task: BPONLINE) -> None:
    obj = _eval(task, task.template_program)
    assert math.isclose(obj, GOLDEN["worst_fit_template"], abs_tol=ATOL), obj


def test_best_fit(task: BPONLINE) -> None:
    code = "import numpy as np\ndef score(item, bins):\n    return -bins\n"
    obj = _eval(task, code)
    assert math.isclose(obj, GOLDEN["best_fit"], abs_tol=ATOL), obj


def test_first_fit(task: BPONLINE) -> None:
    code = "import numpy as np\ndef score(item, bins):\n    return np.ones_like(bins)\n"
    obj = _eval(task, code)
    assert math.isclose(obj, GOLDEN["first_fit"], abs_tol=ATOL), obj


def test_best_fit_beats_first_fit(task: BPONLINE) -> None:
    # Sanity ordering — best-fit should beat first-fit which should beat worst-fit.
    bf  = _eval(task, "import numpy as np\ndef score(item, bins): return -bins\n")
    ff  = _eval(task, "import numpy as np\ndef score(item, bins): return np.ones_like(bins)\n")
    wf  = _eval(task, task.template_program)
    assert bf < ff < wf, (bf, ff, wf)
