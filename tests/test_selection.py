"""Unit tests for parent_selection + population_management."""
from __future__ import annotations

import random

import pytest

from eoh.ea.selection import parent_selection, population_management


def _ind(obj: float | None, code: str = "x") -> dict:
    return {"algorithm": "a", "code": code, "objective": obj, "other_inf": None}


def test_population_management_filters_none() -> None:
    pop = [_ind(None), _ind(0.1), _ind(None), _ind(0.05)]
    out = population_management(pop, size=5)
    assert [x["objective"] for x in out] == [0.05, 0.1]


def test_population_management_dedups_by_objective() -> None:
    pop = [_ind(0.1, "a"), _ind(0.1, "b"), _ind(0.2, "c")]
    out = population_management(pop, size=5)
    # Only the first 0.1 survives; second is dropped despite different code.
    assert len(out) == 2
    assert [x["objective"] for x in out] == [0.1, 0.2]


def test_population_management_topN_lower_is_better() -> None:
    pop = [_ind(0.5), _ind(0.1), _ind(0.3), _ind(0.2), _ind(0.4)]
    out = population_management(pop, size=3)
    assert [x["objective"] for x in out] == [0.1, 0.2, 0.3]


def test_population_management_size_larger_than_pop() -> None:
    pop = [_ind(0.1), _ind(0.2)]
    out = population_management(pop, size=10)
    assert len(out) == 2


def test_parent_selection_empty_pop_raises() -> None:
    with pytest.raises(ValueError):
        parent_selection([], m=2)


def test_parent_selection_returns_m_items() -> None:
    pop = [_ind(0.1), _ind(0.2), _ind(0.3), _ind(0.4)]
    random.seed(0)
    out = parent_selection(pop, m=3)
    assert len(out) == 3
    for p in out:
        assert p in pop


def test_parent_selection_top_ranked_higher_freq() -> None:
    """Over many draws, rank-0 should be sampled more than rank-3."""
    pop = [_ind(float(i)) for i in range(4)]
    random.seed(0)
    counts = [0, 0, 0, 0]
    for _ in range(4000):
        [pick] = parent_selection(pop, m=1)
        counts[int(pick["objective"])] += 1
    # weights ∝ 1/(r+1+4) = [1/5, 1/6, 1/7, 1/8] → relative ratio top:bottom = 1.6
    assert counts[0] > counts[3]
    ratio = counts[0] / counts[3]
    assert 1.3 < ratio < 1.9, f"unexpected ratio {ratio}"
