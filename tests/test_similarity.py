"""Unit tests for similarity primitives."""
from __future__ import annotations

from eoh.analysis.similarity import ast_signature, token_jaccard


def test_ast_signature_alpha_rename_invariant() -> None:
    a = """
def score(item, bins):
    leftover = bins - item
    return -leftover
"""
    b = """
def score(item, bins):
    x = bins - item
    return -x
"""
    assert ast_signature(a) == ast_signature(b)


def test_ast_signature_constant_invariant() -> None:
    a = "def score(item, bins): return bins * 1.5"
    b = "def score(item, bins): return bins * 2.7"
    # Both multiply a name by a float constant — normaliser strips the value.
    assert ast_signature(a) == ast_signature(b)


def test_ast_signature_structural_change_differs() -> None:
    a = "def score(item, bins): return bins"
    b = "def score(item, bins): return -bins"
    assert ast_signature(a) != ast_signature(b)


def test_ast_signature_docstring_ignored() -> None:
    a = """def score(item, bins):
    \"\"\"long docstring that should not affect signature.\"\"\"
    return -bins
"""
    b = "def score(item, bins):\n    return -bins\n"
    assert ast_signature(a) == ast_signature(b)


def test_ast_signature_syntax_error_returns_none() -> None:
    assert ast_signature("def score(") is None


def test_token_jaccard_identical_codes() -> None:
    code = "import numpy as np\ndef score(item, bins): return -bins"
    assert token_jaccard(code, code) == 1.0


def test_token_jaccard_disjoint_codes() -> None:
    a = "def score(item, bins): return -bins"
    b = "class Foo: pass"
    j = token_jaccard(a, b)
    assert 0 <= j < 0.5


def test_token_jaccard_partial_overlap() -> None:
    a = "def score(item, bins): return -bins"
    b = "def score(item, bins): return bins"  # differs by one `-`
    j = token_jaccard(a, b)
    assert 0.8 < j < 1.0, j
