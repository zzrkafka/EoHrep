"""Failure reason labels for offspring with no objective."""
from __future__ import annotations

from enum import Enum


class FailureReason(str, Enum):
    LLM_NO_RESPONSE      = "llm_no_response"
    EXTRACT_FAILED       = "extract_failed"
    DUPLICATE_GIVEUP     = "duplicate_giveup"
    EVAL_TIMEOUT         = "eval_timeout"
    EVAL_ERROR           = "eval_error"
    EVAL_NON_FINITE      = "eval_non_finite"
    GENERATION_EXCEPTION = "generation_exception"
