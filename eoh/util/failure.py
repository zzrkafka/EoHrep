"""FailureReason enum + canonical labels for `objective=None` samples.

When an offspring is recorded with `objective=None`, we need to know WHY
without trawling the debug log. The Evolution layer is responsible for
attaching exactly one of these labels to every failure record.
"""
from __future__ import annotations

from enum import Enum


class FailureReason(str, Enum):
    LLM_NO_RESPONSE      = "llm_no_response"        # all LLM retries returned None
    EXTRACT_FAILED       = "extract_failed"         # response did not yield (algorithm, code)
    DUPLICATE_GIVEUP     = "duplicate_giveup"       # 2 dedup retries exhausted; code matched pop member
    EVAL_TIMEOUT         = "eval_timeout"           # sandbox subprocess wall-clock > timeout
    EVAL_ERROR           = "eval_error"             # sandbox returned None (exec/runtime error)
    EVAL_NON_FINITE      = "eval_non_finite"        # objective was NaN or +/-inf
    GENERATION_EXCEPTION = "generation_exception"   # unexpected exception in get_offspring
