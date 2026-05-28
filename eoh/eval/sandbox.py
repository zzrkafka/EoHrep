"""Subprocess sandbox for safe evaluation of LLM-generated code."""
from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import traceback
from dataclasses import dataclass, field

logger = logging.getLogger("eoh")

_MP_CTX = multiprocessing.get_context("spawn")


@dataclass
class EvalResult:
    """Result of one sandbox evaluation."""
    ok: bool
    value: float | None = None
    error_type: str | None = None
    error_msg: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "value": self.value,
            "error_type": self.error_type,
            "error_msg": self.error_msg,
            "traceback": self.traceback,
        }


def _eval_worker(queue, problem, code) -> None:
    """Subprocess entry: evaluate code and put the result dict in the queue."""
    if sys.platform != "win32":
        os.setsid()
    try:
        result = problem.evaluate(code)
        if isinstance(result, EvalResult):
            queue.put(result.to_dict())
        elif result is None:
            queue.put(EvalResult(ok=False, error_type="EvalReturnedNone",
                                  error_msg="problem.evaluate returned None").to_dict())
        else:
            queue.put(EvalResult(ok=True, value=float(result)).to_dict())
    except BaseException as e:
        queue.put(EvalResult(
            ok=False,
            error_type=type(e).__name__,
            error_msg=str(e)[:500] if str(e) else type(e).__name__,
            traceback=traceback.format_exc()[-4000:],
        ).to_dict())


def eval_with_timeout(problem, code: str, timeout: float) -> EvalResult:
    """Run `problem.evaluate(code)` in a spawn subprocess; enforce timeout."""
    q = _MP_CTX.Queue()
    p = _MP_CTX.Process(target=_eval_worker, args=(q, problem, code))
    p.start()
    p.join(timeout)
    if p.is_alive():
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        p.terminate()
        p.join()
        return EvalResult(
            ok=False,
            error_type="SandboxTimeout",
            error_msg=f"Sandbox killed after {timeout}s",
        )
    try:
        raw = q.get_nowait()
    except Exception as e:
        return EvalResult(
            ok=False,
            error_type="SandboxQueueEmpty",
            error_msg=f"Subprocess produced no result: {e}",
        )
    return EvalResult(**raw)
