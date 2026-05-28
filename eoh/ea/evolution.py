"""Build prompt, call LLM, extract code, evaluate, and return offspring."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from eoh.ea.prompts import build_prompt
from eoh.ea.selection import parent_selection
from eoh.eval.sandbox import eval_with_timeout
from eoh.llm.client import LLMClient
from eoh.util.code_extract import extract, prepend_imports
from eoh.util.failure import FailureReason

logger = logging.getLogger("eoh")


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _empty_offspring(op: str) -> dict:
    return {
        "id": _new_id(),
        "operator": op,
        "algorithm": None,
        "code": None,
        "objective": None,
        "other_inf": None,
        "parent_ids": [],
        "failure": None,
        "error_type": None,
        "error_msg": None,
        "llm_retries": 0,
        "dedup_retries": 0,
        "llm_ms": 0,
        "eval_ms": 0,
        "total_ms": 0,
        "_prompts": [],
        "_responses": [],
        "_eval_traceback": None,
    }


class Evolution:
    def __init__(self, ea_cfg, problem, llm: LLMClient) -> None:
        self.task_description = problem.task_description
        self.template_program = problem.template_program
        self.problem = problem
        self.llm = llm
        self.n_parents = ea_cfg.n_parents
        self.timeout = problem.timeout
        self.n_processes = problem.n_processes

    def _call_llm(self, prompt: str, off: dict) -> tuple[str | None, str | None]:
        """Try up to 4 times to get (algorithm, code) from the LLM."""
        algorithm: list[str] = []
        code: list[str] = []
        off["_prompts"].append(prompt)
        for attempt in range(4):
            t0 = time.perf_counter()
            response = self.llm.get_response(prompt)
            off["llm_ms"] += int((time.perf_counter() - t0) * 1000)
            off["llm_retries"] += 1
            off["_responses"].append(response)
            if response:
                logger.debug("  [response] attempt %d/4: %.500s", attempt + 1, response)
            algorithm, code = extract(response)
            if algorithm and code:
                break
            logger.debug("  [extract] attempt %d/4 failed.", attempt + 1)

        if not algorithm or not code:
            return None, None
        return prepend_imports(code[0], self.template_program), algorithm[0]

    def _generate(self, population: list[dict], operator: str, off: dict):
        if operator == "i1":
            parents: Any = None
            prompt = build_prompt("i1", self.task_description, self.template_program, None)
        elif operator in ("e1", "e2"):
            if not population:
                raise ValueError(f"Operator '{operator}' requires a non-empty population.")
            parents = parent_selection(population, self.n_parents)
            off["parent_ids"] = [p["id"] for p in parents]
            prompt = build_prompt(operator, self.task_description, self.template_program, parents)
        elif operator in ("m1", "m2", "m3"):
            if not population:
                raise ValueError(f"Operator '{operator}' requires a non-empty population.")
            parents = parent_selection(population, 1)[0]
            off["parent_ids"] = [parents["id"]]
            prompt = build_prompt(operator, self.task_description, self.template_program, parents)
        else:
            raise ValueError(f"Unknown operator: {operator}")

        logger.debug("  [prompt/%s] %d chars: %.400s", operator, len(prompt), prompt)
        code, algorithm = self._call_llm(prompt, off)
        if code:
            logger.debug("  [extract] algorithm: %.120r", algorithm)
            logger.debug("  [extract] code (%d chars): %.400s", len(code), code)
        return parents, code, algorithm

    def get_offspring(self, population: list[dict], operator: str) -> tuple[Any, dict]:
        """Generate one offspring. On failure, sets `failure` and leaves `objective` as None."""
        off = _empty_offspring(operator)
        t_total = time.perf_counter()
        try:
            parents, code, algorithm = self._generate(population, operator, off)
            if code is None:
                if not off["_responses"] or all(r is None for r in off["_responses"]):
                    off["failure"] = FailureReason.LLM_NO_RESPONSE.value
                else:
                    off["failure"] = FailureReason.EXTRACT_FAILED.value
                off["total_ms"] = int((time.perf_counter() - t_total) * 1000)
                return parents, off

            while self._is_duplicate(population, code) and off["dedup_retries"] < 2:
                logger.debug("  [offspring] duplicate — retrying...")
                off["dedup_retries"] += 1
                _, code, algorithm = self._generate(population, operator, off)
                if code is None:
                    off["failure"] = (
                        FailureReason.LLM_NO_RESPONSE.value
                        if not off["_responses"] or all(r is None for r in off["_responses"])
                        else FailureReason.EXTRACT_FAILED.value
                    )
                    off["total_ms"] = int((time.perf_counter() - t_total) * 1000)
                    return parents, off

            off["algorithm"] = algorithm
            off["code"] = code

            t_eval = time.perf_counter()
            result = eval_with_timeout(self.problem, code, self.timeout)
            off["eval_ms"] = int((time.perf_counter() - t_eval) * 1000)

            if not result.ok:
                off["error_type"] = result.error_type
                off["error_msg"] = result.error_msg
                off["_eval_traceback"] = result.traceback
                if result.error_type == "SandboxTimeout":
                    off["failure"] = FailureReason.EVAL_TIMEOUT.value
                else:
                    off["failure"] = FailureReason.EVAL_ERROR.value
                logger.debug(
                    "  [eval] %s (%s): %s after %dms",
                    off["failure"], result.error_type, result.error_msg, off["eval_ms"],
                )
            else:
                rounded = float(np.round(result.value, 5))
                if np.isfinite(rounded):
                    off["objective"] = rounded
                else:
                    off["failure"] = FailureReason.EVAL_NON_FINITE.value
                    off["error_msg"] = f"non-finite objective: {result.value}"

            off["total_ms"] = int((time.perf_counter() - t_total) * 1000)
            return parents, off

        except Exception as e:
            logger.debug("  [offspring] %s: %s", type(e).__name__, e)
            off["failure"] = FailureReason.GENERATION_EXCEPTION.value
            off["total_ms"] = int((time.perf_counter() - t_total) * 1000)
            return None, off

    def get_algorithm(self, population: list[dict], operators: list[str]):
        """Generate one offspring per operator. Uses parallel workers if n_processes > 1."""
        if self.n_processes == 1:
            results = [self.get_offspring(population, op) for op in operators]
        else:
            try:
                results = Parallel(n_jobs=self.n_processes)(
                    delayed(self.get_offspring)(population, op) for op in operators
                )
            except Exception as e:
                logger.warning(
                    "  [parallel] %s: %s — falling back to sequential",
                    type(e).__name__, e,
                )
                results = [self.get_offspring(population, op) for op in operators]
        parents = [p for p, _ in results]
        offspring = [o for _, o in results]
        return parents, offspring

    @staticmethod
    def _is_duplicate(population: list[dict], code: str) -> bool:
        return any(ind["code"] == code for ind in population)
