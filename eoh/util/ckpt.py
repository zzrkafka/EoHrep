"""Sample journal, population checkpoints, and prompt forensics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("eoh")

SAMPLE_BATCH = 200

SAMPLE_FIELDS = (
    "sample_order", "id", "operator", "parent_ids",
    "algorithm", "code", "objective", "failure",
    "error_type", "error_msg",
    "llm_retries", "dedup_retries", "llm_ms", "eval_ms", "total_ms",
)

POP_FIELDS = ("id", "algorithm", "code", "objective", "other_inf")


class SampleJournal:
    def __init__(self, run_dir: Path):
        self.samples_dir = run_dir / "results" / "samples"
        self.prompts_dir = run_dir / "results" / "prompts"
        self.pops_dir = run_dir / "results" / "pops"
        self.pops_best_dir = run_dir / "results" / "pops_best"
        for d in (self.samples_dir, self.prompts_dir, self.pops_dir, self.pops_best_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._flushed = 0
        self.count = 0
        self.best_obj: float | None = None
        self.best_id: str | None = None
        self.failure_counts: dict[str, int] = {}
        self.operator_stats: dict[str, dict[str, int]] = {}

    def record(self, op: str, offspring: dict) -> bool:
        """Record one offspring. Returns True if it is a new global best."""
        self.count += 1

        obj = offspring.get("objective")
        is_new_best = obj is not None and (self.best_obj is None or obj < self.best_obj)
        if is_new_best:
            self.best_obj = obj
            self.best_id = offspring["id"]

        stat = self.operator_stats.setdefault(
            op, {"attempted": 0, "succeeded": 0, "new_best": 0}
        )
        stat["attempted"] += 1
        if obj is not None:
            stat["succeeded"] += 1
        if is_new_best:
            stat["new_best"] += 1

        if offspring.get("failure"):
            self.failure_counts[offspring["failure"]] = (
                self.failure_counts.get(offspring["failure"], 0) + 1
            )

        record = {k: offspring.get(k) for k in SAMPLE_FIELDS}
        record["sample_order"] = self.count
        record["operator"] = op

        self._write_prompt_log(offspring)

        self._buffer.append(record)
        if len(self._buffer) >= SAMPLE_BATCH:
            self._flush()
        if is_new_best:
            self._write_best(record)
        return is_new_best

    def flush(self) -> None:
        self._flush()

    def save_population(self, population: list[dict], gen: int) -> None:
        slim = [{k: ind.get(k) for k in POP_FIELDS} for ind in population]
        path = self.pops_dir / f"population_generation_{gen}.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(slim, f, indent=4)
        except OSError as e:
            logger.warning("Could not save checkpoint to %s: %s", path, e)

        if slim:
            best_path = self.pops_best_dir / f"population_generation_{gen}.json"
            try:
                with best_path.open("w", encoding="utf-8") as f:
                    json.dump(slim[0], f, indent=4)
            except OSError as e:
                logger.warning("Could not save best individual to %s: %s", best_path, e)

    def _write_prompt_log(self, offspring: dict) -> None:
        """Write prompts, responses, and eval traceback to prompts/sample_<id>.json."""
        prompts = offspring.get("_prompts") or []
        responses = offspring.get("_responses") or []
        tb = offspring.get("_eval_traceback")
        if not prompts and not responses and not tb:
            return
        path = self.prompts_dir / f"sample_{offspring['id']}.json"
        blob = {
            "sample_id": offspring["id"],
            "operator": offspring.get("operator"),
            "parent_ids": offspring.get("parent_ids", []),
            "failure": offspring.get("failure"),
            "error_type": offspring.get("error_type"),
            "error_msg": offspring.get("error_msg"),
            "attempts": [
                {"prompt": p, "response": r}
                for p, r in zip(prompts, responses)
            ],
            "evaluation_error": (
                {
                    "error_type": offspring.get("error_type"),
                    "error_msg": offspring.get("error_msg"),
                    "traceback": tb,
                }
                if tb
                else None
            ),
        }
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning("Could not write prompt log to %s: %s", path, e)

    def _flush(self) -> None:
        if not self._buffer:
            return
        lo = self._flushed + 1
        hi = self._flushed + len(self._buffer)
        path = self.samples_dir / f"samples_{lo}~{hi}.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(self._buffer, f, indent=4)
            self._flushed += len(self._buffer)
            self._buffer = []
        except OSError as e:
            logger.warning("Could not flush samples to %s: %s", path, e)

    def _write_best(self, record: dict) -> None:
        path = self.samples_dir / "samples_best.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=4)
        except OSError as e:
            logger.warning("Could not write best sample to %s: %s", path, e)


def strip_private(ind: dict) -> dict:
    """Return a copy of ind with underscore-prefixed keys removed."""
    return {k: v for k, v in ind.items() if not k.startswith("_")}
