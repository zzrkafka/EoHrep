"""Main EoH evolution loop."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

from eoh.config import Config
from eoh.ea.evolution import Evolution
from eoh.ea.selection import population_management
from eoh.llm.client import LLMClient
from eoh.tasks.base import BaseProblem
from eoh.util.ckpt import POP_FIELDS, SampleJournal, strip_private

logger = logging.getLogger("eoh")


def _log_header(cfg: Config) -> None:
    logger.info("=" * 54)
    logger.info("  EoH")
    logger.info("  LLM      : %s @ %s", cfg.llm.model, cfg.llm.base_url)
    logger.info(
        "  EC       : gen=%d  pop=%d  ops=[%s]",
        cfg.ea.n_pop, cfg.ea.pop_size, " ".join(cfg.ea.operators),
    )
    init_n = 2 * cfg.ea.pop_size
    logger.info(
        "  Sampling : init=%d (2×pop)  per_gen=%d  parallel=%d",
        init_n, cfg.ea.pop_size, cfg.task.n_processes,
    )
    logger.info(
        "  Timeout  : llm=%ds  eval=%ds",
        cfg.llm.timeout, cfg.task.timeout,
    )
    logger.info("=" * 54)


def _format_sample_line(journal: SampleJournal, op: str, off: dict, is_new_best: bool) -> str:
    if off.get("objective") is None:
        score_str = f"None ({off.get('failure') or 'unknown'})"
    else:
        score_str = str(off["objective"])
    best = str(journal.best_obj) if journal.best_obj is not None else "N/A"
    marker = "  *" if is_new_best else ""
    timing = f"{off['total_ms']}ms"
    return (
        f"  #{journal.count:<4} [{op}]  {score_str:<32}  best={best}  "
        f"id={off['id']}  t={timing}{marker}"
    )


def _write_summary(run_dir: Path, journal: SampleJournal, total_ms: int) -> None:
    summary = {
        "best_objective": journal.best_obj,
        "best_sample_id": journal.best_id,
        "total_samples": journal.count,
        "total_time_ms": total_ms,
        "total_time_min": round(total_ms / 60000, 2),
        "failure_counts": dict(journal.failure_counts),
        "operator_stats": dict(journal.operator_stats),
    }
    path = run_dir / "results" / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_eoh(
    cfg: Config,
    llm: LLMClient,
    problem: BaseProblem,
    run_dir: Path,
    resume_path: str | None = None,
) -> dict | None:
    random.seed(cfg.ea.seed)
    journal = SampleJournal(run_dir)
    evolution = Evolution(cfg.ea, problem, llm)

    _log_header(cfg)
    t0 = time.perf_counter()

    if resume_path is not None:
        with Path(resume_path).open("r", encoding="utf-8") as f:
            population = json.load(f)
        logger.info("[Resume] loaded %d individuals from %s", len(population), resume_path)
        gen_start = _gen_id_from_resume_path(resume_path)
    else:
        logger.info("")
        logger.info("[Init]  (%d samples → pop=%d)", 2 * cfg.ea.pop_size, cfg.ea.pop_size)
        raw: list[dict] = []
        init_ops = ["i1"] * cfg.ea.pop_size
        for _ in range(2):
            _, batch = evolution.get_algorithm([], init_ops)
            for op, off in zip(init_ops, batch):
                is_new_best = journal.record(op, off)
                logger.info(_format_sample_line(journal, op, off, is_new_best))
                if off.get("objective") is not None:
                    raw.append(strip_private(off))
        population = population_management(raw, cfg.ea.pop_size)
        if not population:
            journal.flush()
            _write_summary(run_dir, journal, int((time.perf_counter() - t0) * 1000))
            raise RuntimeError(
                "Initial population is empty. Check LLM connectivity and that evaluate_program "
                f"returns a valid float. See {run_dir}/results/ for per-sample failure records."
            )
        elapsed_min = (time.perf_counter() - t0) / 60
        logger.info(
            "  Init done: %d/%d evaluated  pop=%d  best=%s  elapsed=%.1fm",
            len(raw), journal.count, len(population), population[0]["objective"], elapsed_min,
        )
        journal.save_population(population, 0)
        gen_start = 0

    for gen in range(gen_start, cfg.ea.n_pop):
        logger.info("")
        logger.info("[Gen %d/%d]", gen + 1, cfg.ea.n_pop)

        selected_ops = random.choices(
            cfg.ea.operators, weights=cfg.ea.operator_weights, k=cfg.ea.pop_size,
        )
        _, offspring = evolution.get_algorithm(population, selected_ops)
        for op, off in zip(selected_ops, offspring):
            is_new_best = journal.record(op, off)
            logger.info(_format_sample_line(journal, op, off, is_new_best))
            if off.get("objective") is not None:
                population.append(strip_private(off))
        population = population_management(population, cfg.ea.pop_size)

        if population:
            journal.save_population(population, gen + 1)
        elapsed_min = (time.perf_counter() - t0) / 60
        best = population[0]["objective"] if population else "N/A"
        logger.info(
            "  --- gen %d done  pop=%d  best=%s  elapsed=%.1fm",
            gen + 1, len(population), best, elapsed_min,
        )

    journal.flush()

    total_ms = int((time.perf_counter() - t0) * 1000)
    best = population[0] if population else None
    best_obj = best["objective"] if best else "N/A"
    logger.info("=" * 54)
    logger.info(
        "  Evolution finished.  best=%s  samples=%d  time=%.1fm",
        best_obj, journal.count, total_ms / 60000,
    )
    logger.info("=" * 54)

    _write_summary(run_dir, journal, total_ms)
    return best


def _gen_id_from_resume_path(resume_path: str) -> int:
    name = Path(resume_path).stem
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        logger.warning("Could not parse generation id from %s, starting at 0.", resume_path)
        return 0
