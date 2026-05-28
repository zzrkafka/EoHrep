"""End-to-end integration test driven by StubLLMClient.

A 2-pop × 2-gen run is fast enough to keep CI under a few seconds while
exercising every component: init, parent_selection, build_prompt, extract,
sandbox eval, population_management, and the journal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eoh.config import Config, EAConfig, LLMConfig, LogConfig, TaskConfig
from eoh.ea.loop import run_eoh
from eoh.llm.stub import DEFAULT_RESPONSE, StubLLMClient
from eoh.tasks.obp import BPONLINE
from eoh.util.logging import setup_logger

BEST_FIT_OBJ = 0.03984   # np.round(0.03984304255961367, 5)


def _build_cfg(tmp: Path) -> Config:
    return Config(
        task=TaskConfig(name="obp", capacity=100, timeout=40, n_processes=1),
        llm=LLMConfig(),
        ea=EAConfig(pop_size=2, n_pop=2, n_parents=2),
        logging=LogConfig(output_dir=str(tmp), debug=False),
    )


def test_full_run_with_stub(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)

    run_dir = tmp_path / "run0"
    (run_dir / "results" / "samples").mkdir(parents=True)
    (run_dir / "results" / "pops").mkdir()
    (run_dir / "results" / "pops_best").mkdir()
    setup_logger(run_dir / "results" / "run_log.txt")

    problem = BPONLINE(capacity=cfg.task.capacity, timeout=cfg.task.timeout)
    llm = StubLLMClient([DEFAULT_RESPONSE])

    best = run_eoh(cfg, llm, problem, run_dir=run_dir)

    # 1. Run produced expected best (stub always emits best-fit).
    assert best is not None
    assert math.isclose(best["objective"], BEST_FIT_OBJ, abs_tol=1e-9)

    # 2. Sample count = 2*pop + n_pop*pop = 4 + 4 = 8.
    samples_path = next((run_dir / "results" / "samples").glob("samples_*~*.json"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    assert len(samples) == 2 * cfg.ea.pop_size + cfg.ea.n_pop * cfg.ea.pop_size == 8

    # 3. Every sample carries an algorithm + code + objective.
    for s in samples:
        assert s["algorithm"]
        assert "def score" in s["code"]
        assert s["objective"] == BEST_FIT_OBJ
        assert s["operator"] in {"i1", "e1", "e2", "m1", "m2"}

    # 4. Per-gen population checkpoints exist for gen 0, 1, 2.
    pops_dir = run_dir / "results" / "pops"
    for g in range(0, cfg.ea.n_pop + 1):
        assert (pops_dir / f"population_generation_{g}.json").exists(), g

    # 5. samples_best.json reflects the global best.
    best_record = json.loads((run_dir / "results" / "samples" / "samples_best.json").read_text())
    assert math.isclose(best_record["objective"], BEST_FIT_OBJ, abs_tol=1e-9)


def test_invalid_code_skipped(tmp_path: Path) -> None:
    """LLM returns syntactically broken code → no individual added to pop."""
    cfg = _build_cfg(tmp_path)
    run_dir = tmp_path / "run1"
    (run_dir / "results" / "samples").mkdir(parents=True)
    (run_dir / "results" / "pops").mkdir()
    (run_dir / "results" / "pops_best").mkdir()
    setup_logger(run_dir / "results" / "run_log.txt")

    problem = BPONLINE(capacity=cfg.task.capacity, timeout=cfg.task.timeout)
    bad_response = "{this idea is broken.}\n```python\ndef score(\n```"
    llm = StubLLMClient([bad_response])

    with pytest.raises(RuntimeError, match="Initial population is empty"):
        run_eoh(cfg, llm, problem, run_dir=run_dir)
