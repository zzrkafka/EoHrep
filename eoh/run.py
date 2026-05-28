"""CLI entry: `python -m eoh.run --config <path> [--dry-run] [--resume <pop.json>]`.

In --dry-run mode the LLM client is replaced by a stub returning a known-good
score function; this exercises the full pipeline (init → generations → eval
→ selection → checkpoints) without hitting Ollama.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from eoh.config import Config, load_yaml
from eoh.util.logging import setup_logger


def _timestamp_dir(parent: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path(parent) / ts
    (path / "results" / "samples").mkdir(parents=True, exist_ok=True)
    (path / "results" / "pops").mkdir(parents=True, exist_ok=True)
    (path / "results" / "pops_best").mkdir(parents=True, exist_ok=True)
    return path


def _dump_effective_config(cfg: Config, run_dir: Path) -> None:
    """Persist the effective config as a YAML so runs are reproducible."""
    blob: dict[str, Any] = {
        "task": vars(cfg.task),
        "llm": vars(cfg.llm),
        "ea": vars(cfg.ea),
        "logging": vars(cfg.logging),
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(blob, f, sort_keys=False, allow_unicode=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eoh", description="EoH-local runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use stub LLM (no Ollama); exercise the pipeline only")
    parser.add_argument("--resume", default=None,
                        help="Path to a population_generation_<g>.json to resume from")
    parser.add_argument("--output-dir", default=None,
                        help="Override config.logging.output_dir")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.output_dir:
        cfg.logging.output_dir = args.output_dir

    run_dir = _timestamp_dir(cfg.logging.output_dir)
    log_path = run_dir / "results" / "run_log.txt"
    logger = setup_logger(log_path, debug=cfg.logging.debug)
    _dump_effective_config(cfg, run_dir)

    # Run-level forensics: env + git + Ollama digest. Captured BEFORE the loop
    # starts so we still have it on a partial / crashed run.
    from eoh.util.runmeta import collect_env_metadata, dump_env
    env = collect_env_metadata(
        cwd=".",
        ollama_base_url=cfg.llm.base_url if cfg.llm.backend == "ollama" else None,
        ollama_model=cfg.llm.model if cfg.llm.backend == "ollama" else None,
    )
    dump_env(env, str(run_dir / "env.json"))
    if env.get("ollama", {}).get("digest"):
        logger.info("Model digest: %s", env["ollama"]["digest"])
    if env.get("git", {}).get("available"):
        dirty = "+dirty" if env["git"]["dirty"] else ""
        logger.info("Git: %s @ %s%s", env["git"]["branch"], env["git"]["commit"][:12], dirty)

    logger.info("=" * 54)
    logger.info("  EoH-local v%s", _pkg_version())
    logger.info("  config   : %s", args.config)
    logger.info("  run_dir  : %s", run_dir)
    logger.info("  dry-run  : %s", args.dry_run)
    logger.info("  resume   : %s", args.resume)
    logger.info("=" * 54)

    from eoh.ea.loop import run_eoh

    problem = _build_problem(cfg)

    if args.dry_run:
        from eoh.llm.stub import StubLLMClient
        llm = StubLLMClient()
        logger.info("LLM: stub (dry-run)")
    else:
        from eoh.llm.ollama import OllamaClient
        llm = OllamaClient(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            num_ctx=cfg.llm.num_ctx,
            timeout=cfg.llm.timeout,
            max_retries=cfg.llm.max_retries,
        )

    run_eoh(cfg, llm, problem, run_dir=run_dir, resume_path=args.resume)
    return 0


def _build_problem(cfg: Config):
    """Task factory dispatch by cfg.task.name."""
    if cfg.task.name == "obp":
        from eoh.tasks.obp import BPONLINE
        return BPONLINE(
            capacity=cfg.task.capacity,
            timeout=cfg.task.timeout,
            n_processes=cfg.task.n_processes,
        )
    raise ValueError(f"Unknown task: {cfg.task.name}")


def _pkg_version() -> str:
    from eoh import __version__
    return __version__


if __name__ == "__main__":
    sys.exit(main())
