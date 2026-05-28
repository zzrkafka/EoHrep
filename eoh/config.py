"""Dataclass config + YAML loader."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskConfig:
    name: str = "obp"
    capacity: int = 100
    timeout: int = 40                # seconds per evaluate()
    n_processes: int = 1


@dataclass
class LLMConfig:
    backend: str = "ollama"          # "ollama" or "stub"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    num_ctx: int = 8192
    timeout: int = 180
    max_retries: int = 5


@dataclass
class EAConfig:
    pop_size: int = 4
    n_pop: int = 20
    operators: list[str] = field(default_factory=lambda: ["e1", "e2", "m1", "m2"])
    operator_weights: list[float] | None = None
    n_parents: int = 2
    seed: int = 2024


@dataclass
class LogConfig:
    output_dir: str = "runs"
    debug: bool = False


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ea: EAConfig = field(default_factory=EAConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    def __post_init__(self) -> None:
        if self.ea.operator_weights is None:
            self.ea.operator_weights = [1.0] * len(self.ea.operators)
        if len(self.ea.operator_weights) != len(self.ea.operators):
            warnings.warn("operator_weights length mismatch, resetting to uniform.")
            self.ea.operator_weights = [1.0] * len(self.ea.operators)
        if self.ea.n_parents > self.ea.pop_size or self.ea.n_parents < 2:
            warnings.warn("n_parents out of range, resetting to 2.")
            self.ea.n_parents = 2


def _coerce(data: dict[str, Any]) -> Config:
    """Build Config from a partial dict, preserving defaults."""
    return Config(
        task=TaskConfig(**(data.get("task") or {})),
        llm=LLMConfig(**(data.get("llm") or {})),
        ea=EAConfig(**(data.get("ea") or {})),
        logging=LogConfig(**(data.get("logging") or {})),
    )


def load_yaml(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}: {path}")
    return _coerce(raw)
