# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html), but pre-1.0 minor bumps may break interfaces.

---

## [Unreleased]

Planned for v0.1:

- M0  Scaffolding: CLI entry, dataclass config, stub LLM client. `--dry-run` passes.
- M1  Ollama client + `i1` operator end-to-end. Extract success rate logged.
- M2  Sandbox + OBP evaluator. Known heuristics (`-bins`, `bins-item`) score correctly.
- M3  One full generation loop (init + 1 gen). Population management verified.
- M4  Full 20-generation run. Best-objective curve monotonically non-increasing.
- M5  Numerical alignment with official EoH on OBP-Weibull-5k within 5 % relative gap.

---

## [0.0.0] — 2026-05-28

Initial planning artefacts; no executable code yet.

### Added
- Design document `notes/06_eoh_repro_design.md` (v0.2 — aligned to official source).
- Full SE doc set under `docs/`: architecture, data, evaluation, prompts, logging, config, testing, reproducibility, threat model, glossary, six ADRs.
- Hardware probe (RTX 4060 Laptop, 8 GiB VRAM); model choice locked to 7B-Q4 default.

### Decisions
- Backend: Ollama (see `docs/adr/0001-llm-backend-ollama.md`).
- Sandbox: spawn subprocess + hard timeout (see `docs/adr/0002-eval-sandbox-spawn.md`).
- Config: YAML + dataclass (see `docs/adr/0003-config-yaml-dataclass.md`).
- Execution: synchronous (see `docs/adr/0004-execution-synchronous.md`).
- Prompts: verbatim mirror of official strings (see `docs/adr/0005-prompts-verbatim-mirror.md`).
- Task abstraction: `BaseProblem` Protocol (see `docs/adr/0006-task-protocol.md`).
