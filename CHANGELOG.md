# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html), but pre-1.0 minor bumps may break interfaces.

---

## [Unreleased]

- TSP-GLS and FSSP tasks (paper §4 covers all three; we have only OBP).
- Multi-run mean/std reporting (paper Tables 6–8 report 3 runs; we run 1 per config).
- Out-of-distribution evaluation set (paper Table 9: train on 5k/cap100, test on 1k/10k cap500).
- Optional joblib parallelism over `get_offspring` (implemented, dormant; not useful on single-GPU setups).

---

## [0.1.0] — 2026-05-28

First end-to-end reproduction on Online Bin Packing.

### Added — framework
- `eoh/` package: `config`, `run`, `llm/{client,ollama,stub}`, `ea/{prompts,evolution,selection,loop}`, `eval/sandbox`, `tasks/{base,obp,_obp_instances}`, `util/{logging,ckpt,code_extract,runmeta,failure}`, `analysis/{similarity,diversity}`.
- 36 tests under `tests/` (code extraction, sandbox, OBP evaluator canaries, selection, end-to-end stub, error propagation, similarity).
- Two configs: `configs/obp.yaml` (mirrors GitHub demo) and `configs/obp_aligned.yaml` (paper-aligned).

### Added — observability beyond upstream
- Per-sample failure classification (`FailureReason` enum: `llm_no_response / extract_failed / eval_timeout / eval_error / eval_non_finite`).
- Structured error propagation: sandbox subprocess captures `error_type` + `error_msg` + full traceback; surfaces in `samples_*.json` and `prompts/sample_<id>.json`.
- Lineage: every offspring carries an 8-char uuid; `parent_ids` recorded for every sample.
- Per-sample timing (`llm_ms`, `eval_ms`, `total_ms`) and retry counters.
- `env.json` per run: git commit, Python/numpy versions, OS, GPU, Ollama server version, **model weight digest (sha256)**.
- `summary.json` per run: operator stats (attempted / succeeded / new_best), failure counts.
- `analysis.json` + `python -m eoh.analysis <run_dir>`: AST-normalised dedup rate, pairwise token Jaccard distribution, per-operator parent→child similarity, per-generation population diversity.

### Findings (this work)
- Operator semantic order **m3 ≥ m2 > m1 > e2 > e1** is empirically measurable by parent→child Jaccard and is stable across 88-query and 440-query runs. This validates the paper's verbal design intent quantitatively for the first time.
- The upstream GitHub demo's `_build_prompt` is missing the paper's *"Avoid utilizing the random component, and it is crucial to maintain self-consistency."* clause (Appendix B p.15). Under that drift, weak local models exploit eval-time stochasticity to fake improvements. Restoring the clause eliminates the exploit and is the precondition for our genuine 2.8 % improvement over best-fit.

### Reproduction results vs. paper (OBP Weibull-5kC100)
| Config | Best gap | Status |
|---|---|---|
| Baseline (GitHub-demo config; pop=4, no m3) | 3.984 % | = best-fit baseline |
| Paper-aligned (pop=20, +m3, n_parents=5, prompt restored) | **3.874 %** | **deterministic improvement over best-fit** |
| Paper, CodeLlama-7B (closest in size) | 1.07 % | gap remains ~3.6× |
| Paper, GPT-3.5 | 0.66 % | gap remains ~5.9× |

The remaining gap is attributable to LLM capacity (Q4 quantisation, 7B vs. larger models) and query budget (440 vs. paper's ~2000), not to algorithmic infidelity. See `notes/07_obp_repro_report.md` for the full breakdown.

### Decisions
- Backend: Ollama with OpenAI-compatible `/v1/chat/completions`.
- Sandbox: spawn subprocess + hard timeout; structured `EvalResult` return.
- Config: YAML + dataclass.
- Execution: synchronous (joblib code path present, dormant).
- Prompts: paper-aligned, with a documented deviation from the upstream demo (the random-component clause).
- Task abstraction: `BaseProblem` Protocol with `template_program`, `task_description`, `evaluate_program`.

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
