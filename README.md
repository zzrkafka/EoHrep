# EoH-local — Local Reproduction of Evolution of Heuristics

A from-scratch reimplementation of **Evolution of Heuristics** (Liu et al., ICML 2024) driven by a **local LLM** via Ollama.

- Original paper: <https://arxiv.org/abs/2401.02051>
- Reference implementation: <https://github.com/FeiLiu36/EoH>
- This reproduction: Online Bin Packing (OBP) task only in v0.1; framework designed for extension.

---

## Why this exists

The official EoH code targets cloud LLM APIs (DeepSeek/OpenAI). This repo answers: **can we get the same algorithm behaviour on a single laptop GPU with an open-weights coder model?** That requires aligning prompts, evaluation, and selection bit-for-bit with the official implementation while substituting only the LLM backend.

The 8 GiB-VRAM laptop constraint is real (RTX 4060 Laptop). Defaults are tuned for 7B-Q4 quantised models.

---

## Quickstart

```powershell
# 1. Install Ollama (https://ollama.com), then pull a coder model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# 2. (One-off) Make sure the Ollama server is up
ollama serve            # leave running in a separate terminal

# 3. Install Python deps
uv sync                 # or: pip install -e .

# 4. Run OBP reproduction
python -m eoh.run --config configs/obp.yaml

# 5. Inspect the run
ls runs/<timestamp>/results/
cat runs/<timestamp>/results/samples/samples_best.json
```

A full OBP run is ~88 LLM calls (init `2·pop=8` + `n_pop·pop=80`). On qwen2.5-coder:7b-q4 each call takes 5–15 s decode + ≤2 s evaluation, so the whole run is ≈ 15–30 minutes wall-clock.

---

## Repository layout

```
EOH/
├── README.md                       ← this file
├── CHANGELOG.md
├── pyproject.toml
├── configs/
│   └── obp.yaml                    ← task/llm/ea config
├── data/obp/instances.py           ← Weibull 5k (vendored from official)
├── docs/                           ← formal specs (see Documentation index below)
├── notes/                          ← research notes (advisor diff, design history)
├── eoh/                            ← source tree
│   ├── run.py                      ← CLI entry
│   ├── config.py
│   ├── llm/{client,ollama}.py
│   ├── ea/{prompts,evolution,selection,loop}.py
│   ├── eval/sandbox.py
│   ├── tasks/{base,obp}.py
│   └── util/{logging,ckpt,code_extract}.py
├── tests/
└── runs/<ts>/                      ← per-run output (gitignored)
```

---

## Documentation index

For implementation details, start with:

| Read first | Then | Reference |
|---|---|---|
| [docs/01_architecture.md](docs/01_architecture.md) — modules & data flow | [docs/04_prompt_spec.md](docs/04_prompt_spec.md) — LLM contract | [notes/06_eoh_repro_design.md](notes/06_eoh_repro_design.md) — original design doc |
| [docs/03_evaluation_protocol.md](docs/03_evaluation_protocol.md) — what we score | [docs/07_test_plan.md](docs/07_test_plan.md) — how we validate | [docs/06_config_reference.md](docs/06_config_reference.md) — every YAML field |
| [docs/08_reproducibility.md](docs/08_reproducibility.md) — seeds & env | [docs/09_threat_model.md](docs/09_threat_model.md) — code-exec safety | [docs/10_glossary.md](docs/10_glossary.md) |

Architectural decisions are tracked as ADRs in `docs/adr/`.

---

## Status

| Version | State | Notes |
|---------|-------|-------|
| v0.0 | **planning** ← *current* | Design doc + specs complete; no code yet. |
| v0.1 | not started | M0–M5 milestones; OBP closes. |

See [CHANGELOG.md](CHANGELOG.md).

---

## License

MIT, following the original EoH project.
