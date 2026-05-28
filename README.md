# EoH-local

A reproduction of *Evolution of Heuristics* (Liu et al., ICML 2024) on Online Bin Packing, run with a 7B local model on an 8 GiB-VRAM laptop. Paper: <https://arxiv.org/abs/2401.02051>. Reference code: <https://github.com/FeiLiu36/EoH>.

## What I went after

The straightforward goal — match the paper's numbers on OBP-Weibull-5kC100 — is unattainable here: the paper used cloud GPT-3.5 with ~2000 LLM queries; I have qwen2.5-coder-7B-Q4 and a wall-clock budget of about an hour. The interesting question is the residual one: *if I cannot match the paper, can I at least understand exactly why I cannot, and can I rule out infidelity in the framework as a cause?* That made the work less about chasing a number and more about identifying every place the reproduction could diverge from the paper, and being explicit about which divergences are forced (hardware) and which were latent (drift between the paper and the official demo).

## What I found by asking that question

The paper and the official `examples/bp_online/runEoH.py` disagree in four places. I read the paper carefully against the upstream source, and corrected each one back toward the paper:

| | Paper | `runEoH.py` (upstream) | This repo |
|---|---|---|---|
| Operator set | 5 (`i1, e1, e2, m1, m2, m3`), §3.4 | 4 (`m3` omitted) | 5 |
| `pop_size` | ~20 (inferred from §5 budget) | 4 | 20 |
| `n_parents` for `e1`/`e2` | 5 (Appendix B: "*I have five existing heuristics…*") | 2 | 5 |
| Prompt tail | "*Avoid utilizing the random component, and it is crucial to maintain self-consistency.*" (Appendix B p.15) | omitted | restored |

The fourth row is the one I am most attached to. I noticed it only because an early run reported a "new best" of 0.0394 from a heuristic that turned out to be `score(item, bins) = … + np.random.normal(0, 0.05, …)`. Across five numpy seeds that code's objective is `{0.03984, 0.03944, 0.03994, 0.03954, 0.03964}`; the run had recorded the lucky tail. Tracking down why the paper apparently avoided this failure mode, I found the randomness clause in the paper's appendix prompts and saw that `_build_prompt` in the upstream repo doesn't carry it. Putting it back eliminated the exploit and produced the first deterministic improvement over best-fit I'd seen.

## Result

OBP gap to LP lower bound (lower is better), on Weibull 5kC100:

| | Gap | Note |
|---|---|---|
| Best fit (handcrafted) | 4.08 % | paper Table 1 |
| EoH, GPT-3.5 | 0.66 % | paper Table 7 |
| EoH, CodeLlama-7B (unquantised) | 1.07 % | paper Table 7, closest in size |
| This repo, qwen2.5-coder-7B-Q4 | **3.87 %** | 440 queries, 77 min, deterministic across 5 seeds |
| Same model under the upstream-demo config | 3.98 % | does not beat best-fit |

The 0.11-point improvement (3.98 → 3.87) is small but it is the difference between "the framework is doing nothing" and "the framework is doing something". The ~3-percentage-point gap to CodeLlama-7B is what remains after I stop being able to blame the framework: it is plausibly Q4 quantisation, a different code-pretrained backbone, and ~4.5× less budget. Distinguishing those would need ablations I did not run.

Full multi-run analysis: [`notes/07_obp_repro_report.md`](notes/07_obp_repro_report.md).

## What I also noticed, that the paper does not report

Across all three large runs, the per-operator parent → child token-Jaccard distance gives a stable ordering: **m3 ≈ m2 > m1 > e2 > e1**, where higher means "child code more similar to parent". This matches the paper's verbal design intent for each operator (`m3` simplifies, `e1` is *"totally different form"*), but the paper only states the intent. Seeing it land empirically — and survive a 5× change in run size — was reassuring; it argues the operator semantics are real and not prompt-noise.

## What I built on top of upstream to enable this

The upstream code records `(sample_order, operator, algorithm, code, objective)` per sample and collapses every evaluation failure to `objective=None`. I would not have diagnosed the random-noise exploit on that schema. So before the second run I added: structured failure classification (`error_type` + full traceback survives the sandbox subprocess), 8-character lineage IDs and `parent_ids` per sample, per-sample LLM and eval timings, raw prompt + response logs in `results/prompts/sample_<id>.json`, an `env.json` per run with the Ollama model's sha256 digest, and an offline `python -m eoh.analysis <run_dir>` that produces the operator-Jaccard table above. Plus 36 tests, all of which would have caught the bugs I introduced while refactoring. The algorithmic behaviour mirrors upstream byte-for-byte; the additions are observational.

## Quickstart

```powershell
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama serve

pip install -e .
python -m eoh.run --config configs/obp.yaml          # GitHub-demo config,  ~13 min
python -m eoh.run --config configs/obp_aligned.yaml  # paper-aligned,       ~75 min
python -m eoh.analysis runs/<timestamp>
pytest
```

## Scope

Single task (OBP), single config-pair, single run per config (`seed=2024`). The paper covers OBP + TSP-GLS + FSSP with 3 runs each. Adding TSP-GLS is mechanical (`tasks/base.py:BaseProblem` is task-agnostic); larger LLM ablations and the paper's OOD evaluation set (Table 9) are the obvious next steps.

## License

MIT, following upstream.
