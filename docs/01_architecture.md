# 01 — Architecture

Companion to [notes/06_eoh_repro_design.md](../notes/06_eoh_repro_design.md) §3–4. That doc states the *contracts*; this doc shows the *shape* — module dependency graph, data flow, and one-generation sequence diagram. Code-level details remain in the design doc.

---

## Module dependency graph

```
                ┌───────────┐
                │   run.py  │   CLI entry
                └─────┬─────┘
                      │ loads
                      ▼
                ┌───────────┐
                │  config   │
                └─────┬─────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌──────────┐  ┌──────────┐
   │   llm   │  │    ea    │  │   eval   │
   │ (ollama)│  │   loop   │  │ sandbox  │
   └────┬────┘  └─────┬────┘  └────┬─────┘
        │             │            │
        │       ┌─────┴─────┐      │
        │       ▼           ▼      │
        │  ┌────────┐  ┌─────────┐ │
        └─▶│evolution│ │selection│ │
           └────┬───┘  └─────────┘ │
                │                  │
                └───────┬──────────┘
                        ▼
                  ┌──────────┐
                  │  tasks   │  BaseProblem + BPONLINE
                  └──────────┘
                        ▲
                        │ reads
                  ┌─────┴────┐
                  │   data   │   Weibull 5k instances
                  └──────────┘
```

Hard rules:

- `llm/` knows nothing of EA. `eval/` knows nothing of LLM. `tasks/` knows nothing of EA or LLM. This is enforced by import order — circular imports indicate a contract violation.
- `ea/` is the only layer that imports both `llm/` and `eval/`.
- `util/` is leaf — no upward imports.

---

## Process model

```
   main process                         worker subprocess (spawn)
   ┌───────────────────────┐            ┌──────────────────────────┐
   │ ea.loop.run()         │            │ eval._eval_worker()      │
   │   ↓                   │            │   exec(code, ns)         │
   │ ea.evolution          │            │   ns['score'](item, bins)│
   │   ↓ Ollama HTTP        │   spawn   │   problem.evaluate_program│
   │ qwen2.5-coder:7b      │ ────────▶  │   → objective            │
   │   ↓ extract code      │ Queue.put  │                          │
   │ eval._eval_with_timeout│ ◀────────  │                          │
   │   ↓ aggregate          │            └──────────────────────────┘
   │ ea.selection           │
   │   ↓ checkpoint         │
   │ util.ckpt              │
   └───────────────────────┘
```

- One subprocess per evaluation. Lifetime ≤ `task.timeout` (default 40 s). Killed via `p.terminate()` on timeout.
- `spawn` is used unconditionally — Windows requires it, and it gives a clean namespace on Unix too (important: LLM-generated code may pollute globals).
- Ollama runs as a separate OS daemon (`ollama serve`), unmanaged by us. We only HTTP-talk to it.

---

## One generation — sequence diagram

```
Loop                  Evolution             Selection             Ollama            Sandbox

  │── selected_ops ────▶│                                                              
  │   = random.choices()│                                                              
  │                     │── parents = parent_selection(pop, m) ─▶│                     
  │                     │◀──────── parents ─────────────────────│                     
  │                     │                                                              
  │                     │── build_prompt(op, parents)                                  
  │                     │── get_response ──────────────────────────▶│                  
  │                     │◀── raw_text ───────────────────────────── │                  
  │                     │── _extract ──▶ (code, algorithm)                             
  │                     │                                                              
  │                     │── eval_with_timeout(code) ────────────────────▶│             
  │                     │                                                  │ spawn     
  │                     │                                                  │  Process  
  │                     │◀── objective ───────────────────────────────────│             
  │                     │                                                              
  │◀── offspring ───────│                                                              
  │                     │                                                              
  │── pop += offspring                                                                 
  │── pop = population_management(pop, size) ─▶│                                       
  │◀──────────────── trimmed pop ──────────────│                                       
  │                                                                                    
  │── checkpoint                                                                       
```

Per generation: `pop_size` independent (parents → LLM → extract → sandbox → offspring) pipelines. In v0.1 these run sequentially.

---

## Concurrency boundaries

| Boundary | v0.1 | v0.2 candidate |
|----------|------|----------------|
| Within one offspring (LLM + eval) | Sequential — eval starts only after extract succeeds | Same |
| Across offspring in one generation | **Sequential** — `[get_offspring(...) for op in selected]` | `ProcessPoolExecutor`, each worker holds its own Ollama session — limited by VRAM (one model instance can't be safely concurrent with itself) |
| Across generations | Sequential by definition (selection depends on previous gen) | Same |

For the local-GPU + 7B-Q4 setting, the wall-clock bottleneck is **LLM decoding** (~5–15 s per call). Parallelism via process pool would require multiple Ollama instances pinned to different GPUs — we have one GPU, so v0.1's sync model is the realistic ceiling.

---

## Failure modes & recovery

| Failure | Where | Recovery |
|---------|-------|----------|
| LLM returns garbage (no code block, no `{...}`) | `_extract` | Retry up to 4× in `_call_llm` |
| LLM returns code duplicate of existing pop member | `get_offspring` | Retry up to 2× with same operator |
| LLM HTTP timeout (>150 s) | Ollama client | Retry counts as 1 of 4; if all fail → offspring = `(None, None)` |
| `exec(code)` raises | Sandbox subprocess | Caught in `_eval_worker`, `queue.put(None)` |
| `score()` raises during eval | Sandbox subprocess | Same |
| Sandbox infinite loop / OOM | Main proc | `p.join(timeout)` → `terminate()` → `objective = None` |
| Offspring = `None` after all retries | Main loop | Logged with `score_str = "None (generation failed)"`, **not added to pop** |

Crucially: `population_management` filters out `None` *and* deduplicates by objective — so failed and identical offspring never poison selection.

---

## State, persistence, resumption

- In-memory state: `pop: list[dict]`, `_sample_count`, `_best_obj`, `_samples_buffer`.
- Persisted every 200 samples → `results/samples/samples_<lo>~<hi>.json`.
- Persisted every generation → `results/pops/population_generation_<g>.json`.
- New best → `results/samples/samples_best.json` (overwritten).
- **Resume**: `config.use_continue = True; continue_path = "<...>/population_generation_<g>.json"; continue_id = g` re-enters at gen g+1.

Resume is intentionally coarse — per-generation only. We accept up to one generation of recomputation on crash; per-sample resume would complicate sample numbering.

---

## What this architecture **cannot** do (yet)

- **Multi-task evolution**: one `Loop` instance binds to one `Task`. Cross-task transfer (à la MoH) requires a meta-loop above this — out of scope for v0.1.
- **Distributed evaluation**: no message broker. A v0.2 with `ProcessPoolExecutor` is local-only.
- **Online prompt mutation**: prompts are hard-coded constants in `ea/prompts.py`. ReEvo's reflection would require a new mutable "Reflector state" passed alongside `pop`.
- **Model hot-swap mid-run**: a run is tied to one LLM. To compare backends, run twice and diff `samples_*.json`.
