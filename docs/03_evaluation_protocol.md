# 03 — Evaluation Protocol

How a candidate heuristic gets a number. Aligned byte-for-byte with `examples/bp_online/prob.py:evaluate_program` in the official repo.

---

## Goal

A `score(item, bins) -> np.ndarray` function packs items by always picking the bin with the highest score among **feasible** bins (those with `remaining ≥ item`). We want to minimise the number of bins used. We report a **relative gap to lower bound**.

---

## Formula (verbatim semantics)

For each candidate `code`:

```python
fitness_per_dataset = []                                   # one element per dataset name
for name, dataset in instances.items():                    # v0.1: dataset name = "Weibull 5k"
    num_bins_list = []
    for instance_id, instance in dataset.items():
        capacity = instance["capacity"]                    # 1000 for Weibull 5k
        items    = np.array(instance["items"])             # length 5000
        bins     = np.array([capacity] * instance["num_items"])   # 5000 spare bins, all empty
        _, bins_after = online_binpack(items, bins, score_func)
        num_bins_list.append(-(bins_after != capacity).sum())     # negative count of "used" bins

    avg = -np.mean(num_bins_list)                          # mean bins used across 5 instances (positive)
    fitness_per_dataset.append((avg - lb[name]) / lb[name])  # relative gap to LB

objective = float(np.mean(fitness_per_dataset))           # mean over datasets — v0.1 has just one
```

Where `online_binpack` does the actual packing:

```python
def online_binpack(items, bins, score_func):
    packing = [[] for _ in bins]
    for item in items:
        valid = np.nonzero((bins - item) >= 0)[0]
        priorities = score_func(item, bins[valid])
        best = valid[np.argmax(priorities)]
        bins[best] -= item
        packing[best].append(item)
    return packing, bins
```

---

## Direction

**Lower is better.** `objective` is the relative gap `(bins_used − lb) / lb` ≥ 0. Zero would mean we hit the LP lower bound exactly (unachievable in practice for integer bin packing).

Selection in `population_management` uses `heapq.nsmallest` — first-class encoding of lower-is-better.

---

## Bin-counting subtlety

The line `(bins_after != capacity).sum()` counts *non-empty* bins. The trick: we pre-allocate `num_items` empty bins (which is always ≥ the number actually needed); after packing, any bin still at full capacity was never touched. The count of touched bins = bins used.

The negation `-(...)` in the inner list, then another negation `-np.mean(...)`, are present in the upstream code; the double-negate cancels to "mean bins used (positive)". Keeping the double-negation in our implementation makes diffs against upstream easier.

---

## Constraint on `score()` outputs

- Must return a numpy array shape-equal to its `bins` input.
- Length-0 input is unreachable (online_binpack guarantees at least one feasible bin because we pre-allocated enough empty bins).
- `np.argmax` ties are broken by index (first occurrence wins) — deterministic for fixed input.

If `score()` raises, returns the wrong shape, or returns NaN, `evaluate_program` itself does not catch — the exception propagates to the sandbox subprocess wrapper, which converts it to `objective = None` (= "evaluation failed").

---

## Baselines (canary heuristics for testing)

| Function | Heuristic name | Expected behaviour |
|----------|---------------|--------------------|
| `lambda i, b: -b` | best-fit (tightest remaining) | Standard strong baseline |
| `lambda i, b:  b` | worst-fit (largest remaining) | Standard weak baseline |
| `lambda i, b:  np.ones_like(b)` | first-fit (always pick index 0 of feasible) | Very weak baseline |
| `lambda i, b: -(b - i)` | "least leftover" (= best-fit, mathematically) | Variant of best-fit |

We use the first three as **numerical canaries** in `tests/test_evaluator.py` — their objectives are deterministic and must match the values we record once on first successful run. Any change in `evaluate_program` that perturbs these numbers is a regression.

> The actual numbers will be filled in `tests/golden_objectives.json` after the first manual evaluator run, then frozen.

---

## Expected target numbers (EoH paper)

The original paper reports OBP results in Table-OBP. Numbers we should ballpark match (lower-is-better, % gap):

| Method | Weibull 5k gap (%) |
|--------|---|
| First-fit | ≈ 5 % (reference) |
| Best-fit | ≈ 4 % (reference) |
| FunSearch (best) | ≈ 1 % |
| **EoH (paper, best)** | **≈ 0.5–1 %** (target for our reproduction) |

**v0.1 acceptance**: best objective in our `samples_best.json` ≤ 1.05× the EoH paper number. Cited via paper Table; values to be transcribed during M5 from the paper PDF / repo logs.

---

## Failure semantics

`evaluate_program` itself does not retry, does not catch, does not log. All robustness lives one layer up in `eval/sandbox.py`:

| Inside sandbox subprocess | Outside (main proc) |
|---|---|
| Code raises during exec | `queue.put(None)` |
| `score()` raises | `queue.put(None)` |
| Numpy / runtime error | `queue.put(None)` |
| Wall-clock > `timeout` | `p.terminate()` → main reads None |
| Result returned | `queue.put(float)` |

The main loop receives either `float` or `None` from `eval_with_timeout`. `None` → individual discarded by `population_management`.

---

## What we deliberately do NOT measure (yet)

- **Wall-clock per heuristic**: heuristic runtime varies but is not part of the fitness — we accept that LLM may produce slow heuristics so long as they're under the 40 s sandbox timeout.
- **Memory used by the heuristic**: not measured. A Python-level OOM would be killed by the OS; the subprocess returns None.
- **Out-of-distribution generalisation**: the `m3` operator targets this conceptually, but we don't have a separate OOD dataset in v0.1. Adding `Weibull 10k_ood` is a v0.2 task.
- **Wall-clock per *run***: tracked in `run_log.txt` for diagnostics, not as part of the comparison metric.
