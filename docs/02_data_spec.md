# 02 — Data Specification

Defines exactly which instances we evaluate on, so that any divergence from official EoH numbers is *not* attributable to data drift.

---

## Source of truth

The OBP instance set is vendored verbatim from the official repository:

```
upstream:  examples/bp_online/get_instance.py        (FeiLiu36/EoH @ main)
local:     data/obp/instances.py                     (byte-identical copy)
```

We do **not** regenerate instances. The data lives in the source file as a Python literal — no external download, no random sampling at load time.

> **Implementation note**: if the file is very large, prefer mirroring it as a pickle (`data/obp/instances.pkl`) generated once by re-executing the upstream module. Either way, the canonical hash must match.

---

## Schema

`GetData().get_instances(capacity)` returns `(instances, lower_bounds)` where:

```python
instances: dict[str, dict[str, Instance]] = {
    "Weibull 5k": {
        "test_0": {"capacity": 1000, "num_items": 5000, "items": [48, 66, 48, 32, 72, ...]},
        "test_1": {...},
        "test_2": {...},
        "test_3": {...},
        "test_4": {...},
    },
}

lower_bounds: dict[str, float] = {
    "Weibull 5k": <float>,            # see §3 for definition
}
```

- **Outer key**: dataset name. v0.1 has exactly one (`Weibull 5k`).
- **Inner key**: instance ID. Five instances per dataset.
- **`capacity`**: per-instance bin capacity. For Weibull 5k, all five instances have `capacity = 1000`.
- **`num_items`**: number of items in the stream. All Weibull 5k instances have `num_items = 5000`.
- **`items`**: list of integers — sizes of the items, in arrival order. Order is significant (this is *online* bin packing).

---

## Item distribution

Items are sampled from a **Weibull distribution** then quantised to integers. The exact generator script is not bundled in the upstream repo; the literal values are. We do not need to regenerate — but the distribution shape is known from the FunSearch (Romera-Paredes et al., 2024) paper:

- Shape parameter `k ≈ 3.0`
- Scale parameter `λ ≈ 45`
- Truncated/clipped to fit in `[1, capacity]`

Empirically the Weibull 5k items observed in `get_instance.py` are mostly in `[10, 90]` with mean ≈ 40 and capacity 1000 — so each instance fills ~200 bins under best-fit.

---

## Lower bounds

The `lb` dict ships precomputed in the upstream file. Conceptually,

```
lb["Weibull 5k"] = mean over instances of (sum(items) / capacity)
```

i.e. the continuous-relaxation lower bound on bins-used (LP relaxation of bin packing). It is **per-dataset average**, not per-instance.

This becomes the denominator in the fitness formula — see [03_evaluation_protocol.md](03_evaluation_protocol.md).

---

## Train / validation split

The official EoH does **not** split instances. All 5 instances of Weibull 5k are used both as the evolutionary search signal and as the reported number.

We follow this for fidelity. Risk noted: nothing stops the evolved heuristic from overfitting to these 5 instances. Future work (v0.2+) should add a held-out test set, e.g. fresh Weibull samples or the FunSearch `Weibull 10k` set.

> If a held-out check is added later: keep search on Weibull 5k, evaluate the final best on a separate `Weibull 10k_ood` and report both numbers.

---

## Adding new tasks (forward-looking)

A task's data spec is fully determined by the `BaseProblem` subclass — there is no external data-registry component. To add e.g. TSP:

1. Drop instances in `data/tsp/instances.py` (or `.pkl`).
2. Implement `TSPGLS(BaseProblem)` in `eoh/tasks/tsp_gls.py` with its own `evaluate_program`.
3. Reference it in a new YAML config.

The EA loop is data-agnostic.

---

## Determinism guarantees

| Element | Deterministic? | Why |
|---------|----------------|-----|
| Item order in each instance | Yes | Fixed list literal |
| Instance ordering within a dataset | Yes | `dict` preserves insertion order in Python 3.7+ |
| Lower-bound values | Yes | Precomputed constants |
| `evaluate_program` over fixed `code` | Yes | No RNG in scoring |
| The fitness number for `def score(item, bins): return -bins` | Yes (independent of run) | Use as canary in tests — see [07_test_plan.md](07_test_plan.md) |
