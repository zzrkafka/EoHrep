"""Compute and write a diversity report for a completed run."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from eoh.analysis.similarity import ast_signature, token_jaccard


def _load_samples(run_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for path in sorted((run_dir / "results" / "samples").glob("samples_*~*.json")):
        with path.open("r", encoding="utf-8") as f:
            samples.extend(json.load(f))
    return samples


def _load_populations(run_dir: Path) -> dict[int, list[dict]]:
    pops: dict[int, list[dict]] = {}
    for path in sorted((run_dir / "results" / "pops").glob("population_generation_*.json")):
        try:
            gen = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        with path.open("r", encoding="utf-8") as f:
            pops[gen] = json.load(f)
    return pops


def _percentiles(xs: list[float], ps=(10, 50, 90)) -> dict[str, float]:
    if not xs:
        return {f"p{p}": float("nan") for p in ps}
    xs_sorted = sorted(xs)
    out = {}
    for p in ps:
        if len(xs_sorted) == 1:
            out[f"p{p}"] = xs_sorted[0]
        else:
            k = (len(xs_sorted) - 1) * (p / 100)
            f, c = int(k), min(int(k) + 1, len(xs_sorted) - 1)
            out[f"p{p}"] = xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)
    return out


def _histogram(xs: Iterable[float], bins: int = 10) -> list[dict]:
    hist = [{"lo": i / bins, "hi": (i + 1) / bins, "count": 0} for i in range(bins)]
    for x in xs:
        idx = min(int(x * bins), bins - 1)
        if idx < 0:
            idx = 0
        hist[idx]["count"] += 1
    return hist


def _summarise(xs: list[float]) -> dict[str, float | int]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "p10": float("nan"),
                "p50": float("nan"), "p90": float("nan")}
    return {"n": len(xs), "mean": statistics.mean(xs), **_percentiles(xs)}


def analyse(run_dir: Path) -> dict[str, Any]:
    samples = _load_samples(run_dir)
    pops = _load_populations(run_dir)
    succeeded = [s for s in samples if s.get("objective") is not None and s.get("code")]

    byte_codes = {s["code"] for s in succeeded}
    sigs = {ast_signature(s["code"]) for s in succeeded}
    sigs.discard(None)
    ast_dedup_rate = (
        1 - (len(sigs) / len(byte_codes)) if byte_codes else 0.0
    )

    pair_jaccs: list[float] = []
    for i in range(len(succeeded)):
        for j in range(i + 1, len(succeeded)):
            pair_jaccs.append(token_jaccard(succeeded[i]["code"], succeeded[j]["code"]))

    pj_block = {
        "n_pairs": len(pair_jaccs),
        "mean": (statistics.mean(pair_jaccs) if pair_jaccs else float("nan")),
        **_percentiles(pair_jaccs),
        "histogram": _histogram(pair_jaccs),
    }

    id_to_code: dict[str, str] = {s["id"]: s["code"] for s in samples if s.get("code")}
    by_op: dict[str, list[float]] = {}
    for s in succeeded:
        op = s.get("operator")
        pids = s.get("parent_ids") or []
        if not pids or op == "i1":
            continue
        for pid in pids:
            parent_code = id_to_code.get(pid)
            if parent_code:
                by_op.setdefault(op, []).append(token_jaccard(parent_code, s["code"]))
    per_op = {op: _summarise(xs) for op, xs in by_op.items()}

    per_gen = []
    for gen in sorted(pops):
        pop = pops[gen]
        codes = [m["code"] for m in pop if m.get("code")]
        gen_pairs = [
            token_jaccard(codes[i], codes[j])
            for i in range(len(codes)) for j in range(i + 1, len(codes))
        ]
        sigs_gen = {ast_signature(c) for c in codes}
        sigs_gen.discard(None)
        per_gen.append({
            "gen": gen,
            "size": len(pop),
            "mean_pairwise_jaccard": (
                statistics.mean(gen_pairs) if gen_pairs else float("nan")
            ),
            "ast_unique": len(sigs_gen),
        })

    failure_breakdown: dict[str, dict[str, int]] = {}
    for s in samples:
        if not s.get("failure"):
            continue
        op = s.get("operator", "?")
        failure_breakdown.setdefault(op, {})
        failure_breakdown[op][s["failure"]] = failure_breakdown[op].get(s["failure"], 0) + 1

    return {
        "total_samples": len(samples),
        "succeeded_samples": len(succeeded),
        "byte_unique": len(byte_codes),
        "ast_unique": len(sigs),
        "ast_dedup_rate": round(ast_dedup_rate, 4),
        "pairwise_jaccard": pj_block,
        "per_operator_parent_child_jaccard": per_op,
        "per_generation_population_diversity": per_gen,
        "operator_failure_breakdown": failure_breakdown,
    }


def write_report(run_dir: Path) -> Path:
    report = analyse(run_dir)
    out = run_dir / "results" / "analysis.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out
