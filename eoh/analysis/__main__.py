"""`python -m eoh.analysis <run_dir>` — write analysis.json + print summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eoh.analysis.diversity import analyse, write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eoh.analysis", description="Diversity report for a run")
    p.add_argument("run_dir", help="Path to a run directory (e.g. runs/20260528_154230)")
    p.add_argument("--json", action="store_true", help="Print full JSON instead of summary")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "results" / "samples").exists():
        print(f"Not a run directory: {run_dir} (missing results/samples)", file=sys.stderr)
        return 2

    report = analyse(run_dir)
    out_path = write_report(run_dir)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    pj = report["pairwise_jaccard"]
    print(f"run: {run_dir}")
    print(f"  samples: {report['total_samples']}  succeeded: {report['succeeded_samples']}")
    print(f"  byte-unique codes: {report['byte_unique']}   ast-unique: {report['ast_unique']}"
          f"   ast-dedup-rate: {report['ast_dedup_rate']:.1%}")
    print(f"  pairwise jaccard: n={pj['n_pairs']}  mean={pj['mean']:.3f}"
          f"  p10={pj['p10']:.3f}  p50={pj['p50']:.3f}  p90={pj['p90']:.3f}")
    print("  parent→child jaccard by operator:")
    for op in sorted(report["per_operator_parent_child_jaccard"]):
        s = report["per_operator_parent_child_jaccard"][op]
        if s["n"] > 0:
            print(f"    {op}: n={s['n']:<3d}  mean={s['mean']:.3f}  "
                  f"p10={s['p10']:.3f}  p50={s['p50']:.3f}  p90={s['p90']:.3f}")
    print("  per-gen population (size, mean pairwise jaccard, ast-unique):")
    for g in report["per_generation_population_diversity"]:
        mj = g["mean_pairwise_jaccard"]
        mj_s = f"{mj:.3f}" if mj == mj else "  N/A"   # nan check
        print(f"    gen {g['gen']:>2d}: size={g['size']}  jaccard={mj_s}  ast_unique={g['ast_unique']}")
    if report["operator_failure_breakdown"]:
        print("  failures by operator:")
        for op, fails in report["operator_failure_breakdown"].items():
            for reason, n in fails.items():
                print(f"    {op}/{reason}: {n}")
    print(f"  -> wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
