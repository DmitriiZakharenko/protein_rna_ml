#!/usr/bin/env python3
"""
54_aggregate_probe.py
---------------------
Aggregate multi-seed results of scripts/51 (models named ``v2_s<seed>`` /
``v4_s<seed>``, ``__full`` aggregation) into mean ± std tables, plus the paired
V4 − V2 difference per seed.

Usage
-----
  python scripts/54_aggregate_probe.py \\
      --summary results/overnight/probe_final/step1_summary.json --split cross_neg_probe
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("auroc", "auprc", "pp_median_auroc", "protein_dependence_dAUROC")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--split", default="cross_neg_probe")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--agg", default="full")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    path = Path(args.summary)
    path = path if path.is_absolute() else ROOT / path
    scores = json.loads(path.read_text())["splits"][args.split]["scores"]

    runs: dict[str, dict[int, dict]] = {}
    for key, table in scores.items():
        m = re.fullmatch(rf"(v[24])_s(\d+)__{args.agg}", key)
        if m and args.subset in table:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = table[args.subset]

    out: dict = {"split": args.split, "subset": args.subset, "models": {}}
    print(f"{'model':<6}{'n_seeds':>8}  " + "  ".join(f"{k:>26}" for k in METRICS))
    for model, by_seed in sorted(runs.items()):
        row = {"seeds": sorted(by_seed)}
        cells = []
        for k in METRICS:
            vals = [v[k] for v in by_seed.values() if v.get(k) is not None]
            if vals:
                row[k] = {"mean": round(float(np.mean(vals)), 4),
                          "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
                          "per_seed": {s: by_seed[s].get(k) for s in sorted(by_seed)}}
                sd = f"{row[k]['std']:.3f}" if row[k]["std"] is not None else "n/a"
                cells.append(f"{row[k]['mean']:.4f} ± {sd}")
            else:
                cells.append("n/a")
        out["models"][model] = row
        print(f"{model:<6}{len(by_seed):>8}  " + "  ".join(f"{c:>26}" for c in cells))

    if "v2" in runs and "v4" in runs:
        common = sorted(set(runs["v2"]) & set(runs["v4"]))
        diffs = [runs["v4"][s]["auroc"] - runs["v2"][s]["auroc"] for s in common
                 if runs["v4"][s].get("auroc") is not None and runs["v2"][s].get("auroc") is not None]
        if diffs:
            out["v4_minus_v2_auroc"] = {"per_seed": dict(zip(common, map(lambda d: round(d, 4), diffs))),
                                        "mean": round(float(np.mean(diffs)), 4)}
            print(f"\nV4 - V2 AUROC per seed {common}: {[round(d, 4) for d in diffs]}  mean={np.mean(diffs):.4f}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
