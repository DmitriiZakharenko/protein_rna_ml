#!/usr/bin/env python3
"""
22a_prepare_rnacompete_training.py

Build the RNAcompete subset used for Phase 3A *training* (not zero-shot benchmark).

Policy
------
  rnacompete_eukarya  — full panel (all proteins)
  rnacompete_rbpzoo   — full panel (all proteins)
  rnacompete_ucrbp    — only 23 reproducible RBPs (Ray & Laverty et al. 2023)

The full ucRBP panel (~613 human experiments) is excluded because most proteins
failed reproducibility QC in Ray & Laverty 2023. Keeping only the 23 validated
RBPs avoids training on noisy assay data.

Output
------
  data/benchmarks/rnacompete/rnacompete_training_phase3a.tsv

Usage
-----
  python scripts/22a_prepare_rnacompete_training.py
  python scripts/22a_prepare_rnacompete_training.py --benchmark_dir data/benchmarks/rnacompete
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.rnacompete_training import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_UCRBP_WHITELIST,
    load_rnacompete_training_subset,
    load_ucrbp_whitelist,
    save_training_subset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--ucrbp_whitelist", default=DEFAULT_UCRBP_WHITELIST)
    parser.add_argument(
        "--out",
        default="data/benchmarks/rnacompete/rnacompete_training_phase3a.tsv",
    )
    parser.add_argument("--no_dedup", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("  RNAcompete Training Subset Builder (Phase 3A)")
    print(f"{'='*65}")
    whitelist = load_ucrbp_whitelist(args.ucrbp_whitelist)
    print(f"  ucRBP whitelist: {len(whitelist)} proteins from {args.ucrbp_whitelist}")

    df = load_rnacompete_training_subset(args.benchmark_dir, args.ucrbp_whitelist)

    n_pos = int((df["binding_label"] == 1).sum())
    n_neg = int((df["binding_label"] == 0).sum())
    by_source = df.groupby("dataset_source").agg(
        rows=("binding_label", "size"),
        proteins=("protein_name", "nunique"),
        positives=("binding_label", lambda s: int((s == 1).sum())),
    ).to_dict("index")

    print(f"\n  Combined: {len(df):,} rows | {n_pos:,}+ {n_neg:,}- | "
          f"{df['protein_name'].nunique()} proteins")
    for src, stats in sorted(by_source.items()):
        print(f"    {src}: {stats['rows']:,} rows, {stats['proteins']} proteins, "
              f"{stats['positives']:,}+")

    df = save_training_subset(df, args.out, deduplicate=not args.no_dedup)

    summary = {
        "purpose": "Phase 3A training subset",
        "policy": {
            "eukarya": "full panel",
            "rbpzoo": "full panel",
            "ucrbp": f"filtered to {len(whitelist)} reproducible RBPs (Ray & Laverty 2023)",
        },
        "ucrbp_whitelist": sorted(whitelist),
        "total_rows": int(len(df)),
        "n_proteins": int(df["protein_name"].nunique()),
        "positives": int((df["binding_label"] == 1).sum()),
        "negatives": int((df["binding_label"] == 0).sum()),
        "by_source": {
            k: {
                "rows": int(v["rows"]),
                "proteins": int(v["proteins"]),
                "positives": int(v["positives"]),
            }
            for k, v in by_source.items()
        },
    }
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved {summary_path}")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
