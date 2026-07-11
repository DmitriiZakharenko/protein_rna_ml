#!/usr/bin/env python3
"""
31_build_external_benchmark.py

Build an expanded literature external benchmark by preserving curated positives and
negatives from ``dataset_without_affinities.xlsx`` and adding generated negatives
for every curated positive.

Negative generation strategies (see ``src/data/negative_sampling.py``):
  - shuffle_uniform       : composition-preserving random permutation
  - shuffle_dinucleotide  : dinucleotide-count-preserving Eulerian shuffle
  - cross_protein         : anchor RNA + non-cognate protein from another positive
  - cross_rna             : anchor protein + non-cognate RNA from another positive

Curated literature negatives are never modified or replaced.

Usage (from protein_rna_ml/):
    python scripts/31_build_external_benchmark.py

    python scripts/31_build_external_benchmark.py \\
        --xlsx data/external/dataset_without_affinities.xlsx \\
        --train_tsv data/generalized_v3a/train.tsv \\
        --out_dir data/external \\
        --seed 42

    # Custom per-strategy counts (default: 1 of each per positive)
    python scripts/31_build_external_benchmark.py \\
        --n_shuffle_uniform 2 \\
        --n_shuffle_dinucleotide 2 \\
        --n_cross_protein 1 \\
        --n_cross_rna 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.external_benchmark import (
    build_expanded_benchmark,
    load_external_xlsx,
    parse_curated_pairs,
    save_benchmark_outputs,
    TrainLeakageIndex,
)
from src.data.negative_sampling import NegativeSamplingConfig


def _default_xlsx() -> str | None:
    for candidate in (
        "data/external/dataset_without_affinities.xlsx",
        "dataset without affinities.xlsx",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand literature external benchmark with generated negatives.",
    )
    parser.add_argument(
        "--xlsx",
        default=None,
        help="Path to dataset_without_affinities.xlsx (auto-detected if omitted).",
    )
    parser.add_argument(
        "--train_tsv",
        default="data/generalized_v3a/train.tsv",
        help="Training TSV for leakage annotation (set empty string to skip).",
    )
    parser.add_argument("--out_dir", default="data/external")
    parser.add_argument("--basename", default="external_benchmark_expanded")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_shuffle_uniform", type=int, default=1)
    parser.add_argument("--n_shuffle_dinucleotide", type=int, default=1)
    parser.add_argument("--n_cross_protein", type=int, default=1)
    parser.add_argument("--n_cross_rna", type=int, default=1)

    parser.add_argument("--min_hamming_absolute", type=int, default=3)
    parser.add_argument("--min_hamming_fraction", type=float, default=0.05)
    parser.add_argument("--max_attempts_per_draw", type=int, default=500)
    parser.add_argument("--min_protein_jaccard_distance", type=float, default=0.05)
    parser.add_argument("--min_rna_jaccard_distance", type=float, default=0.05)

    args = parser.parse_args()

    xlsx_path = args.xlsx or _default_xlsx()
    if not xlsx_path or not os.path.exists(xlsx_path):
        print("ERROR: Could not find external xlsx. Pass --xlsx explicitly.")
        sys.exit(1)

    cfg = NegativeSamplingConfig(
        seed=args.seed,
        n_shuffle_uniform=args.n_shuffle_uniform,
        n_shuffle_dinucleotide=args.n_shuffle_dinucleotide,
        n_cross_protein=args.n_cross_protein,
        n_cross_rna=args.n_cross_rna,
        min_hamming_absolute=args.min_hamming_absolute,
        min_hamming_fraction=args.min_hamming_fraction,
        max_attempts_per_draw=args.max_attempts_per_draw,
        min_protein_jaccard_distance=args.min_protein_jaccard_distance,
        min_rna_jaccard_distance=args.min_rna_jaccard_distance,
    )

    print(f"\n=== Loading curated external dataset: {xlsx_path} ===")
    df_raw = load_external_xlsx(xlsx_path)
    curated, parse_stats, cols = parse_curated_pairs(df_raw)

    print(f"  Raw rows:                 {parse_stats.raw_rows}")
    print(f"  Skipped (ambiguous label): {parse_stats.skipped_ambiguous_label}")
    print(f"  Skipped (missing seq):     {parse_stats.skipped_missing_sequence}")
    print(f"  Skipped (invalid RNA):     {parse_stats.skipped_invalid_rna}")
    print(f"  Skipped (invalid protein): {parse_stats.skipped_invalid_protein}")
    print(f"  Usable curated pairs:      {parse_stats.usable_rows}")
    print(f"    positives: {sum(p.binding_label == 1 for p in curated)}")
    print(f"    negatives: {sum(p.binding_label == 0 for p in curated)}")
    print(f"  Columns: {cols}")

    train_index = None
    if args.train_tsv and os.path.exists(args.train_tsv):
        print(f"\n=== Building train leakage index: {args.train_tsv} ===")
        train_index = TrainLeakageIndex.from_train_tsv(args.train_tsv)
        print(f"  Train rows indexed: {train_index.n_train_rows:,}")
        print(f"  Unique train proteins: {len(train_index.train_protein_names):,}")
    elif args.train_tsv:
        print(f"\n  WARNING: train_tsv not found ({args.train_tsv}); skipping leakage checks.")

    print("\n=== Generating negatives ===")
    print(f"  Per positive: shuffle_uniform={cfg.n_shuffle_uniform}, "
          f"shuffle_dinucleotide={cfg.n_shuffle_dinucleotide}, "
          f"cross_protein={cfg.n_cross_protein}, cross_rna={cfg.n_cross_rna}")
    print(f"  Hamming threshold: max({cfg.min_hamming_absolute}, "
          f"ceil(len * {cfg.min_hamming_fraction}))")
    print(f"  Seed: {cfg.seed}")

    df, report = build_expanded_benchmark(curated, cfg, train_index)
    report.parse_stats = parse_stats

    tsv_path, json_path = save_benchmark_outputs(
        df, report, args.out_dir, basename=args.basename,
    )

    pos = int((df["binding_label"] == 1).sum())
    neg = int((df["binding_label"] == 0).sum())

    print(f"\n{'=' * 60}")
    print("  EXTERNAL BENCHMARK EXPANSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total pairs:           {len(df)}")
    print(f"    curated positive:    {report.n_curated_positive}")
    print(f"    curated negative:    {report.n_curated_negative}")
    print(f"    generated negative:  {report.n_generated_negative}")
    print(f"  Class balance:         {pos} pos / {neg} neg ({pos / max(len(df), 1):.1%} positive)")
    print(f"  Proteins both classes: {report.proteins_with_both_classes_after}")
    print(f"  Proteins single class: {report.single_class_proteins_after}")
    print(f"  Neg strategy counts:   {report.neg_strategy_counts}")
    if report.generation_failures:
        print(f"  Generation failures:   {report.generation_failures}")
    if report.train_leakage_summary:
        print(f"  Train leakage flags:   {report.train_leakage_summary}")
    print(f"\n  TSV      → {tsv_path}")
    print(f"  Manifest → {json_path}")
    print(f"\n  Evaluate with:")
    print(f"    python scripts/11_evaluate_external.py \\")
    print(f"      --benchmark_tsv {tsv_path} \\")
    print(f"      --v2_dir models/saved/generalized_v2 \\")
    print(f"      --prot_max 700 --no_cuda")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
