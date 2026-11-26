#!/usr/bin/env python3
"""
17_prepare_rnacompete_benchmark.py

DECISION: RNAcompete is used as Option B — an INDEPENDENT BENCHMARK ONLY.

Rationale (documented at bottom of this script and in DATA.md):
  - 13.9M pairs across 1,087 experiments / 26 organisms — far larger than training data
  - Proteins are construct sequences (RNA-binding domains), NOT full-length; different
    from the full-length sequences used in HTR-SELEX / RBNS training set
  - RNA probes are short (35–41 nt) synthetic oligomers — different distribution from
    HTR-SELEX ~40 nt enriched sequences (similar length but fully synthetic)
  - Organisms span 26 eukaryotes; merging would severely contaminate HTR-SELEX/RBNS
    protein-aware splits with paralogs from divergent species
  - Using it as benchmark gives a ZERO-SHOT generalization score across:
      • unseen organisms
      • unseen protein families (non-human RBPs from diverse eukaryotes)
      • different assay technology (RNAcompete vs SELEX/RBNS)
  - This is structurally equivalent to how ZHMolGraph's hard split (unseen proteins + RNAs)
    works, and directly measures motif-learning ability vs sequence memorization

Benchmark subsets prepared:
  1. rnacompete_all.tsv     — full combined benchmark (~13.9M pairs, subsampled to 2M)
  2. rnacompete_human.tsv   — human-only RBPs from ucRBP (most directly comparable to training)
  3. rnacompete_nonhuman.tsv— non-human eukaryotes (true zero-shot)
  4. rnacompete_rbpzoo.tsv  — RBPZoo sub-dataset only (cleanest negatives, Sasse et al. 2025)

Usage:
  python scripts/17_prepare_rnacompete_benchmark.py \
      --rnacompete_dir /path/to/rnacompete_analysis \
      --output_dir data/benchmarks/rnacompete \
      [--max_pairs 2000000] \
      [--seed 42]
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Schema mapping ─────────────────────────────────────────────────────────
# RNAcompete columns  → project schema columns
COLUMN_MAP = {
    "rna_sequence":    "rna_sequence",
    "binding_label":   "binding_label",
    "probe_intensity": "probe_intensity",    # extra diagnostic column
    "hyb_id":          "experiment_id",
    "target_name":     "protein_name",
    "protein_sequence":"protein_sequence",
    "organism":        "organism",
    "probe_id":        "probe_id",
    "probe_set":       "probe_set",
}

PROJECT_SCHEMA = [
    "rna_sequence", "protein_sequence", "protein_name",
    "binding_label", "experiment_id", "organism",
    "probe_id", "probe_set", "probe_intensity", "dataset_source",
]

SUB_DATASETS = {
    "rbpzoo":  "rbpzoo/results/ml_dataset_rbpzoo_clean.tsv.gz",
    "eukarya": "eukarya/results/ml_dataset_eukarya_clean.tsv.gz",
    "ucrbp":   "ucrbp/results/ml_dataset_ucrbp_clean.tsv.gz",
}

# Organisms considered "human-equivalent" (for human subset)
HUMAN_ORGANISMS = {"Homo sapiens", "Mus musculus"}


# ── Loader ─────────────────────────────────────────────────────────────────

def load_subdataset(base_dir: str, sub_key: str, path: str) -> pd.DataFrame | None:
    full_path = os.path.join(base_dir, path)
    if not os.path.exists(full_path):
        print(f"  [SKIP] {sub_key}: not found at {full_path}", file=sys.stderr)
        return None
    print(f"  [LOAD] {sub_key}: {full_path}")
    try:
        df = pd.read_csv(full_path, sep="\t", compression="gzip", low_memory=False)
    except Exception as e:
        print(f"  [ERROR] {sub_key}: {e}", file=sys.stderr)
        return None
    df["dataset_source"] = f"rnacompete_{sub_key}"
    df = df.rename(columns=COLUMN_MAP)
    return df


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    for col in PROJECT_SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[PROJECT_SCHEMA].copy()
    df["binding_label"] = df["binding_label"].astype(int)
    df["rna_sequence"]  = df["rna_sequence"].str.upper().str.replace("T", "U", regex=False)
    # Normalise organism names: ucRBP uses underscores, others use spaces
    if "organism" in df.columns:
        df["organism"] = df["organism"].str.replace("_", " ", regex=False)
    return df


def compute_stats(df: pd.DataFrame, label: str):
    n_pos  = df["binding_label"].sum()
    n_neg  = (df["binding_label"] == 0).sum()
    n_prot = df["protein_name"].nunique() if "protein_name" in df.columns else "?"
    n_exp  = df["experiment_id"].nunique() if "experiment_id" in df.columns else "?"
    orgs   = df["organism"].nunique()      if "organism"     in df.columns else "?"
    print(f"  {label}: {len(df):>10,} pairs | "
          f"{n_pos:>8,} pos | {n_neg:>8,} neg | "
          f"{n_prot} proteins | {n_exp} experiments | {orgs} organisms")


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"  → saved {path}  ({len(df):,} rows)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rnacompete_dir", required=True,
                        help="Root of rnacompete_analysis repository")
    parser.add_argument("--output_dir",     default="data/benchmarks/rnacompete",
                        help="Output directory for benchmark TSV files")
    parser.add_argument("--max_pairs",      type=int, default=2_000_000,
                        help="Max pairs in merged benchmark (stratified subsample, default 2M)")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  RNAcompete → Project Schema Converter (Benchmark Mode)")
    print(f"{'='*65}")
    print(f"  rnacompete_dir : {args.rnacompete_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  max_pairs      : {args.max_pairs:,}\n")

    rng = np.random.default_rng(args.seed)

    # ── 1. Load all sub-datasets ───────────────────────────────────────────
    sub_dfs = {}
    for key, rel_path in SUB_DATASETS.items():
        df = load_subdataset(args.rnacompete_dir, key, rel_path)
        if df is not None:
            df = standardise(df)
            sub_dfs[key] = df

    if not sub_dfs:
        sys.exit("No sub-datasets loaded. Check --rnacompete_dir.")

    # ── 2. Per-sub-dataset benchmarks ─────────────────────────────────────
    print("\n--- Per-sub-dataset benchmarks ---")
    for key, df in sub_dfs.items():
        compute_stats(df, key)
        save_tsv(df, os.path.join(args.output_dir, f"rnacompete_{key}.tsv"))

    # ── 3. Full merge ─────────────────────────────────────────────────────
    print("\n--- Merging all sub-datasets ---")
    combined = pd.concat(list(sub_dfs.values()), ignore_index=True)
    compute_stats(combined, "combined (raw)")

    # ── 4. Deduplicate on (rna_sequence, protein_name) ────────────────────
    before = len(combined)
    combined = combined.drop_duplicates(subset=["rna_sequence", "protein_name"])
    print(f"  Dedup: {before:,} → {len(combined):,} ({before - len(combined):,} removed)")

    # ── 5. Stratified subsample if needed ─────────────────────────────────
    if len(combined) > args.max_pairs:
        print(f"  Subsampling to {args.max_pairs:,} (stratified by label + dataset_source)...")
        sampled_chunks = []
        frac = args.max_pairs / len(combined)
        for (label, ds), grp in combined.groupby(["binding_label", "dataset_source"]):
            n = max(1, int(np.round(len(grp) * frac)))
            idx = rng.choice(len(grp), min(n, len(grp)), replace=False)
            sampled_chunks.append(grp.iloc[idx])
        combined = pd.concat(sampled_chunks, ignore_index=True)
        print(f"  After subsample: {len(combined):,} pairs")

    compute_stats(combined, "combined (final)")
    save_tsv(combined, os.path.join(args.output_dir, "rnacompete_all.tsv"))

    # ── 6. Human subset ───────────────────────────────────────────────────
    print("\n--- Human / near-human subset ---")
    if "organism" in combined.columns:
        human_df = combined[combined["organism"].isin(HUMAN_ORGANISMS)]
        compute_stats(human_df, "human")
        if len(human_df) > 0:
            save_tsv(human_df, os.path.join(args.output_dir, "rnacompete_human.tsv"))

        nonhuman_df = combined[~combined["organism"].isin(HUMAN_ORGANISMS)]
        compute_stats(nonhuman_df, "non-human")
        if len(nonhuman_df) > 0:
            save_tsv(nonhuman_df, os.path.join(args.output_dir, "rnacompete_nonhuman.tsv"))

    # ── 7. Summary JSON ───────────────────────────────────────────────────
    import json
    summary = {
        "decision": "Option B — independent benchmark (NOT merged into training)",
        "rationale": [
            "Construct sequences (RBDs) differ from full-length proteins in training set",
            "Short probes (35-41 nt) from synthetic pool vs SELEX/RBNS enriched sequences",
            "26 eukaryotic organisms; merging contaminates protein-aware splits with paralogs",
            "Zero-shot generalization metric: unseen organism + unseen protein family",
            "Structurally equivalent to ZHMolGraph hard split — fair comparison basis",
        ],
        "sub_datasets": {k: int(len(v)) for k, v in sub_dfs.items()},
        "total_pairs_raw": before,
        "total_pairs_final": int(len(combined)),
        "human_pairs": int(len(human_df)) if "organism" in combined.columns else None,
        "seed": args.seed,
    }
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  → saved {summary_path}")

    print(f"\n{'='*65}")
    print("  DECISION RATIONALE (short form)")
    print("  RNAcompete → Option B: INDEPENDENT BENCHMARK ONLY")
    print()
    for r in summary["rationale"]:
        print(f"  • {r}")
    print(f"{'='*65}")
    print(f"\nDone. Benchmark files in: {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION DECISION — FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
#
# Option A — Merge into training pool
# ------------------------------------
# Pros:
#   + 13.9M new pairs; ~8× current training size → better statistical coverage
#   + 1,087 new protein constructs; massive protein diversity
#   + High-quality negatives (experimentally confirmed, not random)
#   + Dual-filter labeling prevents ambiguous boundary cases
#   + Extends to 26 organisms → cross-species generalization in training
#
# Cons:
#   - CRITICAL: protein sequences are construct sequences (RBDs, ~258 aa median)
#     vs full-length proteins in HTR-SELEX/RBNS training (~400–800 aa).
#     Merging would force the model to learn from two protein representations
#     simultaneously without explicit domain annotation.
#   - Distribution shift: short 35–41 nt synthetic probes vs SELEX/RBNS 30–60 nt
#     enriched sequences (different probe generation procedure, different negative
#     semantics in RNAcompete).
#   - Mixing training and benchmark invalidates motif-learning evaluation —
#     you cannot test whether the model learned HNRNPC motifs if HNRNPC was
#     in training data from both HTR-SELEX and RNAcompete.
#   - Protein-aware split becomes harder: 1,087 RNAcompete proteins include
#     paralogs of the 169 HTR-SELEX/RBNS proteins → homology leakage.
#   - Practical issue: V2 CNN uses one-hot encoding up to prot_max=800 aa.
#     Most RNAcompete constructs are 74–631 aa; padding is safe but full-length
#     proteins in training will dominate because they fill more of the
#     convolution receptive field.
#
# Option B — Independent benchmark (SELECTED)
# ─────────────────────────────────────────────
# Pros:
#   + Zero-shot generalization: models trained only on HTR-SELEX + RBNS are
#     evaluated on completely unseen proteins, organisms, and assay.
#   + Direct comparison with ZHMolGraph (hard split logic is equivalent).
#   + Clean separation: motif-learning ability is measured without memorization.
#   + Per-organism stratification: human RBPs (ucRBP, 613 experiments) gives
#     the most direct comparison; non-human gives true cross-species zero-shot.
#   + Can be used to measure improvement after every clean retraining.
#   + Probe intensity is available as a continuous regression target if
#     affinity-aware evaluation is needed later.
#
# Cons:
#   - Leaves 13.9M high-quality pairs unused for training (can be revisited
#     after architecture exploration, as Phase 3 dataset).
#   - Human RBPs in ucRBP may overlap with HGNC names in HTR-SELEX (HNRNPA1,
#     FUS, etc.) — requires protein-level dedup before evaluation to avoid
#     reporting inflated scores for "seen" proteins.
#
# VERDICT: Option B.
#   After V2-CLEAN retraining, run:
#     python scripts/17_prepare_rnacompete_benchmark.py ...
#     python scripts/16_analyze_predictions.py --predictions_tsv ...
#   to get a ZHMolGraph-comparable zero-shot AUROC across 1,087 RBPs.
#
# Future (Phase 3):
#   Consider Option A ONLY after:
#   1. Homology leakage between RNAcompete and training proteins is quantified
#      (MMseqs2, 30% sequence identity threshold)
#   2. A domain-aware protein encoder is adopted that handles both construct
#      and full-length sequences
#   3. A separate regression head for probe_intensity is added (multi-task)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
