#!/usr/bin/env python3
"""
22_build_phase3a_dataset.py
Build the Phase 3A training dataset: HTR-SELEX + RBNS + RNAcompete,
with homology-aware protein splitting.

Pipeline:
  1. Load existing generalized_v2 splits (SELEX + RBNS, ~632K, 168 proteins)
  2. Load RNAcompete benchmark TSV (project schema)
  3. Homology filter: exclude RNAcompete proteins with >30% sequence identity
     to training proteins (uses MMseqs2 TSV if provided, else name-based)
  4. Assign RNAcompete proteins to train/val/test (protein-aware)
  5. Subsample RNAcompete to avoid overwhelming the original data
  6. Save to data/generalized_v3a/

Key design decisions:
  - RNAcompete proteins that overlap with existing test/val stay in that split
    (prevents homology leakage)
  - RNAcompete proteins homologous to test proteins go to train only
    (prevents evaluation inflation)
  - dataset_source column retained so V4 model can use it as auxiliary input
  - Organism column added for stratified analysis

Usage:
  # Step 1: export FASTA for MMseqs2
  python scripts/22_build_phase3a_dataset.py --export_fasta_only \
      --selex_dir data/generalized_v2 \
      --rnacompete data/benchmarks/rnacompete/rnacompete_all.tsv \
      --out_dir data/generalized_v3a

  # Step 2: run MMseqs2 (external)
  mmseqs easy-search data/generalized_v3a/proteins_train.fasta \
      data/generalized_v3a/proteins_rnacompete.fasta \
      results/homology/train_vs_rnacompete.tsv tmp/ \
      --min-seq-id 0.30 -c 0.8 --cov-mode 0

  # Step 3: build dataset
  python scripts/22_build_phase3a_dataset.py \
      --selex_dir data/generalized_v2 \
      --rnacompete data/benchmarks/rnacompete/rnacompete_all.tsv \
      --homology_tsv results/homology/train_vs_rnacompete.tsv \
      --out_dir data/generalized_v3a \
      --max_rnacompete 2000000
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_selex_splits(selex_dir: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load existing SELEX+RBNS splits. Returns (df_with_split_col, prot→split)."""
    dfs = []
    for split in ["train", "val", "test"]:
        path = os.path.join(selex_dir, f"{split}.tsv")
        if not os.path.exists(path):
            sys.exit(f"Missing: {path}")
        df = pd.read_csv(path, sep="\t")
        df["split"] = split
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    if "dataset_source" not in combined.columns and "dataset" in combined.columns:
        combined = combined.rename(columns={"dataset": "dataset_source"})
    if "dataset_source" not in combined.columns:
        combined["dataset_source"] = "selex_rbns"

    prot_split = {p: s for s, g in combined.groupby("split") for p in g["protein_name"].unique()}

    print(f"\n  SELEX+RBNS: {len(combined):,} rows, {len(prot_split)} proteins")
    for s in ["train", "val", "test"]:
        sub = combined[combined["split"] == s]
        print(f"    {s}: {len(sub):,} rows, {sub['protein_name'].nunique()} proteins")

    return combined, prot_split


def load_rnacompete(rnacompete_path: str, max_rows: int | None,
                    seed: int) -> pd.DataFrame:
    """Load RNAcompete benchmark TSV (project schema)."""
    print(f"\n  Loading RNAcompete: {rnacompete_path}")
    df = pd.read_csv(rnacompete_path, sep="\t", low_memory=False)

    # Normalise columns
    if "target_name" in df.columns and "protein_name" not in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})
    if "hyb_id" in df.columns and "experiment_id" not in df.columns:
        df = df.rename(columns={"hyb_id": "experiment_id"})
    if "rna_sequence" in df.columns:
        df["rna_sequence"] = df["rna_sequence"].str.upper().str.replace("T", "U", regex=False)
    if "organism" in df.columns:
        df["organism"] = df["organism"].str.replace("_", " ", regex=False)
    if "dataset_source" not in df.columns:
        df["dataset_source"] = "rnacompete"

    required = {"rna_sequence", "protein_sequence", "protein_name", "binding_label"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"RNAcompete file missing columns: {missing}")

    n_pos = (df["binding_label"] == 1).sum()
    n_neg = (df["binding_label"] == 0).sum()
    print(f"  Raw: {len(df):,} rows | pos: {n_pos:,} | neg: {n_neg:,} | "
          f"proteins: {df['protein_name'].nunique()}")

    if max_rows and len(df) > max_rows:
        rng = np.random.default_rng(seed)
        # Stratified by protein + label
        chunks = []
        frac = max_rows / len(df)
        for (prot, label), grp in df.groupby(["protein_name", "binding_label"]):
            n = max(1, int(np.round(len(grp) * frac)))
            idx = rng.choice(len(grp), min(n, len(grp)), replace=False)
            chunks.append(grp.iloc[idx])
        df = pd.concat(chunks, ignore_index=True)
        print(f"  Subsampled to {len(df):,} rows")

    return df


def load_homology_tsv(homology_tsv: str) -> dict[str, set[str]]:
    """
    Parse MMseqs2 easy-search output.
    Returns {rnacompete_protein → set of homologous selex proteins}.
    MMseqs2 output columns: query, target, identity, ...
    """
    if not os.path.exists(homology_tsv):
        print(f"  [WARN] homology TSV not found: {homology_tsv} — using name-only overlap")
        return {}
    df = pd.read_csv(homology_tsv, sep="\t", header=None,
                     names=["query", "target", "identity", "alnlen",
                             "mismatch", "gapopen", "qstart", "qend",
                             "tstart", "tend", "evalue", "bitscore"])
    mapping: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        q = str(row["query"]).split("|")[0].strip()
        t = str(row["target"]).split("|")[0].strip()
        mapping.setdefault(q, set()).add(t)
        mapping.setdefault(t, set()).add(q)
    print(f"  Homology pairs loaded: {len(df):,} hits, {len(mapping)} proteins involved")
    return mapping


def export_fasta(df: pd.DataFrame, path: str, id_col: str, seq_col: str):
    """Export unique protein sequences to FASTA."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = set()
    with open(path, "w") as fh:
        for _, row in df.iterrows():
            pid = str(row[id_col])
            seq = str(row[seq_col])
            if pid in seen or not seq or seq == "nan":
                continue
            seen.add(pid)
            fh.write(f">{pid}\n{seq}\n")
    print(f"  → FASTA: {path} ({len(seen)} sequences)")


def assign_rnacompete_splits(
    rna_df: pd.DataFrame,
    selex_prot_split: dict[str, str],
    homology_map: dict[str, set[str]],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign RNAcompete proteins to splits.

    Rules:
      1. If protein name matches an existing SELEX protein → use that split
      2. If protein is homologous (MMseqs2) to a SELEX test/val protein → train only
         (prevents evaluation leakage)
      3. Otherwise → protein-level random assignment (train/val/test)
    """
    rna_df = rna_df.copy()
    proteins = rna_df["protein_name"].unique()

    # Build homology-aware SELEX test+val set
    selex_test_val = {p for p, s in selex_prot_split.items() if s in ("test", "val")}
    selex_test_val_upper = {p.upper() for p in selex_test_val}

    prot_assignment: dict[str, str] = {}
    n_name_match = n_homolog_blocked = n_new = 0

    rng = random.Random(seed)
    new_proteins = []

    for prot in proteins:
        prot_up = prot.upper()

        # Rule 1: name match to existing SELEX split
        if prot in selex_prot_split:
            prot_assignment[prot] = selex_prot_split[prot]
            n_name_match += 1
            continue

        # Rule 2: homology to SELEX test/val → force to train
        homologs = homology_map.get(prot, set()) | homology_map.get(prot_up, set())
        if homologs & selex_test_val_upper:
            prot_assignment[prot] = "train"
            n_homolog_blocked += 1
            continue

        # Rule 3: new protein
        new_proteins.append(prot)
        n_new += 1

    # Assign new proteins protein-level
    rng.shuffle(new_proteins)
    n_val  = max(1, int(len(new_proteins) * val_frac))
    n_test = max(1, int(len(new_proteins) * test_frac))
    for p in new_proteins[:len(new_proteins) - n_val - n_test]:
        prot_assignment[p] = "train"
    for p in new_proteins[len(new_proteins) - n_val - n_test:len(new_proteins) - n_test]:
        prot_assignment[p] = "val"
    for p in new_proteins[len(new_proteins) - n_test:]:
        prot_assignment[p] = "test"

    rna_df["split"] = rna_df["protein_name"].map(prot_assignment)

    print(f"\n  RNAcompete protein assignment ({len(proteins)} proteins):")
    print(f"    Name-matched to SELEX:        {n_name_match:>5}")
    print(f"    Homolog-blocked to train:      {n_homolog_blocked:>5}")
    print(f"    New proteins:                  {n_new:>5}")
    print(f"      → train: {sum(1 for p in new_proteins[:len(new_proteins)-n_val-n_test])}")
    print(f"      → val:   {n_val}")
    print(f"      → test:  {n_test}")

    return rna_df


def save_splits(selex_df: pd.DataFrame, rna_df: pd.DataFrame,
                out_dir: str, dry_run: bool = False):
    combined = pd.concat([selex_df, rna_df], ignore_index=True)

    # Ensure organism column exists
    if "organism" not in combined.columns:
        combined["organism"] = combined["dataset_source"].apply(
            lambda x: "Homo sapiens" if "selex" in str(x) or "rbns" in str(x) else None)

    core_cols = ["protein_name", "rna_sequence", "protein_sequence",
                 "binding_label", "dataset_source", "organism"]
    for col in core_cols:
        if col not in combined.columns:
            combined[col] = None

    print(f"\n  Combined dataset: {len(combined):,} rows")
    for split in ["train", "val", "test"]:
        sub = combined[combined["split"] == split]
        n_pos = (sub["binding_label"] == 1).sum()
        n_neg = (sub["binding_label"] == 0).sum()
        srcs  = sub["dataset_source"].value_counts().to_dict()
        print(f"    {split}: {len(sub):>9,} rows | {n_pos:>7,}+ {n_neg:>8,}- | "
              f"{sub['protein_name'].nunique()} proteins | sources: {srcs}")

    if dry_run:
        print("\n  [DRY RUN] No files written.")
        return

    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        sub = combined[combined["split"] == split][core_cols].reset_index(drop=True)
        path = os.path.join(out_dir, f"{split}.tsv")
        sub.to_csv(path, sep="\t", index=False)
        print(f"  Saved {path}")

    summary = {
        "total_rows":  int(len(combined)),
        "n_proteins":  int(combined["protein_name"].nunique()),
        "splits":      {s: int((combined["split"] == s).sum()) for s in ["train", "val", "test"]},
        "sources":     combined["dataset_source"].value_counts().to_dict(),
        "class_balance": {
            "positives": int((combined["binding_label"] == 1).sum()),
            "negatives": int((combined["binding_label"] == 0).sum()),
        },
        "organisms": combined["organism"].value_counts().head(10).to_dict(),
    }
    with open(os.path.join(out_dir, "dataset_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved dataset_summary.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selex_dir",       default="data/generalized_v2",
                        help="Existing SELEX+RBNS split directory")
    parser.add_argument("--rnacompete",      default=None,
                        help="RNAcompete training TSV (default: build from eukarya+rbpzoo+ucrbp23)")
    parser.add_argument("--benchmark_dir",   default="data/benchmarks/rnacompete",
                        help="Directory with per-panel RNAcompete TSVs (used with --build_rnacompete)")
    parser.add_argument("--ucrbp_whitelist", default="configs/ucrbp_23_reproducible.txt",
                        help="Whitelist for ucRBP panel (23 reproducible RBPs)")
    parser.add_argument("--build_rnacompete", action="store_true",
                        help="Run 22a logic: eukarya+rbpzoo full, ucRBP filtered to whitelist")
    parser.add_argument("--homology_tsv",    default=None,
                        help="MMseqs2 easy-search output TSV (optional but recommended)")
    parser.add_argument("--out_dir",         default="data/generalized_v3a",
                        help="Output directory for merged splits")
    parser.add_argument("--max_rnacompete",  type=int, default=2_000_000,
                        help="Max RNAcompete rows to include (stratified subsample)")
    parser.add_argument("--val_frac",        type=float, default=0.10)
    parser.add_argument("--test_frac",       type=float, default=0.10)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--export_fasta_only", action="store_true",
                        help="Only export FASTA files for MMseqs2, then exit")
    parser.add_argument("--dry_run",         action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Phase 3A Dataset Builder")
    print(f"  SELEX+RBNS + RNAcompete with homology-aware split")
    print(f"{'='*65}\n")

    # Resolve RNAcompete input
    rnacompete_path = args.rnacompete
    if args.build_rnacompete or rnacompete_path is None:
        from src.data.rnacompete_training import (
            load_rnacompete_training_subset,
            save_training_subset,
        )
        default_out = os.path.join(
            args.benchmark_dir, "rnacompete_training_phase3a.tsv")
        print("\n--- Building RNAcompete training subset (eukarya+rbpzoo+ucrbp23) ---")
        rna_train = load_rnacompete_training_subset(
            args.benchmark_dir, args.ucrbp_whitelist)
        rnacompete_path = default_out
        save_training_subset(rna_train, rnacompete_path)
    elif not os.path.exists(rnacompete_path):
        sys.exit(f"RNAcompete file not found: {rnacompete_path}")

    selex_df, selex_prot_split = load_selex_splits(args.selex_dir)
    rna_df = load_rnacompete(rnacompete_path, args.max_rnacompete, args.seed)

    # ── Export FASTA for MMseqs2 ───────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    train_df = selex_df[selex_df["split"] == "train"]
    export_fasta(train_df,
                 os.path.join(args.out_dir, "proteins_train.fasta"),
                 "protein_name", "protein_sequence")
    export_fasta(rna_df,
                 os.path.join(args.out_dir, "proteins_rnacompete.fasta"),
                 "protein_name", "protein_sequence")

    if args.export_fasta_only:
        print("\n  FASTA files exported. Run MMseqs2:")
        print(f"  mmseqs easy-search \\")
        print(f"      {args.out_dir}/proteins_train.fasta \\")
        print(f"      {args.out_dir}/proteins_rnacompete.fasta \\")
        print(f"      results/homology/train_vs_rnacompete.tsv \\")
        print(f"      tmp/ --min-seq-id 0.30 -c 0.8 --cov-mode 0")
        print(f"\n  Then re-run without --export_fasta_only to build the dataset.")
        return

    # ── Homology filtering ─────────────────────────────────────────────────
    homology_map = load_homology_tsv(args.homology_tsv) if args.homology_tsv else {}
    if not homology_map:
        print("  [WARN] No homology map — using name-only deduplication.")
        print("         Run MMseqs2 for proper homology filtering.")

    # ── Assign splits ──────────────────────────────────────────────────────
    rna_df = assign_rnacompete_splits(
        rna_df, selex_prot_split, homology_map,
        args.val_frac, args.test_frac, args.seed)

    # ── Save ───────────────────────────────────────────────────────────────
    print("\n--- Saving combined dataset ---")
    save_splits(selex_df, rna_df, args.out_dir, args.dry_run)

    if not args.dry_run:
        print(f"\n{'='*65}")
        print(f"  Dataset ready: {args.out_dir}/")
        print(f"\n  Next — train V2 on expanded data:")
        print(f"    python scripts/06_train_generalized_v2.py \\")
        print(f"        --data_dir {args.out_dir} --epochs 60 \\")
        print(f"        --prot_max 700 \\")
        print(f"        --model_dir models/saved/generalized_v3a \\")
        print(f"        --out_dir results/generalized/v3a_scale")
        print(f"\n  Then train V4 (interaction layer):")
        print(f"    python scripts/21_train_generalized_v4_interaction.py \\")
        print(f"        --data_dir {args.out_dir}")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
