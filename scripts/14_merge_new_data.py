"""
Script 14: Merge eCLIP + RNAInter data into our existing training splits.

Merging strategy:
  1. Load existing generalized splits (train/val/test from SELEX+RBNS)
  2. Load new data: data/eclip/eclip_all.tsv + data/rnainter/rnainter_human.tsv
  3. Apply protein-aware assignment:
     - Proteins already in existing splits → stay in the same split
     - New proteins → assign to train (80%), val (10%), test (10%)
       with stratified sampling to balance class ratio
  4. Save updated splits to data/generalized_v2/

Critical invariant: no protein appears in more than one split.
This is the key design principle from Phase 1.

Usage:
    python scripts/14_merge_new_data.py
    python scripts/14_merge_new_data.py --out_dir data/generalized_v2 --dry_run
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_existing_splits(data_dir: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load existing train/val/test splits.
    Returns merged DataFrame + protein→split mapping.
    """
    dfs = []
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path, sep="\t")
        df["split"] = split
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # Build protein → split assignment (invariant: one protein, one split)
    prot_split = {}
    for split in ["train", "val", "test"]:
        prots = combined[combined["split"] == split]["protein_name"].unique()
        for p in prots:
            prot_split[p] = split

    print(f"  Existing data: {len(combined):,} rows, {len(prot_split)} proteins")
    for s in ["train", "val", "test"]:
        n = (combined["split"] == s).sum()
        prots = sum(1 for v in prot_split.values() if v == s)
        print(f"    {s}: {n:,} rows, {prots} proteins")

    return combined, prot_split


def load_new_data(eclip_path: str | None, rnainter_path: str | None) -> pd.DataFrame:
    """Load and validate new datasets."""
    dfs = []

    if eclip_path and os.path.exists(eclip_path):
        df = pd.read_csv(eclip_path, sep="\t")
        # Ensure required columns
        required = ["protein_name", "rna_sequence", "protein_sequence", "binding_label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ⚠️  eCLIP missing columns: {missing} — skipping")
        else:
            print(f"  eCLIP: {len(df):,} rows, {df['protein_name'].nunique()} proteins")
            print(f"    pos={( df['binding_label']==1).sum():,}  "
                  f"neg={(df['binding_label']==0).sum():,}")
            dfs.append(df)
    else:
        print(f"  ⚠️  eCLIP file not found: {eclip_path}")

    if rnainter_path and os.path.exists(rnainter_path):
        df = pd.read_csv(rnainter_path, sep="\t")
        required = ["protein_name", "rna_sequence", "protein_sequence", "binding_label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ⚠️  RNAInter missing columns: {missing} — skipping")
        else:
            print(f"  RNAInter: {len(df):,} rows, {df['protein_name'].nunique()} proteins")
            print(f"    pos={(df['binding_label']==1).sum():,}  "
                  f"neg={(df['binding_label']==0).sum():,}")
            dfs.append(df)
    else:
        print(f"  ⚠️  RNAInter file not found: {rnainter_path}")

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # Normalize RNA sequences: T→U, uppercase
    merged["rna_sequence"] = merged["rna_sequence"].str.upper().str.replace("T", "U", regex=False)

    # Drop rows with missing sequences
    before = len(merged)
    merged = merged.dropna(subset=["rna_sequence", "protein_sequence"])
    merged = merged[merged["rna_sequence"].str.len() >= 10]
    merged = merged[merged["protein_sequence"].str.len() >= 20]
    print(f"  After quality filter: {len(merged):,} / {before:,} rows retained")

    return merged


def assign_splits(new_df: pd.DataFrame, existing_prot_split: dict[str, str],
                  val_frac: float = 0.1, test_frac: float = 0.1,
                  random_seed: int = 42,
                  train_only_sources: list = None) -> pd.DataFrame:
    """
    Assign each row in new_df to train/val/test.

    Key invariants:
    - Proteins already in existing_prot_split → keep their assigned split
      BUT if source is in train_only_sources → force to train regardless
    - New proteins from train_only_sources → always train
    - New proteins from other sources → assign protein-level to train/val/test

    train_only_sources: list of 'dataset' values to restrict to train only.
      Use this for eCLIP to prevent domain-shift contamination of val/test.
      eCLIP positives (in vivo IP peaks) and SELEX positives (enriched synthetic)
      are fundamentally different — mixing in val makes metrics incomparable.
    """
    if train_only_sources is None:
        train_only_sources = []

    new_df = new_df.copy()

    # Separate train-only sources (e.g. eCLIP) → always go to train
    train_only_mask = (new_df.get("dataset", pd.Series([""] * len(new_df)))
                       .isin(train_only_sources))
    df_train_only = new_df[train_only_mask].copy()
    df_rest       = new_df[~train_only_mask].copy()

    if len(df_train_only):
        df_train_only["split"] = "train"
        print(f"\n  Train-only (domain-restricted, no val/test): {len(df_train_only):,}")
        print(f"    Sources: {df_train_only['dataset'].value_counts().to_dict()}")

    # For remaining: known proteins → their existing split
    known_mask = df_rest["protein_name"].isin(existing_prot_split)
    df_known   = df_rest[known_mask].copy()
    df_new     = df_rest[~known_mask].copy()

    print(f"\n  Protein assignment (non-train-only data):")
    print(f"    Rows with proteins in existing splits: {len(df_known):,}")
    print(f"    Rows with NEW proteins: {len(df_new):,}")

    df_known["split"] = df_known["protein_name"].map(existing_prot_split)

    # Assign new proteins: protein-level split
    new_prots = list(df_new["protein_name"].unique())
    print(f"    New unique proteins: {len(new_prots)}")

    if new_prots:
        random.seed(random_seed)
        random.shuffle(new_prots)

        n_val  = max(1, int(len(new_prots) * val_frac))
        n_test = max(1, int(len(new_prots) * test_frac))
        n_train = len(new_prots) - n_val - n_test

        prot_split_new = {}
        for p in new_prots[:n_train]:
            prot_split_new[p] = "train"
        for p in new_prots[n_train:n_train + n_val]:
            prot_split_new[p] = "val"
        for p in new_prots[n_train + n_val:]:
            prot_split_new[p] = "test"

        df_new["split"] = df_new["protein_name"].map(prot_split_new)
        print(f"    New protein split: "
              f"train={n_train}, val={n_val}, test={n_test}")

    result = pd.concat([df_train_only, df_known, df_new], ignore_index=True)
    return result


def save_splits(existing_df: pd.DataFrame, new_df: pd.DataFrame,
                out_dir: str, dry_run: bool = False) -> None:
    """Merge existing + new data and save updated split files."""
    os.makedirs(out_dir, exist_ok=True)

    # Add dataset source if missing
    if "dataset" not in existing_df.columns:
        existing_df["dataset"] = "selex_rbns"
    if "source" not in existing_df.columns:
        existing_df["source"] = "existing"

    combined = pd.concat([existing_df, new_df], ignore_index=True)

    print(f"\n  Combined dataset: {len(combined):,} rows")
    for split in ["train", "val", "test"]:
        sub = combined[combined["split"] == split]
        n_pos = (sub["binding_label"] == 1).sum()
        n_neg = (sub["binding_label"] == 0).sum()
        prots = sub["protein_name"].nunique()
        print(f"    {split}: {len(sub):,} rows ({n_pos:,}+ / {n_neg:,}-), {prots} proteins")

    if dry_run:
        print("\n  [dry_run] No files written.")
        return

    # Save each split
    core_cols = ["protein_name", "rna_sequence", "protein_sequence",
                 "binding_label", "dataset", "source"]
    for split in ["train", "val", "test"]:
        sub = combined[combined["split"] == split][core_cols].reset_index(drop=True)
        out_path = os.path.join(out_dir, f"{split}.tsv")
        sub.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved {split}.tsv: {len(sub):,} rows → {out_path}")

    # Save summary
    summary = {
        "total_rows": int(len(combined)),
        "n_proteins": int(combined["protein_name"].nunique()),
        "splits": {s: int((combined["split"] == s).sum()) for s in ["train", "val", "test"]},
        "sources": combined["dataset"].value_counts().to_dict(),
        "class_balance": {
            "positives": int((combined["binding_label"] == 1).sum()),
            "negatives": int((combined["binding_label"] == 0).sum()),
        },
    }
    import json
    summary_path = os.path.join(out_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing_dir",  default="data/generalized",
                        help="Directory with existing train/val/test.tsv")
    parser.add_argument("--eclip_path",    default="data/eclip/eclip_all.tsv")
    parser.add_argument("--rnainter_path", default="data/rnainter/rnainter_human.tsv")
    parser.add_argument("--out_dir",       default="data/generalized_v2",
                        help="Output directory for updated splits")
    parser.add_argument("--val_frac",      type=float, default=0.10)
    parser.add_argument("--test_frac",     type=float, default=0.10)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--dry_run",       action="store_true")
    args = parser.parse_args()

    print("\n=== Merging datasets ===")
    print(f"  Existing:  {args.existing_dir}")
    print(f"  eCLIP:     {args.eclip_path}")
    print(f"  RNAInter:  {args.rnainter_path}")
    print(f"  Output:    {args.out_dir}")

    # 1. Load existing
    print("\n--- Existing data ---")
    existing_df, prot_split = load_existing_splits(args.existing_dir)

    # 2. Load new data
    print("\n--- New data ---")
    new_df = load_new_data(args.eclip_path, args.rnainter_path)

    if new_df.empty:
        print("\n⚠️  No new data found. Run scripts 12 and 13 first.")
        print("  To run scripts 12 and 13:")
        print("    python scripts/12_download_eclip.py")
        print("    python scripts/13_download_rnainter.py")
        return

    # 3. Assign splits to new data
    # eCLIP goes to train ONLY — in-vivo IP peaks ≠ in-vitro SELEX positives.
    # Mixing eCLIP into val/test makes metrics incomparable across runs.
    new_df_split = assign_splits(new_df, prot_split, args.val_frac, args.test_frac, args.seed,
                                 train_only_sources=["eclip"])

    # 4. Save
    print("\n--- Saving ---")
    save_splits(existing_df, new_df_split, args.out_dir, args.dry_run)

    if not args.dry_run:
        print(f"\n{'='*55}")
        print(f"  DONE")
        print(f"  New splits in: {args.out_dir}/")
        print(f"\n  To retrain V2 CNN on expanded data:")
        print(f"    python scripts/06_train_generalized_v2.py \\")
        print(f"      --data_dir {args.out_dir} \\")
        print(f"      --model_dir models/saved/generalized_v2_expanded \\")
        print(f"      --out_dir results/generalized_v2_expanded")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
