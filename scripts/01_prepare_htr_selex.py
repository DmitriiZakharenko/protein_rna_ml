"""
Script 01: Prepare HTR-SELEX dataset for ML training.

Steps:
  1. Load and validate ml_dataset_simple_clean.tsv
  2. Exploratory data analysis (EDA)
  3. Protein-aware train/val/test split
  4. k-mer encoding of RNA and protein sequences
  5. Save splits (TSV) and encoded features (npz)

Usage:
    python scripts/01_prepare_htr_selex.py \
        --input  ../htr_selex_analysis/results/ml_dataset_simple_clean.tsv \
        --outdir data/ \
        --config configs/htr_selex_validation.yaml
"""

import argparse
import json
import os
import re
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RNA_ALPHABET  = "AUGC"
AA_ALPHABET   = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids


def build_kmer_index(alphabet: str, k: int) -> dict:
    """Return {kmer: index} for all k-mers over alphabet."""
    kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def kmer_freq_vector(seq: str, kmer_index: dict, k: int, normalize: bool = True) -> np.ndarray:
    """
    Count k-mers in seq and return frequency vector.
    Unknown characters are skipped.
    """
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if kmer in kmer_index:
            vec[kmer_index[kmer]] += 1
            n += 1
    if normalize and n > 0:
        vec /= n
    return vec


def encode_dataset(df: pd.DataFrame, rna_k: int, prot_k: int, normalize: bool) -> np.ndarray:
    """
    Encode each row as: [rna_kmer_vector | protein_kmer_vector]
    Returns numpy array of shape (N, n_rna_features + n_prot_features)
    """
    rna_idx  = build_kmer_index(RNA_ALPHABET,  rna_k)
    prot_idx = build_kmer_index(AA_ALPHABET,   prot_k)

    n_rna  = len(rna_idx)
    n_prot = len(prot_idx)
    X = np.zeros((len(df), n_rna + n_prot), dtype=np.float32)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Encoding")):
        rna_vec  = kmer_freq_vector(row["rna_sequence"],     rna_idx,  rna_k,  normalize)
        prot_vec = kmer_freq_vector(row["protein_sequence"], prot_idx, prot_k, normalize)
        X[i, :n_rna]  = rna_vec
        X[i, n_rna:]  = prot_vec

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_RNA   = re.compile(r"^[AUGCaugc]+$")
VALID_PROT  = re.compile(rf"^[{AA_ALPHABET}{AA_ALPHABET.lower()}]+$")


def validate_sequences(df: pd.DataFrame) -> dict:
    """
    Check sequence validity. Returns a report dict.
    """
    issues = {
        "null_rna":      int(df["rna_sequence"].isna().sum()),
        "null_protein":  int(df["protein_sequence"].isna().sum()),
        "invalid_rna":   int((~df["rna_sequence"].fillna("").str.match(r"^[AUGCaugc]+$")).sum()),
        "invalid_prot":  int((~df["protein_sequence"].fillna("").str.match(rf"^[{AA_ALPHABET}{AA_ALPHABET.lower()}]+$")).sum()),
        "has_T_in_rna":  int(df["rna_sequence"].fillna("").str.contains("T|t").sum()),
    }
    return issues


def eda_report(df: pd.DataFrame, out_path: str):
    """Generate and save EDA summary."""
    rna_lengths  = df["rna_sequence"].str.len()
    prot_lengths = df["protein_sequence"].str.len()

    report = {
        "total_examples":    len(df),
        "positive_examples": int((df["binding_label"] == 1).sum()),
        "negative_examples": int((df["binding_label"] == 0).sum()),
        "n_unique_proteins": int(df["protein_name"].nunique()),
        "rna_length": {
            "min":    int(rna_lengths.min()),
            "max":    int(rna_lengths.max()),
            "mean":   float(rna_lengths.mean()),
            "median": float(rna_lengths.median()),
        },
        "protein_length": {
            "min":    int(prot_lengths.min()),
            "max":    int(prot_lengths.max()),
            "mean":   float(prot_lengths.mean()),
            "median": float(prot_lengths.median()),
        },
        "examples_per_protein": {
            "mean":   float(df.groupby("protein_name").size().mean()),
            "min":    int(df.groupby("protein_name").size().min()),
            "max":    int(df.groupby("protein_name").size().max()),
        },
        "sequence_issues": validate_sequences(df),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== EDA Summary ===")
    print(f"  Total examples  : {report['total_examples']:,}")
    print(f"  Positive        : {report['positive_examples']:,}  ({report['positive_examples']/report['total_examples']*100:.1f}%)")
    print(f"  Negative        : {report['negative_examples']:,}  ({report['negative_examples']/report['total_examples']*100:.1f}%)")
    print(f"  Unique proteins : {report['n_unique_proteins']}")
    print(f"  RNA length      : {report['rna_length']['min']}–{report['rna_length']['max']} nt  (median {report['rna_length']['median']:.0f})")
    print(f"  Protein length  : {report['protein_length']['min']}–{report['protein_length']['max']} aa  (median {report['protein_length']['median']:.0f})")

    if report["sequence_issues"]["invalid_rna"] > 0:
        print(f"  ⚠️  Invalid RNA sequences: {report['sequence_issues']['invalid_rna']}")
    if report["sequence_issues"]["has_T_in_rna"] > 0:
        print(f"  ⚠️  RNA sequences with T (should be U): {report['sequence_issues']['has_T_in_rna']}")
    if report["sequence_issues"]["invalid_prot"] > 0:
        print(f"  ⚠️  Invalid protein sequences: {report['sequence_issues']['invalid_prot']}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def protein_aware_split(df: pd.DataFrame, train_frac: float, val_frac: float,
                         seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by protein name so no protein appears in >1 split.

    Returns: (train_df, val_df, test_df, split_map_df)

    split_map_df has columns [protein_name, split]
    """
    rng = np.random.default_rng(seed)
    proteins = np.array(df["protein_name"].unique())
    n = len(proteins)

    shuffled = rng.permutation(proteins)
    n_train = int(np.round(n * train_frac))
    n_val   = int(np.round(n * val_frac))

    train_proteins = set(shuffled[:n_train])
    val_proteins   = set(shuffled[n_train : n_train + n_val])
    test_proteins  = set(shuffled[n_train + n_val :])

    split_map = []
    for p in proteins:
        if p in train_proteins:
            split_map.append({"protein_name": p, "split": "train"})
        elif p in val_proteins:
            split_map.append({"protein_name": p, "split": "val"})
        else:
            split_map.append({"protein_name": p, "split": "test"})

    split_map_df = pd.DataFrame(split_map)
    protein_to_split = dict(zip(split_map_df["protein_name"], split_map_df["split"]))

    df = df.copy()
    df["split"] = df["protein_name"].map(protein_to_split)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"\n=== Protein-Aware Split ===")
    print(f"  Train: {len(train_proteins)} proteins, {len(train_df):,} examples")
    print(f"  Val  : {len(val_proteins)}  proteins, {len(val_df):,} examples")
    print(f"  Test : {len(test_proteins)} proteins, {len(test_df):,} examples")

    # Verify no protein overlap
    assert not (train_proteins & val_proteins), "ERROR: protein overlap between train and val!"
    assert not (train_proteins & test_proteins), "ERROR: protein overlap between train and test!"
    assert not (val_proteins   & test_proteins), "ERROR: protein overlap between val and test!"
    print("  ✅ No protein overlap between splits")

    return train_df, val_df, test_df, split_map_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare HTR-SELEX dataset for ML")
    parser.add_argument("--input",  required=True,  help="Path to ml_dataset_simple_clean.tsv")
    parser.add_argument("--outdir", required=True,  help="Base output directory (data/)")
    parser.add_argument("--config", required=True,  help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    splits_dir    = os.path.join(args.outdir, "splits",    cfg["dataset"]["name"])
    processed_dir = os.path.join(args.outdir, "processed", cfg["dataset"]["name"])
    results_dir   = os.path.join("results",   cfg["dataset"]["name"], "metrics")

    os.makedirs(splits_dir,    exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir,   exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.input}")
    df = pd.read_csv(args.input, sep="\t")
    df.columns = df.columns.str.strip()

    # Normalize sequences to uppercase
    df["rna_sequence"]     = df["rna_sequence"].str.upper()
    df["protein_sequence"] = df["protein_sequence"].str.upper()

    # Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["protein_name", "rna_sequence", "binding_label"])
    n_after = len(df)
    if n_before != n_after:
        print(f"  ⚠️  Dropped {n_before - n_after} duplicate rows")

    # ── 2. EDA ───────────────────────────────────────────────────────────────
    eda_report(df, out_path=os.path.join(results_dir, "eda_summary.json"))

    # ── 3. Split ─────────────────────────────────────────────────────────────
    scfg = cfg["splitting"]
    train_df, val_df, test_df, split_map_df = protein_aware_split(
        df,
        train_frac=scfg["train_frac"],
        val_frac=scfg["val_frac"],
        seed=scfg["seed"],
    )

    # Save split tables (without encoded features — keep them lightweight)
    cols = ["protein_name", "protein_sequence", "rna_sequence", "binding_label", "source"]
    train_df[cols].to_csv(os.path.join(splits_dir, "train.tsv"), sep="\t", index=False)
    val_df[cols].to_csv(  os.path.join(splits_dir, "val.tsv"),   sep="\t", index=False)
    test_df[cols].to_csv( os.path.join(splits_dir, "test.tsv"),  sep="\t", index=False)
    split_map_df.to_csv(  os.path.join(splits_dir, "split_map.tsv"), sep="\t", index=False)
    print(f"\n  Split files saved to: {splits_dir}/")

    # ── 4. Encode ────────────────────────────────────────────────────────────
    ecfg = cfg["encoding"]
    rna_k  = ecfg["rna_kmer_k"]
    prot_k = ecfg["protein_kmer_k"]
    norm   = ecfg["normalize"]

    print(f"\n  Encoding with RNA {rna_k}-mer + Protein {prot_k}-mer (normalize={norm})")

    X_train = encode_dataset(train_df, rna_k, prot_k, norm)
    y_train = train_df["binding_label"].values.astype(np.int8)

    X_val   = encode_dataset(val_df,   rna_k, prot_k, norm)
    y_val   = val_df["binding_label"].values.astype(np.int8)

    X_test  = encode_dataset(test_df,  rna_k, prot_k, norm)
    y_test  = test_df["binding_label"].values.astype(np.int8)

    print(f"  Feature matrix shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Save encoded arrays
    np.savez_compressed(os.path.join(processed_dir, "train_kmer.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(processed_dir, "val_kmer.npz"),   X=X_val,   y=y_val)
    np.savez_compressed(os.path.join(processed_dir, "test_kmer.npz"),  X=X_test,  y=y_test)

    print(f"\n  Encoded arrays saved to: {processed_dir}/")
    print("\n✅ Preprocessing complete. Next step: run scripts/02_train_validation_model.py")


if __name__ == "__main__":
    main()
