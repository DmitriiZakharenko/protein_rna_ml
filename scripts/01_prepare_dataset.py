"""
Script 01: Universal dataset preparation for any RBP-RNA dataset.

Handles column name differences across datasets via config column_map.
Steps: load → validate → protein-aware split → k-mer encode → save.

Usage (run from protein_rna_ml/ folder):
    python scripts/01_prepare_dataset.py --config configs/htr_selex_validation.yaml
    python scripts/01_prepare_dataset.py --config configs/rbns_validation.yaml
    python scripts/01_prepare_dataset.py --config configs/htr_selex_prjeb47428_validation.yaml
"""

import argparse
import json
import os
import re
from itertools import product

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

RNA_ALPHABET = "AUGC"
AA_ALPHABET  = "ACDEFGHIKLMNPQRSTVWY"


# ── Encoding ──────────────────────────────────────────────────────────────────

def build_kmer_index(alphabet, k):
    return {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}

def kmer_freq_vector(seq, kmer_index, k, normalize=True):
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_index:
            vec[kmer_index[kmer]] += 1
            n += 1
    if normalize and n > 0:
        vec /= n
    return vec

def encode_dataset(df, rna_col, prot_col, rna_k, prot_k, normalize):
    rna_idx  = build_kmer_index(RNA_ALPHABET, rna_k)
    prot_idx = build_kmer_index(AA_ALPHABET,  prot_k)
    n_rna, n_prot = len(rna_idx), len(prot_idx)
    X = np.zeros((len(df), n_rna + n_prot), dtype=np.float32)
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="  Encoding")):
        X[i, :n_rna] = kmer_freq_vector(str(row[rna_col]),  rna_idx,  rna_k,  normalize)
        X[i, n_rna:] = kmer_freq_vector(str(row[prot_col]), prot_idx, prot_k, normalize)
    return X


# ── Validation ────────────────────────────────────────────────────────────────

def validate_and_report(df, protein_col, rna_col, prot_col, label_col, results_dir):
    rna_lens  = df[rna_col].str.len()
    prot_lens = df[prot_col].str.len()
    n_pos = int((df[label_col] == 1).sum())
    n_neg = int((df[label_col] == 0).sum())
    n_proteins = int(df[protein_col].nunique())

    invalid_rna  = int((~df[rna_col].str.match(r'^[AUGCaugc]+$')).sum())
    has_T        = int(df[rna_col].str.contains('T|t', na=False).sum())

    report = {
        "total": len(df),
        "positive": n_pos,
        "negative": n_neg,
        "pos_frac": round(n_pos / len(df), 4),
        "n_proteins": n_proteins,
        "rna_length":  {"min": int(rna_lens.min()),  "max": int(rna_lens.max()),  "median": float(rna_lens.median())},
        "prot_length": {"min": int(prot_lens.min()), "max": int(prot_lens.max()), "median": float(prot_lens.median())},
        "issues": {"invalid_rna": invalid_rna, "has_T_in_rna": has_T},
    }

    print(f"\n  {'─'*50}")
    print(f"  Total: {report['total']:,}  |  Pos: {n_pos:,} ({report['pos_frac']:.1%})  |  Neg: {n_neg:,}")
    print(f"  Proteins: {n_proteins}  |  RNA length: {report['rna_length']['min']}–{report['rna_length']['max']} nt")
    if invalid_rna: print(f"  ⚠️  Invalid RNA chars: {invalid_rna}")
    if has_T:       print(f"  ⚠️  T instead of U in RNA: {has_T}")
    print(f"  {'─'*50}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "eda_summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


# ── Splitting ─────────────────────────────────────────────────────────────────

def protein_aware_split(df, protein_col, train_frac, val_frac, seed):
    rng      = np.random.default_rng(seed)
    proteins = np.array(df[protein_col].unique())
    n        = len(proteins)
    shuffled = rng.permutation(proteins)
    n_train  = int(round(n * train_frac))
    n_val    = int(round(n * val_frac))

    train_p = set(shuffled[:n_train])
    val_p   = set(shuffled[n_train:n_train+n_val])
    test_p  = set(shuffled[n_train+n_val:])

    assert not (train_p & val_p) and not (train_p & test_p) and not (val_p & test_p)

    df = df.copy()
    df["_split"] = df[protein_col].apply(lambda p: "train" if p in train_p else ("val" if p in val_p else "test"))

    train_df = df[df["_split"]=="train"].drop(columns=["_split"]).reset_index(drop=True)
    val_df   = df[df["_split"]=="val"].drop(columns=["_split"]).reset_index(drop=True)
    test_df  = df[df["_split"]=="test"].drop(columns=["_split"]).reset_index(drop=True)
    split_map = pd.DataFrame([{protein_col: p, "split": df.loc[df[protein_col]==p, "_split"].iloc[0]}
                               for p in proteins])

    print(f"\n  Split → Train: {len(train_p)} proteins / {len(train_df):,} rows")
    print(f"          Val  : {len(val_p)} proteins / {len(val_df):,} rows")
    print(f"          Test : {len(test_p)} proteins / {len(test_df):,} rows")
    print(f"  ✅ No protein overlap")
    return train_df, val_df, test_df, split_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cmap         = cfg["dataset"]["column_map"]
    protein_col  = cmap["protein_col"]
    prot_seq_col = cmap["protein_seq"]
    rna_col      = cmap["rna_seq"]
    label_col    = cmap["label"]

    splits_dir    = cfg["dataset"]["splits_dir"]
    processed_dir = cfg["dataset"]["processed_dir"]
    results_dir   = os.path.dirname(cfg["output"]["metrics_path"])
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Load
    raw_path = cfg["dataset"]["raw_path"]
    print(f"\nLoading: {raw_path}")
    df = pd.read_csv(raw_path, sep="\t")
    df[rna_col]      = df[rna_col].str.upper()
    df[prot_seq_col] = df[prot_seq_col].str.upper()

    before = len(df)
    df = df.drop_duplicates(subset=[protein_col, rna_col, label_col]).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before-len(df)} duplicates")

    # EDA
    validate_and_report(df, protein_col, rna_col, prot_seq_col, label_col, results_dir)

    # Split
    scfg = cfg["splitting"]
    train_df, val_df, test_df, split_map = protein_aware_split(
        df, protein_col, scfg["train_frac"], scfg["val_frac"], scfg["seed"])

    keep_cols = [protein_col, prot_seq_col, rna_col, label_col]
    if "source" in df.columns:
        keep_cols.append("source")

    train_df[keep_cols].to_csv(os.path.join(splits_dir, "train.tsv"), sep="\t", index=False)
    val_df[keep_cols].to_csv(  os.path.join(splits_dir, "val.tsv"),   sep="\t", index=False)
    test_df[keep_cols].to_csv( os.path.join(splits_dir, "test.tsv"),  sep="\t", index=False)
    split_map.to_csv(os.path.join(splits_dir, "split_map.tsv"), sep="\t", index=False)
    print(f"\n  Split TSVs → {splits_dir}/")

    # Encode
    ecfg = cfg["encoding"]
    rna_k, prot_k, norm = ecfg["rna_kmer_k"], ecfg["protein_kmer_k"], ecfg["normalize"]
    print(f"\n  RNA {rna_k}-mer + Protein {prot_k}-mer encoding...")

    X_train = encode_dataset(train_df, rna_col, prot_seq_col, rna_k, prot_k, norm)
    X_val   = encode_dataset(val_df,   rna_col, prot_seq_col, rna_k, prot_k, norm)
    X_test  = encode_dataset(test_df,  rna_col, prot_seq_col, rna_k, prot_k, norm)

    print(f"  Feature shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    np.savez_compressed(os.path.join(processed_dir,"train_kmer.npz"), X=X_train, y=train_df[label_col].values.astype(np.int8))
    np.savez_compressed(os.path.join(processed_dir,"val_kmer.npz"),   X=X_val,   y=val_df[label_col].values.astype(np.int8))
    np.savez_compressed(os.path.join(processed_dir,"test_kmer.npz"),  X=X_test,  y=test_df[label_col].values.astype(np.int8))
    print(f"  Encoded arrays → {processed_dir}/")

    print(f"\n✅ Done. Next: python scripts/02_train_validation_model.py --config {args.config}")

if __name__ == "__main__":
    main()
