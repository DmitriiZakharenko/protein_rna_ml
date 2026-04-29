"""
Script 04 (v2): Build unified multi-dataset training set for Phase 2.

Key fixes vs v1:
  1. Dataset one-hot (3 features) — model knows which dataset each example came from
  2. RNA length (1 feature, normalized) — accounts for 20 nt RBNS vs 40 nt HTR-SELEX
  3. StandardScaler fitted on train → applied to val/test (removes cross-dataset scale bias)

Feature vector layout (8263 total):
  [RNA 4-mer (256) | Protein 3-mer (8000) | dataset_onehot (3) | rna_len_norm (1) | expt_type (1) | is_flagged (1) | is_flagged (1)]

Usage (from protein_rna_ml/):
    python scripts/04_build_generalized_dataset.py
"""

import os, json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS = [
    {"name": "htr_selex_25907",
     "path": "../htr_selex_analysis/results/ml_dataset_simple_clean.tsv",
     "protein_col": "protein_name", "experiment_type": "in_vitro"},
    {"name": "rbns",
     "path": "../rbns_analysis/results/ml_dataset_rbns_clean.tsv",
     "protein_col": "target_name",  "experiment_type": "in_vitro"},
    {"name": "htr_selex_47428",
     "path": "../htr_selex_prjeb47428/results/final/ml_dataset_top1000_lastcycle_clean.tsv",
     "protein_col": "protein_name", "experiment_type": "in_vitro"},
]
DATASET_NAMES = [d["name"] for d in DATASETS]

FLAGGED_PROTEINS = {"RBM4", "RBM4B", "XRCC6"}
SEED        = 42
TRAIN_FRAC  = 0.75
VAL_FRAC    = 0.11
RNA_K, PROT_K = 4, 3
RNA_ALPHA   = "AUGC"
AA_ALPHA    = "ACDEFGHIKLMNPQRSTVWY"
OUT_DIR     = "data/generalized"


def build_kmer_index(alpha, k):
    return {"".join(p): i for i, p in enumerate(product(alpha, repeat=k))}

def kmer_vec(seq, idx, k, norm=True):
    v = np.zeros(len(idx), dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k]
        if km in idx: v[idx[km]] += 1; n += 1
    if norm and n > 0: v /= n
    return v


def load_dataset(cfg):
    df = pd.read_csv(cfg["path"], sep="\t")
    df["rna_sequence"]     = df["rna_sequence"].str.upper()
    df["protein_sequence"] = df["protein_sequence"].str.upper()
    df = df.rename(columns={cfg["protein_col"]: "protein_name"})
    df["dataset_source"]   = cfg["name"]
    df["experiment_type"]  = cfg["experiment_type"]
    df["is_flagged"]       = df["protein_name"].isin(FLAGGED_PROTEINS).astype(int)
    keep = ["protein_name","protein_sequence","rna_sequence",
            "binding_label","dataset_source","experiment_type","is_flagged"]
    df = df[keep].drop_duplicates(
            subset=["protein_name","rna_sequence","binding_label"]).reset_index(drop=True)
    rna_lens = df["rna_sequence"].str.len()
    print(f"  {cfg['name']:<22}  {len(df):>7,} rows  "
          f"{df['protein_name'].nunique():>3} proteins  "
          f"RNA {rna_lens.min()}–{rna_lens.max()} nt  "
          f"{(df['binding_label']==1).sum()/len(df):.1%} pos")
    return df


def global_protein_split(df, train_frac, val_frac, seed):
    rng = np.random.default_rng(seed)
    proteins = np.array(df["protein_name"].unique())
    n = len(proteins)
    shuffled = rng.permutation(proteins)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    train_p = set(shuffled[:n_train])
    val_p   = set(shuffled[n_train:n_train+n_val])
    test_p  = set(shuffled[n_train+n_val:])
    df = df.copy()
    df["split"] = df["protein_name"].apply(
        lambda p: "train" if p in train_p else ("val" if p in val_p else "test"))
    for sp, g in df.groupby("split"):
        print(f"  {sp:<6}: {g['protein_name'].nunique():>3} proteins  {len(g):>7,} rows")
    return df


def encode(df, rna_idx, prot_idx, desc=""):
    n_rna, n_prot = len(rna_idx), len(prot_idx)
    n_ds = len(DATASET_NAMES)
    # layout: [rna_kmer | prot_kmer | dataset_onehot | rna_len_norm | expt_type | is_flagged]
    n_feat = n_rna + n_prot + n_ds + 1 + 1 + 1
    X = np.zeros((len(df), n_feat), dtype=np.float32)
    rna_max_len = df["rna_sequence"].str.len().max()

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"  Encoding {desc}")):
        X[i, :n_rna]              = kmer_vec(row["rna_sequence"],     rna_idx, RNA_K)
        X[i, n_rna:n_rna+n_prot]  = kmer_vec(row["protein_sequence"], prot_idx, PROT_K)
        ds_idx = DATASET_NAMES.index(row["dataset_source"])
        X[i, n_rna+n_prot+ds_idx] = 1.0          # dataset one-hot
        X[i, n_rna+n_prot+n_ds]   = len(row["rna_sequence"]) / rna_max_len  # RNA length
        X[i, n_rna+n_prot+n_ds+1] = 1.0 if row["experiment_type"]=="in_vivo" else 0.0
        X[i, n_rna+n_prot+n_ds+2] = float(row["is_flagged"])
    return X


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    print("\n=== Loading datasets ===")
    df = pd.concat([load_dataset(cfg) for cfg in DATASETS], ignore_index=True)
    print(f"\n  Combined: {len(df):,} rows  {df['protein_name'].nunique()} unique proteins")

    # Check cross-dataset proteins
    prot_ds = df.groupby("protein_name")["dataset_source"].nunique()
    n_multi = (prot_ds > 1).sum()
    print(f"  Proteins in 2+ datasets: {n_multi}")

    # Split
    print("\n=== Global protein-aware split ===")
    df = global_protein_split(df, TRAIN_FRAC, VAL_FRAC, SEED)

    # Verify
    for prot, g in df.groupby("protein_name"):
        assert g["split"].nunique() == 1, f"Protein {prot} in multiple splits!"
    print("  ✅ No protein overlap")

    # Save TSVs
    cols = ["protein_name","protein_sequence","rna_sequence",
            "binding_label","dataset_source","experiment_type","is_flagged"]
    for sp in ("train","val","test"):
        df[df["split"]==sp][cols].to_csv(
            os.path.join(OUT_DIR, f"{sp}.tsv"), sep="\t", index=False)
    df[cols+["split"]].to_csv(os.path.join(OUT_DIR,"full_dataset.tsv"), sep="\t", index=False)

    # Encode
    print(f"\n=== Encoding (RNA {RNA_K}-mer + Protein {PROT_K}-mer + dataset + length) ===")
    rna_idx  = build_kmer_index(RNA_ALPHA, RNA_K)
    prot_idx = build_kmer_index(AA_ALPHA,  PROT_K)

    splits = {}
    for sp in ("train","val","test"):
        sub = df[df["split"]==sp].reset_index(drop=True)
        X = encode(sub, rna_idx, prot_idx, sp)
        y = sub["binding_label"].values.astype(np.int8)
        splits[sp] = (X, y, sub)

    # Fit StandardScaler on train, apply to all splits
    print("\n=== StandardScaler (fit on train) ===")
    scaler = StandardScaler()
    splits["train"] = (scaler.fit_transform(splits["train"][0]),) + splits["train"][1:]
    splits["val"]   = (scaler.transform(splits["val"][0]),)       + splits["val"][1:]
    splits["test"]  = (scaler.transform(splits["test"][0]),)      + splits["test"][1:]

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print(f"  Scaler saved. Feature dim: {splits['train'][0].shape[1]}")

    # Save npz
    for sp, (X, y, _) in splits.items():
        path = os.path.join(OUT_DIR, f"{sp}_kmer.npz")
        np.savez_compressed(path, X=X.astype(np.float32), y=y)
        print(f"  {sp}: {X.shape}  →  {path}")

    # Stats
    feature_dim = splits["train"][0].shape[1]
    stats = {
        "total": int(len(df)),
        "n_proteins": int(df["protein_name"].nunique()),
        "n_proteins_in_multiple_datasets": int(n_multi),
        "datasets": {d["name"]: int((df["dataset_source"]==d["name"]).sum()) for d in DATASETS},
        "split_sizes": {sp: int((df["split"]==sp).sum()) for sp in ("train","val","test")},
        "feature_dim": feature_dim,
        "feature_layout": f"RNA_4mer(256) + Prot_3mer(8000) + dataset_onehot(3) + rna_len(1) + expt_type(1) + is_flagged(1) = {feature_dim}",
        "normalization": "StandardScaler fitted on train split",
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Total: {stats['total']:,}  |  Proteins: {stats['n_proteins']}  |  Multi-dataset proteins: {n_multi}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  {stats['feature_layout']}")
    print(f"\n✅ Done → {OUT_DIR}/")
    print(f"   Next: python scripts/05_train_generalized_v1.py")

if __name__ == "__main__":
    main()
