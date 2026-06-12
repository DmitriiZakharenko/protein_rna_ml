#!/usr/bin/env python3
"""
25_train_rna_only_per_protein.py
---------------------------------
Train simple RNA-only classifiers per protein (no protein sequence features).

For each RBP: encode RNA 4-mer frequencies → compare Logistic Regression,
Random Forest, and XGBoost.

Default (--honest): dedupe by rna_sequence, stratified 60/20/20 train/val/test,
model selection on validation only, reported metrics on held-out test (AUROC + AUPRC).

Legacy (--simple_split): stratified 80/20 train/test, model picked on test (optimistic).

Designed as a baseline / sanity check: can RNA sequence alone separate
binding vs non-binding for a given protein?

Usage:
    python scripts/25_train_rna_only_per_protein.py \\
        --data_file ../htr_selex_analysis/results/ml_dataset_simple_clean.tsv \\
        --dataset htr_selex

    python scripts/25_train_rna_only_per_protein.py \\
        --data_file ../rbns_analysis/results/ml_dataset_rbns_clean.tsv \\
        --dataset rbns --protein_col target_name

    python scripts/25_train_rna_only_per_protein.py \\
        --data_file ../rnacompete_analysis/eukarya/results/ml_dataset_eukarya_clean.tsv.gz \\
        --dataset rnacompete_eukarya --rnacompete_best_experiment

Post-hoc scoring on extracted examples:
    python scripts/25_train_rna_only_per_protein.py \\
        --score_examples results/top_bottom_examples/all_protocols_summary.tsv \\
        --models_dir results/rna_only_per_protein
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------------------------------------------------------------------
# k-mer encoding (RNA only)
# ---------------------------------------------------------------------------

RNA_ALPHABET = "AUGC"


def portable_path(path: Path | str) -> str:
    """Write repo-relative paths in stats JSON (avoid absolute home directories)."""
    p = Path(path).expanduser().resolve()
    if not p.is_absolute():
        return str(path)
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        try:
            return os.path.relpath(p, base)
        except ValueError:
            continue
    return p.name


def build_kmer_index(alphabet: str, k: int) -> dict[str, int]:
    return {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}


def kmer_freq_vector(
    seq: str, kmer_index: dict[str, int], k: int, normalize: bool = True
) -> np.ndarray:
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    seq = str(seq).upper().replace("T", "U")
    n = 0
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k]
        if km in kmer_index:
            vec[kmer_index[km]] += 1
            n += 1
    if normalize and n > 0:
        vec /= n
    return vec


def encode_rna_matrix(
    seqs: pd.Series, kmer_index: dict[str, int], k: int
) -> np.ndarray:
    X = np.zeros((len(seqs), len(kmer_index)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        X[i] = kmer_freq_vector(seq, kmer_index, k, normalize=True)
    return X


# ---------------------------------------------------------------------------
# Data preprocessing helpers
# ---------------------------------------------------------------------------

def select_best_rnacompete_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one experiment per protein (highest mean positive probe_intensity)."""
    id_col = next((c for c in ("experiment_id", "hyb_id") if c in df.columns), None)
    if id_col is None:
        return df

    protein_col = "protein_name"
    parts = []
    for prot, sub in df.groupby(protein_col):
        if sub[id_col].nunique() <= 1:
            parts.append(sub)
            continue
        sub = sub.copy()
        sub["probe_intensity"] = pd.to_numeric(sub["probe_intensity"], errors="coerce")
        pos = sub[sub["binding_label"] == 1]
        if pos.empty or pos["probe_intensity"].isna().all():
            parts.append(sub)
            continue
        best_eid = pos.groupby(id_col)["probe_intensity"].mean().idxmax()
        parts.append(sub[sub[id_col] == best_eid])
    return pd.concat(parts, ignore_index=True)


def filter_modal_length(df: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    """Restrict to modal RNA length when multiple lengths exist."""
    lengths = df["rna_sequence"].str.len()
    if lengths.nunique() <= 1:
        return df, int(lengths.iloc[0]) if len(lengths) else None
    mode_len = Counter(lengths).most_common(1)[0][0]
    return df[df["rna_sequence"].str.len() == mode_len].copy(), mode_len


def dedupe_by_sequence(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    One row per unique rna_sequence. Conflicting labels → majority vote; ties dropped.
    Returns (deduped_df, n_rows_removed).
    """
    n_before = len(df)
    rows: list[dict] = []
    for seq, grp in df.groupby("rna_sequence", sort=False):
        labels = grp["binding_label"].values
        counts = Counter(labels)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue  # ambiguous label — drop sequence
        label = top[0][0]
        row = grp.iloc[0].to_dict()
        row["binding_label"] = int(label)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out, n_before - len(out)


def stratified_three_way_split(
    n: int,
    y: np.ndarray,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, val, test index arrays (60/20/20 when test_frac=val_frac=0.2)."""
    idx = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, stratify=y, random_state=seed
    )
    y_tv = y[idx_trainval]
    rel_val = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=rel_val, stratify=y_tv, random_state=seed
    )
    return idx_train, idx_val, idx_test


def apply_modal_length_from_reference(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int | None]:
    """Fix length to modal length computed from train split only (no leakage)."""
    lengths = train["rna_sequence"].str.len()
    if lengths.nunique() <= 1:
        mode_len = int(lengths.iloc[0]) if len(lengths) else None
        return train, val, test, mode_len
    mode_len = Counter(lengths).most_common(1)[0][0]
    mask = lambda df: df[df["rna_sequence"].str.len() == mode_len]
    return mask(train), mask(val), mask(test), mode_len


def load_data(
    path: Path,
    protein_col: str | None,
    rnacompete_best_experiment: bool,
    modal_length: bool,
) -> pd.DataFrame:
    opener = pd.read_csv
    df = opener(path, sep="\t", low_memory=False)

    if protein_col and protein_col in df.columns and protein_col != "protein_name":
        df = df.rename(columns={protein_col: "protein_name"})
    elif "target_name" in df.columns and "protein_name" not in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})

    required = {"rna_sequence", "binding_label", "protein_name"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing columns: {missing}")

    df["binding_label"] = pd.to_numeric(df["binding_label"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["rna_sequence", "binding_label", "protein_name"])
    df["binding_label"] = df["binding_label"].astype(int)

    if rnacompete_best_experiment:
        df = select_best_rnacompete_experiment(df)

    return df


def preprocess_protein(
    sub: pd.DataFrame, modal_length: bool
) -> tuple[pd.DataFrame, int | None]:
    if modal_length:
        return filter_modal_length(sub)
    mode_len = int(sub["rna_sequence"].str.len().mode().iloc[0]) if len(sub) else None
    return sub, mode_len


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def get_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    return models


def train_one_protein(
    sub: pd.DataFrame,
    protein: str,
    kmer_index: dict[str, int],
    k: int,
    test_frac: float,
    val_frac: float,
    seed: int,
    model_dir: Path,
    save_models: bool,
    honest: bool,
    dedupe: bool,
    modal_length: bool,
    modal_length_train_only: bool,
    min_examples: int,
) -> tuple[list[dict], dict | None]:
    """Train LR/RF/XGB for one protein. Returns comparison rows + best model info."""
    n_deduped = 0
    if dedupe:
        sub, n_deduped = dedupe_by_sequence(sub)

    n_pos = int((sub["binding_label"] == 1).sum())
    n_neg = int((sub["binding_label"] == 0).sum())
    if n_pos == 0 or n_neg == 0 or len(sub) < min_examples:
        return [], None

    y_all = sub["binding_label"].values

    from sklearn.base import clone

    if honest:
        try:
            idx_tr, idx_va, idx_te = stratified_three_way_split(
                len(sub), y_all, test_frac, val_frac, seed
            )
        except ValueError:
            return [], None

        train_df = sub.iloc[idx_tr].reset_index(drop=True)
        val_df = sub.iloc[idx_va].reset_index(drop=True)
        test_df = sub.iloc[idx_te].reset_index(drop=True)

        mode_len = None
        if modal_length:
            train_df, val_df, test_df, mode_len = apply_modal_length_from_reference(
                train_df, val_df, test_df
            )

        for part in (train_df, val_df, test_df):
            if part["binding_label"].nunique() < 2 or len(part) < 20:
                return [], None

        X_tr = encode_rna_matrix(train_df["rna_sequence"], kmer_index, k)
        X_va = encode_rna_matrix(val_df["rna_sequence"], kmer_index, k)
        X_te = encode_rna_matrix(test_df["rna_sequence"], kmer_index, k)
        y_tr = train_df["binding_label"].values
        y_va = val_df["binding_label"].values
        y_te = test_df["binding_label"].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        comparison_rows: list[dict] = []
        best_name: str | None = None
        best_val_auroc = -1.0
        best_val_auprc = -1.0
        best_template = None
        val_metrics_best: dict = {}
        test_metrics_best: dict = {}

        for name, template in get_models().items():
            m = clone(template)
            m.fit(X_tr_s, y_tr)
            prob_val = m.predict_proba(X_va_s)[:, 1]
            prob_te = m.predict_proba(X_te_s)[:, 1]
            val_m = compute_metrics(y_va, prob_val)
            test_m = compute_metrics(y_te, prob_te)
            comparison_rows.append(
                {
                    "protein_name": protein,
                    "model": name,
                    "val_auroc": val_m["auroc"],
                    "val_auprc": val_m["auprc"],
                    "test_auroc": test_m["auroc"],
                    "test_auprc": test_m["auprc"],
                    "n_train": len(y_tr),
                    "n_val": len(y_va),
                    "n_test": len(y_te),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n_deduped_removed": n_deduped,
                }
            )
            pick = (val_m["auroc"], val_m["auprc"])
            best_pick = (best_val_auroc, best_val_auprc)
            if pick > best_pick:
                best_val_auroc, best_val_auprc = pick
                best_name = name
                best_template = template
                val_metrics_best = val_m
                test_metrics_best = test_m

        if best_name is None or best_template is None:
            return comparison_rows, None

        # Refit best model type on train+val for deployment
        X_tv = np.vstack([X_tr, X_va])
        y_tv = np.concatenate([y_tr, y_va])
        scaler_final = StandardScaler()
        X_tv_s = scaler_final.fit_transform(X_tv)
        final_model = clone(best_template)
        final_model.fit(X_tv_s, y_tv)

        if save_models:
            prot_safe = protein.replace("/", "_")
            joblib.dump(scaler_final, model_dir / f"{prot_safe}_scaler.pkl")
            joblib.dump(
                {"model": final_model, "model_name": best_name, "k": k},
                model_dir / f"{prot_safe}_best.pkl",
            )

        summary = {
            "protein_name": protein,
            "best_model": best_name,
            "auroc": test_metrics_best["auroc"],
            "auprc": test_metrics_best["auprc"],
            "val_auroc": val_metrics_best["auroc"],
            "val_auprc": val_metrics_best["auprc"],
            "accuracy": test_metrics_best["accuracy"],
            "f1": test_metrics_best["f1"],
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_deduped_removed": n_deduped,
            "rna_length_mode": mode_len,
            "split_mode": "honest_60_20_20",
        }
        return comparison_rows, summary

    # Legacy simple split: 80/20 train/test, model selected on test (optimistic)
    if modal_length:
        sub, _ = filter_modal_length(sub)

    X = encode_rna_matrix(sub["rna_sequence"], kmer_index, k)
    y = sub["binding_label"].values

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_frac, stratify=y, random_state=seed
        )
    except ValueError:
        return [], None

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    comparison_rows = []
    best_name, best_auroc, best_model, best_metrics = None, -1.0, None, {}

    for name, template in get_models().items():
        m = clone(template)
        m.fit(X_tr_s, y_tr)
        prob = m.predict_proba(X_te_s)[:, 1]
        metrics = compute_metrics(y_te, prob)
        comparison_rows.append(
            {
                "protein_name": protein,
                "model": name,
                "val_auroc": np.nan,
                "val_auprc": np.nan,
                "test_auroc": metrics["auroc"],
                "test_auprc": metrics["auprc"],
                "n_train": len(y_tr),
                "n_val": 0,
                "n_test": len(y_te),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_deduped_removed": n_deduped,
            }
        )
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_name = name
            best_model = m
            best_metrics = metrics

    if best_model is None or best_name is None:
        return comparison_rows, None

    if save_models:
        prot_safe = protein.replace("/", "_")
        joblib.dump(scaler, model_dir / f"{prot_safe}_scaler.pkl")
        joblib.dump(
            {"model": best_model, "model_name": best_name, "k": k},
            model_dir / f"{prot_safe}_best.pkl",
        )

    summary = {
        "protein_name": protein,
        "best_model": best_name,
        "auroc": best_metrics["auroc"],
        "auprc": best_metrics["auprc"],
        "val_auroc": np.nan,
        "val_auprc": np.nan,
        "accuracy": best_metrics["accuracy"],
        "f1": best_metrics["f1"],
        "n_train": int(len(y_tr)),
        "n_val": 0,
        "n_test": int(len(y_te)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_deduped_removed": n_deduped,
        "split_mode": "simple_80_20",
    }
    return comparison_rows, summary


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    data_path = Path(args.data_file)
    out_root = Path(args.output_dir)
    dataset = args.dataset
    model_dir = out_root / dataset
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {data_path}")
    df = load_data(
        data_path,
        args.protein_col,
        args.rnacompete_best_experiment,
        modal_length=False,
    )
    print(f"  {len(df):,} rows, {df['protein_name'].nunique()} proteins")

    kmer_index = build_kmer_index(RNA_ALPHABET, args.kmer_k)
    proteins = sorted(df["protein_name"].unique())
    if args.max_proteins:
        proteins = proteins[: args.max_proteins]

    comparison_all: list[dict] = []
    summary_all: list[dict] = []
    skipped: list[str] = []

    split_label = "honest 60/20/20 + dedupe" if args.honest else "simple 80/20"
    print(
        f"\nDataset: {dataset}  |  k={args.kmer_k}  |  split: {split_label}"
        f"  |  models: LR, RF" + (", XGB" if HAS_XGB else "")
    )
    print("-" * 64)

    for prot in proteins:
        sub = df[df["protein_name"] == prot].copy()
        mode_len = None
        if not args.honest and args.modal_length:
            sub, mode_len = preprocess_protein(sub, modal_length=True)

        rows, summary = train_one_protein(
            sub,
            prot,
            kmer_index,
            args.kmer_k,
            args.test_frac,
            args.val_frac,
            args.seed,
            model_dir,
            save_models=not args.no_save_models,
            honest=args.honest,
            dedupe=args.dedupe,
            modal_length=args.modal_length,
            modal_length_train_only=args.honest,
            min_examples=args.min_examples,
        )
        if not summary:
            skipped.append(prot)
            continue

        summary["dataset"] = dataset
        if mode_len is not None:
            summary["rna_length_mode"] = mode_len
        comparison_all.extend(rows)
        summary_all.append(summary)
        print(
            f"  OK   {prot:40s}  best={summary['best_model']:22s}  "
            f"test_AUROC={summary['auroc']:.3f}  test_AUPRC={summary['auprc']:.3f}  "
            f"n={summary['n_train']+summary.get('n_val',0)+summary['n_test']}"
        )

    if not summary_all:
        print("\nWARN: no proteins trained.")
        return

    comp_df = pd.DataFrame(comparison_all)
    summ_df = pd.DataFrame(summary_all)

    comp_path = out_root / f"{dataset}_model_comparison.tsv"
    summ_path = out_root / f"{dataset}_per_protein_metrics.tsv"
    comp_df.to_csv(comp_path, sep="\t", index=False)
    summ_df.to_csv(summ_path, sep="\t", index=False)

    # Per-model aggregate stats (validation for selection; test for reporting)
    model_stats = {}
    for model_name in comp_df["model"].unique():
        sub = comp_df[comp_df["model"] == model_name]
        model_stats[model_name] = {
            "median_val_auroc": float(sub["val_auroc"].median()) if sub["val_auroc"].notna().any() else None,
            "median_val_auprc": float(sub["val_auprc"].median()) if sub["val_auprc"].notna().any() else None,
            "median_test_auroc": float(sub["test_auroc"].median()),
            "median_test_auprc": float(sub["test_auprc"].median()),
            "mean_test_auroc": float(sub["test_auroc"].mean()),
            "mean_test_auprc": float(sub["test_auprc"].mean()),
            "n_proteins": len(sub),
            "wins": int((summ_df["best_model"] == model_name).sum()),
        }

    stats = {
        "dataset": dataset,
        "data_file": portable_path(data_path),
        "split_mode": "honest_60_20_20" if args.honest else "simple_80_20",
        "dedupe_sequences": args.dedupe,
        "n_proteins_trained": len(summary_all),
        "n_proteins_skipped": len(skipped),
        "kmer_k": args.kmer_k,
        "model_comparison": model_stats,
        "recommended_model": max(
            model_stats, key=lambda m: model_stats[m]["median_test_auroc"] or 0
        ),
        "median_test_auroc": float(summ_df["auroc"].median()),
        "median_test_auprc": float(summ_df["auprc"].median()),
    }
    stats_path = out_root / f"{dataset}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nSummary: {summ_path}  ({len(summ_df)} proteins)")
    print(f"Comparison: {comp_path}")
    print(f"Stats: {stats_path}")
    print(
        f"\nHeld-out TEST  median AUROC={stats['median_test_auroc']:.3f}  "
        f"median AUPRC={stats['median_test_auprc']:.3f}"
    )
    print("Model comparison (median test AUROC / test AUPRC, wins on val):")
    for m, s in sorted(model_stats.items(), key=lambda x: -(x[1]["median_test_auroc"] or 0)):
        print(
            f"  {m:25s}  test_AUROC={s['median_test_auroc']:.3f}  "
            f"test_AUPRC={s['median_test_auprc']:.3f}  wins={s['wins']}"
        )

    # Merge into all_datasets_summary if multiple datasets exist
    merge_all_summaries(out_root)


def merge_all_summaries(out_root: Path) -> None:
    summaries = sorted(out_root.glob("*_per_protein_metrics.tsv"))
    if not summaries:
        return
    frames = [pd.read_csv(p, sep="\t") for p in summaries]
    master = pd.concat(frames, ignore_index=True)
    master_path = out_root / "all_datasets_summary.tsv"
    master.to_csv(master_path, sep="\t", index=False)
    print(f"\nMaster summary: {master_path}  ({len(master)} rows, {master['dataset'].nunique()} datasets)")


# ---------------------------------------------------------------------------
# Post-hoc scoring on extracted examples
# ---------------------------------------------------------------------------

DATASET_TO_DIR = {
    "HTR-SELEX": "htr_selex",
    "RBNS": "rbns",
    "RNAcompete_Eukarya": "rnacompete_eukarya",
    "RNAcompete_RBPZoo": "rnacompete_rbpzoo",
    "RNAcompete_ucRBP23": "rnacompete_ucrbp23",
}


def score_top_bottom_examples(examples_path: Path, models_root: Path) -> None:
    """Score extracted pos/neg examples with trained per-protein models."""
    df = pd.read_csv(examples_path, sep="\t")
    if "dataset" not in df.columns:
        sys.exit("ERROR: examples file needs 'dataset' column")

    kmer_index = build_kmer_index(RNA_ALPHABET, 4)
    results: list[dict] = []

    for (dataset, prot), grp in df.groupby(["dataset", "protein_name"]):
        dir_name = DATASET_TO_DIR.get(dataset)
        if not dir_name:
            continue
        prot_safe = str(prot).replace("/", "_")
        model_path = models_root / dir_name / f"{prot_safe}_best.pkl"
        scaler_path = models_root / dir_name / f"{prot_safe}_scaler.pkl"
        if not model_path.exists() or not scaler_path.exists():
            results.append(
                {
                    "dataset": dataset,
                    "protein_name": prot,
                    "status": "no_model",
                    "pos_median": np.nan,
                    "neg_median": np.nan,
                    "all_pos_gt_neg": False,
                }
            )
            continue

        bundle = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        model = bundle["model"]
        k = bundle.get("k", 4)

        X = encode_rna_matrix(grp["rna_sequence"], kmer_index, k)
        X_s = scaler.transform(X)
        scores = model.predict_proba(X_s)[:, 1]

        grp = grp.copy()
        grp["pred_score"] = scores
        pos_scores = grp[grp["split"] == "positive"]["pred_score"]
        neg_scores = grp[grp["split"] == "negative"]["pred_score"]

        pos_med = float(pos_scores.median()) if len(pos_scores) else np.nan
        neg_med = float(neg_scores.median()) if len(neg_scores) else np.nan
        all_pos_gt = (
            bool((pos_scores.min() > neg_scores.max()).item())
            if len(pos_scores) and len(neg_scores)
            else False
        )

        results.append(
            {
                "dataset": dataset,
                "protein_name": prot,
                "status": "ok",
                "pos_median": pos_med,
                "neg_median": neg_med,
                "pos_min": float(pos_scores.min()) if len(pos_scores) else np.nan,
                "neg_max": float(neg_scores.max()) if len(neg_scores) else np.nan,
                "all_pos_gt_neg": all_pos_gt,
                "pos_gt_neg_median": pos_med > neg_med if not np.isnan(pos_med) else False,
            }
        )

    res_df = pd.DataFrame(results)
    out_path = models_root / "top_bottom_scoring.tsv"
    res_df.to_csv(out_path, sep="\t", index=False)

    ok = res_df[res_df["status"] == "ok"]
    n_total = len(ok)
    n_median = int(ok["pos_gt_neg_median"].sum()) if n_total else 0
    n_strict = int(ok["all_pos_gt_neg"].sum()) if n_total else 0

    summary = {
        "n_proteins_scored": n_total,
        "n_no_model": int((res_df["status"] == "no_model").sum()),
        "frac_pos_median_gt_neg": n_median / n_total if n_total else 0,
        "frac_all_pos_gt_all_neg": n_strict / n_total if n_total else 0,
    }
    summary_path = models_root / "top_bottom_scoring_stats.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nTop/bottom scoring: {out_path}")
    print(f"  Scored: {n_total} proteins")
    print(f"  pos_median > neg_median: {n_median}/{n_total} ({summary['frac_pos_median_gt_neg']:.1%})")
    print(f"  all pos > all neg: {n_strict}/{n_total} ({summary['frac_all_pos_gt_all_neg']:.1%})")
    print(f"Stats: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data_file", default=None, help="Input clean TSV[.gz]")
    p.add_argument("--dataset", default=None, help="Dataset label (e.g. htr_selex, rbns)")
    p.add_argument("--output_dir", default="results/rna_only_per_protein_honest")
    p.add_argument("--protein_col", default=None, help="Protein column if not protein_name/target_name")
    p.add_argument("--kmer_k", type=int, default=4, help="RNA k-mer size (default: 4)")
    p.add_argument("--test_frac", type=float, default=0.2, help="Test fraction per protein (default: 0.2)")
    p.add_argument("--val_frac", type=float, default=0.2, help="Val fraction per protein (default: 0.2)")
    p.add_argument("--min_examples", type=int, default=100, help="Min rows per protein")
    p.add_argument(
        "--honest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dedupe sequences, 60/20/20 split, model select on val, report test AUROC+AUPRC (default: on)",
    )
    p.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate identical rna_sequence per protein before split (default: on)",
    )
    p.add_argument(
        "--simple_split",
        action="store_true",
        help="Legacy 80/20 train/test, model picked on test (disables --honest)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_proteins", type=int, default=None)
    p.add_argument(
        "--rnacompete_best_experiment",
        action="store_true",
        help="For RNAcompete: keep best experiment per protein by mean pos intensity",
    )
    p.add_argument(
        "--modal_length",
        action="store_true",
        help="Filter to modal RNA length when multiple lengths exist",
    )
    p.add_argument("--no_save_models", action="store_true", help="Skip saving per-protein model files")

    p.add_argument(
        "--score_examples",
        default=None,
        help="Path to all_protocols_summary.tsv for post-hoc scoring",
    )
    p.add_argument(
        "--models_dir",
        default="results/rna_only_per_protein",
        help="Root dir with trained models (for --score_examples)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.score_examples:
        score_top_bottom_examples(Path(args.score_examples), Path(args.models_dir))
        return

    if not args.data_file or not args.dataset:
        sys.exit("ERROR: --data_file and --dataset are required (unless --score_examples)")

    if args.simple_split:
        args.honest = False
        args.dedupe = False

    # RNAcompete Eukarya and similar multi-experiment data
    if args.rnacompete_best_experiment is False and "rnacompete" in (args.dataset or ""):
        args.rnacompete_best_experiment = True
    if args.modal_length is False and args.dataset in ("htr_selex", "rnacompete_eukarya"):
        args.modal_length = True

    run_training(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
