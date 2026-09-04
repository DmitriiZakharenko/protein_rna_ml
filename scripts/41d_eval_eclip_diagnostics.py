#!/usr/bin/env python3
"""
41d_eval_eclip_diagnostics.py
-----------------------------
Diagnostic evaluation for Skipper eCLIP benchmarks.

Reports:
  - Composition baselines (GC%, AU fraction) without a neural model
  - Full-test CNN metrics (optional, if --checkpoint set)
  - RNA-unseen subset: test rows whose RNA never appeared in train
  - RNA-seen subset (leakage check)
  - GC-matched negatives: subsample negs per protein to match pos GC

Works on the Jose-style protein-disjoint model without retraining (post-hoc),
or on any train/test TSV pair after a new split (41c).

Usage
-----
  # Composition + subsets only (no GPU)
  python scripts/41d_eval_eclip_diagnostics.py \\
    --train_tsv data/benchmarks/skipper_eclip/jose_style/train.tsv \\
    --test_tsv data/benchmarks/skipper_eclip/jose_style/test.tsv \\
    --out_dir results/skipper_eclip/jose_style_diagnostics

  # Include V2 CNN scores
  python scripts/41d_eval_eclip_diagnostics.py \\
    --train_tsv data/benchmarks/skipper_eclip/jose_style/train.tsv \\
    --test_tsv data/benchmarks/skipper_eclip/jose_style/test.tsv \\
    --checkpoint models/saved/skipper_eclip_v2_rna151/best_model.pt \\
    --rna_max 151 --prot_max 700 \\
    --out_dir results/skipper_eclip/jose_style_diagnostics

  # Reuse pre-scored pairs (skip inference)
  python scripts/41d_eval_eclip_diagnostics.py \\
    --train_tsv ... --test_tsv ... \\
    --scored_tsv results/skipper_eclip/jose_style_v2_train/external_pairs_scored.tsv \\
    --prob_col prob_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cnn_model import RNABindingCNN

_eval_path = ROOT / "scripts" / "11_evaluate_external.py"
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("evaluate_external_11", _eval_path)
_eval11 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval11)
score_pair = _eval11.score_pair


def resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def gc_fraction(seq: str) -> float:
    s = str(seq).upper()
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def au_fraction(seq: str) -> float:
    s = str(seq).upper()
    if not s:
        return 0.0
    return (s.count("A") + s.count("U")) / len(s)


def safe_auroc_auprc(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(labels) == 0:
        return {"n": 0, "auroc": None, "auprc": None, "note": "empty"}
    if len(set(labels)) < 2:
        return {"n": int(len(labels)), "auroc": None, "auprc": None, "note": "single_class"}
    return {
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "auroc": round(float(roc_auc_score(labels, scores)), 4),
        "auprc": round(float(average_precision_score(labels, scores)), 4),
    }


def per_protein_summary(df: pd.DataFrame, score_col: str, label_col: str = "binding_label") -> dict:
    rows = []
    for prot, g in df.groupby("protein_name"):
        labs = g[label_col].values
        if len(set(labs)) < 2:
            continue
        m = safe_auroc_auprc(labs, g[score_col].values)
        m["protein"] = prot
        rows.append(m)
    if not rows:
        return {"n_proteins": 0, "median_auroc": None, "median_auprc": None, "min_auroc": None}
    aurocs = [r["auroc"] for r in rows if r["auroc"] is not None]
    auprcs = [r["auprc"] for r in rows if r["auprc"] is not None]
    return {
        "n_proteins": len(rows),
        "median_auroc": round(float(np.median(aurocs)), 4) if aurocs else None,
        "median_auprc": round(float(np.median(auprcs)), 4) if auprcs else None,
        "min_auroc": round(float(np.min(aurocs)), 4) if aurocs else None,
        "per_protein": rows,
    }


def gc_matched_subset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Per protein: pair each positive with the closest-GC unused negative."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, g in df.groupby("protein_name"):
        pos = g[g["binding_label"] == 1].sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        neg = g[g["binding_label"] == 0].copy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        picked_neg_idx: list = []
        for _, prow in pos.iterrows():
            if neg.empty:
                break
            neg["gc_dist"] = (neg["gc"] - prow["gc"]).abs()
            best_idx = neg["gc_dist"].idxmin()
            picked_neg_idx.append(best_idx)
            neg = neg.drop(index=best_idx)
        if not picked_neg_idx:
            continue
        neg_pick = g.loc[picked_neg_idx]
        n = min(len(pos), len(neg_pick))
        parts.append(pd.concat([pos.iloc[:n], neg_pick.iloc[:n]], ignore_index=True))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def load_test_with_features(test_path: Path) -> pd.DataFrame:
    df = pd.read_csv(test_path, sep="\t", low_memory=False)
    if "binding_label" not in df.columns:
        raise SystemExit(f"test TSV missing binding_label: {test_path}")
    df["gc"] = df["rna_sequence"].map(gc_fraction)
    df["au_frac"] = df["rna_sequence"].map(au_fraction)
    return df


def score_with_checkpoint(
    df: pd.DataFrame,
    checkpoint: Path,
    rna_max: int,
    prot_max: int,
    device: torch.device,
    win_step: int = 30,
) -> np.ndarray:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = RNABindingCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = []
    n = len(df)
    for i, row in df.iterrows():
        if (len(probs) + 1) % 500 == 0 or len(probs) == 0:
            print(f"  Scoring {len(probs)+1}/{n}...", end="\r", flush=True)
        p = score_pair(
            model,
            row["rna_sequence"],
            row["protein_sequence"],
            rna_max,
            prot_max,
            device,
            win_step,
        )
        probs.append(p)
    print(f"  Scored {n} pairs.          ")
    return np.array(probs)


def evaluate_block(
    df: pd.DataFrame,
    score_col: str,
    name: str,
    *,
    include_per_protein: bool = False,
) -> dict:
    block = {
        "name": name,
        "aggregate": safe_auroc_auprc(df["binding_label"].values, df[score_col].values),
        "gc_pos_mean": round(100.0 * float(df.loc[df["binding_label"] == 1, "gc"].mean()), 1)
        if (df["binding_label"] == 1).any()
        else None,
        "gc_neg_mean": round(100.0 * float(df.loc[df["binding_label"] == 0, "gc"].mean()), 1)
        if (df["binding_label"] == 0).any()
        else None,
    }
    if include_per_protein:
        block["per_protein"] = per_protein_summary(df, score_col)
    else:
        pp = per_protein_summary(df, score_col)
        block["per_protein_summary"] = {
            k: pp[k]
            for k in ("n_proteins", "median_auroc", "median_auprc", "min_auroc")
        }
    return block


def main() -> None:
    ap = argparse.ArgumentParser(description="eCLIP benchmark diagnostics (41d)")
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--val_tsv", default=None)
    ap.add_argument("--checkpoint", default=None, help="V2 checkpoint for CNN scores")
    ap.add_argument("--scored_tsv", default=None, help="Pre-scored test TSV with prob column")
    ap.add_argument("--prob_col", default="prob_v2")
    ap.add_argument("--rna_max", type=int, default=151)
    ap.add_argument("--prot_max", type=int, default=700)
    ap.add_argument("--win_step", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/skipper_eclip/diagnostics")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    train_path = resolve(args.train_tsv)
    test_path = resolve(args.test_tsv)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rnas = set(
        pd.read_csv(train_path, sep="\t", usecols=["rna_sequence"])["rna_sequence"]
    )
    test_df = load_test_with_features(test_path)
    test_df["rna_in_train"] = test_df["rna_sequence"].isin(train_rnas)

    if args.scored_tsv:
        scored = pd.read_csv(resolve(args.scored_tsv), sep="\t", low_memory=False)
        if args.prob_col not in scored.columns:
            raise SystemExit(f"Column {args.prob_col!r} not in {args.scored_tsv}")
        merge_cols = ["protein_name", "rna_sequence", "binding_label"]
        merge_cols = [c for c in merge_cols if c in scored.columns]
        test_df = test_df.drop(columns=[c for c in [args.prob_col] if c in test_df.columns])
        test_df = test_df.merge(
            scored[merge_cols + [args.prob_col]],
            on=merge_cols,
            how="left",
            validate="m:1",
        )
        if test_df[args.prob_col].isna().any():
            n_miss = int(test_df[args.prob_col].isna().sum())
            print(f"  Warning: {n_miss} test rows missing scores after merge")
    elif args.checkpoint:
        if torch.cuda.is_available() and not args.no_cuda:
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"  Device: {device}")
        test_df[args.prob_col] = score_with_checkpoint(
            test_df,
            resolve(args.checkpoint),
            args.rna_max,
            args.prot_max,
            device,
            args.win_step,
        )
    else:
        args.prob_col = None

    results: dict = {
        "script": "41d_eval_eclip_diagnostics.py",
        "train_tsv": str(train_path),
        "test_tsv": str(test_path),
        "checkpoint": str(resolve(args.checkpoint)) if args.checkpoint else None,
        "n_test": int(len(test_df)),
        "rna_leakage": {
            "n_test_rows": int(len(test_df)),
            "n_test_rows_rna_in_train": int(test_df["rna_in_train"].sum()),
            "pct_test_rows_rna_in_train": round(
                100.0 * test_df["rna_in_train"].mean(), 2
            ),
            "n_unique_test_rnas": int(test_df["rna_sequence"].nunique()),
            "n_unique_test_rnas_in_train": int(
                test_df.loc[test_df["rna_in_train"], "rna_sequence"].nunique()
            ),
        },
        "composition_baselines": {},
        "subsets": {},
    }

    print("\n=== Composition baselines (test) ===")
    for feat, col in [("gc_fraction", "gc"), ("au_fraction", "au_frac")]:
        block = evaluate_block(test_df, col, feat)
        results["composition_baselines"][feat] = block
        agg = block["aggregate"]
        pp = block["per_protein_summary"]
        print(
            f"  {feat:<14} aggregate AUROC={agg['auroc']}  "
            f"pp-median={pp['median_auroc']}  "
            f"(GC% pos={block['gc_pos_mean']} neg={block['gc_neg_mean']})"
        )

    gc_matched = gc_matched_subset(test_df, seed=args.seed)
    print(f"\n=== GC-matched negatives (test, n={len(gc_matched):,}) ===")
    for feat, col in [("gc_fraction", "gc"), ("au_fraction", "au_frac")]:
        block = evaluate_block(gc_matched, col, f"gc_matched_{feat}")
        results["composition_baselines"][f"gc_matched_{feat}"] = block
        agg = block["aggregate"]
        pp = block["per_protein_summary"]
        print(
            f"  {feat:<14} aggregate AUROC={agg['auroc']}  pp-median={pp['median_auroc']}"
        )

    if args.prob_col:
        print(f"\n=== CNN ({args.prob_col}) ===")
        subsets = {
            "full_test": test_df,
            "rna_unseen": test_df[~test_df["rna_in_train"]],
            "rna_seen": test_df[test_df["rna_in_train"]],
            "gc_matched": gc_matched,
        }
        for key, sub in subsets.items():
            if len(sub) < 20:
                print(f"  {key}: skipped (n={len(sub)})")
                continue
            block = evaluate_block(sub, args.prob_col, key)
            results["subsets"][key] = block
            agg = block["aggregate"]
            pp = block["per_protein_summary"]
            print(
                f"  {key:<14} n={agg['n']:>6,}  "
                f"AUROC={agg['auroc']}  AUPRC={agg['auprc']}  "
                f"pp-med={pp['median_auroc']}"
            )

        out_scored = out_dir / "test_pairs_scored.tsv"
        cols = [
            c
            for c in test_df.columns
            if c not in ("protein_sequence",) or c == args.prob_col
        ]
        test_df[cols].to_csv(out_scored, sep="\t", index=False)
        results["scored_tsv"] = str(out_scored)

    out_json = out_dir / "eclip_diagnostics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Wrote {out_json}")


if __name__ == "__main__":
    main()
