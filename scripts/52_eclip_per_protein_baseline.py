#!/usr/bin/env python3
"""
52_eclip_per_protein_baseline.py
--------------------------------
Per-protein RNA k-mer logistic regression with grouped cross-validation on
eCLIP pair TSVs. It estimates how much protein-specific RNA signal exists when
the protein is *seen* (an upper-bound check for the pooled protein-disjoint
models in scripts 06/21/51).

Run it on both negative types:
  fixlen_151_all.tsv                   genomic non-peak negatives
  fixlen_151_cross_protein_neg_all.tsv other proteins' positives as negatives

Usage
-----
  python scripts/52_eclip_per_protein_baseline.py \\
      --pairs_tsv data/benchmarks/skipper_eclip/fixlen_151_cross_protein_neg_all.tsv \\
      --out_json results/skipper_eclip/per_protein_cross_protein_neg.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def cv_auroc(df: pd.DataFrame, k_max: int, n_splits: int) -> float | None:
    if df["binding_label"].nunique() < 2 or len(df) < 2 * n_splits:
        return None
    rna = df["rna_sequence"].str.upper().str.replace("T", "U")
    x = CountVectorizer(analyzer="char", ngram_range=(1, k_max), lowercase=False).fit_transform(rna)
    x = x.multiply(1.0 / rna.str.len().values[:, None]).tocsr()
    y = df["binding_label"].values
    oof = np.full(len(df), np.nan)
    # group by RNA so identical windows never straddle folds
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups=df["rna_sequence"].values):
        if len(set(y[tr])) < 2:
            return None
        sc = StandardScaler(with_mean=False).fit(x[tr])
        clf = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(x[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(x[te]))[:, 1]
    return float(roc_auc_score(y, oof))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs_tsv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--k_max", type=int, default=4)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.pairs_tsv if not Path(args.pairs_tsv).is_absolute() else args.pairs_tsv,
                     sep="\t", usecols=["protein_name", "rna_sequence", "binding_label"])
    rows = []
    for prot, g in df.groupby("protein_name"):
        a = cv_auroc(g.reset_index(drop=True), args.k_max, args.n_splits)
        if a is not None:
            rows.append({"protein": prot, "n": int(len(g)), "auroc": round(a, 4)})
    vals = np.array([r["auroc"] for r in rows])
    summary = {
        "pairs_tsv": args.pairs_tsv,
        "k_max": args.k_max,
        "n_proteins": len(rows),
        "median_auroc": round(float(np.median(vals)), 4),
        "mean_auroc": round(float(vals.mean()), 4),
        "frac_gt_0.6": round(float((vals > 0.6).mean()), 3),
        "frac_gt_0.7": round(float((vals > 0.7).mean()), 3),
        "per_protein": rows,
    }
    out = Path(args.out_json)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print({k: v for k, v in summary.items() if k != "per_protein"})


if __name__ == "__main__":
    main()
