#!/usr/bin/env python3
"""
63_parpi_rna_only_audit.py
--------------------------
RNA-only controls on the PaRPI training data (Zenodo 10.5281/zenodo.14878562, `dataset/clip_data/<RBP>_<cell>.tsv`;
5000 top-peak positives, 10000 random transcriptome-window negatives, 101 nt).

For every dataset (261) a k-mer logistic regression (k = 1..4 frequencies, no protein information at all) is trained on
80% of the windows and tested on 20% (the random stratified split of the original code), reporting AUROC for

  random        the original protocol (random 80/20 split of windows),
  transcript    windows of the same transcript never in both train and test (GroupKFold-style single holdout),
  cross_rbp     negatives replaced by the positives of OTHER RBPs profiled in the same cell line (windows that are also
                positive for the target RBP are removed): this asks whether the RNA window alone identifies WHICH RBP
                binds, and has composition-matched negatives,
  gc            GC content alone on the original task.

Usage:  python scripts/63_parpi_rna_only_audit.py [--max_datasets N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/parpi/src_data/dataset/clip_data"
spec = importlib.util.spec_from_file_location("knn59", ROOT / "scripts/59_knn_protein_baseline.py")
knn = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = knn
spec.loader.exec_module(knn)


def load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, sep="\t", usecols=["Name", "Seq", "label"])
    d["Seq"] = d.Seq.str.upper().str.replace("T", "U")
    d["tx"] = d.Name.str.split("|").str[0]
    return d


def fit_eval(Xtr, ytr, Xte, yte) -> float:
    lr = LogisticRegression(C=1.0, max_iter=300).fit(Xtr, ytr)
    return float(roc_auc_score(yte, lr.decision_function(Xte)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_datasets", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = sorted(DATA.glob("*.tsv"))
    if args.max_datasets:
        files = files[: args.max_datasets]
    rng = np.random.default_rng(args.seed)
    data = {f.stem: load(f) for f in files}
    cell = {k: k.rsplit("_", 1)[1] for k in data}
    pos_pool = {k: d[d.label == 1] for k, d in data.items()}
    rows = []
    for i, (k, d) in enumerate(data.items()):
        X = knn.features(d.Seq.tolist())[0]
        y = d.label.to_numpy()
        r = {"dataset": k, "cell": cell[k], "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
        tr, te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=args.seed)
        r["random"] = fit_eval(X[tr], y[tr], X[te], y[te])
        gc = d.Seq.map(lambda s: (s.count("G") + s.count("C")) / max(len(s), 1)).to_numpy()
        r["gc"] = float(roc_auc_score(y[te], gc[te]))
        # transcript holdout
        txs = d.tx.unique()
        hold = set(rng.choice(txs, max(int(0.2 * len(txs)), 1), replace=False))
        te2 = np.where(d.tx.isin(hold))[0]
        tr2 = np.where(~d.tx.isin(hold))[0]
        r["transcript"] = fit_eval(X[tr2], y[tr2], X[te2], y[te2]) if len(set(y[te2])) > 1 and len(set(y[tr2])) > 1 else np.nan
        # cross-RBP negatives within the cell line
        others = [o for o in data if cell[o] == cell[k] and o != k]
        if others:
            own = set(d.loc[d.label == 1, "Seq"])
            pool = pd.concat([pos_pool[o] for o in others]).drop_duplicates("Seq")
            pool = pool[~pool.Seq.isin(own)]
            if len(pool) >= 1000:
                neg = pool.sample(min(len(pool), 10000), random_state=args.seed)
                pos = d[d.label == 1]
                dd = pd.concat([pos, neg])
                yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                XX = knn.features(dd.Seq.tolist())[0]
                a, b = train_test_split(np.arange(len(dd)), test_size=0.2, stratify=yy, random_state=args.seed)
                r["cross_rbp"] = fit_eval(XX[a], yy[a], XX[b], yy[b])
                r["n_other_rbps"] = len(others)
        rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(data)}  last {k}: random {r['random']:.3f} cross_rbp {r.get('cross_rbp', float('nan')):.3f}", flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / "results/parpi_audit"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "rna_only_per_dataset.tsv", sep="\t", index=False)
    summ = {c: {"median": float(res[c].median()), "mean": float(res[c].mean()), "n": int(res[c].notna().sum())}
            for c in ("random", "transcript", "cross_rbp", "gc")}
    summ["by_cell"] = res.groupby("cell")[["random", "transcript", "cross_rbp"]].median().round(3).to_dict("index")
    (out / "rna_only_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "by_cell"}, indent=2))
    print(res.groupby("cell")[["random", "transcript", "cross_rbp"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
