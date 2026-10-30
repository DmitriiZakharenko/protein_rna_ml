#!/usr/bin/env python3
"""
72_affinity_probe_intensity.py
------------------------------
Affinity-style test on RNAcompete probe intensities of the v3b test proteins: for each test protein, Spearman correlation between a
protein-agnostic or homology-transferred RNA score and the measured probe intensity (all probes, not only the binary labels).

Scorers (all use only RNA sequences + the training proteins; nothing sees the test protein's own data except `own_cv`):
  prior      mean over all training proteins' k-mer LR models (protein-independent RNA prior)
  nn1        k-mer LR of the nearest training protein (script 70 neighbours; mean model if no hit)
  own_cv     ridge regression on k-mer features fitted on the protein's OWN probes, 5-fold by probe: an upper reference that
             needs the protein's own measurements (not available for a new protein)
Per-protein Spearman -> median / mean over proteins with a protein bootstrap CI, and by MMseqs2 identity bin.
Optional --scores TSV(s) with protein_name, rna_sequence, score add further scorers (e.g. neural models scoring the same probes).

Usage
-----
  python scripts/72_affinity_probe_intensity.py --out_dir results/v3b_affinity
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load59():
    spec = importlib.util.spec_from_file_location("knn59", ROOT / "scripts" / "59_knn_protein_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnacompete", default="data/benchmarks/rnacompete/rnacompete_all.tsv")
    ap.add_argument("--test_tsv", default="data/benchmarks/v3b_cross_neg/cross_neg_probe/test.tsv", help="only its protein names are used")
    ap.add_argument("--nn_dir", default="results/hybrid")
    ap.add_argument("--identity_tsv", default="results/homology_audit/v3b_test_protein_identity.tsv")
    ap.add_argument("--scores", action="append", default=[], help="name=path.tsv with protein_name, rna_sequence, score")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/v3b_affinity")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    k59 = load59()
    test_prots = set(pd.read_csv(resolve(args.test_tsv), sep="\t", usecols=["protein_name"]).protein_name)
    d = pd.read_csv(resolve(args.rnacompete), sep="\t", usecols=["protein_name", "rna_sequence", "probe_intensity"])
    d = d[d.protein_name.isin(test_prots)]
    d = d.groupby(["protein_name", "rna_sequence"], as_index=False).probe_intensity.mean()  # average replicate experiments
    z = np.load(resolve(args.nn_dir) / "nn_models.npz", allow_pickle=False)
    names = list(z["names"])
    W, B = z["W"], z["B"]
    nb = pd.read_csv(resolve(args.nn_dir) / "neighbours.tsv", sep="\t", keep_default_na=False)
    nb = nb[nb.split == "test"].set_index("protein_name")
    ident = pd.read_csv(resolve(args.identity_tsv), sep="\t").set_index("protein_name").identity_bin.to_dict()
    extra = {}
    for spec in args.scores:
        n, p = spec.split("=", 1)
        extra[n] = pd.read_csv(resolve(p), sep="\t")

    rows = []
    for prot, g in d.groupby("protein_name"):
        if len(g) < 200:
            continue
        y = g.probe_intensity.to_numpy()
        X = k59.features(g.rna_sequence.tolist())[0]
        r = {"protein": prot, "n_probes": len(g), "identity_bin": ident.get(prot, "NA")}
        nn_name = nb.nn_name.get(prot, "") if prot in nb.index else ""
        s_prior = X @ z["W_mean"] + z["B_mean"]
        s_nn = X @ W[names.index(nn_name)] + B[names.index(nn_name)] if nn_name in names else s_prior
        r["prior"] = float(spearmanr(s_prior, y).statistic)
        r["nn1"] = float(spearmanr(s_nn, y).statistic)
        cv = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=args.seed).split(X):
            cv[te] = Ridge(alpha=10.0).fit(X[tr], np.argsort(np.argsort(y[tr]))).predict(X[te])
        r["own_cv"] = float(spearmanr(cv, y).statistic)
        for n, t in extra.items():
            m = g[["rna_sequence"]].merge(t[t.protein_name == prot][["rna_sequence", "score"]].drop_duplicates("rna_sequence"), on="rna_sequence", how="left")
            ok = m.score.notna().to_numpy()
            r[n] = float(spearmanr(m.score[ok], y[ok]).statistic) if ok.sum() > 100 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    scorers = [c for c in df.columns if c not in ("protein", "n_probes", "identity_bin")]
    summ = {"n_proteins": len(df), "scorers": {}, "by_identity_bin": {}}
    print(f"{len(df)} test proteins with RNAcompete intensities")
    print(f"{'scorer':<10}{'median':>8}{'mean':>8}  95% CI of the mean (protein bootstrap)")
    for sc in scorers:
        v = df[sc].dropna().to_numpy()
        bm = [rng.choice(v, len(v)).mean() for _ in range(args.n_boot)]
        summ["scorers"][sc] = {"median": float(np.median(v)), "mean": float(v.mean()),
                               "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], "n": int(len(v))}
        print(f"{sc:<10}{np.median(v):>8.3f}{v.mean():>8.3f}  [{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]")
    for b, g in df.groupby("identity_bin"):
        summ["by_identity_bin"][b] = {"n": int(len(g)), **{sc: float(g[sc].mean()) for sc in scorers}}
        print(f"  bin {b:<8} n={len(g):>3}  " + "  ".join(f"{sc} {g[sc].mean():.3f}" for sc in scorers))
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_protein.tsv", sep="\t", index=False)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
