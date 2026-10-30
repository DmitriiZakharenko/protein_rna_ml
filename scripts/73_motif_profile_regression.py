#!/usr/bin/env python3
"""
73_motif_profile_regression.py
------------------------------
Protein sequence -> RNA motif profile, evaluated on held-out homology clusters.

Target of each protein: 5-mer log2 enrichment of its positive vs negative RNAs (1024 numbers, the profile of script 61).
Predictors (frozen ESM-2 mean embedding of the protein, script 66):
  prior   mean profile of the training proteins
  knn1    profile of the training protein with the most similar ESM-2 embedding (cosine); knn5 = similarity-weighted mean of 5
  ridge   multi-output ridge regression embedding -> profile (alpha chosen on the val proteins)
Metric: per protein, Spearman between predicted and true profile (median / mean with protein bootstrap CI) and top-20 overlap;
additionally, each predicted profile scores the RNAs of the cross-protein probe (mean predicted log-enrichment over the RNA's
5-mers) -> probe AUROC, comparable to the nearest-protein baseline (0.627) and the neural models.

Usage
-----
  python scripts/73_motif_profile_regression.py --out_dir results/v3b_profile
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
K = 5


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load59():
    spec = importlib.util.spec_from_file_location("knn59", ROOT / "scripts" / "59_knn_protein_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def profile(k59, rnas: list[str], y: np.ndarray, max_rows: int, rng) -> np.ndarray | None:
    if len(np.unique(y)) < 2:
        return None
    if len(rnas) > max_rows:
        ix = rng.choice(len(rnas), max_rows, replace=False)
        rnas, y = [rnas[i] for i in ix], y[ix]
    c = k59.kmer_counts(k59.encode(rnas), K)
    pos, neg = c[y == 1].sum(0) + 1.0, c[y == 0].sum(0) + 1.0
    return np.log2(pos / pos.sum()) - np.log2(neg / neg.sum())


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra, rb = np.apply_along_axis(rankdata, 1, a), np.apply_along_axis(rankdata, 1, b)
    ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-12)


def top20(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ta, tb = np.argsort(-a, axis=1)[:, :20], np.argsort(-b, axis=1)[:, :20]
    return np.array([len(set(x) & set(y)) / 20 for x, y in zip(ta, tb)])


def kmer_index(rna: str) -> np.ndarray:
    m = {c: i for i, c in enumerate("ACGU")}
    s = rna.upper().replace("T", "U")
    ids = [m.get(c, -1) for c in s]
    out = []
    for i in range(len(ids) - K + 1):
        w = ids[i:i + K]
        if min(w) >= 0:
            v = 0
            for x in w:
                v = v * 4 + x
            out.append(v)
    return np.array(out, dtype=np.int64)


def probe_auroc(df: pd.DataFrame, prof: dict[str, np.ndarray]) -> float:
    cache, sc = {}, np.zeros(len(df))
    for i, (p, r) in enumerate(zip(df.protein_name.values, df.rna_sequence.values)):
        if r not in cache:
            cache[r] = kmer_index(r)
        ix = cache[r]
        sc[i] = prof[p][ix].mean() if len(ix) else 0.0
    return float(roc_auc_score(df.binding_label.values, sc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/generalized_v3b")
    ap.add_argument("--emb_dir", default="data/embeddings_v3b")
    ap.add_argument("--probe_tsv", default="data/benchmarks/v3b_cross_neg/cross_neg_probe/test.tsv")
    ap.add_argument("--max_rows", type=int, default=30000)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/v3b_profile")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    k59 = load59()
    data = resolve(args.data_dir)
    cols = ["protein_name", "rna_sequence", "binding_label"]
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prof, split = {}, {}
    for sp in ("train", "val", "test"):
        df = pd.read_csv(data / f"{sp}.tsv", sep="\t", usecols=cols, low_memory=False)
        for p, g in df.groupby("protein_name"):
            pr = profile(k59, g.rna_sequence.tolist(), g.binding_label.to_numpy(), args.max_rows, rng)
            if pr is not None:
                prof[p], split[p] = pr, sp
        print(f"profiles {sp}: {sum(1 for v in split.values() if v == sp)}", flush=True)
    z = np.load(resolve(args.emb_dir) / "prot_esm2.npz", allow_pickle=False)
    emb = {n: e.astype(np.float32) for n, e in zip(z["names"], z["emb"])}
    names = [p for p in prof if p in emb]
    S = {sp: [p for p in names if split[p] == sp] for sp in ("train", "val", "test")}
    print({k: len(v) for k, v in S.items()})
    Y = {sp: np.stack([prof[p] for p in S[sp]]) for sp in S}
    E = {sp: np.stack([emb[p] for p in S[sp]]) for sp in S}
    mu, sd = E["train"].mean(0), E["train"].std(0) + 1e-6
    Z = {sp: (E[sp] - mu) / sd for sp in S}
    unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)  # noqa: E731
    sim = {sp: unit(Z[sp]) @ unit(Z["train"]).T for sp in ("val", "test")}

    def predict(sp: str, kind: str, alpha: float = 1.0) -> np.ndarray:
        if kind == "prior":
            return np.tile(Y["train"].mean(0), (len(S[sp]), 1))
        if kind in ("knn1", "knn5"):
            k = 1 if kind == "knn1" else 5
            o = np.argsort(-sim[sp], axis=1)[:, :k]
            w = np.take_along_axis(sim[sp], o, 1).clip(1e-3)
            return (Y["train"][o] * (w / w.sum(1, keepdims=True))[:, :, None]).sum(1)
        return Ridge(alpha=alpha).fit(Z["train"], Y["train"]).predict(Z[sp])

    alphas = [1, 10, 100, 1000, 3000, 10000, 30000, 100000]
    val_sp = {a: float(np.median(spearman_rows(predict("val", "ridge", a), Y["val"]))) for a in alphas}
    best_a = max(val_sp, key=val_sp.get)
    print("ridge alpha selection on val (median Spearman):", {a: round(v, 3) for a, v in val_sp.items()}, "-> alpha", best_a)

    res = {"n": {k: len(v) for k, v in S.items()}, "ridge_alpha": best_a, "val_alpha_scan": val_sp, "test": {}}
    probe = pd.read_csv(resolve(args.probe_tsv), sep="\t", usecols=["protein_name", "rna_sequence", "binding_label"])
    probe = probe[probe.protein_name.isin(S["test"])]
    print(f"\n{'predictor':<8}{'Spearman median':>16}{'mean [95% CI]':>26}{'top20':>8}{'probe AUROC':>13}")
    for kind in ("prior", "knn1", "knn5", "ridge"):
        P = predict("test", kind, best_a)
        sp = spearman_rows(P, Y["test"])
        bm = [rng.choice(sp, len(sp)).mean() for _ in range(args.n_boot)]
        au = probe_auroc(probe, dict(zip(S["test"], P)))
        res["test"][kind] = {"spearman_median": float(np.median(sp)), "spearman_mean": float(sp.mean()),
                             "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                             "top20": float(top20(P, Y["test"]).mean()), "probe_auroc": au}
        print(f"{kind:<8}{np.median(sp):>16.3f}{sp.mean():>14.3f} [{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]"
              f"{top20(P, Y['test']).mean():>8.3f}{au:>13.3f}", flush=True)
        pd.DataFrame({"protein": S["test"], "spearman": sp}).to_csv(out / f"per_protein_{kind}.tsv", sep="\t", index=False)
    # oracle: the protein's own profile scoring its probe rows (upper reference for the 5-mer-profile scorer)
    res["test"]["oracle_own_profile"] = {"probe_auroc": probe_auroc(probe, dict(zip(S["test"], Y["test"])))}
    print(f"oracle (true own profile) probe AUROC: {res['test']['oracle_own_profile']['probe_auroc']:.3f}")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    np.savez(out / "profiles.npz", names=np.array(names), split=np.array([split[p] for p in names]), profile=np.stack([prof[p] for p in names]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
