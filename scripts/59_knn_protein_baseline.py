#!/usr/bin/env python3
"""
59_knn_protein_baseline.py
--------------------------
"Nearest training protein" baseline: the key control for any claim that a model has learned
protein x RNA interaction rather than family/homology memorisation.

For every training protein a small RNA-only model is fitted on its own pairs
(`lr`: logistic regression on k-mer frequencies k=1..4; `pwm`: naive-Bayes log-odds on
4-mer counts, i.e. a position-independent PWM). A test pair (protein P, RNA r) is scored by
the models of the training proteins most similar to P (MMseqs2 identity; weights = identity):

  nn1   best-identity neighbour
  nn3   top-3 neighbours, identity-weighted mean of logits
  nn5   top-5 neighbours
  mean  mean over ALL training proteins (protein-independent RNA prior; also the fallback
        for test proteins with no MMseqs2 hit)

No protein information beyond sequence similarity is used, so on a cross-protein-negative probe
(same RNA, different protein) any AUROC above 0.5 is pure homology transfer.

Usage
-----
  python scripts/59_knn_protein_baseline.py --data_dir data/generalized_v3b \\
      --eval standard=data/generalized_v3b/test.tsv \\
      --eval probe=data/benchmarks/v3b_cross_neg/cross_neg_probe/test.tsv \\
      --identity_tsv results/homology_audit/v3b_test_protein_identity.tsv --out_name v3b
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0.0, "<30%"), (0.30, "30-50%"), (0.50, "50-70%"), (0.70, "70-90%"), (0.90, ">=90%")]  # lower edges; <30% includes no hit
LUT = np.full(256, 4, dtype=np.uint8)
for i, c in enumerate("ACGU"):
    LUT[ord(c)] = i
LUT[ord("T")] = 3


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def encode(seqs: list[str]) -> np.ndarray:
    L = max(len(s) for s in seqs)
    arr = np.full((len(seqs), L), 4, dtype=np.uint8)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = LUT[np.frombuffer(s.upper().encode(), dtype=np.uint8)]
    return arr


def kmer_counts(codes: np.ndarray, k: int) -> np.ndarray:
    n, L = codes.shape
    out = np.zeros((n, 4 ** k), dtype=np.float32)
    if L < k:
        return out
    idx = np.zeros((n, L - k + 1), dtype=np.int64)
    valid = np.ones((n, L - k + 1), dtype=bool)
    for j in range(k):
        c = codes[:, j: L - k + 1 + j].astype(np.int64)
        valid &= c < 4
        idx = idx * 4 + np.minimum(c, 3)
    rows = np.repeat(np.arange(n), L - k + 1).reshape(n, -1)
    np.add.at(out, (rows[valid], idx[valid]), 1.0)
    return out


def features(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(freq k=1..4 [340], counts k=4 [256])"""
    codes = encode(seqs)
    parts, c4 = [], None
    for k in (1, 2, 3, 4):
        c = kmer_counts(codes, k)
        if k == 4:
            c4 = c
        parts.append(c / np.maximum(c.sum(1, keepdims=True), 1.0))
    return np.concatenate(parts, axis=1), c4


def fit_models(train: pd.DataFrame, max_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    W_lr, B_lr, W_nb, B_nb, names = [], [], [], [], []
    for name, g in train.groupby("protein_name"):
        if g.binding_label.nunique() < 2:
            continue
        if len(g) > max_rows:
            g = g.iloc[rng.choice(len(g), max_rows, replace=False)]
        X, c4 = features(g.rna_sequence.tolist())
        y = g.binding_label.to_numpy()
        lr = LogisticRegression(C=1.0, max_iter=300).fit(X, y)
        W_lr.append(lr.coef_[0].astype(np.float32)); B_lr.append(float(lr.intercept_[0]))
        a = 1.0
        pos, neg = c4[y == 1].sum(0) + a, c4[y == 0].sum(0) + a
        w = np.log(pos / pos.sum()) - np.log(neg / neg.sum())
        W_nb.append(w.astype(np.float32)); B_nb.append(0.0)
        names.append(name)
    return names, np.stack(W_lr), np.array(B_lr), np.stack(W_nb), np.array(B_nb)


def mmseqs_neighbours(q: dict[str, str], t: dict[str, str], topn: int = 5) -> dict[str, list[tuple[str, float]]]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text("".join(f">{k}\n{v}\n" for k, v in q.items()))
        (td / "t.fa").write_text("".join(f">{k}\n{v}\n" for k, v in t.items()))
        subprocess.run(["mmseqs", "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "h.m8"), str(td / "tmp"),
                        "-s", "7.5", "-e", "1e-3", "--max-seqs", "1000",
                        "--format-output", "query,target,fident,qcov,tcov,bits"], check=True, stdout=subprocess.DEVNULL)
        h = pd.read_csv(td / "h.m8", sep="\t", names=["q", "t", "fident", "qcov", "tcov", "bits"])
    h = h.sort_values(["q", "bits"], ascending=[True, False]).groupby("q").head(topn)
    return {k: list(zip(g.t, g.fident)) for k, g in h.groupby("q")}


def auroc(y, s):
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def pp_median(df: pd.DataFrame, col: str):
    v = [auroc(g.binding_label, g[col]) for _, g in df.groupby("protein_name")]
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def boot_ci(df: pd.DataFrame, col: str, n: int, rng) -> list[float] | None:
    prots = df.protein_name.unique()
    if len(prots) < 5 or n <= 0:
        return None
    groups = {p: g for p, g in df.groupby("protein_name")}
    vals = []
    for _ in range(n):
        s = pd.concat([groups[p] for p in rng.choice(prots, len(prots))])
        a = auroc(s.binding_label, s[col])
        if a is not None:
            vals.append(a)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--eval", action="append", required=True, help="name=path (tsv with protein_name, protein_sequence, rna_sequence, binding_label)")
    ap.add_argument("--identity_tsv", default=None, help="script 55 table for identity-bin stratification")
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--max_rows_per_protein", type=int, default=6000)
    ap.add_argument("--max_eval_rows", type=int, default=150000)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = resolve(args.data_dir)
    cols = ["protein_name", "protein_sequence", "rna_sequence", "binding_label"]
    train = pd.read_csv(data / "train.tsv", sep="\t", usecols=cols, low_memory=False)
    train_prot = dict(zip(train.protein_name, train.protein_sequence))
    print(f"train: {len(train):,} rows, {len(train_prot)} proteins")
    names, Wl, Bl, Wn, Bn = fit_models(train, args.max_rows_per_protein, args.seed)
    idx = {n: i for i, n in enumerate(names)}
    print(f"fitted {len(names)} per-protein models")
    Wl_mean, Bl_mean, Wn_mean = Wl.mean(0), Bl.mean(), Wn.mean(0)
    del train

    ident = None
    if args.identity_tsv:
        ident = pd.read_csv(resolve(args.identity_tsv), sep="\t").set_index("protein_name")

    summary = {"data_dir": args.data_dir, "n_train_proteins": len(names), "evals": {}}
    for spec in args.eval:
        ename, path = spec.split("=", 1)
        df = pd.read_csv(resolve(path), sep="\t", usecols=cols, low_memory=False)
        if len(df) > args.max_eval_rows:
            df = df.sample(args.max_eval_rows, random_state=args.seed).reset_index(drop=True)
        uniq = df.drop_duplicates("protein_name")
        nb = mmseqs_neighbours(dict(zip(uniq.protein_name, uniq.protein_sequence)), {k: train_prot[k] for k in names})
        X, c4 = features(df.rna_sequence.tolist())
        df["mean_lr"] = X @ Wl_mean + Bl_mean
        df["mean_pwm"] = c4 @ Wn_mean
        for col in ("nn1_lr", "nn3_lr", "nn5_lr", "nn1_pwm", "nn3_pwm", "nn5_pwm"):
            df[col] = np.nan
        for p, ix in df.groupby("protein_name").indices.items():
            hits = [(idx[t], f) for t, f in nb.get(p, []) if t in idx]
            for m in (1, 3, 5):
                if not hits:
                    df.loc[ix, f"nn{m}_lr"] = df.loc[ix, "mean_lr"].to_numpy()
                    df.loc[ix, f"nn{m}_pwm"] = df.loc[ix, "mean_pwm"].to_numpy()
                    continue
                sel = hits[:m]
                w = np.array([f for _, f in sel]) + 1e-6
                w = w / w.sum()
                df.loc[ix, f"nn{m}_lr"] = sum(wi * (X[ix] @ Wl[j] + Bl[j]) for wi, (j, _) in zip(w, sel))
                df.loc[ix, f"nn{m}_pwm"] = sum(wi * (c4[ix] @ Wn[j]) for wi, (j, _) in zip(w, sel))
        df["best_identity"] = df.protein_name.map(lambda p: nb[p][0][1] if p in nb and nb[p] else 0.0)
        scorers = ["mean_lr", "nn1_lr", "nn3_lr", "nn5_lr", "mean_pwm", "nn1_pwm", "nn3_pwm", "nn5_pwm"]
        res = {"n_rows": int(len(df)), "n_proteins": int(df.protein_name.nunique()), "overall": {}, "by_identity_bin": {}}
        print(f"\n== {ename} ({res['n_rows']:,} rows, {res['n_proteins']} proteins) ==")
        print(f"{'scorer':<10}{'AUROC':>8}  {'95% CI (protein bootstrap)':<24}{'pp-median':>10}")
        for sc in scorers:
            a = auroc(df.binding_label, df[sc])
            ci = boot_ci(df, sc, args.n_boot, rng)
            res["overall"][sc] = {"auroc": a, "ci": ci, "pp_median": pp_median(df, sc)}
            print(f"{sc:<10}{a:>8.4f}  {str(ci):<24}{res['overall'][sc]['pp_median']:>10.4f}")
        edges = [b[0] for b in BINS]
        df["bin"] = pd.cut(df.best_identity, bins=edges + [1.01], labels=[b[1] for b in BINS], right=False, include_lowest=True)
        print(f"\n{'bin':<9}{'n_prot':>7}{'n_rows':>9}" + "".join(f"{s:>9}" for s in ("mean_lr", "nn1_lr", "nn5_lr", "nn1_pwm")))
        for b, sub in df.groupby("bin", observed=True):
            r = {s: auroc(sub.binding_label, sub[s]) for s in scorers}
            res["by_identity_bin"][str(b)] = {"n_proteins": int(sub.protein_name.nunique()), "n_rows": int(len(sub)), **r}
            print(f"{str(b):<9}{sub.protein_name.nunique():>7}{len(sub):>9}" + "".join(
                f"{r[s]:>9.3f}" if r[s] is not None else f"{'n/a':>9}" for s in ("mean_lr", "nn1_lr", "nn5_lr", "nn1_pwm")))
        summary["evals"][ename] = res
        out_dir = resolve("results/knn_baseline")
        out_dir.mkdir(parents=True, exist_ok=True)
        df[["protein_name", "binding_label", "best_identity"] + scorers].to_csv(
            out_dir / f"{args.out_name}_{ename}_scores.tsv.gz", sep="\t", index=False)

    out = resolve("results/knn_baseline")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.out_name}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out}/{args.out_name}_summary.json")


if __name__ == "__main__":
    main()
