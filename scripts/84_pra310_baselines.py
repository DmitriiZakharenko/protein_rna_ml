#!/usr/bin/env python3
"""Script 84: sequence-only baselines for protein-RNA binding free energy on PRA310 and PRA201 (tables of the HARP-RPA authors).

Per split file and fold, models are fitted on the training complexes and scored on the held-out complexes: Pearson (PCC) and
Spearman (SCC) correlation between predicted and measured free energy, mean and sd over the five folds (the metric of the paper).

Models
  nn_protein_k   mean free energy of the complexes of the k most similar training proteins (MMseqs2 bit score), training mean without hit
  nn_rna_k       the same with the k most similar training RNAs (cosine of 3-mer frequencies)
  ridge_rna      ridge regression on RNA 3-mer frequencies and log length
  ridge_protein  ridge regression on amino-acid composition and log length
  ridge_both     both feature sets
  mean           training mean (PCC is 0 by construction, reported as the reference of the RMSE)

Split files (data/external/harp_rpa/): PRA310.csv (benchmark folds, CD-HIT clusters at 70% identity), PRA310_sequence_aware_p50.csv
(protein identity below 50% between folds) and PRA310_S.csv (folds by free-energy bins, the same protein can be in training and
validation). Data licence: not stated by the authors, so the tables are not tracked in git.

Usage (project root; MMseqs2 required)
  python scripts/84_pra310_baselines.py
  python scripts/84_pra310_baselines.py audit    # share of validation complexes with a similar or identical training protein
  python scripts/84_pra310_baselines.py selftest
"""
from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
RN = "ACGU"
K3 = ["".join(x) for x in itertools.product(RN, repeat=3)]
SPLITS = {"PRA310 benchmark folds (CD-HIT 70%)": "PRA310.csv", "PRA310 protein identity below 50%": "PRA310_sequence_aware_p50.csv",
          "PRA310 free-energy-stratified folds (protein overlap)": "PRA310_S.csv", "PRA201 benchmark folds": "PRA201.csv",
          "PRA201 free-energy-stratified folds (protein overlap)": "PRA201_S.csv"}


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load(path: Path) -> pd.DataFrame:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    df = pd.DataFrame(rows)
    df["P"] = df["Protein sequences"].str.split(":", n=1).str[-1].str.upper()
    df["R"] = df["RNA sequences"].str.split(":", n=1).str[-1].str.upper().str.replace("T", "U")
    df["dg"] = df["△G(kcal/mol)"].astype(float)
    return df


def prot_feat(seqs) -> np.ndarray:
    return np.array([[s.count(a) / max(len(s), 1) for a in AA] + [np.log(len(s) + 1)] for s in seqs])


def rna_feat(seqs) -> np.ndarray:
    out = []
    for s in seqs:
        n = max(len(s) - 2, 1)
        out.append([sum(1 for i in range(len(s) - 2) if s[i:i + 3] == k) / n for k in K3] + [np.log(len(s) + 1)])
    return np.array(out)


def mmseqs_hits(q: list[str], t: list[str], threads: int = 4) -> dict[str, list[tuple[str, float]]]:
    """For each query protein the training proteins sorted by bit score."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(q)))
        (td / "t.fa").write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(t)))
        subprocess.run(["mmseqs", "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "h.m8"), str(td / "tmp"), "-e", "1e-3", "--max-seqs", "500",
                        "--threads", str(threads), "--format-output", "query,target,bits,qcov", "-s", "7.5"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        best: dict[int, dict[int, float]] = {}
        for line in open(td / "h.m8"):
            a, b, bits, cov = line.split("\t")
            if float(cov) >= 0.5:
                best.setdefault(int(a), {})
                best[int(a)][int(b)] = max(best[int(a)].get(int(b), 0), float(bits))
    return {q[i]: sorted(((t[j], v) for j, v in d.items()), key=lambda x: -x[1]) for i, d in best.items()}


def best_identity(q: list[str], t: list[str], threads: int = 4) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text("".join(f">{i}\n{x}\n" for i, x in enumerate(q)))
        (td / "t.fa").write_text("".join(f">{i}\n{x}\n" for i, x in enumerate(t)))
        subprocess.run(["mmseqs", "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "h.m8"), str(td / "tmp"), "-e", "1e-3", "--max-seqs", "500",
                        "--threads", str(threads), "--format-output", "query,target,fident,qcov", "-s", "7.5"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        b: dict[int, float] = {}
        for line in open(td / "h.m8"):
            a, _, f, c = line.split("\t")
            if float(c) >= 0.5:
                b[int(a)] = max(b.get(int(a), 0.0), float(f))
    return {q[i]: b.get(i, 0.0) for i in range(len(q))}


def fold_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Share of validation complexes with a training protein at the given identity and with an identical training RNA, per fold."""
    rows = []
    for f in [c for c in df.columns if c.startswith("fold_")]:
        tr, va = df[df[f] == "train"], df[df[f] == "val"]
        bi = best_identity(list(va.P.unique()), list(tr.P.unique()))
        v = np.array([bi[p] for p in va.P])
        rows.append(dict(fold=f, n_val=len(va), ident30=float((v >= 0.3).mean()), ident50=float((v >= 0.5).mean()), ident70=float((v >= 0.7).mean()),
                         identical_protein=float((v >= 0.999).mean()), identical_rna=float(va.R.isin(set(tr.R)).mean())))
    return pd.DataFrame(rows)


def score(y, p) -> tuple[float, float]:
    if np.std(p) < 1e-12 or len(y) < 3:
        return 0.0, 0.0
    return float(pearsonr(y, p)[0]), float(spearmanr(y, p)[0])


def run_split(df: pd.DataFrame, ks=(1, 3), use_mmseqs: bool = True) -> pd.DataFrame:
    folds = [c for c in df.columns if c.startswith("fold_")]
    rows = []
    for f in folds:
        tr, va = df[df[f] == "train"], df[df[f] == "val"]
        y, ytr = va.dg.to_numpy(), tr.dg.to_numpy()
        preds = {"mean": np.full(len(va), ytr.mean())}
        if use_mmseqs:
            hits = mmseqs_hits(list(va.P.unique()), list(tr.P.unique()))
            by_prot = tr.groupby("P").dg.mean()
            for k in ks:
                preds[f"nn_protein_{k}"] = np.array([np.mean([by_prot[t] for t, _ in hits.get(p, [])[:k]]) if hits.get(p) else ytr.mean() for p in va.P])
        Xr_tr, Xr_va = rna_feat(tr.R), rna_feat(va.R)
        for k in ks:
            c = (Xr_va[:, :-1] / np.maximum(np.linalg.norm(Xr_va[:, :-1], axis=1, keepdims=True), 1e-9)) @ (Xr_tr[:, :-1] / np.maximum(np.linalg.norm(Xr_tr[:, :-1], axis=1, keepdims=True), 1e-9)).T
            top = np.argsort(-c, axis=1)[:, :k]
            preds[f"nn_rna_{k}"] = ytr[top].mean(1)
        Xp_tr, Xp_va = prot_feat(tr.P), prot_feat(va.P)
        for name, A, B in (("ridge_rna", Xr_tr, Xr_va), ("ridge_protein", Xp_tr, Xp_va), ("ridge_both", np.hstack([Xr_tr, Xp_tr]), np.hstack([Xr_va, Xp_va]))):
            mu, sd = A.mean(0), A.std(0) + 1e-9
            m = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit((A - mu) / sd, ytr)
            preds[name] = m.predict((B - mu) / sd)
        for name, p in preds.items():
            pcc, scc = score(y, p)
            rows.append(dict(fold=f, model=name, pcc=pcc, scc=scc, rmse=float(np.sqrt(np.mean((p - y) ** 2))), n_val=len(va)))
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("model")
    return pd.DataFrame(dict(pcc_mean=g.pcc.mean(), pcc_sd=g.pcc.std(), scc_mean=g.scc.mean(), scc_sd=g.scc.std(), rmse_mean=g.rmse.mean())).reset_index()


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 120
    P = ["".join(rng.choice(list(AA), 80)) for _ in range(n)]
    R = ["".join(rng.choice(list(RN), 12)) for _ in range(n)]
    dg = np.array([-5 - 10 * sum(s.count("A") for s in [r]) / 12 for r in R]) + rng.normal(0, 0.3, n)
    fold = np.arange(n) % 5
    df = pd.DataFrame(dict(P=P, R=R, dg=dg))
    for f in range(5):
        df[f"fold_{f}"] = np.where(fold == f, "val", "train")
    res = summarise(run_split(df, use_mmseqs=False))
    r = res.set_index("model")
    assert r.loc["ridge_rna", "pcc_mean"] > 0.8, "planted RNA effect not recovered"
    assert abs(r.loc["mean", "pcc_mean"]) < 1e-9, "mean model has correlation"
    assert score(np.arange(5.0), np.arange(5.0))[0] > 0.999
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="run", choices=["run", "audit", "selftest"])
    ap.add_argument("--data_dir", default="data/external/harp_rpa")
    ap.add_argument("--out_dir", default="results/pra310_baselines")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        rows = []
        for name, fn in SPLITS.items():
            f = resolve(args.data_dir) / fn
            if f.exists():
                a = fold_audit(load(f))
                rows.append(a.drop(columns=["fold", "n_val"]).mean().to_frame().T.assign(split=name))
        res = pd.concat(rows)[["split", "ident30", "ident50", "ident70", "identical_protein", "identical_rna"]]
        res.to_csv(out / "fold_audit.tsv", sep="\t", index=False)
        print(res.round(3).to_string(index=False))
        return
    allres = []
    for name, fn in SPLITS.items():
        f = resolve(args.data_dir) / fn
        if not f.exists():
            print("missing", f, file=sys.stderr)
            continue
        df = load(f)
        res = run_split(df)
        res.insert(0, "split", name)
        allres.append(res)
        s = summarise(res).round(3)
        print(f"== {name}: {len(df)} complexes, folds {sorted(res.fold.unique())}")
        print(s.to_string(index=False))
    pd.concat(allres).to_csv(out / "per_fold.tsv", sep="\t", index=False)
    pd.concat([summarise(r).assign(split=r.split.iloc[0]) for r in allres]).to_csv(out / "summary.tsv", sep="\t", index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
