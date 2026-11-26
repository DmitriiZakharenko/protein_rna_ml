#!/usr/bin/env python3
"""
78_audit_coral_benchmark.py
---------------------------
Shortcut and leakage audit of the published CORAL benchmark folds (Zenodo 10.5281/zenodo.19730809, 25 folds: redundant, pairwise,
component-wise at two identity thresholds; positives from RNAInter and RPI sets, negatives = random re-pairings).

For every fold (train.csv / val.csv, columns RNA_id, Protein_id, RNA_seqs, Prot_seqs and a 0/1 label):
  overlap     share of validation rows whose protein id, RNA id, protein sequence, RNA sequence or whole pair occurs in train
  identity    share of validation rows whose best training hit (MMseqs2, query coverage >= 0.5) has identity >= 30 / 50 / 70 %
              for the protein and >= 50 / 80 % for the RNA (nucleotide search); needs `mmseqs` (--mmseqs path)
  baselines   AUROC train -> validation of scorers that see only one molecule or only counts:
                rna_only      logistic regression on RNA 3-mer frequencies
                prot_only     logistic regression on amino-acid and dipeptide frequencies
                concat_lr     logistic regression on both feature sets
                concat_hgb    gradient boosting on both feature sets
                degree        mean positive rate of the protein id and of the RNA id in train (0.5 when unseen, so exactly
                              0.5 AUROC when validation and train share no ids)

Usage
-----
  python scripts/78_audit_coral_benchmark.py --data_dir data/datasets --out_dir results/coral_audit
  python scripts/78_audit_coral_benchmark.py --data_dir data/datasets --folds componentwise_splits_60_80/fold_2161 --no_identity
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
LABELS = ["label", "Label", "labels", "interaction", "y", "Y", "class"]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_split(d: Path, name: str) -> pd.DataFrame:
    f = next((x for x in d.iterdir() if x.stem.lower() == name and x.suffix.lower() in (".csv", ".tsv", ".gz")), None)
    if f is None:
        raise FileNotFoundError(f"{name} not found in {d}")
    df = pd.read_csv(f, sep="\t" if f.suffix == ".tsv" else ",")
    lab = next((c for c in LABELS if c in df.columns), None)
    if lab is None:
        lab = next(c for c in df.columns if set(df[c].dropna().unique()) <= {0, 1})
    df = df.rename(columns={lab: "label"})
    df["RNA_seqs"] = df.RNA_seqs.str.upper().str.replace("U", "T")
    return df


def freq_features(seqs: pd.Series, alphabet: str, ks: tuple[int, ...]) -> np.ndarray:
    cols = [("".join(p)) for k in ks for p in itertools.product(alphabet, repeat=k)]
    idx = {c: i for i, c in enumerate(cols)}
    out = np.zeros((len(seqs), len(cols)), dtype=np.float32)
    for r, s in enumerate(seqs.values):
        for k in ks:
            for i in range(len(s) - k + 1):
                j = idx.get(s[i:i + k])
                if j is not None:
                    out[r, j] += 1
        out[r] /= max(1, out[r].sum())
    return out


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def baselines(tr: pd.DataFrame, va: pd.DataFrame, max_train: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if len(tr) > max_train:
        tr = tr.iloc[rng.choice(len(tr), max_train, replace=False)]
    res = {}
    r_tr, r_va = freq_features(tr.RNA_seqs, "ACGT", (3,)), freq_features(va.RNA_seqs, "ACGT", (3,))
    p_tr, p_va = freq_features(tr.Prot_seqs, AA, (1, 2)), freq_features(va.Prot_seqs, AA, (1, 2))
    y_tr, y_va = tr.label.to_numpy(), va.label.to_numpy()
    for name, (a, b) in {"rna_only": (r_tr, r_va), "prot_only": (p_tr, p_va),
                         "concat_lr": (np.hstack([r_tr, p_tr]), np.hstack([r_va, p_va]))}.items():
        sc = StandardScaler().fit(a)
        m = LogisticRegression(max_iter=300, C=0.5).fit(sc.transform(a), y_tr)
        res[name] = auroc(y_va, m.predict_proba(sc.transform(b))[:, 1])
    m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(np.hstack([r_tr, p_tr]), y_tr)
    res["concat_hgb"] = auroc(y_va, m.predict_proba(np.hstack([r_va, p_va]))[:, 1])
    # with no id overlap between train and validation (component-wise folds) every row scores 0.5, so the AUROC is 0.5 by construction
    pr = tr.groupby("Protein_id").label.mean()
    rr = tr.groupby("RNA_id").label.mean()
    res["degree"] = auroc(y_va, 0.5 * (va.Protein_id.map(pr).fillna(0.5) + va.RNA_id.map(rr).fillna(0.5)).to_numpy())
    return res


def overlap(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    train_pairs = set(zip(tr.Protein_id, tr.RNA_id))
    return {"protein_id_in_train": float(va.Protein_id.isin(set(tr.Protein_id)).mean()),
            "rna_id_in_train": float(va.RNA_id.isin(set(tr.RNA_id)).mean()),
            "protein_seq_in_train": float(va.Prot_seqs.isin(set(tr.Prot_seqs)).mean()),
            "rna_seq_in_train": float(va.RNA_seqs.isin(set(tr.RNA_seqs)).mean()),
            "pair_in_train": float(np.mean([p in train_pairs for p in zip(va.Protein_id, va.RNA_id)]))}


def best_identity(q: dict[str, str], t: dict[str, str], mmseqs: str, rna: bool, threads: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, d in (("q", q), ("t", t)):
            (td / f"{name}.fa").write_text("".join(f">{k}\n{v}\n" for k, v in d.items()))
        cmd = [mmseqs, "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "hits.m8"), str(td / "tmp"), "-e", "1e-3",
               "--max-seqs", "500", "--threads", str(threads), "--format-output", "query,target,fident,qcov"]
        cmd += ["--search-type", "3"] if rna else ["-s", "7.5"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        h = pd.read_csv(td / "hits.m8", sep="\t", names=["query", "target", "fident", "qcov"])
    h = h[h.qcov >= 0.5]
    return h.groupby("query").fident.max().to_dict()


def identity_shares(tr: pd.DataFrame, va: pd.DataFrame, mmseqs: str, threads: int) -> dict:
    out = {}
    for tag, id_col, seq_col, rna, cuts in (("protein", "Protein_id", "Prot_seqs", False, (0.3, 0.5, 0.7)),
                                              ("rna", "RNA_id", "RNA_seqs", True, (0.5, 0.8))):
        q = dict(zip(va[id_col], va[seq_col]))
        t = dict(zip(tr[id_col], tr[seq_col]))
        best = best_identity(q, t, mmseqs, rna, threads)
        b = va[id_col].map(best).fillna(0.0)
        for c in cuts:
            out[f"{tag}_identity_ge_{int(c * 100)}"] = float((b >= c).mean())
        out[f"{tag}_median_best_identity"] = float(b.median())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/datasets")
    ap.add_argument("--folds", nargs="*", default=None, help="<strategy>/<fold> paths under data_dir; default all")
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--no_identity", action="store_true")
    ap.add_argument("--max_train", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/coral_audit")
    args = ap.parse_args()

    root = resolve(args.data_dir)
    folds = args.folds or sorted(str(p.relative_to(root)) for p in root.glob("*/*") if p.is_dir())
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in folds:
        d = root / f
        tr, va = read_split(d, "train"), read_split(d, "val")
        row = {"strategy": f.split("/")[0], "fold": f.split("/")[-1], "n_train": len(tr), "n_val": len(va),
               "val_pos_frac": float(va.label.mean())}
        row.update(overlap(tr, va))
        if not args.no_identity:
            row.update(identity_shares(tr, va, args.mmseqs, args.threads))
        row.update({f"auroc_{k}": v for k, v in baselines(tr, va, args.max_train, args.seed).items()})
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        pd.DataFrame(rows).to_csv(out / "per_fold.tsv", sep="\t", index=False)
    df = pd.DataFrame(rows)
    num = df.drop(columns=["fold"]).groupby("strategy").agg(["mean", "std"]).round(3)
    num.to_csv(out / "per_strategy.tsv", sep="\t")
    print(num.T.to_string())


if __name__ == "__main__":
    main()
