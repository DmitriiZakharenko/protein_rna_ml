#!/usr/bin/env python3
"""
60_rpi_benchmark_shortcut_audit.py
----------------------------------
Run the classic RPI benchmarks (RPI369 / RPI488 / RPI1446 / RPI2241, files in data/external/rpi_ggcn) with an
RPISeq-style model (RNA 4-mer + protein 3-mer in a 7-letter alphabet, Random Forest) and with single-molecule
shortcut controls:

  rna_only      RNA 4-mer composition only
  protein_only  protein 3-mer composition only
  both          RPISeq-style concatenation
  label_prior   per-protein / per-RNA positive rate learnt on the training folds (pure identity memorisation)

Folds: random 5-fold (as in the original papers), protein-disjoint (GroupKFold by protein sequence) and
RNA-disjoint. If a single-molecule baseline matches `both` under random folds, the benchmark does not
test interaction.

Usage:  python scripts/60_rpi_benchmark_shortcut_audit.py
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/external/rpi_ggcn"
GROUPS = {"A": "AGV", "I": "ILMV".replace("V", ""), "Y": "FWY"}
# conjoint-triad 7-class alphabet
CLASSES = ["AGV", "ILFP", "YMTS", "HNQW", "RK", "DE", "C"]
AA2C = {a: i for i, g in enumerate(CLASSES) for a in g}
RNA_K = {"".join(p): i for i, p in enumerate(product("ACGU", repeat=4))}
PRO_K = {"".join(p): i for i, p in enumerate(product("0123456", repeat=3))}


def read_fasta(p: Path) -> dict[str, str]:
    out, k = {}, None
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            k = line[1:].strip()
            out[k] = ""
        elif k:
            out[k] += line.strip()
    return out


def load(name: str) -> pd.DataFrame:
    d = D / name
    if name == "RPI488":
        rows = [l.split("\t") for l in (d / "RPI488_pairs.fa").read_text().splitlines() if l.strip()]
        return pd.DataFrame([(r[0], r[0], int(r[1]), r[2], r[3]) for r in rows], columns=["pid", "rid", "label", "prot", "rna"])
    if name == "RPI369":
        rows = [l.split("\t") for l in (d / "RPI369_pairs.fa").read_text().splitlines() if len(l.split("\t")) == 5]
        return pd.DataFrame([(r[0], r[1], int(r[2]), r[3], r[4]) for r in rows], columns=["pid", "rid", "label", "prot", "rna"])
    pf = d / f"{name}_all_NP_pairs.fa" if name == "RPI2241" else d / f"{name}_pairs.fa"
    prot = read_fasta(next(d.glob("*protein.fa")))
    rna = read_fasta(next(d.glob("*[rR][nN][aA].fa")))
    rows = []
    for l in pf.read_text().splitlines():
        f = l.split("\t")
        if len(f) == 3 and f[0] in prot and f[1] in rna:
            rows.append((f[0], f[1], int(f[2]), prot[f[0]], rna[f[1]]))
    return pd.DataFrame(rows, columns=["pid", "rid", "label", "prot", "rna"])


def rna_feat(s: str) -> np.ndarray:
    s = s.upper().replace("T", "U")
    v = np.zeros(256, dtype=np.float32)
    for i in range(len(s) - 3):
        j = RNA_K.get(s[i:i + 4])
        if j is not None:
            v[j] += 1
    return v / max(v.sum(), 1)


def prot_feat(s: str) -> np.ndarray:
    c = "".join(str(AA2C.get(a, 0)) for a in s.upper())
    v = np.zeros(343, dtype=np.float32)
    for i in range(len(c) - 2):
        v[PRO_K[c[i:i + 3]]] += 1
    return v / max(v.sum(), 1)


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def cv_scores(df, XR, XP, folds):
    y = df.label.to_numpy()
    res = {k: np.zeros(len(df)) for k in ("rna_only", "protein_only", "both", "label_prior")}
    for tr, te in folds:
        for k, X in (("rna_only", XR), ("protein_only", XP), ("both", np.hstack([XR, XP]))):
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
            res[k][te] = rf.predict_proba(X[te])[:, 1]
        # label prior: mean label of the same protein (else same RNA) seen in training; global mean otherwise
        d = df.iloc[tr]
        pm, rm, g = d.groupby("prot").label.mean(), d.groupby("rna").label.mean(), y[tr].mean()
        res["label_prior"][te] = [(pm.get(p) if p in pm.index else (rm.get(r) if r in rm.index else g))
                                  for p, r in zip(df.prot.iloc[te], df.rna.iloc[te])]
    return {k: round(auc(y, v), 3) for k, v in res.items()}


def main() -> None:
    out = {}
    print(f"{'dataset':<9}{'folds':<18}{'n':>6}{'#prot':>7}{'#rna':>6}  {'rna_only':>9}{'prot_only':>10}{'both':>7}{'label_prior':>12}")
    for name in ("RPI369", "RPI488", "RPI1446", "RPI2241"):
        df = load(name).drop_duplicates(["prot", "rna"]).reset_index(drop=True)
        XR = np.stack([rna_feat(s) for s in df.rna])
        XP = np.stack([prot_feat(s) for s in df.prot])
        y = df.label.to_numpy()
        pg = df.prot.factorize()[0]
        rg = df.rna.factorize()[0]
        splits = {
            "random 5-fold": list(StratifiedKFold(5, shuffle=True, random_state=0).split(XR, y)),
            "protein-disjoint": list(GroupKFold(5).split(XR, y, pg)),
            "RNA-disjoint": list(GroupKFold(5).split(XR, y, rg)),
        }
        out[name] = {"n": len(df), "n_pos": int(y.sum()), "n_proteins": int(df.prot.nunique()), "n_rna": int(df.rna.nunique())}
        for sp, folds in splits.items():
            r = cv_scores(df, XR, XP, folds)
            out[name][sp] = r
            print(f"{name:<9}{sp:<18}{len(df):>6}{df.prot.nunique():>7}{df.rna.nunique():>6}  {r['rna_only']:>9}{r['protein_only']:>10}{r['both']:>7}{r['label_prior']:>12}", flush=True)
    p = ROOT / "results/rpi_benchmark_audit"
    p.mkdir(parents=True, exist_ok=True)
    (p / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
