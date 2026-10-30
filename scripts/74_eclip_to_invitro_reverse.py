#!/usr/bin/env python3
"""
74_eclip_to_invitro_reverse.py
------------------------------
Reverse transfer: RNA models trained ONLY on the eCLIP/CLIP data of a protein (PaRPI clip_data, 101-nt windows) score the in vitro RNAs
(HTR-SELEX / RBNS / RNAcompete, v3b train+val+test rows) of the same protein. Per-protein AUROC of in vitro positives vs negatives.

For each of the ~50 proteins present in both resources, two k-mer logistic-regression models (k = 1..4 frequencies, script 59 features):
  genomic   eCLIP positives vs the dataset's own genomic negatives (label 0)  -> mixes protein signal and "peak-likeness"
  crossrbp  eCLIP positives vs positives of OTHER RBPs of the same cell line  -> composition-neutral, protein-specific signal
Reference points: per-protein in vitro k-mer LR with 5-fold CV on the in vitro rows themselves (seen-protein ceiling), and the
protein-independent RNA prior (mean in vitro model). Everything is protein-level, no neural network.

Usage
-----
  python scripts/74_eclip_to_invitro_reverse.py --out_dir results/reverse_transfer
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data/external/parpi/src_data/dataset/clip_data"


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load59():
    spec = importlib.util.spec_from_file_location("knn59", ROOT / "scripts" / "59_knn_protein_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def base(s: str) -> str:
    return re.sub(r"[-_ ]?(construct\d+|c\d+)$", "", str(s), flags=re.I).upper()


def auroc(y, s):
    return np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/generalized_v3b")
    ap.add_argument("--max_clip_rows", type=int, default=6000, help="eCLIP rows per class and protein for training")
    ap.add_argument("--max_other", type=int, default=12000, help="other-RBP positives per cell line (negatives of the cross-RBP model)")
    ap.add_argument("--max_invitro", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/reverse_transfer")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    k59 = load59()

    cols = ["protein_name", "rna_sequence", "binding_label", "dataset_source"]
    vit = pd.concat([pd.read_csv(resolve(args.data_dir) / f"{s}.tsv", sep="\t", usecols=cols, low_memory=False) for s in ("train", "val", "test")])
    vit = vit[~vit.dataset_source.astype(str).str.contains("eclip", case=False)]  # in vitro rows only (v3b train also holds eCLIP rows)
    vit["key"] = vit.protein_name.map(base)
    files = sorted(CLIP.glob("*.tsv"))
    by_prot: dict[str, list[Path]] = {}
    for f in files:
        by_prot.setdefault(f.stem.rsplit("_", 1)[0].upper(), []).append(f)
    shared = sorted(set(vit.key) & set(by_prot))
    print(f"{len(shared)} proteins in both resources")

    # positives of every dataset per cell line (negatives of the cross-RBP model), subsampled
    cell_pos: dict[str, dict[str, list[str]]] = {}
    for f in files:
        prot, cell = f.stem.rsplit("_", 1)
        cell_pos.setdefault(cell, {})
    rows, prior_rows = [], []
    z = np.load(resolve("results/hybrid") / "nn_models.npz", allow_pickle=False)
    for prot in shared:
        pos_seqs, gen_seqs, cells = [], [], []
        for f in by_prot[prot]:
            d = pd.read_csv(f, sep="\t", usecols=["Seq", "label"])
            pos_seqs += d.Seq[d.label == 1].tolist()
            gen_seqs += d.Seq[d.label == 0].tolist()
            cells.append(f.stem.rsplit("_", 1)[1])
        if len(pos_seqs) < 200:
            continue
        pos = [pos_seqs[i] for i in rng.choice(len(pos_seqs), min(len(pos_seqs), args.max_clip_rows), replace=False)]
        gen = [gen_seqs[i] for i in rng.choice(len(gen_seqs), min(len(gen_seqs), args.max_clip_rows), replace=False)]
        other = []
        for cell in set(cells):
            for f in sorted(CLIP.glob(f"*_{cell}.tsv")):
                if f.stem.rsplit("_", 1)[0].upper() == prot:
                    continue
                if len(other) >= args.max_other * len(set(cells)):
                    break
                d = pd.read_csv(f, sep="\t", usecols=["Seq", "label"])
                s = d.Seq[d.label == 1].tolist()
                other += [s[i] for i in rng.choice(len(s), min(len(s), 200), replace=False)]
        other = [other[i] for i in rng.choice(len(other), min(len(other), len(pos)), replace=False)]
        Xp, Xg, Xo = (k59.features(x)[0] for x in (pos, gen, other))
        m_gen = LogisticRegression(C=1.0, max_iter=300).fit(np.vstack([Xp, Xg]), np.r_[np.ones(len(Xp)), np.zeros(len(Xg))])
        m_cr = LogisticRegression(C=1.0, max_iter=300).fit(np.vstack([Xp, Xo]), np.r_[np.ones(len(Xp)), np.zeros(len(Xo))])

        v = vit[vit.key == prot]
        if len(v) > args.max_invitro:
            v = v.sample(args.max_invitro, random_state=args.seed)
        y = v.binding_label.to_numpy()
        if len(np.unique(y)) < 2 or min(y.sum(), len(y) - y.sum()) < 50:
            continue
        Xv = k59.features(v.rna_sequence.tolist())[0]
        r = {"protein": prot, "n_clip_pos": len(pos_seqs), "n_invitro": len(v), "sources": ",".join(sorted(v.dataset_source.unique())),
             "genomic": auroc(y, m_gen.decision_function(Xv)), "crossrbp": auroc(y, m_cr.decision_function(Xv)),
             "prior": auroc(y, Xv @ z["W_mean"] + z["B_mean"])}
        cv = cross_val_predict(LogisticRegression(C=1.0, max_iter=300), Xv, y, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
                               method="decision_function")
        r["invitro_own_cv"] = auroc(y, cv)
        rows.append(r)
        print(f"  {prot:<8} genomic {r['genomic']:.3f}  crossrbp {r['crossrbp']:.3f}  prior {r['prior']:.3f}  own_cv {r['invitro_own_cv']:.3f}  ({r['sources']})", flush=True)

    df = pd.DataFrame(rows)
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_protein.tsv", sep="\t", index=False)
    summ = {"n_proteins": len(df)}
    print(f"\n{len(df)} proteins; median AUROC on the in vitro rows of the same protein")
    for c in ("genomic", "crossrbp", "prior", "invitro_own_cv"):
        v = df[c].dropna().to_numpy()
        bm = [rng.choice(v, len(v)).mean() for _ in range(1000)]
        summ[c] = {"median": float(np.median(v)), "mean": float(v.mean()), "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                   "frac_gt_0.6": float((v > 0.6).mean())}
        print(f"  {c:<15} median {np.median(v):.3f}  mean {v.mean():.3f} [{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]  frac>0.6 {(v > 0.6).mean():.2f}")
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
