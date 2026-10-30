#!/usr/bin/env python3
"""
70_prepare_nn_tables.py
-----------------------
Tables for the hybrid "nearest protein + residual learner" (script 71). Run locally (needs MMseqs2; the VM has none).

Writes results/hybrid/:
  nn_models.npz      names (n,), W (n, 340) k-mer LR weights (k=1..4 frequencies), B (n,), W_mean, B_mean
  neighbours.tsv     protein_name, split, nn_name, nn_ident, nn_cov   (best training neighbour; empty nn_name = no hit)

Training proteins get their best neighbour OUTSIDE their own cluster (cluster_id from results/v3b/protein_split_map.tsv), so
the neighbour is as remote as it is for held-out proteins. val/test proteins get their best training neighbour.

Usage
-----
  python scripts/70_prepare_nn_tables.py --data_dir data/generalized_v3b --split_map results/v3b/protein_split_map.tsv
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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


def search(q: dict[str, str], t: dict[str, str]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text("".join(f">{k}\n{v}\n" for k, v in q.items()))
        (td / "t.fa").write_text("".join(f">{k}\n{v}\n" for k, v in t.items()))
        subprocess.run(["mmseqs", "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "h.m8"), str(td / "tmp"),
                        "-s", "7.5", "-e", "1e-3", "--max-seqs", "1000",
                        "--format-output", "query,target,fident,qcov,tcov,bits"], check=True, stdout=subprocess.DEVNULL)
        h = pd.read_csv(td / "h.m8", sep="\t", names=["q", "t", "fident", "qcov", "tcov", "bits"])
    return h.sort_values(["q", "bits"], ascending=[True, False])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/generalized_v3b")
    ap.add_argument("--split_map", default="results/v3b/protein_split_map.tsv")
    ap.add_argument("--out_dir", default="results/hybrid")
    ap.add_argument("--max_rows_per_protein", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    k59 = load59()
    data = resolve(args.data_dir)
    cols = ["protein_name", "protein_sequence", "rna_sequence", "binding_label"]
    train = pd.read_csv(data / "train.tsv", sep="\t", usecols=cols, low_memory=False)
    prot_seq = dict(zip(train.protein_name, train.protein_sequence))
    names, Wl, Bl, _, _ = k59.fit_models(train, args.max_rows_per_protein, args.seed)
    print(f"fitted {len(names)} per-protein LR models")
    del train

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "nn_models.npz", names=np.array(names), W=Wl, B=Bl, W_mean=Wl.mean(0), B_mean=Bl.mean())

    cl = pd.read_csv(resolve(args.split_map), sep="\t").set_index("protein_name").cluster_id.to_dict()
    tgt = {n: prot_seq[n] for n in names}
    rows = []
    # train -> train, other cluster
    h = search({n: tgt[n] for n in names}, tgt)
    for q, g in h.groupby("q"):
        g = g[(g.t != q) & (g.t.map(cl.get) != cl.get(q))]
        if len(g):
            r = g.iloc[0]
            rows.append((q, "train", r.t, float(r.fident), float(r.qcov)))
    seen = {r[0] for r in rows}
    rows += [(n, "train", "", 0.0, 0.0) for n in names if n not in seen]
    # val / test -> train
    for split in ("val", "test"):
        df = pd.read_csv(data / f"{split}.tsv", sep="\t", usecols=["protein_name", "protein_sequence"], low_memory=False).drop_duplicates("protein_name")
        q = dict(zip(df.protein_name, df.protein_sequence))
        h = search(q, tgt).drop_duplicates("q")
        hit = {r.q: r for r in h.itertuples()}
        for n in q:
            if n in hit:
                rows.append((n, split, hit[n].t, float(hit[n].fident), float(hit[n].qcov)))
            else:
                rows.append((n, split, "", 0.0, 0.0))
    tab = pd.DataFrame(rows, columns=["protein_name", "split", "nn_name", "nn_ident", "nn_cov"])
    tab.to_csv(out / "neighbours.tsv", sep="\t", index=False)
    print(tab.groupby("split").agg(n=("protein_name", "size"), no_hit=("nn_name", lambda s: int((s == "").sum())),
                                   median_ident=("nn_ident", "median")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
