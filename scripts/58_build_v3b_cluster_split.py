#!/usr/bin/env python3
"""
58_build_v3b_cluster_split.py
-----------------------------
Re-split the v3a pairs by protein CLUSTERS (MMseqs2 all-vs-all, single linkage)
instead of protein names, so that test/val proteins have no homolog linked by >=40% identity (>=50% coverage) or >=70% identity (>=10% coverage) in train. Linking at 30% identity chains all RRM proteins into one cluster (>400 proteins), so 40% is the strictest usable threshold.
Clusters are assigned to train/val/test stratified by Pfam family (script 56 labels), with
~12% of rows in test and ~11% in val.

Stages
------
  split : cluster all v3a proteins, write results/v3b/protein_split_map.tsv + summary
  build : write data/generalized_v3b/{train,val,test}.tsv from the map (needs v3a tsvs)
  --holdout_family KH  puts every cluster of that family into test (for leave-family-out);
                       output goes to data/generalized_v3b_lofo_<FAMILY>/ and results/v3b_lofo_<FAMILY>/

Usage
-----
  python scripts/58_build_v3b_cluster_split.py split
  python scripts/58_build_v3b_cluster_split.py build
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3A = ROOT / "data/generalized_v3a"
FAM = ROOT / "results/homology_audit/v3a_pfam_family.tsv"
COLS = ["protein_name", "protein_sequence"]


def v3a_proteins() -> pd.DataFrame:
    frames = []
    for sp in ("train", "val", "test"):
        df = pd.read_csv(V3A / f"{sp}.tsv", sep="\t", usecols=COLS)
        g = df.groupby("protein_name").agg(protein_sequence=("protein_sequence", "first"), n_rows=("protein_name", "size"))
        g["v3a_split"] = sp
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def cluster(prots: pd.DataFrame, min_id: float, cov: float, hi_id: float = 0.7) -> pd.Series:
    """Single-linkage clusters from an all-vs-all MMseqs2 search.

    Two proteins are linked if identity >= min_id with query or target coverage >= cov, or
    identity >= hi_id with coverage >= 0.1 (short shared domains of close paralogs, e.g. PABP).
    Greedy `easy-cluster` is not transitive and left near-identical proteins in different clusters.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "p.fa").write_text("".join(f">{n}\n{s}\n" for n, s in zip(prots.protein_name, prots.protein_sequence)))
        subprocess.run(["mmseqs", "easy-search", str(td / "p.fa"), str(td / "p.fa"), str(td / "h.m8"), str(td / "tmp"),
                        "-s", "7.5", "-e", "1e-3", "--max-seqs", "2000",
                        "--format-output", "query,target,fident,qcov,tcov"], check=True, stdout=subprocess.DEVNULL)
        h = pd.read_csv(td / "h.m8", sep="\t", names=["q", "t", "fident", "qcov", "tcov"])
    mx = h[["qcov", "tcov"]].max(axis=1)
    link = h[((h.fident >= min_id) & (mx >= cov)) | ((h.fident >= hi_id) & (mx >= 0.1))]
    parent = {n: n for n in prots.protein_name}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for q, t in zip(link.q, link.t):
        rq, rt = find(q), find(t)
        if rq != rt:
            parent[rq] = rt
    return pd.Series({n: find(n) for n in prots.protein_name})


def assign(clusters: pd.DataFrame, seed: int, f_test: float, f_val: float, holdout: str | None) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    out: dict[int, str] = {}
    for fam, g in clusters.groupby("family"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        if holdout and fam == holdout:
            for cid in g.cluster_id:
                out[cid] = "test"
            continue
        tot, acc_t, acc_v = g.n_rows.sum(), 0, 0
        for cid, n in zip(g.cluster_id, g.n_rows):
            if n > 0.5 * f_test * tot:  # giant clusters stay in train so test/val sizes remain controlled
                out[cid] = "train"
            elif acc_t < f_test * tot:
                out[cid], acc_t = "test", acc_t + n
            elif acc_v < f_val * tot:
                out[cid], acc_v = "val", acc_v + n
            else:
                out[cid] = "train"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["split", "build"])
    ap.add_argument("--min_id", type=float, default=0.4)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--hi_id", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f_test", type=float, default=0.12)
    ap.add_argument("--f_val", type=float, default=0.11)
    ap.add_argument("--holdout_family", default=None)
    args = ap.parse_args()

    tag = "v3b" if not args.holdout_family else f"v3b_lofo_{args.holdout_family}"
    res = ROOT / "results" / tag
    res.mkdir(parents=True, exist_ok=True)
    map_path = res / "protein_split_map.tsv"

    if args.stage == "split":
        prots = v3a_proteins()
        # protein names may appear in several v3a splits only if v3a leaked; keep the first sequence/rows sum
        prots = prots.groupby("protein_name").agg(protein_sequence=("protein_sequence", "first"), n_rows=("n_rows", "sum"),
                                                  v3a_split=("v3a_split", "first")).reset_index()
        rep = cluster(prots, args.min_id, args.cov, args.hi_id)
        prots["cluster_rep"] = prots.protein_name.map(rep)
        prots["cluster_id"] = prots.cluster_rep.astype("category").cat.codes
        fam = pd.read_csv(FAM, sep="\t").set_index("protein_name").pfam_family
        prots["family"] = prots.protein_name.map(fam).fillna("other")
        cl = prots.groupby("cluster_id").agg(n_rows=("n_rows", "sum"), n_prot=("protein_name", "size"),
                                             family=("family", lambda s: s.value_counts().index[0])).reset_index()
        cl["split"] = cl.cluster_id.map(assign(cl, args.seed, args.f_test, args.f_val, args.holdout_family))
        prots["split"] = prots.cluster_id.map(cl.set_index("cluster_id").split)
        prots.drop(columns="protein_sequence").to_csv(map_path, sep="\t", index=False)
        by = prots.groupby("split").agg(n_prot=("protein_name", "size"), n_rows=("n_rows", "sum"))
        fam_tab = pd.crosstab(prots.family, prots.split)
        summary = {"tag": tag, "min_id": args.min_id, "cov": args.cov, "n_proteins": len(prots), "n_clusters": int(cl.shape[0]),
                   "largest_cluster": int(cl.n_prot.max()), "singleton_clusters": int((cl.n_prot == 1).sum()),
                   "by_split": by.to_dict("index"), "family_by_split": fam_tab.to_dict("index"),
                   "v3a_vs_v3b_split_proteins": pd.crosstab(prots.v3a_split, prots.split).to_dict("index")}
        (res / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
        print(json.dumps(summary, indent=2, default=int))
        return

    mp = pd.read_csv(map_path, sep="\t").set_index("protein_name").split
    out = ROOT / "data" / ("generalized_" + tag)
    out.mkdir(parents=True, exist_ok=True)
    first = {s: True for s in ("train", "val", "test")}
    counts = {s: 0 for s in first}
    for sp in ("train", "val", "test"):
        for chunk in pd.read_csv(V3A / f"{sp}.tsv", sep="\t", chunksize=500_000):
            chunk["_s"] = chunk.protein_name.map(mp)
            for s, g in chunk.groupby("_s"):
                g.drop(columns="_s").to_csv(out / f"{s}.tsv", sep="\t", index=False, mode="w" if first[s] else "a", header=first[s])
                first[s] = False
                counts[s] += len(g)
    (out / "dataset_summary.json").write_text(json.dumps({"rows": counts, "source": "v3a re-split by cluster", "map": str(map_path.relative_to(ROOT))}, indent=2))
    print(counts)


if __name__ == "__main__":
    main()
