#!/usr/bin/env python3
"""
80_esm_layer_profile_similarity.py
----------------------------------
Which ESM-2 layer ranks RNAcompete proteins by binding profile similarity?

For every protein, the published RNAcompete 7-mer Z-scores (16,384 values; mean over the protein's experiments) give a binding
vector; Pearson correlation between proteins gives an n x n binding-similarity matrix. For every ESM-2 layer, mean-pooled
residue embeddings give an n x n cosine matrix. Per layer, the Spearman correlation between the upper triangles of the two
matrices is reported (cosine of raw embeddings and of embeddings centred over proteins), with a Mantel permutation p-value.

Proteins longer than 1,022 residues are embedded with a sliding window (window 1,022, overlap 511); per-residue
representations of overlapping windows are averaged before mean pooling.

Stages
------
  meta     experiment_id, protein_name, protein_sequence from a pairs table (column subset, read in chunks) -> out_dir/experiments.tsv
  embed    ESM-2 650M, all layers 0..33, mean pooled -> out_dir/esm2_layers.npz           (GPU)
  analyze  similarity matrices, per-layer Spearman, figure -> out_dir/layer_correlation.tsv, fig_layer_correlation.png

Usage
-----
  python scripts/80_esm_layer_profile_similarity.py meta --pairs data/benchmarks/rnacompete/rnacompete_human.tsv
  python scripts/80_esm_layer_profile_similarity.py embed
  python scripts/80_esm_layer_profile_similarity.py analyze --zscores rnacompete_analysis/data/raw/Zscores_RNAcompete2025.txt.gz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
WINDOW, STRIDE = 1022, 511
AA = set("ACDEFGHIKLMNPQRSTVWY")


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def stage_meta(args) -> None:
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen = []
    for ch in pd.read_csv(resolve(args.pairs), sep="\t", usecols=["experiment_id", "protein_name", "protein_sequence"], chunksize=500000):
        seen.append(ch.drop_duplicates())
    df = pd.concat(seen).drop_duplicates()
    df["protein_sequence"] = df.protein_sequence.map(lambda s: "".join(c for c in str(s).upper() if c in AA))
    df = df[df.protein_sequence.str.len() > 0].reset_index(drop=True)
    df.to_csv(out / "experiments.tsv", sep="\t", index=False)
    n_long = int((df.drop_duplicates("protein_sequence").protein_sequence.str.len() > WINDOW).sum())
    print(f"{len(df)} experiments, {df.protein_sequence.nunique()} distinct sequences, {n_long} longer than {WINDOW}")


def embed_one(model, converter, seq: str, device, layers: list[int]) -> np.ndarray:
    import torch
    n = len(seq)
    starts = list(range(0, max(1, n - WINDOW) + 1, STRIDE))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    acc = np.zeros((len(layers), n, 1280), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    for s in starts:
        w = seq[s:s + WINDOW]
        _, _, tok = converter([("p", w)])
        with torch.no_grad():
            rep = model(tok.to(device), repr_layers=layers)["representations"]
        for k, L in enumerate(layers):
            acc[k, s:s + len(w)] += rep[L][0, 1:len(w) + 1].float().cpu().numpy()
        cnt[s:s + len(w)] += 1
    return (acc / cnt[None, :, None]).mean(axis=1)


def stage_embed(args) -> None:
    import torch
    import esm
    out = resolve(args.out_dir)
    seqs = sorted(pd.read_csv(out / "experiments.tsv", sep="\t").protein_sequence.unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    conv = alphabet.get_batch_converter()
    layers = list(range(34))
    emb, t0 = [], time.time()
    for i, s in enumerate(seqs, 1):
        emb.append(embed_one(model, conv, s, device, layers))
        if i % 25 == 0:
            print(f"  {i}/{len(seqs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "esm2_layers.npz", emb=np.stack(emb).astype(np.float32), seqs=np.array(seqs, dtype=object))
    print(f"saved {len(seqs)} proteins x {len(layers)} layers")


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices_from(m, k=1)]


def cosine(e: np.ndarray, centre: bool) -> np.ndarray:
    if centre:
        e = e - e.mean(0, keepdims=True)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    return e @ e.T


def spearman_mantel(a: np.ndarray, b: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    ra = rankdata(upper(a))
    rho = float(np.corrcoef(ra, rankdata(upper(b)))[0, 1])
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(len(b))
        r = float(np.corrcoef(ra, rankdata(upper(b[np.ix_(p, p)])))[0, 1])
        ge += abs(r) >= abs(rho)
    return rho, (ge + 1) / (n_perm + 1)


def stage_analyze(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = resolve(args.out_dir)
    exp = pd.read_csv(out / "experiments.tsv", sep="\t")
    z = np.load(out / "esm2_layers.npz", allow_pickle=True)
    seqs = list(z["seqs"])
    zs = pd.read_csv(resolve(args.zscores), sep="\t", index_col=0)
    zs = zs.loc[:, ~zs.columns.duplicated()]
    cols = {c: c for c in zs.columns}
    vec, keep = [], []
    for i, s in enumerate(seqs):
        ids = [e for e in exp[exp.protein_sequence == s].experiment_id.unique() if e in cols]
        if ids:
            vec.append(zs[ids].mean(axis=1).to_numpy(dtype=np.float64))
            keep.append(i)
    print(f"{len(keep)} of {len(seqs)} proteins with Z-scores; 7-mers {zs.shape[0]}", flush=True)
    V = np.vstack(vec)
    bind = np.corrcoef(V - V.mean(axis=1, keepdims=True))
    emb = z["emb"][keep]
    rng = np.random.default_rng(args.seed)
    rows = []
    for L in range(emb.shape[1]):
        r = {"layer": L}
        for tag, centre in (("raw", False), ("centred", True)):
            rho, p = spearman_mantel(cosine(emb[:, L], centre), bind, args.n_perm, rng)
            r[f"spearman_{tag}"], r[f"mantel_p_{tag}"] = rho, p
        rows.append(r)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.insert(1, "n_proteins", len(keep))
    df.to_csv(out / "layer_correlation.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(df.layer, df.spearman_raw, marker="o", ms=3, label="cosine, raw")
    ax.plot(df.layer, df.spearman_centred, marker="s", ms=3, label="cosine, centred over proteins")
    ax.set_xlabel("ESM-2 650M layer")
    ax.set_ylabel(f"Spearman with 7-mer Z-score Pearson (n={len(keep)})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_layer_correlation.png", dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["meta", "embed", "analyze"])
    ap.add_argument("--pairs", default="data/benchmarks/rnacompete/rnacompete_human.tsv")
    ap.add_argument("--zscores", default="rnacompete_analysis/data/raw/Zscores_RNAcompete2025.txt.gz")
    ap.add_argument("--out_dir", default="results/esm_layer_profile")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"meta": stage_meta, "embed": stage_embed, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
