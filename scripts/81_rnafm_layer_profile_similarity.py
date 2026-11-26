#!/usr/bin/env python3
"""
81_rnafm_layer_profile_similarity.py
------------------------------------
RNA-side counterpart of script 80: which RNA-FM layer ranks RNAcompete probes by their binding pattern across proteins?

For a random subset of probes (30-41 nt as present in the score file, no length filter is applied, SetA and SetB), the binding vector is the normalised probe intensity in each of the 175
proteins of script 80 (mean over the experiments of the same protein sequence, every protein column standardised over all probes).
The Pearson correlation between the vectors of two probes gives an n x n binding-similarity matrix; for every RNA-FM layer the cosine
similarity of mean-pooled nucleotide embeddings (raw and centred over probes) gives a second matrix. Per layer, the Spearman correlation
of the two upper triangles is reported with a bootstrap interval over probes (resampling with replacement, identical pairs dropped).
Control: cosine similarity of 3-mer frequency vectors, and the difference of each layer to it.

Stages
------
  subset   probe subset + binding vectors from the published per-probe score matrix -> out_dir/probes.tsv, out_dir/profiles.npz  (local)
  embed    RNA-FM (multimolecule/rnafm), hidden states of all layers, mean over nucleotides -> out_dir/rnafm_layers.npz           (GPU)
  analyze  per-layer Spearman, intervals, figure -> out_dir/layer_correlation.tsv, fig_layer_correlation.png

Usage
-----
  python scripts/81_rnafm_layer_profile_similarity.py subset --scores rnacompete_analysis/data/raw/norm_probe_scores.txt.gz
  python scripts/81_rnafm_layer_profile_similarity.py embed
  python scripts/81_rnafm_layer_profile_similarity.py analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def stage_subset(args) -> None:
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exp = pd.read_csv(resolve(args.experiments), sep="\t")
    d = pd.read_csv(resolve(args.scores), sep="\t", low_memory=False)
    ids = [c for c in d.columns[3:]]
    X = d[ids].apply(pd.to_numeric, errors="coerce")
    X = (X - X.mean()) / X.std()
    cols, keep = [], []
    for s, g in exp.groupby("protein_sequence"):
        e = [x for x in g.experiment_id.unique() if x in X.columns]
        if e:
            cols.append(X[e].mean(axis=1).to_numpy())
            keep.append(s)
    M = np.stack(cols, axis=1)
    ok = ~np.isnan(M).any(axis=1)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(np.where(ok)[0], args.n_probes, replace=False))
    sub = d.iloc[idx][["Probe_Set", "Probe_ID", "RNA_Seq"]].rename(columns={"Probe_Set": "probe_set", "Probe_ID": "probe_id", "RNA_Seq": "rna_sequence"})
    sub.to_csv(out / "probes.tsv", sep="\t", index=False)
    np.savez(out / "profiles.npz", profiles=M[idx].astype(np.float32), proteins=np.array(keep, dtype=object))
    print(f"{len(idx)} probes x {len(keep)} proteins; probes with complete data {int(ok.sum())} of {len(d)}")


def stage_embed(args) -> None:
    import torch
    out = resolve(args.out_dir)
    seqs = pd.read_csv(out / "probes.tsv", sep="\t").rna_sequence.tolist()
    spec = importlib.util.spec_from_file_location("embed66", ROOT / "scripts" / "66_embed_v3b_frozen.py")
    m66 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m66)
    RnaFmModel, RnaTokenizer = m66.import_multimolecule()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel.from_pretrained("multimolecule/rnafm").to(device).eval()
    embs, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch):
            chunk = [m66.clean(s, "ACGU") for s in seqs[i:i + args.batch]]
            x = tok(chunk, return_tensors="pt", padding=True).to(device)
            hs = torch.stack(model(**x, output_hidden_states=True).hidden_states, dim=1).float()
            for j, s in enumerate(chunk):
                embs.append(hs[j, :, 1:len(s) + 1].mean(dim=1).cpu().numpy())
            print(f"  {i + len(chunk)}/{len(seqs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "rnafm_layers.npz", emb=np.stack(embs).astype(np.float32))
    print(f"saved {len(embs)} probes x {embs[0].shape[0]} hidden states")


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices_from(m, k=1)]


def cosine(e: np.ndarray, centre: bool) -> np.ndarray:
    if centre:
        e = e - e.mean(0, keepdims=True)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    return e @ e.T


def kmer_freq(seqs: list[str], k: int = 3) -> np.ndarray:
    idx = {"".join(p): i for i, p in enumerate(itertools.product("ACGU", repeat=k))}
    f = np.zeros((len(seqs), len(idx)))
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None:
                f[r, j] += 1
        f[r] /= max(1, f[r].sum())
    return f


def stage_analyze(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = resolve(args.out_dir)
    P = np.load(out / "profiles.npz", allow_pickle=True)["profiles"].astype(np.float64)
    emb = np.load(out / "rnafm_layers.npz")["emb"]
    seqs = pd.read_csv(out / "probes.tsv", sep="\t").rna_sequence.tolist()
    n, n_layers = len(P), emb.shape[1]
    bind = np.corrcoef(P - P.mean(axis=1, keepdims=True))
    mats = {"raw": [cosine(emb[:, L], False) for L in range(n_layers)], "centred": [cosine(emb[:, L], True) for L in range(n_layers)]}
    kf = kmer_freq(seqs)
    ctrl = {"raw": cosine(kf, False), "centred": cosine(kf, True)}
    rng = np.random.default_rng(args.seed)
    iu = np.triu_indices(n, k=1)
    point = {t: [] for t in mats}
    ra = rankdata(bind[iu])
    for t in mats:
        for L in range(n_layers):
            point[t].append(float(np.corrcoef(ra, rankdata(mats[t][L][iu]))[0, 1]))
        point[t + "_kmer3"] = float(np.corrcoef(ra, rankdata(ctrl[t][iu]))[0, 1])
    boot = {t: np.zeros((args.n_boot, n_layers)) for t in mats}
    bctrl = {t: np.zeros(args.n_boot) for t in mats}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        a, c = idx[iu[0]], idx[iu[1]]
        ok = a != c
        rb = rankdata(bind[a[ok], c[ok]])
        for t in mats:
            for L in range(n_layers):
                boot[t][b, L] = np.corrcoef(rb, rankdata(mats[t][L][a[ok], c[ok]]))[0, 1]
            bctrl[t][b] = np.corrcoef(rb, rankdata(ctrl[t][a[ok], c[ok]]))[0, 1]
        if (b + 1) % 20 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}", flush=True)
    q = lambda x: np.percentile(x, [2.5, 97.5], axis=0)  # noqa: E731
    df = pd.DataFrame({"layer": range(n_layers), "n_probes": n})
    for t in mats:
        lo, hi = q(boot[t])
        dl, dh = q(boot[t] - bctrl[t][:, None])
        df[f"spearman_{t}"], df[f"spearman_{t}_lo"], df[f"spearman_{t}_hi"] = point[t], lo, hi
        df[f"diff_vs_kmer3_{t}_lo"], df[f"diff_vs_kmer3_{t}_hi"] = dl, dh
        df[f"kmer3_{t}"] = point[t + "_kmer3"]
    df.to_csv(out / "layer_correlation.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for t, c in (("raw", "C0"), ("centred", "C1")):
        ax.plot(df.layer, df[f"spearman_{t}"], marker="o", ms=3, color=c, label=f"cosine, {t}")
        ax.fill_between(df.layer, df[f"spearman_{t}_lo"], df[f"spearman_{t}_hi"], color=c, alpha=0.15)
        ax.axhline(point[t + "_kmer3"], color=c, ls="--", lw=1)
    ax.set_xlabel("RNA-FM hidden state (0 = token embedding, 12 = last)")
    ax.set_ylabel(f"Spearman with probe binding similarity (n={n})")
    ax.legend(frameon=False, title="dashed: 3-mer frequencies", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_layer_correlation.png", dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["subset", "embed", "analyze"])
    ap.add_argument("--scores", default="rnacompete_analysis/data/raw/norm_probe_scores.txt.gz")
    ap.add_argument("--experiments", default="results/esm_layer_profile/experiments.tsv")
    ap.add_argument("--out_dir", default="results/rna_layer_profile")
    ap.add_argument("--n_probes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"subset": stage_subset, "embed": stage_embed, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
