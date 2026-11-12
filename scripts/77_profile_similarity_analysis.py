#!/usr/bin/env python3
"""
77_profile_similarity_analysis.py
---------------------------------
How well does protein similarity predict similarity of RNA-binding preferences?  Analysis on measured data, no model training.

Per protein: the 5-mer log2-enrichment profile (script 73, results/v3b_profile/profiles.npz, 1024 values).
Pairwise protein similarities: (a) sequence identity (local alignment, matches / length of the shorter protein),
(b) cosine of the standardized ESM-2 embeddings (script 66).  Pairwise profile similarity: Pearson correlation of the profiles.

Outputs in --out_dir
  pairs.tsv.gz              one row per protein pair: identity, cosine, pearson, same_family
  identity_curve.tsv        median Pearson per identity bin (all pairs / same family / different family) with 95% bootstrap CI
  nearest_neighbour.tsv     Pearson between a protein and its nearest neighbour under each similarity (identity, embedding, random, best possible)
  summary.json              Spearman correlations (identity vs Pearson, cosine vs Pearson) with permutation p-values
  fig_identity_curve.png, fig_similarity_scatter.png, fig_profile_clustermap.png

Usage
-----
  python scripts/77_profile_similarity_analysis.py --out_dir results/profile_similarity
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
BINS = [0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]
BIN_LABELS = ["<20%", "20-30%", "30-40%", "40-50%", "50-70%", ">=70%"]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


_ALIGNER = None


def _aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from Bio import Align
        from Bio.Align import substitution_matrices
        a = Align.PairwiseAligner()
        a.mode = "local"
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score, a.extend_gap_score = -10.0, -0.5
        _ALIGNER = a
    return _ALIGNER


def pair_identity(args) -> float:
    a, b = args
    aln = _aligner().align(a, b)[0]
    m = 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        m += sum(x == y for x, y in zip(a[a0:a1], b[b0:b1]))
    return m / max(1, min(len(a), len(b)))


def identity_matrix(seqs: list[str], cache: Path, workers: int) -> np.ndarray:
    n = len(seqs)
    if cache.exists():
        m = np.load(cache)
        if m.shape == (n, n):
            return m
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    with Pool(workers) as pool:
        vals = pool.map(pair_identity, [(seqs[i], seqs[j]) for i, j in pairs], chunksize=200)
    m = np.eye(n)
    for (i, j), v in zip(pairs, vals):
        m[i, j] = m[j, i] = v
    np.save(cache, m)
    return m


def corr_matrix(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(1, keepdims=True)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x @ x.T


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a) - rankdata(a).mean(), rankdata(b) - rankdata(b).mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum() + 1e-12))


def boot_median(v: np.ndarray, rng, n=500) -> tuple[float, float, float]:
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    b = [np.median(rng.choice(v, len(v))) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", default="results/v3b_profile/profiles.npz")
    ap.add_argument("--emb", default="data/embeddings_v3b/prot_esm2.npz")
    ap.add_argument("--seq_tsvs", nargs="+", default=["data/generalized_v3b/train.tsv", "data/generalized_v3b/val.tsv",
                                                       "data/generalized_v3b/test.tsv"])
    ap.add_argument("--split_map", default="results/v3b/protein_split_map.tsv")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/profile_similarity")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(resolve(args.profiles), allow_pickle=False)
    names = [str(n) for n in z["names"]]
    prof = z["profile"].astype(np.float64)
    e = np.load(resolve(args.emb), allow_pickle=False)
    emb = dict(zip([str(n) for n in e["names"]], e["emb"].astype(np.float32)))
    seqs = {}
    for f in args.seq_tsvs:
        d = pd.read_csv(resolve(f), sep="\t", usecols=["protein_name", "protein_sequence"], low_memory=False).drop_duplicates("protein_name")
        seqs.update(dict(zip(d.protein_name, d.protein_sequence)))
    fam = pd.read_csv(resolve(args.split_map), sep="\t").set_index("protein_name").family.to_dict()
    keep = [i for i, n in enumerate(names) if n in emb and n in seqs]
    names, prof = [names[i] for i in keep], prof[keep]
    n = len(names)
    print(f"{n} proteins with profile, embedding and sequence", flush=True)

    P = corr_matrix(prof)
    E = np.stack([emb[x] for x in names])
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    U = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    C = U @ U.T
    I = identity_matrix([seqs[x] for x in names], out / "identity_matrix.npy", args.workers)
    f = np.array([fam.get(x, "NA") for x in names])
    same = (f[:, None] == f[None, :]) & (f[:, None] != "NA")

    iu = np.triu_indices(n, 1)
    pairs = pd.DataFrame({"a": np.array(names)[iu[0]], "b": np.array(names)[iu[1]], "identity": I[iu], "cosine": C[iu],
                          "pearson": P[iu], "same_family": same[iu]})
    pairs.to_csv(out / "pairs.tsv.gz", sep="\t", index=False, float_format="%.4f")

    # identity curve
    pairs["bin"] = pd.cut(pairs.identity, BINS, labels=BIN_LABELS, right=False)
    rows = []
    for label, sub in [("all pairs", pairs), ("same family", pairs[pairs.same_family]), ("different family", pairs[~pairs.same_family])]:
        for b in BIN_LABELS:
            v = sub.pearson[sub.bin == b].to_numpy()
            med, lo, hi = boot_median(v, rng)
            rows.append({"subset": label, "identity_bin": b, "n_pairs": len(v), "median_pearson": med, "ci_low": lo, "ci_high": hi})
    curve = pd.DataFrame(rows)
    curve.to_csv(out / "identity_curve.tsv", sep="\t", index=False)
    print(curve.to_string(index=False))

    # correlations with permutation test (protein labels of the profile matrix shuffled)
    def perm_p(sim: np.ndarray) -> tuple[float, float]:
        obs = spearman(sim[iu], P[iu])
        null = []
        for _ in range(args.n_perm):
            q = rng.permutation(n)
            null.append(spearman(sim[iu], P[q][:, q][iu]))
        return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))

    summary = {"n_proteins": n, "n_pairs": int(len(iu[0]))}
    for name, sim in (("identity", I), ("embedding_cosine", C)):
        obs, p = perm_p(sim)
        summary[f"spearman_{name}_vs_pearson"] = {"rho": obs, "perm_p": p}
        print(f"Spearman({name}, profile Pearson) = {obs:.3f}  permutation p = {p:.3g}")

    # nearest neighbour: how similar is the profile of the nearest protein under each similarity
    off = ~np.eye(n, dtype=bool)
    best = lambda s: np.where(off, s, -np.inf).argmax(1)  # noqa: E731
    nn = {"identity": best(I), "embedding_cosine": best(C), "best_possible (oracle)": best(P),
          "random": np.array([rng.choice(np.delete(np.arange(n), i)) for i in range(n)])}
    nn_rows = [{"neighbour_by": k, "mean_pearson": float(P[np.arange(n), v].mean()), "median_pearson": float(np.median(P[np.arange(n), v]))}
               for k, v in nn.items()]
    pd.DataFrame(nn_rows).to_csv(out / "nearest_neighbour.tsv", sep="\t", index=False)
    summary["nearest_neighbour"] = nn_rows
    print(pd.DataFrame(nn_rows).to_string(index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, col in (("all pairs", "#555555"), ("same family", "#1b7837"), ("different family", "#b2182b")):
        c = curve[curve.subset == label]
        x = np.arange(len(BIN_LABELS))
        ax.errorbar(x, c.median_pearson, yerr=[c.median_pearson - c.ci_low, c.ci_high - c.median_pearson], marker="o", capsize=3,
                    label=label, color=col)
    ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
    ax.set_xlabel("sequence identity of the protein pair")
    ax.set_ylabel("median Pearson of the 5-mer profiles")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_identity_curve.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (x, lab) in zip(axs, ((pairs.identity, "sequence identity"), (pairs.cosine, "ESM-2 cosine"))):
        ax.hexbin(x, pairs.pearson, gridsize=50, bins="log", cmap="YlGnBu", mincnt=1)
        ax.set_xlabel(lab)
        ax.set_ylabel("Pearson of the 5-mer profiles")
    fig.tight_layout()
    fig.savefig(out / "fig_similarity_scatter.png", dpi=200)
    plt.close(fig)

    order = dendrogram(linkage(prof, "average", metric="correlation"), no_plot=True)["leaves"]
    fam_order = sorted(set(f))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(fam_order))))
    cm = {k: colors[i] for i, k in enumerate(fam_order)}
    fig, axs = plt.subplots(1, 2, figsize=(10, 8.5), gridspec_kw={"width_ratios": [1, 0.04], "wspace": 0.02})
    axs[0].imshow(P[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axs[0].set_title("Pearson correlation of 5-mer profiles, proteins ordered by profile clustering")
    axs[0].set_xticks([]), axs[0].set_yticks([])
    axs[1].imshow(np.array([cm[f[i]] for i in order])[:, None, :], aspect="auto")
    axs[1].set_xticks([]), axs[1].set_yticks([])
    axs[1].set_title("family", fontsize=8)
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=cm[k], label=k) for k in fam_order[:20]]
    axs[0].legend(handles=handles, fontsize=6, loc="upper left", bbox_to_anchor=(1.08, 1), frameon=False)
    fig.savefig(out / "fig_profile_clustermap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
