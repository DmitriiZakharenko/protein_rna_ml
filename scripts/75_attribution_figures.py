#!/usr/bin/env python3
"""
75_attribution_figures.py
-------------------------
Figures for one-hot protein-RNA models (V2 / V4 checkpoints), in the style of sequence-logo saliency maps:

  motif_<protein>.png    5-mer enrichment measured in the data (scatter + logo of the top k-mers) next to the motif the model predicts
                         (mean logit of every 5-mer inserted into real RNA contexts) - the model's recovery of the known motif
  saliency_<protein>.png RNA sequence; in-silico single-nucleotide mutations (signed logo); sliding-window deletion (signed letters);
                         integrated gradients (RNA); integrated gradients over the protein with the Pfam RNA-binding domains shaded

Positive letters point up, negative letters are flipped below the axis. Statistics (Spearman of predicted vs measured 5-mer profile) go to stats.json.
These are illustrations of the method on the chosen checkpoint; whether a model transfers is decided by the probe, not by these figures.

Usage
-----
  python scripts/75_attribution_figures.py --model "v2,type=v2,path=models/saved/generalized_v2/best_model.pt,prot_max=300" \\
      --data_tsv data/generalized_v2/train.tsv --proteins RBFOX2,HNRNPC,PCBP2,TIA1 --out_dir results/mechanistic/figures
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NT = "ACGU"
COL = {"A": "#009E73", "C": "#0072B2", "G": "#E69F00", "U": "#D55E00"}
K = 5
KMERS = ["".join(p) for p in product(NT, repeat=K)]
FONT = FontProperties(family="DejaVu Sans", weight="bold")


def load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def letter(ax, ch, x, y0, h, w=0.86):
    """Draw letter ch centred at x with signed height h starting at y0 (negative h flips the letter)."""
    if abs(h) < 1e-9:
        return
    tp = TextPath((0, 0), ch, size=1, prop=FONT)
    bb = tp.get_extents()
    tr = Affine2D().translate(-bb.x0, -bb.y0).scale(w / bb.width, h / bb.height).translate(x - w / 2, y0)
    ax.add_patch(PathPatch(tr.transform_path(tp), facecolor=COL.get(ch, "grey"), edgecolor="none"))


def logo_from_kmers(kmers: list[str], weights=None) -> np.ndarray:
    """Information-content heights (bits) per position and nucleotide from a set of k-mers."""
    w = np.ones(len(kmers)) if weights is None else np.asarray(weights, float)
    f = np.zeros((K, 4))
    for km, wi in zip(kmers, w):
        for i, c in enumerate(km):
            f[i, NT.index(c)] += wi
    f /= f.sum(1, keepdims=True)
    h = -(f * np.log2(np.clip(f, 1e-9, 1))).sum(1)
    return f * (2 - h)[:, None]


def draw_logo(ax, heights: np.ndarray, ymax=None):
    for i in range(heights.shape[0]):
        y = 0.0
        for j in np.argsort(heights[i]):
            letter(ax, NT[j], i, y, heights[i, j])
            y += heights[i, j]
    ax.set_xlim(-0.6, heights.shape[0] - 0.4)
    ax.set_ylim(0, ymax or float(heights.sum(1).max()) * 1.2)
    ax.set_xticks(range(heights.shape[0]))
    ax.set_ylabel("bits")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def signed_logo(ax, mat: np.ndarray, lim=None):
    """mat: (4, L) signed effects; letters stacked above (positive) and below (negative, flipped) the axis."""
    L = mat.shape[1]
    top = bot = 1e-9
    for i in range(L):
        pos, neg = 0.0, 0.0
        for j in np.argsort(np.abs(mat[:, i])):
            v = mat[j, i]
            if v > 0:
                letter(ax, NT[j], i, pos, v)
                pos += v
            elif v < 0:
                letter(ax, NT[j], i, neg, v)
                neg += v
        top, bot = max(top, pos), min(bot, neg)
    lim = lim or max(top, -bot) * 1.1
    ax.set_xlim(-0.6, L - 0.4)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def enrichment(k59, rnas, y):
    c = k59.kmer_counts(k59.encode(list(rnas)), K)
    pos, neg = c[y == 1].sum(0) + 1.0, c[y == 0].sum(0) + 1.0
    return np.log2(pos / pos.sum()) - np.log2(neg / neg.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_tsv", required=True)
    ap.add_argument("--proteins", required=True, help="comma-separated protein names present in the data table")
    ap.add_argument("--top_kmers", type=int, default=12)
    ap.add_argument("--n_ctx", type=int, default=40)
    ap.add_argument("--max_rows", type=int, default=60000)
    ap.add_argument("--out_dir", default="results/mechanistic/figures")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "mps" if torch.backends.mps.is_available() and not args.no_cuda else "cpu")
    sf, m61, m68, k59 = (load_mod("suite51", "51_eclip_eval_suite.py"), load_mod("mech61", "61_motif_recovery_and_mutagenesis.py"),
                         load_mod("attr68", "68_attribution_maps.py"), load_mod("knn59", "59_knn_protein_baseline.py"))
    cfg = m68.parse_model(args.model)
    model, fwd = sf.load_model(cfg["type"], resolve(cfg["path"]), device)
    model.eval()
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolve(args.data_tsv), sep="\t", usecols=["protein_name", "protein_sequence", "rna_sequence", "binding_label"], low_memory=False)
    pool = df.rna_sequence.drop_duplicates()
    pool = pool[(pool.str.len() >= 20) & (pool.str.len() <= cfg["rna_max"])].str.upper().str.replace("T", "U")
    bg = pool.sample(args.n_ctx, random_state=args.seed).tolist()
    probes, owners = [], []
    for i, km in enumerate(KMERS):
        for b in bg:
            pos = int(rng.integers(0, len(b) - K + 1))
            probes.append(b[:pos] + km + b[pos + K:])
            owners.append(i)
    owners = np.array(owners)

    names = [p.strip() for p in args.proteins.split(",")]
    seqs = {n: df[df.protein_name == n].protein_sequence.iloc[0] for n in names if (df.protein_name == n).any()}
    dom = m61.domain_positions_from_hmmscan(seqs)
    stats = {}
    for n in names:
        if n not in seqs:
            print(f"{n}: not in the data table, skipped")
            continue
        g = df[df.protein_name == n]
        if len(g) > args.max_rows:
            g = g.sample(args.max_rows, random_state=args.seed)
        prot = seqs[n]
        true = enrichment(k59, g.rna_sequence, g.binding_label.to_numpy())
        lg = m61.logits(model, fwd, sf, probes, prot, cfg["rna_max"], cfg["prot_max"], device)
        pred = np.bincount(owners, weights=lg, minlength=len(KMERS)) / args.n_ctx
        rho = float(spearmanr(pred, true).statistic)
        stats[n] = {"spearman_pred_vs_true": rho, "n_rows": int(len(g))}
        top_t, top_p = np.argsort(-true)[:args.top_kmers], np.argsort(-pred)[:args.top_kmers]

        f, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
        ax = axs[0]
        ax.scatter(true, pred, s=5, alpha=0.35, color="#555555")
        ax.scatter(true[top_t], pred[top_t], s=14, color="#D55E00", label="top measured 5-mers")
        ax.set_xlabel("measured log2 enrichment (data)")
        ax.set_ylabel("model: mean logit")
        ax.set_title(f"{n}: Spearman {rho:.2f}", loc="left", fontsize=12)
        for i in top_t[:4]:
            ax.annotate(KMERS[i], (true[i], pred[i]), xytext=(4, 4), textcoords="offset points", fontsize=9, color="#D55E00")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        draw_logo(axs[1], logo_from_kmers([KMERS[i] for i in top_t]))
        axs[1].set_title(f"measured motif (top {args.top_kmers} 5-mers)", loc="left", fontsize=12)
        draw_logo(axs[2], logo_from_kmers([KMERS[i] for i in top_p]))
        axs[2].set_title(f"model-predicted motif (top {args.top_kmers})", loc="left", fontsize=12)
        f.tight_layout()
        f.savefig(out / f"motif_{n}.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(f)

        # ---- saliency map for the best-scoring positive RNA of moderate length
        pos = g[(g.binding_label == 1) & (g.rna_sequence.str.len().between(20, 60))].rna_sequence.drop_duplicates().head(400).tolist()
        if not pos:
            continue
        sc = m61.logits(model, fwd, sf, pos, prot, cfg["rna_max"], cfg["prot_max"], device)
        rna = pos[int(np.argmax(sc))].upper().replace("T", "U")
        r, p = m68.encode(sf, rna, prot, cfg, device)
        ig_r, ig_p = m68.integrated_gradients(model, fwd, r, p)
        ism, win = m68.mutagenesis(model, fwd, r, p, len(rna)), m68.window_deletion(model, fwd, r, p, len(rna))
        plen = min(len(prot), cfg["prot_max"])
        ds = [(int(a), int(b)) for a, b in zip(dom[dom.protein_name == n].start, dom[dom.protein_name == n].end) if a <= plen]
        L = len(rna)
        f = plt.figure(figsize=(13, 10))
        gs = f.add_gridspec(5, 1, height_ratios=[0.7, 1.7, 1.4, 1.2, 1.5], hspace=0.5)
        ax = f.add_subplot(gs[0])
        for i, ch in enumerate(rna):
            letter(ax, ch, i, 0, 1.0)
        ax.set_xlim(-0.6, L - 0.4)
        ax.set_ylim(0, 1.05)
        ax.axis("off")
        ax.set_title(f"{n} | positive RNA | model logit {sc.max():.2f}", loc="left", fontsize=13)
        ax = f.add_subplot(gs[1])
        signed_logo(ax, ism[:, :L])
        ax.set_ylabel("in-silico mutation\n(delta logit)")
        ax = f.add_subplot(gs[2])
        signed_logo(ax, np.stack([np.where(np.array(list(rna)) == c, win[:L], 0.0) for c in NT]))
        ax.set_ylabel("sliding-window\ndeletion (delta logit)")
        ax = f.add_subplot(gs[3])
        ax.bar(range(L), ig_r[:L], color=["#0072B2" if v >= 0 else "#D55E00" for v in ig_r[:L]])
        ax.set_xlim(-0.6, L - 0.4)
        ax.set_ylabel("integrated\ngradients (RNA)")
        ax.set_xlabel("RNA position")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax = f.add_subplot(gs[4])
        for a, b in ds:
            ax.axvspan(a - 1, b, color="#999999", alpha=0.25, lw=0)
        ax.plot(np.arange(plen), ig_p[:plen], color="#333333", lw=0.9)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlim(0, plen)
        ax.set_ylabel("integrated\ngradients (protein)")
        ax.set_xlabel("protein residue (grey = Pfam RNA-binding domain)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        f.savefig(out / f"saliency_{n}.png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(f)
        in_dom = float(np.abs(ig_p[:plen])[np.isin(np.arange(plen), [i for a, b in ds for i in range(a - 1, b)])].sum() / max(np.abs(ig_p[:plen]).sum(), 1e-9)) if ds else float("nan")
        stats[n].update({"rna": rna, "logit": float(sc.max()), "domains": ds, "ig_share_in_domain": in_dom,
                         "domain_fraction": float(np.isin(np.arange(plen), [i for a, b in ds for i in range(a - 1, b)]).mean()) if ds else 0.0})
        print(f"{n}: Spearman {rho:.2f}; domains {ds}; share of |IG| in domain {in_dom:.2f} (domain covers {stats[n]['domain_fraction']:.2f})", flush=True)
    (out / "stats.json").write_text(json.dumps({"model": cfg, "proteins": stats}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
