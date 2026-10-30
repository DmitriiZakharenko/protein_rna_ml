#!/usr/bin/env python3
"""
68_attribution_maps.py
----------------------
Saliency maps for one-hot protein-RNA models (V2 / V4 checkpoints, scripts/06, 21, 65) with quantitative checks.

Per (protein, RNA) pair, four RNA views and one protein view:
  * in-silico single-nucleotide mutagenesis (delta logit for every substitution),
  * sliding-window deletion (delta logit when a window is zeroed),
  * integrated gradients on the RNA,
  * integrated gradients on the protein, drawn over the Pfam domain track.

Quantitative summaries over many proteins (the figures alone are anecdotes):
  * domain enrichment: share of |IG| inside annotated RNA-binding domains divided by the share of residues inside them,
    with a permutation p-value from length-matched random windows; the same on residue-shuffled proteins (position-bias control);
  * motif check (--motifs NAME=KMER,...): mean RNA IG on motif occurrences vs elsewhere in positive RNAs.

Usage
-----
  python scripts/68_attribution_maps.py --name v4x_v3b --model "v4x,type=v4,path=models/saved/v4x_v3b_seed42/best_model.pt" \\
      --test_tsv data/generalized_v3b/test.tsv --domain_positions results/mechanistic/v3b_domain_positions.tsv \\
      --max_proteins 20 --n_rna 30 --plot_top 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NT = "ACGU"
NT_COLOR = {"A": "#009E73", "C": "#0072B2", "G": "#E69F00", "U": "#D55E00"}


def load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def parse_model(spec: str) -> dict:
    parts = spec.split(",")
    cfg = {"name": parts[0], "rna_max": 60, "prot_max": 700}
    for kv in parts[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = int(v) if k in ("rna_max", "prot_max") else v
    return cfg


def encode(sf, rna: str, prot: str, cfg: dict, device):
    r = torch.from_numpy(sf.one_hot([rna], cfg["rna_max"], sf.RNA_LUT, 4)).to(device)
    p = torch.from_numpy(sf.one_hot([prot], cfg["prot_max"], sf.AA_LUT, 20)).to(device)
    return r, p


@torch.no_grad()
def logit(model, fwd, r, p) -> np.ndarray:
    return fwd(model, r, p.expand(len(r), -1, -1) if p.shape[0] == 1 else p).float().cpu().numpy().reshape(-1)


def integrated_gradients(model, fwd, r, p, steps: int = 32):
    """IG with an all-zero baseline for both inputs; returns attribution per position (summed over channels)."""
    a = torch.linspace(1.0 / steps, 1.0, steps, device=r.device).view(-1, 1, 1)
    ri = (a * r).requires_grad_(True)
    pi = (a * p).requires_grad_(True)
    out = fwd(model, ri, pi).sum()
    gr, gp = torch.autograd.grad(out, [ri, pi])
    ig_r = (gr.mean(0) * r[0]).sum(-1).detach().cpu().numpy()
    ig_p = (gp.mean(0) * p[0]).sum(-1).detach().cpu().numpy()
    return ig_r, ig_p


def mutagenesis(model, fwd, r, p, n: int) -> np.ndarray:
    base = logit(model, fwd, r, p)[0]
    muts = []
    for i in range(n):
        for c in range(4):
            x = r.clone()
            x[0, i, :] = 0
            x[0, i, c] = 1
            muts.append(x)
    lg = logit(model, fwd, torch.cat(muts), p).reshape(n, 4)
    return (lg - base).T  # (4, n)


def window_deletion(model, fwd, r, p, n: int, w: int = 5) -> np.ndarray:
    base = logit(model, fwd, r, p)[0]
    xs = []
    for i in range(n):
        x = r.clone()
        x[0, max(0, i - w // 2): i + w // 2 + 1, :] = 0
        xs.append(x)
    d = base - logit(model, fwd, torch.cat(xs), p)
    return d


def domain_stats(ig_p: np.ndarray, plen: int, doms: list[tuple[int, int]], rng, n_perm: int = 500):
    """share of |IG| in domains / share of residues in domains, permutation p from random same-length windows."""
    a = np.abs(ig_p[:plen])
    tot = a.sum()
    if tot <= 0 or not doms:
        return None
    mask = np.zeros(plen, bool)
    for s, e in doms:
        mask[max(0, s - 1):min(plen, e)] = True
    frac = mask.mean()
    if frac <= 0 or frac >= 1:
        return None
    obs = a[mask].sum() / tot
    ratio = obs / frac
    k = int(mask.sum())
    null = []
    for _ in range(n_perm):
        st = int(rng.integers(0, plen - k + 1))
        null.append(a[st:st + k].sum() / tot / (k / plen))
    return {"enrichment": float(ratio), "p_perm": float((np.sum(np.array(null) >= ratio) + 1) / (n_perm + 1)),
            "domain_frac": float(frac)}


def plot_pair(path: Path, title: str, rna: str, prot: str, ism, win, ig_r, ig_p, doms, plen: int):
    n = len(rna)
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(5, 1, height_ratios=[0.6, 1.6, 1.0, 1.0, 1.6], hspace=0.55)
    ax = fig.add_subplot(gs[0])
    for i, ch in enumerate(rna):
        ax.text(i, 0.5, ch, ha="center", va="center", color=NT_COLOR.get(ch, "grey"), fontsize=9, family="monospace")
    ax.set_xlim(-0.5, n - 0.5), ax.set_ylim(0, 1), ax.axis("off"), ax.set_title(title, loc="left", fontsize=11)
    ax = fig.add_subplot(gs[1])
    v = float(np.abs(ism[:, :n]).max()) or 1.0
    im = ax.imshow(ism[:, :n], aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_yticks(range(4)), ax.set_yticklabels(list(NT)), ax.set_ylabel("in-silico mutation\n(delta logit)")
    fig.colorbar(im, cax=ax.inset_axes([1.01, 0.0, 0.012, 1.0]))
    ax = fig.add_subplot(gs[2])
    ax.bar(range(n), win[:n], color=["#0072B2" if x >= 0 else "#D55E00" for x in win[:n]])
    ax.set_xlim(-0.5, n - 0.5), ax.set_ylabel("window deletion\n(delta logit)")
    ax = fig.add_subplot(gs[3])
    ax.bar(range(n), ig_r[:n], color=["#0072B2" if x >= 0 else "#D55E00" for x in ig_r[:n]])
    ax.set_xlim(-0.5, n - 0.5), ax.set_ylabel("integrated gradients\n(RNA)"), ax.set_xlabel("RNA position")
    ax = fig.add_subplot(gs[4])
    for s, e in doms:
        ax.axvspan(s - 1, e, color="#999999", alpha=0.25, lw=0)
    ax.plot(np.arange(plen), ig_p[:plen], color="#444444", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(0, plen), ax.set_ylabel("integrated gradients\n(protein)"), ax.set_xlabel("protein residue (grey = Pfam RNA-binding domain)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="name,type=v2|v4,path=...,rna_max=60,prot_max=700")
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--domain_positions", required=True)
    ap.add_argument("--max_proteins", type=int, default=0)
    ap.add_argument("--n_rna", type=int, default=30, help="positive RNAs per protein for the quantitative summaries")
    ap.add_argument("--plot_top", type=int, default=6, help="figures for this many proteins (best positive RNA each)")
    ap.add_argument("--motifs", default="", help="NAME=KMER,... positive control, e.g. RBFOX2=GCAUG,HNRNPC=UUUUU")
    ap.add_argument("--out_dir", default="results/mechanistic/saliency")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    sf = load_mod("suite51", "51_eclip_eval_suite.py")
    cfg = parse_model(args.model)
    model, fwd = sf.load_model(cfg["type"], resolve(cfg["path"]), device)
    model.eval()
    out = resolve(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    cols = ["protein_name", "protein_sequence", "rna_sequence", "binding_label"]
    test = pd.read_csv(resolve(args.test_tsv), sep="\t", usecols=cols, low_memory=False)
    dom = pd.read_csv(resolve(args.domain_positions), sep="\t")
    dom_by = {n: list(zip(g.start, g.end)) for n, g in dom.groupby("protein_name")}
    names = [n for n in dom_by if n in set(test.protein_name)]
    if args.max_proteins:
        names = names[: args.max_proteins]
    motifs = dict(kv.split("=") for kv in args.motifs.split(",") if kv)
    print(f"{len(names)} proteins with a domain; model {cfg['name']}; device {device}")

    rows, plotted = [], 0
    for n in names:
        g = test[(test.protein_name == n) & (test.binding_label == 1)]
        g = g[(g.rna_sequence.str.len() >= 15) & (g.rna_sequence.str.len() <= cfg["rna_max"])]
        if len(g) < 5:
            continue
        prot = g.protein_sequence.iloc[0]
        plen = min(len(prot), cfg["prot_max"])
        doms = [(s, e) for s, e in dom_by[n] if s <= plen]
        rnas = g.rna_sequence.drop_duplicates().sample(min(args.n_rna, g.rna_sequence.nunique()), random_state=args.seed).tolist()
        shuf = "".join(rng.permutation(list(prot)))
        stats, stats_sh, mot_in, mot_out, best = [], [], [], [], (-1e9, None)
        igp_sum = np.zeros(cfg["prot_max"])
        for rna in rnas:
            r, p = encode(sf, rna, prot, cfg, device)
            ig_r, ig_p = integrated_gradients(model, fwd, r, p)
            lg = float(logit(model, fwd, r, p)[0])
            igp_sum += ig_p
            s = domain_stats(ig_p, plen, doms, rng)
            if s:
                stats.append(s["enrichment"])
            _, p_sh = encode(sf, rna, shuf, cfg, device)
            _, igp_sh = integrated_gradients(model, fwd, r, p_sh)
            s = domain_stats(igp_sh, plen, doms, rng)
            if s:
                stats_sh.append(s["enrichment"])
            m = motifs.get(n)
            if m:
                u = rna.upper().replace("T", "U")
                hit = np.zeros(len(u), bool)
                for i in range(len(u) - len(m) + 1):
                    if u[i:i + len(m)] == m:
                        hit[i:i + len(m)] = True
                if hit.any() and (~hit).any():
                    mot_in.append(ig_r[:len(u)][hit].mean())
                    mot_out.append(ig_r[:len(u)][~hit].mean())
            if lg > best[0]:
                best = (lg, rna)
        agg = domain_stats(igp_sum, plen, doms, rng)
        row = {"protein": n, "n_rna": len(rnas), "domain_enrichment_mean": float(np.mean(stats)) if stats else np.nan,
               "domain_enrichment_shuffled_protein": float(np.mean(stats_sh)) if stats_sh else np.nan,
               "domain_enrichment_summed": agg["enrichment"] if agg else np.nan,
               "domain_p_perm_summed": agg["p_perm"] if agg else np.nan,
               "domain_frac": agg["domain_frac"] if agg else np.nan}
        if mot_in:
            row.update({"motif": motifs[n], "ig_in_motif": float(np.mean(mot_in)), "ig_outside_motif": float(np.mean(mot_out)),
                        "motif_in_gt_out_frac": float(np.mean(np.array(mot_in) > np.array(mot_out)))})
        rows.append(row)
        if plotted < args.plot_top and best[1]:
            rna = best[1]
            r, p = encode(sf, rna, prot, cfg, device)
            ig_r, ig_p = integrated_gradients(model, fwd, r, p)
            ism, win = mutagenesis(model, fwd, r, p, len(rna)), window_deletion(model, fwd, r, p, len(rna))
            plot_pair(out / f"{n}.png", f"{n} | {cfg['name']} | logit {best[0]:.2f} | positive RNA", rna, prot, ism, win,
                      ig_r, ig_p, doms, plen)
            plotted += 1
        print(f"  {n}: enrichment {row['domain_enrichment_mean']:.2f} (shuffled protein {row['domain_enrichment_shuffled_protein']:.2f})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_protein.tsv", sep="\t", index=False)
    summ = {"model": cfg, "n_proteins": len(df),
            "domain_enrichment_median": float(df.domain_enrichment_mean.median()),
            "domain_enrichment_shuffled_protein_median": float(df.domain_enrichment_shuffled_protein.median()),
            "frac_proteins_enrichment_gt1": float((df.domain_enrichment_mean > 1).mean()),
            "frac_proteins_p_perm_lt_0.05": float((df.domain_p_perm_summed < 0.05).mean())}
    try:
        from scipy.stats import wilcoxon
        d = (df.domain_enrichment_mean - df.domain_enrichment_shuffled_protein).dropna()
        summ["wilcoxon_p_real_vs_shuffled"] = float(wilcoxon(d).pvalue)
    except Exception:
        pass
    if "motif_in_gt_out_frac" in df:
        mm = df.dropna(subset=["motif_in_gt_out_frac"])
        summ["motif_check"] = {"n_proteins": len(mm), "mean_ig_in_motif": float(mm.ig_in_motif.mean()),
                               "mean_ig_outside": float(mm.ig_outside_motif.mean())}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
