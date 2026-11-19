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
  domains  Pfam hmmscan on the proteins (local, needs hmmscan and data/external/pfam/Pfam-A.hmm) -> out_dir/domains.tsv (RNA-binding
           families only); the embed stage then also saves embeddings pooled over the domain residues only (emb_domain)
  bootstrap  95% intervals over proteins (resampling with replacement, pairs of the same protein dropped) for every layer and for
           the difference to layer 0 and to layer 33 -> out_dir/layer_bootstrap.tsv

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


FAMILIES = [("RRM", "RRM"), ("KH_", "KH"), ("zf-CCCH", "CCCH"), ("zf-CCHC", "CCHC"), ("zf-C2H2", "C2H2"), ("DEAD", "DEAD"),
            ("dsrm", "dsRBD"), ("PUF", "PUF"), ("CSD", "CSD"), ("La", "La"), ("YTH", "YTH"), ("S1", "S1"), ("Piwi", "Piwi"), ("PAZ", "PAZ")]


def stage_domains(args) -> None:
    import subprocess
    import tempfile
    out = resolve(args.out_dir)
    seqs = sorted(pd.read_csv(out / "experiments.tsv", sep="\t").protein_sequence.unique())
    with tempfile.TemporaryDirectory() as td:
        fa, tbl = Path(td) / "q.fa", Path(td) / "dom.tbl"
        fa.write_text("".join(f">s{i}\n{q}\n" for i, q in enumerate(seqs)))
        subprocess.run(["hmmscan", "--cpu", "4", "-E", "1e-3", "--domE", "1e-3", "--domtblout", str(tbl), str(resolve(args.pfam)), str(fa)],
                       check=True, stdout=subprocess.DEVNULL)
        rows = []
        for line in tbl.read_text().splitlines():
            if line.startswith("#"):
                continue
            f = line.split()
            fam = next((fm for pre, fm in FAMILIES if f[0] == pre or f[0].startswith(pre)), None)
            if fam:
                i = int(f[3][1:])
                rows.append((seqs[i], i, f[0], fam, float(f[12]), int(f[19]), int(f[20])))
    df = pd.DataFrame(rows, columns=["protein_sequence", "seq_index", "pfam", "family", "domain_ievalue", "env_from", "env_to"])
    df.to_csv(out / "domains.tsv", sep="\t", index=False)
    n = df.protein_sequence.nunique()
    print(f"{n} of {len(seqs)} proteins with an RNA-binding Pfam domain; families {df.drop_duplicates(['seq_index', 'family']).family.value_counts().to_dict()}")


def embed_one(model, converter, seq: str, device, layers: list[int], mask: np.ndarray | None = None):
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
    res = acc / cnt[None, :, None]
    nan = np.full((len(layers), 1280), np.nan, dtype=np.float32)
    dom = res[:, mask].mean(axis=1) if mask is not None and mask.any() else nan
    outside = res[:, ~mask].mean(axis=1) if mask is not None and (~mask).sum() >= 10 else nan
    return res.mean(axis=1), dom, outside


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
    masks = {}
    if (out / "domains.tsv").exists():
        for r in pd.read_csv(out / "domains.tsv", sep="\t").itertuples():
            m = masks.setdefault(r.protein_sequence, np.zeros(len(r.protein_sequence), dtype=bool))
            m[r.env_from - 1:r.env_to] = True
    emb, dom, outs, t0 = [], [], [], time.time()
    for i, s in enumerate(seqs, 1):
        a, b, c = embed_one(model, conv, s, device, layers, masks.get(s))
        emb.append(a)
        dom.append(b)
        outs.append(c)
        if i % 25 == 0:
            print(f"  {i}/{len(seqs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "esm2_layers.npz", emb=np.stack(emb).astype(np.float32), emb_domain=np.stack(dom).astype(np.float32), emb_outside=np.stack(outs).astype(np.float32),
             seqs=np.array(seqs, dtype=object))
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


def load_inputs(args, out: Path):
    exp = pd.read_csv(out / "experiments.tsv", sep="\t")
    z = np.load(out / "esm2_layers.npz", allow_pickle=True)
    seqs = list(z["seqs"])
    zs = pd.read_csv(resolve(args.zscores), sep="\t", index_col=0)
    zs = zs.loc[:, ~zs.columns.duplicated()]
    vec, keep = [], []
    for i, s in enumerate(seqs):
        ids = [e for e in exp[exp.protein_sequence == s].experiment_id.unique() if e in zs.columns]
        if ids:
            vec.append(zs[ids].mean(axis=1).to_numpy(dtype=np.float64))
            keep.append(i)
    V = np.vstack(vec)
    return np.corrcoef(V - V.mean(axis=1, keepdims=True)), z["emb"][keep], zs.shape[0]


def stage_bootstrap(args) -> None:
    out = resolve(args.out_dir)
    bind, emb, _ = load_inputs(args, out)
    n, n_layers = len(bind), emb.shape[1]
    mats = {tag: [cosine(emb[:, L], c) for L in range(n_layers)] for tag, c in (("raw", False), ("centred", True))}
    rng = np.random.default_rng(args.seed)
    iu = np.triu_indices(n, k=1)
    res = {tag: np.zeros((args.n_boot, n_layers)) for tag in mats}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        a, c = idx[iu[0]], idx[iu[1]]
        ok = a != c
        rb = rankdata(bind[a[ok], c[ok]])
        for tag in mats:
            for L in range(n_layers):
                res[tag][b, L] = np.corrcoef(rb, rankdata(mats[tag][L][a[ok], c[ok]]))[0, 1]
        if (b + 1) % 25 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}", flush=True)
    q = lambda x: np.percentile(x, [2.5, 97.5], axis=0)  # noqa: E731
    df = pd.DataFrame({"layer": range(n_layers)})
    for tag in mats:
        lo, hi = q(res[tag])
        df[f"spearman_{tag}_lo"], df[f"spearman_{tag}_hi"] = lo, hi
        for ref in (0, n_layers - 1):
            d = res[tag] - res[tag][:, [ref]]
            dl, dh = q(d)
            df[f"diff_vs{ref}_{tag}_lo"], df[f"diff_vs{ref}_{tag}_hi"] = dl, dh
    df.to_csv(out / "layer_bootstrap.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False))


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
    full = None
    if args.pool in ("domain", "outside"):
        dom = z["emb_domain" if args.pool == "domain" else "emb_outside"][keep]
        ok = np.isfinite(dom[:, 0, 0])
        print(f"{args.pool} pooling: {int(ok.sum())} of {len(ok)} proteins usable (domain hit; outside: at least 10 residues outside)", flush=True)
        bind, full, emb, keep = bind[np.ix_(ok, ok)], emb[ok], dom[ok], [k for k, o in zip(keep, ok) if o]
    rng = np.random.default_rng(args.seed)
    rows = []
    for L in range(emb.shape[1]):
        r = {"layer": L}
        for tag, centre in (("raw", False), ("centred", True)):
            rho, p = spearman_mantel(cosine(emb[:, L], centre), bind, args.n_perm, rng)
            r[f"spearman_{tag}"], r[f"mantel_p_{tag}"] = rho, p
            if full is not None:
                r[f"spearman_{tag}_full_pool"], r[f"mantel_p_{tag}_full_pool"] = spearman_mantel(cosine(full[:, L], centre), bind, args.n_perm, rng)
        rows.append(r)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.insert(1, "n_proteins", len(keep))
    suffix = "" if args.pool == "full" else f"_{args.pool}"
    df.to_csv(out / f"layer_correlation{suffix}.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(df.layer, df.spearman_raw, marker="o", ms=3, label="cosine, raw")
    ax.plot(df.layer, df.spearman_centred, marker="s", ms=3, label="cosine, centred over proteins")
    ax.set_xlabel("ESM-2 650M layer")
    ax.set_ylabel(f"Spearman with 7-mer Z-score Pearson (n={len(keep)})")
    ax.legend(frameon=False)
    fig.tight_layout()
    if full is not None:
        ax.plot(df.layer, df.spearman_raw_full_pool, ls="--", color="C0", label="raw, whole sequence (same proteins)")
        ax.plot(df.layer, df.spearman_centred_full_pool, ls="--", color="C1", label="centred, whole sequence (same proteins)")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / f"fig_layer_correlation{suffix}.png", dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["meta", "domains", "embed", "analyze", "bootstrap"])
    ap.add_argument("--pairs", default="data/benchmarks/rnacompete/rnacompete_human.tsv")
    ap.add_argument("--zscores", default="rnacompete_analysis/data/raw/Zscores_RNAcompete2025.txt.gz")
    ap.add_argument("--out_dir", default="results/esm_layer_profile")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--pool", choices=["full", "domain", "outside"], default="full", help="analyze: pooling over the whole sequence or the Pfam RNA-binding domains")
    ap.add_argument("--pfam", default="data/external/pfam/Pfam-A.hmm")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"meta": stage_meta, "domains": stage_domains, "embed": stage_embed, "analyze": stage_analyze, "bootstrap": stage_bootstrap}[args.stage](args)


if __name__ == "__main__":
    main()
