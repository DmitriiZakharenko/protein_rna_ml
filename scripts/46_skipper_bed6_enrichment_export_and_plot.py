#!/usr/bin/env python3
"""
46_skipper_bed6_enrichment_export_and_plot.py
---------------------------------------------
1) Plot: Dropbox fixlen_151 BED col5 vs Figshare enrichment_l2or_mean (matched windows).
2) Export: BED6 with enrichment_l2or_mean in column 5 from Figshare (new files for the CLIPper-style workflow).

Examples
--------
  python scripts/46_skipper_bed6_enrichment_export_and_plot.py --plot-only

  python scripts/46_skipper_bed6_enrichment_export_and_plot.py \\
    --export-sample PTBP1_K562 --out_bed results/skipper_eclip/bed6/PTBP1_K562.reproducible_enriched.bed6
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

DROPBOX_TAR = ROOT / "data/raw/eclip_skipper/archives/dropbox/eclip_various_pos_neg_sets.hg38.tar.xz"
FIG_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.K562.tar"
FIG_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.HepG2.tar"
INFO = ROOT / "data/raw/eclip_skipper/manifests/eclip_various_pos_neg_sets.hg38.info.tsv"
PANEL = ROOT / "results/skipper_eclip/enrichment_l2or_col5_panel.tsv"
FIG_DIR = ROOT / "results/skipper_eclip/figures"
BED6_DIR = ROOT / "results/skipper_eclip/bed6"


def load_enriched_intervals(gz_bytes: bytes) -> dict[str, list[tuple[int, int, float]]]:
    intervals: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    with gzip.open(io.BytesIO(gz_bytes), "r") as f:
        header = f.readline().decode().rstrip("\n").split("\t")
        ix_chr, ix_s, ix_e = header.index("chr"), header.index("start"), header.index("end")
        i_l2 = header.index("enrichment_l2or_mean")
        for line in f:
            p = line.decode().rstrip("\n").split("\t")
            try:
                s, e = int(p[ix_s]), int(p[ix_e])
                intervals[p[ix_chr]].append((s, e, float(p[i_l2])))
            except (ValueError, IndexError):
                continue
    for c in intervals:
        intervals[c].sort()
    return intervals


def max_overlap_l2or(chrom: str, start: int, end: int, intervals: dict) -> float | None:
    ivs = intervals.get(chrom, [])
    best = None
    for s, e, l2 in ivs:
        if s >= end:
            break
        if e <= start:
            continue
        if best is None or l2 > best:
            best = l2
    return best


def preload_figshare() -> dict[tuple[str, str], bytes]:
    out: dict[tuple[str, str], bytes] = {}
    for cell, tar_path in [("K562", FIG_K562), ("HepG2", FIG_HEPG2)]:
        with tarfile.open(tar_path, "r:") as tf:
            for name in tf.getnames():
                if not name.endswith(".reproducible_enriched_windows.tsv.gz"):
                    continue
                key = name.split("/")[-1].replace(".reproducible_enriched_windows.tsv.gz", "")
                raw = tf.extractfile(name)
                if raw:
                    out[(cell, key)] = raw.read()
    return out


def collect_matched_pairs(
    fig_gz: dict[tuple[str, str], bytes],
    max_experiments: int = 80,
    max_points_per_exp: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    panel = pd.read_csv(PANEL, sep="\t")
    panel = panel.dropna(subset=["spearman_col5_vs_enrichment_l2or_mean"])
    panel = panel.sort_values("spearman_col5_vs_enrichment_l2or_mean", ascending=False)
    eids = panel["eclip_id"].tolist()[:max_experiments]

    rows: list[dict] = []
    interval_cache: dict[tuple[str, str], dict] = {}

    with tarfile.open(DROPBOX_TAR, "r:xz") as drop_tf:
        for eid in eids:
            parts = eid.split("_")
            if len(parts) < 3:
                continue
            cell = parts[1]
            sample_key = f"{parts[0]}_{cell}"
            ck = (cell, sample_key)
            if ck not in fig_gz:
                continue
            if ck not in interval_cache:
                interval_cache[ck] = load_enriched_intervals(fig_gz[ck])
            intervals = interval_cache[ck]
            bed_m = f"pos_neg_fasta_out/{eid}.fixlen_151.positives.bed"
            f = drop_tf.extractfile(bed_m)
            if f is None:
                continue
            pairs = []
            for line in f.read().decode().splitlines():
                p = line.split("\t")
                if len(p) < 5:
                    continue
                chrom, s, e = p[0], int(p[1]), int(p[2])
                col5 = float(p[4])
                l2 = max_overlap_l2or(chrom, s, e, intervals)
                if l2 is not None:
                    pairs.append((col5, l2))
            if len(pairs) > max_points_per_exp:
                pairs = rng.sample(pairs, max_points_per_exp)
            for col5, l2 in pairs:
                rows.append({"eclip_id": eid, "symbol": parts[0], "bed_col5": col5, "enrichment_l2or_mean": l2})
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_path: Path) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = df["bed_col5"].to_numpy()
    y = df["enrichment_l2or_mean"].to_numpy()
    rho = float(spearmanr(x, y).statistic)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax = axes[0]
    hb = ax.hexbin(x, y, gridsize=55, cmap="YlOrRd", mincnt=1, bins="log")
    ax.plot([0, max(x.max(), y.max())], [0, max(x.max(), y.max())], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Dropbox fixlen_151 BED column 5")
    ax.set_ylabel("Figshare enrichment_l2or_mean (max overlap)")
    ax.set_title(f"Pooled matched windows (n={len(df):,})\nSpearman ρ = {rho:.3f}")
    fig.colorbar(hb, ax=ax, label="log10(count)")

    ax2 = axes[1]
    panel = pd.read_csv(PANEL, sep="\t")
    s = panel["spearman_col5_vs_enrichment_l2or_mean"].dropna()
    ax2.hist(s, bins=25, color="#4e79a7", edgecolor="white")
    ax2.axvline(s.median(), color="#e15759", ls="--", lw=1.5, label=f"median ρ = {s.median():.2f}")
    ax2.set_xlabel("Per-experiment Spearman(col5, enrichment_l2or_mean)")
    ax2.set_ylabel("# experiments")
    ax2.set_title(f"Across panel (n={len(s)} exps)")
    ax2.legend(fontsize=8)

    fig.suptitle("Skipper enrichment: existing BED col5 vs Figshare enrichment_l2or", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"n_points": len(df), "pooled_spearman": rho, "out_png": str(out_path)}


def export_figshare_bed6(sample_key: str, out_path: Path) -> int:
    """Write BED6 from reproducible_enriched_windows (col5 = enrichment_l2or_mean)."""
    cell = sample_key.split("_")[-1]
    if cell not in ("K562", "HepG2"):
        raise SystemExit(f"Expected SYMBOL_K562 or SYMBOL_HepG2, got {sample_key}")
    tar_path = FIG_K562 if cell == "K562" else FIG_HEPG2
    member = f"reproducible_enriched_windows/{sample_key}.reproducible_enriched_windows.tsv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r:") as tf, out_path.open("w") as out:
        raw = tf.extractfile(member)
        if raw is None:
            raise SystemExit(f"Missing {member} in {tar_path}")
        with gzip.open(io.BytesIO(raw.read()), "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
            ix = {c: header.index(c) for c in ["chr", "start", "end", "name", "strand", "enrichment_l2or_mean"]}
            for line in f:
                p = line.rstrip("\n").split("\t")
                try:
                    score = float(p[ix["enrichment_l2or_mean"]])
                except (ValueError, KeyError):
                    continue
                out.write(
                    f"{p[ix['chr']]}\t{p[ix['start']]}\t{p[ix['end']]}\t{p[ix['name']]}\t{score}\t{p[ix['strand']]}\n"
                )
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-sample", type=str, default="", help="e.g. PTBP1_K562")
    ap.add_argument("--out_bed", type=str, default="")
    ap.add_argument("--max-experiments", type=int, default=80)
    args = ap.parse_args()

    if args.export_sample:
        out = Path(args.out_bed) if args.out_bed else BED6_DIR / f"{args.export_sample}.enrichment_l2or.bed6"
        if not out.is_absolute():
            out = ROOT / out
        n = export_figshare_bed6(args.export_sample, out)
        print(f"Wrote {n} lines to {out}")

    print("Loading Figshare tables …", flush=True)
    fig_gz = preload_figshare()
    print("Collecting matched window pairs …", flush=True)
    df = collect_matched_pairs(fig_gz, max_experiments=args.max_experiments)
    cache = ROOT / "results/skipper_eclip/enrichment_col5_l2or_matched_sample.tsv"
    df.to_csv(cache, sep="\t", index=False)
    meta = make_plot(df, FIG_DIR / "enrichment_bed_col5_vs_l2or_mean.png")
    meta["matched_points_tsv"] = str(cache.relative_to(ROOT))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
