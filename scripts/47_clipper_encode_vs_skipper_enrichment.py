#!/usr/bin/env python3
"""
47_clipper_encode_vs_skipper_enrichment.py
------------------------------------------
Compare ENCODE eCLIP narrowPeak signal (CLIPper pipeline, log2-style signalValue)
to Skipper Figshare enrichment_l2or_mean and Dropbox fixlen_151 BED column 5.

ENCODE IDR thresholded peaks are preferred per experiment (ENCSR from eclip_id).

Outputs:
  results/skipper_eclip/clipper_vs_skipper_panel.tsv
  results/skipper_eclip/clipper_vs_skipper_summary.json
  results/skipper_eclip/figures/clipper_signal_vs_skipper_l2or.png
"""

from __future__ import annotations

import gzip
import io
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

# Reuse ENCODE helpers from script 12
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("eclip12", ROOT / "scripts/12_download_eclip.py")
_e12 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_e12)
_get = _e12._get
ENCODE_BASE = _e12.ENCODE_BASE
HEADERS_JSON = _e12.HEADERS_JSON

INFO = ROOT / "data/raw/eclip_skipper/manifests/eclip_various_pos_neg_sets.hg38.info.tsv"
FIG_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.K562.tar"
FIG_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.HepG2.tar"
DROPBOX_TAR = ROOT / "data/raw/eclip_skipper/archives/dropbox/eclip_various_pos_neg_sets.hg38.tar.xz"
OUT = ROOT / "results/skipper_eclip"
FIG_DIR = OUT / "figures"

import tarfile


def download_narrowpeaks(href: str) -> list[dict]:
    url = ENCODE_BASE + href
    r = _get(url, {}, stream=True)
    content = b"".join(r.iter_content(chunk_size=65536))
    peaks = []
    with gzip.open(io.BytesIO(content), "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            try:
                signal = float(cols[6])
            except ValueError:
                signal = float("nan")
            peaks.append(
                {
                    "chrom": cols[0],
                    "start": int(cols[1]),
                    "end": int(cols[2]),
                    "signal": signal,
                }
            )
    return peaks


def pick_idr_bed_file(exp_accession: str) -> str | None:
    files = []
    for assembly in ("GRCh38", "GRCh37"):
        file_url = (
            f"{ENCODE_BASE}/search/?type=File"
            f"&dataset=/experiments/{exp_accession}/"
            f"&output_type=peaks"
            f"&file_format=bed"
            f"&assembly={assembly}"
            f"&status=released"
            f"&format=json"
            f"&frame=object"
            f"&limit=all"
        )
        try:
            r = _get(file_url, HEADERS_JSON)
            files = r.json().get("@graph", [])
        except requests.HTTPError:
            continue
        if files:
            break
    if not files:
        return None
    idr = [f for f in files if len(f.get("biological_replicates") or []) == 2]
    chosen = idr[0] if idr else files[0]
    return chosen.get("href")


def load_skipper_intervals(gz_bytes: bytes) -> dict[str, list[tuple[int, int, float]]]:
    intervals: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    with gzip.open(io.BytesIO(gz_bytes), "r") as f:
        header = f.readline().decode().rstrip("\n").split("\t")
        if "enrichment_l2or_mean" not in header:
            return {}
        ix_chr, ix_s, ix_e = header.index("chr"), header.index("start"), header.index("end")
        i_l2 = header.index("enrichment_l2or_mean")
        for line in f:
            p = line.decode().rstrip("\n").split("\t")
            try:
                intervals[p[ix_chr]].append(
                    (int(p[ix_s]), int(p[ix_e]), float(p[i_l2]))
                )
            except (ValueError, IndexError):
                continue
    for c in intervals:
        intervals[c].sort()
    return intervals


def max_overlap_val(chrom: str, start: int, end: int, intervals: dict) -> float | None:
    best = None
    for s, e, v in intervals.get(chrom, []):
        if s >= end:
            break
        if e <= start:
            continue
        if best is None or v > best:
            best = v
    return best


def peak_intervals(peaks: list[dict]) -> dict[str, list[tuple[int, int, float]]]:
    out: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for p in peaks:
        if np.isfinite(p["signal"]):
            out[p["chrom"]].append((p["start"], p["end"], p["signal"]))
    for c in out:
        out[c].sort()
    return out


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


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-experiments", type=int, default=0, help="0 = all fixlen_151")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    info = pd.read_csv(INFO, sep="\t")
    exps = info.loc[info["pos_bed"].str.contains("fixlen_151", na=False), "eclip_id"].tolist()
    if args.max_experiments:
        exps = exps[: args.max_experiments]

    print("Preloading Skipper Figshare …", flush=True)
    fig_gz = preload_figshare()

    rows = []
    pooled_clip, pooled_skip = [], []
    cache_enc: dict[str, list[dict]] = {}

    for i, eid in enumerate(exps):
        m = re.match(r"^(.+)_([^_]+)_(ENCSR[A-Z0-9]+)$", eid)
        if not m:
            continue
        symbol, cell, encsr = m.group(1), m.group(2), m.group(3)
        sample_key = f"{symbol}_{cell}"

        if encsr not in cache_enc:
            href = pick_idr_bed_file(encsr)
            if not href:
                cache_enc[encsr] = []
            else:
                try:
                    cache_enc[encsr] = download_narrowpeaks(href)
                    time.sleep(0.15)
                except Exception:
                    cache_enc[encsr] = []
        peaks = cache_enc[encsr]
        if not peaks:
            rows.append({"eclip_id": eid, "n_clipper_peaks": 0})
            continue

        sk_key = (cell, sample_key)
        if sk_key not in fig_gz:
            continue
        skip_iv = load_skipper_intervals(fig_gz[sk_key])
        if not skip_iv:
            continue
        clip_iv = peak_intervals(peaks)

        clip_vals, skip_vals = [], []
        for chrom, ivs in skip_iv.items():
            for s, e, l2 in ivs:
                sig = max_overlap_val(chrom, s, e, clip_iv)
                if sig is not None:
                    clip_vals.append(sig)
                    skip_vals.append(l2)

        rho = float("nan")
        if len(clip_vals) >= 30:
            rho = float(spearmanr(clip_vals, skip_vals).statistic)
            n_sample = min(400, len(clip_vals))
            idx = random.Random(args.seed + i).sample(range(len(clip_vals)), n_sample)
            for j in idx:
                pooled_clip.append(clip_vals[j])
                pooled_skip.append(skip_vals[j])

        rows.append(
            {
                "eclip_id": eid,
                "symbol": symbol,
                "cell_line": cell,
                "encode_experiment": encsr,
                "n_clipper_peaks": len(peaks),
                "n_skipper_windows_matched_clipper": len(clip_vals),
                "spearman_clipper_signal_vs_skipper_l2or": rho,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(exps)}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "clipper_vs_skipper_panel.tsv", sep="\t", index=False)

    s = df["spearman_clipper_signal_vs_skipper_l2or"].dropna()
    summary = {
        "description": "ENCODE narrowPeak signalValue (CLIPper IDR peaks) vs Skipper enrichment_l2or_mean on overlapping windows",
        "n_experiments": len(exps),
        "n_with_clipper_peaks": int((df["n_clipper_peaks"] > 0).sum()),
        "n_with_spearman": int(len(s)),
        "spearman_per_experiment": {
            "median": float(s.median()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
            "pct_gt_0.5": float((s > 0.5).mean()) if len(s) else None,
        },
        "pooled_overlap_windows": len(pooled_clip),
        "pooled_spearman": float(spearmanr(pooled_clip, pooled_skip).statistic)
        if len(pooled_clip) >= 100
        else None,
    }
    (OUT / "clipper_vs_skipper_summary.json").write_text(json.dumps(summary, indent=2))

    if len(pooled_clip) >= 100:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        x, y = np.array(pooled_clip), np.array(pooled_skip)
        rho = spearmanr(x, y).statistic
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
        ax.hexbin(x, y, gridsize=50, cmap="viridis", mincnt=1, bins="log")
        lim = max(x.max(), y.max())
        ax.plot([0, lim], [0, lim], "w--", lw=0.8, alpha=0.6)
        ax.set_xlabel("ENCODE CLIPper IDR peak signalValue (narrowPeak col7)")
        ax.set_ylabel("Skipper enrichment_l2or_mean")
        ax.set_title(f"Overlapping windows (n={len(x):,})\nSpearman ρ = {rho:.3f}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "clipper_signal_vs_skipper_l2or.png", bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
