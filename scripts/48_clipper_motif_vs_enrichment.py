#!/usr/bin/env python3
"""
48_clipper_motif_vs_enrichment.py
---------------------------------
On fixlen_151 positive windows (Dropbox FASTA+BED), compare:
  - CLIPper IDR peak signalValue (max overlap on window)
  - Skipper BED column 5
  - HOMER motif1 PWM log-score (Figshare)

Answers whether CLIPper enrichment correlates with motifs on the same windows
alongside col5.

Caches ENCODE IDR peaks under results/skipper_eclip/cache/encode_idr_peaks/.

Outputs:
  results/skipper_eclip/clipper_motif_enrichment_panel.tsv
  results/skipper_eclip/clipper_motif_enrichment_summary.json
"""

from __future__ import annotations

import importlib.util
import json
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

_spec47 = importlib.util.spec_from_file_location("s47", ROOT / "scripts/47_clipper_encode_vs_skipper_enrichment.py")
s47 = importlib.util.module_from_spec(_spec47)
_spec47.loader.exec_module(s47)

_spec45 = importlib.util.spec_from_file_location("s45", ROOT / "scripts/45_skipper_enrichment_homer_qc.py")
s45 = importlib.util.module_from_spec(_spec45)
_spec45.loader.exec_module(s45)

INFO = ROOT / "data/raw/eclip_skipper/manifests/eclip_various_pos_neg_sets.hg38.info.tsv"
DROPBOX_TAR = ROOT / "data/raw/eclip_skipper/archives/dropbox/eclip_various_pos_neg_sets.hg38.tar.xz"
FASTA_DIR = ROOT / "data/raw/eclip_skipper/extracted/fixlen_151_fasta"
HOMER_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_homer.K562.tgz"
HOMER_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_homer.HepG2.tgz"
OUT = ROOT / "results/skipper_eclip"
CACHE = OUT / "cache/encode_idr_peaks"


def load_peaks_cached(encsr: str) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{encsr}.json"
    if path.is_file():
        return json.loads(path.read_text())
    href = s47.pick_idr_bed_file(encsr)
    if not href:
        peaks = []
    else:
        try:
            peaks = s47.download_narrowpeaks(href)
            time.sleep(0.12)
        except Exception:
            peaks = []
    path.write_text(json.dumps(peaks))
    return peaks


def preload_homer() -> dict[tuple[str, str], tuple[str, np.ndarray]]:
    out: dict[tuple[str, str], tuple[str, np.ndarray]] = {}
    for cell, tar_path in [("K562", HOMER_K562), ("HepG2", HOMER_HEPG2)]:
        with tarfile.open(tar_path, "r:gz") as tf:
            for name in tf.getnames():
                m = re.search(
                    r"homer/finemapped_results/([^/]+)/homerResults/motif1\.motif$",
                    name,
                )
                if not m:
                    continue
                raw = tf.extractfile(name)
                if not raw:
                    continue
                parsed = s45.parse_homer_motif_bytes(raw.read())
                if parsed:
                    out[(cell, m.group(1))] = parsed
    return out


def main() -> None:
    info = pd.read_csv(INFO, sep="\t")
    exps = info.loc[info["pos_bed"].str.contains("fixlen_151", na=False), "eclip_id"].tolist()

    print("Loading HOMER motif1 …", flush=True)
    homer = preload_homer()

    rows = []
    with tarfile.open(DROPBOX_TAR, "r:xz") as drop_tf:
        for i, eid in enumerate(exps):
            m = re.match(r"^(.+)_([^_]+)_(ENCSR[A-Z0-9]+)$", eid)
            if not m:
                continue
            symbol, cell, encsr = m.group(1), m.group(2), m.group(3)
            hm = (cell, f"{symbol}_{cell}")
            if hm not in homer:
                continue
            _, pwm = homer[hm]
            fa = FASTA_DIR / f"{eid}.fixlen_151.positives.fa"
            if not fa.is_file():
                continue
            seqs = s45.parse_fasta(fa)
            peaks = load_peaks_cached(encsr)
            if not peaks:
                continue
            clip_iv = s47.peak_intervals(peaks)

            bed_m = f"pos_neg_fasta_out/{eid}.fixlen_151.positives.bed"
            f = drop_tf.extractfile(bed_m)
            if f is None:
                continue

            col5_l, clip_l, mot_l = [], [], []
            for line in f.read().decode().splitlines():
                p = line.split("\t")
                if len(p) < 5:
                    continue
                hid, chrom = p[3], p[0]
                s, e = int(p[1]), int(p[2])
                if hid not in seqs:
                    continue
                ms = s45.motif_log_score(seqs[hid], pwm)
                if not np.isfinite(ms):
                    continue
                cs = s47.max_overlap_val(chrom, s, e, clip_iv)
                if cs is None:
                    continue
                col5_l.append(float(p[4]))
                clip_l.append(cs)
                mot_l.append(ms)

            if len(col5_l) < 30:
                continue
            rows.append(
                {
                    "eclip_id": eid,
                    "symbol": symbol,
                    "n_windows": len(col5_l),
                    "spearman_clipper_signal_vs_motif": float(spearmanr(clip_l, mot_l).statistic),
                    "spearman_bed_col5_vs_motif": float(spearmanr(col5_l, mot_l).statistic),
                    "spearman_clipper_signal_vs_col5": float(spearmanr(clip_l, col5_l).statistic),
                }
            )
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(exps)}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "clipper_motif_enrichment_panel.tsv", sep="\t", index=False)

    def agg(s: pd.Series) -> dict:
        s = s.dropna()
        return {"n": int(len(s)), "median": float(s.median()), "mean": float(s.mean())}

    summary = {
        "unit": "fixlen_151 positive windows with CLIPper IDR overlap + HOMER motif1 PWM",
        "note": "CLIPper signal = max overlapping IDR narrowPeak signalValue on each 151nt window",
        "n_experiments": len(df),
        "clipper_signal_vs_motif_pwm": agg(df["spearman_clipper_signal_vs_motif"]),
        "bed_col5_vs_motif_pwm": agg(df["spearman_bed_col5_vs_motif"]),
        "clipper_signal_vs_bed_col5": agg(df["spearman_clipper_signal_vs_col5"]),
    }
    (OUT / "clipper_motif_enrichment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
