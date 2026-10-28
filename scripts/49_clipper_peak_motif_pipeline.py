#!/usr/bin/env python3
"""
49_clipper_peak_motif_pipeline.py
-----------------------------------
Full CLIPper-based eCLIP enrichment workflow (CLIPper / ENCODE IDR):

1. Export BED6 files: 75 nt windows centered on CLIPper peak summits,
   column 5 = narrowPeak signalValue (log2 IP/input enrichment).
2. Fetch RNA sequences at those windows (Ensembl REST).
3. Score Skipper/HOMER motif1 PWM; Spearman(enrichment, motif) per experiment.
4. Quartile motif-rate QC (high vs low enrichment peaks).

Uses cached IDR peaks from scripts/47/48 (encode_idr_peaks cache).

Outputs:
  results/skipper_eclip/bed6/clipper_idr_75nt/*.bed6
  results/skipper_eclip/bed6/skipper_figshare_l2or/*.bed6  (if --export-skipper)
  results/skipper_eclip/clipper_peak_pipeline_summary.json
  results/skipper_eclip/figures/clipper_peak_75nt_signal_vs_motif.png
  results/skipper_eclip/clipper_peak_75nt_motif_panel.tsv
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import random
import re
import tarfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
INFO = ROOT / "data/raw/eclip_skipper/manifests/eclip_various_pos_neg_sets.hg38.info.tsv"
FIG_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.K562.tar"
FIG_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.HepG2.tar"
HOMER_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_homer.K562.tgz"
HOMER_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_homer.HepG2.tgz"
OUT = ROOT / "results/skipper_eclip"
BED_CLIPPER = OUT / "bed6/clipper_idr_75nt"
BED_SKIPPER = OUT / "bed6/skipper_figshare_l2or"
FIG_DIR = OUT / "figures"
PEAK_CACHE = OUT / "cache/encode_idr_peaks"
SEQ_CACHE = OUT / "cache/ensembl_peak75nt"

WIN_LEN = 75
HALF = WIN_LEN // 2  # 37 + 38 = 75

_spec12 = importlib.util.spec_from_file_location("e12", ROOT / "scripts/12_download_eclip.py")
e12 = importlib.util.module_from_spec(_spec12)
_spec12.loader.exec_module(e12)

_spec47 = importlib.util.spec_from_file_location("s47", ROOT / "scripts/47_clipper_encode_vs_skipper_enrichment.py")
s47 = importlib.util.module_from_spec(_spec47)
_spec47.loader.exec_module(s47)

_spec45 = importlib.util.spec_from_file_location("s45", ROOT / "scripts/45_skipper_enrichment_homer_qc.py")
s45 = importlib.util.module_from_spec(_spec45)
_spec45.loader.exec_module(s45)


def parse_narrowpeak_peaks(raw: bytes) -> list[dict]:
    peaks = []
    with gzip.open(io.BytesIO(raw), "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            chrom = cols[0]
            if "_" in chrom and chrom.startswith("chr"):
                # skip many random/unplaced for Ensembl success rate
                if "random" in chrom or "alt" in chrom or "fix" in chrom:
                    continue
            start, end = int(cols[1]), int(cols[2])
            name = cols[3]
            strand = cols[5] if cols[5] in ("+", "-") else "+"
            try:
                signal = float(cols[6])
            except ValueError:
                continue
            if len(cols) > 9 and cols[9] not in (".", ""):
                summit = start + int(cols[9])
            else:
                summit = start + (end - start) // 2
            w_start = max(0, summit - HALF)
            w_end = w_start + WIN_LEN
            peaks.append(
                {
                    "chrom": chrom,
                    "peak_start": start,
                    "peak_end": end,
                    "summit": summit,
                    "win_start": w_start,
                    "win_end": w_end,
                    "name": name,
                    "strand": strand,
                    "signalValue": signal,
                }
            )
    return peaks


def load_peaks_for_encsr(encsr: str) -> list[dict]:
    PEAK_CACHE.mkdir(parents=True, exist_ok=True)
    np_cache = PEAK_CACHE / f"{encsr}.narrowPeak.gz"
    if np_cache.is_file():
        return parse_narrowpeak_peaks(np_cache.read_bytes())
    href = s47.pick_idr_bed_file(encsr)
    if not href:
        return []
    url = s47.ENCODE_BASE + href
    raw = b"".join(e12._get(url, {}, stream=True).iter_content(chunk_size=65536))
    np_cache.write_bytes(raw)
    time.sleep(0.1)
    return parse_narrowpeak_peaks(raw)


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


def seq_cache_key(chrom: str, start: int, end: int, strand: str) -> Path:
    SEQ_CACHE.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{chrom}_{start}_{end}_{strand}")
    return SEQ_CACHE / f"{safe}.rna"


def fetch_seq(chrom: str, start: int, end: int, strand: str) -> str | None:
    path = seq_cache_key(chrom, start, end, strand)
    if path.is_file():
        return path.read_text().strip() or None
    # Ensembl uses 1-based inclusive
    reg = {"chrom": chrom, "start": start + 1, "end": end, "strand": strand}
    seqs = e12.fetch_sequences_batch([reg])
    seq = seqs[0] if seqs else None
    if seq:
        path.write_text(seq)
    return seq


def stratified_sample(peaks: list[dict], n: int, seed: int) -> list[dict]:
    if len(peaks) <= n:
        return peaks
    signals = np.array([p["signalValue"] for p in peaks])
    qs = np.quantile(signals, [0.25, 0.5, 0.75])
    bins = [[] for _ in range(4)]
    for p in peaks:
        s = p["signalValue"]
        if s <= qs[0]:
            bins[0].append(p)
        elif s <= qs[1]:
            bins[1].append(p)
        elif s <= qs[2]:
            bins[2].append(p)
        else:
            bins[3].append(p)
    rng = random.Random(seed)
    per = max(1, n // 4)
    out = []
    for b in bins:
        if not b:
            continue
        out.extend(rng.sample(b, min(per, len(b))))
    if len(out) < n:
        rest = [p for p in peaks if p not in out]
        out.extend(rng.sample(rest, min(n - len(out), len(rest))))
    return out[:n]


def write_clipper_bed6(path: Path, peaks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for p in peaks:
            out.write(
                f"{p['chrom']}\t{p['win_start']}\t{p['win_end']}\t{p['name']}\t{p['signalValue']}\t{p['strand']}\n"
            )


def export_skipper_bed6(sample_key: str, cell: str, out_path: Path) -> int:
    tar_path = FIG_K562 if cell == "K562" else FIG_HEPG2
    member = f"reproducible_enriched_windows/{sample_key}.reproducible_enriched_windows.tsv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r:") as tf:
        if member not in tf.getnames():
            return 0
        raw = tf.extractfile(member)
        if raw is None:
            return 0
        with gzip.open(io.BytesIO(raw.read()), "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
            if "enrichment_l2or_mean" not in header:
                return 0
            ix = {c: header.index(c) for c in ["chr", "start", "end", "name", "strand", "enrichment_l2or_mean"]}
            with out_path.open("w") as out:
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
    ap.add_argument("--max-peaks-motif", type=int, default=150, help="Peaks per ENCSR for Ensembl+motif QC")
    ap.add_argument("--export-skipper-bed6", action="store_true", help="Export all Skipper Figshare BED6")
    ap.add_argument("--bed6-only", action="store_true", help="Only write CLIPper 75nt BED6 (no Ensembl)")
    ap.add_argument("--motif-only", action="store_true", help="Skip BED6 write; run motif QC on existing peaks cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    info = pd.read_csv(INFO, sep="\t")
    exps = info.loc[info["pos_bed"].str.contains("fixlen_151", na=False), "eclip_id"].unique()

    by_encsr: dict[str, tuple[str, str, str]] = {}
    for eid in exps:
        m = re.match(r"^(.+)_([^_]+)_(ENCSR[A-Z0-9]+)$", eid)
        if not m:
            continue
        sym, cell, encsr = m.group(1), m.group(2), m.group(3)
        by_encsr.setdefault(encsr, (sym, cell, eid))

    homer = preload_homer()
    print(f"Unique ENCODE experiments: {len(by_encsr)}", flush=True)

    rows = []
    pooled_sig, pooled_mot = [], []
    q_top_mot, q_bot_mot = [], []

    for i, (encsr, (symbol, cell, eid)) in enumerate(sorted(by_encsr.items())):
        sample_key = f"{symbol}_{cell}"
        hm = (cell, sample_key)
        if hm not in homer:
            continue
        _, pwm = homer[hm]
        peaks = load_peaks_for_encsr(encsr)
        if not peaks:
            continue

        bed_path = BED_CLIPPER / f"{symbol}_{cell}_{encsr}.clipper_idr_75nt.bed6"
        if not args.motif_only:
            write_clipper_bed6(bed_path, peaks)

        if args.bed6_only:
            continue

        sample = stratified_sample(peaks, args.max_peaks_motif, args.seed + i)
        signals, motifs = [], []
        for p in sample:
            seq = fetch_seq(p["chrom"], p["win_start"], p["win_end"], p["strand"])
            if not seq or len(seq) < 8:
                continue
            ms = s45.motif_log_score(seq, pwm)
            if not np.isfinite(ms):
                continue
            signals.append(p["signalValue"])
            motifs.append(ms)

        if len(signals) < 30:
            continue
        rho = float(spearmanr(signals, motifs).statistic)
        sig_arr = np.array(signals)
        mot_arr = np.array(motifs)
        q75, q25 = np.quantile(sig_arr, [0.75, 0.25])
        top = mot_arr[sig_arr >= q75]
        bot = mot_arr[sig_arr <= q25]
        rows.append(
            {
                "encode_experiment": encsr,
                "symbol": symbol,
                "cell_line": cell,
                "n_idr_peaks": len(peaks),
                "n_peaks_sequenced": len(signals),
                "spearman_signalValue_vs_motif_pwm": rho,
                "motif_pwm_mean_top_quartile_signal": float(top.mean()) if len(top) else float("nan"),
                "motif_pwm_mean_bottom_quartile_signal": float(bot.mean()) if len(bot) else float("nan"),
                "bed6_path": str(bed_path.relative_to(ROOT)),
            }
        )
        n_pool = min(80, len(signals))
        idx = random.Random(args.seed).sample(range(len(signals)), n_pool)
        for j in idx:
            pooled_sig.append(signals[j])
            pooled_mot.append(motifs[j])
        if len(top) and len(bot):
            q_top_mot.append(top.mean())
            q_bot_mot.append(bot.mean())

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(by_encsr)}", flush=True)

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "clipper_peak_75nt_motif_panel.tsv", sep="\t", index=False)

    s = df["spearman_signalValue_vs_motif_pwm"].dropna() if len(df) else pd.Series(dtype=float)
    summary = {
        "workflow": "CLIPper IDR peaks -> 75nt summit-centered BED6 (col5=signalValue) -> Ensembl RNA -> HOMER motif1 PWM",
        "window_bp": WIN_LEN,
        "n_encode_experiments": len(by_encsr),
        "n_with_bed6_and_motif_qc": len(df),
        "n_clipper_bed6_files": len(list(BED_CLIPPER.glob("*.bed6"))),
        "spearman_signalValue_vs_motif_per_experiment": {
            "median": float(s.median()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
            "pct_gt_0.3": float((s > 0.3).mean()) if len(s) else None,
        }
        if len(s)
        else None,
        "pooled_peaks_sequenced": len(pooled_sig),
        "pooled_spearman": float(spearmanr(pooled_sig, pooled_mot).statistic)
        if len(pooled_sig) >= 100
        else None,
        "motif_pwm_mean_top_vs_bottom_quartile_signal": {
            "mean_top_q": float(np.mean(q_top_mot)) if q_top_mot else None,
            "mean_bottom_q": float(np.mean(q_bot_mot)) if q_bot_mot else None,
        },
        "paths": {
            "clipper_bed6_dir": str(BED_CLIPPER.relative_to(ROOT)),
            "skipper_bed6_dir": str(BED_SKIPPER.relative_to(ROOT)) if args.export_skipper_bed6 else None,
            "panel_tsv": "results/skipper_eclip/clipper_peak_75nt_motif_panel.tsv",
        },
        "prior_analyses": {
            "fixlen_151_bed_col5_vs_l2or": "results/skipper_eclip/enrichment_homer_motif_qc.json",
            "clipper_vs_skipper_l2or": "results/skipper_eclip/clipper_vs_skipper_summary.json",
        },
    }
    if args.export_skipper_bed6:
        BED_SKIPPER.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        n_exported = 0
        for eid in exps:
            m = re.match(r"^(.+)_([^_]+)_", eid)
            if not m:
                continue
            sk = f"{m.group(1)}_{m.group(2)}"
            if sk in seen:
                continue
            seen.add(sk)
            n = export_skipper_bed6(sk, m.group(2), BED_SKIPPER / f"{sk}.enrichment_l2or.bed6")
            if n:
                n_exported += 1
                print(f"  Skipper BED6 {sk}: {n} lines", flush=True)
        summary["paths"]["skipper_bed6_dir"] = str(BED_SKIPPER.relative_to(ROOT))
        summary["n_skipper_bed6_exported"] = n_exported

    (OUT / "clipper_peak_pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    if len(pooled_sig) >= 100:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        x, y = np.array(pooled_sig), np.array(pooled_mot)
        rho = spearmanr(x, y).statistic
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
        ax.hexbin(x, y, gridsize=45, cmap="magma", mincnt=1, bins="log")
        ax.set_xlabel("CLIPper IDR signalValue (log2 enrichment)")
        ax.set_ylabel("HOMER motif1 PWM log-score")
        ax.set_title(f"75 nt summit-centered peaks (pooled n={len(x):,})\nSpearman ρ = {rho:.3f}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "clipper_peak_75nt_signal_vs_motif.png", bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
