#!/usr/bin/env python3
"""
45_skipper_enrichment_homer_qc.py
---------------------------------
Validate Dropbox fixlen_151 BED column 5 vs Figshare Skipper enrichment_l2or_mean,
and correlate col5 with top HOMER motif PWM scores (positives only).

Outputs:
  results/skipper_eclip/enrichment_l2or_col5_panel.tsv
  results/skipper_eclip/homer_pwm_motif_panel.tsv
  results/skipper_eclip/enrichment_homer_motif_qc.json
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

DROPBOX_TAR = ROOT / "data/raw/eclip_skipper/archives/dropbox/eclip_various_pos_neg_sets.hg38.tar.xz"
FIG_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.K562.tar"
FIG_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.HepG2.tar"
HOMER_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_homer.K562.tgz"
HOMER_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_homer.HepG2.tgz"
INFO = ROOT / "data/raw/eclip_skipper/manifests/eclip_various_pos_neg_sets.hg38.info.tsv"
FASTA_DIR = ROOT / "data/raw/eclip_skipper/extracted/fixlen_151_fasta"
OUT_DIR = ROOT / "results/skipper_eclip"


def load_enriched_intervals(gz_bytes: bytes) -> dict[str, list[tuple[int, int, float]]]:
    intervals: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    with gzip.open(io.BytesIO(gz_bytes), "r") as f:
        header = f.readline().decode().rstrip("\n").split("\t")
        if "enrichment_l2or_mean" not in header:
            raise ValueError(f"missing enrichment_l2or_mean in columns: {header[:8]}...")
        ix = header.index("chr"), header.index("start"), header.index("end")
        i_l2 = header.index("enrichment_l2or_mean")
        for line in f:
            p = line.decode().rstrip("\n").split("\t")
            try:
                s, e = int(p[ix[1]]), int(p[ix[2]])
                l2 = float(p[i_l2])
            except (ValueError, IndexError):
                continue
            intervals[p[ix[0]]].append((s, e, l2))
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


def read_bed_from_tar(tf: tarfile.TarFile, member: str) -> list[tuple[str, int, int, float]]:
    f = tf.extractfile(member)
    if f is None:
        return []
    rows = []
    for line in f.read().decode().splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        if len(p) < 5:
            continue
        try:
            rows.append((p[0], int(p[1]), int(p[2]), float(p[4])))
        except ValueError:
            pass
    return rows


def parse_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    cur, chunks = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                out[cur] = "".join(chunks)
            cur = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip().upper())
    if cur:
        out[cur] = "".join(chunks)
    return out


def parse_homer_motif_bytes(raw: bytes) -> tuple[str, np.ndarray] | None:
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(">"):
        return None
    consensus = lines[0].split(">")[1].split("\t")[0]
    pwm_rows = []
    for ln in lines[1:]:
        if ln.startswith(">"):
            break
        pwm_rows.append([float(x) for x in ln.split("\t")])
    if not pwm_rows:
        return None
    return consensus, np.array(pwm_rows, dtype=float)


def motif_log_score(seq: str, pwm: np.ndarray) -> float:
    """Max window log-score (HOMER ACGT rows; U in RNA treated as T)."""
    seq = seq.upper().replace("U", "T")
    base_to_i = {"A": 0, "C": 1, "G": 2, "T": 3}
    w = pwm.shape[0]
    if len(seq) < w:
        return float("nan")
    eps = 1e-6
    best = -1e18
    for off in range(len(seq) - w + 1):
        s = 0.0
        ok = True
        for j in range(w):
            b = seq[off + j]
            if b not in base_to_i:
                ok = False
                break
            p = max(pwm[j, base_to_i[b]], eps)
            s += np.log(p / 0.25)
        if ok:
            best = max(best, s)
    return best if best > -1e17 else float("nan")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(INFO, sep="\t")
    exps = info.loc[info["pos_bed"].str.contains("fixlen_151", na=False), "eclip_id"].tolist()

    print("Preloading Figshare enriched_windows …", flush=True)
    fig_gz: dict[tuple[str, str], bytes] = {}
    for cell, tar_path in [("K562", FIG_K562), ("HepG2", FIG_HEPG2)]:
        with tarfile.open(tar_path, "r:") as tf:
            for name in tf.getnames():
                m = re.search(
                    r"reproducible_enriched_windows/([^/]+)\.reproducible_enriched_windows\.tsv\.gz$",
                    name,
                )
                if not m:
                    continue
                raw = tf.extractfile(name)
                if raw is None:
                    continue
                fig_gz[(cell, m.group(1))] = raw.read()
    print(f"  loaded {len(fig_gz)} sample tables", flush=True)

    print("Preloading HOMER motif1 …", flush=True)
    homer_motifs: dict[tuple[str, str], tuple[str, np.ndarray]] = {}
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
                if raw is None:
                    continue
                parsed = parse_homer_motif_bytes(raw.read())
                if parsed:
                    homer_motifs[(cell, m.group(1))] = parsed
    print(f"  loaded {len(homer_motifs)} HOMER motif1 PWMs", flush=True)

    l2or_rows = []
    motif_rows = []
    interval_cache: dict[tuple[str, str], dict] = {}
    fig_errors: list[str] = []

    with tarfile.open(DROPBOX_TAR, "r:xz") as drop_tf:
        drop_names = set(drop_tf.getnames())
        try:
            for i, eid in enumerate(exps):
                parts = eid.split("_")
                if len(parts) < 3:
                    continue
                symbol, cell = parts[0], parts[1]
                sample_key = f"{symbol}_{cell}"
                bed_m = f"pos_neg_fasta_out/{eid}.fixlen_151.positives.bed"
                if bed_m not in drop_names:
                    continue
                bed = read_bed_from_tar(drop_tf, bed_m)
                if not bed:
                    continue

                matched = 0
                rho_l2 = float("nan")
                cache_key = (cell, sample_key)
                if cache_key in fig_gz:
                    if cache_key not in interval_cache:
                        try:
                            interval_cache[cache_key] = load_enriched_intervals(fig_gz[cache_key])
                        except Exception as exc:
                            fig_errors.append(f"{sample_key}: {exc}")
                            interval_cache[cache_key] = {}
                    intervals = interval_cache[cache_key]
                    col5, l2 = [], []
                    for chrom, s, e, sc in bed:
                        v = max_overlap_l2or(chrom, s, e, intervals)
                        if v is not None:
                            col5.append(sc)
                            l2.append(v)
                    matched = len(col5)
                    if matched >= 30:
                        rho_l2 = float(spearmanr(col5, l2).statistic)

                l2or_rows.append(
                    {
                        "eclip_id": eid,
                        "symbol": symbol,
                        "cell_line": cell,
                        "n_bed_pos": len(bed),
                        "n_overlap_figshare": matched,
                        "frac_overlap": matched / len(bed) if bed else 0.0,
                        "spearman_col5_vs_enrichment_l2or_mean": rho_l2,
                    }
                )

                fa = FASTA_DIR / f"{eid}.fixlen_151.positives.fa"
                hm_key = (cell, sample_key)
                if hm_key in homer_motifs and fa.is_file():
                    cons, pwm = homer_motifs[hm_key]
                    seqs = parse_fasta(fa)
                    # map CLIP_* headers to bed order via id in renamed scheme: use all seqs
                    scores_col5 = []
                    scores_motif = []
                    bed_by_id = {}
                    fbed = drop_tf.extractfile(bed_m)
                    for line in fbed.read().decode().splitlines():
                        p = line.split("\t")
                        if len(p) >= 5:
                            bed_by_id[p[3]] = float(p[4])
                    for hid, seq in seqs.items():
                        if hid not in bed_by_id:
                            # headers CLIP_n without bed id match — match by order if same count
                            continue
                        ms = motif_log_score(seq, pwm)
                        if np.isfinite(ms):
                            scores_col5.append(bed_by_id[hid])
                            scores_motif.append(ms)
                    if len(scores_col5) < 30:
                        # fallback: zip by sorted ids
                        ids = sorted(set(seqs) & set(bed_by_id))
                        scores_col5, scores_motif = [], []
                        for hid in ids:
                            ms = motif_log_score(seqs[hid], pwm)
                            if np.isfinite(ms):
                                scores_col5.append(bed_by_id[hid])
                                scores_motif.append(ms)
                    rho_m = float("nan")
                    if len(scores_col5) >= 30:
                        rho_m = float(spearmanr(scores_col5, scores_motif).statistic)
                    motif_rows.append(
                        {
                            "eclip_id": eid,
                            "symbol": symbol,
                            "cell_line": cell,
                            "homer_sample": sample_key,
                            "motif1_consensus": cons,
                            "n_scored": len(scores_col5),
                            "spearman_col5_vs_homer_pwm": rho_m,
                        }
                    )

                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(exps)}", flush=True)
        finally:
            pass

    l2df = pd.DataFrame(l2or_rows)
    mdf = pd.DataFrame(motif_rows)
    l2_path = OUT_DIR / "enrichment_l2or_col5_panel.tsv"
    m_path = OUT_DIR / "homer_pwm_motif_panel.tsv"
    l2df.to_csv(l2_path, sep="\t", index=False)
    mdf.to_csv(m_path, sep="\t", index=False)

    def agg_stats(s: pd.Series) -> dict:
        s = s.dropna()
        if s.empty:
            return {"n": 0}
        return {
            "n": int(len(s)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "pct_gt_0.3": float((s > 0.3).mean()),
            "pct_gt_0.5": float((s > 0.5).mean()),
        }

    summary = {
        "n_experiments_fixlen_151": len(exps),
        "enrichment_l2or_vs_bed_col5": {
            "n_with_figshare_overlap": int((l2df["n_overlap_figshare"] > 0).sum()),
            "median_frac_windows_matched": float(l2df["frac_overlap"].median()),
            "spearman_col5_vs_enrichment_l2or_mean": agg_stats(l2df["spearman_col5_vs_enrichment_l2or_mean"]),
            "note": (
                "Join: genomic overlap fixlen_151 BED vs Figshare reproducible_enriched_windows; "
                "compare BED col5 to max overlapping enrichment_l2or_mean. "
                "Figshare TSV column 'score' is zero; use enrichment_l2or_*."
            ),
        },
        "homer_motif1_pwm_vs_col5": {
            "n_experiments_with_homer": int(len(mdf)),
            "spearman_col5_vs_pwm_logscore": agg_stats(mdf["spearman_col5_vs_homer_pwm"]),
        },
        "figshare_load_errors": fig_errors[:20],
        "n_figshare_load_errors": len(fig_errors),
        "paths": {
            "panel_l2or": str(l2_path.relative_to(ROOT)),
            "panel_homer": str(m_path.relative_to(ROOT)),
        },
    }
    out_json = OUT_DIR / "enrichment_homer_motif_qc.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
