#!/usr/bin/env python3
"""
50_export_skipper_rew_bed10.py
------------------------------
Export Skipper Figshare ``reproducible_enriched_windows`` as 10-column BED files,
analogous to ENCODE CLIPper IDR narrowPeak (cols 7–8 = signal / -log10 p).

Column mapping (Skipper REW TSV → BED):
  1–3  chr, start, end
  4    name  (default: {SYMBOL}_{CELL}_Skipper_REW)
  5    enrichment_l2or_mean  (CLIPper uses 1000 here; we keep l2or per Skipper)
  6    strand
  7    enrichment_l2or_mean  (narrowPeak signalValue analogue)
  8    -log10(p_min)         (narrowPeak pValue analogue; cap 400)
  9    -log10(q_min)         (narrowPeak qValue analogue; cap 400)
  10   -1                    (no summit offset for these windows)

Outputs:
  results/skipper_eclip/bed10/skipper_rew/{SYMBOL}_{CELL}_Skipper_REW.bed
  results/skipper_eclip/bed10/skipper_rew_export_summary.json
  results/skipper_eclip/bed10/skipper_rew_bed10.zip  (unless --no-zip)

Usage
-----
  python scripts/50_export_skipper_rew_bed10.py
  python scripts/50_export_skipper_rew_bed10.py --sample PTBP1_K562
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_K562 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.K562.tar"
FIG_HEPG2 = ROOT / "data/raw/eclip_skipper/archives/figshare/ENCODE_reproducible_enriched_windows.HepG2.tar"
OUT_DIR = ROOT / "results/skipper_eclip/bed10/skipper_rew"
SUMMARY = ROOT / "results/skipper_eclip/bed10/skipper_rew_export_summary.json"
ZIP_PATH = ROOT / "results/skipper_eclip/bed10/skipper_rew_bed10.zip"

REQUIRED = [
    "chr",
    "start",
    "end",
    "strand",
    "enrichment_l2or_mean",
    "p_min",
    "q_min",
]
NEG_LOG_CAP = 400.0


def neg_log10(x: float) -> float:
    if x <= 0 or not math.isfinite(x):
        return -1.0
    return min(NEG_LOG_CAP, -math.log10(x))


def parse_sample_key(member: str) -> str | None:
    m = re.search(r"/([^/]+)\.reproducible_enriched_windows\.tsv\.gz$", member)
    return m.group(1) if m else None


def tsv_to_bed10(gz_bytes: bytes, bed_name: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(io.BytesIO(gz_bytes), "rt") as f, out_path.open("w") as out:
        header = f.readline().rstrip("\n").split("\t")
        if "enrichment_l2or_mean" not in header:
            return 0
        ix = {c: header.index(c) for c in REQUIRED if c in header}
        if len(ix) < len(REQUIRED):
            return 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            try:
                chrom = p[ix["chr"]]
                start = int(p[ix["start"]])
                end = int(p[ix["end"]])
                strand = p[ix["strand"]] if p[ix["strand"]] in ("+", "-") else "."
                l2or = float(p[ix["enrichment_l2or_mean"]])
                p_min = float(p[ix["p_min"]])
                q_min = float(p[ix["q_min"]])
            except (ValueError, KeyError, IndexError):
                continue
            pval = neg_log10(p_min)
            qval = neg_log10(q_min)
            out.write(
                f"{chrom}\t{start}\t{end}\t{bed_name}\t{l2or}\t{strand}\t"
                f"{l2or}\t{pval}\t{qval}\t-1\n"
            )
            n += 1
    return n


def export_tar(tar_path: Path, only_sample: str | None) -> list[dict]:
    rows: list[dict] = []
    if not tar_path.is_file():
        raise SystemExit(f"Missing archive: {tar_path}")
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".reproducible_enriched_windows.tsv.gz"):
                continue
            sample_key = parse_sample_key(member.name)
            if not sample_key:
                continue
            if only_sample and sample_key != only_sample:
                continue
            raw = tf.extractfile(member)
            if raw is None:
                continue
            gz_bytes = raw.read()
            bed_name = f"{sample_key}_Skipper_REW"
            out_path = OUT_DIR / f"{sample_key}_Skipper_REW.bed"
            n = tsv_to_bed10(gz_bytes, bed_name, out_path)
            if n == 0:
                rows.append({"sample_key": sample_key, "status": "skipped", "reason": "bad_header_or_empty"})
                continue
            rows.append(
                {
                    "sample_key": sample_key,
                    "status": "ok",
                    "n_peaks": n,
                    "bed_path": str(out_path.relative_to(ROOT)),
                }
            )
            print(f"  {sample_key}: {n:,} peaks → {out_path.name}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=str, default="", help="Export one SYMBOL_CELL only")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    only = args.sample.strip() or None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for tar_path in (FIG_K562, FIG_HEPG2):
        print(f"Reading {tar_path.name} …", flush=True)
        rows.extend(export_tar(tar_path, only))

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "script": "50_export_skipper_rew_bed10.py",
        "format": "BED10 (CLIPper narrowPeak-style)",
        "column_map": {
            "5": "enrichment_l2or_mean",
            "7": "enrichment_l2or_mean (signalValue)",
            "8": "-log10(p_min)",
            "9": "-log10(q_min)",
            "10": "-1 (no summit)",
        },
        "n_exported": len(ok),
        "n_skipped": len(rows) - len(ok),
        "total_peaks": int(sum(r["n_peaks"] for r in ok)),
        "out_dir": str(OUT_DIR.relative_to(ROOT)),
        "files": ok,
        "skipped": [r for r in rows if r.get("status") != "ok"],
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))

    if not args.no_zip and not only and ok:
        zip_base = ZIP_PATH.with_suffix("")
        if ZIP_PATH.is_file():
            ZIP_PATH.unlink()
        shutil.make_archive(str(zip_base), "zip", root_dir=OUT_DIR)
        summary["zip_path"] = str(ZIP_PATH.relative_to(ROOT))
        SUMMARY.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {ZIP_PATH}", flush=True)

    print(json.dumps({"n_exported": len(ok), "total_peaks": summary["total_peaks"]}, indent=2))


if __name__ == "__main__":
    main()
