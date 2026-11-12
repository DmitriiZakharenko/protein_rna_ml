#!/usr/bin/env python3
"""
76_build_tf_chipseq_pairs.py
----------------------------
Transcription-factor ChIP-seq peaks (ENCODE, GRCh38, K562 and HepG2, optimal IDR thresholded narrowPeak) as protein-nucleic-acid
pairs with an explicit data type, in the column layout of the generalized TSVs.

Stages (run in this order; every stage skips work that is already on disk):
  manifest   ENCODE API -> results/tf_chipseq/manifest.tsv (one experiment per TF and cell line)
  download   peaks, UniProt sequences (human, reviewed) and the hg38 FASTA -> data/raw/tf_chipseq/
  build      windows around the peak summit (151 nt, random strand) and three kinds of rows -> data/benchmarks/tf_chipseq/pairs.tsv
               label 1                  peak window of the TF
               label 0, neg_type xtf    peak window of another TF of the same cell line that does not overlap any peak of this TF
                                        (same sequence, different protein: the layout of the cross-protein probe)
               label 0, neg_type gc     random genomic window without overlap, GC within +-0.05 of the paired positive

Columns: protein_name, rna_sequence (DNA written with U, as the RNA pipeline expects), protein_sequence, binding_label,
dataset_source (chip_K562 / chip_HepG2), data_type (dna_chip), neg_type, chrom, start, end, strand.

Usage
-----
  python scripts/76_build_tf_chipseq_pairs.py manifest
  python scripts/76_build_tf_chipseq_pairs.py download
  python scripts/76_build_tf_chipseq_pairs.py build --max_peaks 1000 --seed 42
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tf_chipseq"
OUT = ROOT / "results" / "tf_chipseq"
BENCH = ROOT / "data" / "benchmarks" / "tf_chipseq"
ENCODE = "https://www.encodeproject.org"
GENOME_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
CELLS = ["K562", "HepG2"]
TAG = re.compile(r"^(eGFP|GFP|YFP|FLAG|3xFLAG|HA|mCherry)-|-human$")


def get_json(path: str, params: dict) -> dict:
    r = requests.get(ENCODE + path, params=params, headers={"Accept": "application/json"}, timeout=180)
    r.raise_for_status()
    return r.json()


def stage_manifest(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = {"assay_title": "TF ChIP-seq", "assembly": "GRCh38", "status": "released", "limit": "all",
            "biosample_ontology.term_name": CELLS}
    exp = get_json("/search/", {**base, "type": "Experiment", "field": ["accession", "target.label", "target.investigated_as",
                                                                         "biosample_ontology.term_name"]})["@graph"]
    rows = {}
    for x in exp:
        t = x.get("target")
        if not t or "transcription factor" not in t.get("investigated_as", []):
            continue
        rows[x["accession"]] = {"experiment": x["accession"], "tf": TAG.sub("", t["label"]), "cell": x["biosample_ontology"]["term_name"]}
    files = get_json("/search/", {**base, "type": "File", "file_format": "bed", "file_type": "bed narrowPeak",
                                  "output_type": "optimal IDR thresholded peaks", "field": ["accession", "dataset", "file_size"]})["@graph"]
    out = []
    for f in files:
        e = f["dataset"].strip("/").split("/")[-1]
        if e in rows:
            out.append({**rows[e], "file": f["accession"], "file_size": f.get("file_size", 0)})
    df = pd.DataFrame(out).sort_values("file_size", ascending=False).drop_duplicates(["tf", "cell"]).sort_values(["cell", "tf"])
    df.to_csv(OUT / "manifest.tsv", sep="\t", index=False)
    print(f"{len(df)} experiments, {df.tf.nunique()} TFs; per cell: {df.cell.value_counts().to_dict()}")


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    for k in range(retries):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            return
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"  retry {dest.name}: {e}", flush=True)
            time.sleep(5)


def stage_download(args) -> None:
    man = pd.read_csv(OUT / "manifest.tsv", sep="\t")
    (RAW / "peaks").mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(man.file, 1):
        dest = RAW / "peaks" / f"{f}.bed.gz"
        if not dest.exists():
            fetch(f"{ENCODE}/files/{f}/@@download/{f}.bed.gz", dest)
        if i % 100 == 0:
            print(f"peaks {i}/{len(man)}", flush=True)
    uni = OUT / "uniprot.tsv"
    have = pd.read_csv(uni, sep="\t") if uni.exists() else pd.DataFrame(columns=["tf", "accession", "sequence"])
    new = []
    for tf in sorted(set(man.tf) - set(have.tf)):
        r = requests.get("https://rest.uniprot.org/uniprotkb/search", timeout=60,
                         params={"query": f"gene_exact:{tf} AND organism_id:9606 AND reviewed:true", "fields": "accession,gene_primary,sequence",
                                 "format": "tsv", "size": 5})
        lines = r.text.strip().split("\n")[1:] if r.ok else []
        recs = [l.split("\t") for l in lines if l]
        pick = next((x for x in recs if x[1] == tf), recs[0] if recs else None)
        new.append({"tf": tf, "accession": pick[0] if pick else "", "sequence": pick[2] if pick else ""})
        time.sleep(0.1)
    have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
    have.to_csv(uni, sep="\t", index=False)
    print(f"UniProt: {int((have.sequence.fillna('') != '').sum())}/{len(have)} TFs with a sequence")
    fa = RAW / "hg38.fa"
    if not fa.exists():
        gz = RAW / "hg38.fa.gz"
        if not gz.exists():
            fetch(GENOME_URL, gz)
        with gzip.open(gz, "rb") as a, open(fa, "wb") as b:
            shutil.copyfileobj(a, b)
        gz.unlink()
    print("download done")


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


class Peaks:
    """Sorted, merged intervals per chromosome for overlap queries."""
    def __init__(self, df: pd.DataFrame):
        self.iv = {}
        for c, g in df.groupby("chrom"):
            a = g.sort_values("start")
            s, e = a.start.values, a.end.values
            ms, me = [s[0]], [e[0]]
            for x, y in zip(s[1:], e[1:]):
                if x <= me[-1]:
                    me[-1] = max(me[-1], y)
                else:
                    ms.append(x), me.append(y)
            self.iv[c] = (np.array(ms), np.array(me))

    def hits(self, chrom: str, start: int, end: int) -> bool:
        if chrom not in self.iv:
            return False
        s, e = self.iv[chrom]
        i = np.searchsorted(s, end)
        return i > 0 and e[i - 1] > start


def gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(1, len(seq))


def stage_build(args) -> None:
    from pyfaidx import Fasta
    rng = np.random.default_rng(args.seed)
    man = pd.read_csv(OUT / "manifest.tsv", sep="\t")
    uni = pd.read_csv(OUT / "uniprot.tsv", sep="\t").fillna("")
    seqs = dict(zip(uni.tf, uni.sequence))
    man = man[man.tf.map(lambda t: 0 < len(seqs.get(t, "")) <= args.max_prot_len)].reset_index(drop=True)
    genome = Fasta(str(RAW / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    chroms = [c for c in genome.keys() if re.fullmatch(r"chr([0-9]+|X)", c)]
    clen = {c: len(genome[c]) for c in chroms}
    half = args.window // 2

    peaks, wins = {}, {}
    for r in man.itertuples():
        df = pd.read_csv(RAW / "peaks" / f"{r.file}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 6, 9],
                         names=["chrom", "start", "end", "signal", "summit"])
        df = df[df.chrom.isin(chroms)].reset_index(drop=True)
        peaks[(r.tf, r.cell)] = Peaks(df)
        if len(df) > args.max_peaks:
            df = df.sample(args.max_peaks, random_state=int(rng.integers(1 << 30)))
        c = df.start.values + df.summit.values
        wins[(r.tf, r.cell)] = pd.DataFrame({"chrom": df.chrom.values, "start": c - half, "end": c - half + args.window})
    print(f"{len(man)} TF/cell experiments with a protein sequence")

    def seq_at(chrom, start, end):
        if start < 0 or end > clen[chrom]:
            return None
        s = genome[chrom][start:end]
        return None if "N" in s or len(s) != end - start else s

    rows = []

    def add(tf, cell, label, neg, chrom, start, end, s, strand=None):
        strand = strand or ("+" if rng.random() < 0.5 else "-")
        s = s if strand == "+" else revcomp(s)
        rows.append((tf, s.replace("T", "U"), seqs[tf], label, f"chip_{cell}", "dna_chip", neg, chrom, start, end, strand))

    for cell in CELLS:
        tfs = [t for t, c in wins if c == cell]
        pool = np.array(tfs)
        for tf in tfs:
            w = wins[(tf, cell)]
            for chrom, st, en in zip(w.chrom, w.start, w.end):
                s = seq_at(chrom, st, en)
                if s is None:
                    continue
                strand = "+" if rng.random() < 0.5 else "-"
                add(tf, cell, 1, "pos", chrom, st, en, s, strand)
                # (1) same window and sequence, protein replaced by another TF of the cell line without a peak there
                for _ in range(20):
                    o = pool[int(rng.integers(len(pool)))]
                    if o != tf and not peaks[(o, cell)].hits(chrom, st, en):
                        add(o, cell, 0, "xtf", chrom, st, en, s, strand)
                        break
                # (2) GC-matched random genomic window of the same TF without peak overlap
                target = gc(s)
                for _ in range(50):
                    c = chroms[int(rng.integers(len(chroms)))]
                    a = int(rng.integers(0, clen[c] - args.window))
                    g = seq_at(c, a, a + args.window)
                    if g is not None and abs(gc(g) - target) <= 0.05 and not peaks[(tf, cell)].hits(c, a, a + args.window):
                        add(tf, cell, 0, "gc", c, a, a + args.window, g)
                        break
        print(f"{cell}: {len(rows):,} rows so far", flush=True)
    cols = ["protein_name", "rna_sequence", "protein_sequence", "binding_label", "dataset_source", "data_type", "neg_type",
            "chrom", "start", "end", "strand"]
    df = pd.DataFrame(rows, columns=cols)
    BENCH.mkdir(parents=True, exist_ok=True)
    df.to_csv(BENCH / "pairs.tsv", sep="\t", index=False)
    print(df.groupby(["neg_type"]).size().to_dict(), "TFs:", df.protein_name.nunique())
    print(f"wrote {BENCH / 'pairs.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["manifest", "download", "build"])
    ap.add_argument("--max_peaks", type=int, default=1000, help="peaks sampled per TF and cell line")
    ap.add_argument("--window", type=int, default=151)
    ap.add_argument("--max_prot_len", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"manifest": stage_manifest, "download": stage_download, "build": stage_build}[args.stage](args)


if __name__ == "__main__":
    main()
