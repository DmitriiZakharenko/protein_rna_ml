#!/usr/bin/env python3
"""
Download and normalize non-CLIP / non-SELEX assay tables into project TSVs.

Outputs under data/sanitized/new_assays/:
  - utexas_aptamer_rna.tsv
  - hughes_invivo_pulldown.tsv
  - prbab_affinity.tsv
  - hurpa_chimeric_gene_pairs.tsv  (from GEO chimericReads; not full PRIMseqTools stats)

Raw files: data/raw/new_assays/

Usage:
    python scripts/42b_download_normalize_new_assays.py
    python scripts/42b_download_normalize_new_assays.py --download-hurpa --skip-download
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "new_assays"
OUT = ROOT / "data" / "sanitized" / "new_assays"
TRAIN_MAP = ROOT / "data" / "sanitized" / "generalized_v3a" / "train.tsv"

HURPA_CHIMERIC_URLS = {
    "HEK_1": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8332nnn/GSM8332737/suppl/GSM8332737_HEK_1_chimericReads.csv.gz",
    "HEK_2": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8332nnn/GSM8332738/suppl/GSM8332738_HEK_2_chimericReads.csv.gz",
    "HEK_3": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8332nnn/GSM8332739/suppl/GSM8332739_HEK_3_chimericReads.csv.gz",
    "K562_1": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8332nnn/GSM8332740/suppl/GSM8332740_K562_1_chimericReads.csv.gz",
    "K562_2": "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8332nnn/GSM8332741/suppl/GSM8332741_K562_2_chimericReads.csv.gz",
}

DOWNLOADS = {
    "utexas": (
        RAW / "utexas_aptamer" / "UTexas_Aptamer_Database_Sept2023.xlsx",
        "https://zenodo.org/records/8387047/files/UTexas%20Aptamer%20Database%20dataset_Sept2023.xlsx?download=1",
    ),
    "hughes_invivo": (
        RAW / "hughes_invivo_pulldown" / "invivo_data.tar.gz",
        "https://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/invivo_data.tar.gz",
    ),
    "prbab2": (
        RAW / "prbab" / "TableS1_PRBAB2.0.xls",
        "http://www.rnabinding.com/PRdeltaGPred/binding_affinity_data/TableS1.xls",
    ),
    "prbab1": (
        RAW / "prbab" / "PRBAB1.0.xls",
        "http://www.rnabinding.com/PRdeltaGPred/binding_affinity_data/Tablesortedbycomplex1.xls",
    ),
}


def _gzip_ok(path: Path) -> bool:
    import gzip

    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except OSError:
        return False


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if dest.suffix == ".gz" and not _gzip_ok(dest):
            print(f"  re-download (corrupt gzip): {dest.name}")
            dest.unlink()
        else:
            print(f"  skip (exists): {dest.name}")
            return
    print(f"  download {url} -> {dest}")
    with urlopen(url, timeout=600) as resp:
        data = resp.read()
    dest.write_bytes(data)
    if dest.suffix == ".gz" and not _gzip_ok(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file failed gzip check: {dest.name}")


def load_protein_sequence_map() -> dict[str, str]:
    if not TRAIN_MAP.exists():
        return {}
    usecols = ["protein_name", "protein_sequence"]
    df = pd.read_csv(TRAIN_MAP, sep="\t", usecols=usecols, low_memory=False)
    df = df.dropna(subset=["protein_name", "protein_sequence"])
    return df.drop_duplicates("protein_name").set_index("protein_name")["protein_sequence"].to_dict()


def parse_kd_to_nm(text: object) -> float | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).strip()
    if not s or s.lower() in {"nan", "na", "n/a", "-"}:
        return None
    m = re.search(
        r"([\d.]+)\s*(?:±|\+/-|\+\-)?\s*[\d.]*\s*(fM|pM|nM|uM|µM|μM|mM|M)\b",
        s,
        re.I,
    )
    if not m:
        m = re.search(r"([\d.]+)\s*e\s*([+-]?\d+)", s, re.I)
        if m:
            val = float(m.group(1)) * (10 ** int(m.group(2)))
            return val * 1e9  # assume mol/L -> nM
        return None
    val = float(m.group(1))
    unit = m.group(2).lower().replace("µ", "u").replace("μ", "u")
    scale = {"fm": 1e-6, "pm": 1e-3, "nm": 1.0, "um": 1e3, "mm": 1e6, "m": 1e9}
    return val * scale.get(unit, 1.0)


def normalize_utexas(protein_map: dict[str, str]) -> pd.DataFrame:
    path = RAW / "utexas_aptamer" / "UTexas_Aptamer_Database_Sept2023.xlsx"
    df = pd.read_excel(path, sheet_name="Dataset")
    na_col = "Type of Nucleic Acid"
    rna_mask = (
        df[na_col]
        .astype(str)
        .str.contains(r"RNA", case=False, na=False)
        & ~df[na_col].astype(str).str.contains("DNA", case=False, na=False)
    )
    df = df[rna_mask].copy()

    def _clean_aptamer_seq(s: object) -> str | None:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return None
        t = str(s).upper().replace("T", "U")
        t = re.sub(r"^5['′]?", "", t)
        t = re.sub(r"3['′]?$", "", t)
        t = re.sub(r"[^ACGU]", "", t)
        return t if len(t) >= 5 else None

    df["rna_sequence"] = df["Aptamer Sequence"].map(_clean_aptamer_seq)
    df = df[df["rna_sequence"].notna()]
    df["protein_name"] = df["Target "].astype(str).str.strip()
    df["affinity_kd_nM"] = df["Affinity"].map(parse_kd_to_nm)
    df["binding_label"] = 1
    df["dataset_source"] = "utexas_aptamer"
    df["experiment_id"] = df.get("Name of Aptamer", pd.Series(dtype=str)).astype(str)
    df["organism"] = "Homo sapiens"
    df["protein_sequence"] = df["protein_name"].map(protein_map)
    return _project_frame(df)


def normalize_hughes_invivo(protein_map: dict[str, str]) -> pd.DataFrame:
    base = RAW / "hughes_invivo_pulldown" / "invivo_data"
    if not base.is_dir():
        tar = RAW / "hughes_invivo_pulldown" / "invivo_data.tar.gz"
        with tarfile.open(tar, "r:gz") as tf:
            tf.extractall(RAW / "hughes_invivo_pulldown")
    rows = []
    for fp in sorted(base.glob("*.txt")):
        protein = fp.stem.split("_")[0].split(".")[0]
        if protein.lower() == "table":
            continue
        with open(fp) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                label_raw, seq = parts[0], parts[1].upper().replace("T", "U")
                label_digits = re.sub(r"[^0-9]", "", label_raw)
                if not label_digits:
                    continue
                label = int(label_digits[-1])
                if not re.fullmatch(r"[ACGU]+", seq):
                    continue
                rows.append(
                    {
                        "protein_name": protein,
                        "rna_sequence": seq,
                        "binding_label": label,
                        "dataset_source": "hughes_invivo_pulldown",
                        "experiment_id": fp.name,
                        "organism": None,
                    }
                )
    df = pd.DataFrame(rows)
    df["protein_sequence"] = df["protein_name"].map(protein_map)
    return _project_frame(df)


def _fetch_pdb_fasta(pdb_id: str) -> dict[str, str]:
    base = pdb_id.strip().split("_")[0][:4].upper()
    url = f"https://files.rcsb.org/download/{base}.fasta"
    with urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    chains: dict[str, str] = {}
    header = None
    seq_parts: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                chains[header] = "".join(seq_parts)
            header = line
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if header is not None:
        chains[header] = "".join(seq_parts)
    return chains


def _pick_chain_sequences(pdb_id: str, fasta: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (protein_seq, rna_seq) heuristically from PDB id like 1HQ1_A:B."""
    prot_chains: list[str] = []
    rna_chains: list[str] = []
    for hdr, seq in fasta.items():
        seq_u = seq.upper().replace("T", "U")
        if re.fullmatch(r"[ACGU]+", seq_u) and len(seq_u) >= 5:
            rna_chains.append(seq_u)
        elif re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
            prot_chains.append(seq)
    m = re.search(r"_([A-Za-z0-9:]+)$", pdb_id.replace(" ", ""))
    if m:
        spec = m.group(1)
        # Often protein:RNA chain ids in table; fall back to first of each type.
        pass
    return (prot_chains[0] if prot_chains else None, rna_chains[0] if rna_chains else None)


def normalize_prbab(fetch_pdb: bool, protein_map: dict[str, str]) -> pd.DataFrame:
    frames = []
    p2 = RAW / "prbab" / "TableS1_PRBAB2.0.xls"
    p1 = RAW / "prbab" / "PRBAB1.0.xls"
    if p2.exists():
        frames.append(pd.read_excel(p2, header=1))
    if p1.exists():
        frames.append(pd.read_excel(p1))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "Protein name": "protein_name",
            "RNA name": "rna_name",
            "PDB ID": "pdb_id",
            "Kd(mol)": "affinity_kd_M",
        }
    )
    df["affinity_kd_nM"] = pd.to_numeric(df["affinity_kd_M"], errors="coerce") * 1e9
    df["binding_label"] = (df["affinity_kd_nM"] <= 1000.0).astype(int)
    df["dataset_source"] = "prbab_affinity"
    df["experiment_id"] = df["pdb_id"]
    df["organism"] = None
    df["rna_sequence"] = None
    df["protein_sequence"] = None
    if fetch_pdb:
        cache: dict[str, tuple[str | None, str | None]] = {}
        for i, row in df.iterrows():
            pid = str(row["pdb_id"])
            if pid not in cache:
                try:
                    cache[pid] = _pick_chain_sequences(pid, _fetch_pdb_fasta(pid))
                except Exception as exc:
                    print(f"  PDB fetch failed {pid}: {exc}")
                    cache[pid] = (None, None)
            ps, rs = cache[pid]
            df.at[i, "protein_sequence"] = ps
            df.at[i, "rna_sequence"] = rs
    df["protein_sequence"] = df["protein_sequence"].fillna(df["protein_name"].map(protein_map))
    return _project_frame(df, extra_cols=["pdb_id", "rna_name", "affinity_kd_nM"])


CHIMERIC_COLS = [
    "readId",
    "R1Tx",
    "R1start",
    "R1end",
    "R1Gene",
    "R1Cigar",
    "R1Strand",
    "R2Tx",
    "R2start",
    "R2end",
    "R2Gene",
    "R2Cigar",
    "R2Strand",
    "R1GeneType",
    "R2GeneType",
]


def normalize_hurpa_chimeric() -> pd.DataFrame:
    hurpa_dir = RAW / "hurpa"
    counts: dict[tuple[str, str], int] = defaultdict(int)
    libs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for gz in sorted(hurpa_dir.glob("*_chimericReads.csv.gz")):
        lib = gz.name.split("_chimeric")[0]
        print(f"  aggregate {gz.name}")
        for chunk in pd.read_csv(
            gz,
            skiprows=1,
            names=CHIMERIC_COLS,
            chunksize=500_000,
            low_memory=False,
        ):
            sub = chunk[["R1Gene", "R2Gene"]].dropna()
            for g1, g2 in zip(sub["R1Gene"], sub["R2Gene"]):
                if g1 == g2:
                    continue
                key = (str(g1), str(g2))
                counts[key] += 1
                libs[key].add(lib)
    rows = []
    for (g1, g2), n in counts.items():
        rows.append(
            {
                "protein_name": g1,
                "rna_gene": g2,
                "rna_sequence": None,
                "binding_label": 1,
                "dataset_source": "prim_chimeric_gene_pair",
                "experiment_id": ",".join(sorted(libs[(g1, g2)])),
                "organism": "Homo sapiens",
                "n_chimeric_reads": n,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("n_chimeric_reads", ascending=False)
    return _project_frame(df, extra_cols=["rna_gene", "n_chimeric_reads"])


def _project_frame(df: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    cols = [
        "rna_sequence",
        "protein_sequence",
        "protein_name",
        "binding_label",
        "dataset_source",
        "experiment_id",
        "organism",
    ]
    for c in extra_cols or []:
        if c not in cols:
            cols.append(c)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def write_summary(outputs: dict[str, pd.DataFrame]) -> None:
    summary = {}
    for name, df in outputs.items():
        summary[name] = {
            "rows": int(len(df)),
            "proteins": int(df["protein_name"].nunique()) if "protein_name" in df.columns else 0,
            "with_protein_sequence": int(df["protein_sequence"].notna().sum()),
            "with_rna_sequence": int(
                df["rna_sequence"].astype(str).str.match(r"^[ACGU]{5,}$", na=False).sum()
            ),
            "positive_rate": float(df["binding_label"].mean()) if len(df) else None,
        }
    path = OUT / "normalization_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--download-hurpa",
        action="store_true",
        help="Download all 5 GEO chimericReads.gz files (~1 GB each).",
    )
    parser.add_argument(
        "--fetch-pdb",
        action="store_true",
        help="Fetch protein/RNA sequences from RCSB for PRBAB PDB ids.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("=== Downloads (tables) ===")
        for _, (dest, url) in DOWNLOADS.items():
            _download(url, dest)
    if args.download_hurpa:
        print("=== Downloads (PRIM chimericReads) ===")
        for _label, url in HURPA_CHIMERIC_URLS.items():
            fname = url.rsplit("/", 1)[-1]
            _download(url, RAW / "hurpa" / fname)

    protein_map = load_protein_sequence_map()
    print(f"Protein sequence map from v3a train: {len(protein_map)} genes")

    outputs: dict[str, pd.DataFrame] = {}

    print("\n=== Normalize UTexas aptamer ===")
    outputs["utexas_aptamer_rna"] = normalize_utexas(protein_map)
    outputs["utexas_aptamer_rna"].to_csv(OUT / "utexas_aptamer_rna.tsv", sep="\t", index=False)
    print(f"  {len(outputs['utexas_aptamer_rna'])} rows")

    print("\n=== Normalize Hughes in vivo pulldown ===")
    outputs["hughes_invivo_pulldown"] = normalize_hughes_invivo(protein_map)
    outputs["hughes_invivo_pulldown"].to_csv(OUT / "hughes_invivo_pulldown.tsv", sep="\t", index=False)
    print(f"  {len(outputs['hughes_invivo_pulldown'])} rows")

    print("\n=== Normalize PRBAB affinity ===")
    outputs["prbab_affinity"] = normalize_prbab(args.fetch_pdb, protein_map)
    outputs["prbab_affinity"].to_csv(OUT / "prbab_affinity.tsv", sep="\t", index=False)
    print(f"  {len(outputs['prbab_affinity'])} rows")

    hurpa_gzs = list((RAW / "hurpa").glob("*_chimericReads.csv.gz"))
    if hurpa_gzs:
        print("\n=== Aggregate PRIM chimeric reads (gene pairs) ===")
        outputs["hurpa_chimeric_gene_pairs"] = normalize_hurpa_chimeric()
        outputs["hurpa_chimeric_gene_pairs"].to_csv(
            OUT / "hurpa_chimeric_gene_pairs.tsv", sep="\t", index=False
        )
        qc = outputs["hurpa_chimeric_gene_pairs"][
            outputs["hurpa_chimeric_gene_pairs"]["n_chimeric_reads"] >= 4
        ]
        qc.to_csv(OUT / "hurpa_chimeric_gene_pairs_min4reads.tsv", sep="\t", index=False)
        print(
            f"  {len(outputs['hurpa_chimeric_gene_pairs'])} unique gene pairs "
            f"({len(qc)} with >=4 reads)"
        )
    else:
        print("\n=== HuRPA: no chimericReads in data/raw/new_assays/hurpa/ (use --download-hurpa) ===")

    manifest = {
        "generated_by": "scripts/42b_download_normalize_new_assays.py",
        "outputs": {k: str(OUT / f"{k}.tsv") for k in outputs},
        "pronab_status": "not_bulk_downloadable; request via https://web.iitm.ac.in/bioinfo2/pronab/",
    }
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    write_summary(outputs)


if __name__ == "__main__":
    main()
