"""
RNAcompete training-subset helpers for Phase 3A.

Training policy:
  - rnacompete_eukarya  — full panel
  - rnacompete_rbpzoo   — full panel
  - rnacompete_ucrbp    — only the 23 reproducible RBPs (Ray & Laverty 2023)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_BENCHMARK_DIR = "data/benchmarks/rnacompete"
DEFAULT_UCRBP_WHITELIST = "configs/ucrbp_23_reproducible.txt"

SUBSET_FILES = {
    "eukarya": "rnacompete_eukarya.tsv",
    "rbpzoo": "rnacompete_rbpzoo.tsv",
    "ucrbp": "rnacompete_ucrbp.tsv",
}

PROJECT_SCHEMA = [
    "rna_sequence", "protein_sequence", "protein_name",
    "binding_label", "experiment_id", "organism",
    "probe_id", "probe_set", "probe_intensity", "dataset_source",
]


def load_ucrbp_whitelist(path: str | None = None) -> set[str]:
    """Load ucRBP protein names from a whitelist file (comments and blanks skipped)."""
    path = path or DEFAULT_UCRBP_WHITELIST
    if not os.path.exists(path):
        raise FileNotFoundError(f"ucRBP whitelist not found: {path}")
    names: set[str] = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(line)
    return names


def load_rnacompete_training_subset(
    benchmark_dir: str = DEFAULT_BENCHMARK_DIR,
    ucrbp_whitelist: str | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Build the Phase 3A RNAcompete training pool:
      eukarya (full) + rbpzoo (full) + ucrbp (23 reproducible RBPs only).
    """
    whitelist = load_ucrbp_whitelist(ucrbp_whitelist)
    parts: list[pd.DataFrame] = []

    for key, fname in SUBSET_FILES.items():
        path = os.path.join(benchmark_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing RNAcompete subset: {path}")

        print(f"  Loading {key}: {path}")
        if key == "ucrbp":
            chunks = []
            n_before = 0
            for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=False):
                n_before += len(chunk)
                sub = chunk[chunk["protein_name"].isin(whitelist)]
                if len(sub):
                    chunks.append(sub)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            n_prot = df["protein_name"].nunique() if len(df) else 0
            print(f"    ucRBP filter: {n_before:,} rows → {len(df):,} rows "
                  f"({n_prot} / {len(whitelist)} whitelist proteins)")
            missing = whitelist - set(df["protein_name"].unique()) if len(df) else whitelist
            if missing:
                print(f"    [WARN] Whitelist proteins not found in ucRBP file: "
                      f"{sorted(missing)}")
        else:
            df = pd.read_csv(path, sep="\t", low_memory=False)
            print(f"    {len(df):,} rows, {df['protein_name'].nunique()} proteins")

        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    for col in PROJECT_SCHEMA:
        if col not in combined.columns:
            combined[col] = None
    combined = combined[PROJECT_SCHEMA].copy()
    combined["binding_label"] = combined["binding_label"].astype(int)
    combined["rna_sequence"] = (
        combined["rna_sequence"].str.upper().str.replace("T", "U", regex=False)
    )
    if "organism" in combined.columns:
        combined["organism"] = combined["organism"].str.replace("_", " ", regex=False)

    return combined


def save_training_subset(
    df: pd.DataFrame,
    out_path: str,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Save training TSV; optionally deduplicate on (rna_sequence, protein_name)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=["rna_sequence", "protein_name"])
        print(f"  Dedup: {before:,} → {len(df):,} ({before - len(df):,} removed)")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved {out_path} ({len(df):,} rows, {df['protein_name'].nunique()} proteins)")
    return df
