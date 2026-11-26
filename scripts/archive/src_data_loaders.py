"""
src/data/loaders.py
Unified dataset loaders for all data sources.

Each loader returns a pandas DataFrame conforming to the PROJECT SCHEMA:
    rna_sequence      str   — RNA sequence (U-alphabet)
    protein_sequence  str   — protein amino acid sequence
    protein_name      str   — gene / RBP name
    binding_label     int   — 0 or 1
    dataset_source    str   — provenance tag
    experiment_id     str   — unique experiment identifier (optional)
    organism          str   — species (optional)

Usage:
    from src.data.loaders import load_dataset, load_rnacompete_benchmark

    df_train = load_dataset("data/generalized_v2/train.tsv")
    df_bench = load_rnacompete_benchmark("data/benchmarks/rnacompete/rnacompete_all.tsv")
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Project schema ─────────────────────────────────────────────────────────

PROJECT_COLUMNS = [
    "rna_sequence",
    "protein_sequence",
    "protein_name",
    "binding_label",
    "dataset_source",
    "experiment_id",
    "organism",
]

# Column aliases: maps common alternative names to canonical names
COLUMN_ALIASES = {
    # protein
    "target_name":      "protein_name",
    "gene_name":        "protein_name",
    "rbp_name":         "protein_name",
    "protein":          "protein_name",
    # RNA
    "rna":              "rna_sequence",
    "sequence":         "rna_sequence",
    "probe_sequence":   "rna_sequence",
    # label
    "label":            "binding_label",
    "class":            "binding_label",
    # dataset
    "dataset":          "dataset_source",
    "source":           "dataset_source",
    "dataset_label":    "dataset_source",
    # experiment
    "hyb_id":           "experiment_id",
    "exp_id":           "experiment_id",
    "rncmpt_id":        "experiment_id",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column alias mapping and canonicalise RNA alphabet."""
    rename = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=rename)
    if "rna_sequence" in df.columns:
        df["rna_sequence"] = df["rna_sequence"].str.upper().str.replace("T", "U", regex=False)
    if "binding_label" in df.columns:
        df["binding_label"] = df["binding_label"].astype(int)
    return df


def _ensure_schema(df: pd.DataFrame, source_tag: Optional[str] = None) -> pd.DataFrame:
    """Add missing schema columns as None; apply source_tag if given."""
    for col in PROJECT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if source_tag and "dataset_source" in df.columns:
        df["dataset_source"] = df["dataset_source"].fillna(source_tag)
    elif source_tag:
        df["dataset_source"] = source_tag
    return df[PROJECT_COLUMNS + [c for c in df.columns if c not in PROJECT_COLUMNS]]


# ── Generic loader ──────────────────────────────────────────────────────────

def load_dataset(
    path: str,
    source_tag: Optional[str] = None,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a project-format TSV file (gzipped or plain).

    Parameters
    ----------
    path        : Path to TSV file
    source_tag  : Override / fill missing dataset_source column
    max_rows    : Subsample to at most this many rows (stratified by binding_label)
    seed        : Random seed for subsampling

    Returns
    -------
    pd.DataFrame conforming to PROJECT_COLUMNS
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    compression = "gzip" if path.endswith(".gz") else "infer"
    df = pd.read_csv(path, sep="\t", compression=compression, low_memory=False)
    df = _normalise_columns(df)
    df = _ensure_schema(df, source_tag or Path(path).stem)

    if max_rows and len(df) > max_rows:
        df = _stratified_sample(df, max_rows, seed)

    return df


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample preserving binding_label distribution."""
    rng = np.random.default_rng(seed)
    chunks = []
    for label, grp in df.groupby("binding_label"):
        k = max(1, round(n * len(grp) / len(df)))
        idx = rng.choice(len(grp), min(k, len(grp)), replace=False)
        chunks.append(grp.iloc[idx])
    result = pd.concat(chunks, ignore_index=True)
    # Final trim to exactly n if rounding produced more
    if len(result) > n:
        result = result.sample(n, random_state=seed)
    return result


# ── RNAcompete benchmark loader ─────────────────────────────────────────────

def load_rnacompete_benchmark(
    path: str,
    subset: Optional[str] = None,
    min_pos_per_protein: int = 10,
    min_neg_per_protein: int = 10,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a RNAcompete benchmark file (prepared by 17_prepare_rnacompete_benchmark.py).

    This data is BENCHMARK ONLY — do not merge into training.

    Parameters
    ----------
    path                 : Path to rnacompete_*.tsv benchmark file
    subset               : 'human', 'nonhuman', 'rbpzoo', or None for all
    min_pos_per_protein  : Drop proteins with fewer positive examples
    min_neg_per_protein  : Drop proteins with fewer negative examples
    max_rows             : Optional subsample cap

    Returns
    -------
    pd.DataFrame with valid proteins only (min_pos_per_protein / min_neg_per_protein met)
    """
    df = load_dataset(path, source_tag=None, max_rows=None, seed=seed)

    # Subset by organism
    if subset == "human" and "organism" in df.columns:
        df = df[df["organism"].isin({"Homo sapiens", "Mus musculus"})].copy()
    elif subset == "nonhuman" and "organism" in df.columns:
        df = df[~df["organism"].isin({"Homo sapiens", "Mus musculus"})].copy()

    # Filter proteins with insufficient pos/neg coverage
    if "protein_name" in df.columns:
        valid_proteins = []
        for prot, grp in df.groupby("protein_name"):
            n_pos = (grp["binding_label"] == 1).sum()
            n_neg = (grp["binding_label"] == 0).sum()
            if n_pos >= min_pos_per_protein and n_neg >= min_neg_per_protein:
                valid_proteins.append(prot)
        before = len(df)
        df = df[df["protein_name"].isin(valid_proteins)].copy()
        n_dropped_proteins = df["protein_name"].nunique()
        if before > len(df):
            warnings.warn(
                f"Dropped {before - len(df):,} rows from proteins with "
                f"<{min_pos_per_protein} pos or <{min_neg_per_protein} neg. "
                f"{df['protein_name'].nunique()} proteins retained.",
                stacklevel=2,
            )

    if max_rows and len(df) > max_rows:
        df = _stratified_sample(df, max_rows, seed)

    return df


# ── Affinity dataset loader ─────────────────────────────────────────────────

def load_affinity_dataset(
    path: str,
    kd_column: str = "Kd_nM",
    binding_threshold_nM: float = 1000.0,
    source_tag: str = "affinity",
) -> pd.DataFrame:
    """
    Load a protein-RNA affinity dataset (Kd values in nM).
    Converts to binary labels: Kd <= threshold → positive, else negative.
    Continuous Kd retained as 'affinity_kd_nM' for regression tasks.

    Parameters
    ----------
    path                 : Path to TSV / Excel file
    kd_column            : Column name with Kd values (nM)
    binding_threshold_nM : Kd threshold for binarisation (default 1000 nM = 1 μM)
    source_tag           : dataset_source tag
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep="\t", low_memory=False)

    df = _normalise_columns(df)

    if kd_column not in df.columns:
        raise ValueError(f"Kd column '{kd_column}' not found. Available: {df.columns.tolist()}")

    df["affinity_kd_nM"] = pd.to_numeric(df[kd_column], errors="coerce")
    df["binding_label"]  = (df["affinity_kd_nM"] <= binding_threshold_nM).astype(int)
    df["dataset_source"] = source_tag
    df = _ensure_schema(df, source_tag)
    return df


# ── eCLIP / CLIP dataset loader ─────────────────────────────────────────────

def load_clip_dataset(
    path: str,
    source_tag: Optional[str] = None,
    bed_like: bool = False,
) -> pd.DataFrame:
    """
    Load an eCLIP / iCLIP / PAR-CLIP TSV file in project format.
    If bed_like=True, interprets as a BED-format file with columns:
      chrom, start, end, name, score, strand, [sequence]

    Parameters
    ----------
    path      : Path to file
    source_tag: dataset_source tag (inferred from filename if None)
    bed_like  : Parse as BED format
    """
    if not source_tag:
        source_tag = Path(path).stem.replace("_eclip", "").replace("_iclip", "")

    if bed_like:
        cols = ["chrom", "start", "end", "name", "score", "strand"]
        df = pd.read_csv(path, sep="\t", header=None, names=cols, low_memory=False)
        df["binding_label"] = 1  # eCLIP peaks are positive by definition
        df["dataset_source"] = source_tag
        if "sequence" in df.columns:
            df = df.rename(columns={"sequence": "rna_sequence"})
        return _ensure_schema(df, source_tag)

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = _normalise_columns(df)
    df["dataset_source"] = source_tag
    return _ensure_schema(df, source_tag)


# ── Quality validation ───────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame, name: str = "dataset") -> dict:
    """
    Run basic integrity checks on a dataset DataFrame.
    Returns a dict of warnings/stats.

    Checks:
      - missing required columns
      - label distribution
      - per-protein AUROC feasibility (pos AND neg present)
      - sequence length distribution
      - null rates
    """
    report = {"name": name, "n_rows": len(df), "warnings": [], "stats": {}}

    # Required columns
    missing = [c for c in ["rna_sequence", "protein_sequence", "protein_name", "binding_label"]
               if c not in df.columns or df[c].isna().all()]
    if missing:
        report["warnings"].append(f"Missing or all-null columns: {missing}")

    if "binding_label" in df.columns:
        n_pos = int((df["binding_label"] == 1).sum())
        n_neg = int((df["binding_label"] == 0).sum())
        pos_rate = n_pos / len(df) if len(df) > 0 else 0
        report["stats"]["n_pos"] = n_pos
        report["stats"]["n_neg"] = n_neg
        report["stats"]["pos_rate"] = round(pos_rate, 4)
        if n_pos < 10:
            report["warnings"].append(f"Very few positives: {n_pos}")
        if pos_rate > 0.8:
            report["warnings"].append(f"High positive rate: {pos_rate:.2%}")

    if "rna_sequence" in df.columns:
        lens = df["rna_sequence"].str.len()
        report["stats"]["rna_len_median"] = int(lens.median())
        report["stats"]["rna_len_max"]    = int(lens.max())

    if "protein_sequence" in df.columns and not df["protein_sequence"].isna().all():
        lens = df["protein_sequence"].dropna().str.len()
        report["stats"]["prot_len_median"] = int(lens.median())
        report["stats"]["prot_len_max"]    = int(lens.max())

    if "protein_name" in df.columns:
        n_prot = df["protein_name"].nunique()
        report["stats"]["n_proteins"] = n_prot
        if "binding_label" in df.columns:
            single_class = sum(
                1 for _, g in df.groupby("protein_name")
                if g["binding_label"].nunique() < 2
            )
            if single_class > 0:
                report["warnings"].append(
                    f"{single_class} proteins have only one class — AUROC undefined for them")

    null_rates = {c: round(df[c].isna().mean(), 3)
                  for c in df.columns if df[c].isna().any()}
    if null_rates:
        report["stats"]["null_rates"] = null_rates

    return report
