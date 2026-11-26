#!/usr/bin/env python3
"""
29_annotate_top_bottom_experiment_ids.py
----------------------------------------
Add experiment_id to all_protocols_summary.tsv and export matching protein FASTA.

Experiment ID sources:
  RNAcompete  hyb_id from clean TSV (best experiment = highest mean positive intensity)
  RBNS        ENCODE experiment_accession from rbns_experiments.tsv (pulldown)
  HTR-SELEX   experiment_accession from fastq_metadata.tsv (mode per protein, non-background)

FASTA headers: >experiment_id,protein_name

Protein sequences are keyed by (experiment_id, protein_name), not protein name alone,
so RNAcompete panels cannot overwrite each other (e.g. fly vs human SRP54).

Usage:
    python scripts/29_annotate_top_bottom_experiment_ids.py

    python scripts/29_annotate_top_bottom_experiment_ids.py \\
        --summary_tsv results/top_bottom_examples/all_protocols_summary.tsv \\
        --out_fasta results/top_bottom_examples/all_protocols_proteins.fasta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RNAcompete_DATASETS = {
    "RNAcompete_Eukarya": "eukarya/results/ml_dataset_eukarya_clean.tsv.gz",
    "RNAcompete_RBPZoo": "rbpzoo/results/ml_dataset_rbpzoo_clean.tsv.gz",
    "RNAcompete_ucRBP23": "ucrbp/results/ml_dataset_ucrbp_clean.tsv.gz",
}
RNAcompete_KMER_DIRS = {
    "RNAcompete_Eukarya": "eukarya/data/kmers",
    "RNAcompete_ucRBP23": "ucrbp/data/kmers",
}


def _mean_top_kmer_score(kmer_dir: Path | None, exp_id: str, top_k: int = 10, min_z: float = 2.0) -> float:
    if kmer_dir is None:
        return -1.0
    path = kmer_dir / f"{exp_id}_zscores.tsv"
    if not path.exists():
        return -1.0
    tbl = pd.read_csv(path, sep="\t")
    if "kmer" not in tbl.columns or "z_setAB" not in tbl.columns:
        return -1.0
    col = pd.to_numeric(tbl["z_setAB"], errors="coerce")
    col = col[col >= min_z]
    if col.empty:
        return -1.0
    return float(col.nlargest(top_k).mean())


def normalize_protein_col(df: pd.DataFrame) -> pd.DataFrame:
    if "protein_name" not in df.columns and "target_name" in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})
    return df


def best_rnacompete_experiment_id(
    df: pd.DataFrame,
    kmer_dir: Path | None = None,
    top_k: int = 10,
    min_z: float = 2.0,
) -> str | None:
    id_col = next((c for c in ("experiment_id", "hyb_id") if c in df.columns), None)
    if id_col is None:
        return None
    if df[id_col].nunique() <= 1:
        return str(df[id_col].iloc[0]) if len(df) else None
    best_eid, best_score = None, -1.0
    for eid in df[id_col].dropna().unique():
        score = _mean_top_kmer_score(kmer_dir, str(eid), top_k, min_z)
        if score > best_score:
            best_score, best_eid = score, str(eid)
    if best_eid and best_score >= 0:
        return best_eid
    pos = df[df["binding_label"] == 1].copy()
    if pos.empty or "probe_intensity" not in pos.columns:
        return str(df[id_col].iloc[0])
    pos["probe_intensity"] = pd.to_numeric(pos["probe_intensity"], errors="coerce")
    return str(pos.groupby(id_col)["probe_intensity"].mean().idxmax())


def load_rnacompete_maps(
    rnacompete_root: Path,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """(dataset, protein) -> experiment_id; (experiment_id, protein) -> sequence."""
    exp_map: dict[tuple[str, str], str] = {}
    seq_map: dict[tuple[str, str], str] = {}
    for dataset, rel in RNAcompete_DATASETS.items():
        path = rnacompete_root / rel
        if not path.exists():
            print(f"  [WARN] RNAcompete clean not found: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df = normalize_protein_col(df)
        df["binding_label"] = pd.to_numeric(df["binding_label"], errors="coerce")
        id_col = next((c for c in ("experiment_id", "hyb_id") if c in df.columns), None)
        for prot, sub in df.groupby("protein_name", sort=False):
            rel_kmer = RNAcompete_KMER_DIRS.get(dataset)
            kmer_dir = (rnacompete_root / rel_kmer) if rel_kmer else None
            if kmer_dir is not None and not kmer_dir.is_dir():
                kmer_dir = None
            eid = best_rnacompete_experiment_id(sub, kmer_dir)
            if eid:
                exp_map[(dataset, prot)] = eid
            if id_col is None:
                continue
            for hyb_id in sub[id_col].dropna().astype(str).unique():
                eid_rows = sub[sub[id_col].astype(str) == hyb_id]
                seqs = eid_rows["protein_sequence"].dropna().astype(str)
                if not seqs.empty:
                    seq_map[(hyb_id, prot)] = seqs.iloc[0].strip()
    return exp_map, seq_map


def load_rbns_map(rbns_experiments: Path) -> tuple[dict[str, str], dict[str, str]]:
    df = pd.read_csv(rbns_experiments, sep="\t")
    pulldown = df[~df["is_control"].astype(bool)].copy()
    exp = pulldown.set_index("target_name")["experiment_accession"].astype(str).to_dict()
    return exp, {}


def load_rbns_sequences(
    rbns_clean: Path,
    rbns_exp: dict[str, str],
    seq_map: dict[tuple[str, str], str],
) -> None:
    df = pd.read_csv(rbns_clean, sep="\t", usecols=["target_name", "protein_sequence"], low_memory=False)
    for prot, sub in df.groupby("target_name", sort=False):
        seqs = sub["protein_sequence"].dropna().astype(str)
        if seqs.empty:
            continue
        eid = rbns_exp.get(prot)
        if eid:
            seq_map[(eid, prot)] = seqs.iloc[0].strip()


def load_htr_map(htr_metadata: Path) -> tuple[dict[str, str], dict[str, str]]:
    df = pd.read_csv(htr_metadata, sep="\t", low_memory=False)
    if "is_background" in df.columns:
        df = df[~df["is_background"].astype(bool)]
    exp_map: dict[str, str] = {}
    for prot, sub in df.groupby("protein_name", sort=False):
        mode = sub["experiment_accession"].astype(str).value_counts().idxmax()
        exp_map[prot] = mode
    return exp_map, {}


def load_htr_sequences(
    htr_clean: Path,
    htr_exp: dict[str, str],
    seq_map: dict[tuple[str, str], str],
) -> None:
    df = pd.read_csv(htr_clean, sep="\t", usecols=["protein_name", "protein_sequence"], low_memory=False)
    for prot, sub in df.groupby("protein_name", sort=False):
        seqs = sub["protein_sequence"].dropna().astype(str)
        if seqs.empty:
            continue
        eid = htr_exp.get(prot)
        if eid:
            seq_map[(eid, prot)] = seqs.iloc[0].strip()


def annotate_summary(
    summary: pd.DataFrame,
    rnacompete_exp: dict[tuple[str, str], str],
    rbns_exp: dict[str, str],
    htr_exp: dict[str, str],
) -> pd.DataFrame:
    out = summary.copy()
    experiment_ids: list[str | None] = []
    missing: list[str] = []

    for _, row in out.iterrows():
        prot = row["protein_name"]
        protocol = row["protocol"]
        eid: str | None = None
        existing = row.get("experiment_id")
        if pd.notna(existing) and str(existing).strip():
            eid = str(existing).strip()
        elif protocol == "rnacompete":
            eid = rnacompete_exp.get((row["dataset"], prot))
        elif protocol == "rbns":
            eid = rbns_exp.get(prot)
        elif protocol == "htr_selex":
            eid = htr_exp.get(prot)
        if eid is None:
            missing.append(f"{protocol}:{prot}")
        experiment_ids.append(eid)

    out["experiment_id"] = experiment_ids
    if missing:
        print(f"  [WARN] {len(missing)} rows missing experiment_id (unique: {len(set(missing))})")
        for m in sorted(set(missing))[:8]:
            print(f"    {m}")
        if len(set(missing)) > 8:
            print(f"    ... and {len(set(missing)) - 8} more")

    cols = list(out.columns)
    if "experiment_id" in cols:
        cols.remove("experiment_id")
    insert_at = cols.index("protein_name") + 1
    cols.insert(insert_at, "experiment_id")
    return out[cols]


def write_fasta(
    summary: pd.DataFrame,
    seq_map: dict[tuple[str, str], str],
    out_path: Path,
) -> int:
    pairs = summary[["experiment_id", "protein_name"]].drop_duplicates()
    pairs = pairs.dropna(subset=["experiment_id"])
    lines: list[str] = []
    skipped = 0
    for _, row in pairs.sort_values(["experiment_id", "protein_name"]).iterrows():
        prot = str(row["protein_name"])
        eid = str(row["experiment_id"])
        seq = seq_map.get((eid, prot))
        if not seq:
            skipped += 1
            continue
        lines.append(f">{eid},{prot}")
        # wrap at 80 aa
        for i in range(0, len(seq), 80):
            lines.append(seq[i: i + 80])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    n = len(pairs) - skipped
    if skipped:
        print(f"  [WARN] FASTA skipped {skipped} proteins (no sequence in clean data)")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Annotate top/bottom summary with experiment IDs + FASTA")
    p.add_argument(
        "--summary_tsv",
        default="results/top_bottom_examples/all_protocols_summary.tsv",
    )
    p.add_argument(
        "--out_summary",
        default=None,
        help="Annotated summary (default: overwrite --summary_tsv)",
    )
    p.add_argument(
        "--out_fasta",
        default="results/top_bottom_examples/all_protocols_proteins.fasta",
    )
    p.add_argument(
        "--rnacompete_root",
        default="../rnacompete_analysis",
    )
    p.add_argument(
        "--rbns_experiments",
        default="../rbns_analysis/data/metadata/rbns_experiments.tsv",
    )
    p.add_argument(
        "--rbns_clean",
        default="../rbns_analysis/results/ml_dataset_rbns_clean.tsv",
    )
    p.add_argument(
        "--htr_metadata",
        default="../htr_selex_analysis/data/metadata/fastq_metadata.tsv",
    )
    p.add_argument(
        "--htr_clean",
        default="../htr_selex_analysis/results/ml_dataset_simple_clean.tsv",
    )
    args = p.parse_args()

    summary_path = Path(args.summary_tsv)
    if not summary_path.exists():
        sys.exit(f"ERROR: summary not found: {summary_path}")

    print(f"Loading summary: {summary_path}")
    summary = pd.read_csv(summary_path, sep="\t")
    seq_map: dict[tuple[str, str], str] = {}

    print("Building RNAcompete experiment map ...")
    rc_exp, rc_seq = load_rnacompete_maps(Path(args.rnacompete_root))
    seq_map.update(rc_seq)

    print("Building RBNS experiment map ...")
    rbns_exp, _ = load_rbns_map(Path(args.rbns_experiments))
    load_rbns_sequences(Path(args.rbns_clean), rbns_exp, seq_map)

    print("Building HTR-SELEX experiment map ...")
    htr_exp, _ = load_htr_map(Path(args.htr_metadata))
    load_htr_sequences(Path(args.htr_clean), htr_exp, seq_map)

    annotated = annotate_summary(summary, rc_exp, rbns_exp, htr_exp)
    out_summary = Path(args.out_summary) if args.out_summary else summary_path
    annotated.to_csv(out_summary, sep="\t", index=False)
    print(f"Annotated summary: {out_summary} ({len(annotated):,} rows)")

    n_fasta = write_fasta(annotated, seq_map, Path(args.out_fasta))
    print(f"Protein FASTA: {args.out_fasta} ({n_fasta} sequences)")
    print("Done.")


if __name__ == "__main__":
    main()
