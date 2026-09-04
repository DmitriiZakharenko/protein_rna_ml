#!/usr/bin/env python3
"""
41_build_skipper_eclip_benchmark.py
-----------------------------------
Build protein–RNA pair TSVs from Skipper/Dropbox eCLIP pos/neg FASTA (fixlen_151).

Joins RNA windows to canonical ENCODE RBP protein sequences and InterPro domain
intervals from ``encode_eclip_rbp_id_best_acc_seq.added_domain_annot.tsv``.

Primary use case: train on generalized_v3a (in vitro) → evaluate on eCLIP pairs
whose gene symbols were **not** seen in v3a training (protein-disjoint OOD).

Inputs (default paths under data/raw/eclip_skipper/):
  extracted/fixlen_151_fasta/*.fixlen_151.{positives,negatives}.fa
  manifests/encode_eclip_rbp_id_best_acc_seq.added_domain_annot.tsv

Outputs (default):
  data/benchmarks/skipper_eclip/fixlen151_all.tsv
  data/benchmarks/skipper_eclip/fixlen151_protein_disjoint_v3a.tsv
  results/skipper_eclip/build_summary.json

Usage
-----
  python scripts/41_build_skipper_eclip_benchmark.py

  python scripts/41_build_skipper_eclip_benchmark.py \\
    --fasta_dir data/raw/eclip_skipper/extracted/fixlen_151_fasta \\
    --train_tsv data/sanitized/generalized_v3a/train.tsv \\
    --max_per_class_per_experiment 200 \\
    --seed 42

Evaluate with script 11 / 21b:
  python scripts/11_evaluate_external.py \\
    --benchmark_tsv data/benchmarks/skipper_eclip/fixlen151_protein_disjoint_v3a.tsv \\
    --v2_dir models/saved/generalized_v2 --rna_max 151 --prot_max 700
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.external_benchmark import TrainLeakageIndex
from src.data.protein_names import base_gene_key
from src.data.protein_sequence import sanitize_protein_sequence, validate_protein_sequence
from src.data.rna_sequence import validate_rna

FASTA_NAME_RE = re.compile(
    r"^(.+)_([^_]+)_(ENCSR[A-Z0-9]+)\.fixlen_151\.(positives|negatives)\.fa$"
)


def resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    cur_id: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                records.append((cur_id, "".join(chunks)))
            cur_id = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if cur_id is not None:
        records.append((cur_id, "".join(chunks)))
    return records


@dataclass(frozen=True)
class EclipExperiment:
    eclip_id: str
    symbol: str
    cell_line: str
    encode_accession: str
    label: int


def parse_fasta_filename(name: str) -> EclipExperiment | None:
    m = FASTA_NAME_RE.match(name)
    if not m:
        return None
    symbol, cell_line, enc, kind = m.groups()
    return EclipExperiment(
        eclip_id=f"{symbol}_{cell_line}_{enc}",
        symbol=symbol,
        cell_line=cell_line,
        encode_accession=enc,
        label=1 if kind == "positives" else 0,
    )


def load_protein_roster(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "symbol" not in df.columns or "protein_sequence" not in df.columns:
        raise SystemExit(f"Unexpected columns in {path}: {list(df.columns)}")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["gene_key"] = df["symbol"].map(base_gene_key)
    dup = df["symbol"].duplicated(keep=False)
    if dup.any():
        raise SystemExit(
            f"Duplicate symbols in roster: {df.loc[dup, 'symbol'].unique()[:10]}"
        )
    return df.set_index("symbol", drop=False)


def subsample_records(
    records: list[tuple[str, str]],
    max_n: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    if max_n <= 0 or len(records) <= max_n:
        return records
    return rng.sample(records, max_n)


def build_pairs(
    fasta_dir: Path,
    roster: pd.DataFrame,
    *,
    max_per_class_per_experiment: int,
    seed: int,
    length_mode: str,
) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    stats = {
        "n_fasta_files": 0,
        "n_files_skipped_parse": 0,
        "n_missing_roster_symbol": 0,
        "n_invalid_rna": 0,
        "n_invalid_protein": 0,
        "n_rows": 0,
        "per_experiment": [],
    }
    missing_symbols: set[str] = set()

    for path in sorted(fasta_dir.glob("*.fixlen_151.*.fa")):
        meta = parse_fasta_filename(path.name)
        if meta is None:
            stats["n_files_skipped_parse"] += 1
            continue
        stats["n_fasta_files"] += 1

        if meta.symbol not in roster.index:
            stats["n_missing_roster_symbol"] += 1
            missing_symbols.add(meta.symbol)
            continue

        prot_row = roster.loc[meta.symbol]
        prot_seq_raw = str(prot_row["protein_sequence"])
        ok_prot, prot_seq = validate_protein_sequence(prot_seq_raw)
        if not ok_prot:
            prot_seq = sanitize_protein_sequence(prot_seq_raw)
            ok_prot, prot_seq = validate_protein_sequence(prot_seq)
        if not ok_prot:
            stats["n_invalid_protein"] += 1
            continue

        records = parse_fasta(path)
        records = subsample_records(records, max_per_class_per_experiment, rng)
        n_kept = 0
        for example_id, rna_raw in records:
            ok_rna, rna_seq = validate_rna(rna_raw)
            if not ok_rna:
                stats["n_invalid_rna"] += 1
                continue
            rows.append(
                {
                    "protein_name": meta.symbol,
                    "protein_sequence": prot_seq,
                    "rna_sequence": rna_seq,
                    "binding_label": meta.label,
                    "dataset": "eclip_skipper",
                    "dataset_source": "eclip_skipper",
                    "eclip_id": meta.eclip_id,
                    "cell_line": meta.cell_line,
                    "encode_accession": meta.encode_accession,
                    "example_id": example_id,
                    "rna_length_mode": length_mode,
                    "uniprot_accession": prot_row.get("best_uniprot_accession", ""),
                    "domain_names": prot_row.get("domain_names", ""),
                    "hit_pos": prot_row.get("hit_pos", ""),
                    "investigated_as": prot_row.get("investigated_as", ""),
                }
            )
            n_kept += 1
        stats["per_experiment"].append(
            {
                "eclip_id": meta.eclip_id,
                "label": meta.label,
                "n_kept": n_kept,
                "n_source": len(records),
            }
        )

    df = pd.DataFrame(rows)
    stats["n_rows"] = len(df)
    stats["missing_roster_symbols"] = sorted(missing_symbols)
    stats["n_pos"] = int((df["binding_label"] == 1).sum()) if len(df) else 0
    stats["n_neg"] = int((df["binding_label"] == 0).sum()) if len(df) else 0
    stats["n_proteins"] = int(df["protein_name"].nunique()) if len(df) else 0
    stats["n_experiments"] = int(df["eclip_id"].nunique()) if len(df) else 0
    return df, stats


def annotate_train_leakage(
    df: pd.DataFrame,
    train_index: TrainLeakageIndex | None,
) -> pd.DataFrame:
    out = df.copy()
    if train_index is None:
        out["protein_name_in_train"] = False
        out["protein_sequence_in_train"] = False
        out["gene_key_in_train"] = False
        return out

    out["protein_name_in_train"] = out["protein_name"].isin(
        train_index.train_protein_names
    )
    out["protein_sequence_in_train"] = out["protein_sequence"].isin(
        train_index.train_protein_sequences
    )
    train_keys = {base_gene_key(n) for n in train_index.train_protein_names}
    out["gene_key"] = out["protein_name"].map(base_gene_key)
    out["gene_key_in_train"] = out["gene_key"].isin(train_keys)
    return out


def protein_disjoint_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose gene_key was not in v3a training (strict OOD proteins)."""
    if "gene_key_in_train" not in df.columns:
        raise ValueError("Run annotate_train_leakage first")
    return df.loc[~df["gene_key_in_train"]].reset_index(drop=True)


def write_summary(
    path: Path,
    *,
    build_stats: dict,
    all_df: pd.DataFrame,
    disjoint_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    summary = {
        "script": "41_build_skipper_eclip_benchmark.py",
        "fasta_dir": str(args.fasta_dir),
        "domain_annot": str(args.domain_annot),
        "train_tsv": str(args.train_tsv) if args.train_tsv else None,
        "max_per_class_per_experiment": args.max_per_class_per_experiment,
        "seed": args.seed,
        "rna_length_mode": args.length_mode,
        "build": {
            k: v
            for k, v in build_stats.items()
            if k != "per_experiment"
        },
        "all": {
            "n_rows": len(all_df),
            "n_pos": int((all_df["binding_label"] == 1).sum()),
            "n_neg": int((all_df["binding_label"] == 0).sum()),
            "n_proteins": int(all_df["protein_name"].nunique()),
            "n_experiments": int(all_df["eclip_id"].nunique()),
            "n_gene_key_in_train": int(all_df["gene_key_in_train"].sum()),
        },
        "protein_disjoint_v3a": {
            "n_rows": len(disjoint_df),
            "n_pos": int((disjoint_df["binding_label"] == 1).sum()),
            "n_neg": int((disjoint_df["binding_label"] == 0).sum()),
            "n_proteins": int(disjoint_df["protein_name"].nunique()),
            "n_experiments": int(disjoint_df["eclip_id"].nunique()),
        },
        "outputs": {
            "all_tsv": str(args.out_all),
            "disjoint_tsv": str(args.out_disjoint),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Skipper eCLIP fixlen_151 benchmark TSVs"
    )
    ap.add_argument(
        "--fasta_dir",
        default="data/raw/eclip_skipper/extracted/fixlen_151_fasta",
    )
    ap.add_argument(
        "--domain_annot",
        default=(
            "data/raw/eclip_skipper/manifests/"
            "encode_eclip_rbp_id_best_acc_seq.added_domain_annot.tsv"
        ),
    )
    ap.add_argument(
        "--train_tsv",
        default="data/sanitized/generalized_v3a/train.tsv",
        help="v3a train TSV for protein-disjoint filter (empty to skip)",
    )
    ap.add_argument(
        "--max_per_class_per_experiment",
        type=int,
        default=200,
        help="Subsample pos/neg per experiment (0 = no cap; ~1.5M rows total)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length_mode", default="fixlen_151")
    ap.add_argument(
        "--out_dir",
        default="data/benchmarks/skipper_eclip",
    )
    ap.add_argument(
        "--summary_json",
        default="results/skipper_eclip/build_summary.json",
    )
    args = ap.parse_args()

    fasta_dir = resolve(args.fasta_dir)
    if not fasta_dir.is_dir():
        raise SystemExit(
            f"FASTA dir not found: {fasta_dir}\n"
            "Extract with: tar -xJf data/raw/eclip_skipper/archives/dropbox/"
            "eclip_various_pos_neg_sets.hg38.tar.xz"
        )

    roster = load_protein_roster(resolve(args.domain_annot))
    print(f"  Protein roster: {len(roster)} symbols")

    df, build_stats = build_pairs(
        fasta_dir,
        roster,
        max_per_class_per_experiment=args.max_per_class_per_experiment,
        seed=args.seed,
        length_mode=args.length_mode,
    )
    if df.empty:
        raise SystemExit("No rows built — check FASTA paths and roster join")

    train_index = None
    if args.train_tsv:
        train_path = resolve(args.train_tsv)
        if not train_path.is_file():
            raise SystemExit(f"train_tsv not found: {train_path}")
        train_index = TrainLeakageIndex.from_train_tsv(train_path)
        print(f"  v3a train proteins: {len(train_index.train_protein_names):,}")

    df = annotate_train_leakage(df, train_index)
    disjoint = protein_disjoint_subset(df)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_all = out_dir / f"{args.length_mode}_all.tsv"
    args.out_disjoint = out_dir / f"{args.length_mode}_protein_disjoint_v3a.tsv"

    df.to_csv(args.out_all, sep="\t", index=False)
    disjoint.to_csv(args.out_disjoint, sep="\t", index=False)

    summary_path = resolve(args.summary_json)
    write_summary(summary_path, build_stats=build_stats, all_df=df, disjoint_df=disjoint, args=args)

    print(f"\n=== Built Skipper eCLIP benchmark ({args.length_mode}) ===")
    print(f"  All rows:      {len(df):,}  ({df['protein_name'].nunique()} proteins)")
    print(f"  Protein-disjoint vs v3a train: {len(disjoint):,}  "
          f"({disjoint['protein_name'].nunique()} proteins)")
    print(f"  Wrote {args.out_all}")
    print(f"  Wrote {args.out_disjoint}")
    print(f"  Summary {summary_path}")


if __name__ == "__main__":
    main()
