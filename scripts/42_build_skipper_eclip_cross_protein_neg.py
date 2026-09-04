#!/usr/bin/env python3
"""
42_build_skipper_eclip_cross_protein_neg.py
-------------------------------------------
Build Jose-style eCLIP pairs: negatives = **positive peaks from other proteins**.

Replaces Skipper ``rand_neg`` windows with cross-protein positives so the model
must learn protein-specific binding rather than generic eCLIP pos vs random background
(see Jose Gil thesis §5.2.1; RPIembeddor eCLIP2 protein-disjoint ~0.72 AUROC).

Workflow
--------
  python scripts/42_build_skipper_eclip_cross_protein_neg.py

  python scripts/41b_split_skipper_eclip_jose_style.py \\
    --pairs_tsv data/benchmarks/skipper_eclip/fixlen_151_cross_protein_neg_all.tsv \\
    --out_dir data/benchmarks/skipper_eclip/jose_cross_protein_neg \\
    --summary_json results/skipper_eclip/jose_cross_protein_neg_split_summary.json

  python scripts/06_train_generalized_v2.py \\
    --data_dir data/benchmarks/skipper_eclip/jose_cross_protein_neg \\
    --rna_max 151 --prot_max 700 \\
    --model_dir models/saved/skipper_eclip_v2_jose_hard \\
    --out_dir results/skipper_eclip/jose_cross_protein_neg_v2_train

Compare to rand_neg Jose-style (~0.89 pp-median) vs thesis protein-disjoint (~0.72).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import protein_aware_split

TRAIN_COLUMNS = [
    "protein_name",
    "protein_sequence",
    "rna_sequence",
    "binding_label",
    "dataset",
    "dataset_source",
]


def resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_build41():
    path = ROOT / "scripts" / "41_build_skipper_eclip_benchmark.py"
    spec = importlib.util.spec_from_file_location("build_skipper_41", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_skipper_41"] = mod
    spec.loader.exec_module(mod)
    return mod


def collect_positives(
    fasta_dir: Path,
    roster: pd.DataFrame,
    *,
    max_per_experiment: int,
    seed: int,
    length_mode: str,
) -> tuple[pd.DataFrame, dict]:
    """Load only *.positives.fa rows (binding_label=1)."""
    mod = _load_build41()
    rng = random.Random(seed)
    rows: list[dict] = []
    stats = {
        "n_pos_fasta_files": 0,
        "n_skipped": 0,
        "n_invalid_rna": 0,
        "n_invalid_protein": 0,
    }

    for path in sorted(fasta_dir.glob("*.fixlen_151.positives.fa")):
        meta = mod.parse_fasta_filename(path.name)
        if meta is None or meta.label != 1:
            stats["n_skipped"] += 1
            continue
        if meta.symbol not in roster.index:
            stats["n_skipped"] += 1
            continue

        stats["n_pos_fasta_files"] += 1
        prot_row = roster.loc[meta.symbol]
        prot_seq_raw = str(prot_row["protein_sequence"])
        ok_prot, prot_seq = mod.validate_protein_sequence(prot_seq_raw)
        if not ok_prot:
            prot_seq = mod.sanitize_protein_sequence(prot_seq_raw)
            ok_prot, prot_seq = mod.validate_protein_sequence(prot_seq)
        if not ok_prot:
            stats["n_invalid_protein"] += 1
            continue

        records = mod.parse_fasta(path)
        records = mod.subsample_records(records, max_per_experiment, rng)
        for example_id, rna_raw in records:
            ok_rna, rna_seq = mod.validate_rna(rna_raw)
            if not ok_rna:
                stats["n_invalid_rna"] += 1
                continue
            rows.append(
                {
                    "protein_name": meta.symbol,
                    "protein_sequence": prot_seq,
                    "rna_sequence": rna_seq,
                    "binding_label": 1,
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
                    "neg_strategy": "native_positive",
                }
            )

    pos_df = pd.DataFrame(rows)
    stats["n_pos_rows"] = len(pos_df)
    stats["n_proteins"] = int(pos_df["protein_name"].nunique()) if len(pos_df) else 0
    return pos_df, stats


def _base_row(pos_row: pd.Series) -> dict:
    return {
        "protein_name": pos_row["protein_name"],
        "protein_sequence": pos_row["protein_sequence"],
        "rna_sequence": pos_row["rna_sequence"],
        "dataset": pos_row["dataset"],
        "dataset_source": pos_row["dataset_source"],
        "eclip_id": pos_row["eclip_id"],
        "cell_line": pos_row["cell_line"],
        "encode_accession": pos_row["encode_accession"],
        "example_id": pos_row["example_id"],
        "rna_length_mode": pos_row["rna_length_mode"],
        "uniprot_accession": pos_row.get("uniprot_accession", ""),
        "domain_names": pos_row.get("domain_names", ""),
        "hit_pos": pos_row.get("hit_pos", ""),
        "investigated_as": pos_row.get("investigated_as", ""),
    }


def build_cross_protein_pairs(
    pos_df: pd.DataFrame,
    *,
    seed: int,
    neg_sampling: str = "match_pos_count",
) -> tuple[pd.DataFrame, dict]:
    """
    For each protein P:
      - keep native positives (label=1)
      - negatives: sample positive RNAs from proteins != P (label=0)
    """
    rng = random.Random(seed)
    out_rows: list[dict] = []
    n_proteins_skipped = 0
    n_neg_sampled = 0

    proteins = sorted(pos_df["protein_name"].unique())
    if len(proteins) < 2:
        raise SystemExit("Need >= 2 proteins with positives for cross-protein negatives")

    donor_pool = pos_df.reset_index(drop=True)

    for protein in proteins:
        pos_sub = pos_df[pos_df["protein_name"] == protein]
        donors = donor_pool[donor_pool["protein_name"] != protein]
        if pos_sub.empty or donors.empty:
            n_proteins_skipped += 1
            continue

        for _, anchor in pos_sub.iterrows():
            row = _base_row(anchor)
            row["binding_label"] = 1
            row["neg_strategy"] = "native_positive"
            row["neg_source_protein"] = ""
            row["neg_source_eclip_id"] = ""
            row["neg_source_example_id"] = ""
            out_rows.append(row)

        n_neg = len(pos_sub)
        donor_indices = rng.choices(donors.index.tolist(), k=n_neg)
        for anchor_idx, d_idx in zip(pos_sub.index.tolist(), donor_indices):
            anchor = pos_sub.loc[anchor_idx]
            donor = donors.loc[d_idx]
            row = _base_row(anchor)
            row["rna_sequence"] = donor["rna_sequence"]
            row["binding_label"] = 0
            row["neg_strategy"] = "cross_protein_pos"
            row["example_id"] = (
                f"xprot_neg_{anchor['example_id']}_from_{donor['protein_name']}"
            )
            row["neg_source_protein"] = donor["protein_name"]
            row["neg_source_eclip_id"] = donor["eclip_id"]
            row["neg_source_example_id"] = donor["example_id"]
            out_rows.append(row)
            n_neg_sampled += 1

    df = pd.DataFrame(out_rows)
    meta = {
        "n_proteins": len(proteins),
        "n_proteins_skipped": n_proteins_skipped,
        "n_neg_sampled": n_neg_sampled,
        "neg_sampling": neg_sampling,
    }
    return df, meta


def to_train_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = "eclip_skipper"
    out["dataset_source"] = "eclip_skipper_jose_hard"
    cols = [c for c in TRAIN_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra]


def split_stats(df: pd.DataFrame, name: str) -> dict:
    if df.empty:
        return {"split": name, "n_rows": 0}
    gc = (
        df["rna_sequence"].str.count("G") + df["rna_sequence"].str.count("C")
    ) / df["rna_sequence"].str.len()
    pos_gc = gc[df["binding_label"] == 1].mean() if (df["binding_label"] == 1).any() else None
    neg_gc = gc[df["binding_label"] == 0].mean() if (df["binding_label"] == 0).any() else None
    return {
        "split": name,
        "n_rows": int(len(df)),
        "n_pos": int((df["binding_label"] == 1).sum()),
        "n_neg": int((df["binding_label"] == 0).sum()),
        "n_proteins": int(df["protein_name"].nunique()),
        "n_unique_rnas": int(df["rna_sequence"].nunique()),
        "gc_frac_pos_mean": round(float(pos_gc), 4) if pos_gc is not None else None,
        "gc_frac_neg_mean": round(float(neg_gc), 4) if neg_gc is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Jose-style cross-protein positive negatives for Skipper eCLIP"
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
        "--max_pos_per_experiment",
        type=int,
        default=200,
        help="Subsample native positives per eCLIP experiment before neg generation",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length_mode", default="fixlen_151")
    ap.add_argument(
        "--out_tsv",
        default="data/benchmarks/skipper_eclip/fixlen_151_cross_protein_neg_all.tsv",
    )
    ap.add_argument(
        "--summary_json",
        default="results/skipper_eclip/cross_protein_neg_build_summary.json",
    )
    ap.add_argument(
        "--write_splits",
        action="store_true",
        help="Also write protein-disjoint train/val/test (41b-style)",
    )
    ap.add_argument(
        "--split_dir",
        default="data/benchmarks/skipper_eclip/jose_cross_protein_neg",
    )
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--val_frac", type=float, default=0.11)
    args = ap.parse_args()

    mod = _load_build41()
    fasta_dir = resolve(args.fasta_dir)
    if not fasta_dir.is_dir():
        raise SystemExit(f"FASTA dir not found: {fasta_dir}")

    roster = mod.load_protein_roster(resolve(args.domain_annot))
    print(f"  Protein roster: {len(roster)} symbols")

    pos_df, pos_stats = collect_positives(
        fasta_dir,
        roster,
        max_per_experiment=args.max_pos_per_experiment,
        seed=args.seed,
        length_mode=args.length_mode,
    )
    if pos_df.empty:
        raise SystemExit("No positive rows collected")

    print(
        f"  Positives: {len(pos_df):,} rows  "
        f"{pos_df['protein_name'].nunique()} proteins  "
        f"{pos_df['rna_sequence'].nunique():,} unique RNAs"
    )

    pairs_df, pair_meta = build_cross_protein_pairs(pos_df, seed=args.seed)
    print(
        f"  Cross-protein pairs: {len(pairs_df):,} rows  "
        f"pos={int((pairs_df.binding_label==1).sum()):,}  "
        f"neg={int((pairs_df.binding_label==0).sum()):,}"
    )

    out_tsv = resolve(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Wrote {out_tsv}")

    summary = {
        "script": "42_build_skipper_eclip_cross_protein_neg.py",
        "description": "Jose thesis negatives: other proteins' eCLIP positives as negatives",
        "reference": "Thesis_Final_Jose — §5.2.1 cross-protein positive negatives",
        "fasta_dir": str(fasta_dir),
        "max_pos_per_experiment": args.max_pos_per_experiment,
        "seed": args.seed,
        "positive_collection": pos_stats,
        "pair_build": pair_meta,
        "dataset": split_stats(pairs_df, "all"),
        "outputs": {"pairs_tsv": str(out_tsv)},
        "split_command": (
            "python scripts/41b_split_skipper_eclip_jose_style.py "
            f"--pairs_tsv {out_tsv.relative_to(ROOT)} "
            f"--out_dir {args.split_dir} "
            "--summary_json results/skipper_eclip/jose_cross_protein_neg_split_summary.json"
        ),
        "train_command": (
            "python scripts/06_train_generalized_v2.py "
            f"--data_dir {args.split_dir} "
            "--rna_max 151 --prot_max 700 "
            "--model_dir models/saved/skipper_eclip_v2_jose_hard "
            "--out_dir results/skipper_eclip/jose_cross_protein_neg_v2_train"
        ),
    }

    if args.write_splits:
        train_df, val_df, test_df, split_map = protein_aware_split(
            pairs_df,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        split_dir = resolve(args.split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        to_train_schema(train_df).to_csv(split_dir / "train.tsv", sep="\t", index=False)
        to_train_schema(val_df).to_csv(split_dir / "val.tsv", sep="\t", index=False)
        to_train_schema(test_df).to_csv(split_dir / "test.tsv", sep="\t", index=False)
        split_map.to_csv(split_dir / "protein_split_map.tsv", sep="\t", index=False)
        summary["splits"] = {
            "train": split_stats(train_df, "train"),
            "val": split_stats(val_df, "val"),
            "test": split_stats(test_df, "test"),
        }
        summary["outputs"]["split_dir"] = str(split_dir)
        print(f"\n  Wrote splits → {split_dir}/{{train,val,test}}.tsv")

    summary_path = resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary {summary_path}")
    print(f"\n  Next:\n    {summary['split_command']}")
    print(f"    {summary['train_command']}")


if __name__ == "__main__":
    main()
