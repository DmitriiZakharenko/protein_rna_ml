#!/usr/bin/env python3
"""
41b_split_skipper_eclip_jose_style.py
-------------------------------------
Protein-disjoint train/val/test splits **within Skipper eCLIP only** (Jose-style).

Unlike script 41's ``protein_disjoint_v3a`` export (train in vitro → test eCLIP),
this holds out eCLIP proteins for test while training on other eCLIP proteins
from the same assay and window length (fixlen_151).

Typical workflow (replicate RPIembeddor / Jose comparison with V2 CNN):

  # 1) Build pair TSV (if not done)
  python scripts/41_build_skipper_eclip_benchmark.py

  # 2) Split by protein
  python scripts/41b_split_skipper_eclip_jose_style.py

  # 3) Train V2 on eCLIP train only
  python scripts/06_train_generalized_v2.py \\
    --data_dir data/benchmarks/skipper_eclip/jose_style \\
    --rna_max 151 --prot_max 700 \\
    --model_dir models/saved/skipper_eclip_v2_rna151 \\
    --out_dir results/skipper_eclip/jose_style_v2_train

  # 4) Eval is built into script 06 (test split); or score test.tsv manually.

Usage
-----
  python scripts/41b_split_skipper_eclip_jose_style.py

  python scripts/41b_split_skipper_eclip_jose_style.py \\
    --pairs_tsv data/benchmarks/skipper_eclip/fixlen_151_all.tsv \\
    --out_dir data/benchmarks/skipper_eclip/jose_style \\
    --train_frac 0.75 --val_frac 0.11 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.external_benchmark import TrainLeakageIndex
from src.data.protein_names import base_gene_key
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
    spec.loader.exec_module(mod)
    return mod


def load_or_build_pairs(args: argparse.Namespace) -> pd.DataFrame:
    pairs_path = resolve(args.pairs_tsv) if args.pairs_tsv else None
    if pairs_path and pairs_path.is_file() and not args.rebuild:
        print(f"  Loading pairs: {pairs_path}")
        return pd.read_csv(pairs_path, sep="\t", low_memory=False)

    if args.rebuild or pairs_path is None or not pairs_path.is_file():
        mod = _load_build41()
        fasta_dir = resolve(args.fasta_dir)
        roster = mod.load_protein_roster(resolve(args.domain_annot))
        df, _ = mod.build_pairs(
            fasta_dir,
            roster,
            max_per_class_per_experiment=args.max_per_class_per_experiment,
            seed=args.seed,
            length_mode=args.length_mode,
        )
        if df.empty:
            raise SystemExit("No pairs built from FASTA")
        if pairs_path:
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(pairs_path, sep="\t", index=False)
            print(f"  Wrote {pairs_path}")
        return df

    raise SystemExit(f"pairs_tsv not found: {pairs_path}")


def proteins_with_both_classes(df: pd.DataFrame) -> set[str]:
    both: set[str] = set()
    for prot, g in df.groupby("protein_name"):
        labels = set(g["binding_label"].astype(int).unique())
        if labels >= {0, 1}:
            both.add(str(prot))
    return both


def to_train_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = "eclip_skipper"
    out["dataset_source"] = "eclip_skipper"
    cols = [c for c in TRAIN_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra]


def split_stats(df: pd.DataFrame, split_name: str) -> dict:
    if df.empty:
        return {"split": split_name, "n_rows": 0, "n_proteins": 0}
    return {
        "split": split_name,
        "n_rows": int(len(df)),
        "n_pos": int((df["binding_label"] == 1).sum()),
        "n_neg": int((df["binding_label"] == 0).sum()),
        "n_proteins": int(df["protein_name"].nunique()),
        "n_experiments": int(df["eclip_id"].nunique()) if "eclip_id" in df.columns else None,
        "proteins_both_classes": int(
            sum(
                1
                for _, g in df.groupby("protein_name")
                if g["binding_label"].nunique() >= 2
            )
        ),
    }


def annotate_v3a_overlap(
    split_map: pd.DataFrame,
    train_tsv: Path | None,
) -> pd.DataFrame:
    out = split_map.copy()
    if train_tsv is None or not train_tsv.is_file():
        out["gene_key_in_v3a_train"] = False
        return out
    idx = TrainLeakageIndex.from_train_tsv(train_tsv)
    train_keys = {base_gene_key(n) for n in idx.train_protein_names}
    out["gene_key"] = out["protein_name"].map(base_gene_key)
    out["gene_key_in_v3a_train"] = out["gene_key"].isin(train_keys)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Jose-style protein-disjoint eCLIP train/val/test splits"
    )
    ap.add_argument(
        "--pairs_tsv",
        default="data/benchmarks/skipper_eclip/fixlen_151_all.tsv",
        help="Input pairs from script 41 (rebuilt if missing and --rebuild)",
    )
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild from FASTA")
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
    ap.add_argument("--max_per_class_per_experiment", type=int, default=200)
    ap.add_argument("--length_mode", default="fixlen_151")
    ap.add_argument(
        "--out_dir",
        default="data/benchmarks/skipper_eclip/jose_style",
    )
    ap.add_argument(
        "--summary_json",
        default="results/skipper_eclip/jose_style_split_summary.json",
    )
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--val_frac", type=float, default=0.11)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow_single_class_proteins",
        action="store_true",
        help="Keep proteins that lack pos or neg (default: require both classes)",
    )
    ap.add_argument(
        "--train_tsv",
        default="data/sanitized/generalized_v3a/train.tsv",
        help="Annotate overlap with in-vitro train (empty to skip)",
    )
    args = ap.parse_args()

    df = load_or_build_pairs(args)

    if args.allow_single_class_proteins:
        eligible = set(df["protein_name"].astype(str).unique())
    else:
        eligible = proteins_with_both_classes(df)
        n_drop = df["protein_name"].nunique() - len(eligible)
        if n_drop:
            print(f"  Dropping {n_drop} proteins without both pos and neg")
        df = df[df["protein_name"].isin(eligible)].reset_index(drop=True)

    if df.empty:
        raise SystemExit("No rows after filtering")

    train_df, val_df, test_df, split_map = protein_aware_split(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        protein_col="protein_name",
    )

    train_tsv_path = resolve(args.train_tsv) if args.train_tsv else None
    split_map = annotate_v3a_overlap(split_map, train_tsv_path)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = to_train_schema(train_df)
    val_out = to_train_schema(val_df)
    test_out = to_train_schema(test_df)

    train_out.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    val_out.to_csv(out_dir / "val.tsv", sep="\t", index=False)
    test_out.to_csv(out_dir / "test.tsv", sep="\t", index=False)
    split_map.to_csv(out_dir / "protein_split_map.tsv", sep="\t", index=False)

    summary = {
        "script": "41b_split_skipper_eclip_jose_style.py",
        "description": "Protein-disjoint split within Skipper eCLIP (Jose / RPIembeddor style)",
        "pairs_tsv": str(resolve(args.pairs_tsv)) if args.pairs_tsv else None,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "length_mode": args.length_mode,
        "n_eligible_proteins": len(eligible),
        "splits": {
            "train": split_stats(train_df, "train"),
            "val": split_stats(val_df, "val"),
            "test": split_stats(test_df, "test"),
        },
        "v3a_train_overlap": {
            "train_split_proteins_in_v3a": int(
                split_map.loc[split_map["split"] == "train", "gene_key_in_v3a_train"]
                .sum()
            )
            if "gene_key_in_v3a_train" in split_map.columns
            else None,
            "test_split_proteins_in_v3a": int(
                split_map.loc[split_map["split"] == "test", "gene_key_in_v3a_train"]
                .sum()
            )
            if "gene_key_in_v3a_train" in split_map.columns
            else None,
        },
        "outputs": {
            "data_dir": str(out_dir),
            "train_tsv": str(out_dir / "train.tsv"),
            "val_tsv": str(out_dir / "val.tsv"),
            "test_tsv": str(out_dir / "test.tsv"),
            "protein_split_map": str(out_dir / "protein_split_map.tsv"),
        },
        "train_command": (
            "python scripts/06_train_generalized_v2.py "
            f"--data_dir {out_dir.relative_to(ROOT)} "
            "--rna_max 151 --prot_max 700 "
            "--model_dir models/saved/skipper_eclip_v2_rna151 "
            "--out_dir results/skipper_eclip/jose_style_v2_train"
        ),
    }

    summary_path = resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Jose-style eCLIP protein split ===")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(
            f"  {name:5s}: {len(part):,} rows  "
            f"{part['protein_name'].nunique()} proteins  "
            f"pos={int((part['binding_label']==1).sum()):,}"
        )
    print(f"\n  Wrote {out_dir}/{{train,val,test}}.tsv")
    print(f"  Wrote {out_dir}/protein_split_map.tsv")
    print(f"  Summary {summary_path}")
    print(f"\n  Next:\n    {summary['train_command']}")


if __name__ == "__main__":
    main()
