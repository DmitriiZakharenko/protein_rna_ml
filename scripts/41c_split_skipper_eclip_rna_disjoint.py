#!/usr/bin/env python3
"""
41c_split_skipper_eclip_rna_disjoint.py
---------------------------------------
Stricter Skipper eCLIP splits for unseen-RNA generalization.

Modes
-----
  rna (default)
      Split by unique ``rna_sequence``. Proteins may appear in train and test
      with different RNAs (ZHMolGraph-style RNA holdout).

  protein_and_rna
      Protein-disjoint split (41b), then drop val/test rows whose RNA appeared
      in train. Test proteins are unseen AND test RNAs never seen in train.

  pair
      Split by unique (protein_name, rna_sequence) pairs.

Typical workflow
----------------
  python scripts/41c_split_skipper_eclip_rna_disjoint.py --mode rna

  python scripts/06_train_generalized_v2.py \\
    --data_dir data/benchmarks/skipper_eclip/rna_disjoint \\
    --rna_max 151 --prot_max 700 \\
    --model_dir models/saved/skipper_eclip_v2_rna151_rna_disjoint \\
    --out_dir results/skipper_eclip/rna_disjoint_v2_train

  python scripts/41d_eval_eclip_diagnostics.py \\
    --checkpoint models/saved/skipper_eclip_v2_rna151/best_model.pt \\
    --train_tsv data/benchmarks/skipper_eclip/jose_style/train.tsv \\
    --test_tsv data/benchmarks/skipper_eclip/jose_style/test.tsv
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

from src.data.splits import (
    drop_eval_rnas_seen_in_train,
    pair_aware_split,
    protein_aware_split,
    rna_aware_split,
)

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
        df, _ = mod.build_pairs(
            resolve(args.fasta_dir),
            mod.load_protein_roster(resolve(args.domain_annot)),
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
        if set(g["binding_label"].astype(int).unique()) >= {0, 1}:
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
        return {"split": split_name, "n_rows": 0, "n_proteins": 0, "n_unique_rnas": 0}
    return {
        "split": split_name,
        "n_rows": int(len(df)),
        "n_pos": int((df["binding_label"] == 1).sum()),
        "n_neg": int((df["binding_label"] == 0).sum()),
        "n_proteins": int(df["protein_name"].nunique()),
        "n_unique_rnas": int(df["rna_sequence"].nunique()),
        "n_experiments": int(df["eclip_id"].nunique()) if "eclip_id" in df.columns else None,
        "proteins_both_classes": int(
            sum(
                1
                for _, g in df.groupby("protein_name")
                if g["binding_label"].nunique() >= 2
            )
        ),
    }


def rna_overlap_stats(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    train_rnas = set(train_df["rna_sequence"])
    eval_rnas = set(eval_df["rna_sequence"])
    overlap = train_rnas & eval_rnas
    rows_overlap = int(eval_df["rna_sequence"].isin(overlap).sum())
    return {
        "n_train_unique_rnas": len(train_rnas),
        "n_eval_unique_rnas": len(eval_rnas),
        "n_shared_rnas": len(overlap),
        "pct_eval_rows_with_train_rna": round(
            100.0 * rows_overlap / max(len(eval_df), 1), 2
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RNA-disjoint / protein+RNA Skipper eCLIP splits"
    )
    ap.add_argument(
        "--mode",
        choices=["rna", "protein_and_rna", "pair"],
        default="rna",
        help="rna: split by RNA; protein_and_rna: protein split + drop train RNAs from eval",
    )
    ap.add_argument(
        "--pairs_tsv",
        default="data/benchmarks/skipper_eclip/fixlen_151_all.tsv",
    )
    ap.add_argument("--rebuild", action="store_true")
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
        default=None,
        help="Default: data/benchmarks/skipper_eclip/{mode}",
    )
    ap.add_argument(
        "--summary_json",
        default=None,
        help="Default: results/skipper_eclip/{mode}_split_summary.json",
    )
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--val_frac", type=float, default=0.11)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--require_both_classes_per_split",
        action="store_true",
        help="Drop proteins lacking pos and neg within each split (can shrink splits)",
    )
    ap.add_argument(
        "--drop_val_rnas_from_test",
        action="store_true",
        help="protein_and_rna only: also exclude val RNAs from test",
    )
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = f"data/benchmarks/skipper_eclip/{args.mode}"
    if args.summary_json is None:
        args.summary_json = f"results/skipper_eclip/{args.mode}_split_summary.json"

    df = load_or_build_pairs(args)
    eligible = proteins_with_both_classes(df)
    df = df[df["protein_name"].isin(eligible)].reset_index(drop=True)
    print(f"  Eligible proteins (both classes globally): {len(eligible)}")

    if args.mode == "rna":
        train_df, val_df, test_df, split_map = rna_aware_split(
            df,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        split_map_name = "rna_split_map.tsv"
    elif args.mode == "pair":
        train_df, val_df, test_df, split_map = pair_aware_split(
            df,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        split_map_name = "pair_split_map.tsv"
    else:
        train_df, val_df, test_df, protein_map = protein_aware_split(
            df,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        n_val_before, n_test_before = len(val_df), len(test_df)
        val_df, test_df = drop_eval_rnas_seen_in_train(
            train_df,
            val_df,
            test_df,
            drop_val_rnas_from_test=args.drop_val_rnas_from_test,
        )
        print(
            f"  protein_and_rna filter: val {n_val_before:,} → {len(val_df):,} rows, "
            f"test {n_test_before:,} → {len(test_df):,} rows"
        )
        split_map = protein_map
        split_map_name = "protein_split_map.tsv"

    if args.require_both_classes_per_split:
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            keep = proteins_with_both_classes(part)
            before = len(part)
            if name == "train":
                train_df = part[part["protein_name"].isin(keep)].reset_index(drop=True)
            elif name == "val":
                val_df = part[part["protein_name"].isin(keep)].reset_index(drop=True)
            else:
                test_df = part[part["protein_name"].isin(keep)].reset_index(drop=True)
            after = len(part[part["protein_name"].isin(keep)])
            print(f"  {name}: kept {len(keep)} proteins, {before:,} → {after:,} rows")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = to_train_schema(train_df)
    val_out = to_train_schema(val_df)
    test_out = to_train_schema(test_df)

    train_out.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    val_out.to_csv(out_dir / "val.tsv", sep="\t", index=False)
    test_out.to_csv(out_dir / "test.tsv", sep="\t", index=False)
    split_map.to_csv(out_dir / split_map_name, sep="\t", index=False)

    summary = {
        "script": "41c_split_skipper_eclip_rna_disjoint.py",
        "mode": args.mode,
        "description": {
            "rna": "RNA-disjoint split (proteins may repeat across splits)",
            "protein_and_rna": "Protein-disjoint + no test/val RNA seen in train",
            "pair": "Unique (protein, RNA) pair disjoint split",
        }[args.mode],
        "pairs_tsv": str(resolve(args.pairs_tsv)),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "splits": {
            "train": split_stats(train_df, "train"),
            "val": split_stats(val_df, "val"),
            "test": split_stats(test_df, "test"),
        },
        "rna_leakage": {
            "val_vs_train": rna_overlap_stats(train_df, val_df),
            "test_vs_train": rna_overlap_stats(train_df, test_df),
        },
        "outputs": {
            "data_dir": str(out_dir),
            "train_tsv": str(out_dir / "train.tsv"),
            "val_tsv": str(out_dir / "val.tsv"),
            "test_tsv": str(out_dir / "test.tsv"),
            "split_map": str(out_dir / split_map_name),
        },
        "train_command": (
            "python scripts/06_train_generalized_v2.py "
            f"--data_dir {out_dir.relative_to(ROOT)} "
            "--rna_max 151 --prot_max 700 "
            f"--model_dir models/saved/skipper_eclip_v2_rna151_{args.mode} "
            f"--out_dir results/skipper_eclip/{args.mode}_v2_train"
        ),
        "diagnostics_command": (
            "python scripts/41d_eval_eclip_diagnostics.py "
            f"--train_tsv {out_dir.relative_to(ROOT)}/train.tsv "
            f"--test_tsv {out_dir.relative_to(ROOT)}/test.tsv "
            f"--checkpoint models/saved/skipper_eclip_v2_rna151_{args.mode}/best_model.pt "
            f"--out_dir results/skipper_eclip/{args.mode}_diagnostics"
        ),
    }

    summary_path = resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== eCLIP split mode={args.mode} ===")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(
            f"  {name:5s}: {len(part):,} rows  "
            f"{part['protein_name'].nunique()} proteins  "
            f"{part['rna_sequence'].nunique():,} unique RNAs"
        )
    leak = summary["rna_leakage"]["test_vs_train"]
    print(
        f"\n  Test RNA leakage vs train: {leak['n_shared_rnas']} shared RNAs, "
        f"{leak['pct_eval_rows_with_train_rna']}% test rows"
    )
    print(f"\n  Wrote {out_dir}/{{train,val,test}}.tsv")
    print(f"  Summary {summary_path}")
    print(f"\n  Next:\n    {summary['train_command']}")


if __name__ == "__main__":
    main()
