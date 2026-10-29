#!/usr/bin/env python3
"""
62_split_integrity_check.py
---------------------------
Integrity audit of a train/val/test directory (v3a, v3b, ...): exact protein-sequence and protein-name overlap between
splits, RNA overlap (share of test/val pairs whose RNA also occurs in train, and as a positive for the same source),
duplicate pairs, class balance, single-class proteins and per-source composition.

Usage:  python scripts/62_split_integrity_check.py --data_dir data/generalized_v3b --out_name v3b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["protein_name", "protein_sequence", "rna_sequence", "binding_label", "dataset_source"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_name", required=True)
    args = ap.parse_args()
    d = Path(args.data_dir)
    d = d if d.is_absolute() else ROOT / d
    sp = {s: pd.read_csv(d / f"{s}.tsv", sep="\t", usecols=COLS, low_memory=False) for s in ("train", "val", "test")}
    out: dict = {"data_dir": args.data_dir, "splits": {}, "overlap": {}}
    tr = sp["train"]
    tr_rna = set(tr.rna_sequence)
    tr_pos_rna = set(tr.loc[tr.binding_label == 1, "rna_sequence"])
    tr_pairs = set(zip(tr.protein_sequence, tr.rna_sequence))
    for s, df in sp.items():
        out["splits"][s] = {
            "rows": len(df), "proteins": int(df.protein_name.nunique()), "pos_frac": round(float(df.binding_label.mean()), 4),
            "single_class_proteins": int((df.groupby("protein_name").binding_label.nunique() < 2).sum()),
            "sources": df.dataset_source.value_counts().to_dict(),
            "duplicate_pairs": int(df.duplicated(["protein_sequence", "rna_sequence", "binding_label"]).sum()),
            "conflicting_labels": int((df.groupby(["protein_sequence", "rna_sequence"]).binding_label.nunique() > 1).sum()),
        }
        if s != "train":
            out["overlap"][s] = {
                "protein_names_in_train": int(df.protein_name.isin(set(tr.protein_name)).sum() and len(set(df.protein_name) & set(tr.protein_name))),
                "protein_sequences_in_train": len(set(df.protein_sequence) & set(tr.protein_sequence)),
                "pct_rows_rna_in_train": round(100 * float(df.rna_sequence.isin(tr_rna).mean()), 2),
                "pct_rows_rna_positive_in_train": round(100 * float(df.rna_sequence.isin(tr_pos_rna).mean()), 2),
                "pct_rows_exact_pair_in_train": round(100 * float(pd.Series(list(zip(df.protein_sequence, df.rna_sequence))).isin(tr_pairs).mean()), 2),
            }
    res = ROOT / "results/split_integrity"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{args.out_name}.json").write_text(json.dumps(out, indent=2, default=int))
    print(json.dumps(out, indent=2, default=int))


if __name__ == "__main__":
    main()
