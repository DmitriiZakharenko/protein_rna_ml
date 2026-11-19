#!/usr/bin/env python3
"""
79_convert_coral_folds.py
-------------------------
Published CORAL folds (train.csv / val.csv with RNA_id, Protein_id, labels, RNA_seqs, Prot_seqs) in the column layout of the
generalized TSVs: protein_name, rna_sequence (T written as U), protein_sequence, binding_label, dataset_source, data_type,
split, fold, rna_id.

One file per fold and split: <out_dir>/<strategy>__<fold>__<split>.tsv; additionally all_unique_molecules.tsv with the unique
proteins and RNAs of the converted folds (input for the embedding script).

Usage
-----
  python scripts/79_convert_coral_folds.py --data_dir data/benchmarks/coral/datasets --out_dir data/benchmarks/coral/tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/benchmarks/coral/datasets")
    ap.add_argument("--out_dir", default="data/benchmarks/coral/tsv")
    args = ap.parse_args()
    root, out = resolve(args.data_dir), resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prot, rna = {}, {}
    for d in sorted(p for p in root.glob("*/*") if p.is_dir()):
        for split in ("train", "val"):
            df = pd.read_csv(d / f"{split}.csv")
            t = pd.DataFrame({
                "protein_name": df.Protein_id, "rna_sequence": df.RNA_seqs.str.upper().str.replace("T", "U"),
                "protein_sequence": df.Prot_seqs, "binding_label": df.labels.astype(int),
                "dataset_source": f"coral_{d.parent.name}", "data_type": "coral_repaired_negatives",
                "split": split, "fold": d.name, "rna_id": df.RNA_id})
            t.to_csv(out / f"{d.parent.name}__{d.name}__{split}.tsv", sep="\t", index=False)
            prot.update(zip(t.protein_name, t.protein_sequence))
            rna.update(zip(t.rna_id, t.rna_sequence))
            print(f"{d.parent.name}/{d.name}/{split}: {len(t)} rows, positives {t.binding_label.mean():.3f}", flush=True)
    mol = pd.concat([pd.DataFrame({"kind": "protein", "id": list(prot), "sequence": list(prot.values())}),
                     pd.DataFrame({"kind": "rna", "id": list(rna), "sequence": list(rna.values())})])
    mol.to_csv(out / "all_unique_molecules.tsv", sep="\t", index=False)
    print(f"unique proteins {len(prot)}, unique RNAs {len(rna)}")


if __name__ == "__main__":
    main()
