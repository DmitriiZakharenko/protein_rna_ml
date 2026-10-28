#!/usr/bin/env python3
"""
53_build_invitro_cross_protein_probe.py
---------------------------------------
In vitro analogue of the eCLIP ``cross_protein_neg`` benchmark, built from the
v3a test split.

For each positive (protein A, RNA r) of a test protein, the negative is the same
RNA r paired with a *different* test protein B from the same ``dataset_source``
for which r is not a positive. The RNA marginals of positives and negatives are
therefore identical, so a model can only score above 0.5 by learning the
protein x RNA interaction (not "protein -> RNA type" or RNA composition).

Writes ``<out_dir>/cross_neg_probe/{train,test}.tsv`` in the format expected by
``scripts/51_eclip_eval_suite.py`` (train.tsv is a symlink to the v3a train set,
used only for the k-mer baseline and RNA-leakage flag).

Usage
-----
  python scripts/53_build_invitro_cross_protein_probe.py \\
      --v3a_dir data/sanitized/generalized_v3a --out_dir data/benchmarks/v3a_cross_neg
  python scripts/51_eclip_eval_suite.py --bench_dir data/benchmarks/v3a_cross_neg \\
      --splits cross_neg_probe --max_train_rows 200000 --model ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3a_dir", default="data/sanitized/generalized_v3a")
    ap.add_argument("--out_dir", default="data/benchmarks/v3a_cross_neg")
    ap.add_argument("--max_pos", type=int, default=40000, help="Cap on positives (random sample)")
    ap.add_argument("--max_tries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v3a = resolve(args.v3a_dir)
    out = resolve(args.out_dir) / "cross_neg_probe"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    test = pd.read_csv(v3a / "test.tsv", sep="\t", low_memory=False)
    pos = test[test["binding_label"] == 1].drop_duplicates(["protein_name", "rna_sequence"])
    if len(pos) > args.max_pos:
        pos = pos.sample(args.max_pos, random_state=args.seed)
    pos = pos.reset_index(drop=True)

    prot_seq = test.groupby("protein_name")["protein_sequence"].first().to_dict()
    pos_set = set(zip(test.loc[test["binding_label"] == 1, "protein_name"],
                      test.loc[test["binding_label"] == 1, "rna_sequence"]))
    prots_by_source = {s: sorted(g["protein_name"].unique()) for s, g in test.groupby("dataset_source")}

    neg_rows, kept = [], []
    for i, r in pos.iterrows():
        cands = [p for p in prots_by_source[r["dataset_source"]] if p != r["protein_name"]]
        if not cands:
            continue
        for _ in range(args.max_tries):
            b = cands[int(rng.integers(len(cands)))]
            if (b, r["rna_sequence"]) not in pos_set:
                neg_rows.append({"protein_name": b, "rna_sequence": r["rna_sequence"],
                                 "protein_sequence": prot_seq[b], "binding_label": 0,
                                 "dataset_source": r["dataset_source"], "organism": r.get("organism")})
                kept.append(i)
                break
    pos = pos.loc[kept]
    df = pd.concat([pos[["protein_name", "rna_sequence", "protein_sequence", "binding_label",
                         "dataset_source", "organism"]], pd.DataFrame(neg_rows)], ignore_index=True)
    df.to_csv(out / "test.tsv", sep="\t", index=False)

    link = out / "train.tsv"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink((v3a / "train.tsv").resolve(), link)

    summary = {
        "n_pos": int(len(pos)), "n_neg": int(len(neg_rows)),
        "n_proteins": int(df["protein_name"].nunique()),
        "by_source": df.groupby("dataset_source")["binding_label"].agg(["size", "mean"]).round(3).to_dict("index"),
        "out_dir": str(out),
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
