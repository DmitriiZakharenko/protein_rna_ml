#!/usr/bin/env python3
"""
57_family_inventory.py
----------------------
Which RNA-binding Pfam families do the candidate extra sources add compared with v3a
train? For every source in `data/sanitized/new_assays` that has protein sequences, run
hmmscan (via script 56) and report per-family protein counts and how many proteins have
no close v3a homolog (MMseqs2 identity to v3a train < 0.30 or no hit).

Usage:  python scripts/57_family_inventory.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fam56", ROOT / "scripts/56_domain_family_audit.py")
fam56 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fam56
spec.loader.exec_module(fam56)

SRC = ROOT / "data/sanitized/new_assays"
SOURCES = {"prbab": "prbab_affinity.tsv", "pronab": "pronab_prapred_affinity.tsv",
           "hughes_invivo": "hughes_invivo_pulldown.tsv"}
OUT = ROOT / "results/homology_audit"


def main() -> None:
    train = pd.read_csv(ROOT / "data/generalized_v3a/train.tsv", sep="\t", usecols=["protein_name", "protein_sequence"]).drop_duplicates("protein_name")
    train_fam = pd.read_csv(OUT / "v3a_pfam_family.tsv", sep="\t").set_index("protein_name").pfam_family
    rows, allp = [], {}
    for src, fn in SOURCES.items():
        df = pd.read_csv(SRC / fn, sep="\t", usecols=["protein_name", "protein_sequence"]).dropna()
        df["protein_sequence"] = df.protein_sequence.str.upper().str.replace(r"[^A-Z]", "X", regex=True)
        df = df[df.protein_sequence.str.len() >= 30].drop_duplicates("protein_sequence")
        for i, (n, s) in enumerate(zip(df.protein_name, df.protein_sequence)):
            key = f"{src}|{i}"
            allp[key] = s
            rows.append((key, src, n))
    meta = pd.DataFrame(rows, columns=["key", "source", "protein_name"])
    hits = fam56.run_hmmscan(allp, OUT / "new_assays_pfam_hits.tsv")
    meta["pfam_family"] = meta.key.map(lambda k: fam56.pfam_family(hits[k]))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "t.fa").write_text("".join(f">{n}\n{s}\n" for n, s in zip(train.protein_name, train.protein_sequence)))
        (td / "q.fa").write_text("".join(f">{k}\n{s}\n" for k, s in allp.items()))
        subprocess.run(["mmseqs", "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "h.m8"), str(td / "tmp"),
                        "-s", "7.5", "-e", "1e-3", "--format-output", "query,target,fident"], check=True, stdout=subprocess.DEVNULL)
        h = pd.read_csv(td / "h.m8", sep="\t", names=["query", "target", "fident"])
    best = h.groupby("query").fident.max()
    meta["best_identity"] = meta.key.map(best).fillna(0.0)
    meta["novel_lt30"] = meta.best_identity < 0.30
    meta.to_csv(OUT / "new_assays_family_inventory.tsv", sep="\t", index=False)

    print("v3a train proteins per family:\n", train_fam.reindex(train.protein_name).value_counts().to_string())
    print("\nproteins per source x family (all / novel <30% identity to v3a train):")
    tab = meta.pivot_table(index="pfam_family", columns="source", values="key", aggfunc="count", fill_value=0)
    nov = meta[meta.novel_lt30].pivot_table(index="pfam_family", columns="source", values="key", aggfunc="count", fill_value=0)
    print(tab.to_string())
    print("\nnovel (<30%) only:\n", nov.to_string())


if __name__ == "__main__":
    main()
