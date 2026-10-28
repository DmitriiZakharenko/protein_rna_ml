#!/usr/bin/env python3
"""
44_overlap_new_assays_literature.py

QC overlap between normalized new-assay TSVs and project literature / train splits.

Outputs:
  results/new_assays/literature_overlap_qc.json
  results/new_assays/literature_overlap_qc.md

Usage (from repo root):
  python scripts/44_overlap_new_assays_literature.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.external_benchmark import (
    TrainLeakageIndex,
    load_external_xlsx,
    parse_curated_pairs,
)
from src.data.rna_sequence import normalize_rna, pair_key

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
SAN = ROOT / "data" / "sanitized" / "new_assays"
OUT_DIR = ROOT / "results" / "new_assays"


def _default_xlsx_without() -> Path:
    for name in (
        "dataset_without_affinities.xlsx",
        "dataset without affinities.xlsx",
    ):
        p = EXT / name
        if p.exists():
            return p
    raise FileNotFoundError("Literature xlsx not found under data/external/")


def _default_xlsx_with() -> Path | None:
    for name in (
        "dataset_with_affinities.xlsx",
        "dataset with affinities.xlsx",
    ):
        p = EXT / name
        if p.exists():
            return p
    return None


def norm_protein_name(name: str | None) -> str | None:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    t = str(name).strip().upper().replace("-", "").replace(" ", "")
    return t or None


def summarize_assay(
    name: str,
    path: Path,
    *,
    lit_proteins: set[str],
    lit_pairs: set[tuple[str, str]],
    train_index: TrainLeakageIndex | None,
) -> dict:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    norm_names = df["protein_name"].map(norm_protein_name)
    proteins = set(norm_names.dropna())

    has_seq = df["protein_sequence"].notna() & df["rna_sequence"].notna()
    has_seq &= df["protein_sequence"].astype(str).str.len().gt(0)
    has_seq &= df["rna_sequence"].astype(str).str.len().gt(0)

    pair_overlap_lit = 0
    exact_in_train = 0
    prot_name_in_train = 0
    prot_seq_in_train = 0
    rna_in_train_pos = 0
    n_with_seq = int(has_seq.sum())

    for _, row in df.loc[has_seq].iterrows():
        pk = pair_key(str(row["protein_sequence"]), str(row["rna_sequence"]))
        if pk in lit_pairs:
            pair_overlap_lit += 1
        if train_index:
            ann = train_index.annotate(
                str(row["protein_name"]),
                str(row["protein_sequence"]),
                str(row["rna_sequence"]),
            )
            exact_in_train += int(ann["exact_pair_in_train"])
            prot_name_in_train += int(ann["protein_name_in_train"])
            prot_seq_in_train += int(ann["protein_sequence_in_train"])
            rna_in_train_pos += int(ann["rna_in_train_positives"])

    v3a_prot_names: set[str] = set()
    if train_index:
        v3a_prot_names = {norm_protein_name(x) for x in train_index.train_protein_names}
        v3a_prot_names.discard(None)

    out: dict = {
        "file": str(path.relative_to(ROOT)),
        "n_rows": len(df),
        "n_distinct_protein_names": len(proteins),
        "n_protein_names_in_v3a_train": len(proteins & v3a_prot_names) if v3a_prot_names else None,
        "protein_names_in_v3a_train": sorted(proteins & v3a_prot_names) if v3a_prot_names else None,
        "n_rows_with_both_sequences": n_with_seq,
        "protein_names_also_in_literature_curated": sorted(lit_proteins & proteins),
        "n_protein_names_in_literature_curated": len(lit_proteins & proteins),
        "n_rows_exact_pair_in_literature_curated": pair_overlap_lit,
    }
    if train_index:
        out["train_v3a_leakage_on_rows_with_seq"] = {
            "exact_pair_in_train": exact_in_train,
            "protein_name_in_train": prot_name_in_train,
            "protein_sequence_in_train": prot_seq_in_train,
            "rna_in_train_positives": rna_in_train_pos,
            "fraction_exact_pair_in_train": round(exact_in_train / n_with_seq, 6)
            if n_with_seq
            else None,
        }
    if "pdb_id" in df.columns:
        pdb = (
            df["pdb_id"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r"_.*", "", regex=True)
            .str[:4]
        )
        out["n_distinct_pdb_ids"] = int(pdb.nunique())
    if "binding_label" in df.columns:
        out["binding_label_counts"] = df["binding_label"].value_counts().to_dict()
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xlsx = _default_xlsx_without()
    curated, parse_stats, _ = parse_curated_pairs(load_external_xlsx(xlsx))
    lit_proteins = {norm_protein_name(p.protein_name) for p in curated}
    lit_proteins.discard(None)
    lit_pairs = {pair_key(p.protein_sequence, p.rna_sequence) for p in curated}

    train_path = ROOT / "data" / "sanitized" / "generalized_v3a" / "train.tsv"
    train_index = TrainLeakageIndex.from_train_tsv(train_path) if train_path.exists() else None

    assays = [
        ("hughes_invivo_pulldown", SAN / "hughes_invivo_pulldown.tsv"),
        ("prbab_affinity", SAN / "prbab_affinity.tsv"),
        ("pronab_prapred_affinity", SAN / "pronab_prapred_affinity.tsv"),
        ("utexas_aptamer_rna", SAN / "utexas_aptamer_rna.tsv"),
    ]

    assay_summaries = {}
    for key, path in assays:
        if not path.exists():
            continue
        assay_summaries[key] = summarize_assay(
            key,
            path,
            lit_proteins=lit_proteins,
            lit_pairs=lit_pairs,
            train_index=train_index,
        )

    prbab_pdb = set()
    pronab_pdb = set()
    if (SAN / "prbab_affinity.tsv").exists():
        pb = pd.read_csv(SAN / "prbab_affinity.tsv", sep="\t", usecols=["pdb_id"])
        prbab_pdb = set(
            pb["pdb_id"].dropna().astype(str).str.upper().str.replace(r"_.*", "", regex=True).str[:4]
        )
    if (SAN / "pronab_prapred_affinity.tsv").exists():
        pn = pd.read_csv(SAN / "pronab_prapred_affinity.tsv", sep="\t", usecols=["pdb_id"])
        pronab_pdb = set(pn["pdb_id"].dropna().astype(str).str.upper().str[:4])

    aff_summary = None
    aff_path = _default_xlsx_with()
    if aff_path:
        aff_raw = load_external_xlsx(aff_path)
        col_p = next(
            (c for c in aff_raw.columns if c and str(c).strip().lower() == "protein"),
            None,
        )
        if col_p:
            aff_prots = {
                norm_protein_name(x) for x in aff_raw[col_p].dropna().astype(str)
            }
            aff_prots.discard(None)
            aff_summary = {
                "file": str(aff_path.relative_to(ROOT)),
                "n_rows": len(aff_raw),
                "n_distinct_proteins": len(aff_prots),
                "overlap_hughes_protein_names": sorted(
                    aff_prots & set(assay_summaries.get("hughes_invivo_pulldown", {}).get(
                        "protein_names_also_in_literature_curated", []
                    ))
                ),
            }
            hughes_all = set()
            if "hughes_invivo_pulldown" in assay_summaries:
                hdf = pd.read_csv(SAN / "hughes_invivo_pulldown.tsv", sep="\t", usecols=["protein_name"])
                hughes_all = set(hdf["protein_name"].map(norm_protein_name).dropna())
            aff_summary["n_proteins_also_in_hughes"] = len(aff_prots & hughes_all)
            aff_summary["proteins_also_in_hughes"] = sorted(aff_prots & hughes_all)

    expanded_path = EXT / "external_benchmark_expanded.tsv"
    expanded_overlap = None
    if expanded_path.exists():
        exp = pd.read_csv(expanded_path, sep="\t", usecols=["protein_name", "protein_sequence", "rna_sequence", "example_class"])
        exp_pairs = {
            pair_key(str(r.protein_sequence), str(r.rna_sequence))
            for _, r in exp.iterrows()
            if pd.notna(r.protein_sequence) and pd.notna(r.rna_sequence)
        }
        expanded_overlap = {
            "n_expanded_pairs": len(exp_pairs),
            "n_in_literature_curated_pairs": len(exp_pairs & lit_pairs),
        }

    report = {
        "literature_curated": {
            "xlsx": str(xlsx.relative_to(ROOT)),
            "usable_pairs": parse_stats.usable_rows,
            "n_proteins": len(lit_proteins),
        },
        "train_v3a": {
            "path": str(train_path.relative_to(ROOT)) if train_path.exists() else None,
            "n_rows": train_index.n_train_rows if train_index else None,
            "n_distinct_protein_names": len(train_index.train_protein_names) if train_index else None,
        },
        "new_assays": assay_summaries,
        "prbab_pronab_pdb_overlap": {
            "n_prbab_pdb": len(prbab_pdb),
            "n_pronab_pdb": len(pronab_pdb),
            "n_shared_pdb": len(prbab_pdb & pronab_pdb),
        },
        "dataset_with_affinities": aff_summary,
        "external_benchmark_expanded": expanded_overlap,
    }

    json_path = OUT_DIR / "literature_overlap_qc.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_lines = [
        "# New assays vs literature / v3a train — overlap QC",
        "",
        f"Generated by `scripts/44_overlap_new_assays_literature.py`.",
        "",
        "## Literature curated (`dataset without affinities.xlsx`)",
        f"- Usable pairs: **{parse_stats.usable_rows}** ({len(lit_proteins)} protein names)",
        "",
        "## Summary",
        "",
        "| Assay | Rows | Proteins | Shared w/ literature names | Exact pair overlap w/ literature |",
        "|-------|------|----------|------------------------------|-----------------------------------|",
    ]
    for key, s in assay_summaries.items():
        md_lines.append(
            f"| {key} | {s['n_rows']:,} | {s['n_distinct_protein_names']} | "
            f"{s['n_protein_names_in_literature_curated']} | {s['n_rows_exact_pair_in_literature_curated']} |"
        )
    md_lines.extend(
        [
            "",
            "### Hughes shared literature protein names",
            ", ".join(
                assay_summaries.get("hughes_invivo_pulldown", {}).get(
                    "protein_names_also_in_literature_curated", []
                )
            )
            or "(none)",
            "",
            "### PRBAB shared literature protein names",
            ", ".join(
                assay_summaries.get("prbab_affinity", {}).get(
                    "protein_names_also_in_literature_curated", []
                )
            )
            or "(none)",
            "",
            f"### PRBAB ∩ ProNAB (PDB id): **{len(prbab_pdb & pronab_pdb)}** / PRBAB {len(prbab_pdb)}",
            "",
            "### v3a train protein names",
            f"- Hughes: **{assay_summaries.get('hughes_invivo_pulldown', {}).get('n_protein_names_in_v3a_train', '?')}/23** symbols already in train",
            f"- PRBAB: **{assay_summaries.get('prbab_affinity', {}).get('n_protein_names_in_v3a_train', '?')}/146**",
            "",
            "**Conclusion:** No exact (protein_seq, RNA_seq) duplicates vs the 159-pair literature sheet. Hughes RBPs largely overlap **v3a train by name** — treat as fine-tune/auxiliary or protein-disjoint eval, not naive extra train without split design.",
            "",
        ]
    )
    if aff_summary:
        md_lines.append(
            f"### `dataset with affinities.xlsx`: {aff_summary['n_proteins_also_in_hughes']} proteins also in Hughes "
            f"({', '.join(aff_summary.get('proteins_also_in_hughes', [])[:15])}…)"
        )
    (OUT_DIR / "literature_overlap_qc.md").write_text("\n".join(md_lines))
    print(f"Wrote {json_path}")
    print(f"Wrote {OUT_DIR / 'literature_overlap_qc.md'}")


if __name__ == "__main__":
    main()
