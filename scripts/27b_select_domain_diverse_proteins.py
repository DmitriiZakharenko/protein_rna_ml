#!/usr/bin/env python3
"""
27b_select_domain_diverse_proteins.py
-------------------------------------
Pick one (protein, hyb_id) pair per unique Table S1 "Domains in construct" architecture
within an RNAcompete panel.

Method
------
1. Load Table S1 (Sasse et al. 2025 supplementary; all RNAcompete experiments).
2. Explode semicolon-separated RNAcompete ID(s) → join to panel clean TSV via hyb_id.
3. Domain architecture is assigned **per hyb_id** (not collapsed across a gene name).
   The full Table S1 string is kept verbatim, e.g. ``RRM;RRM;KH;KH;KH;KH``.
4. Within each unique domain_architecture, keep the (protein_name, hyb_id) with the highest
   mean positive probe_intensity on that hyb_id (same metric as script 27).
5. Optional ``--include_proteins``: always add named proteins (e.g. showcase examples)
   even if they did not win their architecture bucket.
6. Writes ``domain_conflicts_{panel}.tsv`` for gene names whose hyb_ids carry >1 domain
   string in Table S1 (informational; does not block selection at hyb_id level).

Usage:
    python scripts/27b_select_domain_diverse_proteins.py \\
        --table_s1 data/raw/rbpzoo/TableS1.xlsx \\
        --data_file ../rnacompete_analysis/eukarya/results/ml_dataset_eukarya_clean.tsv.gz \\
        --dataset rnacompete_eukarya \\
        --output results/rnacompete_intensity_spectrum/domain_diverse_selection_eukarya.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_table_s1(path: Path) -> pd.DataFrame:
    ts1 = pd.read_excel(path)
    rows: list[dict] = []
    for _, r in ts1.iterrows():
        dom = str(r["Domains in construct"]).strip() if pd.notna(r["Domains in construct"]) else ""
        rncmpt_raw = str(r["RNAcompete ID(s)"]).strip() if pd.notna(r["RNAcompete ID(s)"]) else ""
        for rid in (x.strip() for x in rncmpt_raw.split(";") if x.strip()):
            rows.append(
                {
                    "hyb_id": rid,
                    "table_protein_name": str(r["Protein name"]).strip() if pd.notna(r["Protein name"]) else "",
                    "domain_architecture": dom,
                    "species": r.get("Species"),
                    "construct_id": r.get("Construct ID"),
                }
            )
    return pd.DataFrame(rows)


def mean_positive_intensity_hyb(df: pd.DataFrame, hyb_id: str, protein: str) -> float:
    sub = df[(df["hyb_id"].astype(str) == str(hyb_id)) & (df["target_name"] == protein)]
    pos = sub[sub["binding_label"] == 1]
    if pos.empty:
        return float("nan")
    return float(pd.to_numeric(pos["probe_intensity"], errors="coerce").mean())


def protein_domain_conflicts(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for prot, g in joined.groupby("target_name", sort=False):
        doms = sorted({d for d in g["domain_architecture"].dropna().astype(str) if d and d != "nan"})
        if len(doms) > 1:
            rows.append(
                {
                    "protein_name": prot,
                    "domain_values": "|".join(doms),
                    "hyb_ids": ";".join(sorted(g["hyb_id"].astype(str).unique())),
                }
            )
    return pd.DataFrame(rows)


def select_domain_diverse(
    df: pd.DataFrame,
    tmap: pd.DataFrame,
    include_proteins: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_hybs = set(df["hyb_id"].astype(str))
    missing = panel_hybs - set(tmap["hyb_id"])
    if missing:
        print(f"  WARNING: {len(missing)} panel hyb_id(s) absent from Table S1: {sorted(missing)}")

    joined = df[["hyb_id", "target_name"]].drop_duplicates().merge(tmap, on="hyb_id", how="left")
    no_dom = joined[joined["domain_architecture"].isna() | (joined["domain_architecture"] == "")]
    if not no_dom.empty:
        print(f"  WARNING: {no_dom['hyb_id'].nunique()} hyb_id(s) without domain in Table S1")

    conflicts = protein_domain_conflicts(joined)
    if not conflicts.empty:
        print(f"  NOTE: {len(conflicts)} protein(s) with multiple domain strings across hyb_ids")

    records: list[dict] = []
    for _, row in joined.iterrows():
        dom = str(row["domain_architecture"]).strip()
        if not dom or dom == "nan":
            continue
        hyb_id = str(row["hyb_id"])
        prot = row["target_name"]
        records.append(
            {
                "protein_name": prot,
                "hyb_id": hyb_id,
                "domain_architecture": dom,
                "mean_positive_intensity": mean_positive_intensity_hyb(df, hyb_id, prot),
            }
        )

    hyb_df = pd.DataFrame(records)
    selected_rows: list[dict] = []
    for arch, g in hyb_df.groupby("domain_architecture", sort=False):
        best = g.sort_values("mean_positive_intensity", ascending=False).iloc[0]
        selected_rows.append(best.to_dict())

    selected = pd.DataFrame(selected_rows).sort_values("domain_architecture").reset_index(drop=True)

    if include_proteins:
        extra_rows: list[dict] = []
        for prot in include_proteins:
            if prot in set(selected["protein_name"]):
                continue
            cand = hyb_df[hyb_df["protein_name"] == prot].sort_values(
                "mean_positive_intensity", ascending=False
            )
            if cand.empty:
                print(f"  WARNING: --include_proteins {prot!r} not found in panel")
                continue
            extra_rows.append(cand.iloc[0].to_dict())
        if extra_rows:
            selected = (
                pd.concat([selected, pd.DataFrame(extra_rows)], ignore_index=True)
                .sort_values("domain_architecture")
                .reset_index(drop=True)
            )

    return selected, conflicts


def main() -> None:
    p = argparse.ArgumentParser(description="Select one RBP per Table S1 domain architecture")
    p.add_argument("--table_s1", required=True, help="Table S1.xlsx path")
    p.add_argument("--data_file", required=True, help="RNAcompete clean TSV[.gz]")
    p.add_argument("--dataset", required=True, help="Dataset label (for logging)")
    p.add_argument("--output", required=True, help="Output TSV with selected proteins")
    p.add_argument(
        "--conflicts_output",
        default=None,
        help="Optional TSV: proteins whose hyb_ids map to multiple domain strings",
    )
    p.add_argument(
        "--include_proteins",
        nargs="*",
        default=None,
        help="Always include these protein names (e.g. showcase RBPs)",
    )
    args = p.parse_args()

    table_path = Path(args.table_s1)
    data_path = Path(args.data_file)
    if not table_path.exists():
        sys.exit(f"ERROR: Table S1 not found: {table_path}")
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    print(f"Loading Table S1: {table_path}")
    tmap = load_table_s1(table_path)
    print(f"  {tmap['hyb_id'].nunique()} hyb_ids in Table S1")

    print(f"Loading panel: {data_path}")
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    if "target_name" in df.columns and "protein_name" not in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})
    if "protein_name" not in df.columns:
        sys.exit("ERROR: need protein_name or target_name column")
    df = df.rename(columns={"protein_name": "target_name"})
    df["binding_label"] = pd.to_numeric(df["binding_label"], errors="coerce").astype(int)
    print(f"  {df['target_name'].nunique()} proteins, {df['hyb_id'].nunique()} hyb_ids")

    print(f"Selecting domain-diverse proteins for {args.dataset} ...")
    selected, conflicts = select_domain_diverse(df, tmap, args.include_proteins)
    print(f"  Selected: {len(selected)} architectures")
    if not conflicts.empty:
        print(f"  Domain conflicts (informational): {len(conflicts)} proteins")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, sep="\t", index=False)
    print(f"Wrote: {out}")

    if args.conflicts_output:
        co = Path(args.conflicts_output)
        co.parent.mkdir(parents=True, exist_ok=True)
        conflicts.to_csv(co, sep="\t", index=False)
        print(f"Wrote: {co}")


if __name__ == "__main__":
    main()
