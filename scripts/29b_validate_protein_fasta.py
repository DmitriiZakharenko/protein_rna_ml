#!/usr/bin/env python3
"""
29b_validate_protein_fasta.py
-------------------------------
Pre-send validation for all_protocols_proteins.fasta.

Checks:
  - FASTA format and sequence hygiene (no *, digits, invalid AA)
  - Completeness vs all_protocols_summary.tsv scope (all hyb_id per panel)
  - Summary "best" experiment_id present for every protein
  - RNAcompete sequences match clean TSV (after sanitize)
  - RNAcompete optional match vs panel experiments.tsv metadata
  - Known regression cases (PUM, U2B'', SRP54, RBM38)
  - Cross-panel gene-name collision warnings

Usage:
    python scripts/29b_validate_protein_fasta.py
    python scripts/29b_validate_protein_fasta.py --strict --report_tsv results/.../fasta_validation_report.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.protein_sequence import sanitize_protein_sequence, validate_protein_sequence

RNAcompete_DATASETS = {
    "RNAcompete_Eukarya": ("eukarya/results/ml_dataset_eukarya_clean.tsv.gz", "eukarya/data/metadata/experiments.tsv"),
    "RNAcompete_RBPZoo": ("rbpzoo/results/ml_dataset_rbpzoo_clean.tsv.gz", "rbpzoo/data/metadata/experiments.tsv"),
    "RNAcompete_ucRBP23": ("ucrbp/results/ml_dataset_ucrbp_clean.tsv.gz", "ucrbp/data/metadata/experiments.tsv"),
}

KEY_CASES = [
    ("RNCMPT00046", "PUM", None),
    ("RNCMPT00101", "PUM", None),
    ("RNCMPT00102", "PUM", None),
    ("RNCMPT00103", "PUM", None),
    ("RNCMPT00104", "PUM", None),
    ("RNCMPT00105", "PUM", None),
    ("RNCMPT00799", "U2B''", 232),
    ("RNCMPT01660", "U2B''", 232),
    ("RNCMPT00272", "SRP54", 240),
    ("RNCMPT00051", "RBM38", None),
]

SRP54_PREFIX = "MVKDEPLSDES"


def normalize_protein_col(df: pd.DataFrame) -> pd.DataFrame:
    if "protein_name" not in df.columns and "target_name" in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})
    return df


def parse_fasta(path: Path) -> list[dict]:
    entries: list[dict] = []
    cur: dict | None = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                entries.append(cur)
            if "," not in line[1:]:
                raise ValueError(f"Malformed header (expected >id,name): {line}")
            eid, prot = line[1:].split(",", 1)
            cur = {"eid": eid.strip(), "prot": prot.strip(), "seq": ""}
        elif cur is not None:
            cur["seq"] += line.strip()
    if cur:
        entries.append(cur)
    return entries


def summary_scope(summary: pd.DataFrame) -> dict[str, set[str]]:
    rc = summary[summary["protocol"] == "rnacompete"]
    scope: dict[str, set[str]] = {}
    for ds, sub in rc.groupby("dataset", sort=False):
        scope[str(ds)] = set(sub["protein_name"].astype(str))
    return scope


def load_expected_keys_from_clean(clean_path: Path, proteins: set[str]) -> set[tuple[str, str]]:
    header = pd.read_csv(clean_path, sep="\t", nrows=0)
    pn = "protein_name" if "protein_name" in header.columns else "target_name"
    df = pd.read_csv(clean_path, sep="\t", usecols=[pn, "hyb_id"], low_memory=False)
    df = normalize_protein_col(df)
    sub = df[df["protein_name"].astype(str).isin(proteins)]
    return {(str(r.hyb_id), str(r.protein_name)) for r in sub.itertuples()}


def load_clean_seq_for_keys(path: Path, keys: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Load sanitized sequences from clean TSV for specific (hyb_id, protein) keys only."""
    if not keys:
        return {}
    header = pd.read_csv(path, sep="\t", nrows=0)
    pn = "protein_name" if "protein_name" in header.columns else "target_name"
    wanted_hyb = {k[0] for k in keys}
    out: dict[tuple[str, str], str] = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["hyb_id", pn, "protein_sequence"],
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = normalize_protein_col(chunk)
        sub = chunk[chunk["hyb_id"].astype(str).isin(wanted_hyb)]
        for (hid, prot), grp in sub.groupby([sub["hyb_id"].astype(str), "protein_name"], sort=False):
            key = (str(hid), str(prot))
            if key not in keys or key in out:
                continue
            seqs = grp["protein_sequence"].dropna().astype(str)
            if seqs.empty:
                continue
            cleaned, _ = sanitize_protein_sequence(seqs.iloc[0], source=f"{hid}:{prot}")
            if cleaned:
                out[key] = cleaned
        if len(out) == len(keys):
            break
    return out


def load_metadata_seq_by_hyb(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    id_col = "hyb_id" if "hyb_id" in df.columns else "experiment_id"
    seq_col = next((c for c in ("protein_sequence", "Construct_AA_seq", "construct_aa_seq") if c in df.columns), None)
    if not seq_col:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        hid = str(row[id_col])
        cleaned, _ = sanitize_protein_sequence(str(row[seq_col]), source=f"meta:{hid}")
        if cleaned:
            out[hid] = cleaned
    return out


def add_issue(issues: list[dict], level: str, check: str, eid: str, prot: str, detail: str) -> None:
    issues.append({"level": level, "check": check, "experiment_id": eid, "protein_name": prot, "detail": detail})


def validate_fasta(
    fasta_path: Path,
    summary_path: Path,
    rnacompete_root: Path,
    *,
    strict: bool = False,
) -> tuple[list[dict], dict]:
    entries = parse_fasta(fasta_path)
    summary = pd.read_csv(summary_path, sep="\t")
    scope = summary_scope(summary)
    issues: list[dict] = []

    fasta_map = {(e["eid"], e["prot"]): e["seq"] for e in entries}
    seen_headers: set[tuple[str, str]] = set()

    # --- per-entry hygiene ---
    for e in entries:
        eid, prot, seq = e["eid"], e["prot"], e["seq"]
        key = (eid, prot)
        if key in seen_headers:
            add_issue(issues, "FAIL", "duplicate_header", eid, prot, "duplicate FASTA header")
        seen_headers.add(key)

        if not seq:
            add_issue(issues, "FAIL", "empty_sequence", eid, prot, "empty sequence")
            continue
        if "*" in seq:
            add_issue(issues, "FAIL", "asterisk", eid, prot, "contains *")
        if any(c.isdigit() for c in seq):
            add_issue(issues, "FAIL", "digits", eid, prot, "contains digits")
        ok, reason = validate_protein_sequence(seq)
        if not ok:
            add_issue(issues, "FAIL", "invalid_aa", eid, prot, reason)
        if seq != seq.upper():
            add_issue(issues, "FAIL", "lowercase", eid, prot, "sequence not uppercase")

    # --- expected RNAcompete scope ---
    expected_rc: list[tuple[str, str, str]] = []
    clean_maps: dict[str, dict[tuple[str, str], str]] = {}
    meta_maps: dict[str, dict[str, str]] = {}
    keys_by_ds: dict[str, set[tuple[str, str]]] = {}

    for ds, (clean_rel, meta_rel) in RNAcompete_DATASETS.items():
        meta_path = rnacompete_root / meta_rel
        proteins = scope.get(ds, set())
        clean_path = rnacompete_root / clean_rel
        if not clean_path.exists():
            add_issue(issues, "WARN", "missing_clean", "", ds, str(clean_path))
            panel_keys: set[tuple[str, str]] = set()
        else:
            panel_keys = load_expected_keys_from_clean(clean_path, proteins)
        keys_by_ds[ds] = panel_keys
        for hid, prot in panel_keys:
            expected_rc.append((hid, prot, ds))

        if clean_path.exists():
            clean_maps[ds] = load_clean_seq_for_keys(clean_path, panel_keys)
        meta_maps[ds] = load_metadata_seq_by_hyb(meta_path)

    for hid, prot, ds in expected_rc:
        if (hid, prot) not in fasta_map:
            add_issue(issues, "FAIL", "missing_expected", hid, prot, f"expected from {ds}")

    # --- summary best IDs ---
    for _, row in summary[summary["protocol"] == "rnacompete"].drop_duplicates(["dataset", "protein_name"]).iterrows():
        eid, prot = str(row["experiment_id"]), str(row["protein_name"])
        if (eid, prot) not in fasta_map:
            add_issue(issues, "FAIL", "missing_summary_best", eid, prot, str(row["dataset"]))

    for _, row in summary[summary["protocol"].isin(["rbns", "htr_selex"])].drop_duplicates(["experiment_id", "protein_name"]).iterrows():
        eid, prot = str(row["experiment_id"]), str(row["protein_name"])
        if (eid, prot) not in fasta_map:
            add_issue(issues, "FAIL", "missing_rbns_htr", eid, prot, str(row["protocol"]))

    # --- sequence match clean TSV ---
    ds_by_key = {(hid, prot): ds for hid, prot, ds in expected_rc}
    for (hid, prot), seq in fasta_map.items():
        if not hid.startswith("RNCMPT"):
            continue
        ds = ds_by_key.get((hid, prot))
        if ds is None:
            add_issue(issues, "WARN", "extra_rnacompete", hid, prot, "RNCMPT not in expected scope")
            continue
        ref = clean_maps.get(ds, {}).get((hid, prot))
        if ref is None:
            add_issue(issues, "FAIL", "no_clean_ref", hid, prot, f"no sequence in {ds} clean TSV")
        elif ref != seq:
            add_issue(issues, "FAIL", "clean_mismatch", hid, prot, f"FASTA len={len(seq)} clean len={len(ref)}")

    # --- metadata match (strict) ---
    if strict:
        for hid, prot, ds in expected_rc:
            seq = fasta_map.get((hid, prot))
            meta = meta_maps.get(ds, {}).get(hid)
            if meta is None:
                add_issue(issues, "WARN", "no_metadata", hid, prot, f"not in {ds} experiments.tsv")
            elif seq and meta != seq:
                add_issue(issues, "FAIL", "metadata_mismatch", hid, prot, f"FASTA != experiments.tsv")

    # --- key regression cases ---
    for eid, prot, exp_len in KEY_CASES:
        seq = fasta_map.get((eid, prot))
        if seq is None:
            add_issue(issues, "FAIL", "key_case_missing", eid, prot, "known regression case missing")
            continue
        if eid == "RNCMPT00272" and not seq.startswith(SRP54_PREFIX):
            add_issue(issues, "FAIL", "key_case_srp54", eid, prot, f"expected fly construct prefix, len={len(seq)}")
        if eid == "RNCMPT00051" and any(c.isdigit() for c in seq):
            add_issue(issues, "FAIL", "key_case_rbm38", eid, prot, "digits in RBM38")
        if exp_len is not None and len(seq) != exp_len:
            add_issue(issues, "FAIL", "key_case_length", eid, prot, f"expected len={exp_len}, got {len(seq)}")

    # --- cross-panel same name warning ---
    by_name: dict[str, list[tuple[str, str]]] = {}
    for (eid, prot), seq in fasta_map.items():
        if eid.startswith("RNCMPT"):
            by_name.setdefault(prot, []).append((eid, seq))
    for prot, items in by_name.items():
        seqs = {s for _, s in items}
        if len(seqs) > 1 and len(items) > 1:
            add_issue(
                issues,
                "INFO",
                "multi_construct",
                "",
                prot,
                f"{len(items)} hyb_ids, {len(seqs)} distinct sequences",
            )

    n_fail = sum(1 for i in issues if i["level"] == "FAIL")
    n_warn = sum(1 for i in issues if i["level"] == "WARN")
    stats = {
        "fasta_entries": len(entries),
        "rnacompete_entries": sum(1 for e in entries if e["eid"].startswith("RNCMPT")),
        "expected_rnacompete": len(expected_rc),
        "fail": n_fail,
        "warn": n_warn,
        "info": sum(1 for i in issues if i["level"] == "INFO"),
        "pass": n_fail == 0,
    }
    return issues, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Validate all_protocols_proteins.fasta before delivery")
    p.add_argument("--fasta", default="results/top_bottom_examples/all_protocols_proteins.fasta")
    p.add_argument("--summary_tsv", default="results/top_bottom_examples/all_protocols_summary.tsv")
    p.add_argument("--rnacompete_root", default="../rnacompete_analysis")
    p.add_argument("--strict", action="store_true", help="Also require match to experiments.tsv metadata")
    p.add_argument("--report_tsv", default=None)
    p.add_argument("--report_json", default=None)
    args = p.parse_args()

    issues, stats = validate_fasta(
        Path(args.fasta),
        Path(args.summary_tsv),
        Path(args.rnacompete_root),
        strict=args.strict,
    )

    print(f"FASTA validation: {args.fasta}")
    print(f"  entries: {stats['fasta_entries']} ({stats['rnacompete_entries']} RNCMPT)")
    print(f"  expected RNCMPT (summary scope): {stats['expected_rnacompete']}")
    print(f"  FAIL: {stats['fail']}  WARN: {stats['warn']}  INFO: {stats['info']}")

    fails = [i for i in issues if i["level"] == "FAIL"]
    warns = [i for i in issues if i["level"] == "WARN"]
    if fails:
        print("\nFAILURES:")
        for i in fails[:30]:
            print(f"  [{i['check']}] {i['experiment_id']},{i['protein_name']}: {i['detail']}")
        if len(fails) > 30:
            print(f"  ... and {len(fails) - 30} more")
    if warns:
        print("\nWARNINGS:")
        for i in warns[:15]:
            print(f"  [{i['check']}] {i['experiment_id']},{i['protein_name']}: {i['detail']}")

    if args.report_tsv:
        pd.DataFrame(issues).to_csv(args.report_tsv, sep="\t", index=False)
        print(f"\nReport: {args.report_tsv}")
    if args.report_json:
        Path(args.report_json).write_text(json.dumps({"stats": stats, "issues": issues}, indent=2) + "\n")
        print(f"JSON: {args.report_json}")

    if stats["pass"]:
        print("\nPASS — safe to send.")
    else:
        print("\nFAIL — do not send until fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
