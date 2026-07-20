#!/usr/bin/env python3
"""
40_sanitize_training_protein_sequences.py
-----------------------------------------
Audit and optionally repair protein_sequence columns in training TSVs.

Known issues found in this project:
  - Embedded residue numbers inside sequences (e.g. RBM38: ...EE708090100110120AVV...)
  - Trailing stop codon '*' (many ucRBP / HEXIM entries)
  - Occasional non-amino-acid characters

Default is audit-only. Pass --apply to write sanitized copies (or inplace with backup).

Usage:
    # Audit only
    python scripts/40_sanitize_training_protein_sequences.py

    # Write sanitized TSVs next to originals as *.sanitized.tsv
    python scripts/40_sanitize_training_protein_sequences.py --apply

    # In-place with .bak
    python scripts/40_sanitize_training_protein_sequences.py --apply --inplace
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.protein_sequence import sanitize_protein_sequence, validate_protein_sequence

DEFAULT_TARGETS = [
    "data/generalized_v2/train.tsv",
    "data/generalized_v2/val.tsv",
    "data/generalized_v2/test.tsv",
    "data/generalized_v3a/train.tsv",
    "data/generalized_v3a/val.tsv",
    "data/generalized_v3a/test.tsv",
    "data/eclip/eclip_all.tsv",
]


def audit_unique_sequences(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, sep="\t", usecols=["protein_name", "protein_sequence"])
    uniq = (
        df.dropna(subset=["protein_sequence"])
        .drop_duplicates(["protein_name", "protein_sequence"])
        .copy()
    )
    rows = []
    n_digit = n_star = n_other = n_changed = 0
    for _, r in uniq.iterrows():
        raw = str(r["protein_sequence"])
        cleaned, warnings = sanitize_protein_sequence(
            raw, source=f"{r['protein_name']}"
        )
        ok, reason = validate_protein_sequence(cleaned)
        changed = cleaned != raw.rstrip() and cleaned != raw
        # count categories from warnings / raw
        has_digit = any(c.isdigit() for c in raw)
        has_star = "*" in raw
        if has_digit:
            n_digit += 1
        if has_star:
            n_star += 1
        if any("non_amino" in w for w in warnings):
            n_other += 1
        if cleaned != raw.upper().rstrip("*") and has_digit:
            changed = True
        if len(cleaned) != len(raw.upper().rstrip("*")) or has_digit or has_star:
            n_changed += 1
        if has_digit or has_star or (not ok) or warnings:
            rows.append(
                {
                    "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "protein_name": r["protein_name"],
                    "raw_length": len(raw),
                    "clean_length": len(cleaned),
                    "has_digits": int(has_digit),
                    "has_star": int(has_star),
                    "valid_after": int(ok),
                    "reason": reason,
                    "warnings": ";".join(warnings),
                    "raw_preview": raw[:120],
                    "clean_preview": cleaned[:120],
                }
            )
    report = pd.DataFrame(rows)
    summary = {
        "file": str(path),
        "n_rows": int(len(df)),
        "n_unique_protein_seq": int(len(uniq)),
        "n_flagged_unique": int(len(report)),
        "n_unique_with_digits": n_digit,
        "n_unique_with_star": n_star,
        "n_unique_with_other": n_other,
    }
    return report, summary


def apply_sanitize(path: Path, inplace: bool, out_dir: Path | None) -> Path:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "protein_sequence" not in df.columns:
        raise ValueError(f"No protein_sequence in {path}")
    cleaned_list = []
    warn_count = 0
    for seq in df["protein_sequence"].fillna("").astype(str):
        cleaned, warnings = sanitize_protein_sequence(seq)
        if warnings:
            warn_count += 1
        cleaned_list.append(cleaned)
    df["protein_sequence"] = cleaned_list
    # drop rows that became empty
    n_empty = int((df["protein_sequence"].astype(str).str.len() == 0).sum())
    if n_empty:
        df = df[df["protein_sequence"].astype(str).str.len() > 0].copy()

    if inplace:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        df.to_csv(path, sep="\t", index=False)
        print(f"  inplace write {path} (backup {bak.name}); rows_with_warnings≈{warn_count}, dropped_empty={n_empty}")
        return path

    if out_dir is None:
        out_path = path.with_name(path.stem + ".sanitized.tsv")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  wrote {out_path} (warnings_rows≈{warn_count}, dropped_empty={n_empty})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--out_dir", default="data/sanitized")
    ap.add_argument("--report_dir", default="results/data_qc")
    args = ap.parse_args()

    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []
    summaries = []
    for rel in args.targets:
        path = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        print(f"Auditing {path} ...")
        report, summary = audit_unique_sequences(path)
        summaries.append(summary)
        if not report.empty:
            all_reports.append(report)
        print(
            f"  unique flagged={summary['n_flagged_unique']} "
            f"(digits={summary['n_unique_with_digits']}, star={summary['n_unique_with_star']})"
        )
        if args.apply:
            apply_sanitize(
                path,
                inplace=args.inplace,
                out_dir=None if args.inplace else ROOT / args.out_dir / path.parent.name,
            )

    if all_reports:
        full = pd.concat(all_reports, ignore_index=True)
        full_path = report_dir / "protein_sequence_contamination_report.tsv"
        full.to_csv(full_path, sep="\t", index=False)
        print(f"Wrote {full_path}")
    else:
        print("No contaminated unique protein sequences found.")

    summary_path = report_dir / "protein_sequence_contamination_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "targets": summaries,
                "apply": bool(args.apply),
                "inplace": bool(args.inplace),
                "retrain_required_if_applied": [
                    "scripts/06_train_generalized_v2.py on cleaned generalized_v2",
                    "scripts/06_train_generalized_v2.py on cleaned generalized_v3a",
                    "external eval scripts/11",
                    "any ESM embedding extraction on these proteins",
                ],
            },
            f,
            indent=2,
        )
    print(f"Wrote {summary_path}")
    if not args.apply:
        print(
            "\nAudit only. To repair: "
            "python scripts/40_sanitize_training_protein_sequences.py --apply"
            "  OR  --apply --inplace"
        )


if __name__ == "__main__":
    main()
