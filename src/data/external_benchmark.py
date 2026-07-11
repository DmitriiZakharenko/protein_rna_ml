"""
Load and expand the curated literature protein–RNA external benchmark.

The primary source is ``dataset_without_affinities.xlsx`` (159 usable pairs after
label/sequence filtering). This module normalises parsing logic shared with
``scripts/11_evaluate_external.py`` and adds train-set leakage annotation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.negative_sampling import (
    CuratedPair,
    GeneratedNegative,
    NegativeSamplingConfig,
    generate_all_negatives_for_anchor,
)
from src.data.protein_sequence import sanitize_protein_sequence
from src.data.rna_sequence import gc_content, normalize_rna, pair_key, validate_rna


def parse_binding_label(val: Any) -> int | None:
    """
    Parse literature yes/no labels.

    Returns 1 (binding), 0 (non-binding), or None (ambiguous → skip).
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("yes", "1", "true"):
        return 1
    if s in ("no", "0", "false"):
        return 0
    return None


def load_external_xlsx(xlsx_path: str | os.PathLike) -> pd.DataFrame:
    """Load the active worksheet from the literature Excel file."""
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to read the external benchmark xlsx. "
            "Install with: pip install openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = list(rows[0])
    clean_header = [str(h).strip().replace("\n", " ") if h else None for h in header]
    return pd.DataFrame(rows[1:], columns=clean_header)


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    for col in df.columns:
        if col and any(k.lower() in str(col).lower() for k in candidates):
            return col
    return None


def resolve_external_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical field names to actual spreadsheet column names."""
    col_protein = _find_column(df, "protein") or "Protein"
    col_rna_seq = _find_column(df, "rna sequence", "rna seq") or "RNA sequence"
    col_label = _find_column(df, "interaction") or "Interaction (yes/no)"

    col_prot_seq = None
    for c in df.columns:
        if c and "protein sequence" in str(c).lower():
            col_prot_seq = c
            break
    col_prot_seq = col_prot_seq or "Protein Sequence"

    col_domain_seq = None
    for c in df.columns:
        cl = str(c).lower()
        if c and "domain" in cl and ("sequence" in cl or "mutation" in cl):
            col_domain_seq = c
            break
    col_domain_seq = col_domain_seq or col_prot_seq

    return {
        "protein_name": col_protein,
        "rna_sequence": col_rna_seq,
        "protein_sequence_full": col_prot_seq,
        "protein_sequence_domain": col_domain_seq,
        "binding_label": col_label,
    }


@dataclass
class ParseStats:
    raw_rows: int = 0
    skipped_ambiguous_label: int = 0
    skipped_missing_sequence: int = 0
    skipped_invalid_rna: int = 0
    skipped_invalid_protein: int = 0
    usable_rows: int = 0


def parse_curated_pairs(df_raw: pd.DataFrame) -> tuple[list[CuratedPair], ParseStats, dict[str, str]]:
    """Parse spreadsheet rows into validated ``CuratedPair`` records."""
    cols = resolve_external_columns(df_raw)
    stats = ParseStats(raw_rows=len(df_raw))
    pairs: list[CuratedPair] = []

    for row_idx, row in df_raw.iterrows():
        label = parse_binding_label(row.get(cols["binding_label"]))
        if label is None:
            stats.skipped_ambiguous_label += 1
            continue

        rna_raw = str(row.get(cols["rna_sequence"], "") or "").strip()
        prot_domain = str(row.get(cols["protein_sequence_domain"], "") or "").strip()
        prot_full = str(row.get(cols["protein_sequence_full"], "") or "").strip()
        prot_raw = prot_domain if len(prot_domain) >= 10 else prot_full
        protein_name = str(row.get(cols["protein_name"], "") or "").strip()

        if len(rna_raw) < 4 or len(prot_raw) < 10 or not protein_name:
            stats.skipped_missing_sequence += 1
            continue

        ok_rna, rna_seq = validate_rna(rna_raw)
        if not ok_rna or len(rna_seq) < 4:
            stats.skipped_invalid_rna += 1
            continue

        prot_seq, prot_warnings = sanitize_protein_sequence(prot_raw, strict=True, source=protein_name)
        if len(prot_seq) < 10:
            stats.skipped_invalid_protein += 1
            continue

        pair_id = f"curated_{row_idx:04d}"
        pairs.append(
            CuratedPair(
                pair_id=pair_id,
                protein_name=protein_name,
                protein_sequence=prot_seq,
                rna_sequence=rna_seq,
                binding_label=int(label),
                source_row=int(row_idx),
                metadata={
                    "protein_warnings": prot_warnings,
                    "used_domain_sequence": prot_raw == prot_domain and len(prot_domain) >= 10,
                },
            )
        )

    stats.usable_rows = len(pairs)
    return pairs, stats, cols


@dataclass
class TrainLeakageIndex:
    """Index of training split content for external benchmark leakage checks."""

    train_pairs: set[tuple[str, str]]
    train_positive_rnas: set[str]
    train_protein_names: set[str]
    train_protein_sequences: set[str]
    n_train_rows: int = 0

    @classmethod
    def from_train_tsv(cls, train_tsv: str | os.PathLike) -> "TrainLeakageIndex":
        path = Path(train_tsv)
        if not path.exists():
            raise FileNotFoundError(f"Training TSV not found: {path}")

        df = pd.read_csv(path, sep="\t", usecols=lambda c: c in {
            "protein_name", "protein_sequence", "rna_sequence", "binding_label",
        })
        df["protein_sequence"] = df["protein_sequence"].astype(str).str.upper()
        df["rna_sequence"] = df["rna_sequence"].astype(str).map(normalize_rna)

        pairs = set(zip(df["protein_sequence"], df["rna_sequence"]))
        pos = df[df["binding_label"] == 1] if "binding_label" in df.columns else df

        return cls(
            train_pairs=pairs,
            train_positive_rnas=set(pos["rna_sequence"].unique()),
            train_protein_names=set(df["protein_name"].astype(str).unique()),
            train_protein_sequences=set(df["protein_sequence"].unique()),
            n_train_rows=len(df),
        )

    def annotate(self, protein_name: str, protein_sequence: str, rna_sequence: str) -> dict[str, bool]:
        pseq = str(protein_sequence).upper()
        rseq = normalize_rna(rna_sequence)
        key = (pseq, rseq)
        return {
            "protein_name_in_train": protein_name in self.train_protein_names,
            "protein_sequence_in_train": pseq in self.train_protein_sequences,
            "rna_in_train_positives": rseq in self.train_positive_rnas,
            "exact_pair_in_train": key in self.train_pairs,
        }


@dataclass
class BenchmarkBuildReport:
    parse_stats: ParseStats
    config: NegativeSamplingConfig
    n_curated_positive: int = 0
    n_curated_negative: int = 0
    n_generated_negative: int = 0
    n_total: int = 0
    generation_failures: dict[str, int] = field(default_factory=dict)
    proteins_with_both_classes_after: int = 0
    single_class_proteins_after: int = 0
    train_leakage_summary: dict[str, int] = field(default_factory=dict)
    neg_strategy_counts: dict[str, int] = field(default_factory=dict)


def curated_to_row(
    pair: CuratedPair,
    *,
    example_class: str,
    neg_strategy: str | None,
    leakage: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "parent_pair_id": pair.pair_id if pair.binding_label == 1 else "",
        "protein_name": pair.protein_name,
        "protein_sequence": pair.protein_sequence,
        "rna_sequence": pair.rna_sequence,
        "binding_label": pair.binding_label,
        "example_class": example_class,
        "neg_strategy": neg_strategy or "",
        "source_xlsx_row": pair.source_row if pair.source_row is not None else "",
        "generation_seed": "",
        "partner_pair_id": "",
        "hamming_to_parent_rna": "",
        "rejection_attempts": "",
        "rna_length": len(pair.rna_sequence),
        "protein_length": len(pair.protein_sequence),
        "gc_content": round(gc_content(pair.rna_sequence), 6),
        "protein_name_in_train": int((leakage or {}).get("protein_name_in_train", False)),
        "protein_sequence_in_train": int((leakage or {}).get("protein_sequence_in_train", False)),
        "rna_in_train_positives": int((leakage or {}).get("rna_in_train_positives", False)),
        "exact_pair_in_train": int((leakage or {}).get("exact_pair_in_train", False)),
    }


def generated_to_row(
    neg: GeneratedNegative,
    *,
    leakage: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": neg.pair_id,
        "parent_pair_id": neg.parent_pair_id,
        "protein_name": neg.protein_name,
        "protein_sequence": neg.protein_sequence,
        "rna_sequence": neg.rna_sequence,
        "binding_label": 0,
        "example_class": "generated_negative",
        "neg_strategy": neg.neg_strategy,
        "source_xlsx_row": "",
        "generation_seed": neg.generation_seed,
        "partner_pair_id": neg.partner_pair_id or "",
        "hamming_to_parent_rna": neg.hamming_to_parent_rna if neg.hamming_to_parent_rna is not None else "",
        "rejection_attempts": neg.rejection_attempts,
        "rna_length": len(neg.rna_sequence),
        "protein_length": len(neg.protein_sequence),
        "gc_content": round(gc_content(neg.rna_sequence), 6),
        "protein_name_in_train": int((leakage or {}).get("protein_name_in_train", False)),
        "protein_sequence_in_train": int((leakage or {}).get("protein_sequence_in_train", False)),
        "rna_in_train_positives": int((leakage or {}).get("rna_in_train_positives", False)),
        "exact_pair_in_train": int((leakage or {}).get("exact_pair_in_train", False)),
    }


def build_expanded_benchmark(
    curated: list[CuratedPair],
    cfg: NegativeSamplingConfig,
    train_index: TrainLeakageIndex | None = None,
) -> tuple[pd.DataFrame, BenchmarkBuildReport]:
    """
    Build the full external benchmark: curated positives/negatives + generated negatives.
    """
    positives = [p for p in curated if p.binding_label == 1]
    negatives = [p for p in curated if p.binding_label == 0]

    forbidden_pairs = {pair_key(p.protein_sequence, p.rna_sequence) for p in curated}
    forbidden_rnas_by_protein: dict[str, set[str]] = {}
    for p in curated:
        forbidden_rnas_by_protein.setdefault(p.protein_name, set()).add(normalize_rna(p.rna_sequence))

    all_generated: list[GeneratedNegative] = []
    failure_totals: dict[str, int] = {}

    for anchor in positives:
        protein_rnas = forbidden_rnas_by_protein.setdefault(anchor.protein_name, set())
        negs, failures = generate_all_negatives_for_anchor(
            anchor,
            positives,
            forbidden_pairs,
            protein_rnas,
            cfg,
        )
        all_generated.extend(negs)
        for k, v in failures.items():
            failure_totals[k] = failure_totals.get(k, 0) + v
        for neg in negs:
            forbidden_pairs.add(pair_key(neg.protein_sequence, neg.rna_sequence))
            protein_rnas.add(normalize_rna(neg.rna_sequence))

    rows: list[dict[str, Any]] = []
    for p in positives:
        leakage = train_index.annotate(p.protein_name, p.protein_sequence, p.rna_sequence) if train_index else {}
        rows.append(curated_to_row(p, example_class="curated_positive", neg_strategy=None, leakage=leakage))

    for p in negatives:
        leakage = train_index.annotate(p.protein_name, p.protein_sequence, p.rna_sequence) if train_index else {}
        rows.append(curated_to_row(p, example_class="curated_negative", neg_strategy="curated", leakage=leakage))

    for neg in all_generated:
        leakage = train_index.annotate(neg.protein_name, neg.protein_sequence, neg.rna_sequence) if train_index else {}
        rows.append(generated_to_row(neg, leakage=leakage))

    df = pd.DataFrame(rows)

    # Per-protein class balance after expansion
    both_classes = 0
    single_class = 0
    for _, grp in df.groupby("protein_name"):
        classes = set(grp["binding_label"].unique())
        if classes == {0, 1}:
            both_classes += 1
        else:
            single_class += 1

    leakage_summary = {
        "exact_pair_in_train": int(df["exact_pair_in_train"].sum()) if len(df) else 0,
        "rna_in_train_positives": int(df["rna_in_train_positives"].sum()) if len(df) else 0,
        "protein_name_in_train": int(df["protein_name_in_train"].sum()) if len(df) else 0,
    }

    strategy_counts = df["neg_strategy"].value_counts().to_dict() if len(df) else {}

    report = BenchmarkBuildReport(
        parse_stats=ParseStats(),  # filled by caller
        config=cfg,
        n_curated_positive=len(positives),
        n_curated_negative=len(negatives),
        n_generated_negative=len(all_generated),
        n_total=len(df),
        generation_failures=failure_totals,
        proteins_with_both_classes_after=both_classes,
        single_class_proteins_after=single_class,
        train_leakage_summary=leakage_summary,
        neg_strategy_counts=strategy_counts,
    )
    return df, report


def load_benchmark_tsv(tsv_path: str | os.PathLike) -> pd.DataFrame:
    """Load a pre-built expanded benchmark TSV for evaluation."""
    path = Path(tsv_path)
    df = pd.read_csv(path, sep="\t")
    required = {"protein_name", "protein_sequence", "rna_sequence", "binding_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark TSV missing columns: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rna = normalize_rna(str(row["rna_sequence"]))
        prot = str(row["protein_sequence"]).upper()
        records.append(
            {
                "pair_id": str(row.get("pair_id", "") or ""),
                "protein_name": str(row["protein_name"]),
                "rna_seq": rna,
                "prot_seq": prot,
                "label": int(row["binding_label"]),
                "rna_len": len(rna),
                "prot_len": len(prot),
                "example_class": str(row.get("example_class", "") or ""),
                "neg_strategy": str(row.get("neg_strategy", "") or ""),
                "parent_pair_id": str(row.get("parent_pair_id", "") or ""),
            }
        )
    return pd.DataFrame(records)


def save_benchmark_outputs(
    df: pd.DataFrame,
    report: BenchmarkBuildReport,
    out_dir: str | os.PathLike,
    *,
    basename: str = "external_benchmark_expanded",
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tsv_path = out / f"{basename}.tsv"
    json_path = out / f"{basename}_manifest.json"

    df.to_csv(tsv_path, sep="\t", index=False)

    manifest = {
        "n_total": report.n_total,
        "n_curated_positive": report.n_curated_positive,
        "n_curated_negative": report.n_curated_negative,
        "n_generated_negative": report.n_generated_negative,
        "neg_strategy_counts": report.neg_strategy_counts,
        "generation_failures": report.generation_failures,
        "proteins_with_both_classes_after": report.proteins_with_both_classes_after,
        "single_class_proteins_after": report.single_class_proteins_after,
        "train_leakage_summary": report.train_leakage_summary,
        "config": asdict(report.config),
    }
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return tsv_path, json_path
