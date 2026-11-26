#!/usr/bin/env python3
"""
27_sample_rnacompete_intensity_spectrum.py
------------------------------------------
Sample RNAcompete probes evenly across the log-intensity spectrum per protein.

For each selected protein:
  1. Keep best experiment (highest mean positive probe_intensity), if multiple.
  2. Restrict to modal probe length (most frequent rna_sequence length).
  3. log10(probe_intensity − Smin + 1) per protein (Smin = min raw intensity in sampling pool).
  4. Sample N probes (default 100) at evenly spaced percentiles of log-intensity.

Also reports correlation between per-protein mean positive intensity and
protein sequence length across the full dataset.

Usage:
    python scripts/27_sample_rnacompete_intensity_spectrum.py \\
        --data_file ../rnacompete_analysis/eukarya/results/ml_dataset_eukarya_clean.tsv.gz \\
        --dataset rnacompete_eukarya

    python scripts/27_sample_rnacompete_intensity_spectrum.py \\
        --data_file ../rnacompete_analysis/rbpzoo/results/ml_dataset_rbpzoo_clean.tsv.gz \\
        --dataset rnacompete_rbpzoo --n_proteins 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def portable_path(path: Path | str) -> str:
    """Write repo-relative paths in stats JSON (avoid absolute home directories)."""
    p = Path(path).expanduser().resolve()
    if not p.is_absolute():
        return str(path)
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        try:
            return os.path.relpath(p, base)
        except ValueError:
            continue
    return str(p)


def normalize_protein_col(df: pd.DataFrame) -> pd.DataFrame:
    if "protein_name" not in df.columns:
        if "target_name" in df.columns:
            df = df.rename(columns={"target_name": "protein_name"})
        else:
            sys.exit("ERROR: need protein_name or target_name column")
    return df


def select_best_rnacompete_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """One experiment per protein: highest mean positive probe_intensity."""
    id_col = next((c for c in ("experiment_id", "hyb_id") if c in df.columns), None)
    if id_col is None:
        return df

    parts: list[pd.DataFrame] = []
    for _, sub in df.groupby("protein_name", sort=False):
        if sub[id_col].nunique() <= 1:
            parts.append(sub)
            continue
        sub = sub.copy()
        sub["probe_intensity"] = pd.to_numeric(sub["probe_intensity"], errors="coerce")
        pos = sub[sub["binding_label"] == 1]
        if pos.empty or pos["probe_intensity"].isna().all():
            parts.append(sub)
            continue
        best_eid = pos.groupby(id_col)["probe_intensity"].mean().idxmax()
        parts.append(sub[sub[id_col] == best_eid])
    return pd.concat(parts, ignore_index=True)


def modal_probe_length(df: pd.DataFrame) -> int:
    lengths = df["rna_sequence"].astype(str).str.len()
    return int(lengths.value_counts().idxmax())


def add_log_intensity(df: pd.DataFrame, s_min: float | None = None) -> tuple[pd.DataFrame, float]:
    """
    log10(S − Smin + 1) per protein (Smin = minimum raw intensity in the sampling pool).
    """
    out = df.copy()
    intensity = pd.to_numeric(out["probe_intensity"], errors="coerce")
    if s_min is None:
        s_min = float(intensity.min()) if intensity.notna().any() else 0.0
    out["intensity_s_min"] = s_min
    out["log_intensity"] = np.log10(intensity - s_min + 1.0)
    return out, s_min


def sample_intensity_spectrum(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Pick n_samples probes at evenly spaced percentiles of log_intensity (no replacement).
    """
    df = df.dropna(subset=["log_intensity"]).copy()
    if df.empty:
        return df

    n_samples = min(n_samples, len(df))
    if n_samples == 0:
        return df.iloc[0:0]

    targets = np.linspace(0.5 / n_samples, 100 - 0.5 / n_samples, n_samples)
    log_vals = df["log_intensity"].to_numpy()
    used_idx: set[int] = set()
    rows: list[pd.Series] = []

    for pct in targets:
        target_val = float(np.percentile(log_vals, pct))
        candidates = df.loc[~df.index.isin(used_idx)]
        if candidates.empty:
            break
        nearest_idx = (candidates["log_intensity"] - target_val).abs().idxmin()
        used_idx.add(nearest_idx)
        row = df.loc[nearest_idx].copy()
        row["target_percentile"] = pct
        row["target_log_intensity"] = target_val
        rows.append(row)

    if not rows:
        return df.iloc[0:0]

    out = pd.DataFrame(rows)
    out["spectrum_rank"] = range(1, len(out) + 1)
    return out


def protein_mean_positive_intensity(df: pd.DataFrame) -> float:
    pos = df[df["binding_label"] == 1]
    if pos.empty:
        return float("nan")
    return float(pd.to_numeric(pos["probe_intensity"], errors="coerce").mean())


def protein_length(df: pd.DataFrame) -> int | None:
    if "protein_sequence" not in df.columns:
        return None
    seqs = df["protein_sequence"].dropna().astype(str)
    if seqs.empty:
        return None
    return int(seqs.iloc[0].strip().__len__())


def compute_length_intensity_correlation(summary: pd.DataFrame) -> dict:
    sub = summary.dropna(subset=["mean_positive_intensity", "protein_length_aa"])
    sub = sub[sub["protein_length_aa"] > 0]
    n = len(sub)
    result = {"n_proteins": n}
    if n < 3:
        result["note"] = "Too few proteins for correlation"
        return result

    x = sub["protein_length_aa"].to_numpy(dtype=float)
    y = sub["mean_positive_intensity"].to_numpy(dtype=float)

    if HAS_SCIPY:
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        result.update(
            {
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
            }
        )
    else:
        result["note"] = "scipy not installed; correlation skipped"

    return result


def rank_proteins_by_mean_positive(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    id_col = next((c for c in ("experiment_id", "hyb_id") if c in df.columns), None)
    for prot, sub in df.groupby("protein_name", sort=False):
        sub = select_best_rnacompete_experiment(sub)
        rows.append(
            {
                "protein_name": prot,
                "mean_positive_intensity": protein_mean_positive_intensity(sub),
                "protein_length_aa": protein_length(sub),
                "n_probes_best_experiment": len(sub),
                "n_positives": int((sub["binding_label"] == 1).sum()),
                "experiment_id": sub[id_col].iloc[0] if id_col else None,
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_positive_intensity", ascending=False)
    out["intensity_rank"] = range(1, len(out) + 1)
    return out


def process_protein(
    df: pd.DataFrame,
    protein: str,
    n_samples: int,
) -> tuple[pd.DataFrame, dict]:
    sub = df[df["protein_name"] == protein].copy()
    sub = select_best_rnacompete_experiment(sub)
    sub["probe_intensity"] = pd.to_numeric(sub["probe_intensity"], errors="coerce")

    mode_len = modal_probe_length(sub)
    sub = sub[sub["rna_sequence"].astype(str).str.len() == mode_len].copy()
    sub, s_min = add_log_intensity(sub)

    sampled = sample_intensity_spectrum(sub, n_samples)
    stats = {
        "protein_name": protein,
        "modal_probe_length": mode_len,
        "intensity_s_min": s_min,
        "log_transform": "log10(intensity - s_min + 1)",
        "n_probes_modal_length": len(sub),
        "n_sampled": len(sampled),
        "mean_positive_intensity": protein_mean_positive_intensity(sub),
        "protein_length_aa": protein_length(sub),
        "log_intensity_min": float(sub["log_intensity"].min()) if len(sub) else None,
        "log_intensity_max": float(sub["log_intensity"].max()) if len(sub) else None,
    }
    if id_col := next((c for c in ("experiment_id", "hyb_id") if c in sub.columns), None):
        stats["experiment_id"] = sub[id_col].iloc[0] if len(sub) else None

    return sampled, stats


def main() -> None:
    p = argparse.ArgumentParser(description="RNAcompete log-intensity spectrum sampling")
    p.add_argument("--data_file", required=True, help="RNAcompete clean TSV[.gz]")
    p.add_argument("--dataset", default="rnacompete", help="Dataset label for outputs")
    p.add_argument(
        "--output_dir",
        default="results/rnacompete_intensity_spectrum",
        help="Output root directory",
    )
    p.add_argument(
        "--n_proteins",
        type=int,
        default=3,
        help="Top N proteins by mean positive intensity (0 = all)",
    )
    p.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        help="Explicit protein names (overrides --n_proteins)",
    )
    p.add_argument("--n_samples", type=int, default=100, help="Probes per protein")
    args = p.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        sys.exit(f"ERROR: file not found: {data_path}")

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    df = normalize_protein_col(df)
    df["binding_label"] = pd.to_numeric(df["binding_label"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["rna_sequence", "binding_label", "protein_name"])
    df["binding_label"] = df["binding_label"].astype(int)
    print(f"  {len(df):,} rows, {df['protein_name'].nunique()} proteins")

    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    protein_summary = rank_proteins_by_mean_positive(df)
    summary_path = out_root / "protein_intensity_summary.tsv"
    protein_summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Protein summary: {summary_path}")

    corr = compute_length_intensity_correlation(protein_summary)
    corr_path = out_root / "length_vs_intensity_correlation.json"
    corr_path.write_text(json.dumps(corr, indent=2) + "\n")
    print(f"Length vs intensity correlation: {corr_path}")
    if "pearson_r" in corr:
        print(
            f"  Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.3g}), "
            f"Spearman rho={corr['spearman_rho']:.3f} (p={corr['spearman_p']:.3g}), "
            f"n={corr['n_proteins']}"
        )

    if args.proteins:
        selected = list(args.proteins)
    elif args.n_proteins > 0:
        selected = protein_summary.head(args.n_proteins)["protein_name"].tolist()
    else:
        selected = protein_summary["protein_name"].tolist()

    print(f"Sampling {args.n_samples} probes for {len(selected)} proteins: {', '.join(selected)}")

    all_sampled: list[pd.DataFrame] = []
    run_stats: list[dict] = []

    for prot in selected:
        sampled, stats = process_protein(df, prot, args.n_samples)
        stats["dataset"] = args.dataset
        run_stats.append(stats)

        if sampled.empty:
            print(f"  SKIP {prot}: no probes after filters")
            continue

        per_path = out_root / f"{prot}.tsv"
        cols = [
            c
            for c in [
                "protein_name",
                "rna_sequence",
                "binding_label",
                "probe_intensity",
                "intensity_s_min",
                "log_intensity",
                "spectrum_rank",
                "target_percentile",
                "target_log_intensity",
                "probe_id",
                "hyb_id",
                "experiment_id",
                "organism",
            ]
            if c in sampled.columns
        ]
        sampled[cols].to_csv(per_path, sep="\t", index=False)
        print(f"  OK   {prot}: {len(sampled)} probes -> {per_path}")
        sampled = sampled.copy()
        sampled["dataset"] = args.dataset
        all_sampled.append(sampled[cols + ["dataset"]])

    if all_sampled:
        combined = pd.concat(all_sampled, ignore_index=True)
        combined_path = out_root / f"spectrum_samples_{args.dataset}.tsv"
        combined.to_csv(combined_path, sep="\t", index=False)
        print(f"Combined samples: {combined_path} ({len(combined)} rows)")

    stats_path = out_root / "sampling_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "data_file": portable_path(data_path),
                "n_samples_requested": args.n_samples,
                "proteins_selected": selected,
                "per_protein": run_stats,
                "length_vs_intensity_correlation": corr,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Stats: {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
