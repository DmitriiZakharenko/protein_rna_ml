#!/usr/bin/env python3
"""
26_visualize_rna_only_results.py
Generate figures for RNA-only per-protein classifiers (script 25) and
top/bottom example extraction (script 24).

Figures produced:
  figures/rna_only_dataset_comparison.png   — median test AUROC/AUPRC per dataset
  figures/rna_only_pp_distributions.png     — per-protein test AUROC violin plots
  figures/rna_only_model_wins.png           — LR vs RF wins per dataset
  figures/rna_only_weakest_proteins.png     — proteins with lowest test AUROC
  figures/top_bottom_examples_overview.png  — extracted examples by protocol/dataset

Usage:
    python scripts/26_visualize_rna_only_results.py
    python scripts/26_visualize_rna_only_results.py \\
        --results_dir results/rna_only_per_protein_honest \\
        --examples_tsv results/top_bottom_examples/all_protocols_summary.tsv \\
        --out_dir figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

BLUE = "#2271B5"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#8F8F8F"
GOLD = "#E69F00"
TEAL = "#56B4E9"

DATASET_LABELS = {
    "htr_selex": "HTR-SELEX",
    "rbns": "RBNS",
    "rnacompete_eukarya": "RNAcompete\nEukarya",
    "rnacompete_rbpzoo": "RNAcompete\nRBPZoo",
    "rnacompete_ucrbp23": "RNAcompete\nucRBP23",
}

DATASET_COLORS = {
    "htr_selex": BLUE,
    "rbns": GREEN,
    "rnacompete_eukarya": ORANGE,
    "rnacompete_rbpzoo": PURPLE,
    "rnacompete_ucrbp23": TEAL,
}


def save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def load_stats(results_dir: Path) -> list[dict]:
    stats = []
    for path in sorted(results_dir.glob("*_stats.json")):
        with open(path) as f:
            stats.append(json.load(f))
    return stats


def load_per_protein(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("*_per_protein_metrics.tsv")):
        df = pd.read_csv(path, sep="\t")
        if "dataset" not in df.columns:
            df["dataset"] = path.name.replace("_per_protein_metrics.tsv", "")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No per-protein metrics in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def fig_dataset_comparison(stats: list[dict], out_path: Path) -> None:
    rows = []
    for s in stats:
        ds = s["dataset"]
        rows.append(
            {
                "dataset": ds,
                "label": DATASET_LABELS.get(ds, ds),
                "n_proteins": s["n_proteins_trained"],
                "median_auroc": s["median_test_auroc"],
                "median_auprc": s["median_test_auprc"],
                "recommended": s.get("recommended_model", ""),
            }
        )
    df = pd.DataFrame(rows).sort_values("median_auroc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(df))
    colors = [DATASET_COLORS.get(d, GRAY) for d in df["dataset"]]

    for ax, metric, title in zip(
        axes,
        ["median_auroc", "median_auprc"],
        ["Median Test AUROC", "Median Test AUPRC"],
    ):
        bars = ax.bar(x, df[metric], color=colors, alpha=0.88, width=0.65)
        ax.axhline(0.9, ls="--", color=GRAY, lw=1.1, alpha=0.8, label="AUROC 0.90 ref.")
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], fontsize=9)
        ax.set_ylim(0.65, 1.02)
        ax.set_ylabel(title)
        ax.set_title(title + " by Dataset", fontweight="bold")
        for bar, val, n in zip(bars, df[metric], df["n_proteins"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    rec = df["recommended"].mode().iloc[0] if len(df) else "random_forest"
    fig.suptitle(
        f"RNA-only 4-mer Classifiers — Honest 60/20/20 Split (recommended: {rec})",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, out_path)


def fig_pp_distributions(pp: pd.DataFrame, out_path: Path) -> None:
    order = sorted(pp["dataset"].unique(), key=lambda d: pp.loc[pp["dataset"] == d, "auroc"].median())
    data = [pp.loc[pp["dataset"] == ds, "auroc"].values for ds in order]
    labels = [DATASET_LABELS.get(ds, ds).replace("\n", " ") for ds in order]
    colors = [DATASET_COLORS.get(ds, GRAY) for ds in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(data, positions=range(len(order)), showmeans=False, showmedians=False)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.35)
        body.set_edgecolor(colors[i])
        body.set_linewidth(1.2)

    bp = ax.boxplot(
        data,
        positions=range(len(order)),
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color=GRAY, linewidth=1),
        capprops=dict(color=GRAY, linewidth=1),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0.9, ls="--", color=GRAY, lw=1.1, label="AUROC 0.90")
    ax.axhline(0.5, ls=":", color=GRAY, lw=1.0, alpha=0.7, label="Random (0.50)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Per-Protein Test AUROC")
    ax.set_title("Per-Protein Test AUROC Distribution (RNA-only 4-mer)", fontweight="bold")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=9, loc="lower right")

    for i, ds in enumerate(order):
        sub = pp.loc[pp["dataset"] == ds, "auroc"]
        below = (sub < 0.9).sum()
        ax.text(
            i,
            0.47,
            f"{below}/{len(sub)} <0.9",
            ha="center",
            fontsize=8,
            color=GRAY,
        )

    fig.tight_layout()
    save(fig, out_path)


def fig_model_wins(stats: list[dict], out_path: Path) -> None:
    rows = []
    for s in stats:
        ds = s["dataset"]
        mc = s.get("model_comparison", {})
        lr_wins = mc.get("logistic_regression", {}).get("wins", 0)
        rf_wins = mc.get("random_forest", {}).get("wins", 0)
        n = s["n_proteins_trained"]
        rows.append(
            {
                "dataset": ds,
                "label": DATASET_LABELS.get(ds, ds),
                "lr": lr_wins,
                "rf": rf_wins,
                "n": n,
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    ax.bar(x, df["lr"], label="Logistic Regression", color=BLUE, alpha=0.85)
    ax.bar(x, df["rf"], bottom=df["lr"], label="Random Forest", color=GREEN, alpha=0.85)

    for i, row in df.iterrows():
        ax.text(x[i], row["n"] + 1, f"n={row['n']}", ha="center", fontsize=8, color=GRAY)
        if row["rf"] > row["lr"]:
            winner = "RF"
        else:
            winner = "LR"
        ax.text(
            x[i],
            row["n"] / 2,
            winner,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("Proteins where model wins (val AUROC)")
    ax.set_title("LR vs RF — Model Selection Wins per Dataset", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, out_path)


def fig_weakest_proteins(pp: pd.DataFrame, out_path: Path, n_show: int = 20) -> None:
    worst = pp.nsmallest(n_show, "auroc").copy()
    worst["label"] = worst.apply(
        lambda r: f"{r['protein_name']} ({DATASET_LABELS.get(r['dataset'], r['dataset']).replace(chr(10), ' ')})",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(9, max(5, n_show * 0.28)))
    colors = [ORANGE if v < 0.8 else GOLD for v in worst["auroc"]]
    ax.barh(worst["label"], worst["auroc"], color=colors, alpha=0.88)
    ax.axvline(0.9, ls="--", color=GRAY, lw=1.1, label="AUROC 0.90")
    ax.axvline(0.5, ls=":", color=GRAY, lw=1.0, alpha=0.7, label="Random")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Test AUROC")
    ax.set_title(f"Lowest {n_show} Per-Protein Test AUROC (RNA-only)", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, out_path)


def fig_top_bottom_overview(examples_path: Path, out_path: Path) -> None:
    if not examples_path.exists():
        print(f"  [skip] examples file not found: {examples_path}")
        return

    df = pd.read_csv(examples_path, sep="\t")
    if "dataset" in df.columns:
        group_cols = ["protocol", "dataset"]
    else:
        group_cols = ["protocol"]

    n_proteins = (
        df.groupby(group_cols, observed=True)["protein_name"]
        .nunique()
        .reset_index(name="n_proteins")
    )
    counts = (
        df.groupby(group_cols + ["split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("positive", "negative"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.merge(n_proteins, on=group_cols, how="left")

    counts["label"] = counts.apply(
        lambda r: (
            f"{r['protocol']}\n{r['dataset']}"
            if "dataset" in counts.columns
            else str(r["protocol"])
        ),
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(counts))
    w = 0.35
    ax.bar(x - w / 2, counts["positive"], w, label="Top positives", color=GREEN, alpha=0.85)
    ax.bar(x + w / 2, counts["negative"], w, label="Bottom negatives", color=ORANGE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(counts["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Number of extracted examples")
    ax.set_title("Top/Bottom Examples per Protocol", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(x, counts["n_proteins"], color=BLUE, alpha=0.85, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(counts["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Unique proteins")
    ax.set_title("Proteins with Extracted Examples", fontweight="bold")
    for i, (n_prot, n_pos, n_neg) in enumerate(
        zip(counts["n_proteins"], counts["positive"], counts["negative"])
    ):
        ax.text(i, n_prot + 2, f"{n_pos + n_neg} rows", ha="center", fontsize=7, color=GRAY)

    fig.suptitle(
        f"Script 24 — Top-5 / Bottom-5 RNA Examples (total {len(df):,} rows)",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RNA-only classifier results")
    parser.add_argument(
        "--results_dir",
        default="results/rna_only_per_protein_honest",
        help="Directory with *_stats.json and *_per_protein_metrics.tsv",
    )
    parser.add_argument(
        "--examples_tsv",
        default="results/top_bottom_examples/all_protocols_summary.tsv",
        help="Combined top/bottom examples TSV from script 24",
    )
    parser.add_argument("--out_dir", default="figures", help="Output directory for PNG figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    print(f"Reading results from {results_dir}")
    stats = load_stats(results_dir)
    if not stats:
        raise SystemExit(f"No *_stats.json files in {results_dir}")

    pp = load_per_protein(results_dir)

    print(f"Writing figures to {out_dir}/")
    fig_dataset_comparison(stats, out_dir / "rna_only_dataset_comparison.png")
    fig_pp_distributions(pp, out_dir / "rna_only_pp_distributions.png")
    fig_model_wins(stats, out_dir / "rna_only_model_wins.png")
    fig_weakest_proteins(pp, out_dir / "rna_only_weakest_proteins.png")
    fig_top_bottom_overview(Path(args.examples_tsv), out_dir / "top_bottom_examples_overview.png")
    print("Done.")


if __name__ == "__main__":
    main()
