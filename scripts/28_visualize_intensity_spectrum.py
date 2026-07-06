#!/usr/bin/env python3
"""
28_visualize_intensity_spectrum.py
----------------------------------
Figures for RNAcompete intensity spectrum sampling (script 27).

Figures:
  figures/rnacompete_length_vs_intensity.png      scatter: protein length vs mean pos. intensity
  figures/rnacompete_intensity_spectrum.png       sampled log-intensity vs percentile (overlay)
  figures/rnacompete_intensity_spectrum_faceted_{dataset}.png
      one subplot per selected protein (recommended when >3 proteins)

Usage:
    python scripts/28_visualize_intensity_spectrum.py
    python scripts/28_visualize_intensity_spectrum.py \\
        --results_dir results/rnacompete_intensity_spectrum_domain_diverse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
RED = "#D62728"

DATASETS = {
    "rnacompete_eukarya": {"label": "RNAcompete Eukarya", "color": ORANGE},
    "rnacompete_rbpzoo": {"label": "RNAcompete RBPZoo", "color": PURPLE},
}

PROTEIN_COLORS = [BLUE, GREEN, RED]


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def load_summary(results_root: Path, dataset: str) -> pd.DataFrame:
    path = results_root / dataset / "protein_intensity_summary.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def load_corr(results_root: Path, dataset: str) -> dict:
    path = results_root / dataset / "length_vs_intensity_correlation.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_samples(results_root: Path, dataset: str) -> pd.DataFrame:
    path = results_root / dataset / f"spectrum_samples_{dataset}.tsv"
    if not path.exists():
        # legacy name from earlier runs
        path = results_root / dataset / "spectrum_samples.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def load_selected(results_root: Path, dataset: str) -> list[str]:
    path = results_root / dataset / "sampling_stats.json"
    if path.exists():
        return json.loads(path.read_text()).get("proteins_selected", [])
    return []


def load_domain_selection(results_root: Path, dataset: str) -> pd.DataFrame | None:
    stem = dataset.replace("rnacompete_", "")
    candidates = [
        results_root / f"domain_diverse_selection_{stem}.tsv",
        results_root.parent / "rnacompete_intensity_spectrum" / f"domain_diverse_selection_{stem}.tsv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, sep="\t")
    return None


def fig_spectrum_faceted(
    samples: pd.DataFrame,
    selected: list[str],
    domain_map: dict[str, str],
    dataset_label: str,
    out_path: Path,
) -> None:
    prots = [p for p in selected if p in set(samples["protein_name"])]
    n = len(prots)
    if n == 0:
        return

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, prot in enumerate(prots):
        ax = axes[i]
        sub = samples[samples["protein_name"] == prot].sort_values("target_percentile")
        ax.plot(sub["target_percentile"], sub["log_intensity"], "o-", ms=3, lw=1.0, color=BLUE)
        dom = domain_map.get(prot, "")
        title = f"{prot}"
        if dom:
            title += f"\n({dom})"
        ax.set_title(title, fontsize=8)
        ax.axhline(0, ls=":", color=GRAY, lw=0.8, alpha=0.7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Intensity percentile (modal-length probes)")
    fig.supylabel("log₁₀(probe_intensity)")
    fig.suptitle(
        f"Spectrum Samples by Domain Architecture — {dataset_label} (n={n} proteins × 100 probes)",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, out_path)


def annotate_corr(ax: plt.Axes, corr: dict, x: float = 0.04, y: float = 0.96) -> None:
    if "pearson_r" not in corr:
        return
    ax.text(
        x,
        y,
        f"Pearson r = {corr['pearson_r']:.3f} (p = {corr['pearson_p']:.3g})\n"
        f"Spearman ρ = {corr['spearman_rho']:.3f} (p = {corr['spearman_p']:.3g})\n"
        f"n = {corr.get('n_proteins', corr.get('n', '?'))}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor=GRAY),
    )


def fig_length_vs_intensity(
    summaries: dict[str, pd.DataFrame],
    corrs: dict[str, dict],
    selected: dict[str, list[str]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (ds, meta) in zip(axes, DATASETS.items()):
        df = summaries[ds]
        ax.scatter(
            df["protein_length_aa"],
            df["mean_positive_intensity"],
            s=22,
            alpha=0.45,
            color=meta["color"],
            edgecolors="none",
        )
        top = set(selected.get(ds, []))
        hi = df[df["protein_name"].isin(top)]
        ax.scatter(
            hi["protein_length_aa"],
            hi["mean_positive_intensity"],
            s=70,
            color=meta["color"],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        for _, row in hi.iterrows():
            ax.annotate(
                row["protein_name"],
                (row["protein_length_aa"], row["mean_positive_intensity"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        annotate_corr(ax, corrs.get(ds, {}))
        ax.set_xlabel("Protein length (aa)")
        ax.set_title(meta["label"], fontweight="bold")
    axes[0].set_ylabel("Mean positive probe intensity")
    fig.suptitle(
        "RNAcompete: Mean Positive Intensity vs Protein Length",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, out_path)


def fig_spectrum_samples(
    samples: dict[str, pd.DataFrame],
    selected: dict[str, list[str]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (ds, meta) in zip(axes, DATASETS.items()):
        df = samples[ds]
        prots = selected.get(ds, df["protein_name"].unique().tolist())
        for i, prot in enumerate(prots):
            sub = df[df["protein_name"] == prot].sort_values("target_percentile")
            color = PROTEIN_COLORS[i % len(PROTEIN_COLORS)]
            ax.plot(
                sub["target_percentile"],
                sub["log_intensity"],
                "o-",
                ms=4,
                lw=1.2,
                color=color,
                alpha=0.9,
                label=prot,
            )
        ax.axhline(0, ls=":", color=GRAY, lw=1, alpha=0.7)
        ax.text(2, 0.05, "log 0 (= Smin)", fontsize=8, color=GRAY)
        ax.set_xlabel("Intensity percentile (modal-length probes)")
        ax.set_title(meta["label"], fontweight="bold")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    axes[0].set_ylabel("log₁₀(probe_intensity)")
    n_sel = max(len(selected.get(ds, [])) for ds in DATASETS)
    overlay_title = (
        "Spectrum Samples: 100 Probes at Evenly Spaced Percentiles"
        + (f" ({n_sel} selected proteins)" if n_sel > 3 else " (selected proteins)")
    )
    fig.suptitle(overlay_title, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize RNAcompete intensity spectrum results")
    p.add_argument(
        "--results_dir",
        default="results/rnacompete_intensity_spectrum",
        help="Root dir from script 27",
    )
    p.add_argument("--out_dir", default="figures", help="Figure output directory")
    p.add_argument(
        "--figure_prefix",
        default="rnacompete",
        help="Output filename prefix (e.g. rnacompete_domain_diverse)",
    )
    args = p.parse_args()

    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    summaries = {ds: load_summary(results_root, ds) for ds in DATASETS}
    corrs = {ds: load_corr(results_root, ds) for ds in DATASETS}
    samples = {ds: load_samples(results_root, ds) for ds in DATASETS}
    selected = {ds: load_selected(results_root, ds) for ds in DATASETS}

    print("Generating figures...")
    prefix = args.figure_prefix
    fig_length_vs_intensity(
        summaries, corrs, selected, out_dir / f"{prefix}_length_vs_intensity.png"
    )
    fig_spectrum_samples(samples, selected, out_dir / f"{prefix}_intensity_spectrum.png")

    for ds, meta in DATASETS.items():
        sel_df = load_domain_selection(results_root, ds)
        domain_map = {}
        if sel_df is not None:
            domain_map = dict(zip(sel_df["protein_name"], sel_df["domain_architecture"]))
        prots = selected.get(ds, [])
        if len(prots) > 3:
            stem = ds.replace("rnacompete_", "")
            fig_spectrum_faceted(
                samples[ds],
                prots,
                domain_map,
                meta["label"],
                out_dir / f"{prefix}_intensity_spectrum_faceted_{stem}.png",
            )
    print("Done.")


if __name__ == "__main__":
    main()
