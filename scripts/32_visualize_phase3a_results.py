#!/usr/bin/env python3
"""
32_visualize_phase3a_results.py

Generate README figures from committed Phase 3A + external validation JSON/TSV.
No model or training data required.

Outputs (figures/):
  phase3a_v2_scale_comparison.png   v2 (169 prot) vs v3a (494 prot) test metrics
  phase3a_per_protein_auroc.png     55 held-out test proteins (v3a)
  phase2_model_comparison.png       V1–V3c + V2-v3a test AUROC (refreshed)
  external_eval_comparison.png      curated vs expanded literature eval
  external_score_distributions.png score by label / example class

Usage:
    python scripts/32_visualize_phase3a_results.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
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
})

BLUE = "#2271B5"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#8F8F8F"
GOLD = "#E69F00"
TEAL = "#17A2B8"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def fig_scale_comparison(v2_phase2: dict, v3a: dict, out_path: str) -> None:
    v2 = v2_phase2["v2_cnn_raw_seq"]
    labels = ["Test AUROC", "Test AUPRC", "Per-protein\nmedian AUROC"]
    v2_vals = [
        v2["test_auroc"],
        v2["test_auprc"],
        v2["per_protein_test"]["median_auroc"],
    ]
    v3a_vals = [
        v3a["test_metrics"]["auroc"],
        v3a["test_metrics"]["auprc"],
        v3a["per_protein_summary"]["median"],
    ]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, v2_vals, w, label="V2 · generalized_v2 (169 prot)", color=GRAY, alpha=0.85)
    b2 = ax.bar(x + w / 2, v3a_vals, w, label="V2 · generalized_v3a (494 prot)", color=GREEN, alpha=0.9)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Phase 3A — V2 CNN before vs after RNAcompete scale-up", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(0.5, ls="--", color=GRAY, lw=0.8, alpha=0.6)
    fig.tight_layout()
    save(fig, out_path)


def fig_v3a_per_protein(v3a: dict, out_path: str) -> None:
    pp = v3a.get("per_protein", [])
    if not pp:
        print("  [skip] no v3a per-protein data")
        return

    df = pd.DataFrame(pp).sort_values("auroc", ascending=True)
    colors = [GREEN if a >= 0.80 else BLUE if a >= 0.65 else ORANGE for a in df["auroc"]]

    fig, ax = plt.subplots(figsize=(9, max(6, len(df) * 0.22)))
    ax.barh(df["protein"], df["auroc"], color=colors, alpha=0.88)
    overall = v3a["test_metrics"]["auroc"]
    median = v3a["per_protein_summary"]["median"]
    ax.axvline(0.5, ls="--", color=GRAY, lw=1.0, alpha=0.7)
    ax.axvline(overall, ls="--", color=GREEN, lw=1.3, alpha=0.9, label=f"Overall test {overall:.3f}")
    ax.axvline(median, ls=":", color=BLUE, lw=1.3, alpha=0.9, label=f"Median {median:.3f}")

    ax.set_xlim(0.15, 1.02)
    ax.set_xlabel("AUROC")
    ax.set_title(f"V2 on v3a — Per-Protein Test AUROC ({len(df)} proteins)", fontweight="bold")
    g = mpatches.Patch(facecolor=GREEN, alpha=0.88, label="≥ 0.80")
    b = mpatches.Patch(facecolor=BLUE, alpha=0.88, label="0.65 – 0.80")
    o = mpatches.Patch(facecolor=ORANGE, alpha=0.88, label="< 0.65")
    ax.legend(handles=[g, b, o,
                       plt.Line2D([0], [0], ls="--", color=GREEN, lw=1.3, label=f"Overall {overall:.3f}"),
                       plt.Line2D([0], [0], ls=":", color=BLUE, lw=1.3, label=f"Median {median:.3f}")],
              fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out_path)


def fig_phase2_updated(phase2_path: str, v3a_path: str, out_path: str) -> None:
    d = load_json(phase2_path)
    v3a = load_json(v3a_path)

    models = [
        ("V1 MLP", d["v1_mlp_kmer"]["test_auroc"], d["v1_mlp_kmer"]["test_auprc"], GRAY),
        ("V2 CNN\n(v2 data)", d["v2_cnn_raw_seq"]["test_auroc"], d["v2_cnn_raw_seq"]["test_auprc"], BLUE),
        ("V2 CNN\n(v3a data)", v3a["test_metrics"]["auroc"], v3a["test_metrics"]["auprc"], GREEN),
        ("V3 ESM-2", d["v3_esm2_frozen_meanpool"]["test_auroc"], d["v3_esm2_frozen_meanpool"]["test_auprc"], ORANGE),
        ("V3b", d["v3b_esm2_plus_cnn"]["test_auroc"], d["v3b_esm2_plus_cnn"]["test_auprc"], PURPLE),
        ("V3c", d["v3c_esm2_residue_cnn"]["test_auroc"], d["v3c_esm2_residue_cnn"]["test_auprc"], GOLD),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [m[0] for m in models]
    x = np.arange(len(names))

    for ax, idx, ylab, title in [
        (axes[0], 1, "Test AUROC", "Generalized Models — Test AUROC"),
        (axes[1], 2, "Test AUPRC", "Generalized Models — Test AUPRC"),
    ]:
        vals = [m[idx] for m in models]
        cols = [m[3] for m in models]
        bars = ax.bar(x, vals, color=cols, alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(0, 0.92)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold")
        if idx == 1:
            ax.axhline(0.5, ls="--", color=GRAY, lw=0.8, alpha=0.6)

    fig.suptitle("Phase 2 baselines + Phase 3A V2 (same architecture, scaled data)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, out_path)


def fig_external_comparison(ext_json: dict, out_path: str) -> None:
    curated = ext_json["stratified_v2"]["curated_only"]
    overall = ext_json["v2_cnn"]
    pos_rate = ext_json["pos_rate"]
    rand_auprc = ext_json["random_auprc_baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # AUROC
    ax = axes[0]
    names = ["Curated\n(159 pairs)", "Expanded\n(540 pairs)"]
    vals = [curated["auroc"], overall["auroc"]]
    bars = ax.bar(names, vals, color=[BLUE, TEAL], alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(0.5, ls="--", color=GRAY, lw=0.8, alpha=0.7, label="Random AUROC (0.5)")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("AUROC")
    ax.set_title("Literature External Validation — AUROC", fontweight="bold")
    ax.legend(fontsize=8)

    # AUPRC with random baseline
    ax = axes[1]
    auprc_vals = [curated["auprc"], overall["auprc"]]
    baselines = [0.717, rand_auprc]
    bars = ax.bar(names, auprc_vals, color=[BLUE, TEAL], alpha=0.88)
    for i, (bar, v, b) in enumerate(zip(bars, auprc_vals, baselines)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}\n(+{v-b:.2f} vs random)",
                ha="center", va="bottom", fontsize=9)
        ax.hlines(b, bar.get_x() - 0.35, bar.get_x() + 0.35, colors=ORANGE, linestyles="--", lw=1.2)
    ax.text(0.02, 0.72, "— random AUPRC baseline (curated 72% pos)", transform=ax.transAxes, fontsize=7, color=ORANGE)
    ax.text(0.02, 0.24, f"— random AUPRC baseline (expanded {pos_rate:.0%} pos)", transform=ax.transAxes, fontsize=7, color=ORANGE)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUPRC")
    ax.set_title("Literature External Validation — AUPRC", fontweight="bold")

    fig.suptitle("V2 v3a checkpoint · sliding-window scoring · not comparable to SELEX test", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, out_path)


def fig_external_scores(scored_tsv: str, out_path: str) -> None:
    df = pd.read_csv(scored_tsv, sep="\t")
    df["group"] = np.where(
        df["example_class"] == "curated_positive", "Curated positive",
        np.where(df["example_class"] == "curated_negative", "Curated negative",
                 np.where(df["example_class"] == "generated_negative", "Generated negative", "Other")),
    )
    order = ["Curated positive", "Generated negative", "Curated negative"]
    plot_df = df[df["group"].isin(order)]

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [plot_df.loc[plot_df["group"] == g, "prob_v2"].values for g in order]
    parts = ax.violinplot(data, positions=range(len(order)), showmeans=True, showmedians=False, widths=0.75)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([GREEN, TEAL, ORANGE][i])
        body.set_alpha(0.55)
    parts["cmeans"].set_color("black")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\n(n={(plot_df['group']==g).sum()})" for g in order], fontsize=9)
    ax.set_ylabel("Predicted P(bind)")
    ax.set_title("External Benchmark — Score Distributions by Example Type", fontweight="bold")
    ax.axhline(0.5, ls="--", color=GRAY, lw=0.8, alpha=0.6)

    for i, g in enumerate(order):
        sub = plot_df[plot_df["group"] == g]["prob_v2"]
        ax.text(i, 0.02, f"median={sub.median():.2f}", ha="center", transform=ax.get_xaxis_transform(), fontsize=8)

    fig.tight_layout()
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2_json", default="results/phase2_summary.json")
    parser.add_argument("--v3a_json", default="results/generalized/v3a_scale/v2_cnn_results.json")
    parser.add_argument("--external_json", default="results/external/v3a_v2_expanded/external_validation_v2.json")
    parser.add_argument("--external_scored", default="results/external/v3a_v2_expanded/external_pairs_scored.tsv")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    print("\n=== Phase 3A + external figures ===")
    phase2 = load_json(args.phase2_json)
    v3a = load_json(args.v3a_json)
    ext = load_json(args.external_json)

    fig_scale_comparison(phase2, v3a, os.path.join(args.out_dir, "phase3a_v2_scale_comparison.png"))
    fig_v3a_per_protein(v3a, os.path.join(args.out_dir, "phase3a_per_protein_auroc.png"))
    fig_phase2_updated(args.phase2_json, args.v3a_json, os.path.join(args.out_dir, "phase2_model_comparison.png"))
    fig_external_comparison(ext, os.path.join(args.out_dir, "external_eval_comparison.png"))
    fig_external_scores(args.external_scored, os.path.join(args.out_dir, "external_score_distributions.png"))
    print("Done.\n")


if __name__ == "__main__":
    main()
