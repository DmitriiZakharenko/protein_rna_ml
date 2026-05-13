#!/usr/bin/env python3
"""
19_compare_models.py
Model comparison framework and leaderboard generator.

Reads result JSON files (single-seed or multi-seed summaries), computes a
normalized leaderboard score, and produces:
  1. Leaderboard table (console + TSV)
  2. Radar / spider chart comparing models across metrics
  3. Per-protein AUROC CDF comparison
  4. Val vs test gap summary with overfitting flags

Leaderboard score (composite):
  score = 0.50 * test_auroc + 0.30 * test_auprc + 0.20 * per_protein_median_auroc
  (ZHMolGraph reference: 0.50*0.798 + 0.30*0.820 + 0.20*0.798 = 0.804)

Usage:
  # Compare all single-seed results:
  python scripts/19_compare_models.py \
      --results_dir results/generalized \
      --output_dir results/analysis/leaderboard

  # Include multi-seed summaries (prefer these if available):
  python scripts/19_compare_models.py \
      --results_dir results/generalized \
      --multiseed_dir results/multiseed \
      --output_dir results/analysis/leaderboard
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Config ────────────────────────────────────────────────────────────────────

ZHMOLGRAPH = {
    "test_auroc": 0.798,
    "test_auprc": 0.820,
    "per_protein_median": 0.798,  # approximate, not published directly
    "label": "ZHMolGraph (target)",
}

LEADERBOARD_WEIGHTS = {
    "test_auroc":         0.50,
    "test_auprc":         0.30,
    "per_protein_median": 0.20,
}

MODEL_FILES = {
    "v1_mlp":           "v1_mlp_results.json",
    "v2_cnn":           "v2_cnn_results.json",
    "v3_esm2":          "v3_esm2_results.json",
    "v3b_esm2_cnn":     "v3b_esm2_cnn_results.json",
    "v3c_esm2_residue": "v3c_esm2_residue_results.json",
}
MODEL_LABELS = {
    "v1_mlp":           "V1 MLP",
    "v2_cnn":           "V2 CNN (anchor)",
    "v3_esm2":          "V3 ESM-2",
    "v3b_esm2_cnn":     "V3b CNN+ESM-2",
    "v3c_esm2_residue": "V3c ESM-2 residue",
}
COLORS = {
    "v1_mlp":              "#4e79a7",
    "v2_cnn":              "#f28e2b",
    "v3_esm2":             "#e15759",
    "v3b_esm2_cnn":        "#76b7b2",
    "v3c_esm2_residue":    "#59a14f",
    "ZHMolGraph (target)": "#000000",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Data extraction ───────────────────────────────────────────────────────────

def load_single_seed(results_dir: str) -> dict[str, dict]:
    records = {}
    for key, fname in MODEL_FILES.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            d = json.load(fh)
        test = d.get("test_metrics", {})
        pp   = d.get("per_protein", [])
        pp_aurocs = [p["auroc"] for p in pp if "auroc" in p]
        records[key] = {
            "label":              MODEL_LABELS.get(key, key),
            "test_auroc":         test.get("auroc"),
            "test_auprc":         test.get("auprc"),
            "best_val_auroc":     d.get("best_val_auroc"),
            "best_val_auprc":     d.get("best_val_auprc"),
            "per_protein_median": float(np.median(pp_aurocs)) if pp_aurocs else None,
            "per_protein_min":    float(np.min(pp_aurocs))    if pp_aurocs else None,
            "n_proteins":         len(pp_aurocs),
            "n_seeds":            1,
            "auroc_std":          None,
            "auprc_std":          None,
            "is_bug_affected":    True,   # pre-Phase-1 models are bug-affected
            "source_file":        path,
            "per_protein_list":   pp_aurocs,
        }
    return records


def load_multiseed(multiseed_dir: str, existing: dict[str, dict]) -> dict[str, dict]:
    if not multiseed_dir or not os.path.exists(multiseed_dir):
        return existing
    for model_dir in sorted(Path(multiseed_dir).iterdir()):
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as fh:
            s = json.load(fh)
        key = model_dir.name  # e.g. "v2_cnn"
        agg = s.get("aggregate", {})
        record = existing.get(key, {})
        record.update({
            "label":          MODEL_LABELS.get(key, key) + " (multi-seed)",
            "n_seeds":        s.get("n_runs", 1),
            "is_bug_affected": False,   # multi-seed runs are post-Phase-1 clean
        })
        for metric in ["test_auroc", "test_auprc", "per_protein_median"]:
            if metric in agg:
                record[metric]            = agg[metric].get("mean")
                record[f"{metric}_std"]   = agg[metric].get("std")
        existing[key + "_clean"] = record
        print(f"  [OK] multi-seed: {key} ({record['n_seeds']} seeds)")
    return existing


# ── Leaderboard score ─────────────────────────────────────────────────────────

def compute_score(record: dict) -> float | None:
    vals = [record.get(m) for m in LEADERBOARD_WEIGHTS]
    if any(v is None for v in vals):
        return None
    return sum(LEADERBOARD_WEIGHTS[m] * record[m] for m in LEADERBOARD_WEIGHTS)


# ── Leaderboard table ─────────────────────────────────────────────────────────

def build_leaderboard(records: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, r in records.items():
        score = compute_score(r)
        rows.append({
            "model":              r.get("label", key),
            "test_auroc":         r.get("test_auroc"),
            "test_auprc":         r.get("test_auprc"),
            "per_protein_median": r.get("per_protein_median"),
            "best_val_auroc":     r.get("best_val_auroc"),
            "val_test_gap":       (r.get("best_val_auroc", 0) or 0) - (r.get("test_auroc", 0) or 0),
            "leaderboard_score":  score,
            "n_seeds":            r.get("n_seeds", 1),
            "is_bug_affected":    r.get("is_bug_affected", False),
            "auroc_std":          r.get("auroc_std"),
        })
    # Add ZHMolGraph reference
    zh_score = sum(LEADERBOARD_WEIGHTS[m] * ZHMOLGRAPH[m] for m in LEADERBOARD_WEIGHTS)
    rows.append({
        "model":              ZHMOLGRAPH["label"],
        "test_auroc":         ZHMOLGRAPH["test_auroc"],
        "test_auprc":         ZHMOLGRAPH["test_auprc"],
        "per_protein_median": ZHMOLGRAPH["per_protein_median"],
        "best_val_auroc":     None,
        "val_test_gap":       None,
        "leaderboard_score":  zh_score,
        "n_seeds":            None,
        "is_bug_affected":    False,
        "auroc_std":          None,
    })
    df = pd.DataFrame(rows).sort_values("leaderboard_score", ascending=False)
    df = df.reset_index(drop=True)
    df.index += 1
    return df


def print_leaderboard(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print(f"  LEADERBOARD  (score = 0.50×AUROC + 0.30×AUPRC + 0.20×pp-median)")
    print(f"{'='*80}")
    display_cols = ["model", "test_auroc", "test_auprc", "per_protein_median",
                    "leaderboard_score", "val_test_gap", "is_bug_affected"]
    sub = df[display_cols].copy()
    for col in ["test_auroc", "test_auprc", "per_protein_median", "leaderboard_score"]:
        sub[col] = sub[col].apply(lambda x: f"{x:.4f}" if x is not None else "—")
    sub["val_test_gap"] = sub["val_test_gap"].apply(
        lambda x: f"{x:+.4f}" if x is not None else "—")
    print(sub.to_string())
    print(f"{'='*80}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_leaderboard_bar(df: pd.DataFrame, records: dict, out_dir: str):
    if not HAS_MATPLOTLIB:
        return
    metrics = ["test_auroc", "test_auprc", "per_protein_median"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    model_order = df["model"].tolist()
    for ax, metric in zip(axes, metrics):
        vals = df.set_index("model")[metric]
        colors = [COLORS.get(
            next((k for k, r in records.items() if r.get("label", k) == m), m),
            "#bab0ac"
        ) if m != ZHMOLGRAPH["label"] else "#000000" for m in model_order]

        ax.barh(model_order[::-1], vals.reindex(model_order[::-1]), color=colors[::-1],
                alpha=0.8, edgecolor="none")
        ax.axvline(ZHMOLGRAPH.get(metric, 0.798), color="black", lw=1.2, ls=":",
                   label=f"ZHMolGraph {ZHMOLGRAPH.get(metric, 0.798):.3f}")
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(f"Test {metric}")
        ax.legend(fontsize=8)
        ax.set_xlim(0.4, 1.0)

    fig.suptitle("Model leaderboard — test set metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "leaderboard_bar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def plot_cdf_comparison(records: dict, out_dir: str):
    if not HAS_MATPLOTLIB:
        return
    models_with_pp = {k: r for k, r in records.items() if r.get("per_protein_list")}
    if not models_with_pp:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, r in models_with_pp.items():
        vals = sorted(r["per_protein_list"])
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        col  = COLORS.get(key, "#888888")
        ls   = "--" if r.get("is_bug_affected") else "-"
        label = r.get("label", key)
        if r.get("is_bug_affected"):
            label += " [bug-affected]"
        ax.plot(vals, cdf, color=col, lw=1.8, ls=ls, label=label)

    ax.axvline(ZHMOLGRAPH["test_auroc"], color="black", lw=1.2, ls=":", alpha=0.8,
               label=f"ZHMolGraph {ZHMOLGRAPH['test_auroc']:.3f}")
    ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5, label="random (0.5)")
    ax.set_xlabel("Per-protein AUROC")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Per-protein AUROC CDF — model comparison")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "cdf_per_protein.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def plot_radar(df: pd.DataFrame, out_dir: str):
    """Spider chart comparing models across AUROC, AUPRC, pp-median, 1-gap."""
    if not HAS_MATPLOTLIB:
        return
    metrics = ["test_auroc", "test_auprc", "per_protein_median"]
    metric_labels = ["AUROC", "AUPRC", "pp-median AUROC"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for _, row in df.iterrows():
        label = row["model"]
        vals  = [row.get(m) for m in metrics]
        if any(v is None for v in vals):
            continue
        vals = [float(v) for v in vals]
        vals_plot = vals + vals[:1]
        col = COLORS.get(label, "#888888")
        ls  = "--" if row.get("is_bug_affected") else "-"
        lw  = 2.5 if label == ZHMOLGRAPH["label"] else 1.8
        ax.plot(angles, vals_plot, color=col, lw=lw, ls=ls)
        ax.fill(angles, vals_plot, color=col, alpha=0.06)
        ax.scatter(angles[:-1], vals, color=col, s=40, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=7)
    ax.set_title("Radar chart — model metrics", pad=18, fontsize=12, fontweight="bold")

    # Legend
    handles = []
    from matplotlib.lines import Line2D
    for _, row in df.iterrows():
        col = COLORS.get(row["model"], "#888888")
        ls  = "--" if row.get("is_bug_affected") else "-"
        handles.append(Line2D([0], [0], color=col, lw=1.6, ls=ls, label=row["model"]))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "radar_chart.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir",   default="results/generalized",
                        help="Dir with single-seed result JSONs")
    parser.add_argument("--multiseed_dir", default=None,
                        help="Dir with multi-seed summaries (optional, preferred over single-seed)")
    parser.add_argument("--output_dir",    default="results/analysis/leaderboard",
                        help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Model Comparison + Leaderboard")
    print(f"{'='*60}")

    records = load_single_seed(args.results_dir)
    if not records:
        sys.exit(f"No result files found in {args.results_dir}")

    if args.multiseed_dir:
        records = load_multiseed(args.multiseed_dir, records)

    os.makedirs(args.output_dir, exist_ok=True)

    df = build_leaderboard(records)
    print_leaderboard(df)

    # Save TSV
    tsv_path = os.path.join(args.output_dir, "leaderboard.tsv")
    df.to_csv(tsv_path, sep="\t", float_format="%.5f")
    print(f"\n  → saved {tsv_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, "leaderboard.json")
    with open(json_path, "w") as fh:
        json.dump({
            "leaderboard": df.to_dict(orient="records"),
            "leaderboard_formula": "0.50*test_auroc + 0.30*test_auprc + 0.20*per_protein_median",
            "zhmolgraph_target": ZHMOLGRAPH,
        }, fh, indent=2, default=str)
    print(f"  → saved {json_path}")

    # Plots
    print("\n--- Generating plots ---")
    plot_leaderboard_bar(df, records, args.output_dir)
    plot_cdf_comparison(records, args.output_dir)
    plot_radar(df, args.output_dir)

    # Val-test gap flags
    gap_df = df[df["val_test_gap"].notna()].copy()
    gap_df["val_test_gap"] = gap_df["val_test_gap"].astype(float)
    overfit = gap_df[gap_df["val_test_gap"] > 0.03]
    if len(overfit) > 0:
        print(f"\n  [WARN] Models with val→test AUROC gap > 0.03 (potential overfitting):")
        for _, row in overfit.iterrows():
            print(f"    {row['model']}: gap = {row['val_test_gap']:+.4f}")

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
