#!/usr/bin/env python3
"""
15_analyze_training.py
Training dynamics and per-protein AUROC analysis across all model versions.

Panels produced:
  A. Training dynamics  — loss curves, AUROC/AUPRC over epochs, val vs test gap
  B. Per-protein        — AUROC distribution, split by dataset_source, outlier table

Usage:
  python scripts/15_analyze_training.py \
      --results_dir results/generalized \
      --output_dir results/analysis/training \
      [--models v1_mlp v2_cnn v3_esm2 v3b_esm2_cnn v3c_esm2_residue] \
      [--format png]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Palette / style ──────────────────────────────────────────────────────────
COLORS = {
    "v1_mlp":              "#4e79a7",
    "v2_cnn":              "#f28e2b",
    "v3_esm2":             "#e15759",
    "v3b_esm2_cnn":        "#76b7b2",
    "v3c_esm2_residue":    "#59a14f",
}
DATASET_COLORS = {
    "htr_selex_25907":  "#4e79a7",
    "htr_selex_47428":  "#76b7b2",
    "rbns":             "#f28e2b",
    "eclip":            "#e15759",
    "unknown":          "#bab0ac",
}
ZHMOLGRAPH = {"auroc": 0.798, "auprc": 0.820}

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
    "v3_esm2":          "V3 ESM-2 mean-pool",
    "v3b_esm2_cnn":     "V3b CNN+ESM-2",
    "v3c_esm2_residue": "V3c ESM-2 residue",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_results(results_dir: str, models: list[str]) -> dict:
    data = {}
    for m in models:
        fname = MODEL_FILES.get(m)
        if fname is None:
            print(f"  [SKIP] unknown model key: {m}", file=sys.stderr)
            continue
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] missing: {path}", file=sys.stderr)
            continue
        with open(path) as fh:
            data[m] = json.load(fh)
        print(f"  [OK]   {m} — {len(data[m].get('history', []))} epochs logged")
    return data


def save_fig(fig: plt.Figure, out_dir: str, name: str, fmt: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ── Panel A: Training dynamics ────────────────────────────────────────────────

def plot_training_dynamics(data: dict, out_dir: str, fmt: str):
    """Loss, AUROC, AUPRC over epochs; one line per model with val vs test gap annotation."""

    models_with_history = {m: d for m, d in data.items() if d.get("history")}
    if not models_with_history:
        print("  [WARN] no epoch history found — skipping training dynamics panel")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_loss, ax_auroc, ax_auprc = axes

    for m, d in models_with_history.items():
        hist = d["history"]
        epochs = [h["epoch"] for h in hist]
        color  = COLORS.get(m, "#888888")
        label  = MODEL_LABELS.get(m, m)

        if "train_loss" in hist[0]:
            ax_loss.plot(epochs, [h["train_loss"] for h in hist],
                         color=color, lw=1.8, label=label)

        if "val_auroc" in hist[0]:
            val_aurocs = [h["val_auroc"] for h in hist]
            ax_auroc.plot(epochs, val_aurocs, color=color, lw=1.8, label=label)
            # Mark best epoch
            best_ep = d.get("best_epoch") or (epochs[int(np.argmax(val_aurocs))])
            best_v  = max(val_aurocs)
            ax_auroc.scatter([best_ep], [best_v], color=color, s=60, zorder=5)
            # Test point
            test_auroc = d.get("test_metrics", {}).get("auroc")
            if test_auroc is not None:
                ax_auroc.axhline(test_auroc, color=color, lw=1, ls="--", alpha=0.6)

        if "val_auprc" in hist[0]:
            val_auprcs = [h["val_auprc"] for h in hist]
            ax_auprc.plot(epochs, val_auprcs, color=color, lw=1.8, label=label)
            test_auprc = d.get("test_metrics", {}).get("auprc")
            if test_auprc is not None:
                ax_auprc.axhline(test_auprc, color=color, lw=1, ls="--", alpha=0.6)

    # ZHMolGraph reference lines
    ax_auroc.axhline(ZHMOLGRAPH["auroc"], color="black", lw=1.5, ls=":", alpha=0.8,
                     label=f"ZHMolGraph {ZHMOLGRAPH['auroc']:.3f}")
    ax_auprc.axhline(ZHMOLGRAPH["auprc"], color="black", lw=1.5, ls=":", alpha=0.8,
                     label=f"ZHMolGraph {ZHMOLGRAPH['auprc']:.3f}")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss (BCE)")
    ax_loss.set_title("A.  Training loss")
    ax_loss.legend(fontsize=8, loc="upper right")

    ax_auroc.set_xlabel("Epoch")
    ax_auroc.set_ylabel("AUROC")
    ax_auroc.set_title("A.  Validation AUROC\n(dashed = test; dot = best epoch)")
    ax_auroc.legend(fontsize=8, loc="lower right")

    ax_auprc.set_xlabel("Epoch")
    ax_auprc.set_ylabel("AUPRC")
    ax_auprc.set_title("A.  Validation AUPRC\n(dashed = test)")
    ax_auprc.legend(fontsize=8, loc="lower right")

    fig.suptitle("Training dynamics — all model versions", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "A_training_dynamics", fmt)

    # Supplementary: val vs test gap table
    _save_gap_table(data, out_dir)


def _save_gap_table(data: dict, out_dir: str):
    rows = []
    for m, d in data.items():
        hist = d.get("history", [])
        best_val_auroc = d.get("best_val_auroc") or (
            max(h["val_auroc"] for h in hist) if hist and "val_auroc" in hist[0] else None)
        best_val_auprc = d.get("best_val_auprc") or (
            max(h["val_auprc"] for h in hist) if hist and "val_auprc" in hist[0] else None)
        test = d.get("test_metrics", {})
        test_auroc = test.get("auroc")
        test_auprc = test.get("auprc")

        rows.append({
            "model":            MODEL_LABELS.get(m, m),
            "best_val_auroc":   round(best_val_auroc, 4) if best_val_auroc else None,
            "test_auroc":       round(test_auroc, 4)     if test_auroc     else None,
            "gap_auroc":        round((best_val_auroc or 0) - (test_auroc or 0), 4),
            "best_val_auprc":   round(best_val_auprc, 4) if best_val_auprc else None,
            "test_auprc":       round(test_auprc, 4)     if test_auprc     else None,
            "gap_auprc":        round((best_val_auprc or 0) - (test_auprc or 0), 4),
            "best_epoch":       d.get("best_epoch"),
            "n_epochs":         len(hist),
        })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "A_val_test_gap.tsv")
    df.to_csv(path, sep="\t", index=False)
    print(f"  → saved {path}")
    print(df.to_string(index=False))


# ── Panel B: Per-protein analysis ─────────────────────────────────────────────

def plot_per_protein(data: dict, out_dir: str, fmt: str):
    """AUROC distribution per model, split by dataset_source; outlier tables."""

    models_with_pp = {m: d for m, d in data.items()
                      if d.get("per_protein") and len(d["per_protein"]) > 0}
    if not models_with_pp:
        print("  [WARN] no per_protein data found — skipping per-protein panel")
        return

    # ── B1: AUROC distribution (violin / strip) ──────────────────────────────
    fig_b1, ax = plt.subplots(figsize=(max(8, len(models_with_pp) * 1.8), 5))

    xs, ys, colors, labels = [], [], [], []
    for i, (m, d) in enumerate(models_with_pp.items()):
        aurocs = [p["auroc"] for p in d["per_protein"]]
        xs.extend([i] * len(aurocs))
        ys.extend(aurocs)
        colors.append(COLORS.get(m, "#888888"))
        labels.append(MODEL_LABELS.get(m, m))

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="random (0.5)")
    ax.axhline(ZHMOLGRAPH["auroc"], color="black", lw=1.2, ls=":",
               label=f"ZHMolGraph {ZHMOLGRAPH['auroc']:.3f}")

    for i, (m, d) in enumerate(models_with_pp.items()):
        aurocs = [p["auroc"] for p in d["per_protein"]]
        col = COLORS.get(m, "#888888")
        parts = ax.violinplot([aurocs], positions=[i], widths=0.6,
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(col)
            pc.set_alpha(0.5)
        parts["cmedians"].set_color(col)
        ax.scatter([i + np.random.uniform(-0.15, 0.15) for _ in aurocs],
                   aurocs, s=12, alpha=0.35, color=col)

    ax.set_xticks(range(len(models_with_pp)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Per-protein AUROC")
    ax.set_title("B.  Per-protein AUROC distribution")
    ax.legend(fontsize=8)
    fig_b1.tight_layout()
    save_fig(fig_b1, out_dir, "B1_per_protein_violin", fmt)

    # ── B2: By dataset_source (best model = v2_cnn, fallback to first) ────────
    best_model = "v2_cnn" if "v2_cnn" in models_with_pp else list(models_with_pp.keys())[-1]
    pp_df = pd.DataFrame(models_with_pp[best_model]["per_protein"])
    if "dataset" not in pp_df.columns and "dataset_source" in pp_df.columns:
        pp_df = pp_df.rename(columns={"dataset_source": "dataset"})
    if "dataset" in pp_df.columns:
        fig_b2, ax2 = plt.subplots(figsize=(9, 5))
        datasets = sorted(pp_df["dataset"].unique())
        for j, ds in enumerate(datasets):
            grp = pp_df[pp_df["dataset"] == ds]["auroc"].values
            col = DATASET_COLORS.get(ds, "#bab0ac")
            parts = ax2.violinplot([grp], positions=[j], widths=0.6,
                                   showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(0.55)
            parts["cmedians"].set_color(col)
            ax2.scatter([j + np.random.uniform(-0.15, 0.15) for _ in grp],
                        grp, s=14, alpha=0.4, color=col)
            ax2.text(j, pp_df["auroc"].min() - 0.03,
                     f"n={len(grp)}", ha="center", fontsize=8, color="gray")
        ax2.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax2.axhline(ZHMOLGRAPH["auroc"], color="black", lw=1.2, ls=":")
        ax2.set_xticks(range(len(datasets)))
        ax2.set_xticklabels(datasets, rotation=15, ha="right")
        ax2.set_ylabel("Per-protein AUROC")
        ax2.set_title(f"B.  Per-protein AUROC by dataset source ({MODEL_LABELS.get(best_model, best_model)})")
        fig_b2.tight_layout()
        save_fig(fig_b2, out_dir, "B2_per_protein_by_dataset", fmt)

    # ── B3: Outlier tables (easy / hard) ─────────────────────────────────────
    _save_outlier_tables(models_with_pp, out_dir)

    # ── B4: Cross-model per-protein comparison heatmap ───────────────────────
    _plot_cross_model_heatmap(models_with_pp, out_dir, fmt)


def _save_outlier_tables(models_with_pp: dict, out_dir: str):
    for m, d in models_with_pp.items():
        df = pd.DataFrame(d["per_protein"])
        if "dataset" not in df.columns and "dataset_source" in df.columns:
            df = df.rename(columns={"dataset_source": "dataset"})
        df = df.sort_values("auroc")
        hard = df.head(10)[["protein", "dataset", "auroc", "n"]].round(4)
        easy = df.tail(10).sort_values("auroc", ascending=False)[["protein", "dataset", "auroc", "n"]].round(4)
        os.makedirs(out_dir, exist_ok=True)
        hard.to_csv(os.path.join(out_dir, f"B3_hard_proteins_{m}.tsv"), sep="\t", index=False)
        easy.to_csv(os.path.join(out_dir, f"B3_easy_proteins_{m}.tsv"), sep="\t", index=False)
    print(f"  → saved outlier tables to {out_dir}/B3_*")


def _plot_cross_model_heatmap(models_with_pp: dict, out_dir: str, fmt: str):
    """For proteins present in multiple models, show AUROC heatmap."""
    all_dfs = []
    for m, d in models_with_pp.items():
        df = pd.DataFrame(d["per_protein"])[["protein", "auroc"]]
        df = df.rename(columns={"auroc": MODEL_LABELS.get(m, m)})
        all_dfs.append(df.set_index("protein"))
    if len(all_dfs) < 2:
        return
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = merged.join(df, how="outer")
    merged = merged.dropna()
    if len(merged) < 3:
        return

    merged = merged.sort_values(merged.columns[0])
    fig, ax = plt.subplots(figsize=(max(6, len(merged.columns) * 1.5),
                                    min(20, max(4, len(merged) * 0.3))))
    im = ax.imshow(merged.values.T, aspect="auto", cmap="RdYlGn",
                   vmin=0.4, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(merged.columns)))
    ax.set_yticklabels(merged.columns, fontsize=9)
    if len(merged) <= 60:
        ax.set_xticks(range(len(merged)))
        ax.set_xticklabels(merged.index.tolist(), rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"{len(merged)} proteins (sorted by V2 AUROC)", fontsize=9)
    plt.colorbar(im, ax=ax, label="Per-protein AUROC", shrink=0.6)
    ax.set_title("B.  Cross-model per-protein AUROC heatmap")
    fig.tight_layout()
    save_fig(fig, out_dir, "B4_cross_model_heatmap", fmt)


# ── Summary bar chart ─────────────────────────────────────────────────────────

def plot_summary_bar(data: dict, out_dir: str, fmt: str):
    models   = list(data.keys())
    test_auc = [data[m].get("test_metrics", {}).get("auroc", float("nan")) for m in models]
    test_apc = [data[m].get("test_metrics", {}).get("auprc", float("nan")) for m in models]

    x   = np.arange(len(models))
    w   = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for i, (m, auc, apc) in enumerate(zip(models, test_auc, test_apc)):
        col = COLORS.get(m, "#888888")
        ax1.bar(i, auc, width=w, color=col, label=MODEL_LABELS.get(m, m), alpha=0.85)
        ax2.bar(i, apc, width=w, color=col, alpha=0.85)

    for ax, ref, metric in [(ax1, ZHMOLGRAPH["auroc"], "AUROC"),
                             (ax2, ZHMOLGRAPH["auprc"], "AUPRC")]:
        ax.axhline(ref, color="black", lw=1.5, ls=":", alpha=0.9,
                   label=f"ZHMolGraph {ref:.3f}")
        ax.set_ylim(0.4, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                            rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(f"Test {metric}")
        ax.set_title(f"Summary — Test {metric}")
        ax.legend(fontsize=8)

    fig.suptitle("Model comparison — test set performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "S_summary_bar", fmt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results/generalized",
                        help="Directory with model result JSON files")
    parser.add_argument("--output_dir",  default="results/analysis/training",
                        help="Output directory for plots and tables")
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_FILES.keys()),
                        help="Model keys to include (default: all)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output figure format")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Training Dynamics + Per-Protein Analysis")
    print(f"{'='*60}")
    print(f"  results_dir : {args.results_dir}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  models      : {args.models}\n")

    data = load_results(args.results_dir, args.models)
    if not data:
        sys.exit("No result files found. Check --results_dir.")

    print("\n--- Panel A: Training dynamics ---")
    plot_training_dynamics(data, args.output_dir, args.format)

    print("\n--- Panel B: Per-protein AUROC ---")
    plot_per_protein(data, args.output_dir, args.format)

    print("\n--- Summary bar chart ---")
    plot_summary_bar(data, args.output_dir, args.format)

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
