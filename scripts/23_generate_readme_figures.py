#!/usr/bin/env python3
"""
23_generate_readme_figures.py
Generate all figures for README.md from saved result JSONs.

No models or datasets required — reads only result JSON files.

Figures produced:
  figures/phase1_validation.png        — Phase 1: per-dataset AUROC bar chart
  figures/phase2_model_comparison.png  — Phase 2: V1–V3c test AUROC + AUPRC
  figures/v2_training_curve.png        — V2 training loss + val AUROC over epochs
  figures/v2_per_protein_auroc.png     — V2 per-protein AUROC sorted bar chart
  figures/rnacompete_overview.png      — RNAcompete: organism AUROC + pp histogram

Usage:
    python scripts/23_generate_readme_figures.py
    python scripts/23_generate_readme_figures.py --results_dir results --out_dir figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

BLUE   = "#2271B5"
ORANGE = "#D55E00"
GREEN  = "#009E73"
PURPLE = "#CC79A7"
GRAY   = "#8F8F8F"
GOLD   = "#E69F00"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 1: Phase 1 validation ─────────────────────────────────────────────

def fig_phase1(phase1_path, out_path):
    d = load_json(phase1_path)
    datasets = {
        "HTR-SELEX\nPRJEB25907": d["datasets"]["htr_selex_prjeb25907"],
        "RBNS":                   d["datasets"]["rbns"],
        "HTR-SELEX\nPRJEB47428": d["datasets"]["htr_selex_prjeb47428"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: val AUROC comparison across models ──
    ax = axes[0]
    models = ["logistic_regression", "random_forest", "xgboost"]
    labels = ["Logistic Regression", "Random Forest", "XGBoost"]
    colors = [BLUE, GREEN, ORANGE]
    x      = np.arange(len(datasets))
    width  = 0.25

    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        vals = []
        for ds in datasets.values():
            v = ds["val"]["all_models"].get(model, {})
            vals.append(v.get("auroc", 0))
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.axhline(0.70, ls="--", color=GRAY, lw=1.2, label="Pass threshold (0.70)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(datasets.keys()))
    ax.set_ylim(0.50, 0.90)
    ax.set_ylabel("Val AUROC")
    ax.set_title("Phase 1 — Validation AUROC by Model", fontweight="bold")
    ax.legend(fontsize=9)

    # ── Right: val vs test AUROC for best model ──
    ax = axes[1]
    ds_names = list(datasets.keys())
    best_val  = []
    best_test = []
    for ds in datasets.values():
        best_model = ds["val"]["best_model"]
        best_val.append(ds["val"]["auroc"])
        best_test.append(ds["test"]["auroc"])

    x = np.arange(len(ds_names))
    w = 0.35
    ax.bar(x - w/2, best_val,  w, label="Val (best model)",  color=BLUE,   alpha=0.85)
    ax.bar(x + w/2, best_test, w, label="Test (best model)", color=ORANGE, alpha=0.85)
    ax.axhline(0.70, ls="--", color=GRAY, lw=1.2, label="Pass threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylim(0.50, 0.90)
    ax.set_ylabel("AUROC")
    ax.set_title("Phase 1 — Val vs Test (Best Model per Dataset)", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Phase 1: Per-Dataset Baseline Validation", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, out_path)


# ── Figure 2: Phase 2 model comparison ───────────────────────────────────────

def fig_phase2_comparison(results_dir, out_path):
    models_cfg = [
        ("V1 MLP\n(k-mer)",        "v1_mlp_results.json",         "test_metrics.auroc",  "test_metrics.auprc",  GRAY),
        ("V2 CNN\n(one-hot)",       "v2_cnn_results.json",         "test_metrics.auroc",  "test_metrics.auprc",  BLUE),
        ("V3 ESM-2\n(mean-pool)",   "v3_esm2_results.json",        "test_metrics.auroc",  "test_metrics.auprc",  ORANGE),
        ("V3b CNN+\nESM-2",         "v3b_esm2_cnn_results.json",   "test_metrics.auroc",  "test_metrics.auprc",  GREEN),
        ("V3c ESM-2\n(residue)",    "v3c_esm2_residue_results.json","test_metrics.auroc", "test_metrics.auprc",  PURPLE),
    ]

    def _get(d, dotpath):
        parts = dotpath.split(".")
        for p in parts:
            if isinstance(d, dict):
                d = d.get(p, 0)
            else:
                return 0
        return float(d) if d else 0

    names, aurocs, auprcs, colors = [], [], [], []
    for name, fname, auroc_key, auprc_key, color in models_cfg:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        d = load_json(path)
        names.append(name)
        aurocs.append(_get(d, auroc_key))
        auprcs.append(_get(d, auprc_key))
        colors.append(color)

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_auroc = ax.bar(x - width/2, aurocs, width, color=colors, alpha=0.9,
                        label="Test AUROC")
    bars_auprc = ax.bar(x + width/2, auprcs, width, color=colors, alpha=0.5,
                        hatch="//", label="Test AUPRC")

    # Value labels on bars
    for bar in bars_auroc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_auprc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.45, 0.78)
    ax.set_ylabel("Score")
    ax.set_title("Phase 2 — Test AUROC and AUPRC by Model Architecture\n(clean checkpoints, protein-aware split, 24 test proteins)",
                 fontweight="bold")
    ax.axhline(0.690, ls="--", color=BLUE, lw=1.0, alpha=0.6)

    # ZHMolGraph reference line
    ax.axhline(0.798, ls=":", color="black", lw=1.5, alpha=0.7, label="ZHMolGraph AUROC 0.798\n(different dataset — reference only)")

    # Legend
    solid_patch  = mpatches.Patch(facecolor=GRAY,  alpha=0.9,  label="AUROC (solid)")
    hatch_patch  = mpatches.Patch(facecolor=GRAY,  alpha=0.5,  hatch="//", label="AUPRC (hatched)")
    v2_line      = plt.Line2D([0], [0], ls="--", color=BLUE, lw=1.0, alpha=0.6, label="V2 AUROC baseline (0.690)")
    zh_line      = plt.Line2D([0], [0], ls=":", color="black", lw=1.5, alpha=0.7, label="ZHMolGraph ref 0.798")
    ax.legend(handles=[solid_patch, hatch_patch, v2_line, zh_line], fontsize=9, loc="lower right")

    fig.tight_layout()
    save(fig, out_path)


# ── Figure 3: V2 training curve ───────────────────────────────────────────────

def fig_v2_training(v2_path, out_path):
    d = load_json(v2_path)
    history = d.get("history", [])
    if not history:
        print("  [skip] no training history in v2_cnn_results.json")
        return

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_auroc  = [h["val_auroc"]  for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    l1, = ax1.plot(epochs, train_loss, color=ORANGE, lw=2, marker="o", ms=4,
                   label="Train loss")
    l2, = ax2.plot(epochs, val_auroc,  color=BLUE,   lw=2, marker="s", ms=4,
                   label="Val AUROC")

    best_epoch = d.get("history", history)
    best_idx   = int(np.argmax(val_auroc))
    ax2.scatter([epochs[best_idx]], [val_auroc[best_idx]], color=BLUE, s=120,
                zorder=5, label=f"Best val AUROC {val_auroc[best_idx]:.4f} (epoch {epochs[best_idx]})")
    ax2.axhline(d["test_metrics"]["auroc"], ls="--", color=BLUE, lw=1.2, alpha=0.7,
                label=f"Test AUROC {d['test_metrics']['auroc']:.3f}")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=ORANGE)
    ax2.set_ylabel("Val AUROC",  color=BLUE)
    ax1.tick_params(axis="y", colors=ORANGE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax1.set_title("V2 CNN — Training Dynamics (clean run)", fontweight="bold")
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax2.legend(loc="center right", fontsize=9)
    ax1.legend(handles=[l1], loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, out_path)


# ── Figure 4: V2 per-protein AUROC ───────────────────────────────────────────

def fig_v2_per_protein(v2_path, out_path):
    d = load_json(v2_path)
    pp = d.get("per_protein", [])
    if not pp:
        print("  [skip] no per-protein data")
        return

    df = pd.DataFrame(pp).sort_values("auroc", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.32)))
    colors = []
    for _, row in df.iterrows():
        if row["auroc"] >= 0.80:
            colors.append(GREEN)
        elif row["auroc"] >= 0.65:
            colors.append(BLUE)
        else:
            colors.append(ORANGE)

    ax.barh(df["protein"], df["auroc"], color=colors, alpha=0.85)
    ax.axvline(0.5,  ls="--", color=GRAY,   lw=1.0, alpha=0.7, label="Random (0.50)")
    ax.axvline(0.690, ls="--", color=BLUE,   lw=1.2, alpha=0.8, label="Overall test AUROC (0.690)")
    median = d["per_protein_summary"]["median"]
    ax.axvline(median, ls=":",  color=GREEN,  lw=1.5, alpha=0.8, label=f"Median (0.{round(median*1000):03d})")

    # Legend patches
    g = mpatches.Patch(facecolor=GREEN,  alpha=0.85, label="AUROC ≥ 0.80")
    b = mpatches.Patch(facecolor=BLUE,   alpha=0.85, label="0.65 ≤ AUROC < 0.80")
    o = mpatches.Patch(facecolor=ORANGE, alpha=0.85, label="AUROC < 0.65")

    ax.set_xlim(0.40, 1.02)
    ax.set_xlabel("AUROC")
    ax.set_title("V2 CNN — Per-Protein Test AUROC (24 proteins)", fontweight="bold")
    ax.legend(handles=[g, b, o,
                        plt.Line2D([0],[0], ls="--", color=GRAY,  lw=1.0, label="Random"),
                        plt.Line2D([0],[0], ls="--", color=BLUE,  lw=1.2, label=f"Overall {d['test_metrics']['auroc']:.3f}"),
                        plt.Line2D([0],[0], ls=":",  color=GREEN, lw=1.5, label=f"Median {median:.3f}")],
              fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out_path)


# ── Figure 5: RNAcompete overview ─────────────────────────────────────────────

def fig_rnacompete(metrics_path, per_protein_path, out_path):
    metrics = load_json(metrics_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: per-organism AUROC (top 15 by n_pairs) ──
    ax = axes[0]
    orgs = metrics["per_organism"]
    # Merge duplicate organism names (e.g. Homo_sapiens / Homo sapiens)
    merged = {}
    for org, vals in orgs.items():
        key = org.replace("_", " ")
        if key in merged:
            # combine weighted
            total_n = merged[key]["n"] + vals["n"]
            merged[key]["auroc"] = (merged[key]["auroc"] * merged[key]["n"] +
                                    vals["auroc"] * vals["n"]) / total_n
            merged[key]["n"] = total_n
        else:
            merged[key] = {"auroc": vals["auroc"], "n": vals["n"]}

    org_df = pd.DataFrame([
        {"organism": k, "auroc": v["auroc"], "n": v["n"]}
        for k, v in merged.items()
    ]).sort_values("n", ascending=False).head(15)

    # Sort by AUROC for display
    org_df = org_df.sort_values("auroc", ascending=True)
    bar_colors = [GREEN if a > 0.7 else BLUE if a > 0.55 else ORANGE
                  for a in org_df["auroc"]]

    bars = ax.barh(org_df["organism"], org_df["auroc"], color=bar_colors, alpha=0.85)

    # n labels
    for bar, (_, row) in zip(bars, org_df.iterrows()):
        n_k = row["n"] // 1000
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
                f"{n_k}k", va="center", fontsize=7.5, color=GRAY)

    ax.axvline(0.5,  ls="--", color=GRAY,  lw=1.0, alpha=0.7)
    ax.axvline(0.55, ls=":",  color=ORANGE,lw=1.0, alpha=0.5)
    ax.set_xlim(0.40, 0.95)
    ax.set_xlabel("AUROC")
    ax.set_title("RNAcompete Zero-Shot\nPer-Organism AUROC (top 15 by pairs)", fontweight="bold")

    # ── Right: per-protein AUROC histogram ──
    ax = axes[1]
    if os.path.exists(per_protein_path):
        pp_df = pd.read_csv(per_protein_path, sep="\t")
    else:
        print("  [warn] per_protein.tsv not found")
        pp_df = None

    if pp_df is not None:
        aurocs = pp_df["auroc"].dropna().values
        seen_mask = pp_df.get("in_training", pd.Series([False]*len(pp_df))).fillna(False)
        unseen_aurocs = aurocs[~seen_mask.values]  if len(seen_mask) == len(aurocs) else aurocs
        seen_aurocs   = aurocs[ seen_mask.values]  if len(seen_mask) == len(aurocs) else np.array([])

        bins = np.linspace(0.1, 1.0, 30)
        ax.hist(unseen_aurocs, bins=bins, color=BLUE,   alpha=0.75, label=f"Unseen (n={len(unseen_aurocs)})")
        if len(seen_aurocs) > 0:
            ax.hist(seen_aurocs,   bins=bins, color=ORANGE, alpha=0.75, label=f"Seen in training (n={len(seen_aurocs)})")
        ax.axvline(np.median(unseen_aurocs), ls="--", color=BLUE, lw=1.5,
                   label=f"Median unseen = {np.median(unseen_aurocs):.3f}")
        ax.axvline(0.5, ls=":",  color=GRAY,  lw=1.2, alpha=0.7, label="Random (0.50)")
    else:
        med = metrics["per_protein"]["median_auroc"]
        ax.axvline(med, ls="--", color=BLUE, lw=1.5, label=f"Median {med:.3f}")
        ax.text(0.5, 0.5, f"per_protein.tsv not found\nMedian AUROC: {med:.3f}",
                transform=ax.transAxes, ha="center", va="center")

    ax.set_xlabel("Per-Protein AUROC")
    ax.set_ylabel("Number of Proteins")
    overall = metrics["overall"]["auroc"]
    ax.set_title(f"RNAcompete Zero-Shot\nPer-Protein AUROC Distribution\n(742 proteins, overall AUROC = {overall:.3f})",
                 fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("RNAcompete Zero-Shot Benchmark — V2 CNN (169 training proteins → 742 test proteins, 26 organisms)",
                 y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out_path)


# ── Figure 6: ESM-2 comparison detail ─────────────────────────────────────────

def fig_esm2_comparison(results_dir, out_path):
    """Per-protein AUROC comparison: V2 vs V3c for same 24 proteins."""
    v2  = load_json(os.path.join(results_dir, "v2_cnn_results.json"))
    v3c = load_json(os.path.join(results_dir, "v3c_esm2_residue_results.json"))

    v2_pp  = {p["protein"]: p["auroc"] for p in v2.get("per_protein", [])}
    v3c_pp = {p["protein"]: p["auroc"] for p in v3c.get("per_protein", [])}
    common = sorted(set(v2_pp) & set(v3c_pp))
    if not common:
        print("  [skip] no common proteins between V2 and V3c")
        return

    v2_vals  = [v2_pp[p]  for p in common]
    v3c_vals = [v3c_pp[p] for p in common]
    deltas   = [v3c_pp[p] - v2_pp[p] for p in common]

    # Sort by delta
    order   = np.argsort(deltas)
    common  = [common[i]  for i in order]
    v2_vals = [v2_vals[i] for i in order]
    v3c_vals= [v3c_vals[i]for i in order]
    deltas  = [deltas[i]  for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(common) * 0.30)))

    # Left: scatter V2 vs V3c
    ax = axes[0]
    colors = [GREEN if d > 0 else ORANGE for d in deltas]
    ax.scatter(v2_vals, v3c_vals, c=colors, s=60, alpha=0.85, zorder=3)
    lim = (0.30, 1.05)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="V2 = V3c")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("V2 CNN per-protein AUROC")
    ax.set_ylabel("V3c ESM-2 residue per-protein AUROC")
    ax.set_title("Per-Protein AUROC: V2 vs V3c", fontweight="bold")
    for i, prot in enumerate(common):
        if abs(deltas[i]) > 0.07:
            ax.annotate(prot, (v2_vals[i], v3c_vals[i]),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
    ax.legend(fontsize=9)

    # Right: delta bar chart
    ax = axes[1]
    bar_colors = [GREEN if d > 0 else ORANGE for d in deltas]
    ax.barh(common, deltas, color=bar_colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("AUROC Δ (V3c − V2)")
    ax.set_title("Per-Protein AUROC Delta: V3c vs V2\n(green = V3c better, orange = V2 better)", fontweight="bold")
    ax.set_xlim(-0.45, 0.45)

    fig.suptitle("V3c ESM-2 Residue CNN vs V2 One-Hot CNN — Per-Protein Analysis",
                 y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results",
                        help="Root results directory")
    parser.add_argument("--out_dir",     default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    R = args.results_dir
    O = args.out_dir

    print(f"\n{'='*55}")
    print(f"  Generating README figures → {O}/")
    print(f"{'='*55}")

    # Figure 1 — Phase 1
    p1 = os.path.join(R, "phase1_summary.json")
    if os.path.exists(p1):
        fig_phase1(p1, os.path.join(O, "phase1_validation.png"))
    else:
        print(f"  [skip] {p1} not found")

    # Figure 2 — Phase 2 model comparison
    gen = os.path.join(R, "generalized")
    if os.path.isdir(gen):
        fig_phase2_comparison(gen, os.path.join(O, "phase2_model_comparison.png"))
    else:
        print(f"  [skip] {gen} not found")

    # Figure 3 — V2 training curve
    v2_path = os.path.join(R, "generalized", "v2_cnn_results.json")
    if os.path.exists(v2_path):
        fig_v2_training(v2_path, os.path.join(O, "v2_training_curve.png"))
    else:
        print(f"  [skip] {v2_path} not found")

    # Figure 4 — V2 per-protein
    if os.path.exists(v2_path):
        fig_v2_per_protein(v2_path, os.path.join(O, "v2_per_protein_auroc.png"))

    # Figure 5 — RNAcompete overview
    rc_metrics  = os.path.join(R, "benchmarks", "rnacompete_v2", "metrics.json")
    rc_pp       = os.path.join(R, "benchmarks", "rnacompete_v2", "per_protein.tsv")
    if os.path.exists(rc_metrics):
        fig_rnacompete(rc_metrics, rc_pp, os.path.join(O, "rnacompete_overview.png"))
    else:
        print(f"  [skip] {rc_metrics} not found")

    # Figure 6 — ESM-2 comparison detail
    v3c_path = os.path.join(R, "generalized", "v3c_esm2_residue_results.json")
    if os.path.exists(v2_path) and os.path.exists(v3c_path):
        fig_esm2_comparison(gen, os.path.join(O, "esm2_vs_v2_comparison.png"))

    print(f"\n  All figures saved to {O}/")
    print(f"  Add to README.md with: ![Caption]({O}/figure_name.png)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
