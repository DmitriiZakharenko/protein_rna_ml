#!/usr/bin/env python3
"""
16_analyze_predictions.py
Prediction diagnostics and error analysis.

Panels produced:
  C. Embedding diagnostics
       RNA length vs prediction score
       Protein length vs prediction score
       GC-content vs prediction score
       Calibration curve (predicted prob vs actual positive rate)
  D. Error analysis
       False positive / false negative score distribution
       Top k-mer enrichment in FP and FN clusters
       Confusion-based per-protein error breakdown

Usage:
  python scripts/16_analyze_predictions.py \
      --predictions_tsv results/analysis/predictions/v2_cnn_test_preds.tsv \
      --model_name v2_cnn \
      --output_dir results/analysis/predictions \
      [--threshold 0.5] [--top_kmers 10] [--kmer_k 6]
"""

import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

DATASET_COLORS = {
    "htr_selex_25907": "#4e79a7",
    "htr_selex_47428": "#76b7b2",
    "rbns":            "#f28e2b",
    "eclip":           "#e15759",
    "unknown":         "#bab0ac",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def gc_content(seq: str) -> float:
    seq = seq.upper().replace("U", "T")
    if not seq:
        return float("nan")
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


def kmer_freq(seq: str, k: int = 6) -> Counter:
    seq = seq.upper().replace("T", "U")
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)
             if all(c in "ACGU" for c in seq[i:i+k])]
    return Counter(kmers)


def enrich_kmers(pos_seqs: list[str], neg_seqs: list[str], k: int, top_n: int
                 ) -> pd.DataFrame:
    """Compute per-k-mer log-enrichment (log2 ratio of freq in pos vs neg)."""
    pos_cnt: Counter = Counter()
    neg_cnt: Counter = Counter()
    for s in pos_seqs:
        pos_cnt.update(kmer_freq(s, k))
    for s in neg_seqs:
        neg_cnt.update(kmer_freq(s, k))
    total_pos = max(sum(pos_cnt.values()), 1)
    total_neg = max(sum(neg_cnt.values()), 1)
    kmers = set(pos_cnt) | set(neg_cnt)
    rows = []
    for km in kmers:
        p = (pos_cnt[km] + 0.5) / (total_pos + 0.5)
        n = (neg_cnt[km] + 0.5) / (total_neg + 0.5)
        rows.append({"kmer": km, "log2_enrich": np.log2(p / n),
                     "pos_count": pos_cnt[km], "neg_count": neg_cnt[km]})
    df = pd.DataFrame(rows).sort_values("log2_enrich", ascending=False)
    return df.head(top_n)


def save_fig(fig: plt.Figure, out_dir: str, name: str, fmt: str = "png"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ── Panel C: Embedding diagnostics ───────────────────────────────────────────

def plot_embedding_diagnostics(df: pd.DataFrame, out_dir: str, fmt: str,
                                model_name: str):
    """
    Scatter plots of sequence-level features vs predicted score.
    df must have columns: rna_sequence, protein_sequence, prob, binding_label
    """
    df = df.copy()
    df["rna_len"]  = df["rna_sequence"].str.len()
    df["prot_len"] = df["protein_sequence"].str.len() if "protein_sequence" in df.columns else np.nan
    df["gc"]       = df["rna_sequence"].apply(gc_content)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    feature_specs = [
        ("rna_len",  "RNA length (nt)",         axes[0]),
        ("prot_len", "Protein length (aa)",      axes[1]),
        ("gc",       "RNA GC content",           axes[2]),
    ]

    for feat, xlabel, ax in feature_specs:
        if df[feat].isna().all():
            ax.set_visible(False)
            continue
        colors = df["binding_label"].map({1: "#e15759", 0: "#4e79a7"})
        ax.scatter(df[feat], df["prob"], c=colors, s=6, alpha=0.15, rasterized=True)
        # Trend line (binned median)
        try:
            bins = pd.cut(df[feat].dropna(), bins=30)
            trend = df.dropna(subset=[feat]).groupby(bins, observed=True)["prob"].median()
            bin_mids = [interval.mid for interval in trend.index]
            ax.plot(bin_mids, trend.values, color="black", lw=2, label="binned median")
        except Exception:
            pass
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Predicted probability")
        ax.set_title(f"C.  {xlabel} vs score")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e15759",
               markersize=8, label="positive (label=1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4e79a7",
               markersize=8, label="negative (label=0)"),
    ]
    axes[0].legend(handles=legend_elems, fontsize=8)

    fig.suptitle(f"Embedding diagnostics — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, f"C_embedding_diagnostics_{model_name}", fmt)

    # Per-dataset breakdown if available
    if "dataset" in df.columns or "dataset_source" in df.columns:
        col = "dataset" if "dataset" in df.columns else "dataset_source"
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        for feat, xlabel, ax in zip(
            ["rna_len", "prot_len", "gc"],
            ["RNA length (nt)", "Protein length (aa)", "RNA GC content"],
            axes2
        ):
            if df[feat].isna().all():
                ax.set_visible(False)
                continue
            for ds in sorted(df[col].unique()):
                sub = df[df[col] == ds]
                c = DATASET_COLORS.get(ds, "#999999")
                try:
                    bins = pd.cut(sub[feat].dropna(), bins=20)
                    trend = sub.dropna(subset=[feat]).groupby(bins, observed=True)["prob"].median()
                    bin_mids = [interval.mid for interval in trend.index]
                    ax.plot(bin_mids, trend.values, color=c, lw=1.6, label=ds, alpha=0.85)
                except Exception:
                    pass
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Median predicted score")
            ax.set_title(f"C.  {xlabel} by dataset")
            ax.legend(fontsize=7)
        fig2.tight_layout()
        save_fig(fig2, out_dir, f"C_diagnostics_by_dataset_{model_name}", fmt)


def plot_calibration(df: pd.DataFrame, out_dir: str, fmt: str, model_name: str):
    """Reliability diagram — expected calibration."""
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        fraction_of_positives, mean_predicted = calibration_curve(
            df["binding_label"], df["prob"], n_bins=15, strategy="quantile"
        )
        ax.plot(mean_predicted, fraction_of_positives, "s-", color="#e15759", lw=1.8,
                label=model_name)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"C.  Calibration — {model_name}")
        ax.legend(fontsize=9)
        # ECE
        n = len(df)
        ece_val = np.sum(
            np.abs(fraction_of_positives - mean_predicted)
        ) / len(fraction_of_positives)
        ax.text(0.05, 0.92, f"ECE ≈ {ece_val:.3f}", transform=ax.transAxes, fontsize=9)
    except Exception as e:
        ax.text(0.5, 0.5, f"Calibration error:\n{e}", transform=ax.transAxes,
                ha="center", va="center")
    fig.tight_layout()
    save_fig(fig, out_dir, f"C_calibration_{model_name}", fmt)


# ── Panel D: Error analysis ────────────────────────────────────────────────────

def plot_score_distributions(df: pd.DataFrame, out_dir: str, fmt: str,
                              model_name: str, threshold: float):
    """Score distributions for TP / TN / FP / FN."""
    df = df.copy()
    df["pred"] = (df["prob"] >= threshold).astype(int)
    df["outcome"] = "TN"
    df.loc[(df["pred"] == 1) & (df["binding_label"] == 1), "outcome"] = "TP"
    df.loc[(df["pred"] == 1) & (df["binding_label"] == 0), "outcome"] = "FP"
    df.loc[(df["pred"] == 0) & (df["binding_label"] == 1), "outcome"] = "FN"

    outcome_colors = {"TP": "#59a14f", "TN": "#4e79a7", "FP": "#e15759", "FN": "#f28e2b"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    counts = df["outcome"].value_counts()
    for i, oc in enumerate(["TP", "TN", "FP", "FN"]):
        ax = axes[i]
        subset = df[df["outcome"] == oc]["prob"]
        if len(subset) == 0:
            ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"D.  {oc} (n=0)")
            continue
        ax.hist(subset, bins=40, color=outcome_colors[oc], alpha=0.75, edgecolor="none")
        ax.axvline(threshold, color="black", lw=1, ls="--")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.set_title(f"D.  {oc} (n={counts.get(oc, 0):,})")

    fig.suptitle(f"Error analysis — score distributions ({model_name}, threshold={threshold:.2f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, f"D_score_distributions_{model_name}", fmt)

    # Confusion matrix
    cm = confusion_matrix(df["binding_label"], df["pred"])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["negative", "positive"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"Confusion matrix\n{model_name} (thr={threshold:.2f})")
    fig_cm.tight_layout()
    save_fig(fig_cm, out_dir, f"D_confusion_matrix_{model_name}", fmt)

    return df


def plot_kmer_enrichment(df: pd.DataFrame, out_dir: str, fmt: str,
                          model_name: str, threshold: float,
                          kmer_k: int = 6, top_n: int = 15):
    """k-mer enrichment in FP vs TN, and FN vs TP."""
    if "rna_sequence" not in df.columns:
        print("  [SKIP] rna_sequence column not found — skipping k-mer enrichment")
        return

    df = df.copy()
    df["pred"] = (df["prob"] >= threshold).astype(int)

    for title, pos_label, neg_label in [
        ("FP vs TN: over-predicted non-binders",
         ((df["pred"] == 1) & (df["binding_label"] == 0)),
         ((df["pred"] == 0) & (df["binding_label"] == 0))),
        ("FN vs TP: under-predicted binders",
         ((df["pred"] == 0) & (df["binding_label"] == 1)),
         ((df["pred"] == 1) & (df["binding_label"] == 1))),
    ]:
        pos_seqs = df.loc[pos_label, "rna_sequence"].tolist()
        neg_seqs = df.loc[neg_label, "rna_sequence"].tolist()
        if len(pos_seqs) < 5 or len(neg_seqs) < 5:
            continue

        # Cap at 3000 each to keep fast
        rng = np.random.default_rng(42)
        if len(pos_seqs) > 3000:
            pos_seqs = list(rng.choice(pos_seqs, 3000, replace=False))
        if len(neg_seqs) > 3000:
            neg_seqs = list(rng.choice(neg_seqs, 3000, replace=False))

        enrich_df = enrich_kmers(pos_seqs, neg_seqs, kmer_k, top_n)

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
        colors = ["#e15759" if v > 0 else "#4e79a7" for v in enrich_df["log2_enrich"]]
        ax.barh(enrich_df["kmer"], enrich_df["log2_enrich"], color=colors, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(f"log2 enrichment vs background")
        ax.set_ylabel(f"{kmer_k}-mer")
        ax.set_title(f"D.  {title}\n({model_name}, top {top_n} {kmer_k}-mers)")
        fig.tight_layout()
        label_slug = title.split(":")[0].replace(" ", "_")
        save_fig(fig, out_dir, f"D_kmer_{label_slug}_{model_name}", fmt)

        # Save table
        tsv_path = os.path.join(out_dir, f"D_kmer_{label_slug}_{model_name}.tsv")
        enrich_df.to_csv(tsv_path, sep="\t", index=False)
        print(f"  → saved {tsv_path}")


def plot_per_protein_errors(df: pd.DataFrame, out_dir: str, fmt: str,
                             model_name: str, threshold: float):
    """Per-protein FPR vs FNR scatter."""
    if "protein_name" not in df.columns and "target_name" not in df.columns:
        return
    prot_col = "protein_name" if "protein_name" in df.columns else "target_name"
    df = df.copy()
    df["pred"] = (df["prob"] >= threshold).astype(int)

    rows = []
    for prot, grp in df.groupby(prot_col):
        tp = ((grp["pred"] == 1) & (grp["binding_label"] == 1)).sum()
        tn = ((grp["pred"] == 0) & (grp["binding_label"] == 0)).sum()
        fp = ((grp["pred"] == 1) & (grp["binding_label"] == 0)).sum()
        fn = ((grp["pred"] == 0) & (grp["binding_label"] == 1)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        rows.append({"protein": prot, "fpr": fpr, "fnr": fnr,
                     "n": len(grp), "tp": tp, "tn": tn, "fp": fp, "fn": fn})

    err_df = pd.DataFrame(rows).dropna()
    if len(err_df) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(err_df["fpr"], err_df["fnr"], s=err_df["n"] / err_df["n"].max() * 120 + 10,
                    alpha=0.55, c=err_df["n"], cmap="viridis", edgecolors="none")
    ax.axline((0, 0), slope=1, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title(f"D.  Per-protein FPR vs FNR — {model_name}")
    plt.colorbar(sc, ax=ax, label="n samples")

    # Label extreme outliers
    extreme = err_df[(err_df["fpr"] > 0.7) | (err_df["fnr"] > 0.7)]
    for _, row in extreme.iterrows():
        ax.annotate(row["protein"], (row["fpr"], row["fnr"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    fig.tight_layout()
    save_fig(fig, out_dir, f"D_per_protein_fpr_fnr_{model_name}", fmt)

    err_df.to_csv(os.path.join(out_dir, f"D_per_protein_errors_{model_name}.tsv"),
                  sep="\t", index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions_tsv", required=True,
                        help=("TSV with columns: rna_sequence, protein_sequence (opt.), "
                              "binding_label, prob, [protein_name|target_name], [dataset]"))
    parser.add_argument("--model_name",    default="model",
                        help="Human-readable model label for plot titles")
    parser.add_argument("--output_dir",    default="results/analysis/predictions",
                        help="Output directory for all plots and tables")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Decision threshold for FP/FN analysis (default: 0.5)")
    parser.add_argument("--top_kmers",     type=int,   default=15,
                        help="Number of top k-mers to show in enrichment plot")
    parser.add_argument("--kmer_k",        type=int,   default=6,
                        help="k-mer length for motif analysis (default: 6)")
    parser.add_argument("--format",        default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Prediction Diagnostics + Error Analysis")
    print(f"{'='*60}")
    print(f"  predictions_tsv : {args.predictions_tsv}")
    print(f"  model_name      : {args.model_name}")
    print(f"  output_dir      : {args.output_dir}")
    print(f"  threshold       : {args.threshold}\n")

    if not os.path.exists(args.predictions_tsv):
        sys.exit(f"File not found: {args.predictions_tsv}")

    df = pd.read_csv(args.predictions_tsv, sep="\t", low_memory=False)
    print(f"  Loaded {len(df):,} rows, columns: {df.columns.tolist()}")

    required = {"binding_label", "prob"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    # Normalise dataset column
    if "dataset" not in df.columns and "dataset_source" in df.columns:
        df = df.rename(columns={"dataset_source": "dataset"})

    auroc = roc_auc_score(df["binding_label"], df["prob"])
    auprc = average_precision_score(df["binding_label"], df["prob"])
    pos_rate = df["binding_label"].mean()
    print(f"\n  AUROC  : {auroc:.4f}")
    print(f"  AUPRC  : {auprc:.4f}")
    print(f"  pos%   : {pos_rate:.2%}")
    print(f"  n_pos  : {df['binding_label'].sum():,}")
    print(f"  n_neg  : {(1-df['binding_label']).sum():,}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Panel C: Embedding diagnostics ---")
    plot_embedding_diagnostics(df, args.output_dir, args.format, args.model_name)
    plot_calibration(df, args.output_dir, args.format, args.model_name)

    print("\n--- Panel D: Error analysis ---")
    df_with_outcomes = plot_score_distributions(
        df, args.output_dir, args.format, args.model_name, args.threshold)
    plot_kmer_enrichment(
        df_with_outcomes, args.output_dir, args.format, args.model_name,
        args.threshold, args.kmer_k, args.top_kmers)
    plot_per_protein_errors(
        df_with_outcomes, args.output_dir, args.format, args.model_name, args.threshold)

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
