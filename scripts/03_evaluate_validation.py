"""
Script 03: Full test-set evaluation + per-protein AUROC + plots.

Usage (from protein_rna_ml/ folder):
    python scripts/03_evaluate_validation.py --config configs/rbns_validation.yaml --model xgboost
"""

import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml, joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                              f1_score, brier_score_loss, roc_curve, precision_recall_curve)


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {"auroc": float(roc_auc_score(y_true, y_prob)),
            "auprc": float(average_precision_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier": float(brier_score_loss(y_true, y_prob))}


def per_protein_auroc(df, y_prob, protein_col, label_col):
    df = df.copy()
    df["_prob"] = y_prob
    rows = []
    for prot, g in df.groupby(protein_col):
        if g[label_col].nunique() < 2:
            rows.append({"protein": prot, "auroc": None, "n": len(g), "note": "single_class"})
        else:
            rows.append({"protein": prot,
                         "auroc": float(roc_auc_score(g[label_col], g["_prob"])),
                         "auprc": float(average_precision_score(g[label_col], g["_prob"])),
                         "n": len(g), "note": "ok"})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False, na_position="last")


def plot_roc_pr(y_true, y_prob, plots_dir, model_name, auroc, auprc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="#2563EB", label=f"AUROC={auroc:.3f}")
    axes[0].plot([0,1],[0,1],"k--",lw=1)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC — {model_name}"); axes[0].legend(); axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, lw=2, color="#16A34A", label=f"AUPRC={auprc:.3f}")
    axes[1].axhline(y_true.mean(), color="gray", ls="--", lw=1, label=f"Baseline={y_true.mean():.2f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR — {model_name}"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_per_protein(pp_df, plots_dir, pass_thr, warn_thr):
    valid = pp_df[pp_df["auroc"].notna()]
    proteins = valid["protein"].tolist()
    aurocs   = valid["auroc"].tolist()

    colors = ["#16A34A" if a >= pass_thr else "#F59E0B" if a >= warn_thr else "#DC2626"
              for a in aurocs]

    fig, ax = plt.subplots(figsize=(max(10, len(proteins)*0.28), 5))
    ax.bar(range(len(proteins)), aurocs, color=colors)
    ax.axhline(pass_thr, color="#16A34A", ls="--", lw=1.5, label=f"Pass ({pass_thr})")
    ax.axhline(warn_thr, color="#F59E0B", ls="--", lw=1.5, label=f"Warn ({warn_thr})")
    ax.axhline(0.5,      color="gray",    ls=":",  lw=1.0, label="Random (0.5)")
    ax.set_xticks(range(len(proteins)))
    ax.set_xticklabels(proteins, rotation=90, fontsize=7)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-Protein AUROC — Test Set")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, "per_protein_auroc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="xgboost",
                        choices=["logistic_regression","random_forest","xgboost"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cmap        = cfg["dataset"]["column_map"]
    protein_col = cmap["protein_col"]
    label_col   = cmap["label"]

    processed_dir = cfg["dataset"]["processed_dir"]
    splits_dir    = cfg["dataset"]["splits_dir"]
    model_dir     = cfg["output"]["model_save_dir"]
    plots_dir     = cfg["output"]["plots_dir"]
    metrics_path  = cfg["output"]["metrics_path"]
    os.makedirs(plots_dir, exist_ok=True)

    model = joblib.load(os.path.join(model_dir, f"{args.model}.pkl"))
    test  = np.load(os.path.join(processed_dir, "test_kmer.npz"))
    X_test, y_test = test["X"], test["y"]
    test_df = pd.read_csv(os.path.join(splits_dir, "test.tsv"), sep="\t")

    y_prob = model.predict_proba(X_test)[:,1]
    overall = compute_metrics(y_test, y_prob)

    print(f"\n=== TEST SET — {cfg['dataset']['name']} — {args.model} ===")
    for k, v in overall.items():
        print(f"  {k:<10}: {v:.4f}")

    pp_df = per_protein_auroc(test_df, y_prob, protein_col, label_col)
    valid_aurocs = pp_df["auroc"].dropna().values
    flagged = pp_df[pp_df["auroc"].notna() & (pp_df["auroc"] < cfg["thresholds"]["per_protein_auroc_min"])]

    print(f"\n=== PER-PROTEIN ({len(pp_df)} proteins) ===")
    print(f"  Median={np.median(valid_aurocs):.4f}  Mean={np.mean(valid_aurocs):.4f}  "
          f"Min={np.min(valid_aurocs):.4f}  Max={np.max(valid_aurocs):.4f}")
    if len(flagged):
        print(f"\n  ⚠️  Flagged proteins (AUROC < {cfg['thresholds']['per_protein_auroc_min']}):")
        for _, row in flagged.iterrows():
            print(f"    {row['protein']}: {row['auroc']:.4f}  (n={row['n']})")
    else:
        print(f"  ✅ All proteins above threshold")

    pt, wt = cfg["thresholds"]["auroc_pass"], cfg["thresholds"]["auroc_warn"]
    if overall["auroc"] >= pt and np.median(valid_aurocs) >= 0.65:
        status, msg = "PASS", "✅ Validated — ready for Phase 2"
    elif overall["auroc"] >= wt:
        status, msg = "WARN", "⚠️  Marginal — investigate before Phase 2"
    else:
        status, msg = "FAIL", "❌ Failed — do NOT include in Phase 2"

    print(f"\n  Decision: {status}  —  {msg}")

    plot_roc_pr(y_test, y_prob, plots_dir, args.model, overall["auroc"], overall["auprc"])
    plot_per_protein(pp_df, plots_dir, pt, cfg["thresholds"]["per_protein_auroc_min"])

    out = {"dataset": cfg["dataset"]["name"], "model": args.model,
           "overall": overall,
           "per_protein_summary": {"median": float(np.median(valid_aurocs)),
                                   "mean":   float(np.mean(valid_aurocs)),
                                   "min":    float(np.min(valid_aurocs)),
                                   "max":    float(np.max(valid_aurocs)),
                                   "n_flagged": len(flagged)},
           "per_protein": pp_df.to_dict(orient="records"),
           "flagged": flagged[["protein","auroc","n"]].to_dict(orient="records"),
           "status": status, "message": msg}

    result_path = metrics_path.replace("validation_results.json","test_evaluation.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results → {result_path}")

if __name__ == "__main__":
    main()
