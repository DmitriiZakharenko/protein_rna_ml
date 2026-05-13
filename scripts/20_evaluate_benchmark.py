#!/usr/bin/env python3
"""
20_evaluate_benchmark.py
Standalone inference + evaluation script for any saved PyTorch model checkpoint.

Loads a trained model, runs inference on a benchmark TSV (e.g. RNAcompete,
eCLIP, or any file in project schema), saves predictions, and reports metrics.

This is the script you run AFTER:
  1. python scripts/17_prepare_rnacompete_benchmark.py  (prep data once)
  2. python scripts/20_evaluate_benchmark.py            (score with model)

Supported model types (auto-detected from checkpoint):
  v2_cnn       — dual-branch CNN (RNABindingCNN)
  v1_mlp       — k-mer MLP (RNABindingMLP)

Usage examples:

  # RNAcompete full benchmark with V2 CNN:
  python scripts/20_evaluate_benchmark.py \
      --checkpoint models/saved/generalized_v2/best_model.pt \
      --benchmark  data/benchmarks/rnacompete/rnacompete_all.tsv \
      --output_dir results/benchmarks/rnacompete_v2 \
      --model_type v2_cnn

  # Human-only subset:
  python scripts/20_evaluate_benchmark.py \
      --checkpoint models/saved/generalized_v2/best_model.pt \
      --benchmark  data/benchmarks/rnacompete/rnacompete_human.tsv \
      --output_dir results/benchmarks/rnacompete_v2_human \
      --model_type v2_cnn

  # After multi-seed training, score the clean V2:
  python scripts/20_evaluate_benchmark.py \
      --checkpoint results/multiseed/v2_cnn_clean/seed_42/best_model.pt \
      --benchmark  data/benchmarks/rnacompete/rnacompete_all.tsv \
      --output_dir results/benchmarks/rnacompete_v2_clean_seed42

Output files:
  predictions.tsv        — original rows + "prob" column
  metrics.json           — AUROC, AUPRC, per-protein summary
  per_protein.tsv        — per-protein AUROC/AUPRC/n
  roc_pr_curves.png      — ROC + PR curves
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Matplotlib (optional) ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Model registry ────────────────────────────────────────────────────────────
# Maps model_type string → (import path, class, default architecture kwargs)
# Architecture kwargs are the hardcoded values from the training scripts.
# If your training script used different values, override via --arch_json.

V2_CNN_DEFAULT_ARCH = {
    "rna_filters":  [128, 256, 256],
    "prot_filters": [128, 256, 256],
    "rna_kernels":  [7, 5, 3],
    "prot_kernels": [11, 7, 5],
    "head_dims":    [256, 64],
}


def build_model(model_type: str, checkpoint_args: dict, arch_override: dict | None):
    """
    Reconstruct model from checkpoint training args + (optional) arch override.
    Returns an uninitialised model (state_dict loaded by caller).
    """
    dropout = checkpoint_args.get("dropout", 0.3)

    if model_type == "v2_cnn":
        from src.models.cnn_model import RNABindingCNN
        arch = V2_CNN_DEFAULT_ARCH.copy()
        if arch_override:
            arch.update(arch_override)
        return RNABindingCNN(**arch, dropout=dropout)

    if model_type == "v1_mlp":
        from src.models.mlp_model import RNABindingMLP
        # MLP takes input_dim; try to read from checkpoint args or fallback
        input_dim = checkpoint_args.get("input_dim", 4096)
        return RNABindingMLP(input_dim=input_dim)

    raise ValueError(f"Unknown model_type: '{model_type}'. "
                     f"Supported: v2_cnn, v1_mlp")


def auto_detect_model_type(checkpoint: dict) -> str:
    """Best-effort model type detection from checkpoint metadata."""
    args = checkpoint.get("args", {})
    model_name = str(checkpoint.get("model_name", "")).lower()
    script = str(args.get("script", "")).lower()

    if "cnn" in model_name or "v2" in model_name:
        return "v2_cnn"
    if "mlp" in model_name or "v1" in model_name:
        return "v1_mlp"
    # Detect by state_dict key pattern
    state_keys = list(checkpoint.get("model_state", {}).keys())
    if any("rna_branch" in k for k in state_keys):
        return "v2_cnn"
    if any("layers" in k for k in state_keys):
        return "v1_mlp"
    return "v2_cnn"  # safe default


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(no_cuda: bool = False):
    import torch
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference_cnn(model, df: pd.DataFrame, rna_max: int, prot_max: int,
                      batch_size: int, device) -> np.ndarray:
    """Run V2 CNN inference on a DataFrame, return probability array."""
    import torch
    from src.data.dataset import SeqDataset

    # SeqDataset reads from file — write a temp TSV, or monkey-patch.
    # Cleaner: use SeqDataset._one_hot_* directly via a simple inline loop.
    from src.data.dataset import RNA_TO_IDX, AA_TO_IDX

    def one_hot_rna(seq: str) -> np.ndarray:
        arr = np.zeros((rna_max, 4), dtype=np.float32)
        for i, c in enumerate(seq.upper()[:rna_max]):
            if c in RNA_TO_IDX:
                arr[i, RNA_TO_IDX[c]] = 1.0
        return arr

    def one_hot_prot(seq: str) -> np.ndarray:
        arr = np.zeros((prot_max, 20), dtype=np.float32)
        for i, c in enumerate(str(seq).upper()[:prot_max]):
            if c in AA_TO_IDX:
                arr[i, AA_TO_IDX[c]] = 1.0
        return arr

    model.eval()
    all_probs = []
    n = len(df)

    for start in range(0, n, batch_size):
        batch = df.iloc[start:start + batch_size]
        rna_batch  = np.stack([one_hot_rna(s)  for s in batch["rna_sequence"]])
        prot_batch = np.stack([one_hot_prot(s) for s in batch["protein_sequence"]])

        with torch.no_grad():
            rna_t  = torch.tensor(rna_batch,  dtype=torch.float32).to(device)
            prot_t = torch.tensor(prot_batch, dtype=torch.float32).to(device)
            logits = model(rna_t, prot_t)
            probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

        if (start // batch_size) % 20 == 0:
            pct = (start + len(batch)) / n * 100
            print(f"  inference {start + len(batch):>8,}/{n:,}  ({pct:.1f}%)", end="\r")

    print()
    return np.concatenate(all_probs)


def run_inference_mlp(model, df: pd.DataFrame, batch_size: int, device) -> np.ndarray:
    """Run V1 MLP inference using k-mer encoding."""
    import torch
    from src.data.preprocessing import encode_kmer  # project utility

    all_probs = []
    model.eval()
    n = len(df)

    for start in range(0, n, batch_size):
        batch = df.iloc[start:start + batch_size]
        feats = np.stack([encode_kmer(r, p)
                          for r, p in zip(batch["rna_sequence"], batch["protein_sequence"])])
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "pos_rate": float(y_true.mean()),
        "random_auprc_baseline": float(y_true.mean()),
    }


def compute_per_protein(df: pd.DataFrame, y_prob: np.ndarray,
                        min_pos: int = 5, min_neg: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["prob"] = y_prob
    prot_col = "protein_name" if "protein_name" in df.columns else "target_name"
    if prot_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for prot, grp in df.groupby(prot_col):
        n_pos = (grp["binding_label"] == 1).sum()
        n_neg = (grp["binding_label"] == 0).sum()
        if n_pos < min_pos or n_neg < min_neg:
            rows.append({"protein": prot, "auroc": None, "auprc": None,
                         "n": len(grp), "n_pos": n_pos, "n_neg": n_neg,
                         "status": "skipped_insufficient_data"})
            continue
        try:
            auroc = float(roc_auc_score(grp["binding_label"], grp["prob"]))
            auprc = float(average_precision_score(grp["binding_label"], grp["prob"]))
            rows.append({"protein": prot, "auroc": auroc, "auprc": auprc,
                         "n": len(grp), "n_pos": n_pos, "n_neg": n_neg,
                         "status": "ok"})
        except Exception as e:
            rows.append({"protein": prot, "auroc": None, "auprc": None,
                         "n": len(grp), "n_pos": n_pos, "n_neg": n_neg,
                         "status": f"error: {e}"})

    return pd.DataFrame(rows)


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray,
                out_dir: str, label: str):
    if not HAS_MATPLOTLIB:
        return
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, lw=2, color="#2563EB", label=f"AUROC = {auroc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title(f"ROC — {label}")
    ax1.legend(); ax1.grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()
    ax2.plot(rec, prec, lw=2, color="#16A34A", label=f"AUPRC = {auprc:.4f}")
    ax2.axhline(baseline, color="gray", ls="--", lw=1,
                label=f"Random baseline = {baseline:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall — {label}")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_per_protein_dist(pp_df: pd.DataFrame, out_dir: str, label: str):
    if not HAS_MATPLOTLIB:
        return
    valid = pp_df[pp_df["auroc"].notna()]["auroc"].values
    if len(valid) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(valid, bins=40, color="#4e79a7", alpha=0.8, edgecolor="none")
    axes[0].axvline(np.median(valid), color="#e15759", lw=2,
                    label=f"Median = {np.median(valid):.4f}")
    axes[0].axvline(0.5, color="gray", ls="--", lw=1, label="Random (0.5)")
    axes[0].axvline(0.798, color="black", ls=":", lw=1.5,
                    label="ZHMolGraph (0.798)")
    axes[0].set_xlabel("Per-protein AUROC")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Per-protein AUROC distribution\n{label}")
    axes[0].legend(fontsize=8)

    sorted_auroc = np.sort(valid)
    cdf = np.arange(1, len(sorted_auroc) + 1) / len(sorted_auroc)
    axes[1].plot(sorted_auroc, cdf, lw=2, color="#4e79a7")
    axes[1].axvline(0.5, color="gray", ls="--", lw=1)
    axes[1].axvline(0.798, color="black", ls=":", lw=1.5)
    axes[1].set_xlabel("Per-protein AUROC")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("CDF")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "per_protein_auroc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--benchmark",   required=True,
                        help="Path to benchmark TSV (project schema)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory for predictions, metrics, plots")
    parser.add_argument("--model_type",  default="auto",
                        choices=["auto", "v2_cnn", "v1_mlp"],
                        help="Model type (default: auto-detect from checkpoint)")
    parser.add_argument("--batch_size",  type=int, default=512,
                        help="Inference batch size (default: 512)")
    parser.add_argument("--rna_max",     type=int, default=None,
                        help="RNA max length — overrides checkpoint value")
    parser.add_argument("--prot_max",    type=int, default=None,
                        help="Protein max length — overrides checkpoint value")
    parser.add_argument("--min_pos",     type=int, default=10,
                        help="Min positives per protein for AUROC (default: 10)")
    parser.add_argument("--min_neg",     type=int, default=10,
                        help="Min negatives per protein for AUROC (default: 10)")
    parser.add_argument("--arch_json",   default=None,
                        help="JSON string with architecture overrides e.g. "
                             "'{\"rna_filters\":[64,128]}'")
    parser.add_argument("--no_cuda",     action="store_true")
    parser.add_argument("--max_rows",    type=int, default=None,
                        help="Subsample benchmark to this many rows (for quick testing)")
    args = parser.parse_args()

    import torch

    print(f"\n{'='*65}")
    print(f"  Benchmark Evaluation")
    print(f"{'='*65}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  benchmark   : {args.benchmark}")
    print(f"  output_dir  : {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    print("  Loading checkpoint...", end=" ", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print("done")

    ckpt_args = ckpt.get("args", {})
    print(f"  Checkpoint epoch  : {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint val    : {ckpt.get('val_metrics', {})}")

    # Resolve model type
    model_type = args.model_type
    if model_type == "auto":
        model_type = auto_detect_model_type(ckpt)
        print(f"  Auto-detected     : {model_type}")

    # Resolve sequence lengths
    rna_max  = args.rna_max  or ckpt_args.get("rna_max",  60)
    prot_max = args.prot_max or ckpt_args.get("prot_max", 300)
    print(f"  rna_max / prot_max: {rna_max} / {prot_max}")

    # Architecture overrides
    arch_override = json.loads(args.arch_json) if args.arch_json else None

    # ── Build + load model ─────────────────────────────────────────────────
    device = get_device(args.no_cuda)
    print(f"  Device            : {device}")

    model = build_model(model_type, ckpt_args, arch_override)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters        : {n_params:,}\n")

    # ── Load benchmark data ────────────────────────────────────────────────
    print(f"  Loading benchmark: {args.benchmark}")
    df = pd.read_csv(args.benchmark, sep="\t", low_memory=False)

    # Normalise columns
    if "target_name" in df.columns and "protein_name" not in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})
    if "dataset_source" not in df.columns and "dataset" in df.columns:
        df = df.rename(columns={"dataset": "dataset_source"})
    if "rna_sequence" in df.columns:
        df["rna_sequence"] = df["rna_sequence"].str.upper().str.replace("T", "U", regex=False)

    required = {"rna_sequence", "protein_sequence", "binding_label"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"Benchmark file missing required columns: {missing}\n"
                 f"Available: {df.columns.tolist()}")

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {args.max_rows:,} rows for quick test")

    n_pos = (df["binding_label"] == 1).sum()
    n_neg = (df["binding_label"] == 0).sum()
    n_prot = df["protein_name"].nunique() if "protein_name" in df.columns else "?"
    print(f"  Rows: {len(df):,} | pos: {n_pos:,} | neg: {n_neg:,} | proteins: {n_prot}\n")

    # ── Inference ──────────────────────────────────────────────────────────
    print("  Running inference...")
    t0 = time.time()

    if model_type == "v2_cnn":
        probs = run_inference_cnn(model, df, rna_max, prot_max, args.batch_size, device)
    elif model_type == "v1_mlp":
        probs = run_inference_mlp(model, df, args.batch_size, device)
    else:
        sys.exit(f"Inference not implemented for: {model_type}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(df)/elapsed:.0f} seqs/s)\n")

    y_true = df["binding_label"].values.astype(int)

    # ── Overall metrics ────────────────────────────────────────────────────
    overall = compute_metrics(y_true, probs)
    print(f"  {'AUROC':<12}: {overall['auroc']:.4f}  (ZHMolGraph: 0.798)")
    print(f"  {'AUPRC':<12}: {overall['auprc']:.4f}  (ZHMolGraph: 0.820)")
    print(f"  {'vs ZHMolGraph':<12}: AUROC {overall['auroc'] - 0.798:+.4f}  "
          f"AUPRC {overall['auprc'] - 0.820:+.4f}")

    # ── Per-protein metrics ────────────────────────────────────────────────
    pp_df = compute_per_protein(df, probs, args.min_pos, args.min_neg)
    valid_aurocs = pp_df[pp_df["auroc"].notna()]["auroc"].values

    if len(valid_aurocs) > 0:
        pp_summary = {
            "n_proteins_total":   int(len(pp_df)),
            "n_proteins_scored":  int(len(valid_aurocs)),
            "n_proteins_skipped": int(len(pp_df) - len(valid_aurocs)),
            "median_auroc":       float(np.median(valid_aurocs)),
            "mean_auroc":         float(np.mean(valid_aurocs)),
            "min_auroc":          float(np.min(valid_aurocs)),
            "max_auroc":          float(np.max(valid_aurocs)),
            "pct_above_random":   float((valid_aurocs > 0.5).mean()),
            "pct_above_07":       float((valid_aurocs > 0.7).mean()),
        }
        print(f"\n  Per-protein AUROC ({len(valid_aurocs)} proteins):")
        print(f"    median : {pp_summary['median_auroc']:.4f}")
        print(f"    min    : {pp_summary['min_auroc']:.4f}")
        print(f"    >0.7   : {pp_summary['pct_above_07']:.1%}")
    else:
        pp_summary = {}
        print("  [WARN] No proteins had sufficient pos+neg for AUROC")

    # ── Per-organism breakdown (if available) ─────────────────────────────
    org_summary = {}
    if "organism" in df.columns:
        print("\n  Per-organism AUROC:")
        for org, grp in df.groupby("organism"):
            y_o = grp["binding_label"].values
            p_o = probs[grp.index]
            if y_o.sum() < 5 or (y_o == 0).sum() < 5:
                continue
            try:
                a = float(roc_auc_score(y_o, p_o))
                org_summary[org] = {"auroc": a, "n": int(len(grp)),
                                    "n_pos": int(y_o.sum())}
                print(f"    {org:<35} AUROC={a:.4f}  n={len(grp):,}")
            except Exception:
                pass

    # ── Save predictions ───────────────────────────────────────────────────
    pred_df = df.copy()
    pred_df["prob"] = probs
    pred_path = os.path.join(args.output_dir, "predictions.tsv")
    pred_df.to_csv(pred_path, sep="\t", index=False, float_format="%.6f")
    print(f"\n  → {pred_path}  ({len(pred_df):,} rows)")

    # ── Save per-protein TSV ───────────────────────────────────────────────
    if len(pp_df) > 0:
        pp_path = os.path.join(args.output_dir, "per_protein.tsv")
        pp_df.to_csv(pp_path, sep="\t", index=False, float_format="%.5f")
        print(f"  → {pp_path}")

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics = {
        "checkpoint":   args.checkpoint,
        "benchmark":    args.benchmark,
        "model_type":   model_type,
        "rna_max":      rna_max,
        "prot_max":     prot_max,
        "n_rows":       int(len(df)),
        "overall":      overall,
        "per_protein":  pp_summary,
        "per_organism": org_summary,
        "vs_zhmolgraph": {
            "auroc": round(overall["auroc"] - 0.798, 4),
            "auprc": round(overall["auprc"] - 0.820, 4),
        },
        "elapsed_s": round(elapsed, 1),
    }
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  → {json_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    benchmark_label = os.path.splitext(os.path.basename(args.benchmark))[0]
    plot_curves(y_true, probs, args.output_dir, benchmark_label)
    if len(valid_aurocs) > 0:
        plot_per_protein_dist(pp_df, args.output_dir, benchmark_label)

    print(f"\n{'='*65}")
    print(f"  RESULT: AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}")
    if pp_summary:
        print(f"          Per-protein median AUROC: {pp_summary.get('median_auroc', '?'):.4f}")
    print(f"  All outputs in: {args.output_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
