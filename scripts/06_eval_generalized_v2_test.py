#!/usr/bin/env python3
"""
Evaluate a trained V2 CNN checkpoint on the held-out test split only.

Use when training was interrupted after best_model.pt was saved.

Usage:
    python scripts/06_eval_generalized_v2_test.py \\
        --data_dir data/generalized_v3a \\
        --checkpoint models/saved/generalized_v2/best_model.pt \\
        --prot_max 700 --no_cuda
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.dataset import SeqDataset
from src.models.cnn_model import RNABindingCNN


def evaluate(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for rna, prot, y in loader:
            rna, prot = rna.to(device), prot.to(device)
            p = torch.sigmoid(model(rna, prot)).cpu().numpy()
            probs_all.append(p)
            labels_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "probs": probs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/generalized_v3a")
    parser.add_argument("--checkpoint", default="models/saved/generalized_v2/best_model.pt")
    parser.add_argument("--out_dir", default="results/generalized/v3a_scale")
    parser.add_argument("--out_json", default=None,
                        help="Default: {out_dir}/v2_cnn_results.json")
    parser.add_argument("--rna_max", type=int, default=60)
    parser.add_argument("--prot_max", type=int, default=700)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loaded {args.checkpoint}")
    print(f"  best epoch: {ckpt.get('epoch')}")
    if "val_metrics" in ckpt:
        vm = ckpt["val_metrics"]
        print(f"  val AUROC: {vm.get('auroc', '?'):.4f}  val AUPRC: {vm.get('auprc', '?'):.4f}")

    test_path = os.path.join(args.data_dir, "test.tsv")
    if not os.path.exists(test_path):
        sys.exit(f"Test split not found: {test_path}")

    print(f"\n=== Loading test data: {test_path} ===")
    test_ds = SeqDataset(test_path, rna_max_len=args.rna_max, prot_max_len=args.prot_max)
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    print(f"  Test pairs: {len(test_ds):,}  |  device: {device}")

    model = RNABindingCNN(
        rna_filters=[128, 256, 256], rna_kernels=[7, 5, 3],
        prot_filters=[128, 256, 256], prot_kernels=[11, 7, 5],
        head_dims=[256, 64], dropout=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    print("\n=== Running test evaluation ===")
    test_m = evaluate(model, loader, device)
    print(f"  AUROC: {test_m['auroc']:.4f}   AUPRC: {test_m['auprc']:.4f}")

    test_df = pd.read_csv(test_path, sep="\t", low_memory=False)
    test_df["prob"] = test_m["probs"]
    if "dataset" not in test_df.columns and "dataset_source" in test_df.columns:
        test_df = test_df.rename(columns={"dataset_source": "dataset"})

    per_protein = []
    for prot, grp in test_df.groupby("protein_name"):
        if grp["binding_label"].nunique() < 2:
            continue
        per_protein.append({
            "protein": prot,
            "dataset": grp["dataset"].iloc[0] if "dataset" in grp.columns else "unknown",
            "auroc": float(roc_auc_score(grp["binding_label"], grp["prob"])),
            "n": len(grp),
            "is_flagged": int(grp["is_flagged"].iloc[0]) if "is_flagged" in grp.columns else 0,
        })
    pp_aurocs = [p["auroc"] for p in per_protein]
    print(f"  Per-protein median AUROC: {np.median(pp_aurocs):.4f}  "
          f"min: {np.min(pp_aurocs):.4f}  (n={len(per_protein)} proteins)")

    val_auroc = ckpt.get("val_metrics", {}).get("auroc")
    val_auprc = ckpt.get("val_metrics", {}).get("auprc")
    results = {
        "model": "generalized_v2_cnn",
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_val_auroc": val_auroc,
        "best_val_auprc": val_auprc,
        "test_metrics": {"auroc": test_m["auroc"], "auprc": test_m["auprc"]},
        "per_protein_summary": {
            "median": float(np.median(pp_aurocs)) if pp_aurocs else None,
            "min": float(np.min(pp_aurocs)) if pp_aurocs else None,
            "n_proteins": len(per_protein),
        },
        "per_protein": per_protein,
        "eval_only": True,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = args.out_json or os.path.join(args.out_dir, "v2_cnn_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
