"""
Script 06: Train Phase 2 Model V2 — Dual-branch CNN on one-hot sequences.

This is a significant step up from k-mer MLP:
  - Learns position-sensitive motif filters instead of bag-of-words
  - Separate convolutional branches for RNA and protein
  - First layer ≈ learned position weight matrix (PWM)
  - Expected AUROC gain: ~0.04–0.08 over MLP

Usage (from protein_rna_ml/):
    python scripts/06_train_generalized_v2.py
    python scripts/06_train_generalized_v2.py --rna_max 60 --prot_max 800
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.cnn_model import RNABindingCNN
from src.data.dataset import SeqDataset


def make_weighted_sampler(labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    weights = [n_neg / n_pos if l == 1 else 1.0 for l in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights), replacement=True)


def evaluate(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for rna, prot, y in loader:
            rna, prot = rna.to(device), prot.to(device)
            p = torch.sigmoid(model(rna, prot)).cpu().numpy()
            probs_all.append(p)
            labels_all.append(y.numpy())
    probs  = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    return {"auroc": float(roc_auc_score(labels, probs)),
            "auprc": float(average_precision_score(labels, probs)),
            "probs": probs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/generalized")
    parser.add_argument("--out_dir",    default="results/generalized")
    parser.add_argument("--model_dir",  default="models/saved/generalized_v2")
    parser.add_argument("--rna_max",    type=int,   default=60)
    parser.add_argument("--prot_max",   type=int,   default=800)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=8)
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cpu":
        print("  ⚠️  No GPU detected. CNN training will be slow (~2–4h).")
        print("  Consider running on a machine with CUDA or reducing batch size.")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n=== Loading sequence data from {args.data_dir}/ ===")
    make_ds = lambda split: SeqDataset(
        os.path.join(args.data_dir, f"{split}.tsv"),
        rna_max_len=args.rna_max, prot_max_len=args.prot_max)

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  RNA padded to {args.rna_max} nt  |  Protein padded to {args.prot_max} aa")

    train_labels = train_ds.df["binding_label"].values.tolist()
    sampler = make_weighted_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RNABindingCNN(
        rna_filters=[128, 256, 256], rna_kernels=[7, 5, 3],
        prot_filters=[128, 256, 256], prot_kernels=[11, 7, 5],
        head_dims=[256, 64], dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    pos_weight = torch.tensor(
        [sum(l == 0 for l in train_labels) / sum(l == 1 for l in train_labels)],
        dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n=== Training CNN (max {args.epochs} epochs, patience={args.patience}) ===")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Val AUROC':>10}  {'Val AUPRC':>10}  {'Time':>6}")
    print(f"  {'─'*50}")

    best_auroc, best_epoch, no_improve = 0.0, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for rna, prot, y in train_loader:
            rna, prot, y = rna.to(device), prot.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rna, prot), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        val_m = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss,4),
                         "val_auroc": round(val_m["auroc"],4),
                         "val_auprc": round(val_m["auprc"],4)})
        print(f"  {epoch:>5}  {train_loss:>8.4f}  {val_m['auroc']:>10.4f}  "
              f"{val_m['auprc']:>10.4f}  {elapsed:>5.1f}s")

        if val_m["auroc"] > best_auroc:
            best_auroc, best_epoch, no_improve = val_m["auroc"], epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val_m, "args": vars(args)},
                       os.path.join(args.model_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\n  Best val AUROC: {best_auroc:.4f} at epoch {best_epoch}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device)
    print(f"\n=== Test set ===")
    print(f"  AUROC: {test_m['auroc']:.4f}   AUPRC: {test_m['auprc']:.4f}")

    # Per-protein on test
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.tsv"), sep="\t")
    test_df["prob"] = test_m["probs"]
    per_protein = []
    for prot, grp in test_df.groupby("protein_name"):
        if grp["binding_label"].nunique() < 2: continue
        per_protein.append({
            "protein": prot, "dataset": grp["dataset_source"].iloc[0],
            "auroc": float(roc_auc_score(grp["binding_label"], grp["prob"])),
            "n": len(grp), "is_flagged": int(grp["is_flagged"].iloc[0])})
    pp_aurocs = [p["auroc"] for p in per_protein]
    print(f"  Per-protein median: {np.median(pp_aurocs):.4f}  "
          f"min: {np.min(pp_aurocs):.4f}")

    results = {"model": "generalized_v2_cnn", "best_val_auroc": best_auroc,
               "test_metrics": {"auroc": test_m["auroc"], "auprc": test_m["auprc"]},
               "per_protein_summary": {"median": float(np.median(pp_aurocs)),
                                        "min": float(np.min(pp_aurocs))},
               "per_protein": per_protein, "history": history}
    with open(os.path.join(args.out_dir, "v2_cnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Phase 2 V2 (CNN) complete.")
    print(f"   Next: Phase 2 V3 — ESM-2 + RNA-FM embeddings (see METHODS.md §6.3)")

if __name__ == "__main__":
    main()
