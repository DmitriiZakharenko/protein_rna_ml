"""
Script 05: Train Phase 2 Model V1 — MLP on k-mer features (multi-dataset).

Trains on the combined generalized dataset built by script 04.
Evaluates on val set every epoch with early stopping.
Saves best checkpoint and per-protein AUROC breakdown.

Usage (from protein_rna_ml/):
    python scripts/05_train_generalized_v1.py
    python scripts/05_train_generalized_v1.py --epochs 100 --lr 5e-4
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
from src.models.mlp_model import RNABindingMLP
from src.data.dataset import KmerDataset


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Over-sample positives to achieve 1:1 ratio during training."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    weights = np.where(y == 1, n_neg / n_pos, 1.0)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            probs = torch.sigmoid(model(X)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/generalized")
    parser.add_argument("--out_dir",    default="results/generalized")
    parser.add_argument("--model_dir",  default="models/saved/generalized_v1")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=8,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"\n  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n=== Loading data from {args.data_dir}/ ===")
    train_ds = KmerDataset(os.path.join(args.data_dir, "train_kmer.npz"))
    val_ds   = KmerDataset(os.path.join(args.data_dir, "val_kmer.npz"))
    test_ds  = KmerDataset(os.path.join(args.data_dir, "test_kmer.npz"))

    input_dim = train_ds.X.shape[1]
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"Test: {len(test_ds):,}  Input dim: {input_dim}")

    # Weighted sampler: 1:1 pos:neg ratio during training
    sampler = make_weighted_sampler(train_ds.y.numpy())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RNABindingMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Pos weight for BCE loss: n_neg / n_pos
    pos_weight = torch.tensor(
        [(train_ds.y == 0).sum() / (train_ds.y == 1).sum()], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n=== Training (max {args.epochs} epochs, patience={args.patience}) ===")
    print(f"  {'Epoch':>5}  {'Train loss':>11}  {'Val AUROC':>10}  "
          f"{'Val AUPRC':>10}  {'LR':>10}  {'Time':>6}")
    print(f"  {'─'*65}")

    best_val_auroc = 0.0
    best_epoch     = 0
    history        = []
    no_improve     = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= len(train_ds)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        print(f"  {epoch:>5}  {train_loss:>11.4f}  {val_metrics['auroc']:>10.4f}  "
              f"{val_metrics['auprc']:>10.4f}  "
              f"{scheduler.get_last_lr()[0]:>10.2e}  {elapsed:>5.1f}s")

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch     = epoch
            no_improve     = 0
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "args": vars(args)},
                       os.path.join(args.model_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print(f"\n  Best val AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n=== Test set evaluation ===")
    ckpt = torch.load(os.path.join(args.model_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"  AUROC : {test_metrics['auroc']:.4f}")
    print(f"  AUPRC : {test_metrics['auprc']:.4f}")

    # Per-protein AUROC on test set
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.tsv"), sep="\t")
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X, _ in test_loader:
            all_probs.append(torch.sigmoid(model(X.to(device))).cpu().numpy())
    test_probs = np.concatenate(all_probs)
    test_df["prob"] = test_probs

    per_protein = []
    for prot, grp in test_df.groupby("protein_name"):
        if grp["binding_label"].nunique() < 2:
            continue
        per_protein.append({
            "protein": prot,
            "dataset": grp["dataset_source"].iloc[0],
            "auroc": float(roc_auc_score(grp["binding_label"], grp["prob"])),
            "auprc": float(average_precision_score(grp["binding_label"], grp["prob"])),
            "n": len(grp),
            "is_flagged": int(grp["is_flagged"].iloc[0]),
        })
    per_protein_df = pd.DataFrame(per_protein).sort_values("auroc", ascending=False)

    pp_aurocs = per_protein_df["auroc"].values
    print(f"\n  Per-protein AUROC — median: {np.median(pp_aurocs):.4f}  "
          f"min: {np.min(pp_aurocs):.4f}  max: {np.max(pp_aurocs):.4f}")

    flagged = per_protein_df[per_protein_df["is_flagged"] == 1]
    if len(flagged):
        print(f"\n  Flagged proteins:")
        for _, r in flagged.iterrows():
            print(f"    {r['protein']:<15} AUROC={r['auroc']:.3f} ({r['dataset']})")

    # Save results
    results = {
        "model": "generalized_v1_mlp",
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "per_protein_summary": {
            "median": float(np.median(pp_aurocs)),
            "mean":   float(np.mean(pp_aurocs)),
            "min":    float(np.min(pp_aurocs)),
            "max":    float(np.max(pp_aurocs)),
        },
        "per_protein": per_protein,
        "history": history,
    }
    out_path = os.path.join(args.out_dir, "v1_mlp_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    per_protein_df.to_csv(
        os.path.join(args.out_dir, "v1_mlp_per_protein.tsv"), sep="\t", index=False)

    print(f"\n  Results → {out_path}")
    print(f"\n✅ Phase 2 V1 complete.")
    print(f"   Next: python scripts/06_train_generalized_v2.py  (CNN model)")

if __name__ == "__main__":
    main()
