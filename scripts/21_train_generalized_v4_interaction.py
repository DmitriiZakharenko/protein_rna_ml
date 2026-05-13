#!/usr/bin/env python3
"""
21_train_generalized_v4_interaction.py
Phase 3B — V4: Dual-branch CNN + Bilinear Interaction Layer.

Upgrade over V2 CNN (script 06):
  - BilinearInteraction(rna_emb, prot_emb) instead of raw concatenation
  - Optional dataset_source embedding (--use_source_emb)
  - Larger MLP head to handle richer input (512→128→32)
  - prot_max defaults to 700 to handle RNAcompete construct sequences
  - Saves full training config in checkpoint for reproducibility

Interaction modes (--interaction):
  concat     — V2 baseline (use to verify script parity)
  bilinear   — bilinear only, no residual concat
  concat_bi  — concat + bilinear residual [DEFAULT, recommended]

Usage:
  # V4 on original SELEX+RBNS data (compare directly to V2):
  python scripts/21_train_generalized_v4_interaction.py \
      --data_dir data/generalized_v2 \
      --interaction concat_bi \
      --out_dir results/generalized/v4_bilinear \
      --model_dir models/saved/generalized_v4

  # V4 on Phase 3A expanded data (RNAcompete + SELEX):
  python scripts/21_train_generalized_v4_interaction.py \
      --data_dir data/generalized_v3a \
      --interaction concat_bi \
      --use_source_emb \
      --prot_max 700 \
      --out_dir results/generalized/v4_phase3a \
      --model_dir models/saved/generalized_v4_phase3a

  # Ablation: V4 concat only (should match V2):
  python scripts/21_train_generalized_v4_interaction.py \
      --data_dir data/generalized_v2 \
      --interaction concat \
      --out_dir results/generalized/v4_concat_ablation
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
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.interaction_model import RNABindingV4, DatasetSourceEmbedding
from src.data.dataset import SeqDataset


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: RNABindingV4, loader: DataLoader, device: torch.device,
             use_source_emb: bool) -> dict:
    model.eval()
    probs_all, labels_all = [], []
    for batch in loader:
        if use_source_emb and len(batch) == 4:
            rna, prot, y, sources = batch
            sources = list(sources)
        else:
            rna, prot, y = batch[:3]
            sources = None
        rna, prot = rna.to(device), prot.to(device)
        p = torch.sigmoid(model(rna, prot, sources)).cpu().numpy()
        probs_all.append(p)
        labels_all.append(y.numpy())
    probs  = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "probs": probs,
    }


# ── Dataset with source tag ───────────────────────────────────────────────────

class SeqDatasetWithSource(SeqDataset):
    """SeqDataset extended to also return dataset_source as string."""

    def __init__(self, tsv_path: str, rna_max_len: int = 60, prot_max_len: int = 700):
        super().__init__(tsv_path, rna_max_len=rna_max_len, prot_max_len=prot_max_len)
        src_col = "dataset_source" if "dataset_source" in self.df.columns else \
                  "dataset"        if "dataset"        in self.df.columns else None
        self.source_col = src_col

    def __getitem__(self, idx):
        rna_oh, prot_oh, label = super().__getitem__(idx)
        if self.source_col:
            src = str(self.df.iloc[idx][self.source_col])
        else:
            src = "unknown"
        return rna_oh, prot_oh, label, src


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    parser.add_argument("--data_dir",    default="data/generalized_v2")
    parser.add_argument("--out_dir",     default="results/generalized/v4_bilinear")
    parser.add_argument("--model_dir",   default="models/saved/generalized_v4")
    # Architecture
    parser.add_argument("--interaction", default="concat_bi",
                        choices=["concat", "bilinear", "concat_bi"])
    parser.add_argument("--inter_dim",   type=int, default=256,
                        help="Bilinear interaction output dim")
    parser.add_argument("--rna_max",     type=int, default=60)
    parser.add_argument("--prot_max",    type=int, default=700)
    parser.add_argument("--use_source_emb", action="store_true",
                        help="Add dataset_source embedding to MLP input")
    parser.add_argument("--source_emb_dim", type=int, default=16)
    # Training
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    if not args.no_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*65}")
    print(f"  V4 CNN + Bilinear Interaction")
    print(f"  interaction={args.interaction}  device={device}")
    print(f"  source_emb={args.use_source_emb}  prot_max={args.prot_max}")
    print(f"{'='*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    DatasetClass = SeqDatasetWithSource if args.use_source_emb else SeqDataset
    make_ds = lambda split: DatasetClass(
        os.path.join(args.data_dir, f"{split}.tsv"),
        rna_max_len=args.rna_max, prot_max_len=args.prot_max)

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    def collate(batch):
        if args.use_source_emb:
            rna   = torch.stack([b[0] for b in batch])
            prot  = torch.stack([b[1] for b in batch])
            y     = torch.stack([b[2] for b in batch])
            srcs  = [b[3] for b in batch]
            return rna, prot, y, srcs
        rna  = torch.stack([b[0] for b in batch])
        prot = torch.stack([b[1] for b in batch])
        y    = torch.stack([b[2] for b in batch])
        return rna, prot, y

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"),
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=2, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=2, collate_fn=collate)

    # ── Model ─────────────────────────────────────────────────────────────
    model = RNABindingV4(
        rna_filters   = [128, 256, 256],
        prot_filters  = [128, 256, 256],
        rna_kernels   = [7, 5, 3],
        prot_kernels  = [11, 7, 5],
        interaction   = args.interaction,
        inter_dim     = args.inter_dim,
        head_dims     = [512, 128, 32],
        dropout       = args.dropout,
        use_source_emb= args.use_source_emb,
        source_emb_dim= args.source_emb_dim,
    ).to(device)

    train_labels = train_ds.df["binding_label"].values.tolist()
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"  pos_weight = {pos_weight.item():.3f}  (n_neg/n_pos = {n_neg}/{n_pos})\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'ValAUROC':>9}  {'ValAUPRC':>9}  "
          f"{'Time':>6}  Note")
    print(f"  {'─'*60}")

    best_auprc = 0.0
    best_auroc = 0.0
    best_epoch = 0
    no_improve = 0
    history    = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if args.use_source_emb and len(batch) == 4:
                rna, prot, y, sources = batch
            else:
                rna, prot, y = batch[:3]
                sources = None
            rna, prot, y = rna.to(device), prot.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rna, prot, sources), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        val_m  = evaluate(model, val_loader, device, args.use_source_emb)
        scheduler.step()
        elapsed = time.time() - t0
        is_best = val_m["auprc"] > best_auprc
        note    = "★ best" if is_best else ""

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_auroc":  round(val_m["auroc"], 4),
            "val_auprc":  round(val_m["auprc"], 4),
        })
        print(f"  {epoch:>5}  {train_loss:>8.4f}  {val_m['auroc']:>9.4f}  "
              f"{val_m['auprc']:>9.4f}  {elapsed:>5.1f}s  {note}")

        if is_best:
            best_auprc, best_auroc, best_epoch, no_improve = (
                val_m["auprc"], val_m["auroc"], epoch, 0)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
                "args":        vars(args),
                "arch": {
                    "interaction":    args.interaction,
                    "inter_dim":      args.inter_dim,
                    "use_source_emb": args.use_source_emb,
                    "rna_max":        args.rna_max,
                    "prot_max":       args.prot_max,
                },
            }, os.path.join(args.model_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stop at epoch {epoch} (patience={args.patience})")
                break

    print(f"\n  Best val AUPRC: {best_auprc:.4f}  AUROC: {best_auroc:.4f}  "
          f"epoch: {best_epoch}")

    # ── Test evaluation ───────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(args.model_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_m = evaluate(model, test_loader, device, args.use_source_emb)
    print(f"\n  Test AUROC: {test_m['auroc']:.4f}  AUPRC: {test_m['auprc']:.4f}")

    # Per-protein test metrics
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.tsv"), sep="\t")
    if "dataset_source" not in test_df.columns and "dataset" in test_df.columns:
        test_df = test_df.rename(columns={"dataset": "dataset_source"})
    test_df["prob"] = test_m["probs"]
    per_protein = []
    for prot, grp in test_df.groupby("protein_name"):
        if grp["binding_label"].nunique() < 2:
            continue
        per_protein.append({
            "protein":    prot,
            "dataset":    grp.get("dataset_source", pd.Series(["unknown"])).iloc[0],
            "auroc":      float(roc_auc_score(grp["binding_label"], grp["prob"])),
            "n":          len(grp),
        })
    pp_aurocs = [p["auroc"] for p in per_protein]
    print(f"  Per-protein median: {np.median(pp_aurocs):.4f}  "
          f"min: {np.min(pp_aurocs):.4f}  n={len(pp_aurocs)}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "model":         f"V4_CNN_{args.interaction}",
        "interaction":   args.interaction,
        "use_source_emb": args.use_source_emb,
        "data_dir":      args.data_dir,
        "best_val_auroc": best_auroc,
        "best_val_auprc": best_auprc,
        "best_epoch":    best_epoch,
        "test_metrics":  {"auroc": test_m["auroc"], "auprc": test_m["auprc"]},
        "per_protein_summary": {
            "median": float(np.median(pp_aurocs)),
            "min":    float(np.min(pp_aurocs)),
            "n":      len(pp_aurocs),
        },
        "per_protein": per_protein,
        "history":     history,
        "vs_v2": {
            "auroc": round(test_m["auroc"] - 0.690, 4),  # V2 bug-affected anchor
            "auprc": round(test_m["auprc"] - 0.580, 4),
        },
    }
    out_name = f"v4_{args.interaction}_results.json"
    json_path = os.path.join(args.out_dir, out_name)
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results → {json_path}")
    print(f"  Model   → {args.model_dir}/best_model.pt")


if __name__ == "__main__":
    main()
