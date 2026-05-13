"""
Script 10: Train Phase 2 Model V3c — RNA CNN + ESM-2 Residue CNN.

Architecture:
    RNA sequence (L×4, one-hot)      →  RNA CNN branch         →  256-d
    ESM-2 residues (L×1280, fp16)    →  Linear(1280→64) per pos
                                     →  Prot Residue CNN branch →  256-d
                                        concat → 512-d
                                        MLP [256, 64] → binding logit

Why this beats V3 and V3b (mean-pool ESM-2):
    Mean-pool dilutes the binding domain signal: a 300-aa protein has only
    ~30-50 residues in the actual RNA-binding domain. Averaging all 300
    dilutes the signal by ~6-10x. Two failed experiments (V3, V3b) confirmed
    this empirically.

    V3c applies Conv1D + global max pooling OVER residue positions, just as
    V2 applied Conv1D + max pooling over sequence characters. The model
    learns which residue windows are discriminative — effectively locating
    the binding domain automatically. This is the per-residue analogue of
    what the one-hot CNN already does for sequence motifs.

Memory note:
    Per-residue ESM-2 embeddings are large: batch of 256 × 300 × 1280 × 4B
    = ~390 MB. The Linear(1280→64) projection is applied per-position first,
    reducing memory to 256 × 300 × 64 × 4B = ~20 MB before Conv1D.
    This is manageable on MPS/CPU.

Usage (from protein_rna_ml/):
    # Step 1 (run once):
    python scripts/07b_extract_esm2_residues.py

    # Step 2:
    python scripts/10_train_generalized_v3c.py
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── One-hot encoding ──────────────────────────────────────────────────────────
RNA_ALPHA  = "AUGC"
AA_ALPHA   = "ACDEFGHIKLMNPQRSTVWY"
RNA_TO_IDX = {c: i for i, c in enumerate(RNA_ALPHA)}
AA_TO_IDX  = {c: i for i, c in enumerate(AA_ALPHA)}


def one_hot_rna(seq, max_len):
    t = torch.zeros(max_len, 4)
    for i, c in enumerate(str(seq).upper()[:max_len]):
        if c in RNA_TO_IDX: t[i, RNA_TO_IDX[c]] = 1.0
    return t


# ── Dataset ───────────────────────────────────────────────────────────────────
class V3cDataset(Dataset):
    """
    Returns (rna_oh, prot_residue_emb, label) for each sample.

    prot_residue_emb : (prot_max, 1280) float32  — per-residue ESM-2 embedding
                       (pre-loaded from npz, padded with zeros beyond actual length)
    """
    def __init__(self, tsv_path, residue_lookup, rna_max=60, prot_max=300):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.residue_lookup = residue_lookup   # dict: protein_name → (prot_max, 1280)
        self.rna_max  = rna_max
        self.prot_max = prot_max

        missing = ~self.df["protein_name"].isin(residue_lookup)
        if missing.any():
            print(f"  ⚠️  {missing.sum()} rows missing residue embedding — dropped")
            self.df = self.df[~missing].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        rna_oh   = one_hot_rna(row["rna_sequence"], self.rna_max)
        prot_emb = torch.tensor(self.residue_lookup[row["protein_name"]], dtype=torch.float32)
        label    = torch.tensor(float(row["binding_label"]), dtype=torch.float32)
        return rna_oh, prot_emb, label


# ── Model ─────────────────────────────────────────────────────────────────────
class RNAConvBranch(nn.Module):
    """Standard one-hot CNN branch (identical to V2)."""
    def __init__(self, in_ch=4, filters=(128, 256, 256), kernels=(7, 5, 3), dropout=0.3):
        super().__init__()
        layers, ch = [], in_ch
        for f, k in zip(filters, kernels):
            layers += [nn.Conv1d(ch, f, k, padding=k//2),
                       nn.BatchNorm1d(f), nn.GELU(), nn.Dropout(dropout)]
            ch = f
        self.net     = nn.Sequential(*layers)
        self.out_dim = filters[-1]

    def forward(self, x):
        # x: (B, L, in_ch) → conv expects (B, in_ch, L)
        return self.net(x.transpose(1, 2)).max(dim=-1).values


class ProtResidueConvBranch(nn.Module):
    """
    ESM-2 per-residue CNN branch.

    Input : (B, prot_max, esm_dim=1280)
    Step 1: Linear projection per position: 1280 → proj_dim (reduces memory)
    Step 2: Conv1D layers on projected residues
    Output: (B, out_dim) via global max pooling

    The linear projection is position-wise (weight shared across positions),
    equivalent to a Conv1d with kernel_size=1.
    """
    def __init__(self, esm_dim=1280, proj_dim=64,
                 filters=(128, 256), kernels=(7, 5), dropout=0.3):
        super().__init__()

        # Position-wise projection: compress 1280 → proj_dim before Conv1D
        # This is critical for memory: 1280 → 64 reduces activation memory 20×
        self.proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Conv1D stack on projected residues
        layers, ch = [], proj_dim
        for f, k in zip(filters, kernels):
            layers += [nn.Conv1d(ch, f, k, padding=k//2),
                       nn.BatchNorm1d(f), nn.GELU(), nn.Dropout(dropout)]
            ch = f
        self.convs   = nn.Sequential(*layers)
        self.out_dim = filters[-1]

    def forward(self, x):
        # x: (B, prot_max, 1280)
        x = self.proj(x)                      # (B, prot_max, proj_dim)
        x = self.convs(x.transpose(1, 2))     # (B, out_dim, prot_max)
        return x.max(dim=-1).values            # (B, out_dim)


class V3cModel(nn.Module):
    """
    V3c: RNA one-hot CNN + ESM-2 residue CNN → MLP head.

    Both branches produce 256-d representations via global max pooling.
    The protein branch applies learned filters over ESM-2 residue space,
    effectively performing motif detection in the evolutionary embedding space.
    """
    def __init__(self,
                 esm_dim=1280,
                 esm_proj_dim=64,
                 rna_filters=(128, 256, 256),
                 rna_kernels=(7, 5, 3),
                 prot_filters=(128, 256),
                 prot_kernels=(7, 5),
                 head_dims=(256, 64),
                 dropout=0.3):
        super().__init__()
        self.rna_branch  = RNAConvBranch(4, list(rna_filters), list(rna_kernels), dropout)
        self.prot_branch = ProtResidueConvBranch(
            esm_dim, esm_proj_dim, list(prot_filters), list(prot_kernels), dropout)

        in_dim = self.rna_branch.out_dim + self.prot_branch.out_dim
        head = []
        for h in head_dims:
            head += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        head.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head)

    def forward(self, rna_oh, prot_emb):
        rna_out  = self.rna_branch(rna_oh)      # (B, 256)
        prot_out = self.prot_branch(prot_emb)   # (B, 256)
        return self.head(torch.cat([rna_out, prot_out], dim=-1)).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────
# WeightedRandomSampler is NOT used — see note in 06_train_generalized_v2.py.
def make_sampler(labels):  # noqa: F401  (kept for reference, not called)
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    w = [n_neg / n_pos if l == 1 else 1.0 for l in labels]
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float32), len(w), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labs = [], []
    for rna, prot_emb, y in loader:
        rna, prot_emb = rna.to(device), prot_emb.to(device)
        p = torch.sigmoid(model(rna, prot_emb)).cpu().numpy()
        probs.append(p); labs.append(y.numpy())
    probs = np.concatenate(probs); labs = np.concatenate(labs)
    return {"auroc": float(roc_auc_score(labs, probs)),
            "auprc": float(average_precision_score(labs, probs)),
            "probs": probs}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/generalized")
    parser.add_argument("--emb_path",    default="data/embeddings/esm2_residue_embeddings.npz")
    parser.add_argument("--out_dir",     default="results/generalized")
    parser.add_argument("--model_dir",   default="models/saved/generalized_v3c")
    parser.add_argument("--rna_max",     type=int,   default=60)
    parser.add_argument("--prot_max",    type=int,   default=300)
    parser.add_argument("--esm_proj",    type=int,   default=64,
                        help="ESM-2 residue projection dim (memory reduction step)")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=128,
                        help="Smaller than V2 due to residue embedding memory")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--no_cuda",     action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if not args.no_cuda:
        if   torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")
    if device.type == "cpu":
        print("  ⚠️  Running on CPU. V3c with residue embeddings is slower than V2.")
        print("  Estimated training time: ~3-4 hours on CPU, ~40 min on MPS.")

    # ── Load ESM-2 residue embeddings ─────────────────────────────────────────
    if not os.path.exists(args.emb_path):
        print(f"\n❌  Residue embeddings not found: {args.emb_path}")
        print("   Run first: python scripts/07b_extract_esm2_residues.py")
        sys.exit(1)

    print(f"\n=== Loading ESM-2 residue embeddings: {args.emb_path} ===")
    t0 = time.time()
    data = np.load(args.emb_path)
    protein_ids = data["protein_ids"].tolist()
    embeddings  = data["embeddings"]    # (N, prot_max, 1280) float16
    lengths     = data["lengths"]       # (N,) actual lengths

    # Build lookup dict: protein_name → (prot_max, 1280) float16 array
    # Keep as float16 in lookup; convert to float32 in dataset __getitem__
    residue_lookup = {pid: embeddings[i] for i, pid in enumerate(protein_ids)}
    print(f"  Loaded {len(residue_lookup)} proteins  "
          f"| shape per protein: ({args.prot_max}, 1280)  "
          f"| file: {os.path.getsize(args.emb_path)/1e6:.0f} MB  "
          f"| {time.time()-t0:.1f}s")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n=== Loading sequence data from {args.data_dir}/ ===")
    make_ds = lambda split: V3cDataset(
        os.path.join(args.data_dir, f"{split}.tsv"),
        residue_lookup, rna_max=args.rna_max, prot_max=args.prot_max)

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  RNA padded to {args.rna_max} nt  |  ESM-2 residues padded to {args.prot_max} aa")

    train_labels = train_ds.df["binding_label"].values.tolist()
    # num_workers=0 to avoid memory issues with large embedding arrays.
    # Class imbalance is handled by pos_weight in the loss only (no sampler).
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = V3cModel(
        esm_dim=1280, esm_proj_dim=args.esm_proj,
        rna_filters=(128, 256, 256), rna_kernels=(7, 5, 3),
        prot_filters=(128, 256), prot_kernels=(7, 5),
        head_dims=(256, 64), dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  V3c model parameters: {n_params:,}")
    print(f"  RNA branch:  one-hot CNN (same as V2)")
    print(f"  Prot branch: ESM-2(1280) → Linear({args.esm_proj}) → Conv1D [128,256] → max pool")

    # ── Training ──────────────────────────────────────────────────────────────
    pos_weight = torch.tensor(
        [sum(l == 0 for l in train_labels) / max(sum(l == 1 for l in train_labels), 1)],
        dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auprc, best_auroc, best_epoch, no_improve = 0.0, 0.0, 0, 0
    ckpt_path = os.path.join(args.model_dir, "best_model.pt")

    print(f"\n=== Training V3c (max {args.epochs} epochs, patience={args.patience}) ===")
    print(f"  Primary metric: val AUPRC  |  Early stop patience: {args.patience}")
    print(f"  batch_size={args.batch_size}  lr={args.lr}")
    print(f"{'─'*65}")

    t_train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for rna, prot_emb, y in train_loader:
            rna, prot_emb, y = rna.to(device), prot_emb.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(rna, prot_emb)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validate every epoch
        val_m = evaluate(model, val_loader, device)

        is_best = val_m["auprc"] > best_auprc
        marker  = " ★" if is_best else ""
        if is_best:
            best_auprc, best_auroc, best_epoch, no_improve = (
                val_m["auprc"], val_m["auroc"], epoch, 0)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
                "args": vars(args),
            }, ckpt_path)
        else:
            no_improve += 1

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} | "
              f"val AUROC={val_m['auroc']:.4f} AUPRC={val_m['auprc']:.4f} | "
              f"{elapsed:.0f}s{marker}")

        if no_improve >= args.patience:
            print(f"\n  Early stop: no improvement for {args.patience} epochs.")
            break

    total_train_time = (time.time() - t_train_start) / 60
    print(f"\n  Best epoch: {best_epoch}  |  Val AUROC={best_auroc:.4f}  AUPRC={best_auprc:.4f}")
    print(f"  Training time: {total_train_time:.1f} min")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n=== Test evaluation (best epoch {best_epoch}) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device)
    print(f"  Test AUROC={test_m['auroc']:.4f}  AUPRC={test_m['auprc']:.4f}")

    # Per-protein test breakdown
    test_df = test_ds.df.copy()
    test_df["prob"] = test_m["probs"]
    per_prot = []
    for prot, grp in test_df.groupby("protein_name"):
        labs = grp["binding_label"].values
        prbs = grp["prob"].values
        if len(set(labs)) > 1:
            auroc = float(roc_auc_score(labs, prbs))
            auprc = float(average_precision_score(labs, prbs))
        else:
            auroc = auprc = float("nan")
        ds_name = grp["dataset"].iloc[0] if "dataset" in grp.columns else "unknown"
        per_prot.append({"protein": prot, "dataset": ds_name,
                         "auroc": round(auroc, 6), "auprc": round(auprc, 6), "n": len(grp)})

    # ── Compare vs V2 and V3/V3b ──────────────────────────────────────────────
    v2_auroc, v2_auprc = 0.7028, 0.5987
    v3b_auroc, v3b_auprc = 0.6656, 0.5675
    print(f"\n{'='*55}")
    print(f"  COMPARISON")
    print(f"  V2  CNN          : AUROC={v2_auroc:.4f}  AUPRC={v2_auprc:.4f}  (current best)")
    print(f"  V3b CNN+ESM-mean : AUROC={v3b_auroc:.4f}  AUPRC={v3b_auprc:.4f}")
    print(f"  V3c ESM-residue  : AUROC={test_m['auroc']:.4f}  AUPRC={test_m['auprc']:.4f}  ← this run")
    delta_v2  = test_m["auroc"] - v2_auroc
    delta_v3b = test_m["auroc"] - v3b_auroc
    print(f"\n  Δ vs V2 : {delta_v2:+.4f}  {'✅ improvement' if delta_v2 > 0 else '❌ degradation'}")
    print(f"  Δ vs V3b: {delta_v3b:+.4f}  {'✅ improvement' if delta_v3b > 0 else '❌ degradation'}")
    if test_m["auroc"] > v2_auroc:
        print(f"\n  ✅ V3c beats V2! ESM-2 residue CNN resolves mean-pool problem.")
    else:
        print(f"\n  ❌ V3c below V2. Consider: cross-attention (Phase 3) or RNA-FM branch.")
    print(f"{'='*55}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model": "generalized_v3c_esm2_residue_cnn",
        "architecture": f"RNA CNN(256) + ESM-2 residue Linear({args.esm_proj}) → Conv1D → max pool(256) → MLP",
        "best_val_auroc": best_auroc,
        "best_val_auprc": best_auprc,
        "best_epoch": best_epoch,
        "test_metrics": {"auroc": test_m["auroc"], "auprc": test_m["auprc"]},
        "vs_v2":  {"auroc": round(test_m["auroc"] - v2_auroc, 4),
                   "auprc": round(test_m["auprc"] - v2_auprc, 4)},
        "vs_v3b": {"auroc": round(test_m["auroc"] - v3b_auroc, 4),
                   "auprc": round(test_m["auprc"] - v3b_auprc, 4)},
        "training_time_min": round(total_train_time, 1),
        "per_protein": per_prot,
    }
    out_json = os.path.join(args.out_dir, "v3c_esm2_residue_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_json}")


if __name__ == "__main__":
    main()
