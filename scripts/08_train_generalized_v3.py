"""
Script 08: Train Phase 2 Model V3 — ESM-2 protein embeddings + RNA CNN.

Architecture:
    RNA sequence (L × 4, one-hot)  →  RNA CNN branch  →  256-d RNA embedding
    ESM-2 embedding (1280-d)       →  Linear(256) + GELU  →  256-d protein embedding
                                      concat → 512-d
                                      MLP [256, 64] → binding logit

Why this beats V2 (one-hot CNN):
    - Protein branch now uses pre-trained evolutionary language model instead of
      random one-hot convolutions that must learn protein grammar from scratch
    - ESM-2 embeddings encode structural domains, active sites, co-evolution
    - RNA branch unchanged — CNN already learns positional motifs well
    - Expected test AUROC: 0.78-0.82 (vs 0.703 for V2)

Early stopping: val AUPRC (more informative than AUROC under class imbalance)

Usage (from protein_rna_ml/):
    python scripts/08_train_generalized_v3.py
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


# ── Dataset ───────────────────────────────────────────────────────────────────

RNA_ALPHA = "AUGC"
RNA_TO_IDX = {c: i for i, c in enumerate(RNA_ALPHA)}


def one_hot_rna(seq: str, max_len: int) -> torch.Tensor:
    arr = torch.zeros(max_len, 4)
    for i, c in enumerate(str(seq).upper()[:max_len]):
        if c in RNA_TO_IDX:
            arr[i, RNA_TO_IDX[c]] = 1.0
    return arr


class ESM2Dataset(Dataset):
    """
    Returns:
        rna_oh     : (rna_max, 4)     one-hot RNA
        prot_emb   : (1280,)          ESM-2 mean-pool embedding (float32)
        label      : scalar float
    """
    def __init__(
        self,
        tsv_path: str,
        emb_lookup: dict,          # protein_name → np.array(1280,)
        rna_max_len: int = 60,
    ):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.emb_lookup = emb_lookup
        self.rna_max = rna_max_len

        # Drop rows where protein has no embedding (shouldn't happen but safe)
        missing = ~self.df["protein_name"].isin(emb_lookup)
        if missing.any():
            n = missing.sum()
            print(f"  ⚠️  {n} rows have no ESM-2 embedding — dropped")
            self.df = self.df[~missing].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rna_oh = one_hot_rna(row["rna_sequence"], self.rna_max)
        prot_emb = torch.tensor(self.emb_lookup[row["protein_name"]], dtype=torch.float32)
        label = torch.tensor(float(row["binding_label"]), dtype=torch.float32)
        return rna_oh, prot_emb, label


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBranch(nn.Module):
    def __init__(self, in_ch, filters, kernels, dropout=0.3):
        super().__init__()
        layers = []
        ch = in_ch
        for f, k in zip(filters, kernels):
            layers += [nn.Conv1d(ch, f, k, padding=k//2), nn.BatchNorm1d(f), nn.GELU(), nn.Dropout(dropout)]
            ch = f
        self.net = nn.Sequential(*layers)
        self.out_dim = filters[-1]

    def forward(self, x):
        return self.net(x.transpose(1, 2)).max(dim=-1).values


class ESM2RNACNNModel(nn.Module):
    """
    ESM-2 embedding (1280-d) + RNA CNN branch → MLP head.

    Protein branch: Linear projection (1280 → 256) replaces one-hot CNN.
    RNA branch: unchanged from V2 (positional motif learning).
    """
    def __init__(
        self,
        esm_dim: int = 1280,
        rna_filters=(128, 256, 256),
        rna_kernels=(7, 5, 3),
        proj_dim: int = 256,
        head_dims=(256, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        # Protein: ESM-2 → projection
        self.prot_proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # RNA: CNN branch
        self.rna_branch = ConvBranch(4, list(rna_filters), list(rna_kernels), dropout)

        # Head
        in_dim = proj_dim + self.rna_branch.out_dim   # 256 + 256 = 512
        head = []
        for h in head_dims:
            head += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        head.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head)

    def forward(self, rna_oh, prot_emb):
        rna_emb  = self.rna_branch(rna_oh)    # (B, 256)
        prot_out = self.prot_proj(prot_emb)   # (B, 256)
        return self.head(torch.cat([rna_emb, prot_out], dim=-1)).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────

# WeightedRandomSampler is NOT used — see note in 06_train_generalized_v2.py.
# Using it together with BCEWithLogitsLoss(pos_weight) double-counts class imbalance.
def make_sampler(labels):  # noqa: F401  (kept for reference, not called)
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    w = [n_neg / n_pos if l == 1 else 1.0 for l in labels]
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float32), len(w), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labs = [], []
    for rna, prot, y in loader:
        rna, prot = rna.to(device), prot.to(device)
        p = torch.sigmoid(model(rna, prot)).cpu().numpy()
        probs.append(p); labs.append(y.numpy())
    probs = np.concatenate(probs); labs = np.concatenate(labs)
    return {
        "auroc": float(roc_auc_score(labs, probs)),
        "auprc": float(average_precision_score(labs, probs)),
        "probs": probs,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/generalized")
    parser.add_argument("--emb_path",   default="data/embeddings/esm2_protein_embeddings.npz")
    parser.add_argument("--out_dir",    default="results/generalized")
    parser.add_argument("--model_dir",  default="models/saved/generalized_v3")
    parser.add_argument("--rna_max",    type=int,   default=60)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=512)  # larger than V2 (no protein CNN)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=8)
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if not args.no_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # ── Load ESM-2 embeddings ─────────────────────────────────────────────────
    if not os.path.exists(args.emb_path):
        print(f"\n❌  ESM-2 embeddings not found: {args.emb_path}")
        print("   Run first: python scripts/07_extract_esm2_embeddings.py")
        sys.exit(1)

    print(f"\n=== Loading ESM-2 embeddings: {args.emb_path} ===")
    emb_data = np.load(args.emb_path, allow_pickle=True)
    protein_ids = emb_data["protein_ids"].tolist()
    embeddings  = emb_data["embeddings"].astype(np.float32)  # float16 → float32
    emb_lookup  = {pid: emb for pid, emb in zip(protein_ids, embeddings)}
    esm_dim = embeddings.shape[1]
    print(f"  Proteins: {len(emb_lookup)}  ESM-2 dim: {esm_dim}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print(f"\n=== Loading sequence data ===")
    make_ds = lambda split: ESM2Dataset(
        os.path.join(args.data_dir, f"{split}.tsv"), emb_lookup, args.rna_max)

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_labels = train_ds.df["binding_label"].values.tolist()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ESM2RNACNNModel(
        esm_dim=esm_dim,
        rna_filters=[128, 256, 256], rna_kernels=[7, 5, 3],
        proj_dim=256, head_dims=[256, 64], dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}  (much faster than V2 — no protein CNN)")

    pos_weight = torch.tensor(
        [sum(l == 0 for l in train_labels) / max(sum(l == 1 for l in train_labels), 1)],
        dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n=== Training V3 (max {args.epochs} epochs, early stop on val AUPRC, patience={args.patience}) ===")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'ValAUROC':>9}  {'ValAUPRC':>9}  {'Time':>6}  {'*':>3}")
    print(f"  {'─'*55}")

    best_auprc, best_epoch, no_improve = 0.0, 0, 0
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

        is_best = val_m["auprc"] > best_auprc
        marker = "★" if is_best else ""
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                         "val_auroc": round(val_m["auroc"], 4),
                         "val_auprc": round(val_m["auprc"], 4)})
        print(f"  {epoch:>5}  {train_loss:>8.4f}  {val_m['auroc']:>9.4f}  "
              f"{val_m['auprc']:>9.4f}  {elapsed:>5.1f}s  {marker}")

        if is_best:
            best_auprc, best_epoch, no_improve = val_m["auprc"], epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val_m, "args": vars(args)},
                       os.path.join(args.model_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\n  Best val AUPRC: {best_auprc:.4f} at epoch {best_epoch}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(args.model_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device)
    print(f"\n=== Test set ===")
    print(f"  AUROC: {test_m['auroc']:.4f}   AUPRC: {test_m['auprc']:.4f}")
    print(f"  ZHMolGraph target: AUROC≥0.798  AUPRC≥0.820")

    test_df = pd.read_csv(os.path.join(args.data_dir, "test.tsv"), sep="\t")
    test_df["prob"] = test_m["probs"]
    per_protein = []
    for prot, grp in test_df.groupby("protein_name"):
        if grp["binding_label"].nunique() < 2: continue
        per_protein.append({
            "protein": prot,
            "dataset": grp["dataset_source"].iloc[0],
            "auroc": float(roc_auc_score(grp["binding_label"], grp["prob"])),
            "auprc": float(average_precision_score(grp["binding_label"], grp["prob"])),
            "n": len(grp),
        })
    pp_aurocs = [p["auroc"] for p in per_protein]
    pp_auprcs = [p["auprc"] for p in per_protein]
    print(f"  Per-protein median AUROC: {np.median(pp_aurocs):.4f}  "
          f"AUPRC: {np.median(pp_auprcs):.4f}")

    results = {
        "model": "generalized_v3_esm2_rna_cnn",
        "best_val_auprc": best_auprc,
        "best_val_auroc": history[best_epoch - 1]["val_auroc"],
        "test_metrics": {"auroc": test_m["auroc"], "auprc": test_m["auprc"]},
        "vs_zhmolgraph": {
            "auroc_gap": round(test_m["auroc"] - 0.798, 4),
            "auprc_gap": round(test_m["auprc"] - 0.820, 4),
        },
        "per_protein_summary": {
            "median_auroc": float(np.median(pp_aurocs)),
            "median_auprc": float(np.median(pp_auprcs)),
            "min_auroc": float(np.min(pp_aurocs)),
        },
        "per_protein": per_protein,
        "history": history,
    }
    out_path = os.path.join(args.out_dir, "v3_esm2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Phase 2 V3 complete. Results: {out_path}")

    if test_m["auroc"] >= 0.798:
        print("  🎯 AUROC target REACHED!")
    elif test_m["auprc"] >= 0.820:
        print("  🎯 AUPRC target REACHED!")
    else:
        gap = 0.798 - test_m["auroc"]
        print(f"  Still {gap:.3f} AUROC below target → run scripts/09_train_generalized_v3b.py")


if __name__ == "__main__":
    main()
