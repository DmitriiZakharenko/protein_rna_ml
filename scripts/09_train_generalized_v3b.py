"""
Script 09: Train Phase 2 Model V3b — RNA CNN + Protein CNN + ESM-2 concat.

Architecture:
    RNA sequence (L×4, one-hot)    →  RNA CNN branch     →  256-d
    Protein seq  (L×20, one-hot)   →  Protein CNN branch →  256-d   [proven in V2]
    ESM-2 embedding (1280-d)       →  Linear(1280→128)   →  128-d   [auxiliary]
                                      concat → 640-d
                                      MLP [256, 64] → binding logit

Why this beats V3 (ESM-2 mean-pool only):
    V2 CNN excels at proteins with strong positional motifs (PUM2 Δ+0.253,
    LARP7 Δ+0.202, TRA2A Δ+0.150). ESM-2 excels at motif-poor proteins
    (UNK Δ+0.245, IGF2BP3 Δ+0.095, TAF15 Δ+0.060).
    Concatenating both lets the MLP head learn which representation to trust
    per protein family — end-to-end, no hand-crafted gating.

    ESM-2 projection is smaller (128-d not 256-d) to prevent it overpowering
    the well-trained CNN branch.

Usage (from protein_rna_ml/):
    python scripts/09_train_generalized_v3b.py

    # Requires ESM-2 embeddings (run once):
    python scripts/07_extract_esm2_embeddings.py
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
RNA_ALPHA = "AUGC"
AA_ALPHA  = "ACDEFGHIKLMNPQRSTVWY"
RNA_TO_IDX  = {c: i for i, c in enumerate(RNA_ALPHA)}
AA_TO_IDX   = {c: i for i, c in enumerate(AA_ALPHA)}

def one_hot_rna(seq, max_len):
    t = torch.zeros(max_len, 4)
    for i, c in enumerate(str(seq).upper()[:max_len]):
        if c in RNA_TO_IDX: t[i, RNA_TO_IDX[c]] = 1.0
    return t

def one_hot_prot(seq, max_len):
    t = torch.zeros(max_len, 20)
    for i, c in enumerate(str(seq).upper()[:max_len]):
        if c in AA_TO_IDX: t[i, AA_TO_IDX[c]] = 1.0
    return t


# ── Dataset ───────────────────────────────────────────────────────────────────
class V3bDataset(Dataset):
    def __init__(self, tsv_path, emb_lookup, rna_max=60, prot_max=300):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.emb_lookup = emb_lookup
        self.rna_max  = rna_max
        self.prot_max = prot_max
        missing = ~self.df["protein_name"].isin(emb_lookup)
        if missing.any():
            print(f"  ⚠️  {missing.sum()} rows missing ESM-2 embedding — dropped")
            self.df = self.df[~missing].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rna_oh   = one_hot_rna(row["rna_sequence"],   self.rna_max)
        prot_oh  = one_hot_prot(row["protein_sequence"], self.prot_max)
        prot_emb = torch.tensor(self.emb_lookup[row["protein_name"]], dtype=torch.float32)
        label    = torch.tensor(float(row["binding_label"]), dtype=torch.float32)
        return rna_oh, prot_oh, prot_emb, label


# ── Model ─────────────────────────────────────────────────────────────────────
class ConvBranch(nn.Module):
    def __init__(self, in_ch, filters, kernels, dropout=0.3):
        super().__init__()
        layers, ch = [], in_ch
        for f, k in zip(filters, kernels):
            layers += [nn.Conv1d(ch,f,k,padding=k//2), nn.BatchNorm1d(f), nn.GELU(), nn.Dropout(dropout)]
            ch = f
        self.net = nn.Sequential(*layers)
        self.out_dim = filters[-1]
    def forward(self, x):
        return self.net(x.transpose(1,2)).max(dim=-1).values


class V3bModel(nn.Module):
    """
    RNA CNN (256) + Protein CNN (256) + ESM-2 projection (128) → MLP head.

    The protein CNN branch learns binding motifs directly from sequence (proven in V2).
    The ESM-2 auxiliary provides evolutionary context for motif-poor proteins.
    The MLP head implicitly learns to gate between representations.
    """
    def __init__(self, esm_dim=1280, rna_filters=(128,256,256), rna_kernels=(7,5,3),
                 prot_filters=(128,256,256), prot_kernels=(11,7,5),
                 esm_proj_dim=128, head_dims=(256,64), dropout=0.3):
        super().__init__()
        self.rna_branch  = ConvBranch(4,  list(rna_filters),  list(rna_kernels),  dropout)
        self.prot_branch = ConvBranch(20, list(prot_filters), list(prot_kernels), dropout)
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, esm_proj_dim),
            nn.LayerNorm(esm_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        in_dim = self.rna_branch.out_dim + self.prot_branch.out_dim + esm_proj_dim
        head = []
        for h in head_dims:
            head += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        head.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head)

    def forward(self, rna_oh, prot_oh, prot_emb):
        rna_out  = self.rna_branch(rna_oh)     # (B, 256)
        prot_out = self.prot_branch(prot_oh)   # (B, 256)
        esm_out  = self.esm_proj(prot_emb)     # (B, 128)
        return self.head(torch.cat([rna_out, prot_out, esm_out], dim=-1)).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────
# WeightedRandomSampler is NOT used — see note in 06_train_generalized_v2.py.
def make_sampler(labels):  # noqa: F401  (kept for reference, not called)
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    w = [n_neg/n_pos if l==1 else 1.0 for l in labels]
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float32), len(w), replacement=True)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labs = [], []
    for rna, prot_oh, prot_emb, y in loader:
        rna, prot_oh, prot_emb = rna.to(device), prot_oh.to(device), prot_emb.to(device)
        p = torch.sigmoid(model(rna, prot_oh, prot_emb)).cpu().numpy()
        probs.append(p); labs.append(y.numpy())
    probs = np.concatenate(probs); labs = np.concatenate(labs)
    return {"auroc": float(roc_auc_score(labs, probs)),
            "auprc": float(average_precision_score(labs, probs)),
            "probs": probs}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/generalized")
    parser.add_argument("--emb_path",    default="data/embeddings/esm2_protein_embeddings.npz")
    parser.add_argument("--out_dir",     default="results/generalized")
    parser.add_argument("--model_dir",   default="models/saved/generalized_v3b")
    parser.add_argument("--rna_max",     type=int,   default=60)
    parser.add_argument("--prot_max",    type=int,   default=300)
    parser.add_argument("--esm_proj",    type=int,   default=128,
                        help="ESM-2 projection dim (smaller = less ESM-2 dominance)")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--no_cuda",     action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    if not args.no_cuda:
        if   torch.cuda.is_available():                               device = torch.device("cuda")
        elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                                          device = torch.device("cpu")
    else: device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # ── ESM-2 embeddings ──────────────────────────────────────────────────────
    if not os.path.exists(args.emb_path):
        print(f"\n❌  ESM-2 embeddings not found: {args.emb_path}")
        print("   Run: python scripts/07_extract_esm2_embeddings.py"); sys.exit(1)

    emb_data   = np.load(args.emb_path, allow_pickle=True)
    emb_lookup = {pid: emb.astype(np.float32)
                  for pid, emb in zip(emb_data["protein_ids"].tolist(),
                                      emb_data["embeddings"])}
    esm_dim = list(emb_lookup.values())[0].shape[0]
    print(f"\n  ESM-2 embeddings: {len(emb_lookup)} proteins, dim={esm_dim}")

    # ── Data ──────────────────────────────────────────────────────────────────
    make_ds = lambda s: V3bDataset(
        os.path.join(args.data_dir, f"{s}.tsv"), emb_lookup, args.rna_max, args.prot_max)
    train_ds = make_ds("train"); val_ds = make_ds("val"); test_ds = make_ds("test")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_labels = train_ds.df["binding_label"].values.tolist()
    kw = {"num_workers": 4, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size*2, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size*2, shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = V3bModel(
        esm_dim=esm_dim, esm_proj_dim=args.esm_proj,
        rna_filters=[128,256,256], rna_kernels=[7,5,3],
        prot_filters=[128,256,256], prot_kernels=[11,7,5],
        head_dims=[256,64], dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  (RNA CNN 256 + Prot CNN 256 + ESM-2 proj {args.esm_proj})")

    pos_weight = torch.tensor(
        [sum(l==0 for l in train_labels) / max(sum(l==1 for l in train_labels),1)],
        dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n=== Training V3b — early stop on val AUPRC, patience={args.patience} ===")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'ValAUROC':>9}  {'ValAUPRC':>9}  {'Time':>6}  *")
    print(f"  {'─'*55}")

    best_auprc, best_auroc, best_epoch, no_improve = 0.0, 0.0, 0, 0
    history = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time(); model.train(); loss_sum = 0.0
        for rna, prot_oh, prot_emb, y in train_loader:
            rna, prot_oh, prot_emb, y = rna.to(device), prot_oh.to(device), prot_emb.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rna, prot_oh, prot_emb), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y)
        train_loss = loss_sum / len(train_ds)
        val_m = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        is_best = val_m["auprc"] > best_auprc
        history.append({"epoch": epoch, "train_loss": round(train_loss,4),
                         "val_auroc": round(val_m["auroc"],4),
                         "val_auprc": round(val_m["auprc"],4)})
        print(f"  {epoch:>5}  {train_loss:>8.4f}  {val_m['auroc']:>9.4f}  {val_m['auprc']:>9.4f}  {elapsed:>5.1f}s  {'★' if is_best else ''}")
        if is_best:
            best_auprc, best_auroc, best_epoch, no_improve = val_m["auprc"], val_m["auroc"], epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val_m, "args": vars(args)},
                       os.path.join(args.model_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}"); break

    print(f"\n  Best val AUPRC: {best_auprc:.4f}  AUROC: {best_auroc:.4f}  at epoch {best_epoch}")

    # ── Test ──────────────────────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(args.model_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device)
    print(f"\n=== Test set ===")
    print(f"  AUROC: {test_m['auroc']:.4f}   AUPRC: {test_m['auprc']:.4f}")
    print(f"  V2 was: AUROC=0.7028  AUPRC=0.5987")

    test_df = pd.read_csv(os.path.join(args.data_dir, "test.tsv"), sep="\t")
    test_df["prob"] = test_m["probs"]
    per_prot = []
    for prot, grp in test_df.groupby("protein_name"):
        if grp["binding_label"].nunique() < 2: continue
        per_prot.append({
            "protein": prot, "dataset": grp["dataset_source"].iloc[0],
            "auroc": float(roc_auc_score(grp["binding_label"], grp["prob"])),
            "auprc": float(average_precision_score(grp["binding_label"], grp["prob"])),
            "n": len(grp),
        })
    pp_aurocs = [p["auroc"] for p in per_prot]
    pp_auprcs = [p["auprc"] for p in per_prot]
    print(f"  Per-protein median AUROC: {np.median(pp_aurocs):.4f}  AUPRC: {np.median(pp_auprcs):.4f}")
    print(f"  V2 was: median AUROC=0.7176")

    delta_auroc = test_m["auroc"] - 0.7028
    delta_auprc = test_m["auprc"] - 0.5987
    print(f"\n  vs V2: AUROC {delta_auroc:+.4f}  AUPRC {delta_auprc:+.4f}")

    results = {
        "model": "generalized_v3b_esm2_plus_cnn",
        "architecture": "RNA CNN(256) + Prot CNN(256) + ESM-2 proj(128) → MLP",
        "best_val_auprc": best_auprc, "best_val_auroc": best_auroc,
        "test_metrics": {"auroc": test_m["auroc"], "auprc": test_m["auprc"]},
        "vs_v2": {"auroc": round(delta_auroc,4), "auprc": round(delta_auprc,4)},
        "per_protein_summary": {
            "median_auroc": float(np.median(pp_aurocs)),
            "median_auprc": float(np.median(pp_auprcs)),
            "min_auroc": float(np.min(pp_aurocs)),
        },
        "per_protein": per_prot, "history": history,
    }
    out = os.path.join(args.out_dir, "v3b_esm2_cnn_results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\n✅ V3b complete — {out}")
    if test_m["auroc"] > 0.7028:
        print("  ✅ Beats V2! → next: fine-tune ESM-2 last 2 layers (V3c)")
    else:
        print("  ⚠️  No improvement over V2 → try: larger ESM-2 proj or V3c residue CNN")

if __name__ == "__main__":
    main()
