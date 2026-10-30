#!/usr/bin/env python3
"""
65_train_v4_crossneg.py
-----------------------
V4 (concat_bi) trained with composition-neutral cross-protein negatives.

Every training positive (protein A, RNA r) is paired in the same step with the same RNA r
and a different training protein B of the same ``dataset_source`` for which r is not a known
positive. The RNA marginals of positives and negatives are therefore identical, so the only
way to lower the loss is to learn the protein x RNA interaction. Loss:

    BCE(s(A,r), 1) + BCE(s(B,r), 0) + lambda_rank * softplus(-(s(A,r) - s(B,r)))

Optionally a fraction of ordinary rows from train.tsv is mixed in (--std_frac).
Early stopping uses AUROC on a fixed cross-protein probe built from val.tsv (same
construction as scripts/53), not the standard val set, which rewards the RNA shortcut.

The checkpoint format matches scripts/21, so scripts/21b (loader), 51 (probe eval) and
61 (motif/mutagenesis) work unchanged.

Usage
-----
  python scripts/65_train_v4_crossneg.py --data_dir data/generalized_v3b \\
      --model_dir models/saved/v4x_v3b_seed42 --out_dir results/v3b_crossneg/v4x_seed42 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import _AA_LUT, _RNA_LUT, one_hot_encode  # noqa: E402
from src.models.interaction_model import RNABindingV4  # noqa: E402


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class ProteinCache:
    def __init__(self, df: pd.DataFrame, prot_max: int):
        seqs = df.drop_duplicates("protein_name").set_index("protein_name")["protein_sequence"]
        self.t = {k: one_hot_encode(v, prot_max, _AA_LUT, 20) for k, v in seqs.items()}

    def get(self, names):
        return torch.stack([self.t[n] for n in names])


def enc_rna(seqs, rna_max):
    return torch.stack([one_hot_encode(s, rna_max, _RNA_LUT, 4) for s in seqs])


def build_pools(df: pd.DataFrame):
    pos = df[df.binding_label == 1].drop_duplicates(["protein_name", "rna_sequence"]).reset_index(drop=True)
    pos_set = set(zip(pos.protein_name, pos.rna_sequence))
    by_src = {s: np.array(sorted(g.protein_name.unique())) for s, g in df.groupby("dataset_source")}
    return pos, pos_set, by_src


def draw_negatives(pos: pd.DataFrame, pos_set, by_src, rng, tries: int = 20):
    """Return (index into pos, partner protein) pairs for positives that got a valid cross-protein negative."""
    keep, partners = [], []
    for i, (p, r, s) in enumerate(zip(pos.protein_name, pos.rna_sequence, pos.dataset_source)):
        cands = by_src[s]
        if len(cands) < 2:
            continue
        for _ in range(tries):
            b = cands[int(rng.integers(len(cands)))]
            if b != p and (b, r) not in pos_set:
                keep.append(i)
                partners.append(b)
                break
    return np.array(keep), partners


def build_val_probe(val: pd.DataFrame, max_pos: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos, pos_set, by_src = build_pools(val)
    if len(pos) > max_pos:
        pos = pos.sample(max_pos, random_state=seed).reset_index(drop=True)
    keep, partners = draw_negatives(pos, pos_set, by_src, rng)
    p = pos.loc[keep].reset_index(drop=True)
    seq = val.drop_duplicates("protein_name").set_index("protein_name")["protein_sequence"]
    n = p.copy()
    n["protein_name"] = partners
    n["protein_sequence"] = [seq[b] for b in partners]
    n["binding_label"] = 0
    return pd.concat([p, n], ignore_index=True)


@torch.no_grad()
def score(model, df, pc, rna_max, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(df), bs):
        sub = df.iloc[i:i + bs]
        r = enc_rna(sub.rna_sequence, rna_max).to(device)
        p = pc.get(sub.protein_name).to(device)
        out.append(torch.sigmoid(model(r, p, None)).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/generalized_v3b")
    ap.add_argument("--out_dir", default="results/v3b_crossneg/v4x_seed42")
    ap.add_argument("--model_dir", default="models/saved/v4x_v3b_seed42")
    ap.add_argument("--rna_max", type=int, default=60)
    ap.add_argument("--prot_max", type=int, default=700)
    ap.add_argument("--inter_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pos_per_epoch", type=int, default=300000, help="positives sampled per epoch")
    ap.add_argument("--batch_pos", type=int, default=128, help="positives per step (each gives 2 rows)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--std_frac", type=float, default=0.0,
                    help="ordinary train rows mixed in per step, as a fraction of batch_pos")
    ap.add_argument("--val_pos", type=int, default=20000)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    out_dir, model_dir, data_dir = resolve(args.out_dir), resolve(args.model_dir), resolve(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cols = ["protein_name", "rna_sequence", "protein_sequence", "binding_label", "dataset_source"]
    train = pd.read_csv(data_dir / "train.tsv", sep="\t", usecols=cols, low_memory=False)
    val = pd.read_csv(data_dir / "val.tsv", sep="\t", usecols=cols, low_memory=False)
    pos, pos_set, by_src = build_pools(train)
    std = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True) if args.std_frac > 0 else None
    pc = ProteinCache(pd.concat([train, val]), args.prot_max)
    val_probe = build_val_probe(val, args.val_pos, args.seed)
    print(f"device={device} train positives={len(pos):,} proteins={train.protein_name.nunique()} "
          f"val probe rows={len(val_probe):,}")

    model = RNABindingV4(rna_filters=[128, 256, 256], prot_filters=[128, 256, 256], rna_kernels=[7, 5, 3],
                         prot_kernels=[11, 7, 5], interaction="concat_bi", inter_dim=args.inter_dim,
                         head_dims=[512, 128, 32], dropout=args.dropout, use_source_emb=False,
                         source_emb_dim=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    best, best_ep, bad, history, std_ptr = -1.0, 0, 0, [], 0
    print(f"{'ep':>3} {'loss':>7} {'valProbeAUROC':>14} {'AUPRC':>7} {'time':>7}")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        sample = pos.sample(min(args.pos_per_epoch, len(pos)), random_state=int(rng.integers(1 << 30)))
        sample = sample.reset_index(drop=True)
        keep, partners = draw_negatives(sample, pos_set, by_src, rng)
        sample = sample.loc[keep].reset_index(drop=True)
        partners = np.array(partners)
        tot, nb = 0.0, 0
        for i in range(0, len(sample), args.batch_pos):
            sub, part = sample.iloc[i:i + args.batch_pos], partners[i:i + args.batch_pos]
            rna = enc_rna(sub.rna_sequence, args.rna_max)
            pa, pb = pc.get(sub.protein_name), pc.get(part)
            r = torch.cat([rna, rna]).to(device)
            p = torch.cat([pa, pb]).to(device)
            y = torch.cat([torch.ones(len(sub)), torch.zeros(len(sub))]).to(device)
            if std is not None:
                k = max(1, int(args.std_frac * len(sub)))
                s = std.iloc[std_ptr:std_ptr + k]
                std_ptr = (std_ptr + k) % max(1, len(std) - k)
                r = torch.cat([r, enc_rna(s.rna_sequence, args.rna_max).to(device)])
                p = torch.cat([p, pc.get(s.protein_name).to(device)])
                y = torch.cat([y, torch.tensor(s.binding_label.values, dtype=torch.float32, device=device)])
            logit = model(r, p, None)
            n = len(sub)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if args.lambda_rank > 0:
                loss = loss + args.lambda_rank * F.softplus(-(logit[:n] - logit[n:2 * n])).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        prob = score(model, val_probe, pc, args.rna_max, device)
        auroc = float(roc_auc_score(val_probe.binding_label, prob))
        auprc = float(average_precision_score(val_probe.binding_label, prob))
        history.append({"epoch": ep, "loss": tot / nb, "val_probe_auroc": auroc, "val_probe_auprc": auprc})
        star = ""
        if auroc > best:
            best, best_ep, bad, star = auroc, ep, 0, " *"
            torch.save({"epoch": ep, "model_state": model.state_dict(), "val_metrics": {"auroc": auroc, "auprc": auprc},
                        "args": vars(args),
                        "arch": {"interaction": "concat_bi", "inter_dim": args.inter_dim, "use_source_emb": False,
                                 "rna_max": args.rna_max, "prot_max": args.prot_max}},
                       model_dir / "best_model.pt")
        else:
            bad += 1
        print(f"{ep:>3} {tot / nb:>7.4f} {auroc:>14.4f} {auprc:>7.4f} {time.time() - t0:>6.0f}s{star}", flush=True)
        if bad >= args.patience:
            print(f"early stop at epoch {ep}")
            break

    res = {"best_val_probe_auroc": best, "best_epoch": best_ep, "args": vars(args), "history": history}
    (out_dir / "v4x_results.json").write_text(json.dumps(res, indent=2))
    print(f"best val probe AUROC {best:.4f} at epoch {best_ep}; model -> {model_dir}/best_model.pt")


if __name__ == "__main__":
    main()
