#!/usr/bin/env python3
"""
71_train_hybrid.py
------------------
Hybrid "nearest protein + residual learner" on v3b.

    logit(A, r) = a * z_nn(A, r) + c * ident(A) * z_nn(A, r) + head(rna_emb(r), prot_emb(A))

z_nn(A, r): k-mer-LR logit of the RNA r under the per-protein RNA model of A's nearest training neighbour (script 70; for training
proteins the neighbour is taken outside their own cluster so it is as remote as for held-out proteins; no neighbour -> mean model).
head: bilinear MLP on frozen ESM-2 / RNA-FM embeddings (scripts 66, 67). Trained with composition-neutral cross-protein negatives
(positive (A, r) vs (B, r) of the same source) and early-stopped on a cross-protein probe from val.

--mode nn_only skips training and scores z_nn alone (sanity check against scripts/59: probe nn1 ~0.627).

Usage
-----
  python scripts/71_train_hybrid.py --mode nn_only --out_dir results/v3b_hybrid/nn_only
  python scripts/71_train_hybrid.py --mode hybrid --seed 42 --out_dir results/v3b_hybrid/hybrid_s42
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class NN:
    """z_nn lookup: RNA k-mer features table x per-protein LR weights of the neighbour."""

    def __init__(self, tab, k59, nn_dir: Path, device):
        z = np.load(nn_dir / "nn_models.npz", allow_pickle=False)
        names = list(z["names"])
        self.W = torch.from_numpy(np.vstack([z["W"], z["W_mean"][None]])).to(device)  # last row = mean model
        self.B = torch.from_numpy(np.r_[z["B"], z["B_mean"]]).float().to(device)
        nb = pd.read_csv(nn_dir / "neighbours.tsv", sep="\t", keep_default_na=False)
        idx = {n: i for i, n in enumerate(names)}
        self.nn_idx = {r.protein_name: (idx[r.nn_name] if r.nn_name in idx else len(names)) for r in nb.itertuples()}
        self.ident = {r.protein_name: float(r.nn_ident) for r in nb.itertuples()}
        self.tab, self.device = tab, device
        seqs = list(tab.rna_idx.keys())
        feats = np.zeros((len(seqs), 340), dtype=np.float16)
        for i in range(0, len(seqs), 20000):
            feats[i:i + 20000] = k59.features(seqs[i:i + 20000])[0].astype(np.float16)
            if (i // 20000) % 10 == 0:
                print(f"  k-mer features {i}/{len(seqs)}", flush=True)
        self.X = torch.from_numpy(feats).to(device)

    def z(self, rna_seqs, prot_names):
        ri = torch.tensor([self.tab.rna_idx[s] for s in rna_seqs], device=self.device)
        ni = torch.tensor([self.nn_idx[p] for p in prot_names], device=self.device)
        z = (self.X[ri].float() * self.W[ni].float()).sum(1) + self.B[ni]
        ident = torch.tensor([self.ident[p] for p in prot_names], device=self.device)
        return z, ident


class Hybrid(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head, self.a, self.c = head, nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.0))

    def forward(self, r, p, z, ident, use_head=True):
        out = (self.a + self.c * ident) * z
        return out + self.head(r, p) if use_head else out


@torch.no_grad()
def score(model, tab, nn_, df, bs=8192, mode="hybrid"):
    model.eval()
    out = []
    for i in range(0, len(df), bs):
        sub = df.iloc[i:i + bs]
        r, p = tab.get(sub.rna_sequence.values, sub.protein_name.values)
        z, ident = nn_.z(sub.rna_sequence.values, sub.protein_name.values)
        out.append(torch.sigmoid(model(r, p, z, ident, use_head=(mode != "nn_only"))).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/generalized_v3b")
    ap.add_argument("--emb_dir", default="data/embeddings_v3b")
    ap.add_argument("--nn_dir", default="results/hybrid")
    ap.add_argument("--probe_tsv", default="data/benchmarks/v3b_cross_neg/cross_neg_probe/test.tsv")
    ap.add_argument("--identity_tsv", default="results/homology_audit/v3b_test_protein_identity.tsv")
    ap.add_argument("--mode", choices=["hybrid", "nn_only"], default="hybrid")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_pos", type=int, default=20000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/v3b_hybrid/run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m67, k59, t65 = (load_mod("frozen67", "67_frozen_embedding_baselines.py"), load_mod("knn59", "59_knn_protein_baseline.py"),
                     load_mod("train65", "65_train_v4_crossneg.py"))
    data = resolve(args.data_dir)
    cols = ["protein_name", "rna_sequence", "protein_sequence", "binding_label", "dataset_source"]
    train = pd.read_csv(data / "train.tsv", sep="\t", usecols=cols, low_memory=False)
    val = pd.read_csv(data / "val.tsv", sep="\t", usecols=cols, low_memory=False)
    test = pd.read_csv(data / "test.tsv", sep="\t", usecols=cols, low_memory=False)
    probe = pd.read_csv(resolve(args.probe_tsv), sep="\t", usecols=cols, low_memory=False)
    ident_tab = pd.read_csv(resolve(args.identity_tsv), sep="\t")
    tab = m67.Tables(resolve(args.emb_dir), device)
    nn_ = NN(tab, k59, resolve(args.nn_dir), device)
    model = Hybrid(m67.Head("bilinear")).to(device)

    best, best_ep, bad, hist, best_state = -1.0, 0, 0, [], None
    if args.mode == "hybrid":
        val_probe = t65.build_val_probe(val, args.val_pos, args.seed)
        pos, pos_set, by_src = t65.build_pools(train)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
        half = args.batch // 2
        for ep in range(1, args.epochs + 1):
            model.train()
            sample = pos.sample(args.steps_per_epoch * half, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
            keep, partners = t65.draw_negatives(sample, pos_set, by_src, rng)
            sample, partners = sample.loc[keep].reset_index(drop=True), np.array(partners)
            tot, n_steps = 0.0, len(sample) // half
            for s in range(n_steps):
                sub, part = sample.iloc[s * half:(s + 1) * half], partners[s * half:(s + 1) * half]
                r, pa = tab.get(sub.rna_sequence.values, sub.protein_name.values)
                _, pb = tab.get(sub.rna_sequence.values, part)
                za, ia = nn_.z(sub.rna_sequence.values, sub.protein_name.values)
                zb, ib = nn_.z(sub.rna_sequence.values, part)
                logit = model(torch.cat([r, r]), torch.cat([pa, pb]), torch.cat([za, zb]), torch.cat([ia, ib]))
                y = torch.cat([torch.ones(len(sub)), torch.zeros(len(sub))]).to(device)
                loss = F.binary_cross_entropy_with_logits(logit, y) + F.softplus(-(logit[:len(sub)] - logit[len(sub):])).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            sched.step()
            v = float(roc_auc_score(val_probe.binding_label, score(model, tab, nn_, val_probe)))
            hist.append({"epoch": ep, "loss": tot / n_steps, "val_probe_auroc": v, "a": float(model.a), "c": float(model.c)})
            star = ""
            if v > best:
                best, best_ep, bad, star = v, ep, 0, " *"
                best_state = {k: x.clone() for k, x in model.state_dict().items()}
            else:
                bad += 1
            print(f"{ep:>3} loss {tot / n_steps:.4f} val-probe AUROC {v:.4f} a={float(model.a):.2f} c={float(model.c):.2f}{star}", flush=True)
            if bad >= args.patience:
                break
        model.load_state_dict(best_state)

    res = {"args": vars(args), "best_epoch": best_ep, "best_val_probe_auroc": best, "history": hist}
    for name, df in [("test_standard", test), ("probe", probe)]:
        prob = score(model, tab, nn_, df, mode=args.mode)
        df[["protein_name", "binding_label"]].assign(prob=prob).to_csv(out / f"scores_{name}.tsv.gz", sep="\t", index=False)
        res[name] = m67.metrics(df, prob, ident_tab)
        print(f"{name}: AUROC {res[name]['auroc']:.4f} pp-median {res[name]['pp_median']:.4f}")
        print("  by bin:", {k: round(v, 3) for k, v in res[name]["by_bin"].items()})
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
