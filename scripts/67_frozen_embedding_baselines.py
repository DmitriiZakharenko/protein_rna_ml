#!/usr/bin/env python3
"""
67_frozen_embedding_baselines.py
--------------------------------
Frozen ESM-2 (protein) + RNA-FM (RNA) mean-pooled embeddings (scripts/66) with a small trainable head, trained on v3b and
evaluated on the v3b standard test and the v3b cross-protein probe, stratified by MMseqs2 identity bin.

Heads (--head):
  rna       RNA embedding only        (composition/RNA shortcut control)
  protein   protein embedding only    (protein prior control)
  concat    [rna, protein] -> MLP
  bilinear  [rna', protein', rna' * protein'] -> MLP, with rna' / protein' linear projections
Training (--train_mode):
  std       ordinary labelled rows of train.tsv
  xneg      each positive paired with the same RNA and another training protein of the same source (script 65 sampling)
--shuffle_protein_eval additionally scores the test sets with proteins permuted among test proteins (control).

Usage
-----
  python scripts/67_frozen_embedding_baselines.py --head bilinear --train_mode xneg --seed 42 \\
      --out_dir results/v3b_frozen/bilinear_xneg_s42
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_script65():
    spec = importlib.util.spec_from_file_location("train65", ROOT / "scripts" / "65_train_v4_crossneg.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Head(nn.Module):
    def __init__(self, kind: str, d_rna=640, d_prot=1280, h=256, dropout=0.3):
        super().__init__()
        self.kind = kind
        self.rna_norm, self.prot_norm = nn.LayerNorm(d_rna), nn.LayerNorm(d_prot)
        self.pr, self.pp = nn.Linear(d_rna, h), nn.Linear(d_prot, h)
        d_in = {"rna": h, "protein": h, "concat": 2 * h, "bilinear": 3 * h}[kind]
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, r, p):
        r, p = F.gelu(self.pr(self.rna_norm(r))), F.gelu(self.pp(self.prot_norm(p)))
        x = {"rna": r, "protein": p, "concat": torch.cat([r, p], 1), "bilinear": torch.cat([r, p, r * p], 1)}[self.kind]
        return self.mlp(x).squeeze(-1)


class Tables:
    def __init__(self, emb_dir: Path, device):
        z = np.load(emb_dir / "rna_rnafm.npz", allow_pickle=False)
        self.rna_idx = {s: i for i, s in enumerate(z["seqs"])}
        self.rna = torch.from_numpy(z["emb"]).to(device)
        z = np.load(emb_dir / "prot_esm2.npz", allow_pickle=False)
        self.prot_idx = {s: i for i, s in enumerate(z["names"])}
        self.prot = torch.from_numpy(z["emb"]).to(device)
        self.device = device

    def get(self, rna_seqs, prot_names):
        ri = torch.tensor([self.rna_idx[s] for s in rna_seqs], device=self.device)
        pi = torch.tensor([self.prot_idx[s] for s in prot_names], device=self.device)
        return self.rna[ri].float(), self.prot[pi].float()


@torch.no_grad()
def score(model, tab: Tables, df: pd.DataFrame, bs=8192, shuffle_protein=False, seed=0):
    model.eval()
    names = df.protein_name.values
    if shuffle_protein:
        rng = np.random.default_rng(seed)
        uniq = np.unique(names)
        perm = dict(zip(uniq, rng.permutation(uniq)))
        names = np.array([perm[n] for n in names])
    out = []
    for i in range(0, len(df), bs):
        r, p = tab.get(df.rna_sequence.values[i:i + bs], names[i:i + bs])
        out.append(torch.sigmoid(model(r, p)).cpu().numpy())
    return np.concatenate(out)


def metrics(df: pd.DataFrame, prob: np.ndarray, ident: pd.DataFrame) -> dict:
    y = df.binding_label.values
    res = {"auroc": float(roc_auc_score(y, prob)), "auprc": float(average_precision_score(y, prob))}
    d = df.assign(prob=prob).merge(ident[["protein_name", "identity_bin"]], on="protein_name", how="left")
    pp = [roc_auc_score(g.binding_label, g.prob) for _, g in d.groupby("protein_name") if g.binding_label.nunique() == 2]
    res["pp_median"] = float(np.median(pp)) if pp else None
    res["by_bin"] = {b: float(roc_auc_score(g.binding_label, g.prob)) for b, g in d.groupby("identity_bin")
                     if g.binding_label.nunique() == 2 and len(g) >= 200}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/generalized_v3b")
    ap.add_argument("--emb_dir", default="data/embeddings_v3b")
    ap.add_argument("--probe_tsv", default="data/benchmarks/v3b_cross_neg/cross_neg_probe/test.tsv")
    ap.add_argument("--identity_tsv", default="results/homology_audit/v3b_test_protein_identity.tsv")
    ap.add_argument("--head", choices=["rna", "protein", "concat", "bilinear"], default="bilinear")
    ap.add_argument("--train_mode", choices=["std", "xneg"], default="std")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_pos", type=int, default=20000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--split_map", default="results/v3b/protein_split_map.tsv")
    ap.add_argument("--cluster_frac", type=float, default=1.0, help="train on a random fraction of the training CLUSTERS (learning curve)")
    ap.add_argument("--seen_probe_frac", type=float, default=0.0,
                    help="positive control: hold out this fraction of the training RNAs (all rows with them), evaluate a cross-protein probe "
                         "on them for the SEEN training proteins")
    ap.add_argument("--out_dir", default="results/v3b_frozen/run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t65 = load_script65()
    data = resolve(args.data_dir)
    cols = ["protein_name", "rna_sequence", "protein_sequence", "binding_label", "dataset_source"]
    train = pd.read_csv(data / "train.tsv", sep="\t", usecols=cols, low_memory=False)
    val = pd.read_csv(data / "val.tsv", sep="\t", usecols=cols, low_memory=False)
    test = pd.read_csv(data / "test.tsv", sep="\t", usecols=cols, low_memory=False)
    probe = pd.read_csv(resolve(args.probe_tsv), sep="\t", usecols=cols, low_memory=False)
    ident = pd.read_csv(resolve(args.identity_tsv), sep="\t")
    if args.cluster_frac < 1.0:
        cl = pd.read_csv(resolve(args.split_map), sep="\t").set_index("protein_name").cluster_id.to_dict()
        ids = sorted({cl[p] for p in train.protein_name.unique() if p in cl})
        keep = set(rng.choice(ids, max(1, int(round(args.cluster_frac * len(ids)))), replace=False).tolist())
        train = train[train.protein_name.map(lambda p: cl.get(p) in keep)].reset_index(drop=True)
        print(f"cluster_frac {args.cluster_frac}: {len(keep)}/{len(ids)} clusters, {train.protein_name.nunique()} proteins, {len(train):,} rows")
    seen_probe = None
    if args.seen_probe_frac > 0:
        uniq = train.rna_sequence.unique()
        held = set(rng.choice(uniq, int(args.seen_probe_frac * len(uniq)), replace=False).tolist())
        mask = train.rna_sequence.isin(held)
        seen_probe = t65.build_val_probe(train[mask].reset_index(drop=True), 20000, args.seed)
        train = train[~mask].reset_index(drop=True)
        print(f"seen-protein probe: {len(seen_probe):,} rows from {len(held):,} held-out RNAs; train rows left {len(train):,}")
    tab = Tables(resolve(args.emb_dir), device)
    val_probe = t65.build_val_probe(val, args.val_pos, args.seed)
    pos, pos_set, by_src = t65.build_pools(train)
    model = Head(args.head, h=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    print(f"device={device} head={args.head} mode={args.train_mode} train rows={len(train):,} pos={len(pos):,}")

    best, best_ep, bad, hist = -1.0, 0, 0, []
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        if args.train_mode == "xneg":
            sample = pos.sample(args.steps_per_epoch * args.batch // 2, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
            keep, partners = t65.draw_negatives(sample, pos_set, by_src, rng)
            sample, partners = sample.loc[keep].reset_index(drop=True), np.array(partners)
            half = args.batch // 2
            n_steps = len(sample) // half
        else:
            sample = train.sample(args.steps_per_epoch * args.batch, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
            n_steps = args.steps_per_epoch
        for s in range(n_steps):
            if args.train_mode == "xneg":
                sub, part = sample.iloc[s * half:(s + 1) * half], partners[s * half:(s + 1) * half]
                r, pa = tab.get(sub.rna_sequence.values, sub.protein_name.values)
                _, pb = tab.get(sub.rna_sequence.values, part)
                r, p = torch.cat([r, r]), torch.cat([pa, pb])
                y = torch.cat([torch.ones(len(sub)), torch.zeros(len(sub))]).to(device)
                logit = model(r, p)
                loss = F.binary_cross_entropy_with_logits(logit, y) + F.softplus(-(logit[:len(sub)] - logit[len(sub):])).mean()
            else:
                sub = sample.iloc[s * args.batch:(s + 1) * args.batch]
                r, p = tab.get(sub.rna_sequence.values, sub.protein_name.values)
                y = torch.tensor(sub.binding_label.values, dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(model(r, p), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        v = float(roc_auc_score(val_probe.binding_label, score(model, tab, val_probe)))
        hist.append({"epoch": ep, "loss": tot / n_steps, "val_probe_auroc": v})
        star = ""
        if v > best:
            best, best_ep, bad, star = v, ep, 0, " *"
            best_state = {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
        print(f"{ep:>3} loss {tot / n_steps:.4f} val-probe AUROC {v:.4f}{star}", flush=True)
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    res = {"args": vars(args), "best_epoch": best_ep, "best_val_probe_auroc": best, "history": hist}
    if seen_probe is not None:
        res["seen_probe_auroc"] = float(roc_auc_score(seen_probe.binding_label, score(model, tab, seen_probe)))
        print(f"seen-protein probe (new RNAs, training proteins): AUROC {res['seen_probe_auroc']:.4f}")
    for name, df in [("test_standard", test), ("probe", probe)]:
        prob = score(model, tab, df)
        df[["protein_name", "binding_label"]].assign(prob=prob).to_csv(out / f"scores_{name}.tsv.gz", sep="\t", index=False)
        res[name] = metrics(df, prob, ident)
        res[name + "_shuffled_protein"] = metrics(df, score(model, tab, df, shuffle_protein=True, seed=args.seed), ident)
        print(f"{name}: AUROC {res[name]['auroc']:.4f} pp-median {res[name]['pp_median']:.4f} "
              f"| shuffled protein {res[name + '_shuffled_protein']['auroc']:.4f}")
        print("  by bin:", {k: round(v, 3) for k, v in res[name]["by_bin"].items()})
    (out / "results.json").write_text(json.dumps(res, indent=2))
    torch.save({"state": best_state, "args": vars(args)}, out / "head.pt")
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
