#!/usr/bin/env python3
"""
64_parpi_cellline_models.py
---------------------------
Our models on the PaRPI data of one cell line (default HeLa, 17 RBP datasets), same random stratified 80/20 split per
dataset as the PaRPI code, but with a proper validation split (10% of train) for epoch selection:

  rna_cnn       per-dataset RNA-only CNN (one-hot, 4 channels)              - protein-free, DeepBind-like
  rna_shape_cnn per-dataset CNN with in vivo icSHAPE (value + missing flag) - protein-free, PrismNet-like
  pooled_v2     one V2 dual-branch CNN (RNA one-hot + protein one-hot) trained on all datasets of the cell line
  pooled_v2_x   same model evaluated with protein sequences permuted across datasets (protein-dependence control)
  pooled_cross  pooled V2 trained and evaluated with cross-RBP negatives (negatives = other RBPs' positives)

Reported: per-dataset test AUROC, medians, and the published PaRPI/HDRNet/PrismNet/DeepBind AUC for the same datasets.

Usage:  python scripts/64_parpi_cellline_models.py --cell Hela --epochs 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.cnn_model import ConvBranch, RNABindingCNN  # noqa: E402

DATA = ROOT / "data/external/parpi/src_data/dataset"
RNA = {c: i for i, c in enumerate("ACGU")}
AA = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
PROT_MAX, L = 700, 101


def onehot(seqs, table, length):
    out = np.zeros((len(seqs), length, len(table)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s.upper().replace("T", "U")[:length]):
            k = table.get(c)
            if k is not None:
                out[i, j, k] = 1.0
    return out


def shape_feat(strs):
    out = np.zeros((len(strs), L, 2), dtype=np.float32)
    for i, s in enumerate(strs):
        v = np.array([float(x) for x in s.split(",")])[:L]
        miss = v < 0
        out[i, : len(v), 0] = np.where(miss, 0.0, v)
        out[i, : len(v), 1] = miss.astype(np.float32)
    return out


def read_fasta(p):
    return "".join(l.strip() for l in open(p) if not l.startswith(">"))


class RnaCnn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = ConvBranch(ch, [64, 128, 128], [9, 5, 3], 0.2)
        self.h = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.b(x)).squeeze(-1)


def predict(model, fwd, tensors, device, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tensors[0]), bs):
            out.append(fwd(model, *[t[i:i + bs].to(device) for t in tensors]).float().cpu().numpy())
    return np.concatenate(out)


def fit(model, fwd, tr, y_tr, va, y_va, device, epochs, lr=1e-3, bs=256, patience=4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1, None, 0
    n = len(y_tr)
    yt = torch.from_numpy(y_tr.astype(np.float32))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(fwd(model, *[t[ix].to(device) for t in tr]), yt[ix].to(device))
            loss.backward()
            opt.step()
        auc = roc_auc_score(y_va, predict(model, fwd, va, device))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="Hela")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_datasets", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    files = sorted((DATA / "clip_data").glob(f"*_{args.cell}.tsv"))
    if args.max_datasets:
        files = files[: args.max_datasets]
    D = {}
    for f in files:
        d = pd.read_csv(f, sep="\t", usecols=["Seq", "icshape", "Protein", "label"])
        D[f.stem] = d
        print(f.stem, len(d), flush=True)
    prot_seq = {k: read_fasta(DATA / "protein" / f"{d.Protein.iloc[0]}.fasta") for k, d in D.items()}
    prot_oh = {k: torch.from_numpy(onehot([s], AA, PROT_MAX)) for k, s in prot_seq.items()}

    res, tests = {}, {}
    packs = {}
    for k, d in D.items():
        y = d.label.to_numpy()
        tr, te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=args.seed)
        tr, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=args.seed)
        x = torch.from_numpy(onehot(d.Seq.tolist(), RNA, L))
        sh = torch.from_numpy(shape_feat(d.icshape.tolist()))
        packs[k] = dict(x=x, xs=torch.cat([x, sh], dim=2), y=y, tr=tr, va=va, te=te)

    # per-dataset protein-free models
    for name, key, ch in (("rna_cnn", "x", 4), ("rna_shape_cnn", "xs", 6)):
        for k, p in packs.items():
            m = RnaCnn(ch).to(device)
            f = lambda mod, a: mod(a)  # noqa: E731
            fit(m, f, [p[key][p["tr"]]], p["y"][p["tr"]], [p[key][p["va"]]], p["y"][p["va"]], device, args.epochs)
            res.setdefault(k, {})[name] = float(roc_auc_score(p["y"][p["te"]], predict(m, f, [p[key][p["te"]]], device)))
        print(name, "median", round(float(np.median([r[name] for r in res.values()])), 4), flush=True)

    # pooled V2 with protein
    def stack(split, protein_map=None):
        X, P, Y, G = [], [], [], []
        for k, p in packs.items():
            ix = p[split]
            X.append(p["x"][ix])
            pk = (protein_map or {}).get(k, k)
            P.append(prot_oh[pk].expand(len(ix), -1, -1))
            Y.append(p["y"][ix])
            G += [k] * len(ix)
        return torch.cat(X), torch.cat(P), np.concatenate(Y), np.array(G)

    fv2 = lambda mod, a, b: mod(a, b)  # noqa: E731
    Xtr, Ptr, Ytr, _ = stack("tr")
    Xva, Pva, Yva, _ = stack("va")
    Xte, Pte, Yte, Gte = stack("te")
    m = RNABindingCNN().to(device)
    fit(m, fv2, [Xtr, Ptr], Ytr, [Xva, Pva], Yva, device, args.epochs)
    s = predict(m, fv2, [Xte, Pte], device)
    keys = list(packs)
    rng = np.random.default_rng(args.seed)
    perm = {k: keys[(i + 1 + int(rng.integers(0, len(keys) - 1))) % len(keys)] for i, k in enumerate(keys)}
    _, Pshuf, _, _ = stack("te", perm)
    s_x = predict(m, fv2, [Xte, Pshuf], device)
    for k in keys:
        idx = Gte == k
        res[k]["pooled_v2"] = float(roc_auc_score(Yte[idx], s[idx]))
        res[k]["pooled_v2_shuffled_protein"] = float(roc_auc_score(Yte[idx], s_x[idx]))
    print("pooled_v2 median", round(float(np.median([r["pooled_v2"] for r in res.values()])), 4),
          "shuffled-protein median", round(float(np.median([r["pooled_v2_shuffled_protein"] for r in res.values()])), 4), flush=True)

    # pooled with cross-RBP negatives: positives of the dataset vs positives of the other RBPs
    pos_x = {k: p["x"][p["y"] == 1] for k, p in packs.items()}
    cx_tr, cp_tr, cy_tr, cx_te, cp_te, cy_te, cg_te = [], [], [], [], [], [], []
    for k, p in packs.items():
        own = pos_x[k]
        others = torch.cat([pos_x[o] for o in packs if o != k])
        others = others[torch.randperm(len(others))[: len(own)]]
        n = len(own)
        cut = int(0.8 * n)
        for sl, (X, P, Y) in (("tr", (cx_tr, cp_tr, cy_tr)), ("te", (cx_te, cp_te, cy_te))):
            a, b = (slice(0, cut), slice(0, cut)) if sl == "tr" else (slice(cut, n), slice(cut, n))
            X += [own[a], others[b]]
            P += [prot_oh[k].expand(len(own[a]) + len(others[b]), -1, -1)]
            Y += [np.r_[np.ones(len(own[a])), np.zeros(len(others[b]))]]
            if sl == "te":
                cg_te += [k] * (len(own[a]) + len(others[b]))
    mc = RNABindingCNN().to(device)
    X_, P_, Y_ = torch.cat(cx_tr), torch.cat(cp_tr), np.concatenate(cy_tr)
    vi = np.random.default_rng(1).permutation(len(Y_))
    nv = int(0.1 * len(Y_))
    fit(mc, fv2, [X_[vi[nv:]], P_[vi[nv:]]], Y_[vi[nv:]], [X_[vi[:nv]], P_[vi[:nv]]], Y_[vi[:nv]], device, args.epochs)
    sc = predict(mc, fv2, [torch.cat(cx_te), torch.cat(cp_te)], device)
    yc, gc = np.concatenate(cy_te), np.array(cg_te)
    for k in keys:
        res[k]["pooled_cross_rbp"] = float(roc_auc_score(yc[gc == k], sc[gc == k]))
    print("pooled_cross_rbp median", round(float(np.median([r["pooled_cross_rbp"] for r in res.values()])), 4), flush=True)

    df = pd.DataFrame(res).T
    df.index.name = "dataset"
    pub = pd.read_excel(ROOT / "data/external/parpi/SupplementaryData1.xlsx", sheet_name="Figure 2(a-c)", header=1).rename(columns={"model": "dataset"})
    df = df.reset_index().merge(pub[["dataset", "PaRPI", "HDRNet", "Prismnet", "deepbind", "iDeep"]], on="dataset", how="left")
    out = ROOT / "results/parpi_audit"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"our_models_{args.cell}.tsv", sep="\t", index=False)
    summ = {c: round(float(df[c].median()), 4) for c in df.columns if c != "dataset"}
    (out / f"our_models_{args.cell}_summary.json").write_text(json.dumps(summ, indent=2))
    print("\nmedian AUROC over", len(df), "datasets in", args.cell)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
