#!/usr/bin/env python3
"""
61_motif_recovery_and_mutagenesis.py
------------------------------------
Mechanistic validation of trained V2/V4 models on held-out proteins.

1. Motif recovery. For every test protein the "true" motif is the 5-mer log-enrichment of its positives over its
   negatives (test split). The "predicted" motif is the mean model logit of each 5-mer inserted at a random position of
   random background RNAs (same backgrounds for all proteins). Reported: per-protein Spearman, top-20 overlap and
   whether the predicted top 5-mer is among the true top-20, versus
     - `prior`   : mean enrichment over all training proteins (protein-agnostic motif),
     - `nn`      : enrichment of the most similar training protein (MMseqs2),
   stratified by identity bin when `--identity_tsv` is given.
2. Protein in silico mutagenesis. For proteins with a Pfam RNA-binding domain (positions from
   `--domain_positions`), residues are replaced by alanine in several conditions and the drop of the model logit on the
   protein's positives is measured:
     domain_all        every residue of the domain
     domain_contact    F/Y/W/K/R inside the domain (RNA-contact-like residues)
     domain_random     the same number of random residues inside the domain
     outside_contact   the same number of F/Y/W/K/R outside the domain
     outside_segment   a random segment of equal length outside the domain
   A model that uses the domain should lose more for domain_all / domain_contact than for the outside controls.

Usage
-----
  python scripts/61_motif_recovery_and_mutagenesis.py --name v3b \
      --model "v2_s42,type=v2,path=models/saved/v2_v3b_seed42/best_model.pt,rna_max=60,prot_max=700" \
      --model "v4_s42,type=v4,path=models/saved/v4_v3b_seed42/best_model.pt,rna_max=60,prot_max=700" \
      --test_tsv data/generalized_v3b/test.tsv --train_tsv data/generalized_v3b/train.tsv \
      --identity_tsv results/homology_audit/v3b_test_protein_identity.tsv \
      --domain_positions results/homology_audit/v3b_test_domain_positions.tsv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
K = 5
KMERS = ["".join(p) for p in product("ACGU", repeat=K)]
KIDX = {k: i for i, k in enumerate(KMERS)}


def load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def kmer_counts(seqs, k=K) -> np.ndarray:
    c = np.zeros(4 ** k, dtype=np.float64)
    for s in seqs:
        s = s.upper().replace("T", "U")
        for i in range(len(s) - k + 1):
            j = KIDX.get(s[i:i + k])
            if j is not None:
                c[j] += 1
    return c


def enrichment(pos, neg) -> np.ndarray:
    cp, cn = kmer_counts(pos) + 1.0, kmer_counts(neg) + 1.0
    return np.log2(cp / cp.sum()) - np.log2(cn / cn.sum())


@torch.no_grad()
def logits(model, fwd, sf, rnas: list[str], prot: str, rna_max: int, prot_max: int, device, batch: int = 2048):
    p = torch.from_numpy(sf.one_hot([prot], prot_max, sf.AA_LUT, 20)).to(device)
    out = np.zeros(len(rnas), dtype=np.float32)
    for i in range(0, len(rnas), batch):
        r = torch.from_numpy(sf.one_hot(rnas[i:i + batch], rna_max, sf.RNA_LUT, 4)).to(device)
        out[i:i + batch] = fwd(model, r, p.expand(len(r), -1, -1)).float().cpu().numpy().reshape(-1)
    return out


def mmseqs_nn(q: dict[str, str], t: dict[str, str]) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text("".join(f">{k}\n{v}\n" for k, v in q.items()))
        (td / "t.fa").write_text("".join(f">{k}\n{v}\n" for k, v in t.items()))
        subprocess.run(["mmseqs", "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "h.m8"), str(td / "tmp"),
                        "-s", "7.5", "-e", "1e-3", "--format-output", "query,target,bits"], check=True, stdout=subprocess.DEVNULL)
        h = pd.read_csv(td / "h.m8", sep="\t", names=["q", "t", "bits"])
    return h.sort_values("bits", ascending=False).drop_duplicates("q").set_index("q").t.to_dict()


def parse_model(spec: str) -> dict:
    parts = spec.split(",")
    cfg = {"name": parts[0], "rna_max": 60, "prot_max": 700}
    for kv in parts[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = int(v) if k in ("rna_max", "prot_max") else v
    return cfg


def domain_positions_from_hmmscan(prots: dict[str, str]) -> pd.DataFrame:
    fam = load_mod("fam56", "56_domain_family_audit.py")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text("".join(f">{k}\n{''.join(c if c.isalpha() else 'X' for c in v.upper())}\n" for k, v in prots.items() if len(v) >= 10))
        subprocess.run(["hmmscan", "--cpu", "4", "-E", "1e-3", "--domE", "1e-3", "--domtblout", str(td / "d.tbl"),
                        str(resolve(fam.PFAM_HMM)), str(td / "q.fa")], check=True, stdout=subprocess.DEVNULL)
        rows = []
        for line in (td / "d.tbl").read_text().splitlines():
            if line.startswith("#"):
                continue
            f = line.split()
            name = f[0]
            famlabel = fam.pfam_family([name])
            if famlabel != "other":
                rows.append((f[3], name, famlabel, int(f[17]), int(f[18]), float(f[12])))
    return pd.DataFrame(rows, columns=["protein_name", "pfam", "family", "start", "end", "ievalue"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--train_tsv", default=None, help="enables the nearest-protein motif baseline")
    ap.add_argument("--identity_tsv", default=None)
    ap.add_argument("--domain_positions", default=None, help="TSV from this script (computed with hmmscan if absent)")
    ap.add_argument("--bg", choices=["real", "random"], default="real", help="insertion contexts: real RNAs or uniform random")
    ap.add_argument("--bg_len", type=int, default=40)
    ap.add_argument("--n_ctx", type=int, default=40)
    ap.add_argument("--min_rows", type=int, default=300, help="min test rows per protein")
    ap.add_argument("--n_mut_rows", type=int, default=200, help="positives per protein for mutagenesis")
    ap.add_argument("--max_proteins", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else
                          "mps" if torch.backends.mps.is_available() and not args.no_cuda else "cpu")
    sf = load_mod("suite51", "51_eclip_eval_suite.py")
    cols = ["protein_name", "protein_sequence", "rna_sequence", "binding_label"]
    test = pd.read_csv(resolve(args.test_tsv), sep="\t", usecols=cols, low_memory=False)
    prot_seq = test.drop_duplicates("protein_name").set_index("protein_name").protein_sequence.to_dict()
    cnt = test.groupby("protein_name").size()
    names = [n for n in cnt.index if cnt[n] >= args.min_rows and test[test.protein_name == n].binding_label.nunique() == 2]
    if args.max_proteins:
        names = names[: args.max_proteins]
    print(f"{len(names)} test proteins, device {device}")

    # ---------- true motifs and baselines
    true_enr = {}
    for n in names:
        g = test[test.protein_name == n]
        true_enr[n] = enrichment(g[g.binding_label == 1].rna_sequence, g[g.binding_label == 0].rna_sequence)
    prior = nn_enr = None
    if args.train_tsv:
        train = pd.read_csv(resolve(args.train_tsv), sep="\t", usecols=cols, low_memory=False)
        tprot = train.drop_duplicates("protein_name").set_index("protein_name").protein_sequence.to_dict()
        nn = mmseqs_nn({n: prot_seq[n] for n in names}, tprot)
        needed = set(nn.values())
        tr_enr = {}
        for t in needed:
            g = train[train.protein_name == t]
            if g.binding_label.nunique() == 2:
                tr_enr[t] = enrichment(g[g.binding_label == 1].rna_sequence, g[g.binding_label == 0].rna_sequence)
        samp = train.sample(min(len(train), 600_000), random_state=args.seed)
        prior = enrichment(samp[samp.binding_label == 1].rna_sequence, samp[samp.binding_label == 0].rna_sequence)
        nn_enr = {n: tr_enr[nn[n]] for n in names if n in nn and nn[n] in tr_enr}
        del train

    ident = pd.read_csv(resolve(args.identity_tsv), sep="\t").set_index("protein_name") if args.identity_tsv else None

    # ---------- predicted motifs
    if args.bg == "real":
        # in-distribution contexts: real RNAs of the data set (keeps library flanks / length distribution)
        pool = test.rna_sequence.drop_duplicates()
        pool = pool[(pool.str.len() >= 20) & (pool.str.len() <= 60)].str.upper().str.replace("T", "U")
        bg = pool.sample(args.n_ctx, random_state=args.seed).tolist()
    else:
        bg = ["".join(rng.choice(list("ACGU"), args.bg_len)) for _ in range(args.n_ctx)]
    probes, owners = [], []
    for i, km in enumerate(KMERS):
        for b in bg:
            pos = int(rng.integers(0, len(b) - K + 1))
            probes.append(b[:pos] + km + b[pos + K:])
            owners.append(i)
    owners = np.array(owners)

    def spearman(a, b):
        return float(spearmanr(a, b).statistic)

    def topk(a, b, k=20):
        return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k

    result = {"name": args.name, "n_proteins": len(names), "models": {}}
    rows_motif = []
    for spec in args.model:
        cfg = parse_model(spec)
        model, fwd = sf.load_model(cfg["type"], resolve(cfg["path"]), device)
        pred = {}
        for n in names:
            lg = logits(model, fwd, sf, probes, prot_seq[n], cfg["rna_max"], cfg["prot_max"], device)
            pred[n] = np.bincount(owners, weights=lg, minlength=len(KMERS)) / args.n_ctx
        for n in names:
            r = {"model": cfg["name"], "protein": n,
                 "spearman": spearman(pred[n], true_enr[n]), "top20": topk(pred[n], true_enr[n]),
                 "top1_in_true_top20": float(int(np.argmax(pred[n])) in set(np.argsort(-true_enr[n])[:20]))}
            if prior is not None:
                r["spearman_prior"] = spearman(prior, true_enr[n])
                r["top20_prior"] = topk(prior, true_enr[n])
            if nn_enr is not None and n in nn_enr:
                r["spearman_nn"] = spearman(nn_enr[n], true_enr[n])
                r["top20_nn"] = topk(nn_enr[n], true_enr[n])
            r["best_identity"] = float(ident.best_identity.get(n, np.nan)) if ident is not None else np.nan
            rows_motif.append(r)
        # protein-shuffle control: use another protein's sequence
        shuf = list(rng.permutation(names))
        sh_sp = []
        for n, m in zip(names, shuf):
            lg = logits(model, fwd, sf, probes, prot_seq[m], cfg["rna_max"], cfg["prot_max"], device)
            sh_sp.append(spearman(np.bincount(owners, weights=lg, minlength=len(KMERS)) / args.n_ctx, true_enr[n]))
        result["models"][cfg["name"]] = {"motif_spearman_shuffled_protein_mean": float(np.mean(sh_sp))}
        result["models"][cfg["name"]]["_cfg"] = cfg
        result["models"][cfg["name"]]["_loaded"] = True
        result.setdefault("_cache", {})[cfg["name"]] = (model, fwd, cfg)
    motif = pd.DataFrame(rows_motif)

    print("\n=== Motif recovery (per-protein 5-mer enrichment; mean over proteins) ===")
    agg_cols = [c for c in ("spearman", "spearman_prior", "spearman_nn", "top20", "top20_prior", "top20_nn", "top1_in_true_top20") if c in motif]
    summ = motif.groupby("model")[agg_cols].mean()
    print(summ.round(3).to_string())
    for m, v in result["models"].items():
        print(f"  {m}: shuffled-protein control Spearman = {v['motif_spearman_shuffled_protein_mean']:.3f}")
    if ident is not None:
        bins = pd.cut(motif.best_identity, [-1, 0.3, 0.5, 0.7, 0.9, 1.01], labels=["<30%", "30-50%", "50-70%", "70-90%", ">=90%"], right=False)
        by = motif.assign(bin=bins).groupby(["model", "bin"], observed=True)[["spearman"] + [c for c in ("spearman_nn", "spearman_prior") if c in motif]].agg(["mean", "count"])
        print("\nBy identity bin (Spearman):")
        print(by.round(3).to_string())
        by.columns = ["_".join(c) for c in by.columns]
        result["motif_by_bin"] = {f"{m}|{b}": v for (m, b), v in by.round(4).to_dict("index").items()}
    result["motif_summary"] = summ.round(4).to_dict("index")

    # ---------- protein mutagenesis
    if args.domain_positions and resolve(args.domain_positions).exists():
        dom = pd.read_csv(resolve(args.domain_positions), sep="\t")
    else:
        dom = domain_positions_from_hmmscan({n: prot_seq[n] for n in names})
        out_dom = ROOT / "results/mechanistic"
        out_dom.mkdir(parents=True, exist_ok=True)
        dom.to_csv(out_dom / f"{args.name}_domain_positions.tsv", sep="\t", index=False)
    dom = dom[dom.protein_name.isin(names)]
    rows_mut = []
    for cfg_name, (model, fwd, cfg) in result.pop("_cache").items():
        for n in dom.protein_name.unique():
            seq = prot_seq[n]
            d = dom[dom.protein_name == n]
            mask_dom = np.zeros(len(seq), dtype=bool)
            for s, e in zip(d.start, d.end):
                mask_dom[s - 1:e] = True
            if mask_dom.sum() < 20 or (~mask_dom).sum() < mask_dom.sum() or mask_dom[cfg["prot_max"]:].any():
                continue
            g = test[test.protein_name == n]
            pos = g[g.binding_label == 1].rna_sequence.drop_duplicates()
            neg = g[g.binding_label == 0].rna_sequence.drop_duplicates()
            pos = pos.sample(min(len(pos), args.n_mut_rows), random_state=args.seed).tolist()
            neg = neg.sample(min(len(neg), args.n_mut_rows), random_state=args.seed).tolist()
            base_p = logits(model, fwd, sf, pos, seq, cfg["rna_max"], cfg["prot_max"], device)
            base_n = logits(model, fwd, sf, neg, seq, cfg["rna_max"], cfg["prot_max"], device)
            arr = np.array(list(seq))
            contact = np.isin(arr, list("FYWKR"))
            n_c = int((mask_dom & contact).sum())
            if n_c < 3:
                continue
            idx_dom, idx_out = np.where(mask_dom)[0], np.where(~mask_dom)[0]
            idx_out_contact = np.where(~mask_dom & contact)[0]
            seg_start = int(rng.integers(0, len(seq) - mask_dom.sum() + 1))
            conds = {
                "domain_all": idx_dom,
                "domain_contact": np.where(mask_dom & contact)[0],
                "domain_random": rng.choice(idx_dom, n_c, replace=False),
                "outside_contact": rng.choice(idx_out_contact, min(n_c, len(idx_out_contact)), replace=False) if len(idx_out_contact) else np.array([], dtype=int),
                "outside_segment": idx_out[rng.integers(0, max(len(idx_out) - int(mask_dom.sum()), 1)):][: int(mask_dom.sum())],
            }
            for cname, ix in conds.items():
                if len(ix) == 0:
                    continue
                mut = arr.copy()
                mut[ix] = "A"
                ms = "".join(mut)
                mp = logits(model, fwd, sf, pos, ms, cfg["rna_max"], cfg["prot_max"], device)
                mn = logits(model, fwd, sf, neg, ms, cfg["rna_max"], cfg["prot_max"], device)
                from sklearn.metrics import roc_auc_score
                y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                rows_mut.append({"model": cfg_name, "protein": n, "family": d.family.iloc[0], "condition": cname, "n_mut": int(len(ix)),
                                 "drop_pos_logit": float(base_p.mean() - mp.mean()),
                                 "d_auroc": float(roc_auc_score(y, np.r_[mp, mn]) - roc_auc_score(y, np.r_[base_p, base_n]))})
    mut = pd.DataFrame(rows_mut)
    if len(mut):
        print("\n=== Protein alanine mutagenesis (mean drop of logit on positives / delta AUROC) ===")
        tab = mut.groupby(["model", "condition"]).agg(n_prot=("protein", "nunique"), drop=("drop_pos_logit", "mean"), d_auroc=("d_auroc", "mean"))
        print(tab.round(3).to_string())
        stat = {}
        for m, g in mut.groupby("model"):
            piv = g.pivot_table(index="protein", columns="condition", values="drop_pos_logit")
            for a, b in (("domain_contact", "outside_contact"), ("domain_contact", "domain_random"), ("domain_all", "outside_segment")):
                if a in piv and b in piv and piv[[a, b]].dropna().shape[0] >= 5:
                    x = piv[[a, b]].dropna()
                    p = wilcoxon(x[a], x[b]).pvalue if (x[a] != x[b]).any() else 1.0
                    stat[f"{m}: {a} > {b}"] = {"n": int(len(x)), "frac_larger": float((x[a] > x[b]).mean()), "wilcoxon_p": float(p)}
        for k, v in stat.items():
            print(f"  {k}: larger in {v['frac_larger']:.0%} of {v['n']} proteins (Wilcoxon p = {v['wilcoxon_p']:.3g})")
        result["mutagenesis_summary"] = tab.round(4).reset_index().to_dict("records")
        result["mutagenesis_tests"] = stat
    out = ROOT / "results/mechanistic"
    out.mkdir(parents=True, exist_ok=True)
    motif.to_csv(out / f"{args.name}_motif_per_protein.tsv", sep="\t", index=False)
    mut.to_csv(out / f"{args.name}_mutagenesis_per_protein.tsv", sep="\t", index=False)
    result["models"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in result["models"].items()}
    (out / f"{args.name}_summary.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {out}/{args.name}_summary.json")


if __name__ == "__main__":
    main()
