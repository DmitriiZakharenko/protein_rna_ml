#!/usr/bin/env python3
"""
55_mmseqs_homology_audit.py
---------------------------
Sequence-level novelty audit of test proteins against the v3a training proteins
(MMseqs2). For every test protein it reports the best local-alignment identity to
any training protein; optionally it stratifies a scores table (scripts/51 output)
by that identity bin.

Usage
-----
  python scripts/55_mmseqs_homology_audit.py --name v3a_test \\
      --test_tsv data/generalized_v3a/test.tsv
  python scripts/55_mmseqs_homology_audit.py --name eclip_pnr \\
      --test_tsv data/benchmarks/skipper_eclip/protein_and_rna/test.tsv
  # stratify probe scores (VM) by identity bin
  python scripts/55_mmseqs_homology_audit.py --name v3a_test \\
      --test_tsv data/generalized_v3a/test.tsv \\
      --scores results/overnight/probe_final/scores_cross_neg_probe.tsv.gz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BINS = [(-1, 0.0, "no_hit"), (0.0, 0.30, "<30%"), (0.30, 0.50, "30-50%"),
        (0.50, 0.70, "50-70%"), (0.70, 0.90, "70-90%"), (0.90, 1.01, ">=90%")]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def unique_proteins(tsv: Path) -> dict[str, str]:
    df = pd.read_csv(tsv, sep="\t", usecols=["protein_name", "protein_sequence"])
    df = df.drop_duplicates("protein_name")
    return dict(zip(df.protein_name, df.protein_sequence))


def write_fasta(prots: dict[str, str], path: Path) -> None:
    with open(path, "w") as f:
        for k, v in prots.items():
            f.write(f">{k}\n{v}\n")


def bin_of(x: float) -> str:
    if x <= 0:
        return "no_hit"
    for lo, hi, name in BINS[1:]:
        if lo <= x < hi:
            return name
    return ">=90%"


def auroc(y, s) -> float | None:
    from sklearn.metrics import roc_auc_score
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def run_mmseqs(args, out: Path):
    train = unique_proteins(resolve(args.train_tsv))
    test = unique_proteins(resolve(args.test_tsv))
    print(f"train proteins {len(train)}, test proteins {len(test)}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_fasta(train, td / "train.fa")
        write_fasta(test, td / "test.fa")
        subprocess.run(
            ["mmseqs", "easy-search", str(td / "test.fa"), str(td / "train.fa"), str(td / "hits.m8"), str(td / "tmp"),
             "-s", args.sensitivity, "-e", "1e-3", "--max-seqs", "1000", "--threads", str(args.threads),
             "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits"],
            check=True, stdout=subprocess.DEVNULL)
        cols = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]
        hits = pd.read_csv(td / "hits.m8", sep="\t", names=cols)

    if args.min_cov > 0:
        hits = hits[hits.qcov >= args.min_cov]
    best = hits.sort_values(["query", "fident", "bits"], ascending=[True, False, False]).drop_duplicates("query")
    best = best.set_index("query")
    rows = []
    for name in test:
        if name in best.index:
            r = best.loc[name]
            rows.append((name, float(r.fident), r.target, float(r.qcov), float(r.evalue)))
        else:
            rows.append((name, 0.0, "", 0.0, np.nan))
    tab = pd.DataFrame(rows, columns=["protein_name", "best_identity", "best_train_hit", "qcov", "evalue"])
    return tab, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--train_tsv", default="data/generalized_v3a/train.tsv")
    ap.add_argument("--scores", default=None, help="scripts/51 scores_*.tsv.gz to stratify by identity bin")
    ap.add_argument("--out_dir", default="results/homology_audit")
    ap.add_argument("--sensitivity", default="7.5")
    ap.add_argument("--min_cov", type=float, default=0.0, help="query coverage filter (0 = none)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--identity_tsv", default=None,
                    help="reuse a previous <name>_protein_identity.tsv instead of running MMseqs2")
    args = ap.parse_args()

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.identity_tsv:
        tab = pd.read_csv(resolve(args.identity_tsv), sep="\t")
        train = {}
        print(f"reusing identities for {len(tab)} test proteins")
    else:
        tab, train = run_mmseqs(args, out)
    tab["identity_bin"] = tab.best_identity.map(bin_of)
    if not args.identity_tsv:
        tab.to_csv(out / f"{args.name}_protein_identity.tsv", sep="\t", index=False)

    order = [b[2] for b in BINS]
    counts = tab.identity_bin.value_counts().reindex(order, fill_value=0)
    summary = {"name": args.name, "n_test_proteins": len(tab), "n_train_proteins": len(train),
               "identity_bin_counts": counts.to_dict(),
               "median_best_identity": round(float(tab.best_identity.median()), 3),
               "frac_ge_30": round(float((tab.best_identity >= 0.30).mean()), 3),
               "frac_ge_50": round(float((tab.best_identity >= 0.50).mean()), 3)}
    print(json.dumps(summary, indent=2))

    if args.scores:
        sc = pd.read_csv(resolve(args.scores), sep="\t")
        sc = sc.merge(tab[["protein_name", "identity_bin", "best_identity"]], on="protein_name", how="left")
        model_cols = [c for c in sc.columns if c.endswith("__full") or c.endswith("__center")]
        strat = {}
        print(f"\n{'bin':<10}{'n_prot':>7}{'n_rows':>9}" + "".join(f"{c.split('__')[0]:>9}" for c in model_cols))
        cum_cols = {"all": sc, "<50% only": sc[sc.best_identity < 0.50], "<30% only": sc[sc.best_identity < 0.30]}
        for b in order:
            sub = sc[sc.identity_bin == b]
            if len(sub) == 0:
                continue
            strat[b] = {"n_proteins": int(sub.protein_name.nunique()), "n_rows": int(len(sub)),
                        **{c: auroc(sub.binding_label, sub[c]) for c in model_cols}}
            print(f"{b:<10}{strat[b]['n_proteins']:>7}{len(sub):>9}" + "".join(
                f"{strat[b][c]:>9.3f}" if strat[b][c] is not None else f"{'n/a':>9}" for c in model_cols))
        for k, sub in cum_cols.items():
            if len(sub) == 0:
                continue
            strat[k] = {"n_proteins": int(sub.protein_name.nunique()), "n_rows": int(len(sub)),
                        **{c: auroc(sub.binding_label, sub[c]) for c in model_cols}}
            print(f"{k:<10}{strat[k]['n_proteins']:>7}{len(sub):>9}" + "".join(
                f"{strat[k][c]:>9.3f}" if strat[k][c] is not None else f"{'n/a':>9}" for c in model_cols))
        summary["stratified_auroc"] = strat

    fn = f"{args.name}_stratified.json" if args.identity_tsv else f"{args.name}_summary.json"
    (out / fn).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / fn}")


if __name__ == "__main__":
    main()
