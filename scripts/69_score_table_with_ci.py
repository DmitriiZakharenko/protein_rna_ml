#!/usr/bin/env python3
"""
69_score_table_with_ci.py
-------------------------
One comparison table for any number of scorers on the same benchmark, with protein-level bootstrap confidence intervals, paired
differences against a reference scorer, and strata by MMseqs2 identity bin and Pfam family.

Each scorer is a table with protein_name, binding_label and a score column:
  --scorer "nn1=results/knn_baseline/v3b_probe_scores.tsv.gz:nn1_lr"
  --scorer "frozen=results/v3b_frozen/bilinear_xneg_s42/scores_probe.tsv.gz"        (column defaults to `prob`)
The first scorer is the reference for paired differences. Proteins are resampled with replacement (the same draws for all
scorers, so differences are paired); AUROC of the pooled rows is computed per draw.

Usage
-----
  python scripts/69_score_table_with_ci.py --name v3b_probe --scorer "nn1=...:nn1_lr" --scorer "v4=...:v4_s42__full" \\
      --identity_tsv results/homology_audit/v3b_test_protein_identity.tsv --split_map results/v3b/protein_split_map.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def auroc(y, s):
    return np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def load_scorer(spec: str) -> tuple[str, pd.DataFrame]:
    name, rest = spec.split("=", 1)
    path, _, col = rest.partition(":")
    df = pd.read_csv(resolve(path), sep="\t", usecols=lambda c: c in ("protein_name", "binding_label", col or "prob"))
    return name, df.rename(columns={col or "prob": "score"})[["protein_name", "binding_label", "score"]]


def boot(groups: dict[str, dict[str, tuple]], prots: list[str], n: int, rng, ref: str):
    """groups[scorer][protein] = (y, s). Returns {scorer: (auroc, lo, hi)} and {scorer: (delta, lo, hi, p_le_0)}."""
    names = list(groups)

    def pooled(sc, ps):
        y = np.concatenate([groups[sc][p][0] for p in ps])
        s = np.concatenate([groups[sc][p][1] for p in ps])
        return auroc(y, s)

    point = {sc: pooled(sc, prots) for sc in names}
    draws = {sc: [] for sc in names}
    for _ in range(n):
        ps = list(rng.choice(prots, len(prots)))
        for sc in names:
            draws[sc].append(pooled(sc, ps))
    ci = {sc: (point[sc], float(np.nanpercentile(draws[sc], 2.5)), float(np.nanpercentile(draws[sc], 97.5))) for sc in names}
    dl = {}
    for sc in names:
        if sc == ref:
            continue
        d = np.array(draws[sc]) - np.array(draws[ref])
        dl[sc] = (point[sc] - point[ref], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)), float(np.nanmean(d <= 0)))
    return ci, dl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--scorer", action="append", required=True)
    ap.add_argument("--identity_tsv", default="results/homology_audit/v3b_test_protein_identity.tsv")
    ap.add_argument("--split_map", default="results/v3b/protein_split_map.tsv")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--min_proteins", type=int, default=4, help="smallest stratum reported")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/v3b_report")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tabs = dict(load_scorer(s) for s in args.scorer)
    ref = next(iter(tabs))
    common = set.intersection(*(set(t.protein_name) for t in tabs.values()))
    groups = {}
    for sc, t in tabs.items():
        groups[sc] = {p: (g.binding_label.values, g.score.values) for p, g in t[t.protein_name.isin(common)].groupby("protein_name")}
    prots = sorted(p for p in common if all(len(np.unique(groups[sc][p][0])) == 2 for sc in tabs))
    ident = pd.read_csv(resolve(args.identity_tsv), sep="\t").set_index("protein_name").identity_bin.to_dict()
    fam = pd.read_csv(resolve(args.split_map), sep="\t").set_index("protein_name").family.to_dict()
    print(f"{args.name}: {len(prots)} proteins with both classes in all scorers; reference = {ref}")

    out = {"name": args.name, "n_proteins": len(prots), "reference": ref, "strata": {}}
    strata = {"all": prots}
    for key, mp in (("identity", ident), ("family", fam)):
        for v in sorted({mp.get(p) for p in prots if mp.get(p)}):
            sub = [p for p in prots if mp.get(p) == v]
            if len(sub) >= args.min_proteins:
                strata[f"{key}:{v}"] = sub
    lines = []
    for sname, sub in strata.items():
        ci, dl = boot(groups, sub, args.n_boot if sname == "all" else max(100, args.n_boot // 2), rng, ref)
        out["strata"][sname] = {"n_proteins": len(sub), "auroc": ci, "delta_vs_ref": dl}
        row = f"{sname:<22}{len(sub):>4}  " + "  ".join(
            f"{sc}: {ci[sc][0]:.3f} [{ci[sc][1]:.3f},{ci[sc][2]:.3f}]" + (f" d={dl[sc][0]:+.3f} [{dl[sc][1]:+.3f},{dl[sc][2]:+.3f}]" if sc in dl else "")
            for sc in tabs)
        lines.append(row)
        print(row, flush=True)
    o = resolve(args.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    (o / f"{args.name}.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {o / (args.name + '.json')}")


if __name__ == "__main__":
    main()
