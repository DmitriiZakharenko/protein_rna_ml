#!/usr/bin/env python3
"""
56_domain_family_audit.py
-------------------------
Sequence-pattern domain-family labels for v3a proteins (no external services) and
probe AUROC stratified by family.

Family labels
-------------
Pfam-A (`hmmscan`, domain E-value < 1e-3; download to data/external/pfam/Pfam-A.hmm and
`hmmpress` it) is mapped to RNA-binding families (RRM, KH, CCCH, DEAD, dsRBD, ...).
Proteins with several families are `multi`, with none `other`. The curated
`domain_class` (data/domains/protein_domains.tsv) is kept alongside as a cross-check
(`--validate`).

Usage
-----
  python scripts/56_domain_family_audit.py --test_tsv data/generalized_v3a/test.tsv \\
      --scores results/overnight/probe_final/scores_cross_neg_probe.tsv.gz
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PFAM_HMM = "data/external/pfam/Pfam-A.hmm"
# Pfam name prefix -> RNA-binding family
FAMILIES = [("RRM", "RRM"), ("KH_", "KH"), ("zf-CCCH", "CCCH"), ("zf-CCHC", "CCHC"), ("zf-C2H2", "C2H2"),
            ("DEAD", "DEAD"), ("dsrm", "dsRBD"), ("PUF", "PUF"), ("CSD", "CSD"), ("La", "La"), ("YTH", "YTH"),
            ("S1", "S1"), ("Piwi", "Piwi"), ("PAZ", "PAZ")]
MANIFEST = "data/raw/eclip_skipper/manifests/encode_eclip_rbp_id_best_acc_seq.added_domain_annot.tsv"


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def norm(name: str) -> str:
    return re.sub(r"-construct\d+$", "", str(name)).upper()


def pfam_family(hit_names: list[str]) -> str:
    fams = []
    for h in hit_names:
        for pre, fam in FAMILIES:
            if h == pre or h.startswith(pre):
                if fam not in fams:
                    fams.append(fam)
                break
    if not fams:
        return "other"
    return fams[0] if len(fams) == 1 else "multi"


def run_hmmscan(prots: dict[str, str], cache: Path, evalue: float = 1e-3) -> dict[str, list[str]]:
    """Return {protein: [Pfam domain names]} (hmmscan, domain E-value < evalue). Cached as TSV."""
    import subprocess, tempfile
    if cache.exists():
        df = pd.read_csv(cache, sep="\t")
        if set(prots) <= set(df.protein_name):
            return {k: g.pfam.dropna().tolist() for k, g in df.groupby("protein_name")}
    with tempfile.TemporaryDirectory() as td:
        fa, tbl = Path(td) / "q.fa", Path(td) / "dom.tbl"
        fa.write_text("".join(f">{k}\n{v}\n" for k, v in prots.items()))
        subprocess.run(["hmmscan", "--cpu", "4", "-E", str(evalue), "--domE", str(evalue), "--domtblout", str(tbl),
                        str(resolve(PFAM_HMM)), str(fa)], check=True, stdout=subprocess.DEVNULL)
        rows = []
        for line in tbl.read_text().splitlines():
            if line.startswith("#"):
                continue
            f = line.split()
            rows.append((f[3], f[0], f[1], float(f[12])))
    df = pd.DataFrame(rows, columns=["protein_name", "pfam", "pfam_acc", "domain_ievalue"])
    for k in prots:
        if k not in set(df.protein_name):
            df.loc[len(df)] = (k, None, None, np.nan)
    df.sort_values(["protein_name", "domain_ievalue"]).to_csv(cache, sep="\t", index=False)
    return {k: g.pfam.dropna().tolist() for k, g in df.groupby("protein_name")}


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_tsv", default="data/generalized_v3a/test.tsv")
    ap.add_argument("--train_tsv", default="data/generalized_v3a/train.tsv")
    ap.add_argument("--identity_tsv", default="results/homology_audit/v3a_test_protein_identity.tsv")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--out_dir", default="results/homology_audit")
    ap.add_argument("--validate", action="store_true", help="Pfam family vs curated class on train proteins")
    args = ap.parse_args()
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cur = pd.read_csv(resolve("data/domains/protein_domains.tsv"), sep="\t")
    cur_map = {k.upper(): c for k, c in zip(cur.protein_key, cur.domain_class)}
    man = pd.read_csv(resolve(MANIFEST), sep="\t")
    man_map = {s.upper(): str(n) for s, n in zip(man.symbol, man.domain_names)}

    test = pd.read_csv(resolve(args.test_tsv), sep="\t", usecols=["protein_name", "protein_sequence"]).drop_duplicates("protein_name")
    train = pd.read_csv(resolve(args.train_tsv), sep="\t", usecols=["protein_name", "protein_sequence"]).drop_duplicates("protein_name")
    allp = {**dict(zip(train.protein_name, train.protein_sequence)), **dict(zip(test.protein_name, test.protein_sequence))}
    hits = run_hmmscan(allp, out / "v3a_pfam_hits.tsv")
    fam_of = {k: pfam_family(v) for k, v in hits.items()}
    pd.DataFrame({"protein_name": list(fam_of), "pfam_family": list(fam_of.values()),
                  "pfam_domains": [";".join(hits[k]) for k in fam_of]}).to_csv(out / "v3a_pfam_family.tsv", sep="\t", index=False)
    rows = []
    for name in test.protein_name:
        k = norm(name)
        rows.append((name, fam_of[name], cur_map.get(k, ""), ";".join(hits[name]), man_map.get(k, "")))
    tab = pd.DataFrame(rows, columns=["protein_name", "family", "curated_class", "pfam_domains", "interpro_domains"])
    ident = pd.read_csv(resolve(args.identity_tsv), sep="\t")[["protein_name", "best_identity", "identity_bin"]]
    tab = tab.merge(ident, on="protein_name")
    tab.to_csv(out / "v3a_test_domain_family.tsv", sep="\t", index=False)

    summary = {"n_test": len(tab), "family_counts": tab.family.value_counts().to_dict(),
               "train_family_counts": pd.Series([fam_of[k] for k in train.protein_name]).value_counts().to_dict()}
    print(json.dumps(summary, indent=2))
    if args.validate:
        tr = train.assign(curated=train.protein_name.map(lambda n: cur_map.get(norm(n))),
                          pfam=train.protein_name.map(fam_of)).dropna(subset=["curated"])
        ct = pd.crosstab(tr.curated, tr.pfam)
        print("\nPfam family vs curated class (train proteins):")
        print(ct)
        summary["validation_crosstab"] = ct.to_dict()

    if args.scores:
        sc = pd.read_csv(resolve(args.scores), sep="\t").merge(tab[["protein_name", "family"]], on="protein_name")
        cols = [c for c in sc.columns if c.endswith("__full")]
        fam_res = {}
        print(f"\n{'family':<9}{'n_prot':>7}{'n_rows':>9}  " + "  ".join(f"{c.split('__')[0]:>7}" for c in cols) + "   v2mean  v4mean")
        for fam, sub in list(sc.groupby("family")) + [("ALL", sc)]:
            r = {c: auroc(sub.binding_label, sub[c]) for c in cols}
            v2 = np.nanmean([v for c, v in r.items() if c.startswith("v2") and v is not None] or [np.nan])
            v4 = np.nanmean([v for c, v in r.items() if c.startswith("v4") and v is not None] or [np.nan])
            fam_res[fam] = {"n_proteins": int(sub.protein_name.nunique()), "n_rows": int(len(sub)), "v2_mean": v2, "v4_mean": v4, **r}
            print(f"{fam:<9}{fam_res[fam]['n_proteins']:>7}{len(sub):>9}  " + "  ".join(
                f"{r[c]:>7.3f}" if r[c] is not None else f"{'n/a':>7}" for c in cols) + f"   {v2:.3f}   {v4:.3f}")
        summary["family_auroc"] = fam_res

        # Within-family identity strata (RRM only has enough proteins)
        sc = sc.merge(tab[["protein_name", "identity_bin"]], on="protein_name")
        rrm = sc[sc.family == "RRM"]
        print("\nRRM only, by identity bin:")
        print(f"{'bin':<9}{'n_prot':>7}{'n_rows':>9}   v2mean  v4mean")
        rrm_res = {}
        for b in ["no_hit", "30-50%", "50-70%", "70-90%", ">=90%"]:
            sub = rrm[rrm.identity_bin == b]
            if sub.empty:
                continue
            r = {c: auroc(sub.binding_label, sub[c]) for c in cols}
            v2 = float(np.nanmean([v for c, v in r.items() if c.startswith("v2") and v is not None]))
            v4 = float(np.nanmean([v for c, v in r.items() if c.startswith("v4") and v is not None]))
            rrm_res[b] = {"n_proteins": int(sub.protein_name.nunique()), "n_rows": int(len(sub)), "v2_mean": v2, "v4_mean": v4}
            print(f"{b:<9}{rrm_res[b]['n_proteins']:>7}{len(sub):>9}   {v2:.3f}   {v4:.3f}")
        summary["rrm_by_identity_bin"] = rrm_res
    (out / "v3a_test_domain_family_summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
