#!/usr/bin/env python3
"""
37_annotate_protein_domains.py
------------------------------
Build a domain annotation table for the cross-protocol roster.

Primary source (offline): Table S1 (Sasse et al. 2025) — construct domain
architecture + residue boundaries.

Optional: UniProt REST features for proteins still missing annotations
(--fetch_uniprot).

Outputs
-------
  data/domains/protein_domains.tsv
  data/domains/protein_domains_summary.json
  results/domain_aware/transfer_by_domain_stats.json  (if transfer metrics exist)

Usage:
    python scripts/37_annotate_protein_domains.py
    python scripts/37_annotate_protein_domains.py --fetch_uniprot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.protein_names import base_gene_key, coarse_domain_class

try:
    import requests
except ImportError:
    requests = None


def resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_boundaries(raw: str) -> list[tuple[int, int]]:
    """Parse Table S1 'Domain Boundaries' like '46;125;126;213' → pairs."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    parts = []
    for tok in s.replace(",", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            parts.append(int(float(tok)))
        except ValueError:
            continue
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        a, b = parts[i], parts[i + 1]
        if b >= a > 0:
            pairs.append((a, b))
    return pairs


def load_table_s1(path: Path) -> pd.DataFrame:
    ts1 = pd.read_excel(path)
    rows = []
    for _, r in ts1.iterrows():
        pname = str(r["Protein name"]).strip() if pd.notna(r["Protein name"]) else ""
        if not pname:
            continue
        dom = (
            str(r["Domains in construct"]).strip()
            if pd.notna(r.get("Domains in construct"))
            else ""
        )
        bounds = (
            str(r["Domain Boundaries"]).strip()
            if pd.notna(r.get("Domain Boundaries"))
            else ""
        )
        pairs = parse_boundaries(bounds)
        species = str(r["Species"]).strip() if pd.notna(r.get("Species")) else ""
        z = r.get("Average Z-score of top 10 7-mers")
        try:
            z = float(z) if pd.notna(z) else float("nan")
        except (TypeError, ValueError):
            z = float("nan")
        construct_seq = ""
        for col in ("Construct AA seq", "RBD or RBR AA Sequence"):
            if col in r.index and pd.notna(r[col]):
                construct_seq = str(r[col]).strip()
                break
        rows.append(
            {
                "protein_key": base_gene_key(pname),
                "protein_name_native": pname,
                "source": "table_s1",
                "species": species,
                "domain_architecture": dom,
                "domain_class": coarse_domain_class(dom),
                "domain_boundaries_raw": bounds,
                "domain_intervals": ";".join(f"{a}-{b}" for a, b in pairs),
                "n_domain_intervals": len(pairs),
                "construct_aa_seq": construct_seq,
                "construct_aa_len": len(construct_seq) if construct_seq else 0,
                "mean_top10_z": z,
                "rnacompete_ids": (
                    str(r["RNAcompete ID(s)"]).strip()
                    if pd.notna(r.get("RNAcompete ID(s)"))
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def pick_best_s1(sub: pd.DataFrame) -> pd.Series:
    human = sub[sub["species"].str.lower() == "homo sapiens"]
    pool = human if not human.empty else sub
    pool = pool.copy()
    pool["_z"] = pd.to_numeric(pool["mean_top10_z"], errors="coerce").fillna(float("-inf"))
    return pool.sort_values("_z", ascending=False).iloc[0]


def fetch_uniprot_domains(gene: str, taxon: int = 9606) -> dict | None:
    if requests is None:
        return None
    # search reviewed human gene
    q = f'(gene:{gene}) AND (organism_id:{taxon}) AND (reviewed:true)'
    url = "https://rest.uniprot.org/uniprotkb/search"
    try:
        r = requests.get(
            url,
            params={"query": q, "format": "json", "size": 1, "fields": "accession,gene_names,ft_domain,ft_region,sequence"},
            timeout=30,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            return None
        hit = results[0]
        acc = hit.get("primaryAccession", "")
        feats = hit.get("features", []) or []
        intervals = []
        labels = []
        for f in feats:
            if f.get("type") not in {"Domain", "Region", "Zinc finger", "Repeat"}:
                continue
            desc = (f.get("description") or f.get("type") or "").upper()
            # keep RNA-binding-ish domains
            keep_toks = ("RRM", "KH", "ZNF", "ZINC", "PUM", "SAM", "CSD", "DSRM", "CCCH", "RNA")
            if not any(t in desc for t in keep_toks) and f.get("type") != "Domain":
                continue
            loc = f.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if start and end:
                intervals.append(f"{int(start)}-{int(end)}")
                labels.append(desc.split()[0] if desc else f.get("type"))
        if not intervals and not labels:
            # fallback: any Domain feature
            for f in feats:
                if f.get("type") != "Domain":
                    continue
                loc = f.get("location", {})
                start = loc.get("start", {}).get("value")
                end = loc.get("end", {}).get("value")
                desc = f.get("description") or "Domain"
                if start and end:
                    intervals.append(f"{int(start)}-{int(end)}")
                    labels.append(str(desc).split()[0])
        arch = ";".join(labels) if labels else ""
        return {
            "uniprot_id": acc,
            "domain_architecture": arch,
            "domain_class": coarse_domain_class(arch) if arch else "unknown",
            "domain_intervals": ";".join(intervals),
            "n_domain_intervals": len(intervals),
            "source": "uniprot",
        }
    except Exception:
        return None


def transfer_domain_stats(transfer_path: Path, out_path: Path) -> dict:
    if not transfer_path.exists():
        return {}
    t = pd.read_csv(transfer_path, sep="\t")
    # restrict to three in vitro for headline stats
    three = {"htr_selex", "rbns", "rnacompete_eukarya"}
    t3 = t[t["train_protocol"].isin(three) & t["test_protocol"].isin(three)].copy()
    by_class = (
        t3.groupby("domain_class")["auroc"]
        .agg(["count", "median", "mean"])
        .reset_index()
        .sort_values("median", ascending=False)
    )
    # same-class pairs already labeled per protein; compare RRM vs unknown etc.
    stats = {
        "n_transfer_3assay": int(len(t3)),
        "median_auroc_3assay": float(t3["auroc"].median()) if len(t3) else None,
        "by_domain_class": by_class.to_dict(orient="records"),
        "note": "Per-protein domain_class from roster/Table S1; not yet construct-masked model.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    by_class.to_csv(out_path.with_name("transfer_by_domain_class.tsv"), sep="\t", index=False)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_protocol.yaml")
    ap.add_argument("--roster", default="results/cross_protocol/protein_roster.tsv")
    ap.add_argument("--out_tsv", default="data/domains/protein_domains.tsv")
    ap.add_argument("--fetch_uniprot", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.35, help="Delay between UniProt calls")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(resolve(args.config)))
    table_s1 = resolve(cfg["paths"]["table_s1"])
    roster = pd.read_csv(resolve(args.roster), sep="\t")
    s1 = load_table_s1(table_s1)

    rows = []
    missing = []
    for _, r in roster.iterrows():
        key = str(r["protein_key"])
        sub = s1[s1["protein_key"] == key]
        if not sub.empty:
            best = pick_best_s1(sub)
            n_arch = sub["domain_architecture"].nunique()
            rows.append(
                {
                    "protein_key": key,
                    "protein_name_roster": r.get("protein_name", key),
                    "source": "table_s1",
                    "table_s1_name": best["protein_name_native"],
                    "species": best["species"],
                    "domain_architecture": best["domain_architecture"],
                    "domain_class": best["domain_class"],
                    "domain_boundaries_raw": best["domain_boundaries_raw"],
                    "domain_intervals": best["domain_intervals"],
                    "n_domain_intervals": int(best["n_domain_intervals"]),
                    "construct_aa_len": int(best["construct_aa_len"]),
                    "has_construct_seq": int(best["construct_aa_len"] > 0),
                    "n_table_s1_constructs": int(len(sub)),
                    "architecture_ambiguous": int(n_arch > 1),
                    "uniprot_id": "",
                    "rnacompete_ids": best["rnacompete_ids"],
                }
            )
        else:
            missing.append(key)
            rows.append(
                {
                    "protein_key": key,
                    "protein_name_roster": r.get("protein_name", key),
                    "source": "missing",
                    "table_s1_name": "",
                    "species": "",
                    "domain_architecture": "",
                    "domain_class": "unknown",
                    "domain_boundaries_raw": "",
                    "domain_intervals": "",
                    "n_domain_intervals": 0,
                    "construct_aa_len": 0,
                    "has_construct_seq": 0,
                    "n_table_s1_constructs": 0,
                    "architecture_ambiguous": 0,
                    "uniprot_id": "",
                    "rnacompete_ids": "",
                }
            )

    if args.fetch_uniprot and missing:
        if requests is None:
            raise SystemExit("requests not installed; pip install requests")
        print(f"Fetching UniProt for {len(missing)} proteins missing Table S1...")
        by_key = {row["protein_key"]: i for i, row in enumerate(rows)}
        for key in missing:
            info = fetch_uniprot_domains(key)
            time.sleep(args.sleep)
            if not info:
                print(f"  no UniProt hit: {key}")
                continue
            i = by_key[key]
            rows[i].update(
                {
                    "source": "uniprot",
                    "domain_architecture": info["domain_architecture"],
                    "domain_class": info["domain_class"],
                    "domain_intervals": info["domain_intervals"],
                    "n_domain_intervals": info["n_domain_intervals"],
                    "uniprot_id": info["uniprot_id"],
                }
            )
            print(f"  {key} → {info['uniprot_id']} {info['domain_class']} {info['domain_intervals']}")

    out = pd.DataFrame(rows)
    out_path = resolve(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

    summary = {
        "n_roster": int(len(out)),
        "n_table_s1": int((out["source"] == "table_s1").sum()),
        "n_uniprot": int((out["source"] == "uniprot").sum()),
        "n_missing": int((out["source"] == "missing").sum()),
        "n_with_intervals": int((out["n_domain_intervals"] > 0).sum()),
        "n_with_construct_seq": int(out["has_construct_seq"].sum()),
        "domain_class_counts": out["domain_class"].value_counts().to_dict(),
        "output": str(out_path.relative_to(ROOT)),
    }
    summary_path = out_path.with_name("protein_domains_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    stats = transfer_domain_stats(
        resolve("results/cross_protocol/transfer_metrics.tsv"),
        resolve("results/domain_aware/transfer_by_domain_stats.json"),
    )

    print(json.dumps(summary, indent=2))
    if stats:
        print("Wrote results/domain_aware/transfer_by_domain_stats.json")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
