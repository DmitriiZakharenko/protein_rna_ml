#!/usr/bin/env python3
"""
33_build_cross_protocol_roster.py
---------------------------------
Build the matched protein panel for cross-protocol comparison.

Strict matching rules
---------------------
- Within a protocol, native names are NEVER merged via synonyms.
- Synonyms (HuR→ELAVL1, etc.) apply only when they do not collide with another
  native already present in that protocol (see src.data.protein_names).
- If several constructs map to one gene in one protocol, pick ONE representative
  (prefer name without -constructN) and record the rule + discarded natives.
- A2BP1 and RBFOX1 both present in the same panel stay as separate keys.

Outputs
-------
  results/cross_protocol/protein_roster.tsv
  results/cross_protocol/protocol_protein_lists.tsv
  results/cross_protocol/match_ambiguities.tsv
  results/cross_protocol/roster_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.protein_names import (
    base_gene_key,
    choose_representative_native,
    coarse_domain_class,
    resolve_match_key,
)


PROTOCOL_METRICS_FALLBACK = {
    "htr_selex": "results/rna_only_per_protein_honest/htr_selex_per_protein_metrics.tsv",
    "rbns": "results/rna_only_per_protein_honest/rbns_per_protein_metrics.tsv",
    "rnacompete_eukarya": "results/rna_only_per_protein_honest/rnacompete_eukarya_per_protein_metrics.tsv",
    "rnacompete_rbpzoo": "results/rna_only_per_protein_honest/rnacompete_rbpzoo_per_protein_metrics.tsv",
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p: str | Path, base: Path = ROOT) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def proteins_from_metrics(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t")
    col = "protein_name" if "protein_name" in df.columns else df.columns[0]
    return {str(x).strip() for x in df[col].dropna().unique() if str(x).strip()}


def proteins_from_eclip(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t", usecols=["protein_name"])
    return {str(x).strip() for x in df["protein_name"].dropna().unique()}


def proteins_from_raw_tsv(path: Path, protein_col: str | None) -> set[str]:
    df = pd.read_csv(path, sep="\t", nrows=5)
    if protein_col and protein_col in df.columns:
        col = protein_col
    elif "protein_name" in df.columns:
        col = "protein_name"
    elif "target_name" in df.columns:
        col = "target_name"
    else:
        raise ValueError(f"No protein column in {path}; columns={list(df.columns)}")
    names: set[str] = set()
    for chunk in pd.read_csv(path, sep="\t", usecols=[col], chunksize=200_000):
        names.update(str(x).strip() for x in chunk[col].dropna().unique() if str(x).strip())
    return names


def assign_keys_for_protocol(natives: set[str]) -> dict[str, tuple[str, str]]:
    """
    Map each native name → (match_key, rule) without within-protocol synonym collisions.

    Pass 1: base keys only (occupancy).
    Pass 2: resolve synonyms with collision blocking.
    """
    bases = {n: base_gene_key(n) for n in natives}
    occupied_bases = {b for b in bases.values() if b}
    out: dict[str, tuple[str, str]] = {}
    for native, base in bases.items():
        if not base:
            out[native] = ("", "empty")
            continue
        key, rule = resolve_match_key(
            native,
            keys_already_in_protocol=occupied_bases,
            apply_synonyms=True,
        )
        out[native] = (key, rule)
    return out


def load_table_s1_domains(path: Path) -> pd.DataFrame:
    """
    Domain rows keyed by base_gene_key (no synonyms).

    If multiple Table S1 constructs exist for one gene, keep ALL rows and mark
    n_constructs; roster join will flag ambiguous domain assignment.
    """
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
        z = r.get("Average Z-score of top 10 7-mers")
        try:
            z = float(z) if pd.notna(z) else float("nan")
        except (TypeError, ValueError):
            z = float("nan")
        species = str(r["Species"]).strip() if pd.notna(r.get("Species")) else ""
        bounds = (
            str(r["Domain Boundaries"]).strip()
            if pd.notna(r.get("Domain Boundaries"))
            else ""
        )
        hyb = str(r["Hyb ID(s)"]).strip() if pd.notna(r.get("Hyb ID(s)")) else ""
        rncmpt = (
            str(r["RNAcompete ID(s)"]).strip()
            if pd.notna(r.get("RNAcompete ID(s)"))
            else ""
        )
        rows.append(
            {
                "protein_name_native": pname,
                "protein_key": base_gene_key(pname),
                "domain_architecture": dom,
                "domain_boundaries": bounds,
                "species": species,
                "hyb_ids": hyb,
                "rnacompete_ids": rncmpt,
                "mean_top10_z": z,
                "domain_class": coarse_domain_class(dom),
            }
        )
    raw = pd.DataFrame(rows)
    raw = raw[raw["protein_key"] != ""].copy()
    return raw


def pick_domain_for_key(domains: pd.DataFrame, key: str) -> dict:
    """Select domain annotation for a match key; prefer human + highest Z; flag ambiguity."""
    empty = {
        "domain_architecture": "",
        "domain_boundaries": "",
        "domain_class": "unknown",
        "domain_species": "",
        "table_s1_name": "",
        "domain_n_constructs": 0,
        "domain_match_status": "missing",
    }
    if domains is None or domains.empty:
        return empty
    # Direct key, also try synonym reverse is not needed — roster keys are already resolved
    sub = domains[domains["protein_key"] == key]
    if sub.empty:
        # try if key is synonym target but Table S1 only has synonym source
        return empty
    archs = sorted({a for a in sub["domain_architecture"].dropna().astype(str) if a and a != "nan"})
    human = sub[sub["species"].str.lower() == "homo sapiens"]
    pool = human if not human.empty else sub
    pool = pool.copy()
    pool["_z"] = pd.to_numeric(pool["mean_top10_z"], errors="coerce").fillna(float("-inf"))
    best = pool.sort_values("_z", ascending=False).iloc[0]
    status = "unique"
    if len(archs) > 1:
        status = "ambiguous_architectures"
    elif len(sub) > 1:
        status = "multiple_constructs_same_architecture"
    return {
        "domain_architecture": str(best["domain_architecture"]),
        "domain_boundaries": str(best["domain_boundaries"]),
        "domain_class": str(best["domain_class"]),
        "domain_species": str(best["species"]),
        "table_s1_name": str(best["protein_name_native"]),
        "domain_n_constructs": int(len(sub)),
        "domain_match_status": status,
        "domain_architectures_all": "|".join(archs),
    }


def build_roster(
    protocol_proteins: dict[str, set[str]],
    domains: pd.DataFrame | None,
    min_protocols: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # protocol -> native -> (key, rule)
    assigned: dict[str, dict[str, tuple[str, str]]] = {}
    for protocol, natives in protocol_proteins.items():
        assigned[protocol] = assign_keys_for_protocol(natives)

    # key -> protocol -> list of natives
    membership: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    ambiguity_rows: list[dict] = []

    for protocol, mapping in assigned.items():
        # group natives by key
        by_key: dict[str, list[str]] = defaultdict(list)
        for native, (key, rule) in mapping.items():
            if not key:
                continue
            by_key[key].append(native)
            if rule == "synonym_blocked_collision":
                ambiguity_rows.append(
                    {
                        "protein_key": key,
                        "protocol": protocol,
                        "protein_name_native": native,
                        "issue": "synonym_blocked_collision",
                        "detail": f"kept base key; synonym blocked",
                    }
                )
        for key, natives in by_key.items():
            membership[key][protocol] = sorted(set(natives))
            if len(set(natives)) > 1:
                chosen, rule = choose_representative_native(natives)
                ambiguity_rows.append(
                    {
                        "protein_key": key,
                        "protocol": protocol,
                        "protein_name_native": ";".join(sorted(set(natives))),
                        "issue": "multiple_natives_same_key",
                        "detail": f"representative={chosen}; rule={rule}",
                    }
                )

    list_rows = []
    roster_rows = []
    for key, by_prot in sorted(membership.items()):
        if len(by_prot) < min_protocols:
            continue
        protocols = sorted(by_prot.keys())
        rep_map: dict[str, str] = {}
        rep_rule: dict[str, str] = {}
        all_natives_map: dict[str, str] = {}
        for p in protocols:
            natives = by_prot[p]
            chosen, rule = choose_representative_native(natives)
            rep_map[p] = chosen
            rep_rule[p] = rule
            all_natives_map[p] = ";".join(sorted(set(natives)))

        preferred_native = rep_map[protocols[0]]
        row = {
            "protein_key": key,
            "protein_name": preferred_native,
            "n_protocols": len(protocols),
            "protocols": ";".join(protocols),
            "protocol_set": "|".join(protocols),
        }
        for p in protocol_proteins:
            row[f"name_in_{p}"] = rep_map.get(p, "")
            row[f"all_names_in_{p}"] = all_natives_map.get(p, "")
            row[f"rep_rule_in_{p}"] = rep_rule.get(p, "")
            row[f"in_{p}"] = int(p in by_prot)

        dom = pick_domain_for_key(domains, key) if domains is not None else {
            "domain_architecture": "",
            "domain_boundaries": "",
            "domain_class": "unknown",
            "domain_species": "",
            "table_s1_name": "",
            "domain_n_constructs": 0,
            "domain_match_status": "missing",
            "domain_architectures_all": "",
        }
        row.update(dom)
        roster_rows.append(row)

        for p, natives in by_prot.items():
            for n in natives:
                list_rows.append(
                    {
                        "protein_key": key,
                        "protocol": p,
                        "protein_name_native": n,
                        "is_representative": int(n == rep_map[p]),
                        "assign_rule": assigned[p].get(n, ("", ""))[1],
                    }
                )

    roster = pd.DataFrame(roster_rows)
    lists = pd.DataFrame(list_rows)
    amb = pd.DataFrame(ambiguity_rows)
    if not roster.empty:
        roster = roster.sort_values(
            ["n_protocols", "protein_key"], ascending=[False, True]
        ).reset_index(drop=True)
    return roster, lists, amb


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cross-protocol protein roster")
    ap.add_argument("--config", default="configs/cross_protocol.yaml")
    ap.add_argument("--scan_raw", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    paths = cfg.get("paths", {})
    out_dir = resolve(args.out_dir or paths.get("out_dir", "results/cross_protocol"))
    out_dir.mkdir(parents=True, exist_ok=True)
    min_protocols = int(cfg.get("min_protocols", 2))

    protocol_proteins: dict[str, set[str]] = {}
    source_notes: dict[str, str] = {}

    for proto in cfg.get("protocols", []):
        pid = proto["id"]
        path_key = proto.get("path_key", pid)
        if pid == "eclip" or path_key == "eclip":
            epath = resolve(paths["eclip"])
            if epath.exists():
                protocol_proteins[pid] = proteins_from_eclip(epath)
                source_notes[pid] = str(epath)
            else:
                print(f"[warn] missing eCLIP: {epath}")
            continue

        if args.scan_raw:
            raw_path = resolve(paths[path_key])
            if not raw_path.exists():
                print(f"[warn] missing raw for {pid}: {raw_path}")
                continue
            protocol_proteins[pid] = proteins_from_raw_tsv(
                raw_path, proto.get("protein_col")
            )
            source_notes[pid] = str(raw_path)
        else:
            cached = paths.get("cached_metrics", {}).get(pid) or PROTOCOL_METRICS_FALLBACK.get(
                pid
            )
            if not cached:
                print(f"[warn] no cached metrics for {pid}; skip (or pass --scan_raw)")
                continue
            cpath = resolve(cached)
            if not cpath.exists():
                print(f"[warn] missing metrics for {pid}: {cpath}")
                continue
            protocol_proteins[pid] = proteins_from_metrics(cpath)
            source_notes[pid] = str(cpath)

    # Reserve roster columns for every configured protocol (empty set if data missing).
    for proto in cfg.get("protocols", []):
        protocol_proteins.setdefault(proto["id"], set())

    domains = None
    table_s1 = resolve(paths.get("table_s1", "data/raw/rbpzoo/TableS1.xlsx"))
    if table_s1.exists():
        domains = load_table_s1_domains(table_s1)
        domains.to_csv(out_dir / "table_s1_domains_all_constructs.tsv", sep="\t", index=False)
        print(f"Table S1 domain rows: {len(domains)} (all constructs)")
    else:
        print(f"[warn] Table S1 not found: {table_s1}")

    roster, lists, amb = build_roster(protocol_proteins, domains, min_protocols)
    roster_path = out_dir / "protein_roster.tsv"
    lists_path = out_dir / "protocol_protein_lists.tsv"
    amb_path = out_dir / "match_ambiguities.tsv"
    roster.to_csv(roster_path, sep="\t", index=False)
    lists.to_csv(lists_path, sep="\t", index=False)
    amb.to_csv(amb_path, sep="\t", index=False)

    combo_counts = Counter(roster["protocol_set"]) if not roster.empty else Counter()
    domain_cov = (
        float((roster["domain_class"] != "unknown").mean()) if not roster.empty else 0.0
    )
    amb_domain = (
        int((roster["domain_match_status"] == "ambiguous_architectures").sum())
        if not roster.empty and "domain_match_status" in roster.columns
        else 0
    )
    summary = {
        "n_roster_proteins": int(len(roster)),
        "min_protocols": min_protocols,
        "proteins_per_protocol": {k: len(v) for k, v in protocol_proteins.items()},
        "source_notes": source_notes,
        "combo_counts": dict(combo_counts.most_common()),
        "domain_annotation_fraction": domain_cov,
        "domain_ambiguous_architecture_count": amb_domain,
        "n_match_ambiguity_rows": int(len(amb)),
        "matching_policy": {
            "within_protocol_synonym_merge": False,
            "construct_auto_merge_for_classifier": False,
            "representative_required": True,
            "notes": "Classifiers must use name_in_<protocol> exact native string only.",
        },
        "domain_class_counts": (
            roster["domain_class"].value_counts().to_dict() if not roster.empty else {}
        ),
    }
    summary_path = out_dir / "roster_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Roster proteins (≥{min_protocols} protocols): {len(roster)}")
    print(f"Domain annotated: {domain_cov:.1%} (ambiguous architectures: {amb_domain})")
    print(f"Match ambiguity rows: {len(amb)}")
    print("Top protocol combos:")
    for combo, n in combo_counts.most_common(10):
        print(f"  {n:3d}  {combo}")
    print(f"Wrote {roster_path}")
    print(f"Wrote {amb_path}")


if __name__ == "__main__":
    main()
