#!/usr/bin/env python3
"""
35_cross_protocol_motif_concordance.py
--------------------------------------
Motif concordance for matched proteins using top/bottom 7-mer anchors
from scripts/24 (all_protocols_summary.tsv).

For each protein × protocol (positives only):
  - unique matched_kmer set (top ranks)
  - pairwise Jaccard between protocols
  - optional Spearman on shared k-mers using enrichment / Z-score columns

Maps summary protocol labels to roster protocol ids:
  htr_selex, rbns, rnacompete→eukarya/rbpzoo/ucrbp via dataset column

Usage:
    python scripts/35_cross_protocol_motif_concordance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p: str | Path, base: Path = ROOT) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def map_summary_protocol(row: pd.Series) -> str | None:
    protocol = str(row.get("protocol", "")).strip().lower()
    dataset = str(row.get("dataset", "")).strip().lower()
    if protocol == "htr_selex":
        return "htr_selex"
    if protocol == "rbns":
        return "rbns"
    if protocol == "rnacompete":
        if "eukarya" in dataset:
            return "rnacompete_eukarya"
        if "rbpzoo" in dataset:
            return "rnacompete_rbpzoo"
        if "ucrbp" in dataset:
            return "rnacompete_ucrbp"
        return "rnacompete"
    if protocol == "eclip":
        return "eclip"
    return None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else float("nan")


def kmer_cores(kmers: set[str], core_k: int = 5) -> set[str]:
    """All core_k substrings of the 7-mers (handles motif register shifts)."""
    cores: set[str] = set()
    for km in kmers:
        s = str(km).upper().replace("T", "U")
        if len(s) < core_k:
            cores.add(s)
            continue
        for i in range(len(s) - core_k + 1):
            cores.add(s[i : i + core_k])
    return cores


def score_column(df: pd.DataFrame) -> str | None:
    for c in ("kmer_z_score", "kmer_enrichment_score"):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def top_kmers_for_group(g: pd.DataFrame, top_n: int) -> tuple[set[str], dict[str, float]]:
    pos = g[g["binding_label"] == 1].copy()
    if pos.empty:
        return set(), {}
    score_col = score_column(pos)
    if score_col:
        pos[score_col] = pd.to_numeric(pos[score_col], errors="coerce")
        pos = pos.dropna(subset=["matched_kmer"])
        # best score per kmer
        ranked = (
            pos.groupby("matched_kmer", as_index=False)[score_col]
            .max()
            .sort_values(score_col, ascending=False)
        )
    else:
        pos = pos.dropna(subset=["matched_kmer"])
        if "rank" in pos.columns:
            ranked = (
                pos.sort_values("rank")
                .drop_duplicates("matched_kmer")
                .assign(_score=lambda d: -pd.to_numeric(d["rank"], errors="coerce"))
            )
            ranked = ranked.rename(columns={"_score": "score"})
            score_col = "score"
            ranked = ranked[["matched_kmer", score_col]].sort_values(
                score_col, ascending=False
            )
        else:
            counts = pos["matched_kmer"].value_counts().reset_index()
            counts.columns = ["matched_kmer", "score"]
            ranked = counts
            score_col = "score"

    top = ranked.head(top_n)
    kmers = set(top["matched_kmer"].astype(str).str.upper().str.replace("T", "U"))
    scores = {
        str(r["matched_kmer"]).upper().replace("T", "U"): float(r[score_col])
        for _, r in top.iterrows()
        if pd.notna(r[score_col])
    }
    return kmers, scores


def spearman_shared(sa: dict[str, float], sb: dict[str, float]) -> tuple[float, int]:
    shared = sorted(set(sa) & set(sb))
    if len(shared) < 3:
        return float("nan"), len(shared)
    xa = [sa[k] for k in shared]
    xb = [sb[k] for k in shared]
    rho, _ = spearmanr(xa, xb)
    return float(rho), len(shared)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_protocol.yaml")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    paths = cfg["paths"]
    out_dir = resolve(args.out_dir or paths["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    roster_path = out_dir / "protein_roster.tsv"
    if not roster_path.exists():
        raise SystemExit(f"Missing roster: {roster_path} (run script 33 first)")
    roster = pd.read_csv(roster_path, sep="\t")
    roster_keys = set(roster["protein_key"])

    summary_path = resolve(paths.get("top_bottom_summary", "results/top_bottom_examples/all_protocols_summary.tsv"))
    summary = pd.read_csv(summary_path, sep="\t")
    summary["protocol_id"] = summary.apply(map_summary_protocol, axis=1)
    summary = summary.dropna(subset=["protocol_id"])
    # Restrict to roster representative natives only (avoid construct merges)
    allowed_natives: dict[str, set[str]] = {}
    native_to_key: dict[str, dict[str, str]] = {}
    for pid in [
        "htr_selex",
        "rbns",
        "rnacompete_eukarya",
        "rnacompete_rbpzoo",
        "rnacompete_ucrbp",
        "eclip",
    ]:
        col = f"name_in_{pid}"
        if col not in roster.columns:
            continue
        sub = roster[roster[col].astype(str).str.len() > 0]
        allowed_natives[pid] = set(sub[col].astype(str))
        native_to_key[pid] = dict(zip(sub[col].astype(str), sub["protein_key"]))

    summary["protein_name"] = summary["protein_name"].astype(str)
    keep_mask = []
    keys = []
    for _, r in summary.iterrows():
        pid = r["protocol_id"]
        native = str(r["protein_name"])
        if pid in allowed_natives and native in allowed_natives[pid]:
            keep_mask.append(True)
            keys.append(native_to_key[pid][native])
        else:
            keep_mask.append(False)
            keys.append("")
    summary = summary.copy()
    summary["_keep"] = keep_mask
    summary["_key"] = keys
    summary = summary.loc[summary["_keep"]].copy()
    summary["protein_key"] = summary["_key"]
    summary = summary.drop(columns=["_keep", "_key"])
    summary = summary[summary["protein_key"].isin(roster_keys)].copy()

    # Build kmer sets per (protein_key, protocol) using representative native only
    motif_rows = []
    sets: dict[tuple[str, str], set[str]] = {}
    scores: dict[tuple[str, str], dict[str, float]] = {}
    for (key, pid), g in summary.groupby(["protein_key", "protocol_id"], sort=False):
        # If multiple natives somehow remain, keep only representative
        rep = native_to_key.get(pid, {})
        # invert: key -> native for this protocol
        natives_for_key = [n for n, k in rep.items() if k == key]
        if natives_for_key:
            g = g[g["protein_name"].isin(natives_for_key)]
        kmers, sc = top_kmers_for_group(g, args.top_n)
        if not kmers:
            continue
        sets[(key, pid)] = kmers
        scores[(key, pid)] = sc
        motif_rows.append(
            {
                "protein_key": key,
                "protocol": pid,
                "protein_name": natives_for_key[0] if natives_for_key else "",
                "n_kmers": len(kmers),
                "kmers": ",".join(sorted(kmers)),
            }
        )
    motifs_df = pd.DataFrame(motif_rows)
    motifs_df.to_csv(out_dir / "motif_sets.tsv", sep="\t", index=False)

    # Pairwise concordance
    pair_rows = []
    by_protein: dict[str, list[str]] = {}
    for key, pid in sets:
        by_protein.setdefault(key, []).append(pid)

    dom_map = dict(zip(roster["protein_key"], roster.get("domain_class", pd.Series(dtype=str))))
    arch_map = dict(
        zip(roster["protein_key"], roster.get("domain_architecture", pd.Series(dtype=str)))
    )

    for key, protocols in by_protein.items():
        protocols = sorted(set(protocols))
        for i, a in enumerate(protocols):
            for b in protocols[i + 1 :]:
                sa, sb = sets[(key, a)], sets[(key, b)]
                ja = jaccard(sa, sb)
                ja_core = jaccard(kmer_cores(sa, 5), kmer_cores(sb, 5))
                rho, n_shared = spearman_shared(scores[(key, a)], scores[(key, b)])
                pair_rows.append(
                    {
                        "protein_key": key,
                        "protocol_a": a,
                        "protocol_b": b,
                        "jaccard_top_kmers": ja,
                        "jaccard_core5": ja_core,
                        "spearman_shared_kmers": rho,
                        "n_shared_kmers": n_shared,
                        "n_kmers_a": len(sa),
                        "n_kmers_b": len(sb),
                        "domain_class": dom_map.get(key, "unknown"),
                        "domain_architecture": arch_map.get(key, ""),
                    }
                )

    conc = pd.DataFrame(pair_rows)
    conc_path = out_dir / "motif_concordance.tsv"
    conc.to_csv(conc_path, sep="\t", index=False)

    # Optional join with transfer metrics if present
    transfer_path = out_dir / "transfer_metrics.tsv"
    joined_path = None
    if transfer_path.exists() and not conc.empty:
        tr = pd.read_csv(transfer_path, sep="\t")
        # symmetrize: match unordered protocol pair to both transfer directions
        merge_rows = []
        for _, r in conc.iterrows():
            sub = tr[
                (tr["protein_key"] == r["protein_key"])
                & (
                    (
                        (tr["train_protocol"] == r["protocol_a"])
                        & (tr["test_protocol"] == r["protocol_b"])
                    )
                    | (
                        (tr["train_protocol"] == r["protocol_b"])
                        & (tr["test_protocol"] == r["protocol_a"])
                    )
                )
            ]
            if sub.empty:
                merge_rows.append({**r.to_dict(), "mean_transfer_auroc": np.nan, "n_transfer_dirs": 0})
            else:
                merge_rows.append(
                    {
                        **r.to_dict(),
                        "mean_transfer_auroc": float(sub["auroc"].mean()),
                        "n_transfer_dirs": int(len(sub)),
                    }
                )
        joined = pd.DataFrame(merge_rows)
        joined_path = out_dir / "motif_vs_transfer.tsv"
        joined.to_csv(joined_path, sep="\t", index=False)

    summary_json = {
        "top_n": args.top_n,
        "n_motif_protein_protocol": int(len(motifs_df)),
        "n_concordance_pairs": int(len(conc)),
        "median_jaccard": float(conc["jaccard_top_kmers"].median()) if not conc.empty else None,
        "median_jaccard_core5": float(conc["jaccard_core5"].median()) if not conc.empty else None,
        "median_jaccard_by_domain": (
            conc.groupby("domain_class")["jaccard_top_kmers"].median().dropna().to_dict()
            if not conc.empty
            else {}
        ),
        "median_jaccard_core5_by_domain": (
            conc.groupby("domain_class")["jaccard_core5"].median().dropna().to_dict()
            if not conc.empty
            else {}
        ),
        "outputs": {
            "motif_sets": "motif_sets.tsv",
            "concordance": "motif_concordance.tsv",
            "motif_vs_transfer": str(joined_path.name) if joined_path else None,
        },
    }
    with open(out_dir / "motif_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Motif protein×protocol sets: {len(motifs_df)}")
    print(f"Concordance pairs: {len(conc)}")
    if not conc.empty:
        print(f"Median Jaccard (exact 7-mer): {conc['jaccard_top_kmers'].median():.3f}")
        print(f"Median Jaccard (core 5-mer):  {conc['jaccard_core5'].median():.3f}")
        print(
            f"Pairs with exact>0 / core5>0: "
            f"{(conc['jaccard_top_kmers'] > 0).sum()} / {(conc['jaccard_core5'] > 0).sum()}"
        )
    print(f"Wrote {conc_path}")


if __name__ == "__main__":
    main()
