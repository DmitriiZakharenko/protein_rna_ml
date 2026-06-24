#!/usr/bin/env python3
"""
24_extract_top_bottom_examples.py
----------------------------------
Extract top-5 positive and bottom-5 negative RNA examples per RBP,
anchored on the top-K enriched 7-mers per protein.

Designed for structural follow-up and motif validation.

Protocols and k-mer sources
----------------------------
rnacompete
  k-mer source : (1) Wide Z-score matrix (--zscore_file; RBPZoo 2025 IDs), or
                 (2) per-experiment TSVs (--kmer_dir; Eukarya/ucRBP legacy IDs:
                 {RNCMPT}_zscores.tsv with z_setAB), matching rnacompete_analysis.
                 Falls back to label-pool enrichment only when neither source matches
                 the experiment ID (emits a warning).
  Positives    : Highest-intensity probes (binding_label=1) containing ≥1 top-K 7-mer.
  Negatives    : Lowest-intensity probes (binding_label=0) containing NO top-K 7-mers.
  Annotation   : matched_kmer + kmer_position (0-based) added to output.

rbns
  Underlying clean data (rbns_analysis):
    Positives  : pulldown-enriched sequences (source=enriched; R_max available).
    Negatives  : 0 nM input-pool sequences absent from positive concentrations
                 (source=background).
  k-mer source : Computed per-protein from binding_label pools.
  Top-5 pos    : binding_label=1, contains ≥1 top-K 7-mer; ranked by R_max (primary),
                 then motif k-mer frequency in positive pool.
  Top-5 neg    : binding_label=0, contains NO top-K 7-mer; ranked by k-mer frequency
                 in the negative pool (typical background composition, no motif hit).
  Note         : RBNS uses column 'target_name' for protein; handled automatically.

htr_selex
  Underlying clean data (htr_selex_analysis):
    Positives  : last-cycle enriched sequences (source=enriched; top-1000 by frequency).
    Negatives  : ZeroCycle background sequences not present in enriched set
                 (source=background; up to 2x positives). Background and enriched are
                 different library types (no-selection control vs post-selection).
  k-mer source : MEME motif files (motif_dir/<protein>_meme/meme.txt), else computed
                 from binding_label pools.
  Top-5 pos    : binding_label=1, contains ≥1 top-K 7-mer; ranked by motif k-mer
                 frequency in positive pool (default), or by last-cycle frequency
                 (--htr_rank_by_last_cycle_frequency + --htr_frequency_dir).
  Top-5 neg    : binding_label=0, contains NO top-K 7-mer; ranked by k-mer frequency
                 in the negative pool. Modal length applied when mixed 26/40 nt libraries.
  Compare      : --htr_compare_ranking writes overlap between motif vs frequency top-5.

ucRBP filtering (--ucrbp_mode)
  Restricts processing to the 23 reproducible ucRBPs identified by the pass/fail
  classifier in Ray & Laverty et al. 2023 (Sci. Rep. 13:5238).
  Pass --ucrbp_whitelist <file> (one protein name per line) for the full list.
  The partial built-in list contains 11 verified names from the paper text.

Output
------
  <output_dir>/<protocol>/<protein>.tsv          per-protein examples
  <output_dir>/<protocol>_summary.tsv            all examples combined
  <output_dir>/<protocol>_stats.json             run statistics

Output columns:
  protein_name, rna_sequence, binding_label, probe_intensity,
  split (positive/negative), rank, matched_kmer, kmer_position,
  kmer_z_score (RNAcompete) or kmer_enrichment_score (RBNS / HTR-SELEX),
  probe_length; RNAcompete rows also include experiment_id, dataset.

References
----------
  RNAcompete Eukarya  : Ray et al. Nature 2013
  RBPZoo (174 RBPs)   : Sasse et al. Nat. Biotechnol. 2025
  ucRBP (23 RBPs)     : Ray & Laverty et al. Sci. Rep. 2023
  RBNS                : Lambert et al. Mol. Cell 2014
  HTR-SELEX           : Jolma et al. Cell 2013; Sasse et al. 2013
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def portable_path(path: Path | str | None) -> str | None:
    """Write repo-relative paths in stats JSON (avoid absolute home directories)."""
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if not p.is_absolute():
        return str(path)
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        try:
            return os.path.relpath(p, base)
        except ValueError:
            continue
    return p.name


# ---------------------------------------------------------------------------
# Known 23 reproducible ucRBPs – Ray & Laverty et al. 2023, Sci. Rep. 13:5238
# Full list is in Supplementary Table S1; partial list verified from paper text.
# Complete via --ucrbp_whitelist or update this list from the supplement.
# ---------------------------------------------------------------------------
UCRBP_23_KNOWN: list[str] = [
    # Class 1 – ribosomal / ribosomal-domain proteins (8)
    # NHP2L1 is L7Ae domain, classified as ribosomal by Ray & Laverty et al. 2023 (Fig. 2)
    "RPL5", "RPL11", "RPL22", "RPL30", "RPS12", "RPS2", "RPS15A", "NHP2L1",
    # Class 2 – non-ribosomal, known RNA-binders (10)
    "RPP25", "PEG10", "SERBP1", "LSM6", "NUDT21", "IFIT2",
    "CNBP", "ZRANB2", "PRPF31", "NUDT16L1",
    # Class 3 – novel bona fide RBPs (5); confirmed from paper main text
    # ILF2 (GC-rich), PURA (GA-rich), SSBP1 (AUG core), GAR1, HARS2
    "GAR1", "PURA", "SSBP1", "HARS2", "ILF2",
    # Source: Ray & Laverty et al. 2023 (Sci. Rep. 13:5238)
    # Experiment IDs: RNCMPT01219, RNCMPT01084, RNCMPT01297, RNCMPT01114, RNCMPT01363,
    #   RNCMPT01071, RNCMPT01740, RNCMPT01411, RNCMPT01299, RNCMPT01313, RNCMPT01267,
    #   RNCMPT00625, RNCMPT01140, RNCMPT01322, RNCMPT00592, RNCMPT01327, RNCMPT01357,
    #   RNCMPT01394, RNCMPT01334, RNCMPT01439, RNCMPT01403, RNCMPT01854, RNCMPT00297
]

IUPAC_RNA = {
    "A": ["A"], "U": ["U"], "G": ["G"], "C": ["C"],
    "R": ["A", "G"], "Y": ["C", "U"], "S": ["G", "C"],
    "W": ["A", "U"], "K": ["G", "U"], "M": ["A", "C"],
    "B": ["C", "G", "U"], "D": ["A", "G", "U"],
    "H": ["A", "C", "U"], "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "U"],
    # DNA aliases (MEME uses T)
    "T": ["U"],
}


# ---------------------------------------------------------------------------
# k-mer utilities
# ---------------------------------------------------------------------------

def normalize_rna_seq(seq: str) -> str:
    return seq.upper().replace("T", "U")


def load_htr_frequency_map(protein: str, freq_dir: Path) -> dict[str, float]:
    """Map RNA sequence → last-cycle mean frequency from enriched_simple table."""
    path = freq_dir / f"{protein}_enriched_simple.tsv"
    if not path.exists():
        return {}
    tbl = pd.read_csv(path, sep="\t")
    if "sequence" not in tbl.columns or "frequency" not in tbl.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in tbl.iterrows():
        seq = normalize_rna_seq(str(row["sequence"]))
        out[seq] = float(row["frequency"])
    return out


def get_kmers(seq: str, k: int) -> list[str]:
    seq = normalize_rna_seq(seq)
    return [seq[i: i + k] for i in range(len(seq) - k + 1)]


def find_first_kmer(seq: str, kmers: list[str]) -> tuple[str, int]:
    """Return (kmer, 0-based position) for the first top-kmer hit in seq."""
    seq_u = seq.upper().replace("T", "U")
    for km in kmers:
        idx = seq_u.find(km)
        if idx != -1:
            return km, idx
    return "-", -1


def contains_any(seq: str, kmer_set: set[str]) -> bool:
    seq_u = seq.upper().replace("T", "U")
    return any(km in seq_u for km in kmer_set)


# ---------------------------------------------------------------------------
# Z-score matrix loader (RNAcompete)
# ---------------------------------------------------------------------------

def load_zscore_matrix(path: Path) -> pd.DataFrame:
    """
    Load the wide-format z-score matrix (rows=7-mer, cols=experiment_id).
    File can be plain TSV or gzip-compressed.
    Returns a DataFrame indexed by kmer.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    print(f"  Z-score matrix: {len(df)} kmers × {len(df.columns)} experiments")
    return df


def get_top_kmers_from_zscore(
    zscore_df: pd.DataFrame,
    exp_id: str,
    top_k: int,
    min_z: float = 2.0,
) -> list[str]:
    """
    Extract top-K 7-mers for one experiment from the z-score matrix.
    Returns kmers sorted by z-score descending, filtered by min_z.
    """
    if exp_id not in zscore_df.columns:
        return []
    col = zscore_df[exp_id].dropna()
    col = col[col >= min_z]
    top = col.nlargest(top_k)
    return list(top.index)


def load_top_kmers_from_kmer_tsv(
    kmer_dir: Path,
    exp_id: str,
    top_k: int,
    min_z: float = 2.0,
) -> tuple[list[str], dict[str, float]]:
    """
    Load official top-K 7-mers from per-experiment file (Eukarya / ucRBP format).
    Files: {RNCMPT}_zscores.tsv with columns kmer, z_setA, z_setB, z_setAB.
    """
    path = kmer_dir / f"{exp_id}_zscores.tsv"
    if not path.exists():
        return [], {}
    tbl = pd.read_csv(path, sep="\t")
    if "kmer" not in tbl.columns:
        return [], {}
    zcol = "z_setAB" if "z_setAB" in tbl.columns else tbl.columns[1]
    sub = tbl[["kmer", zcol]].copy()
    sub[zcol] = pd.to_numeric(sub[zcol], errors="coerce")
    sub = sub.dropna(subset=[zcol])
    sub = sub[sub["kmer"].astype(str).str.len() == 7]
    sub = sub[sub[zcol] >= min_z].nlargest(top_k, zcol)
    if sub.empty:
        return [], {}
    kmers = sub["kmer"].astype(str).tolist()
    enrich = dict(zip(kmers, sub[zcol].astype(float)))
    return kmers, enrich


def resolve_rnacompete_top_kmers(
    exp_id: str,
    zscore_df: pd.DataFrame | None,
    kmer_dir: Path | None,
    top_k: int,
    min_z: float,
) -> tuple[list[str], dict[str, float], str]:
    """Return (top_kmers, score_map, source_tag). source: zscore_matrix | kmer_tsv | none."""
    eid = str(exp_id)
    if zscore_df is not None:
        kmers = get_top_kmers_from_zscore(zscore_df, eid, top_k, min_z)
        if kmers:
            enrich = {
                km: float(zscore_df.loc[km, eid])
                for km in kmers
                if km in zscore_df.index
            }
            return kmers, enrich, "zscore_matrix"
    if kmer_dir is not None:
        kmers, enrich = load_top_kmers_from_kmer_tsv(kmer_dir, eid, top_k, min_z)
        if kmers:
            return kmers, enrich, "kmer_tsv"
    return [], {}, "none"


def mean_top_kmer_score(
    exp_id: str,
    zscore_df: pd.DataFrame | None,
    kmer_dir: Path | None,
    top_k: int,
    min_z: float,
) -> float:
    kmers, enrich, _ = resolve_rnacompete_top_kmers(
        exp_id, zscore_df, kmer_dir, top_k, min_z
    )
    if not kmers:
        return -1.0
    if enrich:
        return float(np.mean([enrich[km] for km in kmers if km in enrich]))
    if zscore_df is not None and exp_id in zscore_df.columns:
        return float(zscore_df[str(exp_id)].nlargest(top_k).mean())
    return -1.0


def infer_kmer_dir(data_path: Path) -> Path | None:
    """If data_file lives under eukarya/ or ucrbp/, use sibling data/kmers/."""
    for parent in [data_path.resolve().parent, *data_path.resolve().parents]:
        if parent.name in ("eukarya", "ucrbp", "rbpzoo"):
            cand = parent / "data" / "kmers"
            if cand.is_dir() and any(cand.glob("*_zscores.tsv")):
                return cand
    return None


# ---------------------------------------------------------------------------
# MEME motif parser (HTR-SELEX)
# ---------------------------------------------------------------------------

def _expand_iupac(seq: str) -> list[str]:
    """Expand IUPAC consensus sequence to all possible RNA sequences."""
    options: list[list[str]] = []
    for ch in seq.upper().replace("T", "U"):
        options.append(IUPAC_RNA.get(ch, [ch]))
    results = [""]
    for choices in options:
        results = [r + c for r in results for c in choices]
    return results


def parse_meme_top_kmers(meme_txt: Path, k: int = 7, top_k: int = 10) -> list[str]:
    """
    Extract top-k 7-mers from a MEME meme.txt file.

    Strategy: read the letter-probability matrix for the first (highest-scoring)
    motif. At each position take the most probable base. Then slide a k-mer window
    over the consensus and collect all unique k-mers; return up to top_k.
    If the motif width < k, fall back to IUPAC consensus expansion.
    """
    if not meme_txt.exists():
        return []

    text = meme_txt.read_text(errors="replace")

    # Find first MOTIF block
    lpm_match = re.search(
        r"letter-probability matrix:[^\n]*\n((?:\s+[0-9. eE+\-]+\n)+)",
        text,
    )
    if not lpm_match:
        # Fall back to IUPAC consensus line
        cons_match = re.search(r"consensus\s+([AUGCTRYWSKMBDHVN]+)", text, re.IGNORECASE)
        if cons_match:
            consensus = cons_match.group(1).upper().replace("T", "U")
            kmers: list[str] = []
            for i in range(len(consensus) - k + 1):
                window = consensus[i: i + k]
                expanded = _expand_iupac(window)
                kmers.extend(expanded)
            return list(dict.fromkeys(kmers))[:top_k]
        return []

    # Parse PPM rows
    rows = []
    for line in lpm_match.group(1).strip().splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) == 4:
            rows.append(vals)  # order: A C G U (MEME default)

    if not rows or len(rows) < k:
        return []

    # Build consensus from PPM (argmax per position: A=0, C=1, G=2, U=3)
    bases = ["A", "C", "G", "U"]
    consensus = "".join(bases[int(np.argmax(row))] for row in rows)

    kmers_out: list[str] = []
    for i in range(len(consensus) - k + 1):
        km = consensus[i: i + k]
        if km not in kmers_out:
            kmers_out.append(km)
        if len(kmers_out) >= top_k:
            break
    return kmers_out


# ---------------------------------------------------------------------------
# Enrichment from data (fallback for RBNS / HTR-SELEX without external files)
# ---------------------------------------------------------------------------

def compute_kmer_enrichment(
    pos_seqs: list[str], neg_seqs: list[str], k: int
) -> dict[str, float]:
    """
    Compute per-kmer enrichment: (freq_pos) / (freq_neg + pseudocount).
    freq = fraction of sequences containing the kmer.
    """
    n_pos = max(len(pos_seqs), 1)
    n_neg = max(len(neg_seqs), 1)

    pos_counts: Counter = Counter()
    for seq in pos_seqs:
        for km in set(get_kmers(seq, k)):
            pos_counts[km] += 1

    neg_counts: Counter = Counter()
    for seq in neg_seqs:
        for km in set(get_kmers(seq, k)):
            neg_counts[km] += 1

    return {
        km: (cnt / n_pos) / ((neg_counts.get(km, 0) + 1) / n_neg)
        for km, cnt in pos_counts.items()
    }


# ---------------------------------------------------------------------------
# Per-protein processing
# ---------------------------------------------------------------------------

def _select_examples(
    df: pd.DataFrame,
    top_kmers: list[str],
    n_examples: int,
    has_intensity: bool,
    protein: str,
    fix_length: bool = False,
    motif_score_col: str = "kmer_enrichment_score",
) -> pd.DataFrame:
    """
    Core selection logic shared across protocols.

    Positives: binding_label==1, contain ≥1 top-kmer, sorted by intensity/score desc.
    Negatives: binding_label==0, contain NO top-kmer, sorted by intensity/score asc
               (or neg_pool_score desc for RBNS/HTR-SELEX if _neg_score column present).

    fix_length: if True (used for RNAcompete where probes vary from 30–41 nt),
                the modal probe length across all top-kmer-containing positives is used
                as the fixed length; all positives and negatives are restricted to that
                length. Maximises the candidate pool vs fixing to the first positive.
                HTR-SELEX (40 nt fixed) and RBNS (20 nt fixed) do not need this flag,
                but it is applied automatically when the source has length variation.
    """
    top_kmer_set = set(top_kmers)
    rows: list[dict] = []

    # ---- positives ----
    cand_pos = df[df["binding_label"] == 1].copy()
    cand_pos = cand_pos[
        cand_pos["rna_sequence"].apply(lambda s: contains_any(s, top_kmer_set))
    ]

    # sort: intensity desc (RNAcompete) or _pos_score desc (RBNS/HTR-SELEX)
    if "_pos_score" in cand_pos.columns:
        cand_pos = cand_pos.sort_values("_pos_score", ascending=False)
    elif has_intensity:
        cand_pos = cand_pos.sort_values("probe_intensity", ascending=False)

    # For RNAcompete: fix length to modal length among top-kmer positives,
    # not the first encountered — this maximises the candidate pool.
    fixed_len: int | None = None
    if fix_length and not cand_pos.empty:
        from collections import Counter as _Counter
        length_counts = _Counter(len(s) for s in cand_pos["rna_sequence"])
        fixed_len = length_counts.most_common(1)[0][0]
        cand_pos = cand_pos[cand_pos["rna_sequence"].str.len() == fixed_len]

    for rank, (_, row) in enumerate(cand_pos.iterrows(), 1):
        seq = row["rna_sequence"]
        if fix_length and fixed_len is not None and len(seq) != fixed_len:
            continue  # safety guard
        matched_km, kpos = find_first_kmer(seq, top_kmers)
        row_out = {
                "protein_name": protein,
                "rna_sequence": seq,
                "binding_label": int(row["binding_label"]),
                "probe_intensity": row.get("probe_intensity", np.nan),
                "R_max": row.get("R_max", np.nan),
                "split": "positive",
                "rank": rank,
                "matched_kmer": matched_km,
                "kmer_position": kpos,
                "probe_length": len(seq),
            }
        row_out[motif_score_col] = row.get("_enrich_score", np.nan)
        rows.append(row_out)
        if len([r for r in rows if r["split"] == "positive"]) >= n_examples:
            break

    # ---- negatives ----
    cand_neg = df[df["binding_label"] == 0].copy()
    cand_neg = cand_neg[
        ~cand_neg["rna_sequence"].apply(lambda s: contains_any(s, top_kmer_set))
    ]

    # Apply same length constraint to negatives
    if fix_length and fixed_len is not None:
        cand_neg = cand_neg[cand_neg["rna_sequence"].str.len() == fixed_len]

    if "_neg_score" in cand_neg.columns:
        cand_neg = cand_neg.sort_values("_neg_score", ascending=False)
    elif has_intensity:
        cand_neg = cand_neg.sort_values("probe_intensity", ascending=True)

    for rank, (_, row) in enumerate(cand_neg.head(n_examples).iterrows(), 1):
        row_out = {
                "protein_name": protein,
                "rna_sequence": row["rna_sequence"],
                "binding_label": int(row["binding_label"]),
                "probe_intensity": row.get("probe_intensity", np.nan),
                "R_max": row.get("R_max", np.nan),
                "split": "negative",
                "rank": rank,
                "matched_kmer": "-",
                "kmer_position": -1,
                "probe_length": len(row["rna_sequence"]),
            }
        row_out[motif_score_col] = np.nan
        rows.append(row_out)

    return pd.DataFrame(rows)


def process_rnacompete(
    df: pd.DataFrame,
    protein: str,
    zscore_df: pd.DataFrame | None,
    kmer_dir: Path | None,
    top_k: int,
    n_examples: int,
    kmer_len: int,
    min_z: float,
) -> pd.DataFrame:
    # For proteins with multiple experiments, restrict to the best one:
    # - pick experiment with highest mean Z across top-K 7-mers (matrix or per-experiment TSV)
    # - otherwise: pick experiment with highest mean positive probe_intensity
    id_col = next(
        (c for c in ("experiment_id", "hyb_id") if c in df.columns), None
    )
    if id_col and df[id_col].nunique() > 1:
        best_eid, best_score = None, -1.0
        for eid in df[id_col].dropna().unique():
            score = mean_top_kmer_score(str(eid), zscore_df, kmer_dir, top_k, min_z)
            if score > best_score:
                best_score, best_eid = score, str(eid)
        if best_eid and best_score >= 0:
            df = df[df[id_col] == best_eid].copy()
        else:
            has_int = "probe_intensity" in df.columns
            if has_int:
                df["probe_intensity"] = pd.to_numeric(df["probe_intensity"], errors="coerce")
                best_eid = (
                    df[df["binding_label"] == 1]
                    .groupby(id_col)["probe_intensity"]
                    .mean()
                    .idxmax()
                )
                df = df[df[id_col] == best_eid].copy()

    has_intensity = "probe_intensity" in df.columns and not df["probe_intensity"].isna().all()

    top_kmers: list[str] = []
    enrich_map: dict[str, float] = {}
    kmer_source = "none"

    if id_col:
        for eid in df[id_col].dropna().unique():
            top_kmers, enrich_map, kmer_source = resolve_rnacompete_top_kmers(
                str(eid), zscore_df, kmer_dir, top_k, min_z
            )
            if top_kmers:
                break

    if not top_kmers:
        eids = (
            [str(e) for e in df[id_col].dropna().unique()]
            if id_col
            else ["?"]
        )
        print(
            f"  [WARN] {protein}: no Z-scores for {eids} "
            f"(zscore_file / kmer_dir) — label-pool enrichment fallback",
            file=sys.stderr,
        )
        pos_pool = df[df["binding_label"] == 1]["rna_sequence"].tolist()
        neg_pool = df[df["binding_label"] == 0]["rna_sequence"].tolist()
        if not pos_pool:
            return pd.DataFrame()
        enrichments = compute_kmer_enrichment(pos_pool, neg_pool, kmer_len)
        top_kmers = sorted(enrichments, key=lambda x: enrichments[x], reverse=True)[:top_k]
        enrich_map = enrichments
        kmer_source = "label_pool_fallback"

    if not top_kmers:
        return pd.DataFrame()

    # Attach per-sequence enrichment score (max Z/enrichment of any top-kmer hit)
    top_kmer_set_local = set(top_kmers)
    df = df.copy()
    if enrich_map:
        df["_enrich_score"] = df["rna_sequence"].apply(
            lambda s: max(
                (enrich_map.get(km, 0.0)
                 for km in get_kmers(s, kmer_len)
                 if km in top_kmer_set_local),
                default=0.0,
            )
        )
    df.attrs["kmer_source"] = kmer_source

    result = _select_examples(
        df, top_kmers, n_examples, has_intensity, protein,
        fix_length=True, motif_score_col="kmer_z_score",
    )
    if id_col and not result.empty:
        eid = str(df[id_col].iloc[0])
        result = result.copy()
        result["experiment_id"] = eid
    return result


def process_rbns(
    df: pd.DataFrame,
    protein: str,
    top_k: int,
    n_examples: int,
    kmer_len: int,
) -> pd.DataFrame:
    """
    RBNS protocol.
    Uses R_max (if present) for positive ranking, computed enrichment for kmer selection.
    """
    pos_pool = df[df["binding_label"] == 1]["rna_sequence"].tolist()
    neg_pool = df[df["binding_label"] == 0]["rna_sequence"].tolist()
    if not pos_pool:
        return pd.DataFrame()

    enrichments = compute_kmer_enrichment(pos_pool, neg_pool, kmer_len)
    top_kmers = sorted(enrichments, key=lambda x: enrichments[x], reverse=True)[:top_k]
    if not top_kmers:
        return pd.DataFrame()

    top_kmer_set = set(top_kmers)

    # Positive pool frequency of top kmers (for ranking)
    pos_kf: Counter = Counter()
    for seq in pos_pool:
        for km in set(get_kmers(seq, kmer_len)):
            pos_kf[km] += 1

    # Negative pool frequency (for ranking negatives)
    neg_kf: Counter = Counter()
    for seq in neg_pool:
        for km in set(get_kmers(seq, kmer_len)):
            neg_kf[km] += 1

    df = df.copy()

    # Positive score: R_max (primary) then top-kmer pool frequency (secondary)
    has_rmax = "R_max" in df.columns and df["R_max"].notna().any()
    if has_rmax:
        df["_pos_score"] = pd.to_numeric(df["R_max"], errors="coerce").fillna(0.0)
    else:
        df["_pos_score"] = df["rna_sequence"].apply(
            lambda s: sum(pos_kf[km] for km in get_kmers(s, kmer_len) if km in top_kmer_set)
        )

    # Negative score: total kmer frequency in negative pool (amplifiable but non-binding)
    df["_neg_score"] = df["rna_sequence"].apply(
        lambda s: sum(neg_kf[km] for km in get_kmers(s, kmer_len))
    )

    df["_enrich_score"] = df["rna_sequence"].apply(
        lambda s: max((enrichments.get(km, 0.0) for km in get_kmers(s, kmer_len)), default=0.0)
    )

    return _select_examples(df, top_kmers, n_examples, has_intensity=False, protein=protein)


def process_htr_selex(
    df: pd.DataFrame,
    protein: str,
    motif_dir: Path | None,
    top_k: int,
    n_examples: int,
    kmer_len: int,
    rank_by_last_cycle: bool = False,
    freq_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    HTR-SELEX protocol.
    Uses MEME meme.txt (if motif_dir provided) for top kmers, else computes from data.

    Positive ranking (after top-kmer filter):
      default: motif representativeness (_pos_score = sum of top-kmer pool frequencies)
      rank_by_last_cycle: per-sequence last-cycle frequency from freq_map
    """
    top_kmers: list[str] = []

    if motif_dir is not None:
        meme_txt = motif_dir / f"{protein}_meme" / "meme.txt"
        top_kmers = parse_meme_top_kmers(meme_txt, k=kmer_len, top_k=top_k)

    if not top_kmers:
        pos_pool = df[df["binding_label"] == 1]["rna_sequence"].tolist()
        neg_pool = df[df["binding_label"] == 0]["rna_sequence"].tolist()
        if not pos_pool:
            return pd.DataFrame()
        enrichments = compute_kmer_enrichment(pos_pool, neg_pool, kmer_len)
        top_kmers = sorted(enrichments, key=lambda x: enrichments[x], reverse=True)[:top_k]
        enrich_map = enrichments
    else:
        pos_pool = df[df["binding_label"] == 1]["rna_sequence"].tolist()
        neg_pool = df[df["binding_label"] == 0]["rna_sequence"].tolist()
        enrich_map = compute_kmer_enrichment(pos_pool, neg_pool, kmer_len) if pos_pool else {}

    if not top_kmers:
        return pd.DataFrame()

    top_kmer_set = set(top_kmers)

    # Score positives by top-kmer frequency in positive pool
    pos_kf: Counter = Counter()
    for seq in pos_pool:
        for km in set(get_kmers(seq, kmer_len)):
            pos_kf[km] += 1

    neg_kf: Counter = Counter()
    for seq in neg_pool:
        for km in set(get_kmers(seq, kmer_len)):
            neg_kf[km] += 1

    df = df.copy()
    if rank_by_last_cycle and freq_map:
        df["_pos_score"] = df["rna_sequence"].map(
            lambda s: freq_map.get(normalize_rna_seq(s), 0.0)
        )
        df["last_cycle_frequency"] = df["_pos_score"]
    else:
        df["_pos_score"] = df["rna_sequence"].apply(
            lambda s: sum(pos_kf[km] for km in get_kmers(s, kmer_len) if km in top_kmer_set)
        )
    df["_neg_score"] = df["rna_sequence"].apply(
        lambda s: sum(neg_kf[km] for km in get_kmers(s, kmer_len))
    )
    df["_enrich_score"] = df["rna_sequence"].apply(
        lambda s: max((enrich_map.get(km, 0.0) for km in get_kmers(s, kmer_len)), default=0.0)
    )

    # Apply modal-length normalisation: guards against datasets where the same
    # protein has sequences of different lengths (e.g. mixed 26/40 nt libraries).
    has_length_variation = df["rna_sequence"].str.len().nunique() > 1
    return _select_examples(
        df, top_kmers, n_examples,
        has_intensity=False, protein=protein,
        fix_length=has_length_variation,
    )


def compare_htr_positive_rankings(
    motif_result: pd.DataFrame,
    freq_result: pd.DataFrame,
    protein: str,
) -> dict:
    """Compare top-5 positive sequences between two HTR ranking modes."""
    motif_seqs = motif_result.loc[motif_result["split"] == "positive", "rna_sequence"].tolist()
    freq_seqs = freq_result.loc[freq_result["split"] == "positive", "rna_sequence"].tolist()
    motif_set = set(motif_seqs)
    freq_set = set(freq_seqs)
    overlap = motif_set & freq_set
    union = motif_set | freq_set
    return {
        "protein_name": protein,
        "n_motif_top5": len(motif_seqs),
        "n_freq_top5": len(freq_seqs),
        "n_overlap": len(overlap),
        "overlap_fraction": len(overlap) / len(motif_seqs) if motif_seqs else float("nan"),
        "jaccard": len(overlap) / len(union) if union else float("nan"),
        "motif_only": ";".join(sorted(motif_set - freq_set)),
        "frequency_only": ";".join(sorted(freq_set - motif_set)),
        "both": ";".join(sorted(overlap)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_file", required=True,
        help=(
            "Input TSV with columns: rna_sequence, binding_label, and one of "
            "protein_name / target_name. "
            "Optional: probe_intensity (RNAcompete), experiment_id (RNAcompete), "
            "R_max / high_confidence (RBNS), dataset_source."
        ),
    )
    p.add_argument(
        "--protocol", required=True,
        choices=["rnacompete", "rbns", "htr_selex"],
        help="Dataset protocol — determines kmer source and ranking strategy.",
    )
    p.add_argument(
        "--output_dir", default="results/top_bottom_examples",
        help="Root output directory (default: results/top_bottom_examples).",
    )
    # RNAcompete-specific
    p.add_argument(
        "--zscore_file", default=None,
        help=(
            "[rnacompete] Path to Z-score matrix TSV[.gz] "
            "(rows=7-mer, cols=experiment_id). "
            "E.g. rnacompete_analysis/data/raw/Zscores_RNAcompete2025.txt.gz"
        ),
    )
    p.add_argument(
        "--kmer_dir", default=None,
        help=(
            "[rnacompete] Directory with per-experiment {RNCMPT}_zscores.tsv "
            "(Eukarya/ucRBP legacy IDs). Auto-inferred from --data_file path when omitted."
        ),
    )
    p.add_argument(
        "--min_z", type=float, default=2.0,
        help="[rnacompete] Minimum Z-score threshold for enriched kmers (default: 2.0).",
    )
    # HTR-SELEX-specific
    p.add_argument(
        "--motif_dir", default=None,
        help=(
            "[htr_selex] Directory containing per-protein MEME subdirs "
            "(<protein>_meme/meme.txt). "
            "E.g. htr_selex_analysis/results/motifs/"
        ),
    )
    p.add_argument(
        "--htr_frequency_dir",
        default="../htr_selex_analysis/results/tables",
        help=(
            "[htr_selex] Directory with per-protein {protein}_enriched_simple.tsv "
            "(last-cycle frequency tables)."
        ),
    )
    p.add_argument(
        "--htr_rank_by_last_cycle_frequency",
        action="store_true",
        help=(
            "[htr_selex] Rank motif-filtered positives by last-cycle frequency "
            "instead of motif representativeness in the positive pool."
        ),
    )
    p.add_argument(
        "--htr_compare_ranking",
        action="store_true",
        help=(
            "[htr_selex] Run both ranking modes and write htr_selex_ranking_overlap.tsv "
            "(output files still use the selected ranking mode)."
        ),
    )
    # Common
    p.add_argument("--top_k_kmers", type=int, default=10,
                   help="Number of top enriched k-mers to use as motif anchors (default: 10).")
    p.add_argument("--n_examples", type=int, default=5,
                   help="Positive and negative examples per protein (default: 5).")
    p.add_argument("--kmer_len", type=int, default=7,
                   help="k-mer length (default: 7).")
    p.add_argument("--filter_source", default=None,
                   help="Filter rows to dataset_source == this value before processing.")
    p.add_argument("--dataset_label", default=None,
                   help="[rnacompete] Dataset tag for master merge (e.g. RNAcompete_Eukarya).")
    p.add_argument("--ucrbp_mode", action="store_true",
                   help="Restrict to 23 reproducible ucRBPs (Ray & Laverty et al. 2023).")
    p.add_argument("--ucrbp_whitelist", default=None,
                   help="File with one ucRBP protein name per line (overrides built-in list).")
    p.add_argument("--high_confidence_only", action="store_true",
                   help="[rbns] Keep only rows where high_confidence==True.")
    p.add_argument("--min_pos", type=int, default=20,
                   help="Min positive sequences required per protein (default: 20).")
    p.add_argument("--min_neg", type=int, default=20,
                   help="Min negative sequences required per protein (default: 20).")
    p.add_argument("--max_proteins", type=int, default=None,
                   help="Process at most N proteins (for testing).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        sys.exit(f"ERROR: --data_file not found: {data_path}")

    if args.protocol == "rnacompete" and args.dataset_label:
        slug = args.dataset_label.lower().replace("rnacompete_", "rnacompete_")
        out_dir = Path(args.output_dir) / slug
    else:
        out_dir = Path(args.output_dir) / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading: {data_path} ...", end=" ", flush=True)
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    print(f"{len(df):,} rows")

    # Normalise protein_name column (RBNS uses target_name)
    if "protein_name" not in df.columns and "target_name" in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})
        print("  Renamed 'target_name' → 'protein_name'")

    required = {"protein_name", "rna_sequence", "binding_label"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing required columns: {missing}")

    df["binding_label"] = df["binding_label"].astype(int)
    df["rna_sequence"] = df["rna_sequence"].str.upper().str.replace("T", "U", regex=False)

    # ── Optional: filter dataset_source ───────────────────────────────────
    if args.filter_source and "dataset_source" in df.columns:
        df = df[df["dataset_source"] == args.filter_source]
        print(f"  Filtered to dataset_source='{args.filter_source}': {len(df):,} rows")

    # ── Optional: RBNS high_confidence filter ─────────────────────────────
    if args.high_confidence_only and "high_confidence" in df.columns:
        # Keep negatives always; filter only positives
        pos_mask = df["binding_label"] == 1
        df = pd.concat([
            df[pos_mask & (df["high_confidence"].astype(str).str.lower() == "true")],
            df[~pos_mask],
        ], ignore_index=True)
        print(f"  After high_confidence filter: {len(df):,} rows")

    # ── Load Z-score matrix (RNAcompete) ───────────────────────────────────
    zscore_df: pd.DataFrame | None = None
    kmer_dir: Path | None = None
    if args.protocol == "rnacompete":
        if args.kmer_dir:
            kmer_dir = Path(args.kmer_dir)
            if not kmer_dir.is_dir():
                sys.exit(f"ERROR: --kmer_dir not found: {kmer_dir}")
            print(f"K-mer Z-scores: {kmer_dir} (per-experiment TSV)")
        else:
            inferred = infer_kmer_dir(data_path)
            if inferred is not None:
                kmer_dir = inferred
                print(f"K-mer Z-scores: {kmer_dir} (auto from data path)")
        if args.zscore_file:
            zpath = Path(args.zscore_file)
            if zpath.exists():
                print(f"Loading z-scores: {zpath}")
                zscore_df = load_zscore_matrix(zpath)
            else:
                print(f"  [WARN] --zscore_file not found: {zpath}")

    # ── ucRBP whitelist ────────────────────────────────────────────────────
    ucrbp_allowed: set[str] | None = None
    if args.ucrbp_mode:
        if args.ucrbp_whitelist:
            wl = Path(args.ucrbp_whitelist)
            if not wl.exists():
                sys.exit(f"ERROR: --ucrbp_whitelist not found: {wl}")
            ucrbp_allowed = set(wl.read_text().strip().splitlines())
            print(f"ucRBP whitelist: {len(ucrbp_allowed)} proteins")
        else:
            ucrbp_allowed = set(UCRBP_23_KNOWN)
            print(
                f"ucRBP mode: built-in partial list ({len(ucrbp_allowed)} proteins).\n"
                "  TIP: pass --ucrbp_whitelist to use the full 23 from Supplementary Table S1."
            )

    # ── MEME motif dir (HTR-SELEX) ─────────────────────────────────────────
    motif_dir: Path | None = None
    htr_freq_dir: Path | None = None
    if args.protocol == "htr_selex":
        if args.motif_dir:
            motif_dir = Path(args.motif_dir)
            if not motif_dir.exists():
                print(f"  [WARN] --motif_dir not found: {motif_dir} — will compute from data")
                motif_dir = None
            else:
                print(f"MEME motif dir: {motif_dir}")
        htr_freq_dir = Path(args.htr_frequency_dir)
        if args.htr_rank_by_last_cycle_frequency or args.htr_compare_ranking:
            if not htr_freq_dir.exists():
                sys.exit(f"ERROR: --htr_frequency_dir not found: {htr_freq_dir}")
            print(f"HTR frequency dir: {htr_freq_dir}")
        if args.htr_rank_by_last_cycle_frequency:
            print("HTR positive ranking: last-cycle frequency")
        elif args.htr_compare_ranking:
            print("HTR positive ranking: motif (default output) + compare overlap")
        else:
            print("HTR positive ranking: motif representativeness")

    # ── Per-protein loop ───────────────────────────────────────────────────
    proteins = sorted(df["protein_name"].unique())
    if args.max_proteins:
        proteins = proteins[: args.max_proteins]

    print(f"\nProtocol      : {args.protocol}")
    print(f"Proteins      : {len(proteins)}")
    print(f"top_k_kmers   : {args.top_k_kmers}  |  n_examples: {args.n_examples}  |  k={args.kmer_len}")
    print("-" * 64)

    all_results: list[pd.DataFrame] = []
    skipped: list[str] = []
    stats: list[dict] = []
    ranking_overlaps: list[dict] = []

    for prot in proteins:
        if ucrbp_allowed is not None and prot not in ucrbp_allowed:
            skipped.append(prot)
            continue

        sub = df[df["protein_name"] == prot].copy()
        n_pos = int((sub["binding_label"] == 1).sum())
        n_neg = int((sub["binding_label"] == 0).sum())

        if n_pos < args.min_pos or n_neg < args.min_neg:
            print(f"  SKIP {prot:40s}  pos={n_pos}  neg={n_neg}  (below threshold)")
            skipped.append(prot)
            continue

        if args.protocol == "rnacompete":
            result = process_rnacompete(
                sub, prot, zscore_df, kmer_dir, args.top_k_kmers, args.n_examples,
                args.kmer_len, args.min_z,
            )
        elif args.protocol == "rbns":
            result = process_rbns(sub, prot, args.top_k_kmers, args.n_examples, args.kmer_len)
        else:
            freq_map = (
                load_htr_frequency_map(prot, htr_freq_dir)
                if htr_freq_dir is not None
                else None
            )
            if args.htr_compare_ranking:
                motif_result = process_htr_selex(
                    sub, prot, motif_dir, args.top_k_kmers, args.n_examples, args.kmer_len,
                    rank_by_last_cycle=False, freq_map=freq_map,
                )
                freq_result = process_htr_selex(
                    sub, prot, motif_dir, args.top_k_kmers, args.n_examples, args.kmer_len,
                    rank_by_last_cycle=True, freq_map=freq_map,
                )
                if not motif_result.empty and not freq_result.empty:
                    ranking_overlaps.append(
                        compare_htr_positive_rankings(motif_result, freq_result, prot)
                    )
                result = (
                    freq_result
                    if args.htr_rank_by_last_cycle_frequency
                    else motif_result
                )
            else:
                result = process_htr_selex(
                    sub, prot, motif_dir, args.top_k_kmers, args.n_examples, args.kmer_len,
                    rank_by_last_cycle=args.htr_rank_by_last_cycle_frequency,
                    freq_map=freq_map,
                )

        if result.empty:
            print(f"  SKIP {prot:40s}  (no valid examples)")
            skipped.append(prot)
            continue

        if args.dataset_label:
            result = result.copy()
            result["dataset"] = args.dataset_label

        n_out_pos = int((result["split"] == "positive").sum())
        n_out_neg = int((result["split"] == "negative").sum())
        print(f"  OK   {prot:40s}  pos={n_out_pos}  neg={n_out_neg}")

        prot_file = out_dir / f"{prot}.tsv"
        result.to_csv(prot_file, sep="\t", index=False)
        all_results.append(result)
        stats.append(
            {"protein_name": prot, "n_input_pos": n_pos, "n_input_neg": n_neg,
             "n_output_pos": n_out_pos, "n_output_neg": n_out_neg}
        )

    # ── Write summaries ────────────────────────────────────────────────────
    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_path = Path(args.output_dir) / f"{out_dir.name}_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"\nSummary TSV  : {summary_path}  ({len(summary_df):,} rows)")

        stats_path = Path(args.output_dir) / f"{args.protocol}_stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "protocol": args.protocol,
                    "data_file": portable_path(data_path),
                    "zscore_file": portable_path(args.zscore_file),
                    "kmer_dir": portable_path(kmer_dir) if kmer_dir else None,
                    "dataset_label": args.dataset_label,
                    "motif_dir": portable_path(args.motif_dir),
                    "htr_frequency_dir": portable_path(args.htr_frequency_dir)
                    if args.protocol == "htr_selex"
                    else None,
                    "htr_rank_by_last_cycle_frequency": args.htr_rank_by_last_cycle_frequency,
                    "htr_compare_ranking": args.htr_compare_ranking,
                    "n_proteins_processed": len(stats),
                    "n_proteins_skipped": len(skipped),
                    "top_k_kmers": args.top_k_kmers,
                    "n_examples": args.n_examples,
                    "kmer_len": args.kmer_len,
                    "proteins": stats,
                },
                indent=2,
            )
        )
        print(f"Stats JSON   : {stats_path}")

        if ranking_overlaps:
            overlap_df = pd.DataFrame(ranking_overlaps)
            overlap_path = Path(args.output_dir) / "htr_selex_ranking_overlap.tsv"
            overlap_df.to_csv(overlap_path, sep="\t", index=False)
            med = overlap_df["overlap_fraction"].median()
            print(
                f"Ranking overlap: {overlap_path}  "
                f"(median top-5 overlap = {med:.2f}, n={len(overlap_df)} proteins)"
            )
    else:
        print("\nWARN: no results — check input data and filters.")

    if skipped:
        shown = skipped[:10]
        suffix = " ..." if len(skipped) > 10 else ""
        print(f"\nSkipped ({len(skipped)}): {', '.join(shown)}{suffix}")

    # ── Optional: merge all protocol summaries into a single master TSV ───────
    root = Path(args.output_dir)
    summary_files = sorted(
        p for p in root.glob("*_summary.tsv")
        if p.name != "all_protocols_summary.tsv"
        and p.name != "rnacompete_summary.tsv"  # superseded by rnacompete_* panels
    )
    if len(summary_files) > 1:
        frames = []
        for p in summary_files:
            tmp = pd.read_csv(p, sep="\t")
            stem = p.stem.replace("_summary", "")
            if stem.startswith("rnacompete_"):
                tmp["protocol"] = "rnacompete"
                if "dataset" not in tmp.columns:
                    panel = stem.replace("rnacompete_", "")
                    name_map = {
                        "eukarya": "RNAcompete_Eukarya",
                        "rbpzoo": "RNAcompete_RBPZoo",
                        "ucrbp23": "RNAcompete_ucRBP23",
                    }
                    tmp["dataset"] = name_map.get(panel, f"RNAcompete_{panel}")
            else:
                tmp["protocol"] = stem
                if "dataset" not in tmp.columns:
                    tmp["dataset"] = {"rbns": "RBNS", "htr_selex": "HTR-SELEX"}.get(stem, stem)
            cols = ["protocol", "dataset"] + [c for c in tmp.columns if c not in ("protocol", "dataset")]
            frames.append(tmp[cols])
        master = pd.concat(frames, ignore_index=True)
        master_path = root / "all_protocols_summary.tsv"
        master.to_csv(master_path, sep="\t", index=False)
        print(
            f"Master TSV   : {master_path}  ({len(master):,} rows, "
            f"{master['protocol'].nunique()} protocols)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
