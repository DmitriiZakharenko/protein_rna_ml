"""
Script 13: Download and process RNAInter protein–RNA interaction dataset.

RNAInter (http://www.rnainter.org) is a curated database of RNA–protein
interactions with both positive and negative examples — already formatted
for machine learning tasks.

What we download:
  - Human protein–RNA interactions (mRNA, lncRNA, miRNA targets)
  - Labelled positives (experimentally confirmed binding)
  - Labelled negatives (confirmed non-binding or random negatives from RNAInter)
  - Filter: only entries with both protein and RNA sequences available

Download options (in order of preference):
  1. RNAInter download API: https://www.rnainter.org/download/
  2. Pre-built ML dataset: RNAInter provides positive/negative sets for several models
  3. Fallback: scrape interaction table for Homo sapiens entries

Output:
  data/rnainter/rnainter_human.tsv — all human interactions with sequences

Usage:
    python scripts/13_download_rnainter.py
    python scripts/13_download_rnainter.py --max_per_protein 500
"""

import argparse
import gzip
import io
import json
import os
import re
import sys
import time

import pandas as pd
import requests

HEADERS_JSON = {"Accept": "application/json"}
HEADERS_TEXT = {"Accept": "text/plain"}
UNIPROT_BASE = "https://rest.uniprot.org"
MAX_RETRIES  = 3


# ── RNAInter download ─────────────────────────────────────────────────────────
def download_rnainter(out_dir: str) -> str | None:
    """
    Attempt to download the RNAInter human dataset.
    RNAInter provides a downloadable table at:
      https://www.rnainter.org/download/
    Returns path to downloaded file, or None if failed.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "rnainter_raw.tsv.gz")

    if os.path.exists(raw_path):
        print(f"  RNAInter raw file already exists: {raw_path}")
        return raw_path

    # RNAInter download URLs (check the download page for current links)
    candidate_urls = [
        "http://www.rnainter.org/download/download/rnainter_homo_sapiens.txt.gz",
        "http://www.rnainter.org/download/RNAInter_homo_sapiens.txt.gz",
        "https://www.rnainter.org/rnainterdb/download/RNAInter_v4.0_homo_sapiens.txt.gz",
        "http://www.rnainter.org/download/RNAInter_v4.0.txt.gz",
    ]

    for url in candidate_urls:
        print(f"  Trying: {url}")
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                size = 0
                with open(raw_path, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                        size += len(chunk)
                print(f"  Downloaded: {size/1e6:.1f} MB → {raw_path}")
                return raw_path
            else:
                print(f"    HTTP {r.status_code}")
        except Exception as e:
            print(f"    Failed: {e}")

    print("\n  ⚠️  All direct download URLs failed.")
    print("  Please download manually from http://www.rnainter.org/download/")
    print("  and save as: data/rnainter/rnainter_raw.tsv.gz")
    print("  Then re-run this script.\n")
    return None


def parse_rnainter(raw_path: str) -> pd.DataFrame:
    """
    Parse RNAInter TSV file.
    Expected columns (RNAInter v4.0 format):
      Interactor1_ID | Interactor1_type | Interactor1_name | Interactor1_species
      Interactor2_ID | Interactor2_type | Interactor2_name | Interactor2_species
      Interaction_score | Evidence | ...
    """
    try:
        if raw_path.endswith(".gz"):
            with gzip.open(raw_path, "rt", encoding="utf-8", errors="replace") as f:
                df = pd.read_csv(f, sep="\t", low_memory=False)
        else:
            df = pd.read_csv(raw_path, sep="\t", low_memory=False)
    except Exception as e:
        print(f"  ⚠️  Parse error: {e}")
        return pd.DataFrame()

    print(f"  Raw rows: {len(df):,}  Columns: {list(df.columns[:10])}")
    return df


def extract_protein_rna_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract protein–RNA pairs from RNAInter table.
    Handles flexible column naming across RNAInter versions.
    """
    if df.empty:
        return df

    cols = list(df.columns)
    print(f"  All columns: {cols}")

    # Detect column layout
    # Version A: Interactor1/2 style
    # Version B: protein_id / RNA_id style
    id1_col  = next((c for c in cols if "interactor1" in c.lower() and "id" in c.lower()), None)
    id2_col  = next((c for c in cols if "interactor2" in c.lower() and "id" in c.lower()), None)
    typ1_col = next((c for c in cols if "interactor1" in c.lower() and "type" in c.lower()), None)
    typ2_col = next((c for c in cols if "interactor2" in c.lower() and "type" in c.lower()), None)
    sp1_col  = next((c for c in cols if "interactor1" in c.lower() and "species" in c.lower()), None)
    sp2_col  = next((c for c in cols if "interactor2" in c.lower() and "species" in c.lower()), None)
    score_col= next((c for c in cols if "score" in c.lower()), None)
    label_col= next((c for c in cols if "label" in c.lower() or "interaction" in c.lower()), None)

    if id1_col is None:
        print("  ⚠️  Could not detect standard RNAInter column layout")
        print(f"  Columns: {cols}")
        return pd.DataFrame()

    # Filter: Homo sapiens only
    mask_human = pd.Series([True] * len(df))
    if sp1_col and sp2_col:
        mask_human = (df[sp1_col].str.contains("sapiens", case=False, na=False) |
                      df[sp2_col].str.contains("sapiens", case=False, na=False))
    df_human = df[mask_human].copy()
    print(f"  Human interactions: {len(df_human):,}")

    # Filter: one interactor is protein, other is RNA
    mask_prot_rna = pd.Series([False] * len(df_human))
    if typ1_col and typ2_col:
        protein_terms = {"protein", "rbp"}
        rna_terms     = {"mrna", "lncrna", "ncrna", "mirna", "snrna", "rrna", "trna", "rna"}
        def is_protein(t):
            return str(t).lower().strip() in protein_terms
        def is_rna(t):
            return any(x in str(t).lower() for x in rna_terms)
        mask_prot_rna = (
            (df_human[typ1_col].apply(is_protein) & df_human[typ2_col].apply(is_rna)) |
            (df_human[typ2_col].apply(is_protein) & df_human[typ1_col].apply(is_rna))
        )
    df_pr = df_human[mask_prot_rna].copy()
    print(f"  Protein–RNA pairs: {len(df_pr):,}")

    return df_pr


# ── Sequence fetching ─────────────────────────────────────────────────────────
def fetch_uniprot_sequence(uniprot_id: str) -> str | None:
    """Fetch canonical protein sequence from UniProt."""
    uid = uniprot_id.strip().split("|")[-1].split(".")[0]
    url = f"{UNIPROT_BASE}/uniprotkb/{uid}.fasta"
    try:
        r = requests.get(url, headers=HEADERS_TEXT, timeout=15)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split("\n")
        return "".join(l for l in lines if not l.startswith(">")) or None
    except Exception:
        return None


def fetch_rna_sequence_from_ensembl(ensembl_id: str) -> str | None:
    """Fetch RNA sequence from Ensembl by transcript ID."""
    tid = ensembl_id.strip().split(".")[0]  # strip version suffix
    if not tid.startswith("ENS"):
        return None
    url = f"https://rest.ensembl.org/sequence/id/{tid}?content-type=text/plain&type=cdna"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        seq = r.text.strip().upper().replace("T", "U")
        return seq if len(seq) > 10 else None
    except Exception:
        return None


def fetch_rna_from_ncbi(ncbi_id: str) -> str | None:
    """Fetch RNA sequence from NCBI Entrez (for non-Ensembl IDs)."""
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           f"?db=nuccore&id={ncbi_id}&rettype=fasta&retmode=text")
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split("\n")
        seq = "".join(l for l in lines if not l.startswith(">"))
        seq = seq.upper().replace("T", "U")
        return seq if len(seq) > 10 else None
    except Exception:
        return None


def window_rna(seq: str, window: int = 60, step: int = 60) -> list[str]:
    """Split a long RNA into non-overlapping windows for training."""
    windows = []
    for i in range(0, len(seq) - window + 1, step):
        windows.append(seq[i:i + window])
    return windows if windows else [seq[:window]]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",           default="data/rnainter")
    parser.add_argument("--max_per_protein",   type=int, default=500,
                        help="Max positive+negative pairs per protein (for balance)")
    parser.add_argument("--rna_window",        type=int, default=60,
                        help="Window size for long RNA sequences")
    parser.add_argument("--skip_download",     action="store_true",
                        help="Skip download, use existing raw file")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    print("\n=== RNAInter Download ===")
    raw_path = None
    if not args.skip_download:
        raw_path = download_rnainter(args.out_dir)
    else:
        # Try to find existing file
        for fname in os.listdir(args.out_dir):
            if "rnainter" in fname.lower() and fname.endswith((".gz", ".tsv", ".txt")):
                raw_path = os.path.join(args.out_dir, fname)
                print(f"  Using existing file: {raw_path}")
                break

    if raw_path is None:
        print("\n❌  No RNAInter file available. Please download manually:")
        print("   1. Go to http://www.rnainter.org/download/")
        print("   2. Download 'Homo sapiens' interactions")
        print(f"   3. Save as: {args.out_dir}/rnainter_raw.tsv.gz")
        print("   4. Re-run: python scripts/13_download_rnainter.py --skip_download")
        sys.exit(0)

    # ── Step 2: Parse ─────────────────────────────────────────────────────────
    print("\n=== Parsing RNAInter ===")
    df_raw = parse_rnainter(raw_path)
    if df_raw.empty:
        print("  ❌  Could not parse file — check format")
        sys.exit(1)

    df_pr = extract_protein_rna_pairs(df_raw)
    if df_pr.empty:
        print("  ❌  No protein–RNA pairs found — check column names")
        sys.exit(1)

    # ── Step 3: Fetch sequences ───────────────────────────────────────────────
    print("\n=== Fetching sequences ===")
    cols = list(df_pr.columns)
    id1_col  = next(c for c in cols if "interactor1" in c.lower() and "id" in c.lower())
    id2_col  = next(c for c in cols if "interactor2" in c.lower() and "id" in c.lower())
    typ1_col = next(c for c in cols if "interactor1" in c.lower() and "type" in c.lower())
    typ2_col = next(c for c in cols if "interactor2" in c.lower() and "type" in c.lower())
    name1_col= next((c for c in cols if "interactor1" in c.lower() and "name" in c.lower()), id1_col)
    name2_col= next((c for c in cols if "interactor2" in c.lower() and "name" in c.lower()), id2_col)
    label_col= next((c for c in cols if "label" in c.lower()), None)

    # Cache sequences to avoid redundant API calls
    prot_seq_cache = {}
    rna_seq_cache  = {}

    records = []
    skipped_no_prot = 0
    skipped_no_rna  = 0

    for _, row in df_pr.iterrows():
        # Determine which interactor is protein, which is RNA
        if "protein" in str(row[typ1_col]).lower():
            prot_id = str(row[id1_col]).strip()
            prot_name = str(row[name1_col]).strip()
            rna_id  = str(row[id2_col]).strip()
            rna_name= str(row[name2_col]).strip()
        else:
            prot_id = str(row[id2_col]).strip()
            prot_name = str(row[name2_col]).strip()
            rna_id  = str(row[id1_col]).strip()
            rna_name= str(row[name1_col]).strip()

        # Determine label
        if label_col:
            label_val = str(row[label_col]).lower()
            label = 1 if any(x in label_val for x in ("1","yes","true","binding","interact")) else 0
        else:
            label = 1  # default: all RNAInter entries are positive

        # Fetch protein sequence
        if prot_id not in prot_seq_cache:
            # Try UniProt ID first, then gene name lookup
            seq = fetch_uniprot_sequence(prot_id)
            if not seq and not prot_id.startswith("P") and not prot_id.startswith("Q"):
                # Try searching UniProt by gene name
                r = requests.get(
                    f"{UNIPROT_BASE}/uniprotkb?query=gene:{prot_name}+AND+organism_id:9606&format=fasta&size=1",
                    timeout=10)
                if r.status_code == 200 and r.text:
                    lines = r.text.strip().split("\n")
                    seq = "".join(l for l in lines if not l.startswith(">"))
            prot_seq_cache[prot_id] = seq or ""
            if seq:
                print(f"  ✓ Protein {prot_name} ({prot_id}): {len(seq)} aa")
            time.sleep(0.1)

        prot_seq = prot_seq_cache.get(prot_id, "")
        if not prot_seq:
            skipped_no_prot += 1
            continue

        # Fetch RNA sequence
        if rna_id not in rna_seq_cache:
            seq = None
            if rna_id.startswith("ENS"):
                seq = fetch_rna_sequence_from_ensembl(rna_id)
            elif re.match(r"[A-Z]{2}_?\d", rna_id):
                seq = fetch_rna_from_ncbi(rna_id)
            rna_seq_cache[rna_id] = seq or ""
            time.sleep(0.07)

        rna_seq_full = rna_seq_cache.get(rna_id, "")
        if not rna_seq_full:
            skipped_no_rna += 1
            continue

        # Split long RNAs into windows
        windows = window_rna(rna_seq_full, args.rna_window)
        for w in windows[:5]:  # max 5 windows per interaction to avoid imbalance
            records.append({
                "protein_name":     prot_name,
                "rna_sequence":     w,
                "protein_sequence": prot_seq,
                "binding_label":    label,
                "dataset":          "rnainter",
                "source":           f"RNAInter_{rna_name}",
            })

    # ── Step 4: Balance and save ──────────────────────────────────────────────
    print(f"\n  Skipped (no protein seq): {skipped_no_prot}")
    print(f"  Skipped (no RNA seq):     {skipped_no_rna}")

    if not records:
        print("\n  ❌  No records with sequences. Check API access and IDs.")
        sys.exit(0)

    df_out = pd.DataFrame(records)

    # Balance per protein
    balanced = []
    for prot, grp in df_out.groupby("protein_name"):
        pos = grp[grp["binding_label"] == 1]
        neg = grp[grp["binding_label"] == 0]
        n_take = min(args.max_per_protein // 2, len(pos))
        balanced.append(pos.sample(min(len(pos), n_take), random_state=42))
        balanced.append(neg.sample(min(len(neg), n_take * 2), random_state=42))
    df_balanced = pd.concat(balanced, ignore_index=True)

    out_path = os.path.join(args.out_dir, "rnainter_human.tsv")
    df_balanced.to_csv(out_path, sep="\t", index=False)

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Total: {len(df_balanced)}")
    print(f"    Positives: {(df_balanced['binding_label']==1).sum()}")
    print(f"    Negatives: {(df_balanced['binding_label']==0).sum()}")
    print(f"  Proteins: {df_balanced['protein_name'].nunique()}")
    print(f"  Saved → {out_path}")
    print(f"\n  Next: python scripts/14_merge_new_data.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
