"""
Script 12: Download and process eCLIP data from ENCODE.

Strategy:
  POSITIVES  — IDR peaks (output_type=peaks, reproducible binding sites)
               Extract ±30 nt around peak summit → 60 nt window
  NEGATIVES  — Flanking regions of the same chromosome, same length,
               matched GC content (±0.05), no overlap with any peak for this RBP.
               These are real expressed RNA regions confirmed NOT to be enriched.

For each RBP:
  1. Query ENCODE API → find peak BED files (GRCh38, released)
  2. Download BED.gz → extract peak coordinates + summits
  3. Fetch sequences via Ensembl REST API (no genome download needed)
  4. Generate flanking negatives; fetch their sequences
  5. Fetch protein sequence from UniProt
  6. Save: data/eclip/{RBP}_eclip.tsv

Output TSV columns (same format as our existing generalized splits):
  protein_name | rna_sequence | protein_sequence | binding_label | dataset | source

Usage (from protein_rna_ml/):
    python scripts/12_download_eclip.py
    python scripts/12_download_eclip.py --rbps HNRNPC TARDBP FUS  # specific RBPs
    python scripts/12_download_eclip.py --rbp TARDBP --max_peaks 500 --dry_run
"""

import argparse
import gzip
import io
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
ENCODE_BASE   = "https://www.encodeproject.org"
ENSEMBL_BASE  = "https://rest.ensembl.org"
UNIPROT_BASE  = "https://rest.uniprot.org"
HEADERS_JSON  = {"Accept": "application/json"}
HEADERS_TEXT  = {"Accept": "text/plain"}

# Target RBPs — well-studied, broad eCLIP coverage in ENCODE
# Format: {gene_name: uniprot_id}  (UniProt IDs for canonical human sequences)
TARGET_RBPS = {
    "HNRNPC":  "P07910",   # Heterogeneous nuclear ribonucleoprotein C
    "TARDBP":  "Q13148",   # TDP-43, ALS-associated
    "FUS":     "P35637",   # FUS/TLS, ALS-associated
    "RBFOX2":  "O43251",   # RNA-binding protein fox-2
    "ELAVL1":  "Q15717",   # HuR, ubiquitous ARE-binding
    "QKI":     "Q96PU8",   # Quaking, brain/oligodendrocyte
    "HNRNPA1": "P09651",   # hnRNP A1
    "IGF2BP1": "Q9NZI8",   # IGF2 mRNA-binding protein 1
    "SLBP":    "Q14493",   # Stem-loop binding protein
    "RBFOX1":  "Q9NWB1",   # RNA-binding protein fox-1 (neuronal)
}

WINDOW_SIZE  = 60    # nucleotides extracted per peak (±30 around summit)
NEG_RATIO    = 2     # negatives per positive
NEG_OFFSET   = 3     # flanking window placed NEG_OFFSET * WINDOW_SIZE away from peak
MAX_RETRIES  = 3
RATE_LIMIT   = 0.07  # seconds between Ensembl API requests (~14 req/s, limit is 15)


# ── ENCODE API ────────────────────────────────────────────────────────────────
def get_eclip_bed_files(rbp_name: str, cell_line: str = None) -> list[dict]:
    """
    Return list of released eCLIP peak BED files for a given RBP (GRCh38).

    Two-step approach:
      1. Find all eCLIP experiments for this RBP (target.label filter)
      2. Fetch peak BED files for each experiment accession
    """
    # Step 1: find experiments
    exp_url = (f"{ENCODE_BASE}/search/?type=Experiment"
               f"&assay_title=eCLIP"
               f"&target.label={rbp_name}"
               f"&status=released"
               f"&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens"
               f"&format=json&limit=all")
    if cell_line:
        exp_url += f"&biosample_ontology.term_name={cell_line}"

    r = _get(exp_url, HEADERS_JSON)
    experiments = r.json().get("@graph", [])
    if not experiments:
        return []

    exp_accessions = [e["accession"] for e in experiments if "accession" in e]
    print(f"    Found {len(exp_accessions)} experiment(s): {exp_accessions}")

    # Step 2: find BED peak files for each experiment
    all_files = []
    for acc in exp_accessions:
        file_url = (f"{ENCODE_BASE}/search/?type=File"
                    f"&dataset=/experiments/{acc}/"
                    f"&output_type=peaks"
                    f"&file_format=bed"
                    f"&assembly=GRCh38"
                    f"&status=released"
                    f"&format=json&limit=all")
        rf = _get(file_url, HEADERS_JSON)
        files = rf.json().get("@graph", [])
        all_files.extend(files)

    return all_files


def download_bed_gz(href: str) -> list[dict]:
    """
    Download a BED.gz from ENCODE and return list of peak dicts.
    eCLIP peaks are in narrowPeak format (BED6+4):
      chr, start, end, name, score, strand, signalValue, pValue, qValue, summit_offset
    """
    url = ENCODE_BASE + href
    r = _get(url, {}, stream=True)
    content = b"".join(r.iter_content(chunk_size=65536))

    peaks = []
    with gzip.open(io.BytesIO(content), "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            chrom, start, end = cols[0], int(cols[1]), int(cols[2])
            # Col 9 (0-indexed) = summit offset from start (narrowPeak format)
            summit_offset = int(cols[9]) if len(cols) > 9 and cols[9] != "." else (end - start) // 2
            summit = start + summit_offset
            strand = cols[5] if cols[5] in ("+", "-") else "+"
            score  = float(cols[4]) if cols[4] != "." else 0.0
            peaks.append({
                "chrom": chrom, "start": start, "end": end,
                "summit": summit, "strand": strand, "score": score,
            })
    return peaks


# ── Sequence extraction (Ensembl REST API) ────────────────────────────────────
def fetch_sequences_batch(regions: list[dict], genome: str = "human") -> list[str | None]:
    """
    Fetch DNA sequences for a list of genomic regions via Ensembl REST API.
    regions: list of {chrom, start, end, strand}  (1-based, inclusive)

    Returns list of sequences in same order (None if fetch failed).
    Automatically handles rate limits and batches of 50.
    """
    results = [None] * len(regions)
    batch_size = 50

    for i in range(0, len(regions), batch_size):
        batch = regions[i:i + batch_size]
        query = {"regions": [
            f"{r['chrom']}:{r['start']}..{r['end']}:{1 if r.get('strand','+') == '+' else -1}"
            for r in batch
        ]}
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{ENSEMBL_BASE}/sequence/region/{genome}",
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    data=json.dumps(query),
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    print(f"    Rate limited — waiting {wait}s ...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    break
                seqs = resp.json()
                for j, seq_obj in enumerate(seqs):
                    seq = seq_obj.get("seq", "")
                    results[i + j] = seq.upper().replace("T", "U") if seq else None
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"    Warning: batch {i//batch_size} failed — {e}")
                time.sleep(2 ** attempt)
        time.sleep(RATE_LIMIT)

    return results


# ── Protein sequence (UniProt REST API) ───────────────────────────────────────
def fetch_uniprot_sequence(uniprot_id: str) -> str | None:
    """Fetch canonical protein sequence from UniProt."""
    url = f"{UNIPROT_BASE}/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(MAX_RETRIES):
        try:
            r = _get(url, HEADERS_TEXT)
            lines = r.text.strip().split("\n")
            seq = "".join(l for l in lines if not l.startswith(">"))
            return seq if seq else None
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  Warning: UniProt fetch failed for {uniprot_id} — {e}")
            time.sleep(2 ** attempt)
    return None


# ── Negative region generation ────────────────────────────────────────────────
def generate_negatives(peaks: list[dict], n_ratio: int = NEG_RATIO,
                       offset_mult: int = NEG_OFFSET, window: int = WINDOW_SIZE
                       ) -> list[dict]:
    """
    For each peak, generate `n_ratio` flanking negative windows.

    Each negative is placed `offset_mult * window` nt away from the peak summit
    on the same chromosome. We alternate upstream/downstream and shuffle to
    avoid any systematic directional bias.

    Negatives that would overlap any known peak are discarded.
    """
    # Build peak intervals per chromosome for overlap checking
    chrom_peaks = defaultdict(list)
    for p in peaks:
        chrom_peaks[p["chrom"]].append((p["start"], p["end"]))

    def overlaps_any_peak(chrom, start, end):
        for ps, pe in chrom_peaks.get(chrom, []):
            if start < pe and end > ps:
                return True
        return False

    negatives = []
    half = window // 2
    directions = [1, -1] * (n_ratio // 2 + 1)

    for peak in peaks:
        chrom, summit, strand = peak["chrom"], peak["summit"], peak["strand"]
        added = 0
        for i, direction in enumerate(directions):
            if added >= n_ratio:
                break
            dist     = offset_mult * window * (i // 2 + 1)
            neg_center = summit + direction * dist
            neg_start  = max(1, neg_center - half)
            neg_end    = neg_start + window
            if overlaps_any_peak(chrom, neg_start, neg_end):
                continue
            negatives.append({
                "chrom":  chrom,
                "start":  neg_start,
                "end":    neg_end,
                "summit": neg_center,
                "strand": strand,
                "score":  0.0,
            })
            added += 1

    return negatives


# ── GC content filter ─────────────────────────────────────────────────────────
def gc_content(seq: str) -> float:
    seq = seq.upper().replace("U", "T")
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq)


def filter_by_gc_match(pos_seqs: list[str], neg_seqs: list[str],
                       tolerance: float = 0.08) -> list[bool]:
    """
    For each negative, check if there is at least one positive with matched GC.
    Returns a boolean mask for negatives to keep.
    If all positives have GC in [neg_gc ± tolerance], keep the negative.
    """
    pos_gcs = [gc_content(s) for s in pos_seqs if s]
    if not pos_gcs:
        return [True] * len(neg_seqs)
    gc_min, gc_max = min(pos_gcs) - tolerance, max(pos_gcs) + tolerance
    return [gc_min <= gc_content(s) <= gc_max if s else False for s in neg_seqs]


# ── HTTP helper ───────────────────────────────────────────────────────────────
def _get(url: str, headers: dict, stream: bool = False) -> requests.Response:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=30, stream=stream)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                time.sleep(5)
                continue
            raise
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


# ── Per-RBP processing pipeline ───────────────────────────────────────────────
def process_rbp(rbp_name: str, uniprot_id: str, out_dir: str,
                max_peaks: int, cell_line: str | None,
                dry_run: bool = False) -> pd.DataFrame | None:
    print(f"\n{'='*60}")
    print(f"  RBP: {rbp_name}  (UniProt: {uniprot_id})")
    print(f"{'='*60}")

    out_path = os.path.join(out_dir, f"{rbp_name}_eclip.tsv")
    if os.path.exists(out_path):
        print(f"  Already processed — loading {out_path}")
        return pd.read_csv(out_path, sep="\t")

    # 1. Find BED files
    bed_files = get_eclip_bed_files(rbp_name, cell_line)
    if not bed_files:
        print(f"  ⚠️  No eCLIP BED files found for {rbp_name} — skipping")
        return None
    print(f"  Found {len(bed_files)} peak BED file(s)")

    # 2. Download and merge peaks from all files (both replicates, both cell lines)
    all_peaks = []
    for bf in bed_files:
        href = bf.get("href", "")
        acc  = bf.get("accession", "?")
        print(f"  Downloading {acc} ...")
        try:
            peaks = download_bed_gz(href)
            all_peaks.extend(peaks)
            print(f"    {len(peaks)} peaks")
        except Exception as e:
            print(f"    ⚠️  Failed: {e}")

    if not all_peaks:
        print(f"  ⚠️  No peaks loaded — skipping")
        return None

    print(f"  Total peaks: {len(all_peaks)}")

    # 3. Sort by score, take top max_peaks
    all_peaks.sort(key=lambda p: p["score"], reverse=True)
    peaks = all_peaks[:max_peaks]
    print(f"  Using top {len(peaks)} peaks (by score)")

    if dry_run:
        print(f"  [dry_run] Would process {len(peaks)} positives + "
              f"{len(peaks)*NEG_RATIO} negatives")
        return None

    # 4. Extract positive sequences: ±30 nt around summit
    half = WINDOW_SIZE // 2
    pos_regions = [{
        "chrom": p["chrom"],
        "start": max(1, p["summit"] - half),
        "end":   p["summit"] + half,
        "strand": p["strand"],
    } for p in peaks]

    print(f"  Fetching {len(pos_regions)} positive sequences via Ensembl REST ...")
    pos_seqs = fetch_sequences_batch(pos_regions)
    n_pos_ok = sum(1 for s in pos_seqs if s)
    print(f"  Fetched: {n_pos_ok}/{len(pos_seqs)} positive sequences")

    # 5. Generate and fetch negative regions
    negatives = generate_negatives(peaks, NEG_RATIO)
    neg_regions = [{
        "chrom":  n["chrom"],
        "start":  n["start"],
        "end":    n["end"],
        "strand": n["strand"],
    } for n in negatives]

    print(f"  Fetching {len(neg_regions)} negative sequences ...")
    neg_seqs = fetch_sequences_batch(neg_regions)
    n_neg_ok = sum(1 for s in neg_seqs if s)
    print(f"  Fetched: {n_neg_ok}/{len(neg_seqs)} negative sequences")

    # 6. GC-content filter on negatives
    valid_neg = filter_by_gc_match(
        [s for s in pos_seqs if s],
        neg_seqs, tolerance=0.08)
    print(f"  Negatives passing GC filter: {sum(valid_neg)}/{len(neg_seqs)}")

    # 7. Fetch protein sequence
    print(f"  Fetching protein sequence (UniProt: {uniprot_id}) ...")
    prot_seq = fetch_uniprot_sequence(uniprot_id)
    if not prot_seq:
        print(f"  ⚠️  Could not fetch protein sequence — skipping")
        return None
    print(f"  Protein length: {len(prot_seq)} aa")

    # 8. Build output DataFrame
    rows = []
    for seq in pos_seqs:
        if seq and len(seq) >= 10:
            rows.append({
                "protein_name":     rbp_name,
                "rna_sequence":     seq[:WINDOW_SIZE],
                "protein_sequence": prot_seq,
                "binding_label":    1,
                "dataset":          "eclip",
                "source":           "ENCODE_eCLIP_peak",
            })
    for seq, keep in zip(neg_seqs, valid_neg):
        if keep and seq and len(seq) >= 10:
            rows.append({
                "protein_name":     rbp_name,
                "rna_sequence":     seq[:WINDOW_SIZE],
                "protein_sequence": prot_seq,
                "binding_label":    0,
                "dataset":          "eclip",
                "source":           "ENCODE_eCLIP_flanking_neg",
            })

    df = pd.DataFrame(rows)
    n_pos = (df["binding_label"] == 1).sum()
    n_neg = (df["binding_label"] == 0).sum()
    print(f"  Output: {n_pos} positives + {n_neg} negatives = {len(df)} total")

    # 9. Save
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved → {out_path}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbps",      nargs="+",
                        default=list(TARGET_RBPS.keys()),
                        help="RBP gene names to process (default: all 10)")
    parser.add_argument("--out_dir",   default="data/eclip",
                        help="Output directory for per-RBP TSV files")
    parser.add_argument("--max_peaks", type=int, default=1000,
                        help="Max peaks per RBP per run (top by score)")
    parser.add_argument("--cell_line", default=None,
                        choices=["K562", "HepG2"],
                        help="Restrict to one cell line (default: both)")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Query API and count files, but don't download sequences")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Validate RBP names
    unknown = [r for r in args.rbps if r not in TARGET_RBPS]
    if unknown:
        print(f"⚠️  Unknown RBPs (no UniProt ID defined): {unknown}")
        print(f"   Known RBPs: {list(TARGET_RBPS.keys())}")
        args.rbps = [r for r in args.rbps if r in TARGET_RBPS]

    print(f"\n=== eCLIP download pipeline ===")
    print(f"  RBPs:       {args.rbps}")
    print(f"  Max peaks:  {args.max_peaks} per RBP")
    print(f"  Cell line:  {args.cell_line or 'both'}")
    print(f"  Out dir:    {args.out_dir}")
    print(f"  Dry run:    {args.dry_run}")

    all_dfs = []
    for rbp in args.rbps:
        uid = TARGET_RBPS[rbp]
        try:
            df = process_rbp(rbp, uid, args.out_dir, args.max_peaks,
                             args.cell_line, args.dry_run)
            if df is not None:
                all_dfs.append(df)
        except Exception as e:
            print(f"  ❌  {rbp} failed: {e}")
            import traceback; traceback.print_exc()

    if not all_dfs or args.dry_run:
        print("\nDry run complete — no files written.")
        return

    # Merge all RBPs into one TSV
    merged = pd.concat(all_dfs, ignore_index=True)
    merged_path = os.path.join(args.out_dir, "eclip_all.tsv")
    merged.to_csv(merged_path, sep="\t", index=False)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total samples: {len(merged)}")
    print(f"    Positives: {(merged['binding_label']==1).sum()}")
    print(f"    Negatives: {(merged['binding_label']==0).sum()}")
    print(f"  RBPs processed: {merged['protein_name'].nunique()}")
    print(f"  Merged file: {merged_path}")
    print(f"\n  Next: python scripts/14_merge_new_data.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
