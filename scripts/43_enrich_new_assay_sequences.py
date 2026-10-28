#!/usr/bin/env python3
"""
Enrich new-assay tables with sequences:
  - PRBAB: RCSB Data API polymer_entity sequences
  - HuRPA: RefSeq transcript (gene -> tx from PRIM refSeq map) + UniProt protein sequences
  - ProNAB: scrape PRA-Pred ProNAB-derived table; document manual full-dump request

Usage:
    python scripts/43_enrich_new_assay_sequences.py --all
    python scripts/43_enrich_new_assay_sequences.py --prbab --hurpa --pronab
"""

from __future__ import annotations

import argparse
import json
import re
import time
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "new_assays"
SAN = ROOT / "data" / "sanitized" / "new_assays"
REFSEQ_MAP = ROOT / "data" / "raw" / "prim_seqtools" / "refSeq_tx_gene_type.csv"
CACHE = ROOT / "data" / "raw" / "new_assays" / "cache"

GENE_TYPE_PRIORITY = (
    "protein_coding",
    "mRNA",
    "lncRNA",
    "transcribed_pseudogene",
    "ncRNA",
    "miRNA",
    "snRNA",
    "snoRNA",
)


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": "protein_rna_ml/43_enrich"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_json(url: str) -> dict | list:
    return json.loads(_http_get(url).decode("utf-8"))


def fetch_pdb_polymer_sequences(pdb_accession: str) -> tuple[str | None, str | None]:
    """Return (protein_seq, rna_seq) from first matching polymer entities."""
    base = pdb_accession.strip().split("_")[0].split()[0][:4].upper()
    try:
        entry = _http_json(f"https://data.rcsb.org/rest/v1/core/entry/{base}")
        entity_ids = entry["rcsb_entry_container_identifiers"]["polymer_entity_ids"]
    except Exception:
        return None, None

    proteins: list[str] = []
    rnas: list[str] = []
    for eid in entity_ids:
        try:
            ent = _http_json(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{base}/{eid}")
            ptype = ent.get("entity_poly", {}).get("type", "")
            seq = ent.get("entity_poly", {}).get("pdbx_seq_one_letter_code") or ""
            if not seq:
                continue
            if "polyribonucleotide" in ptype or "RNA" in ptype.upper():
                rnas.append(seq.upper().replace("T", "U"))
            elif "polypeptide" in ptype:
                proteins.append(seq)
        except Exception:
            continue
    return (proteins[0] if proteins else None, rnas[0] if rnas else None)


def enrich_prbab() -> None:
    path = SAN / "prbab_affinity.tsv"
    df = pd.read_csv(path, sep="\t", low_memory=False)
    cache_path = CACHE / "prbab_pdb_sequences.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    for i, row in df.iterrows():
        pid = str(row.get("pdb_id", ""))
        if not pid or pid in cache:
            continue
        ps, rs = fetch_pdb_polymer_sequences(pid)
        cache[pid] = {"protein_sequence": ps, "rna_sequence": rs}
        time.sleep(0.15)

    cache_path.write_text(json.dumps(cache, indent=2))
    for i, row in df.iterrows():
        pid = str(row.get("pdb_id", ""))
        hit = cache.get(pid, {})
        if hit.get("protein_sequence"):
            df.at[i, "protein_sequence"] = hit["protein_sequence"]
        if hit.get("rna_sequence"):
            df.at[i, "rna_sequence"] = hit["rna_sequence"]

    out = SAN / "prbab_affinity.tsv"
    df.to_csv(out, sep="\t", index=False)
    n_p = df["protein_sequence"].notna().sum()
    n_r = df["rna_sequence"].notna().sum()
    print(f"PRBAB: wrote {out} (protein_seq={n_p}/{len(df)}, rna_seq={n_r}/{len(df)})")


def build_gene_to_transcript() -> pd.DataFrame:
    df = pd.read_csv(REFSEQ_MAP, header=None, names=["transcript_id", "gene_name", "gene_type"])
    df["prio"] = df["gene_type"].map(
        lambda g: GENE_TYPE_PRIORITY.index(g) if g in GENE_TYPE_PRIORITY else 99
    )
    df = df.sort_values(["gene_name", "prio", "transcript_id"])
    best = df.groupby("gene_name", as_index=False).first()
    return best[["gene_name", "transcript_id", "gene_type"]]


def _parse_fasta(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur_id = None
    parts: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                out[cur_id] = "".join(parts)
            cur_id = line[1:].split()[0]
            parts = []
        else:
            parts.append(line.strip())
    if cur_id is not None:
        out[cur_id] = "".join(parts)
    return out


def fetch_refseq_transcripts(tx_ids: list[str], batch: int = 100) -> dict[str, str]:
    try:
        from Bio import Entrez
    except ImportError:
        raise SystemExit("Biopython required: pip install biopython")

    Entrez.email = "protein_rna_ml@local.dev"
    seqs: dict[str, str] = {}
    cache_fa = CACHE / "refseq_transcripts.fasta"
    if cache_fa.exists():
        cached = _parse_fasta(cache_fa.read_text())
        for tid in tx_ids:
            base = tid.split(".")[0]
            for k, v in cached.items():
                if k.startswith(base):
                    seqs[tid] = v.upper().replace("T", "U")
                    break
        missing = [t for t in tx_ids if t not in seqs]
        if not missing:
            return seqs
        tx_ids = missing

    for start in range(0, len(tx_ids), batch):
        chunk = tx_ids[start : start + batch]
        handle = Entrez.efetch(db="nucleotide", id=chunk, rettype="fasta", retmode="text")
        text = handle.read()
        handle.close()
        for k, v in _parse_fasta(text).items():
            seqs[k.split(".")[0]] = v.upper().replace("T", "U")
        # map versioned ids
        for tid in chunk:
            base = tid.split(".")[0]
            if base in seqs and tid not in seqs:
                seqs[tid] = seqs[base]
        time.sleep(0.34)

    # append to cache fasta
    CACHE.mkdir(parents=True, exist_ok=True)
    with cache_fa.open("a") as fh:
        for tid, seq in seqs.items():
            if tid.count(".") == 0:
                fh.write(f">{tid}\n{seq}\n")
    return seqs


def load_uniprot_human_gene_sequences() -> dict[str, str]:
    """Build gene symbol -> sequence from UniProt reviewed human stream (one download)."""
    cache_path = CACHE / "uniprot_human_reviewed.tsv"
    CACHE.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        print("  Downloading UniProt human reviewed stream (one-time)...")
        url = (
            "https://rest.uniprot.org/uniprotkb/stream?format=tsv"
            "&fields=accession,gene_primary,gene_oln,sequence"
            "&query=reviewed:true+AND+organism_id:9606"
        )
        data = _http_get(url, timeout=600)
        cache_path.write_bytes(data)

    u = pd.read_csv(cache_path, sep="\t", low_memory=False)
    found: dict[str, str] = {}
    for _, row in u.iterrows():
        seq = row.get("Sequence")
        if not isinstance(seq, str) or not seq:
            continue
        primary = row.get("Gene Names (primary)")
        if isinstance(primary, str) and primary:
            found.setdefault(primary, seq)
        oln = row.get("Gene Names (OLN)")
        if isinstance(oln, str):
            for g in re.split(r"[;\s]+", oln):
                g = g.strip()
                if g:
                    found.setdefault(g, seq)
    return found


def fetch_uniprot_sequences_by_gene(genes: list[str]) -> dict[str, str]:
    all_map = load_uniprot_human_gene_sequences()
    return {g: all_map[g] for g in genes if g in all_map}


def enrich_hurpa(max_genes: int | None = None) -> None:
    pairs_path = SAN / "hurpa_chimeric_gene_pairs_min4reads.tsv"
    df = pd.read_csv(pairs_path, sep="\t", usecols=lambda c: True, low_memory=False)

    rna_genes = sorted(df["rna_gene"].dropna().unique().tolist())
    prot_genes = sorted(df["protein_name"].dropna().unique().tolist())
    if max_genes:
        rna_genes = rna_genes[:max_genes]
        prot_genes = prot_genes[:max_genes]

    print(f"  HuRPA unique rna_gene={len(rna_genes)}, protein_name={len(prot_genes)}")
    g2t = build_gene_to_transcript()
    g2t = g2t[g2t["gene_name"].isin(rna_genes)]

    tx_ids = g2t["transcript_id"].tolist()
    print(f"  Fetching {len(tx_ids)} RefSeq transcripts via NCBI efetch...")
    tx_seq = fetch_refseq_transcripts(tx_ids)

    rna_map = SAN / "hurpa_rna_gene_sequences.tsv"
    rows_rna = []
    for _, r in g2t.iterrows():
        seq = tx_seq.get(r["transcript_id"]) or tx_seq.get(r["transcript_id"].split(".")[0])
        rows_rna.append(
            {
                "rna_gene": r["gene_name"],
                "transcript_id": r["transcript_id"],
                "gene_type": r["gene_type"],
                "rna_sequence": seq,
            }
        )
    rna_df = pd.DataFrame(rows_rna)
    rna_df.to_csv(rna_map, sep="\t", index=False)
    print(f"  Wrote {rna_map} ({rna_df.rna_sequence.notna().sum()}/{len(rna_df)} with sequence)")

    print("  Fetching UniProt protein sequences...")
    prot_seq = fetch_uniprot_sequences_by_gene(prot_genes)
    prot_df = pd.DataFrame(
        {"protein_name": list(prot_seq.keys()), "protein_sequence": list(prot_seq.values())}
    )
    prot_map_path = SAN / "hurpa_protein_gene_sequences.tsv"
    prot_df.to_csv(prot_map_path, sep="\t", index=False)
    print(f"  Wrote {prot_map_path} ({len(prot_df)} genes)")

    rna_lookup = rna_df.set_index("rna_gene")
    df["rna_transcript_id"] = df["rna_gene"].map(rna_lookup["transcript_id"])
    df["rna_sequence"] = df["rna_gene"].map(rna_lookup["rna_sequence"])
    df["protein_sequence"] = df["protein_name"].map(prot_seq)

    join_hurpa_pairs_from_maps()


def join_hurpa_pairs_from_maps() -> None:
    """Join pre-built gene maps onto min4reads table (fast re-run)."""
    pairs_path = SAN / "hurpa_chimeric_gene_pairs_min4reads.tsv"
    rna_map = SAN / "hurpa_rna_gene_sequences.tsv"
    prot_map = SAN / "hurpa_protein_gene_sequences.tsv"
    if not rna_map.exists() or not prot_map.exists():
        raise FileNotFoundError("Run --hurpa first to build gene maps")

    rna_df = pd.read_csv(rna_map, sep="\t")
    prot_df = pd.read_csv(prot_map, sep="\t")
    rna_lookup = rna_df.set_index("rna_gene")
    prot_lookup = prot_df.set_index("protein_name")

    import os
    import subprocess

    out = SAN / "hurpa_chimeric_gene_pairs_min4reads_enriched.tsv.gz"
    tmp = out.with_name(f"{out.name}.part.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    rna_seq_n = 0
    prot_seq_n = 0
    n_rows = 0
    first = True
    with tmp.open("wb") as raw_out:
        gz = subprocess.Popen(
            ["gzip", "-6", "-c"],
            stdin=subprocess.PIPE,
            stdout=raw_out,
            text=True,
        )
        try:
            for chunk in pd.read_csv(pairs_path, sep="\t", chunksize=200_000, low_memory=False):
                chunk["rna_transcript_id"] = chunk["rna_gene"].map(rna_lookup["transcript_id"])
                chunk["rna_sequence"] = chunk["rna_gene"].map(rna_lookup["rna_sequence"])
                chunk["protein_sequence"] = chunk["protein_name"].map(prot_lookup["protein_sequence"])
                rna_seq_n += int(chunk["rna_sequence"].notna().sum())
                prot_seq_n += int(chunk["protein_sequence"].notna().sum())
                n_rows += len(chunk)
                chunk.to_csv(gz.stdin, sep="\t", index=False, header=first)
                first = False
        finally:
            if gz.stdin:
                gz.stdin.close()
            rc = gz.wait()
    if rc != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"gzip failed with exit code {rc}")
    subprocess.run(["gzip", "-t", str(tmp)], check=True)
    tmp.replace(out)
    print(
        f"  Wrote {out} rows={n_rows} "
        f"rna_seq={rna_seq_n} prot_seq={prot_seq_n}"
    )


def scrape_prapred_pronab_subset() -> None:
    """187 ProNAB-derived complexes from PRA-Pred (HTML table)."""
    url = "https://web.iitm.ac.in/bioinfo2/prapred/dataset.html"
    html = _http_get(url).decode("utf-8", errors="replace")
    tables = pd.read_html(StringIO(html))
    raw = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "PDB" in cols and "Source" in cols:
            raw = t
            break
    if raw is None:
        raise RuntimeError("No PDB/Source table on PRA-Pred dataset page")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(subset=["PDB"])
    raw = raw.rename(
        columns={
            "PDB": "pdb_id",
            "RNA strand": "rna_strand",
            "Functional class": "functional_class",
            "Experimental ΔG(kcal/mol)": "delta_g_kcal_mol",
            "Source": "source_db",
        }
    )
    out_dir = RAW / "pronab"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pronab_prapred_subset.csv"
    raw.to_csv(out, index=False)
    print(f"ProNAB subset (PRA-Pred): {len(raw)} rows -> {out}")

    request_md = out_dir / "FULL_DUMP_REQUEST.md"
    request_md.write_text(
        """# ProNAB full database request

The ProNAB site does not expose a stable public bulk CSV URL (search/export pages return 404).

**Action:** submit the [Data Download](https://web.iitm.ac.in/bioinfo2/pronab/) form on the ProNAB homepage
or email **gromiha@iitm.ac.in** requesting the full Protein–RNA subset (Kd, sequences, conditions).

**Local interim:** `pronab_prapred_subset.csv` — 217 complexes from PRA-Pred (ProNAB + PRBAB refs).

After receiving the dump, place at:
`data/raw/new_assays/pronab/pronab_full.csv`
and run: `python scripts/43_enrich_new_assay_sequences.py --pronab-full`
"""
    )
    print(f"  Wrote {request_md}")


def enrich_pronab_prapred_subset() -> None:
    sub = RAW / "pronab" / "pronab_prapred_subset.csv"
    if not sub.exists():
        return
    df = pd.read_csv(sub)
    cache_path = CACHE / "prbab_pdb_sequences.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    rows = []
    for _, r in df.iterrows():
        pdb = str(r["pdb_id"]).strip().lower()
        hit = cache.get(pdb.upper()) or cache.get(pdb)
        if not hit:
            ps, rs = fetch_pdb_polymer_sequences(pdb)
            hit = {"protein_sequence": ps, "rna_sequence": rs}
            cache[pdb.upper()] = hit
            time.sleep(0.15)
        rows.append(
            {
                "pdb_id": pdb,
                "protein_name": pdb,
                "rna_name": r.get("rna_strand"),
                "protein_sequence": hit.get("protein_sequence"),
                "rna_sequence": hit.get("rna_sequence"),
                "affinity_delta_g_kcal_mol": r.get("delta_g_kcal_mol"),
                "binding_label": 1,
                "dataset_source": "pronab_prapred_subset",
                "experiment_id": r.get("source_db"),
            }
        )
    cache_path.write_text(json.dumps(cache, indent=2))
    out = SAN / "pronab_prapred_affinity.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"ProNAB PRA-Pred enriched: {len(rows)} -> {out}")


def normalize_pronab_full_if_present() -> None:
    path = RAW / "pronab" / "pronab_full.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, low_memory=False)
    # heuristic column normalize
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    na_col = next((c for c in df.columns if "nucleic" in c and "type" in c), None)
    if na_col:
        df = df[df[na_col].astype(str).str.contains("RNA", case=False, na=False)]
    out = SAN / "pronab_rna_affinity.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"ProNAB full: {len(df)} RNA rows -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--prbab", action="store_true")
    p.add_argument("--hurpa", action="store_true")
    p.add_argument("--pronab", action="store_true")
    p.add_argument("--pronab-full", action="store_true")
    p.add_argument("--hurpa-max-genes", type=int, default=None, help="Debug: limit genes fetched")
    p.add_argument(
        "--hurpa-join-only",
        action="store_true",
        help="Only join existing hurpa_*_gene_sequences maps onto min4reads",
    )
    args = p.parse_args()
    if args.all:
        args.prbab = args.hurpa = args.pronab = True

    if args.prbab:
        print("=== PRBAB + RCSB ===")
        enrich_prbab()
    if args.hurpa_join_only:
        print("=== HuRPA join maps only ===")
        join_hurpa_pairs_from_maps()
    elif args.hurpa:
        print("=== HuRPA + RefSeq/UniProt ===")
        enrich_hurpa(max_genes=args.hurpa_max_genes)
    if args.pronab:
        print("=== ProNAB (PRA-Pred subset + request doc) ===")
        scrape_prapred_pronab_subset()
        enrich_pronab_prapred_subset()
    if args.pronab_full:
        normalize_pronab_full_if_present()


if __name__ == "__main__":
    main()
