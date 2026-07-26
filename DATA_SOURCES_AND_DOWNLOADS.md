# Data Sources & Download Instructions

**Last updated**: 2026-07-21  
**Purpose**: Every cross-protocol / domain step must be traceable to a primary source.
Do not invent tables. If a file is missing, download it with the steps below and
record the path in `configs/cross_protocol.yaml`.

---

## 1. Required for Week-1 cross-protocol

| Asset | Role | Default path | How to obtain |
|-------|------|--------------|---------------|
| HTR-SELEX clean TSV | Protocol A | `../htr_selex_analysis/results/ml_dataset_simple_clean.tsv` | Sibling repo `htr_selex_analysis` (Ray et al. 2019 pipeline output) |
| RBNS clean TSV | Protocol B | `../rbns_analysis/results/ml_dataset_rbns_clean.tsv` | Sibling repo `rbns_analysis` (Lambert et al. 2020) |
| RNAcompete Eukarya | Protocol C | `../rnacompete_analysis/eukarya/results/ml_dataset_eukarya_clean.tsv.gz` | Sibling `rnacompete_analysis` |
| RNAcompete RBPZoo | Protocol D | `../rnacompete_analysis/rbpzoo/results/ml_dataset_rbpzoo_clean.tsv.gz` | Sibling `rnacompete_analysis` (Sasse et al. 2025) |
| eCLIP panel | Protocol E | `data/eclip/eclip_all.tsv` | Built by `scripts/12_download_eclip.py` from ENCODE |
| Table S1 (domains) | Domain labels | `data/raw/rbpzoo/TableS1.xlsx` | See §2 below |
| Top/bottom summary | Motif Jaccard | `results/top_bottom_examples/all_protocols_summary.tsv` | `scripts/24` + `29` |
| RNA-only metrics | Roster names / optional LR cache | `results/rna_only_per_protein_honest/*` | `scripts/25` |

**Integrity check before every classifier run:**

```bash
python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(open("configs/cross_protocol.yaml"))
paths = cfg["paths"]
keys = ["htr_selex","rbns","rnacompete_eukarya","rnacompete_rbpzoo","eclip","table_s1","top_bottom_summary"]
for k in keys:
    p = Path(paths[k] if k != "top_bottom_summary" else paths["top_bottom_summary"])
    if not p.is_absolute():
        p = Path.cwd() / p
    print(("OK " if p.exists() else "MISSING "), p, f"({p.stat().st_size if p.exists() else 0} bytes)")
PY
```

If any sibling clean TSV is `MISSING`, do **not** invent substitutes from `generalized_v3a`
(`selex_rbns` merges HTR-SELEX+RBNS and breaks protocol identity).

---

## 2. Table S1 — domain architectures (Sasse et al. 2025)

**What it is**: Supplementary Table S1 from the RBPZoo / EuPRI RNAcompete paper
(Sasse et al., *Nature Biotechnology* 2025). Columns used:

- `Protein name`
- `Domains in construct`
- `Domain Boundaries`
- `Species`
- `RNAcompete ID(s)` / `Hyb ID(s)`
- `Average Z-score of top 10 7-mers`

**Expected local file**: `data/raw/rbpzoo/TableS1.xlsx`  
(Currently present in this clone: ~157 KB.)

### Download if missing

1. Open the Sasse et al. 2025 article page (Nature Biotechnology) and go to
   **Supplementary information**.
2. Download **Supplementary Table 1** (often named `41587_2024_XXXX_MOESM*_ESM.xlsx`
   or similar — verify the caption says construct / domain annotations for RNAcompete).
3. Save as:

```bash
mkdir -p data/raw/rbpzoo
# after download:
mv ~/Downloads/<Supplementary_Table_S1>.xlsx data/raw/rbpzoo/TableS1.xlsx
```

4. Verify columns:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_excel("data/raw/rbpzoo/TableS1.xlsx")
need = ["Protein name", "Domains in construct", "Domain Boundaries", "Species",
        "RNAcompete ID(s)", "Average Z-score of top 10 7-mers"]
print("rows", len(df))
print("missing cols", [c for c in need if c not in df.columns])
print("have", list(df.columns)[:12])
PY
```

5. Re-run roster:

```bash
python scripts/33_build_cross_protocol_roster.py
```

**Do not** hand-edit domain strings. If a gene has multiple constructs with different
architectures, script 33 flags `domain_match_status=ambiguous_architectures`.

**Roster join aliases** (same Table S1 row, different gene spelling in SELEX/RBNS):

| Roster key | Table S1 key |
|------------|--------------|
| `HNRPLL` | `HNRNPLL` |
| `RBFOX1` | `A2BP1` |
| `PUM1` | `PUM` |

Wired in `scripts/37_annotate_protein_domains.py` (`TABLE_S1_JOIN_ALIASES`).

---

## 2b. UniProt bulk TSV — SELEX/RBNS-only genes absent from Table S1

**Why**: ~12 roster proteins appear in HTR-SELEX/RBNS but not in Sasse Table S1
(RNAcompete constructs). Prefer **one downloaded TSV**, not ad-hoc per-gene API
scrapes mixed into the annotation table.

**Local file**: `data/raw/uniprot/roster_missing_domains.tsv`

**Genes covered** (reviewed human): `IGF2BP1`, `BOLL`, `CELF1`, `DAZ3`, `ELAVL4`,
`RBFOX3`, `RBM4B`, `RBMS2`, `RC3H1`, `THUMPD1`, `ZFP36`, `ZRANB2`.

### Re-download

```bash
mkdir -p data/raw/uniprot
curl -fsSL -o data/raw/uniprot/roster_missing_domains.tsv \
  'https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession%2Cgene_primary%2Cft_domain%2Cft_zn_fing%2Clength%2Csequence&query=%28%28gene_exact%3AIGF2BP1%29%20OR%20%28gene_exact%3ABOLL%29%20OR%20%28gene_exact%3ACELF1%29%20OR%20%28gene_exact%3ADAZ3%29%20OR%20%28gene_exact%3AELAVL4%29%20OR%20%28gene_exact%3ARBFOX3%29%20OR%20%28gene_exact%3ARBM4B%29%20OR%20%28gene_exact%3ARBMS2%29%20OR%20%28gene_exact%3ARC3H1%29%20OR%20%28gene_exact%3ATHUMPD1%29%20OR%20%28gene_exact%3AZFP36%29%20OR%20%28gene_exact%3AZRANB2%29%29%20AND%20%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29'
```

Then rebuild annotations (Table S1 first, then this TSV):

```bash
python scripts/37_annotate_protein_domains.py \
  --uniprot_tsv data/raw/uniprot/roster_missing_domains.tsv
```

`source=uniprot_tsv` rows give full-length UniProt domain intervals (not RNAcompete
construct boundaries). Use Table S1 for construct-mask experiments when available.

---

## 3. eCLIP (ENCODE)

```bash
python scripts/12_download_eclip.py
# or subset:
python scripts/12_download_eclip.py --rbps TARDBP FUS RBFOX2
```

Uses ENCODE API peak BEDs + Ensembl sequence fetch. Output: `data/eclip/{RBP}_eclip.tsv`
and `data/eclip/eclip_all.tsv`.

For **Skipper** reprocessing (later track): workflow is public
(Boyle et al.; STAR Protocols 2024) — raw BAMs from ENCODE, not the pre-called
IDR peaks alone. Instructions will live in a future `SKIPPER_SETUP.md`.

---

## 4. Protein-sequence QC (mandatory before protein-aware retrain)

```bash
python scripts/40_sanitize_training_protein_sequences.py
# report → results/data_qc/protein_sequence_contamination_report.tsv
```

Known contamination (2026-07-20):

- `RBM38` in `generalized_v3a/train.tsv`: embedded residue numbers
  (`...EE708090100110120AVV...`)
- Trailing `*` stop codons (HEXIM1/2, several ucRBP entries)

Repair (writes copies under `data/sanitized/` unless `--inplace`):

```bash
python scripts/40_sanitize_training_protein_sequences.py --apply
```

Then **retrain** any protein-aware model (V2/V3/V4) from sanitized TSVs.
RNA-only cross-protocol scripts do not read protein sequences.

---

## 5. Verification checklist (before claiming numbers)

- [ ] All paths in §1 integrity check print `OK`
- [ ] Roster rebuilt after any Table S1 / metrics update (`scripts/33`)
- [ ] `match_ambiguities.tsv` reviewed (construct reps, blocked synonyms)
- [ ] Classifiers use exact `name_in_<protocol>` (script 34 policy)
- [ ] Cached within-AUROC only from `*_model_comparison.tsv` **logistic_regression**
      `test_auroc` — never RF `best_model` summaries
- [ ] Transfer summary reports `exact_rna_overlap_fraction` (leakage diagnostic)
- [ ] Protein QC report exists if citing V2/V3a protein-aware scores

---

## 6. Citation anchors

- Ray et al., *Genome Res.* 2019 — HTR-SELEX PRJEB25907
- Lambert et al., *Nature* 2020 — RBNS
- Ray et al., *Nature* 2013 — RNAcompete eukarya
- Sasse et al., *Nat. Biotechnol.* 2025 — RBPZoo / Table S1 domains
- UniProt — reviewed human entries for SELEX/RBNS-only domain fill-in (§2b)
- ENCODE eCLIP — Van Nostrand et al. / ENCODE portal peak files
- Boyle et al., Skipper — *Cell Genomics* / STAR Protocols 2024 (future track)
