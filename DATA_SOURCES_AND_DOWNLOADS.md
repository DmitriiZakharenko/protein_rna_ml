# Data Sources & Download Instructions

**Last updated**: 2026-07-20  
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
- ENCODE eCLIP — Van Nostrand et al. / ENCODE portal peak files
- Boyle et al., Skipper — *Cell Genomics* / STAR Protocols 2024 (future track)
