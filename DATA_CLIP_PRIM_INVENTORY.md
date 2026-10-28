# CLIP review + PRIM-seq — dataset inventory for `protein_rna_ml`

**Created**: 2026-09-24  
**Sources**:
- [PubMed 40925998](https://pubmed.ncbi.nlm.nih.gov/40925998/) — PRIM-seq / HuRPA ([Nat Biotechnol](https://www.nature.com/articles/s41587-025-02780-z))
- [Xiang et al. 2024](https://doi.org/10.1038/s41576-024-00749-3) — *Decoding protein–RNA interactions using CLIP-based methodologies* (Nat Rev Genet; review, **no primary deposit**)

**Local downloads**: `data/raw/geo_prim/`, `data/raw/prim_seqtools/`  
**Already integrated in project**: ENCODE eCLIP via `scripts/12`, Skipper benchmark `data/raw/eclip_skipper/` (Boyle et al. 2023, cited in review ref. 13).

---

## 1. PRIM-seq paper (PMID 40925998) — primary data

| ID | Type | Stats | Local path | Download |
|----|------|-------|------------|----------|
| **GSE270010** | PRIM-seq PE RNA | 5 libraries: HEK293T ×3, K562 ×2 | `data/raw/geo_prim/suppl/GSE270010_RAW.tar` (**2.4 GB** ✅) | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270010) |
| **GSE270009** | PHGDH RIP-seq + IgG | 4 libraries, HEK293T | `data/raw/geo_prim/suppl/GSE270009_RAW.tar` (**830 KB**) | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270009) |
| **PRJNA1124853** | BioProject (PRIM) | — | — | Linked from GSE270010 |
| **PRJNA1124851** | BioProject (RIP) | — | — | Linked from GSE270009 |
| **HuRPA network** | Processed associations | Paper: **>350k** associations, **~7k** RNAs, **~11k** proteins, **2,610** proteins with ≥10 RNA partners | Build via [PRIMseqTools](https://github.com/Zhong-Lab-UCSD/PRIMseqTools) or [web portal](https://sysbiocomp.ucsd.edu/prim/) (portal returned 500 on 2026-09-24) | GitHub + GEO RAW |
| **refSeq_tx_gene_type.csv** | Transcript → gene type (GENCODE-style) | **163,808** rows | `data/raw/prim_seqtools/refSeq_tx_gene_type.csv` | [raw GitHub](https://raw.githubusercontent.com/Zhong-Lab-UCSD/PRIMseqTools/master/refSeq_tx_gene_type.csv) |

**Fetch large PRIM RAW (manual / VM):**

```bash
curl -C - -o data/raw/geo_prim/suppl/GSE270010_RAW.tar \
  ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270010/suppl/GSE270010_RAW.tar
```

### Usefulness for our project (PRIM-seq)

| Asset | Rating | Rationale |
|-------|--------|-----------|
| GSE270010 / HuRPA | **Medium (eval / Phase 6)** | **Transcript-level** RNA–protein association network (de novo RBP discovery), not motif-level SELEX labels. Useful for in vivo bridge, prioritizing RBPs, **not** as drop-in train mix with `generalized_v3a` without label harmonization. |
| GSE270009 RIP-seq | **Low–medium** | Single-protein validation; small N; good sanity check for PHGDH-associated RNAs vs HuRPA. |
| refSeq_tx_gene_type | **Medium (annotation)** | RNA class labels for long-transcript / CLIP peak interpretation (lncRNA vs mRNA). |

---

## 2. Nature Reviews Genetics 2024 — atlases & databases cited (actionable)

The review does **not** release new data. Below: **highlighted** primary resources from editorial notes + central CLIP thread. Full reference list: 212 papers in the PDF.

### Tier A — already in repo or on critical path

| Resource | Paper (review ref.) | Scale (published) | Access | Local / script | Usefulness |
|----------|---------------------|-------------------|--------|----------------|------------|
| **ENCODE eCLIP (2016 panel)** | Van Nostrand *Nat Methods* 2016 (ref. 8) | **73** RBPs, HepG2 + K562 | [ENCODE](https://www.encodeproject.org/) | `scripts/12_download_eclip.py` → `data/sanitized/eclip/eclip_all.tsv` — **26,362** pairs, **9** RBPs in current clone, **60 nt** windows, pos rate **34%** | **High** — site-level in vivo positives + GC-matched flanks; matches Phase 6 / RPIembeddor2 direction |
| **Skipper reprocessed eCLIP** | Boyle *Cell Genom* 2023 (ref. 13) | **264** experiments, **173** gene symbols, 4 length modes | [Figshare 21206009](https://doi.org/10.6084/m9.figshare.21206009) + Dropbox bundle | `data/raw/eclip_skipper/` — **fixlen_151**: **736,347** pos + **736,347** neg windows (**151 nt** each), **185** proteins in domain manifest | **Very high** — primary long-window in vivo benchmark (`scripts/41*`) |
| **ENCODE RBP encyclopedia (2020)** | Van Nostrand *Nature* 2020 (ref. 46) | **356** RBPs, **1,223** replicated datasets, **5** assays (eCLIP, RBNS, KD RNA-seq, ChIP, localization) | ENCODE publication bundles: [eCLIP ENCSR456FVU](https://www.encodeproject.org/publication-data/ENCSR456FVU/), [RBNS ENCSR876DCD](https://www.encodeproject.org/publication-data/ENCSR876DCD/), KD RNA-seq HepG2/K562, etc. | RBNS overlap with `../rbns_analysis/`; eCLIP peaks via ENCODE API (same as script 12) | **Very high** — RBNS + eCLIP under one RBP roster; RBNS already core train |
| **Principles / 150 RBP eCLIP** | Van Nostrand *Genome Biol* 2020 (ref. 45) | **150** RBPs (subset of ENCODE expansion) | ENCODE + [GSE107768](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107768) (+50 eCLIP) | Not bulk-downloaded here | **High** — motif/peak biology; same ENCODE stack |

### Tier B — strong for Phase 6+ / auxiliary, not yet downloaded

| Resource | Paper (review ref.) | Scale | Access | Usefulness |
|----------|---------------------|-------|--------|------------|
| **starBase v2.0** | Li *NAR* 2014 (ref. 101) | Large-scale CLIP-derived **miRNA–ceRNA–protein–RNA** networks | [starBase](http://starbase.sysu.edu.cn/) | **Low** for motif RBP task (miRNA-centric); **medium** for lncRNA–protein literature-style pairs |
| **Tethered function screen** | Luo *NSMB* 2020 (ref. 14) | Systematic **RBP tethering** → mRNA stability / translation | Primary paper GEO (not fetched) | **Medium** — functional readouts beyond binding; different label than binary bind |
| **126 RBP miRNA processing** | Nussbacher *Mol Cell* 2018 (ref. 103) | **126** RBPs, miRNA interaction profiles | ENCODE/Yeo lab supplementary | **Low** — miRNA pathway, not generic RBP–motif |
| **Mammalian mRNA-binding proteome** | Castello *Cell* 2012 (ref. 207) | Proteome-wide RBP discovery | GEO / PRIDE (per paper) | **Low–medium** — protein-centric, not pairwise sequence labels |
| **Temporal-iCLIP** | Cordiner *Nat Commun* 2023 (ref. 20) | Time-resolved co-transcriptional loading | GEO per paper | **Medium** — dynamics; niche for nascent RNA |
| **Antibody-barcode eCLIP** | Lorenz *Nat Methods* 2023 (ref. 183) | Multiplexed eCLIP | ENCODE / Eclipse | **Medium** — more RBPs at lower cost; same peak semantics |
| **SARS-CoV-2 RNA–host maps** | Schmidt *Nat Microbiol* 2021 (ref. 148); Flynn *Cell* 2021 (ref. 149) | Viral–host RBP networks | GEO per study | **Low** — viral context, off-target for SELEX-trained models |
| **VIR-CLASP** | Kim *Mol Cell* 2020 (ref. 154) | Multiplex viral RNA–protein | GEO per study | **Low** |
| **TF–RNA UV MS** | Oksuz *Mol Cell* 2022 (ref. 34) | ~half of TFs bind RNA | Proteomics + meta-analysis | **Low** for current classifier |
| **TRIBE / STAMP / APOBEC1 editors** | refs. 184–186, 192 | Cell-type RBP targets via editing | Per-lab GEO | **Low–medium** — site maps, custom assays |
| **hRBPome repository** | Ghosh *bioRxiv* 2018 (cited in PRIM paper) | Human RBP catalogue | bioRxiv / derived DB | **Low** — annotation only |

### Tier C — methodology papers (no single “dataset” row)

HITS-CLIP, PAR-CLIP, iCLIP, irCLIP, psiCLIP, proximity-CLIP, etc. (refs. 4–7, 9, 41, 197, 521) — use their **individual GEO accessions** when a specific RBP is needed; not bulk-imported.

---

## 3. Cross-check: claims vs our pipeline

| Claim | Verified |
|-------|----------|
| PRIM-seq = **pair-level interactome**, not SELEX motifs | Yes — SMART-display + REILIS, HuRPA counts |
| Review’s main train-relevant CLIP data = **peaks / windows** | Yes — eCLIP 60 nt train; Skipper 101–201 nt eval |
| PRIM negatives ≠ eCLIP flanks | PRIM has no “negative windows”; associations are **positive-centric** network edges |
| Literature external ≈ same **task family** as starBase/lncRNA catalogs | Yes — molecule/pair level (see `DATA.md` §2.2) |

---

## 4. Recommended acquisition order (project-aligned)

1. **Finish** Skipper `fixlen_151` archives if missing (`data/raw/eclip_skipper/README.md`).
2. **Expand** `scripts/12` RBP list toward ENCODE **356** roster (overlap with Skipper **173** symbols).
3. **Optional VM job**: `GSE270010_RAW.tar` → build HuRPA edges for **eval only**.
4. **Defer** starBase / PRIM train merge until Phase 6 protocol for **assay-stratified** metrics is fixed.
5. **Do not** bulk-download entire ENCSR456FVU without a per-RBP manifest (multi-TB).

---

## 5. File index (this inventory)

```
data/raw/geo_prim/
  MANIFEST.json
  GSE270010.soft
  GSE270009.soft
  suppl/GSE270009_RAW.tar          # downloaded
  suppl/GSE270010_RAW.tar          # run curl command above (~2.4 GB)
data/raw/prim_seqtools/
  refSeq_tx_gene_type.csv          # downloaded
data/raw/eclip_skipper/
  manifests/ ...                   # see dataset_summary.json
```

**Maintenance**: extend this file when a new CLIP source is wired into `scripts/` or `DATA_SOURCES_AND_DOWNLOADS.md`.
