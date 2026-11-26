# High-quality dataset hunt — aligned with `protein_rna_ml`

**Created**: 2026-09-24  
**Goal**: Motif-level RBP–RNA binding (in vitro first, site-level in vivo second).  
**Not in scope here**: NPInter/RPI catalogs, PRIM HuRPA interactome, starBase miRNA networks.

**Related**: `DATA.md`, `DATA_SOURCES_AND_DOWNLOADS.md`, `DATA_CLIP_PRIM_INVENTORY.md`, `RESEARCH_ROADMAP.md` §7.

---

## New assays only (exclude CLIP, SELEX/RBNS, RNAcompete, current `v3a` train)

Use this section when hunting **orthogonal** data — not more ENCODE/Skipper/RBPZoo.

| Resource | Assay type | Scale (order of magnitude) | Access | Fit for motif embedder | Notes |
|----------|------------|----------------------------|--------|------------------------|-------|
| **PRIM-seq → HuRPA** | In-lysate chimeric DNA maps protein–RNA pairs | **~365k** edges; **11k** proteins × **7k** RNAs | [GEO GSE270010](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270010); edges via [HuRPA portal](https://sysbiocomp.ucsd.edu/prim/); pipeline [PRIMseqTools](https://github.com/Zhong-Lab-UCSD/PRIMseqTools) | ⭐ **interactome eval only** | Raw **2.4 GB** in `data/raw/geo_prim/`; labels are **transcript–protein**, not sites; same shortcut risks as catalogs unless sliced with read-pair evidence |
| **ProNAB** | Literature **Kd / ΔG** (gel shift, SPR, ITC, …) | **~5.4k** protein–**RNA** rows (subset of 20k total NA) | [web.iitm.ac.in/bioinfo2/pronab](https://web.iitm.ac.in/bioinfo2/pronab/) — per-query download; full dump on request | ⭐⭐ **Phase 3C affinity** | Pairs are **defined RNA sequences** (often aptamer/PDB fragments), not random pools; complements `dataset_with_affinities.xlsx` |
| **PRBAB 2.0** | PDB complexes + curated **pKD** | **~72** new + legacy (**~145** RNA complexes total) | [rnabinding.com/PRdeltaGPred](http://www.rnabinding.com/PRdeltaGPred/) | ⭐⭐ structural/affinity eval | Small but clean; overlaps PDB contact benchmarks |
| **UTexas Aptamer DB** | Literature aptamer–target (**Kd** when reported) | **~1.4k** aptamer records (RNA + DNA) | [Zenodo 8387047](https://zenodo.org/records/8387047) (`dataset_Sept2023.xlsx`) | ⭐⭐ affinity / **specialized RNA** | Many selections are historically SELEX-derived — use as **aptamer modality**, not mRNA motif train |
| **PRADB** (ZeroFold) | Merged affinity compendium | **2,621** unique protein–RNA pairs | Not yet a public dump; sources = ProNAB, BioLiP2, UTexas, PDBbind+ | ⭐⭐ when released | Watch [arXiv:2603.23583](https://arxiv.org/abs/2603.23583) / authors for tables |
| **BioLiP / PDB nt–aa contacts** | X-ray/NMR interfaces | Hundreds of complexes | BioLiP2, RPI / contact predictors’ splits | ⭐⭐ **Phase 7** residue–nucleotide | Different task (contacts), not binary site windows |
| **RNP-MaP** | In-cell chemical probing of **multi-protein** occupancy on one RNA | Few RNAs (XIST, RNase P, RMRP, U1, …) | [GSE152483](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152483) processed tar | ⭐ niche | Nucleotide resolution but **not** genome-wide RBP panel |
| **Hughes RNAcompete supp. “in vivo pulldown”** | Literature pulldowns (0/1 bound + sequence) | Variable | [Eukarya supp](https://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/) → `Sequence data` tar | ⭐ auxiliary classification | **Not** probe pools. Overlap QC (`scripts/44`): **0** exact pairs vs 159-pair literature; **3** shared symbols (ELAVL1, RBM4, TARDBP); **11/23** Hughes symbols in `v3a` train — use protein-disjoint or holdout RBPs if mixing train. |

**Explicitly not “new experimental labels” for us** (predictions, CLIP-derived catalogs, or proteome-only):

- **RNAct / catRAPID omics** — predicted interactomes ([download](https://rnact.tartaglialab.com/download)).
- **NPInter v5 bulk files** — still “coming soon” on [download page](http://bigdata.ibp.ac.cn/npinter5/download/); v4 is mostly mined CLIP/literature pairs.
- **XRNAX / RBDmap / GRPIp (PXD054800)** — RBP discovery or domain mapping, **no** per-row RNA sequence for binding.
- **in situ STAMP (GSE240521)** — site mapper analogous to CLIP family.

**Local normalized TSVs** (`python scripts/42b_download_normalize_new_assays.py`):

| File | Rows (2026-09-24) |
|------|-------------------|
| `data/sanitized/new_assays/utexas_aptamer_rna.tsv` | 583 |
| `data/sanitized/new_assays/hughes_invivo_pulldown.tsv` | 216,588 |
| `data/sanitized/new_assays/prbab_affinity.tsv` | 218 (**217** with RCSB protein+RNA seq; `scripts/43`) |
| `data/sanitized/new_assays/pronab_prapred_affinity.tsv` | 217 (PRA-Pred interim; full ProNAB on manual request) |
| `data/sanitized/new_assays/hurpa_chimeric_gene_pairs.tsv` | ~13.8M (5 GEO libraries) |
| `data/sanitized/new_assays/hurpa_chimeric_gene_pairs_min4reads.tsv` | 3,753,739 |
| `data/sanitized/new_assays/hurpa_rna_gene_sequences.tsv` | 18,397 genes (**18,378** RefSeq RNA) |
| `data/sanitized/new_assays/hurpa_protein_gene_sequences.tsv` | 14,698 genes (UniProt human reviewed) |
| `data/sanitized/new_assays/hurpa_chimeric_gene_pairs_min4reads_enriched.tsv.gz` | min4reads + sequences (`--hurpa-join-only`) |

Manifest: `data/raw/new_assays/MANIFEST.json` · ProNAB request: `data/raw/new_assays/pronab/FULL_DUMP_REQUEST.md`

**Literature / train overlap QC:** `python scripts/44_overlap_new_assays_literature.py` → `results/new_assays/literature_overlap_qc.{json,md}`

**Suggested order (new-assay track only):**

1. HuRPA QC + homology holdout vs `v3a` proteins; optional transcript **windowing** for embedder (**eval only**).
2. ProNAB **full** export (manual request) → `--pronab-full`.
3. Merge aptamer + `dataset_with_affinities.xlsx` for **affinity regression** eval.

---

## Quality criteria (how we score)

| Criterion | Why it matters for us |
|-----------|------------------------|
| **Label = binding site or probe enrichment** | Same semantics as SELEX/RNAcompete, not “whole transcript interacts” |
| **Hard negatives** | GC/length-matched flanks, non-enriched pool, or explicit decoys — not random lncRNA pairs |
| **Protein + RNA sequence per row** | Fits `generalized_v3a` schema |
| **Homology-aware split possible** | Protein-aware minimum; MMseqs2 for train vs benchmark |
| **Scale** | Enough proteins to beat 494-panel ceiling on zero-shot |

**Rating**: ⭐⭐⭐ = train/eval ready with modest ETL | ⭐⭐ = strong after preprocessing | ⭐ = auxiliary / eval only

---

## Tier 1 — Best fit (do these first)

### 1. Skipper / ENCODE eCLIP windows ⭐⭐⭐

| Field | Value |
|-------|--------|
| **What** | Peak vs background RNA windows, 101–201 nt, 264 experiments |
| **Paper** | Boyle et al., *Cell Genomics* 2023 |
| **Access** | [Figshare 21206009](https://doi.org/10.6084/m9.figshare.21206009), Dropbox bundle — see `data/raw/eclip_skipper/README.md` |
| **Stats (local manifest)** | **173** gene symbols; **fixlen_151**: **736,347** pos + **736,347** neg (**151 nt**); domain annot **185** proteins |
| **Status** | Manifests in git; full archives may need download on VM |
| **Use** | **Train** (site-level, long context) + **protein-disjoint eval** (`scripts/41–41d`) |
| **Why top** | Same task family as RPIembeddor2; negatives are real genomic non-peaks, not catalog random pairs |

**Next action**: Verify `archives/figshare/` on VM; build train TSV with `max_per_class_per_experiment` + overlap audit vs `v3a` proteins.

---

### 2. ENCODE eCLIP IDR peaks (expand script 12) ⭐⭐⭐

| Field | Value |
|-------|--------|
| **What** | Reproducible IDR peaks + size-matched input; extract ±30 nt → **60 nt** (current pipeline) |
| **Papers** | Van Nostrand *Nat Methods* 2016 (**73** RBPs); *Nature* 2020 (**~488** experiments → **356** RBPs, **1,223** replicates) |
| **Access** | [ENCSR456FVU](https://www.encodeproject.org/publication-data/ENCSR456FVU/); extra **50** sets [GSE107768](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107768) |
| **Status** | **9** RBPs, **26,362** pairs in `data/sanitized/eclip/eclip_all.tsv` |
| **Use** | Merge into train (train-only assay tag) + assay-stratified metrics |
| **Why top** | Gold-standard in vivo **sites**; complements SELEX; flanks = hard negatives (`DATA.md` §1.4) |

**Next action**: Extend `TARGET_RBPS` in `scripts/12_download_eclip.py` to overlap Skipper roster; prefer RBPs already in `v3a` for transfer experiments.

---

### 3. RNAcompete compendium (in vitro) ⭐⭐⭐

| Panel | Proteins (approx.) | Clean TSV (sibling repo) | GEO / web |
|-------|-------------------|--------------------------|-----------|
| **Eukarya** | ~200 | `rnacompete_analysis/eukarya/.../ml_dataset_eukarya_clean.tsv.gz` | [Hughes Eukarya supp](https://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/) |
| **RBPZoo** (Sasse 2025) | **174** new + legacy | `rbpzoo/.../ml_dataset_rbpzoo_clean.tsv.gz` | [RBPZoo supp](https://hugheslab.ccbr.utoronto.ca/supplementary-data/RBPZoo/), **GSE192895** (~1 GB raw) |
| **ucRBP** (Ray 2023) | **492** attempted; **~63** sequence-specific | Partial in train via **23** whitelist | **GSE215198**, [datah.ccbr.utoronto.ca/ucRBP](http://datah.ccbr.utoronto.ca/ucRBP/) |

**Already in project**: Eukarya + RBPZoo + **23** reproducible ucRBP (`configs/ucrbp_23_reproducible.txt`) → `generalized_v3a`.

**High-value expansion** (not yet fully exploited):

1. **RBPZoo processed `txt.gz`** — confirm all **174** passed QC in `ml_dataset_rbpzoo_clean` (Table S1 domains already in `data/raw/rbpzoo/TableS1.xlsx`).
2. **ucRBP full panel** — use [RNAcompete_classifier](https://github.com/morrislab/RNAcompete_classifier) pass probabilities; add proteins above threshold with homology-aware split (not the failed 63 non-specific).
3. **Morris pipeline** — [rnacompete](https://github.com/morrislab/rnacompete) for reprobing raw batches if clean TSV gaps.

**Why top**: Same assay as core train; **probe_intensity** → Phase **3C** affinity; best negatives in the field (designed pool + flags).

---

### 4. RBNS + HTR-SELEX (ENCODE-linked) ⭐⭐⭐

| Field | Value |
|-------|--------|
| **What** | In vitro random pools; **R_max** affinity; 20 nt (RBNS) / 40 nt (SELEX) |
| **Access** | Sibling `rbns_analysis`, `htr_selex_analysis`; ENCODE [ENCSR876DCD](https://www.encodeproject.org/publication-data/ENCSR876DCD/) |
| **Status** | Core of `generalized_v3a` (**494** proteins, **2.66M** pairs) |
| **Use** | Keep as anchor; improve **negatives** (`scripts/15` hard negs — `DATA.md` §3) |
| **Why top** | Quantitative; same task; ENCODE 2020 links **same RBP names** as eCLIP/RBNS |

**Next action**: Pull `R_max` into Phase 3C; MMseqs2 homology audit (publication blocker).

---

## Tier 2 — Strong after ETL (site-level in vivo)

### 5. bbad307 unified CLIP benchmark ⭐⭐

| Field | Value |
|-------|--------|
| **Paper** | Paggi & Gagneur, *Brief Bioinform* 2023 — systematic benchmark of **11** methods |
| **Datasets** | **ENCODE eCLIP** (~223 exps, **150** RBPs); **iONMF** (31 exps, **19** RBPs); **Mukherjee PAR-CLIP** (59 exps, HEK293) |
| **Processing** | Homogenized peaks, **101 nt** windows, two negative-sampling strategies |
| **Use** | Import their peak→window protocol for **eval** and hard-negative ablations; compare to Skipper |
| **Link** | [bbad307](https://doi.org/10.1093/bib/bbad307) — supplementary lists all CLIP accessions |

**Why**: Negative-generation methodology directly addresses our SELEX easy-negative problem.

---

### 6. POSTAR3 bulk human CLIP sites ⭐⭐

| Field | Value |
|-------|--------|
| **What** | **1,499** CLIP-seq datasets curated to genomic sites |
| **Access** | [POSTAR3](http://postar.ncrnalab.org/) → RBP module → bulk **human.txt.gz** |
| **Use** | Mine peaks → extract windows → pair with ENCODE protein sequences; **not** raw train dump without assay tags |
| **Caveat** | Large; mixed protocols; must filter to eCLIP/iCLIP and match our RBP roster |

---

### 7. Antibody-barcode eCLIP (ABC) ⭐⭐

| Field | Value |
|-------|--------|
| **Paper** | Lorenz et al., *Nat Methods* 2023 |
| **GEO** | [GSE205536](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205536) |
| **Use** | More RBPs at lower cost; peaks comparable to ENCODE eCLIP (paper Fig. 1) |

---

### 8. iCLIP peaks (TDP-43, FUS, …) ⭐⭐

| Field | Value |
|-------|--------|
| **Roadmap** | `RESEARCH_ROADMAP.md` Phase 6 |
| **Use** | In vivo bridge for ALS-relevant RBPs; site-level if peaks + flanks extracted like eCLIP |
| **Access** | ENCODE / GEO per study (review Xiang 2024 refs.) |

---

## Tier 3 — Specialized (Phase 3C–5)

| Resource | What | Access | Use |
|----------|------|--------|-----|
| **Literature + affinities** | Mutants, comparative binding | `dataset_with_affinities.xlsx` (repo) | Phase **5** Δscore, hard negatives |
| **EuPRI / JPLE** | **504** in vitro + **34k** predicted motifs | RBPZoo portal, GSE192895 | Motif priors, **not** pairwise labels for all |
| **PDB protein–RNA** | Contact ground truth | PDB | Phase **7** structural benchmark subset |
| **DMS / ClinVar** | Variant effects | ClinVar, literature DMS | Phase **5** |

---

## Deprioritize (for motif-specific embedder)

| Resource | Reason |
|----------|--------|
| **PRIM-seq / HuRPA** | Transcript–protein association network (raw: `data/raw/geo_prim/suppl/GSE270010_RAW.tar` ✅) |
| **NPInter / RPI1460 / ZHMolGraph train** | Molecule-level DB labels |
| **starBase v2** | miRNA–ceRNA centric |
| **SARS-CoV-2 CLIP maps** | Viral context |
| **PRIM** until processed | Process with PRIMseqTools → optional Phase 6 **interactome** track only |

---

## Recommended acquisition order (next 2–4 weeks)

```text
1. Skipper archives complete → train/eval TSVs (fixlen_151)
2. eCLIP script 12 → 50–100 RBPs overlapping v3a / Skipper
3. ucRBP: classifier-pass subset from GSE215198 → homology merge (expand beyond 23)
4. Verify RBPZoo clean TSV covers all 174 in Sasse Table S1
5. bbad307: copy negative-sampling spec into `scripts/15_generate_hard_negatives.py` design
6. MMseqs2 on all protein sources before any new merge into `generalized_v3a`
```

---

## Quick comparison: where the leverage is

| If you want… | Best dataset |
|--------------|----------------|
| More **proteins** for zero-shot | RNAcompete **RBPZoo + ucRBP (QC-pass)** |
| **Long RNA** + site labels | **Skipper 151 nt** + eCLIP expansion |
| **Affinity** (T1) | RNAcompete `probe_intensity`, RBNS `R_max` |
| **Hard negatives** | eCLIP/Skipper flanks; bbad307 sampling; SELEX dinucleotide decoys |
| **In vivo eval** | Skipper protein-disjoint; iCLIP TDP-43/FUS |
| **Interactome** (separate head) | PRIM HuRPA (after pipeline) |

---

## Downloads started / local

| Asset | Path | Size |
|-------|------|------|
| GSE270009 RIP-seq | `data/raw/geo_prim/suppl/GSE270009_RAW.tar` | ~0.8 MB |
| GSE270010 PRIM-seq | `data/raw/geo_prim/suppl/GSE270010_RAW.tar` | **2.4 GB** ✅ |
| PRIM refSeq types | `data/raw/prim_seqtools/refSeq_tx_gene_type.csv` | 163,808 rows |

Update this file when a tier-1 source is merged into `data/sanitized/` or `scripts/`.
