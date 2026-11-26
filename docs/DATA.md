# Data Provenance, Leakage Risks, and Negative Sampling

**Last updated**: 2026-11-26

This document describes all datasets used in training and evaluation,
their known biases, their leakage risks, and the negative sampling strategy
applied to each. It is the authoritative reference for understanding what the
model has and has not been exposed to.

---

## 1. Training Datasets

### 1.1 HTR-SELEX PRJEB25907

| Property | Value |
|----------|-------|
| Reference | Ray et al., *Genome Research* 2019 (PMID 31285289) |
| Type | In vitro (HTR-SELEX, high-throughput RNA bind-n-seq) |
| RBPs | 93 |
| Rows in generalized pool | ~279,000 |
| Positive rate | ~33% |
| RNA lengths | 40 nt (fixed SELEX window) |
| Protein lengths | 50–500 aa (constructs, not full-length) |
| Negatives source | Sequences from final selection cycle with zero enrichment |
| Phase 1 best val AUROC | 0.825 (XGBoost) |

**Leakage risks**:
- Constructs (e.g. `ESRP1-construct3`) are protein sub-domains, not full proteins.
  Multiple constructs from the same protein can appear in the same split; the split
  key is construct name, not gene name.
- No homology clustering. Paralogs (RRM-domain family, hnRNP family) can be distributed
  across train and test.

**Negative quality**: Medium. Negatives are sequences from the same library that were
NOT enriched. They have the same GC content and length distribution as positives
(both come from the same random RNA pool), which makes them a reasonable baseline.
They are NOT naturally-occurring non-binding RNA fragments.

---

### 1.2 RBNS

| Property | Value |
|----------|-------|
| Reference | Lambert et al., *Nature* 2020 (PMID 33106656) |
| Type | In vitro (RNA Bind-n-Seq, affinity-based) |
| RBPs | 96 |
| Rows in generalized pool | ~284,642 |
| Positive rate | ~33% |
| RNA lengths | 20 nt (RBNS uses shorter windows than HTR-SELEX) |
| Protein lengths | variable |
| Negatives source | Random RNA sequences from the same pool (not enriched) |
| Phase 1 best val AUROC | 0.758 (RF) |
| Flagged proteins | RBM4, RBM4B (paralogs with near-identical binding), XRCC6 (atypical) |

**Critical length mismatch**: RBNS RNA = 20 nt, HTR-SELEX = 40 nt. This mismatch
breaks k-mer frequency models (k-mer counts are length-dependent). CNN with global
max pooling handles it correctly. Any future k-mer feature work must address this.

**Leakage risks**:
- RBM4 and RBM4B are confirmed paralogs that ended up in different splits.
  Their binding profiles are near-identical. Any model that sees RBM4 in training
  effectively "knows" RBM4B's binding specificity.
- RBNS uses random sequences as negatives; these are simpler than HTR-SELEX negatives.

**Negative quality**: Low. Negatives are randomly generated sequences, not sequences
that were tested and confirmed to not bind. The model learns to distinguish
"statistically unlikely sequence given the selected pool" from a positive.

---

### 1.3 HTR-SELEX PRJEB47428

| Property | Value |
|----------|-------|
| Reference | Laverty et al., *Nucleic Acids Research* 2022 |
| Type | In vitro (HTR-SELEX) |
| RBPs | 23 (small dataset) |
| Rows in generalized pool | ~69,000 |
| Positive rate | ~33% |
| RNA lengths | 40 nt |
| Negatives source | Same as PRJEB25907 |
| Phase 1 best val AUROC | 0.817 (LR) |
| Warning | Only 4 test proteins. Per-protein metrics have very high variance. |

---

### 1.4 eCLIP (partial, train-only)

| Property | Value |
|----------|-------|
| Reference | Van Nostrand et al., *Nature Methods* 2016; ENCODE |
| Type | In vivo (enhanced CLIP-seq) |
| RBPs | 10 (downloaded: HNRNPC, TARDBP, FUS, RBFOX2, ELAVL1, QKI, HNRNPA1, IGF2BP1, SLBP, RBFOX1) |
| Rows | ~26,362 (in generalized_v2 combined pool) |
| Split assignment | TRAIN ONLY — eCLIP positives (IP peaks) are NOT comparable to SELEX positives (enriched synthetic sequences). Mixing in val/test would make metrics incomparable. |
| Positive rate | ~33% |
| Negatives source | Flanking regions of same chromosome, same length, GC-matched (±0.05), no overlap with peaks |

**Why train-only**: eCLIP measures binding in cellular context (HepG2/K562). SELEX/RBNS
measures affinity in vitro with random sequences. The definitions of "positive" are
fundamentally different. Using eCLIP data in val/test while training on SELEX would
create a task where val/test labels have a different meaning than training labels.

**Negative quality**: HIGH. Flanking regions are real expressed RNA sequences with
matched GC content. This is the most biologically grounded negative set in the project.

**Scope**: 26,362 rows, 4% of the 659,004 rows of `generalized_v2`. An ablation of these rows was not run.

---

## 2. Evaluation Datasets

### 2.1 Internal Test Set

| Dataset | Rows | Proteins | Role |
|---------|------|----------|------|
| `generalized_v2` test | 98,662 | 169 (24 in test) | Phase 2 primary benchmark |
| `generalized_v3a` test | 322,275 | 494 (55 in test) | Phase 3A test set, protein-name split |
| `generalized_v3b` test | 321,027 | 494 (64 in test) | same pairs split by protein cluster, test set for claims about new proteins |
| v3a / v3b cross-protein probe | about 80,000 pairs (v3a), 63 proteins (v3b) | test proteins | same RNA paired with another protein of the same assay, built by `scripts/53` |

Same negative quality as training (SELEX/RBNS pool negs + eCLIP flanking in train only).

### 2.2 External Literature Dataset (`dataset_without_affinities.xlsx`)

- 159 pairs, 96 proteins, manually curated from literature
- 72% positive rate; 88% of proteins single-class (no per-protein AUROC on curated alone)
- 51-57% of RNAs exceed 60 nt (the count differs between `results/phase2_summary.json`, METHODS and this file, not re-counted), scored by window-max, which raises scores of long RNAs
- V2 (v3a checkpoint): AUROC **0.763**, AUPRC 0.915 (random AUPRC baseline = 0.717)
- These numbers are not comparable with SELEX or v3a test metrics (different RNA length, labels and negatives)

### 2.3 Expanded External Benchmark (`external_benchmark_expanded.tsv`)

Built by `scripts/31_build_external_benchmark.py`:
- **540 pairs** = 114 curated pos + 45 curated neg + **381 generated neg**
- Generated strategies per positive: `shuffle_uniform`, `shuffle_dinucleotide`, `cross_protein`, `cross_rna`
- 21% positive rate → random AUPRC baseline = **0.211** (not 0.5)
- V2 eval: AUROC 0.688, AUPRC 0.488, gain over random +0.28
- **Diagnostic only** — generated negs are decoys, not experimentally validated non-binders
- Manifest: `data/external/external_benchmark_expanded_manifest.json`

### 2.4 External Dataset with Affinities (`dataset_with_affinities.xlsx`)

- Contains mutation data: wild-type vs mutant binding comparisons
- Not yet used in training or evaluation
- Useful for: (a) auxiliary affinity regression, (b) mutation effect prediction,
  (c) hard negatives (confirmed non-binding mutants)

---

## 3. Negative Sampling — Current vs Recommended

| Dataset | Current negatives | Quality | Recommended |
|---------|------------------|---------|-------------|
| HTR-SELEX | Non-enriched seqs from same pool | Medium | Dinucleotide-matched decoys from full library |
| RBNS | Randomly generated sequences | Low | Dinucleotide + length-matched from same pool |
| eCLIP | Flanking regions, GC-matched | High | Keep as-is; expand to more RBPs |

### How to generate better SELEX/RBNS negatives

For each positive RNA sequence in the SELEX/RBNS pool:
1. Compute GC content and RNA length.
2. Sample N=2 sequences from the same protein's unselected pool that match
   GC content ±0.05 and length ±2 nt and have edit distance > 5 from the positive.
3. This preserves nucleotide composition and selection round context while
   ensuring the negative is not a near-duplicate of the positive.

This is implemented conceptually in `scripts/12_download_eclip.py` for eCLIP.
The same procedure for SELEX and RBNS is not implemented (the script name `15_generate_hard_negatives.py` used in an earlier plan does not exist). The in vitro cross-protein probe (`scripts/53`) is the negative set that removes RNA composition as a signal.

---

## 4. Split Strategy — Current Limitations

### 4.1 Protein-Aware (implemented)

All examples for a given protein are assigned to exactly one split.
No protein appears in more than one of train/val/test.
Implementation: `src/data/splits.py:protein_aware_split`.
Verified with assertion checks.

### 4.2 Homology-aware (implemented as `generalized_v3b`)

Two proteins with more than 30% sequence identity can have similar binding specificities. The v2 and v3a splits are by protein name, so homologs can fall on both sides. The homology audit (`scripts/55`, MMseqs2) gives for the 55 v3a test proteins: no hit 6, 30-50% identity 15, 50-70% 14, 70-90% 13, >= 90% 7, so 62% have a training protein at >= 50% identity (examples: ESRP1-construct3 against ESRP1, RBM4 against RBM4B). Probe AUROC follows this identity (V4: 0.525 without hit, 0.884 at 70-90%). `generalized_v3b` (`scripts/58`) assigns whole protein clusters (40% identity at 50% coverage, or 70% identity at 10% coverage) to one split, and 3% of its 64 test proteins have a training protein at >= 50% identity. Details: `EXPERIMENT_LOG.md` (2026-10-29) and `docs/METHODS.md` section 3.4. Not implemented: an MMseqs2 audit of the 24 `generalized_v2` test proteins, and a threshold of 30% (it joins all RRM proteins into one cluster).

### 4.3 RNA-Aware (NOT implemented — required for fair comparison to ZHMolGraph)

ZHMolGraph holds out both unseen proteins AND unseen RNA sequences simultaneously.
Our current split only holds out proteins. The same RNA sequences can appear in both
training (with protein A) and test (with protein B).

**Impact**: models that memorize RNA-level features will appear to generalize when
they are actually exploiting RNA sequence identity.

---

## 5. Column Name Conventions

Different pipeline stages use different column names for the same field.
This caused a bug where per-protein dataset labels were all "unknown".

| Stage | Protein label column | Dataset source column |
|-------|---------------------|----------------------|
| `data/generalized/` (scripts/04) | `protein_name` | `dataset_source` |
| `data/generalized_v2/` (scripts/14) | `protein_name` | `dataset` |
| All training scripts | expect `protein_name` + `dataset` | normalise at load time |

**Fix applied in scripts/06**: at load time, `dataset_source` is renamed to `dataset`
if the latter column is absent. All future scripts should follow this convention.

**Standard going forward**: use `dataset` as the column name in all output TSVs.
The `dataset_source` name in `data/generalized/` is legacy and should not be replicated.

---

## 6. Data files, dataset versions and upstream repositories

`data/` is not tracked in git (`.gitignore`), apart from small files. Files are rebuilt with the scripts or copied from the GPU VM.

| Path | Content | Built by |
|---|---|---|
| `data/generalized_v2/{train,val,test}.tsv` | Phase 2 pool, 659,004 rows (about 490k / 71k / 99k), 169 proteins | `scripts/04`, `scripts/14` |
| `data/embeddings/esm2_*_embeddings.npz` | ESM-2 mean-pool (168 proteins x 1280) and residue embeddings (fp16) | `scripts/07`, `07b` |
| `data/generalized_v3a/{train,val,test}.tsv` | 2,658,999 pairs, 494 proteins | `scripts/22` |
| `data/sanitized/generalized_v3a/` | the same TSVs with cleaned `protein_sequence` | `scripts/40 --apply` |
| `data/generalized_v3b/` | v3a pairs split by protein cluster | `scripts/58 build` |
| `data/external/` | literature pairs (`dataset_without_affinities.xlsx`, expanded benchmark) | `scripts/31` |
| `data/benchmarks/` | RNAcompete benchmark, Skipper eCLIP splits, probes | scripts 17, 41-42, 53 |
| `data/generalized/` | legacy pool of `scripts/04` | superseded by `generalized_v2` |

**Sanitization (`scripts/40`)**: audit-only by default. With `--apply` it writes `*.sanitized.tsv` (or a copy under `data/sanitized/`, or in place with `.bak`). `src/data/protein_sequence.py` strips a trailing `*`, removes embedded digits (for example residue numbers in RBM38) and other non-amino-acid characters, and rows whose sequence becomes empty are dropped. Scripts 38, 41, 51, 53 and `scripts/run_eclip_step1.sh` read `data/sanitized/generalized_v3a/`. Scripts 55, 58 and the Phase 3A training read `data/generalized_v3a/`. Row counts of the two copies were not compared here.

**Upstream repositories**: the raw SELEX, RBNS and RNAcompete tables are produced by `htr_selex_analysis`, `htr_selex_prjeb47428`, `rbns_analysis` and `rnacompete_analysis`, which are separate git repositories (configs address them as `../<name>/`, next to this folder; in the present working copy they are inside the project folder). They are needed to rebuild the pools from scratch (see `docs/DATA_SOURCES_AND_DOWNLOADS.md`).
