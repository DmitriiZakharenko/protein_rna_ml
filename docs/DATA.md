# Data Provenance, Leakage Risks, and Negative Sampling

**Last updated**: 2026-07-11

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

**Current limitation**: Only 4% of the total training data. Not yet confirmed whether
the eCLIP examples are helping or hurting SELEX/RBNS generalization.

---

## 2. Evaluation Datasets

### 2.1 Internal Test Set

| Dataset | Rows | Proteins | Role |
|---------|------|----------|------|
| `generalized_v2` test | 98,662 | 169 (24 in test) | Phase 2 primary benchmark |
| **`generalized_v3a` test** | **322,275** | **494 (55 in test)** | **Phase 3A primary benchmark** |

Same negative quality as training (SELEX/RBNS pool negs + eCLIP flanking in train only).

### 2.2 External Literature Dataset (`dataset_without_affinities.xlsx`)

- 159 pairs, 96 proteins, manually curated from literature
- 72% positive rate; 88% of proteins single-class (no per-protein AUROC possible)
- 57% of RNAs exceed 60 nt (scored by window-max, inflates AUPRC for long lncRNAs)
- 159 pairs, 96 proteins, manually curated from literature
- 72% positive rate; 88% of proteins single-class (no per-protein AUROC on curated alone)
- 57% of RNAs exceed 60 nt (scored by window-max, inflates AUPRC for long lncRNAs)
- V2 (v3a checkpoint): AUROC **0.763**, AUPRC 0.915 (random AUPRC baseline = 0.717)
- **DO NOT** compare these numbers to SELEX/v3a test metrics

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
Equivalent logic needs to be added for SELEX/RBNS in a new script `scripts/15_generate_hard_negatives.py`.

---

## 4. Split Strategy — Current Limitations

### 4.1 Protein-Aware (implemented)

All examples for a given protein are assigned to exactly one split.
No protein appears in more than one of train/val/test.
Implementation: `src/data/splits.py:protein_aware_split`.
Verified with assertion checks.

### 4.2 Homology-Aware (NOT implemented — required before publication)

**Problem**: Two proteins with >30% sequence identity may have nearly identical
binding specificities. If one is in training and the other in test, the model
can transfer knowledge via homology.

**Known affected pairs** (not exhaustive):
- RBM4 / RBM4B (confirmed paralogs in RBNS, split across train/test)
- hnRNP family members (multiple constructs)
- SR protein family (SRSF1–SRSF9 have similar RRM domains)

**Required action**:
1. Run MMseqs2 `easy-cluster` at 30% identity, 80% coverage on all 169 protein sequences.
2. Assign entire clusters to the same split.
3. Report metrics separately for "clean" test proteins (no training-set homolog) and
   "contaminated" test proteins (training-set homolog exists).

Until this is done, val/test AUROC numbers may overestimate true generalization.

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

## 6. Data Files On Disk vs In-Repo

| File | In repo | Notes |
|------|---------|-------|
| `data/generalized_v2/train.tsv` | Yes | ~490k rows, use this for all training |
| `data/generalized_v2/val.tsv` | Yes | ~71k rows |
| `data/generalized_v2/test.tsv` | Yes | ~99k rows |
| `data/embeddings/esm2_protein_embeddings.npz` | Yes | 168 proteins × 1280-d, mean-pool |
| `data/embeddings/esm2_residue_embeddings.npz` | Yes | 168 proteins × 300 aa × 1280-d, fp16 |
| `data/external/*.tsv` | Yes | eCLIP per-protein output |
| Raw SELEX/RBNS TSVs | **NO** | Live in sibling repo dirs (`../htr_selex_analysis/`, `../rbns_analysis/`) |
| `data/generalized/` (legacy) | Yes (small files) | Generated by scripts/04. Legacy. Use `generalized_v2` instead. |

**Reproducibility gap**: A fresh clone of this repository cannot reproduce the
generalized dataset from scratch without access to the raw SELEX/RBNS TSVs which
live outside this repo. The `data/generalized_v2/` TSVs are committed and are the
canonical starting point for all Phase 2 training.
