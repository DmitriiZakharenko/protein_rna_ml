# Research Roadmap — Beyond Binary Binding Classification

**Project**: Protein–RNA Binding Prediction  
**Status**: Planning document (extends Phase 3A)  
**Last updated**: 2026-06-25  
**Prerequisite**: Phase 3A dataset (`data/generalized_v3a/`) and scaled model training  
**Related documents**: `PHASE3A_PLAN.md`, `STRATEGY.md`, `METHODS.md`, `DATA.md`

---

## 1. Purpose

This document defines the medium-term research direction for the project. Phase 1–3A
establish that high-quality binding signal exists in the data and that cross-protein
generalization requires more training diversity and explicit interaction modeling.

The next stages shift the scientific question from:

> *Does this protein bind this RNA?*

to:

> *How strongly, where, under what conditions, and with what consequence for
> sequence variation and cellular context?*

The goal is a model family with interpretable biological outputs and evaluable
relevance to RNA regulation and disease-associated variation.

---

## 2. Current Scope and Limitations

### 2.1 What the current pipeline measures

| Capability | Status |
|------------|--------|
| Per-protein binding classification (RNA-only) | Strong (AUROC 0.95–0.99) |
| Cross-protein binding classification | Moderate (V2 test AUROC 0.69) |
| Zero-shot generalization to unseen RBPs | Weak (RNAcompete median AUROC 0.55) |
| Binding affinity (continuous) | Not modeled |
| Binding site localization | Not modeled |
| Variant effect prediction | Not modeled |
| In vivo validation | Partial (eCLIP train-only, small fraction) |

### 2.2 Structural constraints of the present task

1. **Binary labels** from enrichment assays do not encode affinity gradients.
2. **In vitro assays** (SELEX, RBNS, RNAcompete) may not reflect in-cell binding.
3. **Construct vs full-length proteins** differ across data sources.
4. **Easy negatives** in SELEX/RBNS may inflate apparent performance.
5. **Protein-aware splits** do not fully control homology leakage.

These limitations are methodological, not evidence of poor data quality. They define
what must change for the model to support biological and clinical interpretation.

---

## 3. Research Objectives

### 3.1 Primary objectives

1. **Quantitative binding** — predict relative or absolute binding strength, not only
   class membership.
2. **Interaction localization** — identify which RNA motif positions and protein regions
   contribute to the predicted interaction.
3. **Cross-assay generalization** — train on in vitro data; evaluate transfer to
   independent assays and, where possible, in vivo binding maps.
4. **Variant sensitivity** — quantify how amino-acid substitutions in RNA-binding
   proteins alter predicted binding to cognate RNA elements.
5. **Interpretable outputs** — link predictions to known domain architectures, motif
   families, and regulatory contexts.

### 3.2 Secondary objectives

1. **Therapeutic sequence design** — use the model as a scorer for candidate RNA
   oligonucleotides (decoys, splice-switching elements, competitive binders).
2. **Benchmark contribution** — publish homology-aware splits, hard negatives, and
   multi-task evaluation protocols alongside model checkpoints.
3. **Assay gap analysis** — systematically measure where in vitro predictions agree
   or disagree with CLIP-derived binding evidence.

---

## 4. Capability Tiers

The project evolves through progressively richer model outputs. Each tier builds on
the previous; skipping tiers risks uninterpretable or unverifiable claims.

| Tier | Model output | Primary data | Evaluation |
|------|--------------|--------------|------------|
| **T0** | Binding probability | SELEX, RBNS, RNAcompete (binary) | AUROC, AUPRC, per-protein median |
| **T1** | Binding probability + affinity rank | `probe_intensity`, `R_max` | Spearman ρ, RMSE on held-out affinities |
| **T2** | T1 + motif / domain attribution | Enriched k-mers, UniProt domains, eCLIP peaks | Attribution consistency, motif recovery |
| **T3** | T2 + variant effect (Δscore) | ClinVar, DMS, literature variant pairs | Pathogenic vs benign separation |
| **T4** | T3 + structural features | RNA secondary structure, predicted domain folds | Agreement with PDB contacts (subset) |
| **T5** | Generative / design scoring | SELEX enriched distributions | Enrichment of designed sequences in assay |

**Current position**: T0 (V2 CNN). Phase 3A–3B target improved T0 with more proteins
and interaction layers. Phase 3C onward advances to T1–T3.

---

## 5. Phased Roadmap

### Phase 3A — Training data scale-up *(complete 2026-07)*

**Goal**: Increase protein diversity by merging SELEX/RBNS with a curated RNAcompete
training subset (Eukarya and RBPZoo full; ucRBP restricted to 23 reproducible RBPs).

**Deliverables**: `data/generalized_v3a/`, V2 retrained — test AUROC **0.813** (2026-07). Multi-seed + RNAcompete re-eval pending.

**Document**: `PHASE3A_PLAN.md`

---

### Phase 3B — Explicit interaction modeling

**Goal**: Replace independent branch concatenation with bilinear or cross-attention
interaction between RNA and protein representations.

**Model**: V4 (`scripts/21_train_generalized_v4_interaction.py`)  
**Hypothesis**: Joint features improve both in-distribution test performance and
zero-shot transfer when combined with Phase 3A data.

**Success criteria**: Test AUROC improvement over V2 on `generalized_v3a` exceeds
multi-seed variance; zero-shot median AUROC ≥ 0.60.

---

### Phase 3C — Affinity and ranking

**Goal**: Add a regression head trained on continuous binding readouts alongside
classification.

| Source | Target variable | Loss |
|--------|-----------------|------|
| RNAcompete | `probe_intensity` (or Z-score) | MSE / Huber on log-scale |
| RBNS | `R_max` | MSE on log-scale |
| SELEX/RBNS | `binding_label` | BCE (existing) |

**Combined loss**: `L = L_cls + λ · L_reg` (λ tuned on validation).

**Evaluation**:
- Classification metrics unchanged (AUROC, AUPRC).
- Ranking: Spearman ρ between predicted and observed intensity per protein.
- Calibration: reliability diagrams on held-out RNAcompete proteins.

**Biological meaning**: Distinguishes strong vs weak binders within the positive
class; essential for modeling partial loss of function under mutation.

---

### Phase 4 — Binding site localization

**Goal**: Predict *where* binding occurs, not only *whether*.

#### 4.1 RNA motif localization

**Auxiliary task**: predict start position (or distribution) of the highest-scoring
7-mer window within the RNA sequence.

**Weak labels**: top enriched k-mers from SELEX/RNAcompete analysis
(`scripts/24_extract_top_bottom_examples.py`).

**Metric**: fraction of positives where predicted window overlaps annotated motif (±1 nt).

#### 4.2 Protein region attribution

**Auxiliary task**: residue-level importance scores on the protein sequence, aligned
with known RNA-binding domains (RRM, KH, Pumilio, etc.).

**Weak labels**: UniProt domain annotations; eCLIP peak-supported regions where
available.

**Metric**: enrichment of attribution mass within annotated domains vs flanking regions.

**Biological meaning**: Connects predictions to mechanistic hypotheses (which domain
mediates recognition) and supports variant interpretation (domain-local vs
dispersed effects).

---

### Phase 5 — Variant effect prediction

**Goal**: Predict change in binding upon amino-acid substitution:

```
Δscore = f(RBP_mut, RNA) − f(RBP_wt, RNA)
```

#### 5.1 Data sources

| Source | Type | Use |
|--------|------|-----|
| ClinVar / gnomAD | Missense in RBP genes | Pathogenic vs benign benchmark |
| Deep mutational scanning (DMS) | Systematic substitutions | Gold-standard Δaffinity |
| Literature | Curated variant–phenotype pairs | Qualitative validation |

#### 5.2 Evaluation protocol

- Hold out entire genes (not individual variants) to prevent leakage.
- Report AUROC for pathogenic vs benign among variants in RNA-binding domains.
- Report correlation with DMS measurements where available.
- Stratify by domain class (RRM, intrinsically disordered, etc.).

#### 5.3 Disease-relevant RBP priorities

Proteins with strong disease genetics and available CLIP or structural data:

| Gene | Context |
|------|---------|
| TARDBP (TDP-43) | ALS / FTD |
| FUS | ALS / FTD |
| RBFOX1/2 | Splicing regulation |
| HNRNPA1/A2B1 | Myopathy, neurodegeneration |
| PTBP1 | Splicing, development |
| MATR3 | ALS |
| QKI | Myelination, cancer |

**Biological meaning**: Direct link between model output and interpretable clinical
variant classification in RNA-binding proteins.

---

### Phase 6 — In vitro / in vivo bridge

**Goal**: Quantify and reduce the distributional gap between selection-based assays
and transcriptome-wide binding maps.

#### 6.1 Data integration

| Assay | Role | Status |
|-------|------|--------|
| HTR-SELEX, RBNS, RNAcompete | Primary training (in vitro) | Integrated |
| eCLIP | Hard negatives + weak localization labels | Partial (train-only) |
| iCLIP | Peak-level binding for TDP-43, FUS | Acquisition planned |
| PAR-CLIP | miRNA/RISC context (AGO2) | Future |

#### 6.2 Evaluation modes

1. **Train in vitro, test on CLIP peaks** — binary binding on peak vs flank pairs.
2. **Assay-stratified metrics** — report performance separately per `dataset_source`.
3. **Assay gap report** — proteins where in vitro and in vivo rankings disagree.

**Biological meaning**: States explicitly whether the model captures cellular binding
or only biochemical preference in a controlled pool.

---

### Phase 7 — Structural and thermodynamic features

**Goal**: Incorporate RNA secondary structure and protein domain structure as
additional input channels, without requiring experimental 3D structures for every
prediction.

#### 7.1 Feature classes

| Feature | Source | Cost |
|---------|--------|------|
| RNA MFE, pairing probability | ViennaRNA / RNAfold | CPU batch |
| Base-pair accessibility | RNAplfold | CPU batch |
| Protein domain boundaries | UniProt, InterPro | Annotation |
| Predicted domain folds | ESMFold or database models | GPU batch (one-time) |
| Known complex contacts | PDB protein–RNA structures | Benchmark subset only |

#### 7.2 Integration strategy

1. Append structure features as additional input channels to the V4 encoder.
2. Evaluate on a PDB-held-out set of protein–RNA complexes (contact prediction).
3. Compare T1/T2 models with and without structure features on the same splits.

**Biological meaning**: Structure-aware models are required for RBPs whose recognition
depends on RNA conformation (e.g. stem–loop binders) and for evaluating predictions
against physical contact data where available.

---

### Phase 8 — Therapeutic and design applications

**Goal**: Demonstrate utility on a single well-defined design problem.

#### 8.1 Candidate tasks (select one for first case study)

| Task | Input | Output | Application area |
|------|-------|--------|------------------|
| Decoy RNA design | RBP sequence | High-affinity, specific RNA oligo | Competitive inhibition |
| Splice-site targeting | Splicing RBP + pre-mRNA context | Binding score landscape | Splice-switching therapy |
| miRNA target ranking | AGO-loaded miRNA + 3′UTR | Site occupancy score | Target validation |

#### 8.2 Protocol

1. Define RBP and regulatory context from literature.
2. Generate candidate RNA sequences (enumerated or generative).
3. Score with trained model (T1 or higher).
4. Validate top candidates against known functional sites or published assays.

**Deliverable**: One documented case study with reproducible scripts and honest
limitations.

---

## 6. Evaluation Framework

Metrics expand with capability tier. All reported numbers require multi-seed variance
(established in `STRATEGY.md`).

### 6.1 Core metrics (all phases)

| Metric | Use |
|--------|-----|
| AUROC, AUPRC | Threshold-independent classification |
| Per-protein median AUROC | Heterogeneity across RBPs |
| Brier score | Calibration |
| Multi-seed mean ± std | Reproducibility |

### 6.2 Tier-specific metrics

| Tier | Additional metrics |
|------|-------------------|
| T1 | Spearman ρ (affinity), Kendall τ, RMSE on log-intensity |
| T2 | Motif recovery rate, domain attribution enrichment |
| T3 | Variant AUROC (pathogenic vs benign), Δscore correlation with DMS |
| T4 | Contact F1 on PDB subset, Δperformance with vs without structure |
| T5 | Enrichment of top-scored designs in retrospective assay data |

### 6.3 Reporting requirements

1. Separate results for in vitro test, zero-shot benchmark, and in vivo holdout.
2. Homology-stratified breakdown (seen vs unseen protein families).
3. Negative-type breakdown (easy vs hard negatives) when hard negatives are introduced.
4. Explicit statement of label semantics per assay (documented in `DATA.md`).

---

## 7. Data Acquisition Plan

Parallel data collection to support Phases 5–7.

### 7.1 High priority

| Resource | Purpose | Phase |
|----------|---------|-------|
| Ray & Laverty 2023 Supplementary Table S1 | Verify ucRBP whitelist | 3A |
| MMseqs2 all-vs-all (SELEX proteins) | Quantify split leakage | 3A–3B |
| ClinVar missense in project RBPs | Variant benchmark | 5 |
| iCLIP peaks (TDP-43, FUS) | In vivo validation | 6 |
| eCLIP expansion (ENCODE) | Hard negatives, localization | 6 |

### 7.2 Medium priority

| Resource | Purpose | Phase |
|----------|---------|-------|
| RBNS `R_max` in committed TSVs | Regression targets | 3C |
| UniProt domain annotations | Attribution labels | 4 |
| ViennaRNA features for training RNAs | Structure channel | 7 |
| PDB protein–RNA complexes | Structural benchmark | 7 |
| DMS datasets for RBPs | Variant gold standard | 5 |

### 7.3 Lower priority

| Resource | Purpose | Phase |
|----------|---------|-------|
| RNAInter | Interaction graph, hard negatives | 6+ |
| Precomputed RNA language model embeddings | Alternative RNA encoder | 7 |
| Expanded literature binding set (>500 pairs) | External validation | 8 |
| DisGeNET / OMIM gene–disease mapping | Application framing | 5, 8 |

---

## 8. Dependencies Between Phases

```
Phase 3A (data scale)
    └── Phase 3B (interaction layer)
            └── Phase 3C (affinity regression)
                    ├── Phase 4 (localization)
                    │       └── Phase 7 (structure features)
                    ├── Phase 5 (variant effects)
                    └── Phase 6 (in vivo bridge)
                            └── Phase 8 (design case study)
```

**Hard dependencies**:
- Phase 5 requires Phase 3C (Δscore on a calibrated affinity scale is more
  interpretable than Δprobability near 0.5).
- Phase 6 requires eCLIP/iCLIP acquisition before in vivo claims.
- Phase 8 requires at least T1 (ranking) for meaningful design scoring.

**Soft dependencies**:
- Phase 4 can start in parallel with Phase 3C (auxiliary losses on existing labels).
- Phase 7 is optional for Phases 5–6 but required for structure-dependent RBPs.

---

## 9. Out of Scope (Current Cycle)

The following are deferred to avoid scope creep before T1–T3 are established:

1. End-to-end structural complex prediction from sequence alone.
2. Full transcriptome-scale binding prediction (requires genomic context and
   expression data not in current datasets).
3. Training on raw CLIP read counts without peak calling and negative-matched controls.
4. Clinical deployment or diagnostic certification.
5. Integration of small-molecule ligands (protein–RNA–small-molecule ternary complexes).

---

## 10. Document Maintenance

Update this file when:
- A phase is completed (change status, add result pointers in `EXPERIMENT_LOG.md`).
- New data sources are integrated (update §7 and `DATA.md`).
- Success criteria are revised based on Phase 3A–3B outcomes.

**Experiment log**: `EXPERIMENT_LOG.md`  
**Immediate execution plan**: `PHASE3A_PLAN.md`  
**Lessons and anti-patterns**: `STRATEGY.md` §4
