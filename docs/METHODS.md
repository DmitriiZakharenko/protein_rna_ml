# Methods & Model Tracking

**Project**: Protein–RNA Binding Prediction  
**Last updated**: 2026-11-26  
**Language**: English  

This document records the methodological choices of the project: encodings, splits, model architectures, training configurations and the result tables of Phases 1-3, with the rationale and literature references. Results of the later work (v3b, probe, frozen-embedding heads, audits) are recorded in `EXPERIMENT_LOG.md`, and section 3.3 and section 9 summarise them.

---

## Table of Contents

1. [Evaluation Protocol](#1-evaluation-protocol)
2. [Sequence Encoding](#2-sequence-encoding)
3. [Dataset Splitting Strategy](#3-dataset-splitting-strategy)
4. [Baseline Models](#4-baseline-models)
5. [Literature Benchmarks](#5-literature-benchmarks)
6. [Methods planned at the start of Phase 2](#6-methods-planned-at-the-start-of-phase-2)
7. [Results Summary Table](#7-results-summary-table)
8. [Methodological limitations](#8-methodological-limitations)
9. [Additional result tables](#9-additional-result-tables)
10. [Domain annotation](#10-domain-annotation)

---

## 1. Evaluation Protocol

### 1.1 Two-Stage Validation Pipeline

**Design choice**: Before a dataset is included in the generalized model, it must pass a standalone validation with a small baseline model.

**Rationale**: Protein–RNA binding datasets vary enormously in quality. In vitro methods (HTR-SELEX, RBNS) enrich RNA sequences over multiple rounds; background controls differ between studies; some proteins have weak or noisy signal. Running a quick baseline model on each dataset independently detects these problems before they contaminate the generalized training pool. A dataset that cannot produce AUROC > 0.70 on unseen proteins with even a simple k-mer model is considered suspect and requires investigation.

**Pass/fail thresholds**:

| Metric | PASS | WARN | FAIL |
|---|---|---|---|
| Overall AUROC (test set) | ≥ 0.70 | 0.60–0.70 | < 0.60 |
| Per-protein median AUROC | ≥ 0.65 | — | — |
| Per-protein min AUROC | ≥ 0.50 | flags individual proteins | — |

**Reference**: General principle from [Chicco & Jurman (2020), BMC Genomics](https://doi.org/10.1186/s12864-019-6413-7) — AUROC and AUPRC as primary metrics for imbalanced binary classification.

---

### 1.2 Primary Metrics

| Metric | Abbreviation | Why used |
|---|---|---|
| Area Under ROC Curve | AUROC | Threshold-independent; standard for binary classification; robust to class imbalance |
| Area Under Precision-Recall Curve | AUPRC | More informative than AUROC when positives are rare or the positive class is the focus; our class ratio is 1:2 |
| Accuracy | Acc | Interpretable for reporting; less informative than AUROC for imbalanced data |
| F1 Score | F1 | Harmonic mean of precision and recall; sensitive to threshold choice |
| Brier Score | BS | Measures probability calibration; lower is better; useful if model outputs are used as confidence scores downstream |

**Note on F1**: Random Forest and XGBoost can show near-zero F1 at default threshold 0.5 while maintaining good AUROC. This happens when the model assigns high ranks to positives but places the decision boundary too high. AUROC is the reliable metric; F1 should be computed after threshold calibration on the validation set.

---

## 2. Sequence Encoding

### 2.1 k-mer Frequency Vectors

**What it is**: For a sequence of length L, count all occurrences of every possible subsequence of length k (k-mer). Divide each count by the total number of k-mers extracted (L − k + 1) to get a normalized frequency vector. Concatenate the RNA and protein frequency vectors into a single feature vector per (protein, RNA) pair.

**Current settings**:
- RNA: k = 4 → 4^4 = **256 features** (alphabet: A, U, G, C)
- Protein: k = 3 → 20^3 = **8,000 features** (20 standard amino acids)
- Total feature vector size: **8,256 per example**

**Rationale for k = 4 (RNA)**:  
4-mers capture the minimal binding motif unit for most RNA-binding proteins. The majority of characterized RBP binding motifs (RBPmap, ATtRACT database) are 4–6 nucleotides long. Using k = 3 loses too much specificity (64 features, underfits); k = 5 gives 1,024 features but is extremely sparse for short 20 nt RBNS sequences (only 16 possible 5-mers per sequence). k = 4 is the established standard in the field.

**Rationale for k = 3 (protein)**:  
20^3 = 8,000 covers all amino acid triplets and captures local residue composition patterns (e.g., RRM domains, KH domains, zinc-finger regions) without excessive dimensionality. k = 2 (400 features) is too coarse; k = 4 (160,000 features, mostly zero) is too sparse for proteins of typical length.

**Known limitations**:  
- Loses all positional information (sequence order is discarded)
- Cannot capture long-range dependencies or secondary structure
- Two sequences with identical composition but different order get the same vector
- Struggles to generalize to proteins with novel domain compositions not seen in training

**References**:
- [Ghandi et al. (2014), PLoS Computational Biology](https://doi.org/10.1371/journal.pcbi.1003711) — k-mer features for DNA binding specificity
- [Ray et al. (2013), Nature](https://doi.org/10.1038/nature12311) — RNA binding motif landscape; motivates 4–6 nt as minimal motif length
- [ATtRACT database](https://attract.cnic.es/) — curated RBP binding motifs confirming 4–6 nt motif sizes

---

## 3. Dataset Splitting Strategy

### 3.1 Protein-Aware Split

**What it is**: The dataset is divided into train / validation / test subsets such that **no protein appears in more than one subset**. All examples (positive and negative) for a given protein are assigned to the same split.

**Current split ratios (by number of proteins)**:

| Dataset | Train | Val | Test |
|---|---|---|---|
| HTR-SELEX PRJEB25907 | 70 proteins (75%) | 10 (11%) | 13 (14%) |
| RBNS | 72 proteins (75%) | 11 (11%) | 13 (14%) |
| HTR-SELEX PRJEB47428 | 16 proteins (70%) | 3 (13%) | 4 (17%) |

**Rationale**:  
A random split at the example level would mix the same protein across train and test. In that case, a model could achieve high test AUROC simply by memorising which RNA sequences co-occur with which protein — without learning any transferable binding principle. Protein-aware splitting enforces that test-set proteins are genuinely unseen during training, providing an honest estimate of generalisation to new RBPs. This is the biologically meaningful evaluation setting and mirrors the real deployment scenario (predicting binding for a protein with no prior SELEX/RBNS data).

This design is stricter than the standard random split used in older RBP prediction papers, and directly mirrors the real deployment scenario of predicting binding for a novel protein with no prior experimental data.

**Random seed**: 42 (fixed throughout; all splits reproducible).

**Reference**:
- [Ghanbari & Ohler (2020), Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbz103) — discusses protein-aware vs random splits for RBP prediction; demonstrates that random splits systematically overestimate performance by 0.05–0.15 AUROC

---

### 3.3 Dataset versions and naming

| Name | Path | Build | Split |
|---|---|---|---|
| `generalized_v2` | `data/generalized_v2/` | `scripts/04`, `scripts/14` | by protein name, 24 test proteins, 98,662 test rows |
| `generalized_v3a` | `data/generalized_v3a/` | `scripts/22` (HTR-SELEX + RBNS + RNAcompete Eukarya/RBPZoo + ucRBP 23) | by protein name, train 393 / val 49 / test 55 proteins, 2,658,999 pairs |
| sanitized v3a | `data/sanitized/generalized_v3a/` | `scripts/40 --apply` | built from the v3a TSVs (row counts of the two copies were not compared). `protein_sequence` is cleaned: trailing `*` removed, digits and non-amino-acid characters removed (`src/data/protein_sequence.py`), rows with an empty sequence dropped |
| v3a, known domains | `known_domains_only` option of `scripts/38` | filter on sanitized v3a | 465 of 619 protein variants kept (variants with a known domain class) |
| `generalized_v3b` | `data/generalized_v3b/` | `scripts/58 build` from `results/v3b/protein_split_map.tsv` | same 2,658,999 pairs split by protein cluster: train 375 / val 55 / test 64 proteins |

`scripts/55`, `scripts/58` and `scripts/40` read `data/generalized_v3a/`. Scripts 38, 41, 51, 53 and `scripts/run_eclip_step1.sh` read the sanitized copy by default. Which copy the Phase 3A checkpoints of V2 (0.813) and V4 (0.829) were trained on is recorded in neither the log nor the summary JSON files. Domain-aware V2 numbers (baseline 0.847) are on the filtered sanitized copy and are not comparable with 0.813.

### 3.4 Homology between splits

The v3a and v2 splits assign proteins by name or construct name, so paralogs can fall on different sides. The homology audit (`scripts/55`, MMseqs2 all-vs-all, best local identity of each test protein to any training protein) gives, for the 55 v3a test proteins: no hit 6, 30-50% 15, 50-70% 14, 70-90% 13, >= 90% 7 (62% at >= 50%). Probe AUROC rises with this identity (log, 2026-10-29). `scripts/58` links two proteins when identity is >= 40% at >= 50% coverage or >= 70% at >= 10% coverage (single linkage, 160 clusters, largest 220 proteins), assigns whole clusters to train, val or test stratified by Pfam family (seed 42) and gives 3% of the 64 test proteins a training protein at >= 50% identity. The 30% identity threshold joins all RRM proteins into one cluster of more than 400 proteins, so 40% is the lowest usable threshold. Domain-level similarity at 30-50% remains for most v3b test proteins.

---

### 3.2 Class Balance

**Current ratio**: ~1 positive : 2 negatives (33% positive rate) in all three datasets.

**Rationale**:  
A 1:1 ratio would oversample positives or undersample negatives, potentially inflating AUROC estimates. A 1:2 ratio (or 1:3) is commonly used in RBP prediction literature and approximates realistic biological conditions where the genome contains fewer binding sites than non-binding regions. No resampling or class weighting is applied at baseline; XGBoost's `scale_pos_weight` may be explored if class balance issues arise on individual proteins.

---

## 4. Baseline Models

### 4.1 Logistic Regression

**Library**: `sklearn.linear_model.LogisticRegression`  
**Input**: 8,256-dimensional k-mer frequency vector  
**Hyperparameters**: C = 1.0, max_iter = 1000, solver = lbfgs (default)

**What it does**: Fits a linear decision boundary in 8,256-dimensional k-mer space. Equivalent to learning a weighted sum of k-mer frequencies.

**Why included**:
- Serves as a linear baseline; if logistic regression performs well, the problem is linearly separable in k-mer space
- Fast (< 3 min on 200k examples)
- Coefficients are interpretable: the k-mers with largest positive weights are those most associated with binding
- Robust to class imbalance in this regime

**Observed performance (val set)**:
- HTR-SELEX PRJEB25907: AUROC = 0.771, AUPRC = 0.640
- RBNS: AUROC = 0.746, AUPRC = 0.636

**Reference**: [Pedregosa et al. (2011), JMLR](https://jmlr.org/papers/v12/pedregosa11a.html) — scikit-learn

---

### 4.2 Random Forest

**Library**: `sklearn.ensemble.RandomForestClassifier`  
**Input**: 8,256-dimensional k-mer frequency vector  
**Hyperparameters**: n_estimators = 200, max_depth = 15, n_jobs = −1

**What it does**: Ensemble of decision trees trained on random subsets of features and examples. Captures non-linear interactions between k-mers.

**Why included**:
- Handles high-dimensional sparse features well
- No feature scaling required
- Provides feature importances (which k-mers drive predictions)
- Robust to irrelevant features

**Known issue observed**: F1 ≈ 0 at default threshold despite AUROC ≈ 0.77. This is a known behaviour of random forests on imbalanced data with many features: the model learns correct rankings (good AUROC) but places probability mass too low on positives for the default 0.5 threshold. Threshold calibration using Platt scaling or isotonic regression would resolve this; not yet implemented.

**Observed performance (val set)**:
- HTR-SELEX PRJEB25907: AUROC = 0.766, AUPRC = 0.659
- RBNS: AUROC = 0.758, AUPRC = 0.684

**Reference**: [Breiman (2001), Machine Learning](https://doi.org/10.1023/A:1010933404324) — original Random Forest paper

---

### 4.3 XGBoost (Gradient Boosted Trees)

**Library**: `xgboost.XGBClassifier`  
**Input**: 8,256-dimensional k-mer frequency vector  
**Hyperparameters**: n_estimators = 300, max_depth = 6, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8, early_stopping_rounds = 20, eval_metric = logloss

**What it does**: Sequentially builds decision trees where each tree corrects the residual errors of the previous ensemble. The shrinkage (learning_rate), subsampling (subsample, colsample_bytree), and early stopping prevent overfitting.

**Why included**:
- Best-performing model on tabular data in most benchmarks
- Early stopping on validation logloss prevents overfitting to training proteins (critical for protein-aware split)
- Handles sparse features efficiently

**Important note on early stopping**: In XGBoost ≥ 2.0, `early_stopping_rounds` must be passed to the constructor, not to `fit()`. This was corrected in the codebase after observing diverging validation logloss (0.623 → 0.830 over 300 rounds on RBNS).

**Observed performance (best model)**:
- HTR-SELEX PRJEB25907: validation AUROC 0.825, AUPRC 0.742, test AUROC 0.796, AUPRC 0.693 (`results/phase1_summary.json`).
- RBNS: validation AUROC 0.661 without early stopping (overfitting), see section 7 for the corrected runs. An earlier note listed HTR-SELEX AUROC 0.844 and AUPRC 0.765. These values do not appear in the result files, which give the values above.

**Reference**: [Chen & Guestrin (2016), KDD](https://doi.org/10.1145/2939672.2939785) — original XGBoost paper

---

## 5. Literature Benchmarks

The following published models are tracked as reference points. All numbers are from the respective papers on their reported test sets.

| Model | Method | Protein split | AUROC | AUPRC | Year | Reference |
|---|---|---|---|---|---|---|
| **ZHMolGraph** | RNA-FM + ProtTrans (frozen) + GNN on interaction network | Unknown protein + unknown RNA (hard, NPInter5) | **0.798** | **0.820** | 2025 | [Liu et al., Commun. Biol.](https://doi.org/10.1038/s42003-025-07694-9) |
| RPITER | Iterative feature refinement + random forest | Random | 0.727 | 0.774 | 2019 | [Peng et al., Int. J. Mol. Sci.](https://doi.org/10.3390/ijms20225543) |
| IPMiner | Deep learning (auto-encoder + deep belief network) | Random | 0.664 | 0.742 | 2016 | [Pan et al., BMC Bioinformatics](https://doi.org/10.1186/s12859-016-1334-2) |
| NPI-GNN | Graph neural network | Random | 0.511 | 0.520 | 2021 | [Yu et al., Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbab255) |
| **Our XGBoost (k-mer)** | XGBoost on k-mer frequency vectors | Protein-disjoint | 0.796 (test) | 0.693 (test) | 2026 | This project, HTR-SELEX PRJEB25907 only (`results/phase1_summary.json`) |

**Note on comparability**: Direct numerical comparison across models is not valid. ZHMolGraph was trained on NPInter2 + RPI7317 (curated literature-derived RPI databases) and evaluated on NPInter5 / TheNovel (entirely unseen protein–RNA pairs from database curation). Our project trains and evaluates on in vitro assay data (HTR-SELEX, RBNS). These are fundamentally different tasks: database RPI prediction vs. in vitro binding specificity prediction. ZHMolGraph AUROC 0.798 is cited as an aspirational performance target, not a directly comparable number.

---

## 6. Methods planned at the start of Phase 2

This section was written before Phase 2. Sections 6.1-6.3 were implemented (V1, V2, V3, V3b, V3c, results in section 7), and V4 (bilinear interaction, `scripts/21`) was added in Phase 3B. Sections 6.4-6.7 describe designs that were not implemented as stated, apart from frozen RNA-FM embeddings in the v3b heads (`scripts/66-67`). The "expected AUROC" figures in this section are expectations written before the experiments.

### 6.1 MLP on k-mer Features

**Description**: Multi-layer perceptron (2–3 hidden layers) trained on the same 8,256-dimensional k-mer vectors. Captures non-linear interactions that logistic regression misses, without requiring sequence-level representations.

**Expected AUROC gain over XGBoost**: ~0.02–0.05  
**Implementation**: PyTorch; ~1 hour training on CPU

---

### 6.2 CNN on One-Hot Encoded Sequences

**Description**: Separate 1D convolutional encoders for RNA (input: L × 4) and protein (input: L × 20), followed by global max pooling and a fully connected classification head. CNNs learn position-sensitive motif filters, unlike k-mer bags.

**Why this matters**: k-mer features lose sequence order entirely. A convolutional filter of width 6 learns which hexamer motifs are present AND in what context (flanking nucleotides). This is how MEME-based motif models work, but in a learnable end-to-end framework.

**Expected AUROC**: 0.85–0.91  
**Implementation**: PyTorch; ~30 min per dataset on GPU (or ~2h on CPU)  
**Reference**: [Alipanahi et al. (2015), Nature Biotechnology](https://doi.org/10.1038/nbt.3300) — DeepBind; CNN for protein–DNA/RNA binding

---

### 6.3 Pretrained Protein Language Model (ESM-2)

**Description**: Extract per-residue embeddings from ESM-2 (650M parameter model, Meta AI) for each protein sequence. Use as protein features in place of k-mer vectors.

**Implemented variants** (see §7 for actual results):

- **V3 — frozen mean-pool**: average over all residues → 1280-d vector. Test AUROC 0.634 against 0.690 for V2.
- **V3b — frozen mean-pool as auxiliary**: concatenate ESM-2 mean-pool (128-d projection) with V2 CNN branches. Test AUROC 0.666 against 0.690 for V2.
- **V3c — frozen residue Conv1D**: Linear(1280→64) per position, then Conv1D + global max pool. Test AUROC 0.685 against 0.690 for V2, above V3b (0.666) and V3 (0.634).

**Why mean-pool fails**: The RNA-binding domain is ~20–50 residues in a protein of 300 aa. Averaging over all residues dilutes the domain signal by 6–15×. Conv1D + max pool (V3c) partially recovers positional selectivity but is insufficient when trained on only 169 proteins.

**Actual test AUROC (frozen, this dataset)**:
- V3 mean-pool: 0.634 (expected 0.85–0.90 based on literature — expectation was wrong for this task)
- V3c residue Conv1D: 0.685

**Conclusion**: Frozen ESM-2 embeddings do NOT improve over one-hot CNN on this dataset. ESM-2 helps motif-poor proteins (UNK +0.25 AUROC) but hurts motif-rich proteins (PUM2 −0.25 AUROC). Mean-pooling is the mechanism of failure, not the model itself.

**Reference**: [Lin et al. (2023), Science](https://doi.org/10.1126/science.ade2574) — ESM-2; [Rives et al. (2021), PNAS](https://doi.org/10.1073/pnas.2016239118) — ESM-1

---

### 6.4 Pretrained RNA Language Model (RNA-FM)

**Description**: Extract per-nucleotide embeddings from RNA-FM (100M parameter model) for each RNA sequence. Average-pool to obtain a 640-dimensional RNA embedding.

**Why this matters**: RNA-FM was trained on 23 million non-coding RNA sequences and captures secondary structure information implicitly — sequences that fold into similar structures get similar embeddings, even with different primary sequences. This is the key limitation of one-hot and k-mer RNA encoding that RNA-FM resolves.

**Modes**: Frozen (planned V3b) → fine-tuned (planned V3c)  
**Reference**: [Chen et al. (2022), arXiv](https://arxiv.org/abs/2204.00300) — RNA-FM

---

### 6.5 Cross-Attention Interaction Module

**Description**: Instead of concatenating protein and RNA embeddings and passing them to an MLP, use a cross-attention layer that allows the protein representation to attend over RNA positions and vice versa. The attention weights are interpretable: they show which RNA regions interact with which protein residues.

**Architecture**:
```
RNA sequence     → RNA Encoder   → RNA token embeddings   (L_rna × D)
Protein sequence → Prot Encoder  → Prot token embeddings  (L_prot × D)
                 → Cross-Attention (RNA attends to Protein, Protein attends to RNA)
                 → Pooling + MLP Head → binding score
```

**Why this matters**: Protein–RNA binding is fundamentally about complementarity between specific protein domains and specific RNA motifs. Cross-attention directly models this: each RNA token learns which protein regions are relevant to it. Simple concatenation loses this pairwise information.

**Expected AUROC**: 0.91–0.96 (with fine-tuned encoders)  
**Reference**: [Vaswani et al. (2017), NeurIPS](https://arxiv.org/abs/1706.03762) — Attention mechanism; [Jumper et al. (2021), Nature](https://doi.org/10.1038/s41586-021-03819-2) — AlphaFold2 uses cross-attention between protein sequence and multiple sequence alignment

---

### 6.6 Multi-Task Learning (Binding + Affinity)

**Description**: Add a regression head alongside the binary classification head. When affinity data is available (e.g., R_max from RBNS, enrichment ratio from HTR-SELEX), train both heads simultaneously using a masked multi-task loss:

```
L_total = L_classification + λ · L_regression (only where affinity label is available)
```

**Why this matters**: Affinity values are more informative than binary labels — a sequence with R_max = 50 is a stronger binder than R_max = 1.6, but both are labelled "1". Training the model to predict R_max as well should improve the ranking quality (AUPRC) even if only the binary label is available at test time.

**Affinity data available**:
- RBNS: R_max per sequence (enrichment ratio across concentrations)
- HTR-SELEX: frequency in last selection cycle (proxy for affinity)

**Reference**: [Caruana (1997), Machine Learning](https://doi.org/10.1023/A:1007379606734) — multi-task learning; [Tasaki et al. (2022), Nature Machine Intelligence](https://doi.org/10.1038/s42256-022-00457-9) — multi-task protein function prediction

---

### 6.7 Experimental Context Label (in vivo / in vitro)

**Description**: Add a binary feature `experiment_type` (0 = in vitro, 1 = in vivo) as an additional input to the MLP classification head. This allows the model to learn that in vivo binding is more selective than in vitro SELEX.

**Datasets by context**:
- in vitro: HTR-SELEX PRJEB25907, HTR-SELEX PRJEB47428, RBNS
- in vivo: eCLIP (ENCODE), iCLIP, PAR-CLIP (planned for Phase 2)

---

## 7. Results Summary Table

Updated after each experiment. Phase 1 numbers are on the **test set** (unseen proteins within each dataset). Phase 2 numbers are on the generalized test set (unseen proteins across all 3 datasets, 24 proteins total).

### Phase 1 — Per-dataset validation (protein-aware split)

| Dataset | Model | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Status | Date |
|---|---|---|---|---|---|---|---|
| HTR-SELEX PRJEB25907 | XGBoost | **0.825** | **0.742** | 0.796 | 0.693 | ✅ PASS | 2026-05 |
| HTR-SELEX PRJEB25907 | Logistic Regression | 0.771 | 0.639 | — | — | ✅ PASS | 2026-05 |
| HTR-SELEX PRJEB25907 | Random Forest | 0.759 | 0.644 | — | — | ✅ PASS | 2026-05 |
| RBNS | Random Forest | **0.758** | **0.684** | 0.632 | 0.507 | ✅ PASS | 2026-04 |
| RBNS | Logistic Regression | 0.746 | 0.636 | — | — | ✅ PASS | 2026-04 |
| RBNS | XGBoost | 0.661 | 0.522 | — | — | ✅ PASS | 2026-04 |
| HTR-SELEX PRJEB47428 | Logistic Regression | **0.817** | **0.736** | 0.590 | — | ✅ PASS | 2026-04 |
| HTR-SELEX PRJEB47428 | XGBoost | 0.629 | 0.454 | 0.628 | 0.436 | ✅ PASS | 2026-04 |

The Status column refers to the validation threshold (val AUROC > 0.70 for the best model). On the test proteins `results/phase1_summary.json` marks RBNS (0.632, 13 proteins) and PRJEB47428 (0.628, 4 proteins) as WARN, and PRJEB25907 (0.796, 13 proteins) as PASS.

**Flagged proteins** (test AUROC < 0.70, hard biological cases, not data quality issues):
- RBNS: RBM4 (0.345), RBM4B (0.323) — closely related paralogs with near-identical binding profiles; XRCC6 (0.496) — atypical non-canonical RBP
- HTR-SELEX PRJEB25907: PCBP1 (0.436) — poly-C binding with structured recognition mode; LARP6 (0.693), RBM6 (0.683), RBMS2 (0.656) — marginal
- HTR-SELEX PRJEB47428: PUM2 (0.504), slbp (0.605) — only 4 test proteins, high variance

### Phase 2 — Generalized model (cross-dataset, 168 proteins, 632k examples)

The pool of the three datasets has 632,642 examples and 168 proteins. `generalized_v2` adds 26,362 eCLIP rows used for training only (659,004 rows, 169 proteins in `EXPERIMENT_LOG.md`). V2, V3, V3b and V3c were rerun on 2026-05-13 after the double class-weighting fix (section 8.1). V1 (`results/generalized/v1_mlp_results.json`, dated 2026-04-27) was produced before the sampler was removed from `scripts/05`, and its rerun is pending.

| Model | Architecture | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Best epoch | Date |
|---|---|---|---|---|---|---|---|
| V1 MLP | k-mer 8262-d → MLP [512,256,128] | 0.716 | 0.636 | 0.674 | 0.544 | 1 (then degraded) | 2026-04 |
| V2 CNN | One-hot → dual CNN → MLP | **0.746** | — | **0.690** | **0.580** | 13 / 13 | 2026-05-13 |
| V3 ESM-2 (frozen mean-pool) | ESM-2(1280) mean-pool → Linear(256) + RNA CNN | 0.715 | 0.675 | 0.634 | 0.547 | 17 / 25 | 2026-05 |
| V3b ESM-2 + CNN concat | RNA CNN(256) + Prot CNN(256) + ESM-2(1280→128) concat | 0.770 | 0.704 | 0.666 | 0.568 | — | 2026-05 |
| V3c ESM-2 residue CNN | RNA CNN(256) + ESM-2 residue Conv1D → max pool(256) | 0.745 | 0.683 | 0.685 | 0.595 | 18 | 2026-05-13 |

**V2 CNN per-protein test results** (24 proteins, median AUROC 0.714, `results/phase2_summary.json`; the table below is from an earlier V2 run and differs by up to 0.01 per protein, for example ESRP1-construct3 0.980 and PUF60 0.934 in the 2026-05-13 run):

| Protein | Dataset | AUROC | AUPRC |
|---|---|---|---|
| ESRP1-construct3 | HTR-SELEX 25907 | 0.981 | — |
| PUF60 | RBNS | 0.924 | — |
| KHDRBS3 | HTR-SELEX 25907 | 0.901 | — |
| RBM28 | HTR-SELEX 25907 | 0.842 | — |
| PUM2 | HTR-SELEX 25907 | 0.835 | — |
| TRA2A | RBNS | 0.800 | — |
| HNRNPA0 | HTR-SELEX 25907 | 0.851 | — |
| CSDA | HTR-SELEX 25907 | 0.819 | — |
| LARP7-construct4 | HTR-SELEX 25907 | 0.774 | — |
| MEX3D-construct3 | HTR-SELEX 25907 | 0.735 | — |
| DAZAP1 | HTR-SELEX 25907 | 0.783 | — |
| PRR3 | RBNS | 0.759 | — |
| IGF2BP2 | RBNS | 0.701 | — |
| PCBP1 | HTR-SELEX 25907 | 0.675 | — |
| ZC3H18 | RBNS | 0.612 | — |
| NUPL2 | RBNS | 0.610 | — |
| RBM6 | HTR-SELEX 25907 | 0.610 | — |
| LARP6 | HTR-SELEX 25907 | 0.585 | — |
| IGF2BP1 | HTR-SELEX 25907 | 0.632 | — |
| SNRNP70 | HTR-SELEX 25907 | 0.574 | — |
| SRSF8 | RBNS | 0.577 | — |
| TAF15 | RBNS | 0.468 | — |
| IGF2BP3 | RBNS | 0.459 | — |
| UNK | RBNS | 0.429 | — |

**V1 MLP failure**: k-mer features are length-dependent — RBNS 20nt vs HTR-SELEX 40nt produce incompatible frequency distributions. Val AUROC peaked at epoch 1 then degraded. StandardScaler does not resolve a structural feature mismatch. V2 CNN resolves this via global max pooling (length-agnostic).

**V3 ESM-2 frozen mean-pool failure**: Mean-pooling over all residues dilutes the binding domain signal. Proteins with strong position-specific motifs (PUM2, LARP7, TRA2A) lost 0.15–0.41 AUROC vs V2. Proteins without clear motifs (UNK, TAF15, IGF2BP3) gained 0.06–0.25 AUROC — ESM-2 provides useful family context for these. Key lesson: V2 CNN and ESM-2 encode *complementary* information. V3b was designed to concatenate both.

**V3b ESM-2 + CNN concat failure**: Adding ESM-2 mean-pool as an auxiliary signal alongside the proven V2 CNN branches still degraded performance (test AUROC 0.666 vs 0.690). This confirms that the failure mode is fundamental to mean-pooling, not to the architecture or weight of the ESM-2 branch. Two independent experiments (V3 and V3b) both degraded relative to pure one-hot CNN — the mean-pool ESM-2 representation is actively harmful for motif-rich proteins even when the CNN branch is present. V3c replaces mean-pool with a residue-level Conv1D branch, letting the model focus on binding-relevant positions via learned filters.

### External Validation — Preliminary Assessment

V2 CNN was evaluated on the literature dataset (`dataset_without_affinities.xlsx`, 159 pairs).
Updated numbers (v3a checkpoint, 2026-07): AUROC=**0.763** AUPRC=0.915 (curated); expanded
540-pair benchmark via `scripts/31` — see `DATA.md §2.3`.

Reported AUROC=0.798 in older notes used a slightly different row filter; treat **0.763** as canonical curated result.

1. **Single-class proteins** (88%): 84 of 96 proteins have only positive *or* only negative examples. Per-protein AUROC is undefined for these; only 9 proteins contribute to the metric.

2. **Sliding window max-pooling bias**: RNAs >60 nt are split into overlapping 60 nt windows; the pair score is the maximum window score. Long RNAs (56% of pairs) receive mean probability 0.809 vs 0.186 for short RNAs, independent of actual binding. This inflates AUROC/AUPRC for datasets where positives tend to be long RNAs.

3. **High positive class rate** (72%): Random classifier AUPRC baseline is already 0.717. The effective improvement over random is +0.21, not the apparent 0.21 above 0.5.

A reliable external benchmark requires ≥500 pairs with both positive and negative examples per protein (≥5 per class per protein). The current dataset is most useful for qualitative validation and dataset expansion efforts.

### External benchmarks (reference only)

| Model | Method | AUROC | AUPRC | Split | Reference |
|---|---|---|---|---|---|
| ZHMolGraph | RNA-FM + ProtTrans (frozen) + GNN | 0.798 | 0.820 | Hard (NPInter5, unseen prot+RNA) ⚠️ different dataset | Liu et al. 2025 |
| RPITER | Iterative feature refinement + RF | 0.727 | 0.774 | Random | Peng et al. 2019 |
| IPMiner | Deep autoencoder + DBN | 0.664 | 0.742 | Random | Pan et al. 2016 |
| NPI-GNN | Graph neural network | 0.511 | 0.520 | Random | Yu et al. 2021 |

**Comparability note**: ZHMolGraph trains on NPInter2/RPI7317 (curated database interactions)
and tests on NPInter5 (TheNovel — completely unseen protein–RNA pairs). Our project trains
on HTR-SELEX/RBNS (in vitro selection assays) and tests on protein-aware splits from the
same assay type. These are different biological tasks, different data sources, and different
negative sampling strategies. ZHMolGraph 0.798 is an aspirational reference only.

### RNA-only per-protein classifiers (script 25, 2026-06)

Independent per-protein baseline: **one classifier per RBP**, RNA 4-mer
frequencies only (256-d), no protein sequence. Models: Logistic Regression (L2, scaled)
and Random Forest (200 trees). XGBoost optional if installed.

**Split protocol (`--honest`, default)**:
1. Deduplicate rows with identical `rna_sequence` (keep first label).
2. Stratified 60/20/20 train / val / test per protein.
3. Pick best model on **validation** AUROC (tie-break: AUPRC).
4. Report **test** AUROC and AUPRC only (held-out).
5. Refit chosen model on train+val for optional export (`--save_models`).

RNAcompete: when multiple `hyb_id` per protein, keep best experiment (highest mean
positive intensity); modal RNA length computed from **train positives only** and applied
to filter val/test (same logic as script 24).

| Dataset | n RBPs | Median test AUROC | Median test AUPRC | RF wins |
|---|---|---|---|---|
| HTR-SELEX | 93 | 0.949 | 0.927 | 73/93 |
| RBNS | 96 | 0.995 | 0.985 | 79/96 |
| RNAcompete Eukarya | 200 | 0.993 | 0.985 | 137/200 |
| RNAcompete RBPZoo | 174 | 0.990 | 0.974 | 153/174 |

Figures: `python scripts/26_visualize_rna_only_results.py` → `figures/rna_only_*.png`.

**Caveats**: per-protein k-mer models are expected to score high on in vitro binding data.
RNAcompete positive labels partially overlap with 7-mer enrichment criteria used in script 24
(more circular). This is **not** comparable to generalized V2 CNN (protein-aware cross-RBP split).

### Top/bottom example extraction (script 24, 2026-06)

For structural follow-up: 5 positives + 5 negatives per RBP, anchored on top-10 enriched
7-mers. RNAcompete: modal length among kmer-positive probes; negatives at same length without
top-10 7-mer. Output master TSV: `results/top_bottom_examples/all_protocols_summary.tsv`.

Motif score columns (protocol-specific):
- **RNAcompete** (`kmer_z_score`): official Hughes / RBPZoo 7-mer Z-score of the matched
  top-kmer (max Z among top-10 hits in the probe).
- **RBNS / HTR-SELEX** (`kmer_enrichment_score`): pos/neg pool k-mer enrichment ratio.
  The master TSV carries both columns; each is populated only for the relevant protocol.

### RNAcompete intensity spectrum sampling (scripts 27-28, archived in `scripts/archive/`, 2026-06)

Probe set for structural / intensity follow-up: **100 modal-length probes per RBP**,
evenly spaced across log₁₀(`probe_intensity`) percentiles, for the **top 3 RBPs by mean
positive intensity** per RNAcompete panel.

**Filters**:
- One `hyb_id` per protein (highest mean positive intensity).
- Modal `rna_sequence` length only (38 nt in both Eukarya and RBPZoo).
- Percentile targets: 0.5%, 1.5%, …, 99.5%; nearest probe by log-intensity, no replacement.

**Length vs intensity** (full panel, not just top 3):

| Panel | n RBPs | Pearson r | p |
|---|---|---|---|
| Eukarya | 200 | −0.050 | 0.49 |
| RBPZoo | 174 | −0.036 | 0.63 |

No significant association. Deliverables: `spectrum_samples_{dataset}.tsv` (300 rows each).
Figure: `docs/archive/figures/rnacompete_domain_diverse_length_vs_intensity.png`.

**Note**: log transform is `log10(intensity − Smin + 1)` per protein (Smin = minimum raw
intensity in the modal-length pool). Probes at Smin map to log 0.

### Top/bottom example experiment IDs (script 29, archived in `scripts/archive/`, 2026-06)

Adds `experiment_id` to `all_protocols_summary.tsv` and exports
`all_protocols_proteins.fasta` (headers `>experiment_id,protein_name`):

| Protocol | ID source |
|---|---|
| RNAcompete | `hyb_id` (best experiment per protein) |
| RBNS | ENCODE `experiment_accession` |
| HTR-SELEX | `experiment_accession` (mode per protein in fastq metadata) |

---

## 8. Methodological limitations

Scope of the reported numbers, stated per item.

### 8.1 Double class-weighting (fixed 2026-05-13)

`WeightedRandomSampler` (about 50/50 batches) was used together with `BCEWithLogitsLoss(pos_weight = n_neg / n_pos, about 2.0)` in scripts 06, 08, 09 and 10, which corresponds to an effective positive weight of about 4. The sampler was removed on 2026-05-13 and `pos_weight` is the only class-imbalance correction. V2, V3, V3b and V3c were retrained and all numbers in this document and in `EXPERIMENT_LOG.md` come from the retrained checkpoints. Script 05 (V1) also had the sampler removed. The V1 result file is dated 2026-04-27 and its rerun is pending.

### 8.2 Homology between splits

Splits in v2 and v3a are by protein name or construct name. The audit of section 3.4 measures the resulting overlap: 62% of the v3a test proteins have a training protein at >= 50% identity, and v3a probe AUROC is 0.52-0.53 (V4) without a detectable homolog and 0.87-0.88 at >= 70% identity. `generalized_v3b` controls homology (3% of test proteins at >= 50%). The v2 split (24 test proteins, for example RBM4 and RBM4B on different sides) was not audited with MMseqs2.

### 8.3 Negatives

SELEX and RBNS negatives are non-enriched or shuffled sequences of the same library. The in vitro cross-protein probe (`scripts/53`) uses the same RNA paired with another protein of the same assay, so RNA composition carries no information (GC and k-mer LR score 0.500). eCLIP negatives of `generalized_v2` (flanking regions, GC and length matched, 26,362 rows) are training-only. The Skipper benchmarks use genomic windows (`protein_disjoint`, `protein_and_rna`) or other proteins' positives (`cross_protein_neg`).

### 8.4 Validation-test gap

Validation AUROC exceeds test AUROC by 0.05-0.06 for V2 on `generalized_v2` (0.746 and 0.690) and by 0.04-0.10 in the other Phase 2 runs (V1 0.042, V3 0.081, V3b 0.104, V3c 0.060). Possible contributors that were not separated: RNA length distributions of RBNS and HTR-SELEX, family composition of the validation and test proteins, and early stopping on validation AUPRC.

### 8.5 Seeds

Phase 1, Phase 2 and V2 on v3a use seed 42. Three seeds (42, 0, 1) are used for V4 on v3a (test AUROC 0.829 +- 0.009, sample SD), the in vitro probe (V2 0.686 +- 0.015, V4 0.754 +- 0.001), the frozen-embedding heads and the TF ChIP-seq comparison. V2 and V4 on v3b rest on seed 42, and the seed 0 and 1 runs on v3b were not completed. Standard deviations are sample SDs over 3 seeds.

### 8.6 Result files of the V4 runs

The V4 numbers (3-seed test AUROC 0.829 +- 0.009, AUPRC 0.732 +- 0.011, external curated 0.737 and expanded 0.666) are recorded in `results/phase3b_summary.json`. The per-run result files (`results/multiseed/v4_concat_bi_v3a/`, `results/generalized/v4_phase3a/` and the V4 external JSON files) are in the repository (small JSON, TSV and the scored external pairs). Model weights are on the GPU VM only. The field `vs_v2` in the V4 result JSON files is the difference to the Phase 2 V2 test AUROC and AUPRC of 0.690 and 0.580 on `generalized_v2`. The comparison on the same test set (`generalized_v3a`) is 0.829 against 0.813 AUROC, +0.016.

---

## 9. Additional result tables

Tables moved from the README. Sources are the log entries and result files named in each item.

### 9.1 Skipper eCLIP benchmark (scripts 41-54)

Skipper eCLIP windows `fixlen_151` (173 gene symbols, 264 experiments, Boyle et al. 2023) form an in vivo long-RNA benchmark. Splits are in `data/benchmarks/skipper_eclip/`.

| Split | Negatives | Protein-disjoint | Purpose |
|---|---|---|---|
| `protein_disjoint` | genomic non-peak windows | yes | peak against background |
| `protein_and_rna` | genomic non-peak windows | yes, and no test RNA in train | variant without RNA overlap |
| `cross_protein_neg` | other proteins' positives, GC matched | yes | protein-specific signal |

RNA-only baselines on the test sets (no protein input):

| Split | GC fraction | RNA k-mer LR (k <= 4) | k-mer LR, GC-matched subset |
|---|---|---|---|
| `protein_disjoint` | 0.744 | 0.808 | 0.680 |
| `protein_and_rna` | 0.746 | 0.808 | 0.680 |
| `cross_protein_neg` | 0.544 | 0.435 | 0.441 |

On `cross_protein_neg` the tested models (V2 and V4 trained from scratch, V2 fine-tuned from v3a, v3a zero-shot) reach GC-matched AUROC 0.50-0.53, while per-protein k-mer LR on proteins seen in training reaches a median 0.737. On the in vitro probe the same models follow sequence identity to training proteins (MMseqs2 audit, V2 and V4: 0.85-0.87 at >= 90% identity, 0.61-0.70 at 30-50%, 0.48-0.53 with no detectable homolog). Source: `EXPERIMENT_LOG.md`, entries of 2026-10-28 and 2026-10-29. eCLIP TSVs built before the "validate reason" fix carry `protein_sequence == "ok"`. Scripts 41, 41b, 41c and 42 abort on such data and script 51 refuses to score it. Rebuild: `scripts/run_eclip_step1.sh` (`STAGES="baselines"` selects a subset).

### 9.2 RNA-only per-protein classifiers (scripts 25-26)

One RNA 4-mer model (LR or RF) per protein, 60/20/20 split per protein after removing duplicate RNAs. Median test AUROC per protein (`results/rna_only_per_protein_honest/all_datasets_summary.tsv`): HTR-SELEX 0.949 (93 proteins), RBNS 0.995 (96), RNAcompete Eukarya 0.993 (200), RNAcompete RBPZoo 0.990 (174). The task is within-protein, so the numbers are not comparable with protein-disjoint results.

### 9.3 RNAcompete intensity spectrum (scripts 27-28, archived)

100 probes per protein across log-intensity percentiles (`scripts/archive/27_sample_rnacompete_intensity_spectrum.py`, `scripts/archive/28_visualize_intensity_spectrum.py`). Pearson correlation of protein length with mean positive intensity: Eukarya r = -0.050 (p = 0.49), RBPZoo r = -0.036 (p = 0.63).

### 9.4 RNAcompete zero-shot, Phase 2 V2 checkpoint (historical)

V2 trained on `generalized_v2` (169 proteins), evaluated with `scripts/archive/20_evaluate_benchmark.py` on `rnacompete_all` (2M pairs sampled from 13.9M, 742 proteins): overall AUROC 0.571, per-protein median 0.551 (0.549 for 715 proteins not in training, 0.629 for 27 seen), 17.3% of proteins above 0.7. Valid for this checkpoint only, since the Eukarya and RBPZoo panels are part of `generalized_v3a` training.

### 9.5 Cross-protocol in vitro runs

`results/cross_protocol_invitro/` (config `configs/cross_protocol_invitro.yaml`, no eCLIP in the roster) holds the quoted numbers: 196 within-protocol and 284 transfer evaluations, mean AUROC 0.974 and 0.791. `results/cross_protocol/` (config `configs/cross_protocol.yaml`) is the earlier run with eCLIP in the roster (9 eCLIP proteins): 204 and 322 evaluations, mean AUROC 0.963 and 0.779.

---

## 10. Domain annotation

Pfam family labels (`scripts/56`, also `scripts/80` and `scripts/68`) come from `hmmscan` against Pfam-A with `-E 1e-3 --domE 1e-3`. The filter therefore applies to the sequence E-value and the conditional (per-domain) E-value of `--domE`. The stored column `domain_ievalue` is the independent E-value (i-Evalue) of the domain table, which is a different quantity from the conditional E-value used for the filter. Sensitivity on the 175 RNAcompete RBPZoo constructs of `results/esm_layer_profile/domains.tsv`: 471 of the 924 domain rows have `domain_ievalue` above 1e-3. With the logged default (conditional E-value only) 175 proteins have a domain and the families are RRM 145, KH 24, CCCH 6, CCHC 3, DEAD 1, with a median domain coverage of 0.641 of the construct (quartiles 0.489-0.724). With an additional filter `domain_ievalue < 1e-3` (453 rows) 174 proteins have a domain, the families are RRM 145, KH 24, CCCH 5, CCHC 2, DEAD 1, and the median coverage is 0.545 (quartiles 0.370-0.644). Scripts 61 and 80 have an optional argument for this filter (default off), so the logged results use the conditional E-value only. The sensitivity of the 448-protein family table of script 56 to the same filter was not computed.
