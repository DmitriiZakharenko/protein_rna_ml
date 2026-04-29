# Methods & Model Tracking

**Project**: Protein–RNA Binding Prediction  
**Last updated**: 2026-04-23  
**Language**: English  

This document records every methodological choice made in the project — encoding strategies, splitting decisions, model architectures, and training configurations — together with the rationale for each choice and literature references. It is updated each time a new method is introduced or an existing one is modified.

---

## Table of Contents

1. [Evaluation Protocol](#1-evaluation-protocol)
2. [Sequence Encoding](#2-sequence-encoding)
3. [Dataset Splitting Strategy](#3-dataset-splitting-strategy)
4. [Baseline Models](#4-baseline-models)
5. [Literature Benchmarks](#5-literature-benchmarks)
6. [Planned Methods — Phase 2](#6-planned-methods--phase-2)
7. [Results Summary Table](#7-results-summary-table)

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

This design is analogous to the "unknown protein" or "hard split" setting used in the ZHMolGraph benchmark (Liu et al., 2025) and is stricter than the standard random split used in older RBP prediction papers.

**Random seed**: 42 (fixed throughout; all splits reproducible).

**Reference**:
- [Ghanbari & Ohler (2020), Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbz103) — discusses protein-aware vs random splits for RBP prediction; demonstrates that random splits systematically overestimate performance
- [Liu et al. (2025), Communications Biology](https://doi.org/10.1038/s42003-025-07657-4) (ZHMolGraph) — uses "unknown RNA + unknown protein" as the primary hard-split benchmark

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
- HTR-SELEX PRJEB25907: AUROC = **0.844**, AUPRC = **0.765** → **PASS ✅**
- RBNS: AUROC = 0.661 (without early stopping; **overfitting confirmed**) → being rerun

**Reference**: [Chen & Guestrin (2016), KDD](https://doi.org/10.1145/2939672.2939785) — original XGBoost paper

---

## 5. Literature Benchmarks

The following published models are tracked as reference points. All numbers are from the respective papers on their reported test sets.

| Model | Method | Protein split | AUROC | AUPRC | Year | Reference |
|---|---|---|---|---|---|---|
| **ZHMolGraph** | RNA-FM + ProtTrans (frozen) + GNN on interaction network | Unknown protein + unknown RNA (hard) | **0.798** | **0.820** | 2025 | [Liu et al., Commun. Biol.](https://doi.org/10.1038/s42003-025-07657-4) |
| RPITER | Iterative feature refinement + random forest | Random | 0.727 | 0.774 | 2019 | [Peng et al., Int. J. Mol. Sci.](https://doi.org/10.3390/ijms20225543) |
| IPMiner | Deep learning (auto-encoder + deep belief network) | Random | 0.664 | 0.742 | 2016 | [Pan et al., BMC Bioinformatics](https://doi.org/10.1186/s12859-016-1334-2) |
| NPI-GNN | Graph neural network | Random | 0.511 | 0.520 | 2021 | [Yu et al., Briefings in Bioinformatics](https://doi.org/10.1093/bib/bbab255) |
| **Our XGBoost (k-mer)** | XGBoost on k-mer frequency vectors | Protein-aware | 0.844 | 0.765 | 2025 | This project — HTR-SELEX only |

**Note on comparability**: Direct numerical comparison across models is difficult because each uses a different dataset, different negatives, and different train/test splits. The ZHMolGraph "hard split" is the closest to our protein-aware evaluation setting.

---

## 6. Planned Methods — Phase 2

The following methods will be implemented after all datasets pass Phase 1 validation. Listed in order of increasing complexity and expected performance.

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

**Description**: Extract per-residue embeddings from ESM-2 (650M parameter model, Meta AI) for each protein sequence. Average-pool over residues to obtain a fixed-size protein embedding (1,280 dimensions). Use as protein features in place of k-mer vectors.

**Why this matters**: ESM-2 was trained on 250 million protein sequences and encodes evolutionary, structural, and functional information that k-mers cannot capture. Proteins with similar binding domains (e.g., all RRM-domain proteins) will have similar embeddings even if their primary sequences diverge.

**Modes**:
- **Frozen** (zero-shot): extract embeddings without fine-tuning; fast; similar to ZHMolGraph approach
- **Fine-tuned**: continue training ESM-2 on our RBP dataset; expected to substantially outperform frozen

**Expected AUROC (frozen)**: 0.85–0.90  
**Expected AUROC (fine-tuned)**: 0.90–0.95  
**Reference**: [Lin et al. (2023), Science](https://doi.org/10.1126/science.ade2574) — ESM-2; [Rives et al. (2021), PNAS](https://doi.org/10.1073/pnas.2016239118) — ESM-1

---

### 6.4 Pretrained RNA Language Model (RNA-FM)

**Description**: Extract per-nucleotide embeddings from RNA-FM (100M parameter model) for each RNA sequence. Average-pool to obtain a 640-dimensional RNA embedding.

**Why this matters**: RNA-FM was trained on 23 million non-coding RNA sequences and captures secondary structure information implicitly (sequences that fold into similar structures get similar embeddings). ZHMolGraph also uses RNA-FM but with frozen weights.

**Modes**: Frozen (baseline) → fine-tuned (target)  
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

Updated after each experiment. All numbers are on the **test set** (unseen proteins).

| Dataset | Model | Encoding | AUROC | AUPRC | Status | Date |
|---|---|---|---|---|---|---|
| HTR-SELEX PRJEB25907 | XGBoost | k-mer (4+3) | **0.844** | **0.765** | ✅ PASS | 2025-04 |
| HTR-SELEX PRJEB25907 | Logistic Regression | k-mer (4+3) | 0.771 | 0.640 | ✅ PASS | 2025-04 |
| HTR-SELEX PRJEB25907 | Random Forest | k-mer (4+3) | 0.766 | 0.659 | ✅ PASS | 2025-04 |
| RBNS | XGBoost (no early stop) | k-mer (4+3) | 0.608 | 0.473 | ⚠️ WARN* | 2025-04 |
| RBNS | Logistic Regression | k-mer (4+3) | — | — | pending eval on test | — |
| RBNS | Random Forest | k-mer (4+3) | — | — | pending eval on test | — |
| HTR-SELEX PRJEB47428 | Logistic Regression | k-mer (4+3) | — | — | pending | — |
| HTR-SELEX PRJEB47428 | XGBoost (with early stop) | k-mer (4+3) | — | — | pending | — |

*RBNS XGBoost failed due to missing early stopping — overfitting confirmed. To be rerun.  
**RBNS flagged proteins**: EIF4G2 (0.49), RBM4 (0.45), NUPL2 (0.43), RBM4B (0.40), SAFB2 (0.37). Likely biological/domain novelty rather than data quality issue. RBNS dataset considered valid for Phase 2 with these proteins flagged as hard cases.

