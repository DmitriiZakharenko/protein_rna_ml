# Protein–RNA Binding Prediction: ML Project Plan

**Goal**: Build a generalizable machine learning model that predicts whether a given protein (amino acid sequence) binds to a given RNA (nucleotide sequence), with planned extensions for experimental context (in vivo / in vitro) and binding affinity.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Datasets & Roadmap](#3-datasets--roadmap)
4. [Phase 0 — Environment Setup](#phase-0--environment-setup)
5. [Phase 1 — Per-Dataset Validation Model (HTR-SELEX)](#phase-1--per-dataset-validation-model-htr-selex)
6. [Phase 2 — Generalized Model](#phase-2--generalized-model)
7. [Preprocessing Pipeline (HTR-SELEX)](#preprocessing-pipeline-htr-selex)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Future Extensions](#future-extensions)

---

## 1. Project Overview

We are building a protein–RNA interaction (PRI) classifier with the following properties:

| Property | Details |
|---|---|
| Input 1 | Amino acid sequence of protein (variable length) |
| Input 2 | Nucleotide sequence of RNA (variable length, uses U not T) |
| Output (now) | Binary: 1 = binds, 0 = does not bind |
| Output (future) | + in vivo / in vitro label; + continuous affinity score |
| Goal | Maximum generalization across proteins AND RNA families |

**Two-stage strategy:**
1. **Validation models** — quick, per-dataset sanity checks. Train a small model on each individual dataset to confirm the data has a learnable signal. If a dataset fails here, it doesn't go into the main training pool.
2. **Generalized model** — trained jointly across all validated datasets, designed to generalize to unseen proteins.

---

## 2. Folder Structure

```
protein_rna_ml/                  ← rename this folder from "ml"
├── PROJECT_PLAN.md              ← this file
├── requirements.txt
├── configs/
│   ├── htr_selex_validation.yaml   ← hyperparameters for HTR-SELEX validation model
│   └── generalized_model.yaml      ← hyperparameters for main model
├── data/
│   ├── raw/
│   │   └── htr_selex/           ← symlink or copy of ml_dataset_simple_clean.tsv
│   ├── processed/
│   │   └── htr_selex/           ← encoded features (numpy/parquet)
│   ├── splits/
│   │   └── htr_selex/           ← train/val/test index files
│   └── future_datasets/         ← placeholder for eCLIP, RBNS, etc.
├── src/
│   ├── data/
│   │   ├── preprocessing.py     ← sequence encoding (k-mer, one-hot)
│   │   ├── dataset.py           ← PyTorch Dataset classes
│   │   └── splits.py            ← protein-aware splitting logic
│   ├── models/
│   │   ├── baseline.py          ← k-mer + sklearn classifiers
│   │   ├── encoders.py          ← CNN/Transformer sequence encoders
│   │   └── classifier.py        ← full neural model (protein encoder + RNA encoder + head)
│   ├── training/
│   │   ├── train.py             ← training loop
│   │   └── evaluate.py          ← evaluation logic + metrics
│   └── utils/
│       ├── metrics.py           ← AUROC, AUPRC, etc.
│       └── visualization.py     ← plots
├── scripts/
│   ├── 01_prepare_htr_selex.py  ← preprocessing + splitting for HTR-SELEX
│   ├── 02_train_validation_model.py  ← train small validation model
│   ├── 03_evaluate_validation.py     ← full evaluation report
│   └── 04_train_generalized.py       ← main generalized model (Phase 2)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_htr_selex_validation.ipynb
│   └── 03_generalized_model.ipynb
├── models/
│   ├── checkpoints/
│   └── saved/
└── results/
    ├── htr_selex/
    │   ├── plots/
    │   └── metrics/
    └── generalized/
```

---

## 3. Datasets & Roadmap

| # | Dataset | Type | Status | Notes |
|---|---|---|---|---|
| 1 | HTR-SELEX (PRJEB25907) | in vitro | ✅ Ready | 279k examples, 93 proteins |
| 2 | eCLIP (ENCODE) | in vivo | 🔜 Planned | ~100+ RBPs, cell line data |
| 3 | RBNS | in vitro | 🔜 Planned | RNA Bind-n-Seq |
| 4 | RNAcompete | in vitro | 🔜 Planned | Microarray-based |
| 5 | CLIP-seq (various) | in vivo | 🔜 Planned | PAR-CLIP, iCLIP, HITS-CLIP |

**Each new dataset must pass Phase 1 validation before being added to Phase 2 training.**

---

## Phase 0 — Environment Setup

```bash
# Create conda environment
conda create -n protein_rna_ml python=3.10
conda activate protein_rna_ml

# Install dependencies
pip install -r requirements.txt
```

**Key packages:**
- `scikit-learn` — baseline models (RF, XGBoost, Logistic Regression)
- `xgboost` — gradient boosting baseline
- `torch` + `torchvision` — neural models
- `numpy`, `pandas`, `scipy` — data handling
- `matplotlib`, `seaborn` — visualization
- `biopython` — sequence utilities
- `pyyaml` — config files
- `tqdm` — progress bars

---

## Phase 1 — Per-Dataset Validation Model (HTR-SELEX)

**Purpose**: Verify that the dataset contains a genuine, learnable signal. A model that cannot beat random on this dataset signals a data quality problem.

### Step 1.1 — Data Loading & EDA

Script: `scripts/01_prepare_htr_selex.py`

- Load `ml_dataset_simple_clean.tsv`
- Report class balance (target: 33% pos / 67% neg — ✅ confirmed)
- Report protein distribution (target: ~1000 pos per protein — ✅ confirmed)
- Report RNA sequence length distribution
- Flag any missing sequences or invalid characters
- Save EDA summary to `results/htr_selex/metrics/eda_summary.json`

### Step 1.2 — Protein-Aware Splitting (CRITICAL)

**Why protein-aware?** If the same protein appears in train and test, the model can learn protein-specific patterns without generalizing. We must test on proteins the model has NEVER seen.

Split strategy for HTR-SELEX (93 proteins):
- **Train**: 70 proteins (~75%) → ~200,000 examples
- **Val**: 10 proteins (~11%) → ~28,000 examples
- **Test**: 13 proteins (~14%) → ~37,000 examples

Rules:
- Split is stratified: each split gets a representative mix of proteins
- Negative examples are assigned to splits based on which protein they were sampled for
- Split indices are saved as TSV files (protein_name → split) for reproducibility
- Random seed: 42

### Step 1.3 — Sequence Encoding

Two encoding strategies are implemented (switchable via config):

**A) k-mer Frequency Vectors (fast baseline)**
- RNA: count all 4-mers (4^4 = 256 features) normalized by sequence length
- Protein: count all 3-mers (20^3 = 8000 features, but in practice ~4000 unique)
- Concatenate → single feature vector per (protein, RNA) pair
- Very fast, no GPU needed

**B) One-Hot Encoding (for CNN models)**
- RNA: L × 4 matrix (A/U/G/C)
- Protein: L × 20 matrix (20 standard amino acids)
- Padded/truncated to fixed length (RNA: 50 nt, Protein: 500 aa)

### Step 1.4 — Baseline Validation Model

Model options (train all, compare):

| Model | Features | Expected Training Time |
|---|---|---|
| Logistic Regression | k-mer | < 1 min |
| Random Forest | k-mer | ~5 min |
| XGBoost | k-mer | ~10 min |
| MLP (2-layer) | k-mer | ~5 min (CPU) |
| CNN (simple) | one-hot | ~30 min (GPU) or ~2h (CPU) |

**Recommended starting point**: XGBoost with k-mer features. Fast, interpretable, strong baseline.

### Step 1.5 — Evaluation

Metrics computed on the **test set** (unseen proteins):

| Metric | Expected for good data | Failure threshold |
|---|---|---|
| AUROC | > 0.70 | < 0.55 |
| AUPRC | > 0.55 | < 0.40 |
| Accuracy | > 0.65 | < 0.55 |
| Per-protein AUROC | Median > 0.65 | Any protein < 0.50 ⚠️ |

**Per-protein analysis is key**: a model might look good overall but fail on specific proteins, which signals data issues for those proteins.

Output: `results/htr_selex/metrics/validation_results.json` + plots in `results/htr_selex/plots/`

### Step 1.6 — Validation Decision

| Result | Decision |
|---|---|
| AUROC > 0.70, per-protein median > 0.65 | ✅ Dataset validated → proceed to Phase 2 |
| AUROC 0.60–0.70 | ⚠️ Marginal — investigate problematic proteins, consider reprocessing |
| AUROC < 0.60 | ❌ Dataset rejected — data quality issue, do not include in Phase 2 |

---

## Phase 2 — Generalized Model

**Purpose**: Train a single model on multiple validated datasets that generalizes to new proteins and new RNA sequences.

### Architecture

```
Protein Sequence  →  [Protein Encoder]  →  protein_embedding (dim D)
RNA Sequence      →  [RNA Encoder]      →  rna_embedding (dim D)
                                               ↓
                              [Interaction Module]
                              (concatenate / dot product / bilinear)
                                               ↓
                                    [MLP Head]
                                               ↓
                              binding_score (sigmoid → 0..1)
```

**Encoder options (start simple, scale up):**
- V1: k-mer → Linear (fastest, proof of concept)
- V2: CNN with 1D convolutions (captures local sequence motifs)
- V3: Pretrained embeddings (ESM-2 for protein, RNA-FM for RNA — best quality)

**Interaction options:**
- Concatenation + MLP (simplest)
- Bilinear interaction (captures pairwise feature interactions)
- Cross-attention (captures which protein regions interact with which RNA regions)

### Multi-Dataset Training

- Each dataset gets a `dataset_source` label
- Optionally: add `experiment_type` feature (in vivo = 1, in vitro = 0) as additional input to MLP head
- Loss: binary cross-entropy (now); add regression head for affinity (later)
- Batch sampling: stratified by dataset to prevent large datasets from dominating

### Generalization Protocol

When training on N datasets, evaluate:
1. **In-distribution test**: test proteins not seen in training (from same dataset)
2. **Cross-dataset test**: train on dataset A, test on dataset B (hardest generalization)
3. **Cross-protein test**: test on protein families never seen (requires protein family annotation)

---

## Preprocessing Pipeline (HTR-SELEX)

Step-by-step from the raw table to train/test datasets:

```
ml_dataset_simple_clean.tsv
        │
        ▼
[Step 1] Load & validate
  - Check for nulls, duplicates
  - Validate sequence characters (RNA: A/U/G/C only; Protein: standard AA)
  - Confirm label distribution
        │
        ▼
[Step 2] Protein-aware split
  - Randomly assign proteins to train/val/test (70/10/13)
  - Save protein→split mapping
        │
        ▼
[Step 3] Encode sequences
  - Option A: k-mer frequency vectors (for baseline)
  - Option B: one-hot matrices (for CNN)
  - Save encoded arrays as .npz files
        │
        ▼
[Step 4] Create balanced batches
  - HTR-SELEX is 1:2 (pos:neg) — acceptable, no resampling needed
  - For future datasets with extreme imbalance: use weighted sampling
        │
        ▼
[Step 5] Save final splits
  - data/splits/htr_selex/train.tsv  (protein_name, rna_sequence, label)
  - data/splits/htr_selex/val.tsv
  - data/splits/htr_selex/test.tsv
  - data/splits/htr_selex/split_map.tsv  (protein_name → split)
  - data/processed/htr_selex/X_train_kmer.npz
  - data/processed/htr_selex/X_val_kmer.npz
  - data/processed/htr_selex/X_test_kmer.npz
```

---

## Evaluation Metrics

| Metric | Why |
|---|---|
| AUROC | Threshold-independent, standard for imbalanced datasets |
| AUPRC | Better than AUROC for imbalanced data (our ratio is 1:2) |
| Accuracy / F1 | Interpretable for reports |
| Per-protein AUROC | Identifies weak proteins / data issues |
| Precision @ K | How good are the top predictions |
| Calibration (Brier score) | Important if we use probabilities downstream |

---

## Future Extensions

### Extension 1: in vivo / in vitro label
- Add binary feature `experiment_type` (0 = in vitro, 1 = in vivo) as input to MLP head
- Train on mixed data → model learns the difference between experimental contexts
- Enables: "predict binding in vivo conditions" vs "predict binding in vitro"

### Extension 2: Binding Affinity
- Where affinity data is available (Kd, ΔG, enrichment ratio), add regression head
- Multi-task loss: `L = L_classification + λ * L_regression`
- Affinity head only activated for samples where affinity is available (masked loss)

### Extension 3: Structural Features
- RNA secondary structure (MFE from ViennaRNA, dot-bracket notation)
- Protein domain annotation (from Pfam/UniProt)
- Adds biological context beyond sequence alone

### Extension 4: Pretrained Embeddings
- **ESM-2** (Meta, 650M params): state-of-the-art protein language model
- **RNA-FM** or **SpliceBERT**: RNA language model
- These embeddings capture evolutionary and structural information
- Plug in as frozen encoders to reduce training data requirements

---

## Quick Start Checklist

- [ ] Rename `ml/` folder to `protein_rna_ml/`
- [ ] Set up conda environment (`pip install -r requirements.txt`)
- [ ] Run `scripts/01_prepare_htr_selex.py` → creates splits + encoded data
- [ ] Run `scripts/02_train_validation_model.py` → trains XGBoost baseline
- [ ] Run `scripts/03_evaluate_validation.py` → generates full report
- [ ] Review `results/htr_selex/metrics/validation_results.json`
- [ ] Decision: AUROC > 0.70? → proceed to Phase 2
