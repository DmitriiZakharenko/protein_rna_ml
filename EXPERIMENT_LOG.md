# Experiment Log

**Project**: Protein–RNA Binding Prediction
**Last updated**: 2026-05-13

This file is the canonical record of every training run. Each entry documents what
was run, what the results were, what failed, and what was learned. Results that are
still believed to be affected by known bugs are explicitly marked.

---

## Phase 1 — Per-Dataset Validation

**Purpose**: Confirm each dataset has a learnable signal before merging.
**Split**: Protein-aware, seed=42, 75/11/14% train/val/test by protein count.
**Encoding**: RNA 4-mer (256 features) + Protein 3-mer (8000 features), normalized.
**Models**: Logistic Regression, Random Forest, XGBoost.

### P1-HTR-25907 — HTR-SELEX PRJEB25907

| Model | Val AUROC | Val AUPRC |
|-------|-----------|-----------|
| XGBoost | **0.825** | **0.742** |
| LR | 0.771 | 0.639 |
| RF | 0.759 | 0.644 |
| XGBoost test-set | 0.796 | 0.693 |

**Status**: PASS ✅
**Flagged proteins**: PCBP1 (0.436), LARP6 (0.693), RBM6 (0.683), RBMS2 (0.656)

### P1-RBNS — RNA Bind-n-Seq

| Model | Val AUROC | Val AUPRC |
|-------|-----------|-----------|
| RF | **0.758** | **0.684** |
| LR | 0.746 | 0.636 |
| XGBoost (no early stop) | 0.661 | 0.522 |
| RF test-set | 0.632 | 0.507 |

**Status**: PASS ✅ (with caveats: RBNS test is lower than val due to length-distribution shift)
**Flagged proteins**: RBM4 (0.345), RBM4B (0.323) — confirmed paralogs; XRCC6 (0.496) — atypical RBP

**Issue discovered**: XGBoost `early_stopping_rounds` must be passed to constructor in v2.x, not to `fit()`.
Without this, XGBoost trained for all 300 rounds with diverging val loss. Bug corrected.

### P1-HTR-47428 — HTR-SELEX PRJEB47428

| Model | Val AUROC | Val AUPRC |
|-------|-----------|-----------|
| LR | **0.817** | **0.736** |
| XGBoost | 0.629 | 0.454 |
| LR test-set | 0.590 | — |
| XGBoost test-set | 0.628 | 0.436 |

**Status**: PASS ✅ (only 4 test proteins — high variance, interpret with caution)

---

## Phase 2 — Generalized Model

**Combined dataset**: 659,004 rows, 169 proteins, 3 SELEX/RBNS datasets + eCLIP (train-only).
**Split**: Global protein-aware, seed=42, ~75/11/14% by protein count.
**Test set**: 24 proteins (98,662 rows).

### EXP-V1 — MLP on k-mer features

| Hyperparameter | Value |
|----------------|-------|
| Date | 2026-04 |
| Architecture | MLP [512, 256, 128], input=8262-d k-mer |
| LR | 1e-3 |
| Epochs | 60 (early stop patience=8) |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.716 | 0.636 |
| Test | 0.674 | 0.544 |

**Result**: FAIL — Val AUROC peaked at epoch 1 then degraded monotonically.

**Root cause**: RNA k-mer frequency is length-dependent. RBNS sequences are 20 nt; HTR-SELEX
are 40 nt. A 4-mer count over 20 nt has a different magnitude and distribution than over
40 nt. StandardScaler cannot fix this because the issue is structural (different feature
spaces, not just different scales). The MLP sees cross-dataset examples with incompatible
representations as if they were comparable.

**Lesson**: Any model that uses positional-length-dependent features must address the
RBNS/HTR-SELEX RNA length mismatch explicitly. Global max pooling in CNNs resolves this.

---

### EXP-V2 — Dual-branch CNN on one-hot sequences

| Hyperparameter | Value |
|----------------|-------|
| Date | 2026-05 |
| Architecture | RNA CNN [128,256,256] kernels [7,5,3] + Prot CNN [128,256,256] kernels [11,7,5] → MLP [256,64] |
| RNA max len | 60 nt |
| Protein max len | 300 aa |
| LR | 5e-4, cosine schedule |
| Batch size | 256 |
| Early stopping | Val AUPRC, patience=8 |
| Device | MPS (Apple Silicon) |
| **Known bug** | WeightedRandomSampler + pos_weight used simultaneously (double class-weighting). Fix applied 2026-05-13 to scripts/06. |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.811 | 0.734 |
| Test | **0.703** | **0.599** |
| Val→Test gap | −0.108 | −0.135 |

**Best epoch**: 20/28

**Per-protein test highlights** (24 proteins):
- Median AUROC: 0.718
- Best: ESRP1-construct3 (0.981), PUF60 (0.924), KHDRBS3 (0.901)
- Worst: UNK (0.429), IGF2BP3 (0.459), TAF15 (0.468)
- HTR-SELEX median: 0.779; RBNS median: 0.611

**Result**: PASS — current best model. Length-agnostic via global max pooling. Fails on
diffuse/low-complexity binders (UNK, TAF15) and multi-domain proteins (IGF2BP3).

**Note on per-protein dataset column**: All per-protein results in `v2_cnn_results.json`
show `"dataset": "unknown"` due to column name mismatch bug. Fix applied to scripts/06.
Re-run to get correctly labelled per-protein breakdown.

---

### EXP-V3 — Frozen ESM-2 mean-pool + RNA CNN

| Hyperparameter | Value |
|----------------|-------|
| Date | 2026-05 |
| Architecture | ESM-2(1280-d mean-pool) → Linear(256) + GELU + RNA CNN [128,256,256] [7,5,3] → MLP [256,64] |
| ESM-2 model | esm2_t33_650M_UR50D (frozen, no fine-tuning) |
| LR | 5e-4 |
| Batch size | 512 |
| **Known bug** | Same double class-weighting as V2. |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.715 | 0.675 |
| Test | 0.634 | 0.547 |

**Best epoch**: 17/25

**Training dynamics**: Very noisy (val AUPRC oscillates ±0.05 per epoch). V2 smoothly
peaked at epoch 20. This instability is diagnostic of the ESM-2 projection layer
conflicting with the RNA branch's learned representations.

**Per-protein vs V2**:
- V3 better: UNK +0.245, HNRNPA0 +0.088, PUF60 +0.069, IGF2BP3 +0.095, TAF15 +0.060
- V3 worse: PRR3 −0.408, PUM2 −0.253, LARP7 −0.202, TRA2A −0.150, DAZAP1 −0.144

**Root cause**: Mean-pooling over all 300 residues dilutes the binding domain signal by
6–15×. Proteins with strong localized motifs (Pumilio domain, La-motif, RRM) lose their
discriminative signal. Proteins without clear sequence motifs benefit from the evolutionary
context in ESM-2.

**Result**: FAIL vs V2 (−0.069 AUROC on test). Key learning: V2 CNN and ESM-2 encode
complementary information. Simple replacement fails; explicit combination is needed.

---

### EXP-V3b — V2 CNN + ESM-2 mean-pool auxiliary

| Hyperparameter | Value |
|----------------|-------|
| Date | 2026-05 |
| Architecture | RNA CNN(256) + Prot CNN(256) + ESM-2(1280→128) auxiliary → concat 640-d → MLP [256,64] |
| LR | 5e-4 |
| Batch size | 256 |
| **Known bug** | Same double class-weighting. |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.770 | 0.704 |
| Test | 0.666 | 0.568 |

**Result**: FAIL vs V2 (−0.037 AUROC on test). ESM-2 mean-pool is actively harmful
even when the proven V2 CNN branches are present. This rules out architecture
(specifically concat vs replace) as the cause of V3's failure. The failure mode is
the mean-pool operation itself.

---

### EXP-V3c — ESM-2 residue Conv1D

| Hyperparameter | Value |
|----------------|-------|
| Date | 2026-05 |
| Architecture | RNA CNN [128,256,256] [7,5,3] + ESM-2 residue Linear(1280→64) → Conv1D [128,256] [7,5] → max pool(256) → MLP [256,64] |
| ESM-2 embeddings | Per-residue (L×1280), fp16, padded to prot_max=300 |
| LR | 3e-4 |
| Batch size | 128 (reduced for memory) |
| Training time | 149.4 min (MPS) |
| **Known bug** | Same double class-weighting. |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.745 | 0.683 |
| Test | **0.685** | **0.595** |

**Best epoch**: 18

**Comparison**:
- Δ vs V2: AUROC −0.018, AUPRC −0.004 ← still below V2
- Δ vs V3b: AUROC +0.019, AUPRC +0.028 ← better than mean-pool variants

**Result**: FAIL vs V2. Residue Conv1D partially addresses the mean-pool dilution
problem (confirming the V3/V3b diagnosis) but does not recover full V2 performance.

**Interpretation**: 169 training proteins provide insufficient signal for the residue
projection layer (1280→64) to learn to suppress non-binding residues and amplify binding
domain residues. The model likely needs either (a) more proteins with diverse binding
modes, (b) supervision on which residues constitute the binding domain (auxiliary task),
or (c) fine-tuning ESM-2 instead of using frozen embeddings.

**Final verdict on frozen ESM-2 variants** (3 experiments, 2 months):
All three approaches (mean-pool, concat, residue CNN) fail to beat one-hot CNN.
The architectural hypothesis that "ESM-2 positional selectivity will recover V2
performance" was falsified by V3c. Do not run further frozen ESM-2 variants without
a fundamentally different design (fine-tuning, cross-attention, binding-domain supervision).

---

## Known Bugs Discovered During Phase 2

| Bug | Discovery | Scripts affected | Fix date |
|-----|-----------|-----------------|----------|
| Double class-weighting (WeightedRandomSampler + pos_weight) | 2026-05 audit | 06, 08, 09, 10 | 2026-05-13 |
| Per-protein "dataset" column always "unknown" | 2026-05 audit | 06 | 2026-05-13 |
| Broken script reference in run_pipeline.sh | 2026-05 audit | run_pipeline.sh | 2026-05-13 |
| Missing transformers/sentencepiece/openpyxl in requirements.txt | 2026-05 audit | requirements.txt | 2026-05-13 |
| XGBoost early_stopping_rounds in constructor vs fit() | 2026-04 | 02_train_validation_model.py | 2026-04 |

---

## Next Planned Experiments

| ID | Name | Purpose | Prerequisites | Expected output |
|----|------|---------|---------------|-----------------|
| EXP-V2-CLEAN | V2 retrain with corrected weighting | Clean anchor baseline | scripts/06 fix (done) | Updated test AUROC ≈ 0.70±0.02 |
| EXP-V2-MULTISEED | V2 ×5 seeds | Quantify variance | V2-CLEAN | AUROC mean±std |
| EXP-HOMOLOGY | Homology audit | Quantify paralog leakage | MMseqs2 install | Leakage fraction |
| EXP-HARDNEG | V2 with hard negatives | Better negatives | scripts/15 (todo) | AUROC on new test |
| EXP-INTERACTION | Bilinear V2 | Pairwise interaction | V2-CLEAN | AUROC vs V2 |
