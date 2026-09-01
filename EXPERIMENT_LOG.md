# Experiment Log

**Project**: Protein–RNA Binding Prediction
**Last updated**: 2026-09-01

This file is the canonical record of every training run. Each entry documents what
was run, what the results were, what failed, and what was learned. All Phase 2 results have been retrained with the double class-weighting bug fixed (2026-05-13).
Numbers in this log reflect clean checkpoints unless otherwise noted.

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
| Date | 2026-05 (clean rerun 2026-05-13) |
| Architecture | RNA CNN [128,256,256] kernels [7,5,3] + Prot CNN [128,256,256] kernels [11,7,5] → MLP [256,64] |
| RNA max len | 60 nt |
| Protein max len | 300 aa |
| LR | 5e-4, cosine schedule |
| Batch size | 256 |
| Early stopping | Val AUPRC, patience=8 |
| Device | MPS (Apple Silicon) |
| pos_weight | n_neg / n_pos ≈ 2.0 (WeightedRandomSampler removed) |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.746 | — |
| Test | **0.690** | **0.580** |
| Val→Test gap | −0.056 | — |

**Best epoch**: 13

**Per-protein test highlights** (24 proteins):
- Median AUROC: 0.714
- Best: ESRP1-construct3 (0.980), PUF60 (0.934), KHDRBS3 (0.901)
- Worst: UNK (0.448), ZC3H18 (0.554), RBM6 (0.537)

**Result**: PASS — current best model. Length-agnostic via global max pooling.
Fails on diffuse/low-complexity binders (UNK) and proteins with complex binding modes.

---

### EXP-V3 — Frozen ESM-2 mean-pool + RNA CNN

| Hyperparameter | Value |
|----------------|-------|
| Date | 2026-05 |
| Architecture | ESM-2(1280-d mean-pool) → Linear(256) + GELU + RNA CNN [128,256,256] [7,5,3] → MLP [256,64] |
| ESM-2 model | esm2_t33_650M_UR50D (frozen, no fine-tuning) |
| LR | 5e-4 |
| Batch size | 512 |
| Rerun | 2026-05-13 (double class-weighting bug fixed) |

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
| Rerun | 2026-05-13 (double class-weighting bug fixed) |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.770 | 0.704 |
| Test | 0.666 | 0.568 |

**Result**: FAIL vs V2 (−0.024 AUROC on test). ESM-2 mean-pool is actively harmful
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
| Rerun | 2026-05-13 (double class-weighting bug fixed) |

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.745 | 0.683 |
| Test | **0.685** | **0.595** |

**Best epoch**: 18

**Comparison**:
- Δ vs V2: AUROC −0.005, AUPRC +0.015 ← close to V2 but not better
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

## Phase 3A — V2 CNN on `generalized_v3a`

**Date**: 2026-07-11  
**Dataset**: `data/generalized_v3a/` — 2,658,999 pairs, 494 proteins (SELEX + RBNS + RNAcompete Eukarya/RBPZoo + ucRBP 23)  
**Scripts**: `scripts/22a`, `scripts/22`, `scripts/06_train_generalized_v2.py`, `scripts/06_eval_generalized_v2_test.py`  
**Checkpoint**: `models/saved/generalized_v2/best_model.pt` (epoch 24, early stop on val AUPRC)  
**Hardware**: VM CPU, `prot_max=700`, batch 512, ~11 h/epoch

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Val | 0.818 | 0.693 |
| Test | **0.813** | **0.713** |
| Per-protein median test AUROC | **0.817** | (55 test proteins) |

**Status**: PASS vs Phase 3A targets (test AUROC ≥ 0.70). Single seed only.

### External validation (same checkpoint)

| Benchmark | n | AUROC | AUPRC | Notes |
|-----------|---|-------|-------|-------|
| Curated literature | 159 | 0.763 | 0.915 | pos rate 72%; baseline AUPRC 0.717 |
| Expanded + generated negs | 540 | 0.688 | 0.488 | pos rate 21%; baseline AUPRC 0.211; diagnostic |

Built expanded set with `scripts/31_build_external_benchmark.py` (shuffle + cross-pair decoys).

**Figures**: `python scripts/32_visualize_phase3a_results.py` →
`figures/phase3a_v2_scale_comparison.png`, `phase3a_per_protein_auroc.png`,
`phase2_model_comparison.png` (refreshed with v3a row),
`external_eval_comparison.png`, `external_score_distributions.png`.

**Summary JSON**: `results/phase3a_summary.json`

### Not yet run on v3a checkpoint

- RNAcompete zero-shot (`scripts/20`) — compare to v2 median 0.549
- Multi-seed variance (`scripts/18`) on v3a

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

| ID | Name | Purpose | Status | Result |
|----|------|---------|--------|--------|
| EXP-V2-CLEAN | V2 retrain with corrected weighting | Clean anchor baseline | **Done** | Test AUROC 0.690, AUPRC 0.580 |
| EXP-RNACOMPETE-V2 | RNAcompete zero-shot on V2 | Generalization test | **Done** | Median AUROC 0.549 |
| EXP-PHASE3A | V2 on SELEX+RNAcompete data | Scale training | **Done** | Test AUROC 0.813 on v3a |
| EXP-EXT-31 | Literature benchmark + generated negs | External diagnostic | **Done** | Curated 0.763; expanded 0.688 |
| EXP-V2-MULTISEED-V3A | V2 ×5 seeds on v3a | Quantify variance | **Next** | — |
| EXP-RNACOMPETE-V3A | RNAcompete zero-shot on v3a ckpt | Generalization test | Next | — |
| EXP-V4-BILINEAR | V4 bilinear interaction | Pairwise interaction layer | Scripted | `scripts/21` ready |

---

## Infrastructure Update — Reproducibility & Multi-seed Runner (2026-05-28)

**Scripts updated**: `scripts/06_train_generalized_v2.py`, `scripts/18_run_multiseed.py`

### Changes

| Component | Change |
|-----------|--------|
| `06_train_generalized_v2.py` | Added `--seed` (seeds random/numpy/torch + DataLoader Generator) |
| `06_train_generalized_v2.py` | Added `--dry_run` (single batch pass, no disk writes) |
| `06_train_generalized_v2.py` | `num_workers=0` default on macOS (prevents multiprocessing freeze) |
| `18_run_multiseed.py` | Auto-injects `--model_dir seed_N/checkpoints` + `--out_dir seed_N/` per seed |
| `18_run_multiseed.py` | `--live` flag streams epoch logs to terminal while still writing `train.log` |
| `18_run_multiseed.py` | Sets `PYTHONUNBUFFERED=1` in child process environment |

### Multi-seed Status

seed_0 checkpoint saved to `results/multiseed/v2_cnn/seed_0/checkpoints/best_model.pt`.
Seeds 1-4 pending. Run command:

```bash
python scripts/18_run_multiseed.py \
    --script scripts/06_train_generalized_v2.py \
    --seeds 1 2 3 4 \
    --output_dir results/multiseed/v2_cnn \
    --extra_args "--data_dir data/generalized_v2 --epochs 60" \
    --live
```

---

## RNAcompete Zero-Shot Benchmark — V2 CNN (Clean Checkpoint)

**Date**: 2026-05-13  
**Script**: `scripts/20_evaluate_benchmark.py`  
**Checkpoint**: `models/saved/generalized_v2/best_model.pt` (clean, bug fixed)  
**Benchmark**: `data/benchmarks/rnacompete/rnacompete_all.tsv` (2M rows subsampled from 13.9M)

### Results

| Metric | Value | Interpretation |
|---|---|---|
| Overall AUROC | **0.571** | Marginally above random (0.5) |
| Overall AUPRC | **0.391** | vs random baseline 0.316 |
| Per-protein median AUROC (all 742) | **0.551** | Near-random |
| Per-protein median — seen in training (27) | **0.629** | Slight memorization benefit |
| Per-protein median — truly unseen (715) | **0.549** | Near-random zero-shot |
| % proteins with AUROC > 0.7 | 17.3% | |
| % proteins with AUROC > 0.5 | 61.9% | |

### Top / Bottom Proteins

**Best** (AUROC > 0.90): TAF15 (0.982), EWSR1 (0.961), RBFOX2 (0.949), ZRANB2 (0.934), ESRP1 (0.916)  
Of these, EWSR1, RBFOX2, ZRANB2, ESRP1 are in the training set — partial memorization.

**Worst** (AUROC < 0.26): ETF1 (0.215), RNF31 (0.215), NUDT6 (0.228), DCN (0.241)  
These are likely non-canonical RBPs (many from ucRBP) with binding specificity
the model has never seen evidence of.

### Per-Organism (selected, after name normalization fix)

| Organism | AUROC | n |
|---|---|---|
| Saccharomyces cerevisiae | 0.830 | 2,277 |
| Puccinia graminis | 0.811 | 5,995 |
| Monodelphis domestica | 0.807 | 10,295 |
| Arabidopsis thaliana | 0.674 | 13,283 |
| Homo sapiens (RBPZoo/Eukarya) | 0.671 | 42,055 |
| Homo sapiens (ucRBP, 613 RBPs) | 0.554 | 1,452,731 |
| Leishmania major | 0.534 | 54,639 |

Human ucRBP (0.554 on 1.45M pairs, 613 proteins) is the most important number:
the model barely generalizes to unseen human RBPs from the same in vitro assay.

### Data Quality Issues Found

- Organism name normalization bug: ucRBP uses underscores (`Homo_sapiens`),
  other sub-datasets use spaces. Fixed in `17_prepare_rnacompete_benchmark.py`.
  Benchmark must be re-generated to get clean per-organism numbers.

### Interpretation

This is a **true zero-shot generalization failure**. The model trained on 169 proteins
from HTR-SELEX + RBNS cannot generalize to 715 unseen proteins across 26 organisms.
The 0.08 gap between seen (0.629) and unseen (0.549) median AUROC shows limited
memorization and limited generalization — the worst possible combination.

This result is from the clean V2 checkpoint with ~170 training proteins.
It establishes the **current baseline** for zero-shot generalization.
The bottleneck is not the model architecture or the bug — it is the number
and diversity of training proteins. Phase 3A (adding RNAcompete to training) is
the highest-leverage next step.

### Action Items

See STRATEGY.md §Next Steps for the full prioritized plan.

---

## 2026-06-16 — RNAcompete intensity spectrum samples (scripts 27–28)

**Request**: 100 probes per protein, evenly sampled across log-intensity percentiles at modal
probe length; top 3 RBPs by mean positive intensity; test protein length vs intensity correlation.

**Ran on**: RNAcompete Eukarya (`ml_dataset_eukarya_clean.tsv.gz`, 200 RBPs) and RBPZoo
(`ml_dataset_rbpzoo_clean.tsv.gz`, 174 RBPs).

| Panel | Top 3 sampled | Modal length | Length vs intensity (Pearson) |
|---|---|---|---|
| Eukarya | ARET, SF2, BRU-3 | 38 nt | r = −0.050, p = 0.49 |
| RBPZoo | LmjF.24.1570, RBFOX2, LmjF.34.4560 | 38 nt | r = −0.036, p = 0.63 |

**Outputs**:
- `results/rnacompete_intensity_spectrum/{dataset}/spectrum_samples_{dataset}.tsv`
- `figures/rnacompete_length_vs_intensity.png`, `figures/rnacompete_intensity_spectrum.png`

**Conclusion**: no evidence that longer RBPs yield higher mean positive RNAcompete intensity.
Spectrum TSVs: `results/rnacompete_intensity_spectrum/*/spectrum_samples_*.tsv`.

---

## 2026-08-30 — Phase 3B: V4 bilinear on generalized_v3a (GPU P100)

**Script**: `scripts/21_train_generalized_v4_interaction.py`  
**Config**: `concat_bi`, `--use_source_emb`, `prot_max=700`, `batch_size=512`, seed=42  
**Checkpoint**: `models/saved/generalized_v4_phase3a/best_model.pt` (epoch 32, early stop 42)

| Split | V2 v3a | V4 concat_bi | Δ |
|-------|--------|--------------|---|
| Test AUROC | 0.813 | **0.829** | +0.016 |
| Test AUPRC | 0.713 | **0.735** | +0.022 |
| Per-protein median test AUROC | 0.817 | **0.851** | +0.034 |

**External (literature, long RNA, sliding window max)** — `scripts/11` / `scripts/21b`:

| Subset | V2 AUROC | V4 AUROC | V2 AUPRC | V4 AUPRC |
|--------|----------|----------|----------|----------|
| Curated (159) | 0.763 | 0.737 | 0.915 | 0.890 |
| Expanded (540) | 0.688 | 0.666 | 0.488 | 0.341 |

**Learned**: Bilinear + source embedding improves held-out **in vitro** test but does **not**
improve (slightly hurts) OOD literature external. Likely factors: assay-specific source_emb
(zero vector at external inference), bilinear overfit to SELEX/RNAcompete pairing patterns.

**Summary JSON**: `results/phase3b_summary.json`  
**Next**: multi-seed V4 (`scripts/18`); cross-protocol (`scripts/33–36`); optional V4 w/o source_emb.

### P3B-V4-MULTISEED — 3 seeds on generalized_v3a

**Script**: `scripts/18_run_multiseed.py` → `scripts/21_train_generalized_v4_interaction.py`  
**Seeds**: 42, 0, 1 — all succeeded  
**Output**: `results/multiseed/v4_concat_bi_v3a/summary.json`

| Metric | mean ± std | min | max |
|--------|------------|-----|-----|
| Test AUROC | **0.829 ± 0.009** | 0.819 | 0.837 |
| Test AUPRC | **0.732 ± 0.011** | 0.720 | 0.741 |
| Per-protein median AUROC | 0.854 ± 0.004 | 0.851 | 0.858 |

Best test seed: **0** (AUROC 0.837). Variance is modest; V4 gain over V2 (0.813) is stable across seeds.

---

## 2026-09-01 — Phase 3B: Cross-protocol in-vitro (scripts 33–36)

**VM**: pleasedimpala-8f147 (`/vol/space/protein_rna_ml`)  
**Config**: `configs/cross_protocol_invitro.yaml` — HTR-SELEX, RBNS, RNAcompete Eukarya/RBPZoo (no eCLIP)  
**Model**: k=4 logistic regression, representative native exact match (roster from script 33)

**Roster**: 84 proteins (≥2 protocols), 82.1% domain-annotated (Table S1)

### Within-protocol (matched proteins, honest train/val/test)

| Protocol | Proteins evaluated | Median AUROC |
|----------|-------------------|--------------|
| HTR-SELEX | 54 | 0.950 |
| RBNS | 60 | 0.993 |
| RNAcompete Eukarya | 58 | 0.992 |
| RNAcompete RBPZoo | 24 | 0.985 |

**Mean within AUROC**: 0.974 (196 protein×protocol rows)

### Cross-protocol transfer (train A → test B, same protein)

| Direction | Median AUROC | n |
|-----------|--------------|---|
| htr_selex → rbns | 0.804 | 33 |
| htr_selex → rnacompete_eukarya | 0.783 | 34 |
| htr_selex → rnacompete_rbpzoo | 0.734 | 12 |
| rbns → htr_selex | **0.703** | 33 |
| rbns → rnacompete_eukarya | 0.766 | 38 |
| rbns → rnacompete_rbpzoo | 0.771 | 14 |
| rnacompete_eukarya → htr_selex | 0.832 | 34 |
| rnacompete_eukarya → rbns | 0.936 | 38 |
| rnacompete_eukarya → rnacompete_rbpzoo | 0.953 | 11 |
| rnacompete_rbpzoo → htr_selex | 0.872 | 12 |
| rnacompete_rbpzoo → rbns | **0.961** | 14 |
| rnacompete_rbpzoo → rnacompete_eukarya | **0.987** | 11 |

**Mean transfer AUROC**: 0.791 (284 rows). **24/284** below 0.55.  
RNA sequence overlap ≈ 0 for SELEX/RBNS/RNAcompete cross-pairs; ~23% for Eukarya↔RBPZoo.

### Motif concordance (script 35, top/bottom 7-mers from script 24)

- 142 protein×protocol pairs; median Jaccard (exact 7-mer) = 0.0; core-5-mer > 0 in 66/142 pairs
- Transfer works without shared top k-mers across assays

**Outputs**: `results/cross_protocol_invitro/`; figures `figures/cross_protocol_*.png`  
**Summary**: `results/phase3b_summary.json` → `cross_protocol_invitro`  
**Next**: domain-conditioned V2 (`scripts/38`)

