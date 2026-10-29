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

---

## 2026-09-04 — Phase 3B: Domain-conditioned V2 (script 38)

**VM**: pleasedimpala-8f147  
**Data**: `generalized_v3a`, `known_domains_only` (465 variants; CCCH/KH/RRM/multi)  
**Model**: V2 RNABindingCNN + optional `domain_class` embedding in head  
**Seed**: 42, `prot_max=700`

| Mode | Test AUROC | Test AUPRC | pp-median | best_epoch |
|------|------------|------------|-----------|------------|
| baseline | 0.8470 | 0.7473 | 0.8594 | 17 |
| domain_conditioned | 0.8418 | 0.7447 | 0.8468 | 17 |
| domain_shuffle | **0.8549** | **0.7528** | **0.8851** | 13 |

Per-protein median by domain: CCCH and multi improve under shuffle; KH best under conditioned; RRM best under shuffle.

**Conclusion**: Coarse Table S1 `domain_class` is not a useful inductive bias for V2 on protein-disjoint v3a. Shuffle beating baseline indicates extra head parameters, not biological domain information. Phase 3B domain track closed as null.

**Outputs**: `results/domain_aware/v2_domain_cond/{mode}/v2_domain_results.json`  
**Summary**: `results/phase3b_summary.json` → `domain_aware_v2`


---

## 2026-10-28 — Skipper eCLIP audit and in vitro interaction probe (scripts 41–53)

**VM**: pleasedimpala-8f147. **Data**: Skipper eCLIP `fixlen_151` rebuilt with real protein sequences
(earlier local TSVs carried `protein_sequence == "ok"` — build scripts now abort on that).
Evaluation: `scripts/51_eclip_eval_suite.py` (protein-bootstrap CIs, GC-matched subset with caliper,
protein-shuffle control).

### Composition baselines (no protein input), test AUROC

| Split | GC | RNA k-mer LR (k≤4) | k-mer LR, GC-matched |
|---|---|---|---|
| `protein_disjoint` / `protein_and_rna` (genomic negatives) | 0.745 | 0.808 | 0.680 |
| `cross_protein_neg` (other proteins' positives) | 0.544 | 0.435 | 0.441 |

Per-protein k-mer LR with grouped CV (seen proteins, script 52): median 0.833 on genomic negatives,
**0.737** on `cross_protein_neg` → protein-specific RNA signal exists when the protein is known.

### Models on eCLIP (GC-matched AUROC unless stated; `prot_dep` = AUROC drop with shuffled proteins)

| Model | `protein_and_rna` | `cross_protein_neg` |
|---|---|---|
| V2 trained on eCLIP | full 0.858, GC-matched 0.772, prot_dep 0.003 | test 0.485 (loss stuck at ln 2) |
| V2 fine-tuned from v3a | full 0.834, GC-matched 0.711, prot_dep 0.013 | full 0.557, GC-matched 0.527, per-protein median 0.50 |
| V2 v3a zero-shot | 0.56–0.60 (full 0.70–0.72) | 0.528 (CI includes 0.5) |
| V4 v3a zero-shot | 0.52–0.54 (full 0.59–0.64) | 0.50–0.51 |

`cross_protein_neg` diagnostics (V2 same proteins in train/test 0.498; V2 lr 1e-4 0.502; V4 bilinear 0.484)
all failed. Conclusion: with composition-neutral negatives no tested model extracts protein-specific eCLIP
signal; the in vivo benchmark with genomic negatives measures "peak-likeness" (RNA-only 0.81).

### In vitro interaction probe (script 53, v3a test, ~55 proteins, ~80k pairs)

Negatives = the same RNA paired with another test protein of the same assay source (RNA marginals identical;
GC and k-mer LR give 0.500).

| Model | AUROC [95% CI, protein bootstrap] | prot_dep | pp-median |
|---|---|---|---|
| V2 (v3a) | 0.704 [0.66, 0.75] | 0.20 | 0.715 |
| V4 concat_bi (v3a) | 0.763 [0.71, 0.81] | 0.26 | 0.741 |

In the standard v3a test, shuffling proteins drops AUROC by 0.21 (V2) / 0.29 (V4).
**In vitro models learn protein×RNA interaction for unseen proteins; eCLIP transfer does not appear.**
Caveats: single seed, one 60k sample, name-level (not MMseqs) protein novelty on eCLIP.

### Decisions
- eCLIP track closed as an audit/negative-transfer result; headline benchmark becomes the cross-protein-negative probe.
- Multi-seed probe done (see above): V4 > V2 confirmed.
- MMseqs2 audit done (see below): probe signal is homology-driven.
- Next: cluster-based test split (≤30% identity), Phase 3C (affinity).

### Multi-seed in vitro probe (3 seeds each, 2026-10-29)

`scripts/54_aggregate_probe.py` on `results/overnight/probe_final` (same probe set; V2 seeds 42/0/1, V4 seeds 42/0/1).

| Model | AUROC | AUPRC | pp-median AUROC | prot_dep |
|---|---|---|---|---|
| V2 (3 seeds) | 0.686 ± 0.015 | 0.684 ± 0.010 | 0.696 ± 0.019 | 0.184 ± 0.016 |
| V4 concat_bi (3 seeds) | 0.754 ± 0.001 | 0.750 ± 0.001 | 0.759 ± 0.003 | 0.251 ± 0.002 |

Paired V4 − V2 AUROC per seed [0, 1, 42]: +0.061, +0.085, +0.058 (mean +0.068). V4 wins in every seed and is far
more stable. On the standard v3a test the V2/V4 gap is only ~0.016, so the bilinear layer helps specifically in
learning protein×RNA interaction. Caveat: n=3 seeds, one probe sample, no MMseqs audit yet.

### MMseqs2 homology audit of the probe (script 55, 2026-10-29)

Best local identity of each v3a test protein (55) to any v3a train protein (393): no hit 6, 30–50% 15, 50–70% 14,
70–90% 13, ≥90% 7 (median 0.60; 62% ≥50%). Several test proteins are identical to train ones (e.g. ESRP1-construct3 vs
ESRP1, IGF2BP1, MEX3D-construct3), so the name-level split leaks. eCLIP `protein_and_rna` test (24 proteins): no hit 16,
≥90% 5.

Probe AUROC by identity bin (mean of 3 seeds, `__full`):

| Bin (proteins) | V2 | V4 |
|---|---|---|
| no hit (6) | 0.475 | 0.525 |
| 30–50% (15) | 0.613 | 0.699 |
| 50–70% (14) | 0.667 | 0.695 |
| 70–90% (13) | 0.789 | 0.884 |
| ≥90% (7) | 0.850 | 0.874 |
| <50% (21) | 0.561 | 0.639 |
| all (55) | 0.686 | 0.758 |

AUROC rises monotonically with identity. On proteins with no detectable homolog both models are at chance
(V2 0.48, V4 0.52); partial transfer exists at 30–50% (V4 0.70 vs V2 0.61, V4 > V2 in every bin except noise at ≥90%).
**Conclusion:** the "unseen protein" claim must be restated as homology-based transfer; headline AUROC 0.75 is inflated
by close homologs. Caveats: only 6 no-hit proteins, no per-bin CI yet. Next: cluster-based split at ≤30% identity.


### Domain annotation coverage for the v3a test proteins (2026-10-29, in progress)

- Curated labels (`data/domains/protein_domains.tsv`, 84 proteins) match only 15 of 55 probe test proteins by name; the
  Skipper InterPro manifest (`encode_eclip_rbp_id_best_acc_seq.added_domain_annot.tsv`, 185 human eCLIP RBPs) adds 5 more
  (20/55 total). Most uncovered test proteins are non-human orthologs (CBG*, LmjF*, Tb_*, Smp_*, PK*).
- A regex heuristic (RNP1/RNP2, KH, CCCH) was tried and rejected: RRM recall 0.4–0.75 with 10–20% false positives against
  curated labels. Domain families are therefore assigned with Pfam-A + `hmmscan` (script 56); Pfam is downloaded to
  `data/external/pfam/` (git-ignored).
- Skipper Figshare enrichment/HOMER data (scripts 46–49): CLIPper signal vs HOMER motif PWM has Spearman median 0.03–0.04
  (only 2.5% of experiments > 0.3), so enrichment is not a clean specificity target; kept as a secondary in vivo
  within-protein ranking test after k-mer/PWM baselines.

### Pfam family labels for v3a proteins (script 56, 2026-10-29)

`hmmscan` against Pfam-A (E < 1e-3) on all 448 v3a proteins mapped to RNA-binding families. Agreement with curated
`domain_class` on train proteins: RRM 51/56, KH 9/11, CCCH 3/3, CSD 2/2, PUF 1/1 (misses go to `other`; none are
mislabelled across families). Probe test set (55 proteins): RRM 33, KH 5, multi 5, CCCH 2, CSD 1, La 1, PUF 1, CCHC 1,
other 6; train (393): RRM 279, other 45, KH 37, CCCH 17, multi 9. The test set is RRM-dominated, so non-RRM families have
very few proteins (n ≤ 5) and per-family AUROCs are indicative only.
Outputs: `results/homology_audit/v3a_pfam_hits.tsv`, `v3a_pfam_family.tsv`, `v3a_test_domain_family.tsv`.

### Probe AUROC by Pfam family (3 seeds, `__full`, 2026-10-29)

| Family (proteins) | V2 | V4 |
|---|---|---|
| RRM (33) | 0.761 | 0.815 |
| KH (5) | 0.731 | 0.813 |
| CSD (1) | 0.754 | 0.838 |
| PUF (1) | 0.778 | 0.842 |
| multi (5) | 0.601 | 0.668 |
| other (6) | 0.482 | 0.536 |
| CCCH (2) | 0.366 | 0.343 |
| CCHC (1) | 0.446 | 0.501 |
| La (1) | 0.638 | 0.546 |
| all (55) | 0.686 | 0.754 |

Families well represented in training (RRM 279 train proteins, KH 37) transfer (V4 0.81); families with few training
proteins or without a recognised RNA-binding Pfam domain (other, CCCH, CCHC, La) are at or below chance (CCCH < 0.5 with
V2 0.37 and V4 0.34, n = 2 proteins, i.e. anti-predictive, possibly a protein-specific artefact). Confound: every
no-hit protein (6) falls in other/La/CCHC, and no RRM protein is a no-hit (RRM identity bins 30–50%: 9, 50–70%: 11,
70–90%: 8, ≥90%: 5), so "sequence novelty" and "family unseen" are largely the same axis in this test set.
Per-family n is small (non-RRM ≤ 5 proteins). Next: RRM-only by identity bin (script 56 now prints it).

### RRM-only probe AUROC by identity bin (3 seeds, 2026-10-29)

| Bin (RRM proteins) | V2 | V4 |
|---|---|---|
| 30–50% (9) | 0.669 | 0.753 |
| 50–70% (11) | 0.670 | 0.714 |
| 70–90% (8) | 0.848 | 0.905 |
| ≥90% (5) | 0.896 | 0.921 |

Within the best-represented family, identity still matters (≥70% identity: 0.85–0.92; 30–70%: 0.67–0.75), but
remote RRM homologs (30–50%) still transfer (V4 0.75, V2 0.67). Family representation and sequence similarity are both
drivers; no RRM test protein lacks a homolog, so true novelty within RRM cannot be tested with this set.

### Family inventory of candidate extra sources (script 57, 2026-10-29)

Pfam families and MMseqs2 novelty (identity to v3a train < 30%) for protein sequences in `data/sanitized/new_assays`
(unique sequences: PRBAB 133, ProNAB 171, Hughes in vivo 8; HuRPA is proteome-wide gene pairs, UTexas aptamers have no
protein, both skipped).

| Family | PRBAB (novel <30%) | ProNAB (novel) |
|---|---|---|
| RRM | 29 (2) | 32 (2) |
| KH | 6 (1) | 7 (1) |
| PUF | 7 (3) | 9 (4) |
| dsRBD | 6 (6) | 7 (6) |
| DEAD | 3 (3) | 4 (3) |
| PAZ | 4 (4) | 4 (4) |
| YTH | 1 (1) | 4 (4) |
| CCCH / CCHC / C2H2 | 2 / 4 / 2 | 2 / 4 / 2 |
| other (no RBD Pfam) | 70 (65) | 93 (86) |

The two structure-derived sets (one Kd per complex, heavily overlapping) add only a handful of proteins per family; dsRBD,
DEAD, PAZ, YTH are absent from v3a train and would be new families, but with 3–6 proteins each. Most "other" entries are
short peptides/non-RBD constructs. Conclusion: the public sources already on disk do not fix the family imbalance
(v3a train: RRM 279, KH 37, CCCH 17); they are usable as small eval-only families (dsRBD, DEAD, PAZ, YTH) rather than as
training data. Output: `results/homology_audit/new_assays_family_inventory.tsv`.

### v3b cluster split (script 58, 2026-10-29)

Protein clusters from an all-vs-all MMseqs2 search with single linkage: link if identity >= 40% at >= 50% coverage, or
identity >= 70% at >= 10% coverage (close paralogs sharing a short domain, e.g. PABP). Attempts: greedy `easy-cluster`
(30%/50% cov) left PABPC1/pabpc4 (94% identity over 40% of the sequence) and CELF1/CELF1-construct2 on different sides;
single linkage at 30% identity chains all RRM proteins into one cluster of 400+ proteins, so 40% is the strictest usable
threshold. Result: 160 clusters (largest 220, RRM, kept in train), stratified by Pfam family, seed 42:
train 375 proteins / 2.06M rows, val 55 / 0.28M, test 64 / 0.32M (RRM 43, KH 8, CCCH 3, other 10).

Leakage check of the built test set (script 55 vs v3b train):

| Best identity to train | v3a test (55) | v3b test (64) |
|---|---|---|
| no hit / <30% | 6 / 0 | 6 / 3 |
| 30–50% | 15 | 53 |
| 50–70% | 14 | 2 |
| 70–90% | 13 | 0 |
| ≥90% | 7 | 0 |

Share with identity ≥50%: 62% → 3% (the two remaining hits cover only 7–14% of the query). Domain-level similarity at
30–50% remains for most test proteins and is unavoidable inside RRM. v3b = same pairs as v3a, re-split by protein; map in
`results/v3b/protein_split_map.tsv`, dataset written by `scripts/58 build` to `data/generalized_v3b/` (git-ignored).
Next: train V2/V4 (3 seeds) on v3b, standard test + cross-protein probe.


### Nearest-protein baseline on v3a (script 59, 2026-10-29)

For each training protein an RNA-only model (k-mer LR, or naive-Bayes 4-mer log-odds = PWM) is fitted on its own pairs;
a test pair is scored by the models of the most similar training proteins (MMseqs2 identity weights). No protein signal
other than sequence similarity. Protein-bootstrap 95% CI in brackets; neural numbers are 3-seed means.

| Scorer | v3a standard test | v3a cross-protein probe | probe pp-median |
|---|---|---|---|
| RNA prior (mean over all train proteins) | 0.606 | 0.500 | 0.547 |
| nearest protein, k-mer LR (nn1) | 0.779 [0.72, 0.83] | 0.724 [0.66, 0.77] | 0.831 |
| top-3 neighbours, k-mer LR (nn3) | 0.788 [0.74, 0.83] | **0.746 [0.70, 0.79]** | 0.831 |
| top-5 neighbours, k-mer LR (nn5) | 0.791 [0.74, 0.84] | 0.740 [0.69, 0.80] | 0.817 |
| nearest protein, PWM (nn1) | 0.772 | 0.714 | 0.820 |
| V2 (3 seeds) | ~0.81 | 0.686 ± 0.015 | 0.696 |
| V4 concat_bi (3 seeds) | ~0.81 | 0.754 ± 0.001 | 0.759 |

By identity bin (v3a probe, nn1 k-mer LR vs V4 from script 55): <30% (incl. no hit) 0.45 vs 0.53; 30–50% 0.71 vs 0.70;
50–70% 0.76 vs 0.70; 70–90% 0.84 vs 0.88; ≥90% 0.85 vs 0.87 (kNN column corrected: an earlier draft of script 59 shifted
the bin labels by one).
**A homology-only baseline with no protein model matches V4 and beats V2 on the
cross-protein probe, and has a much higher per-protein median AUROC (0.83 vs 0.76).** The neural models therefore have
not yet been shown to learn anything beyond homology transfer. Caveat: bin definitions differ slightly (script 59 uses
the best-bits hit, script 55 the best identity), pooled-AUROC CIs overlap; v3b results follow.

### Nearest-protein baseline on v3b (script 59, 2026-10-29)

| Scorer | v3b standard test (64 proteins) | v3b cross-protein probe (63 proteins) | probe pp-median |
|---|---|---|---|
| RNA prior (mean over train proteins) | 0.591 | 0.500 | 0.493 |
| nearest protein, k-mer LR (nn1) | **0.676 [0.61, 0.74]** | **0.627 [0.57, 0.68]** | 0.703 |
| top-3 / top-5, k-mer LR | 0.644 / 0.636 | 0.607 / 0.592 | 0.666 / 0.652 |
| nearest protein, PWM (nn1) | 0.675 | 0.625 | 0.706 |

On the cluster split (most test proteins are 30–50% identical to their nearest training protein, 24 below 30%) the
homology-only baseline drops from 0.79/0.75 (v3a) to 0.68/0.63; these are the numbers the V2/V4 models trained on v3b
must beat (pipeline `scripts/run_v3b_overnight.sh`). The bin columns printed by the first run were mislabelled (shifted
by one); the script was fixed and v3a/v3b were re-run.

### Literature comparison notes (2026-10-29; from the README reference table and the PaRPI full text, PMC12485180)

- Classical RPI predictors (RPISeq, RPI-Pred, IPMiner, RPI-SAN, RPITER, RPI-SE, NPI-GNN, Graph-RPI, SeqMG-RPI, BioLLMNet)
  solve molecule-level lncRNA/ncRNA–protein pairing on small benchmarks (RPI369/488/1807/2241, NPInter) with random or
  network-derived negatives under random-pair CV; reported AUC 0.92–0.99 are not comparable to our binding-specificity task.
  The only hard-split numbers (unknown RNA and protein, ZHMolGraph 2025 comparison): ZHMolGraph 0.798, RPITER 0.727,
  IPMiner 0.664, NPI-GNN ~0.51 AUROC. None of them reports homology-stratified results or an RNA-only / nearest-protein
  baseline.
- PaRPI (Commun. Biol. 2025; closest to our setting): 261 eCLIP/CLIP datasets (172 RBPs), ESM-2 protein embeddings +
  BERT/icSHAPE/RNAplfold RNA graph + cross-attention. Negatives are random transcriptome windows (1:2), exactly the
  genomic-background type we showed to be composition-solvable (RNA-only k-mer LR 0.81 on Skipper eCLIP); no RNA-only
  baseline is reported. Headline median AUC 0.90 on a within-protein, 80%-RNA-identity-filtered split. "Unseen RBPs" =
  4 proteins (<30% identity), positives only, accuracy 0.67 (not an AUC); homology is assessed only in 3 similarity bins.
  Median baselines compared: HDRNet, PrismNet, DeepBind, GraphProt, iDeep.
- Our distinguishing elements: composition-neutral cross-protein negatives, identity/family-stratified and cluster-split
  evaluation, nearest-protein and RNA-only baselines in every bin, and an in vivo vs in vitro transfer test.
- Missing comparisons to add on the probe/v3b: RPISeq-style (k-mer composition + RF), frozen RNA-FM/ESM-2 embeddings +
  MLP (ZHMolGraph/BioLLMNet-style).

### Shortcut audit of the classic RPI benchmarks (script 60, 2026-10-29)

RPISeq-style Random Forest (RNA 4-mer + protein 3-mer in a 7-class alphabet) on the benchmark files in
`data/external/rpi_ggcn`, plus single-molecule controls (AUROC; unique protein–RNA pairs after de-duplication):

| Dataset (pairs / proteins / RNAs) | Folds | RNA only | Protein only | Both | Per-protein label prior |
|---|---|---|---|---|---|
| RPI369 (447 / 280 / 229) | random 5-fold | 0.854 | **0.964** | 0.974 | 0.937 |
| | protein-disjoint | 0.857 | 0.909 | 0.932 | 0.820 |
| | RNA-disjoint | 0.585 | 0.949 | 0.948 | 0.846 |
| RPI488 (486 / 247 / 25) | random 5-fold | **0.909** | 0.422 | 0.906 | 0.503 |
| | protein-disjoint | 0.902 | 0.580 | 0.904 | 0.902 |
| RPI1446 (2684 / 635 / 99) | random 5-fold | 0.794 | 0.609 | 0.922 | 0.594 |
| | protein-disjoint | 0.794 | 0.507 | 0.844 | 0.792 |
| RPI2241 (2884 / 923 / 416) | random 5-fold | 0.831 | 0.594 | 0.874 | 0.595 |
| | protein-disjoint | 0.832 | 0.556 | 0.847 | 0.820 |
| | RNA-disjoint | 0.753 | 0.599 | 0.830 | 0.568 |

In these benchmarks a single molecule explains most of the signal: protein-only 0.96 vs both 0.97 (RPI369), RNA-only 0.91
vs both 0.91 (RPI488, only 25 distinct RNAs), RNA-only 0.83 vs both 0.87 (RPI2241). Hard-split numbers of 0.7–0.8 reported
by newer methods are therefore compatible with learning molecule-level priors. Caveat: our loader keeps de-duplicated
unique pairs (RPI369: 447 of 738 lines), a single RF without tuning, no PDB-redundancy removal.

### Recent related work found by search (2026-10-29; abstracts/snippets and the NPBIP full text only, not independently verified)

- **NPBIP (2026, PMC13341135)**: SimBind (Pfam/HMMER nucleic-acid-binding-domain alignment + softmax-weighted mean over the top
  m=14 similar training RBPs' predictions) + NucProNet (ESM-2 mean-pooled protein tower, dilated-conv + transformer
  RNA tower, cosine bilinear head), outputs z-scored and summed. RNAcompete, 88 held-out test experiments (split by RBP,
  no identity filter): NPBIP r = 0.414, NucProNet 0.394, SimBind (similarity only) 0.367, JPLE 0.321, k-mer upper bound
  0.46; performance vs maximum training similarity: r = 0.63. This independently supports our two findings (similarity-
  only transfer rivals deep models; performance tracks homology) and our planned hybrid kNN + neural design. It
  evaluates on RNAcompete intensities (Pearson), not cross-protein-negative classification.
- **"Generalizable deep-learning-based mRNA-protein interaction prediction strongly depends on protein diversity"
  (J. Cheminformatics 2026)**: per the search snippet, high reported scores reflect RBP overlap between train and test,
  drop sharply on unseen RBPs, protein-language-model encodings do not fix it; releases an RBP-aware benchmark. Full text
  not accessible (cookie redirect) — read before citing.
- ZeRPI (Methods 2025; GNN, zero-shot RNA–protein interactions, contrastive loss), RBPsuite 2.0 (BMC Biol. 2025), IRIS
  (bioRxiv 2025; biophysical affinity + motif discovery), RMDNet (2025), RBPSignal (bioRxiv 2025): not yet read.
- PaRPI data/code: Zenodo 10.5281/zenodo.14878562 (dataset.7z 499 MB, BERT_Model.7z 319 MB), GitHub ljquanlab/PaRPI;
  downloaded to `data/external/parpi/` (git-ignored).

### Split integrity audit (script 62, 2026-10-29)

| Check | v3a | v3b |
|---|---|---|
| Test/val protein names in train | 1 test, 2 val | 0 |
| Test/val protein sequences identical to train | 1 test, 5 val | 0 |
| Rows with the exact pair in train | test 0%, val 0.14% | 0% |
| Test rows whose RNA occurs in train | 69.4% | 73.3% |
| Test rows whose RNA is a positive in train (any protein) | 50.3% | 51.8% |
| Duplicate pairs (train) / conflicting labels | 1129 / 0 | 1567 / 0 |
| Positive fraction train/val/test | 0.310 / 0.292 / 0.306 | 0.305 / 0.309 / 0.318 |

The high RNA overlap (~70%) comes from assays reusing the same probe sets across proteins, not from protein leakage; it
is why the standard test is partly solvable from RNA alone (RNA-prior 0.59–0.61) and why the cross-protein probe, where
RNA marginals are identical by construction, is the primary metric. Sequential plan: `NEXT_STEPS.md`.
