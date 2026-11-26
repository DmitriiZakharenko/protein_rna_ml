# Research Strategy

> **Status (2026-11-26)**: sections 1-2 and 5-7 record the lessons of Phase 2 and the RNAcompete zero-shot diagnosis (2026-05). Sections 3, 4, 6 and 8 were updated on 2026-11-26 to the state after the homology audit and v3b. Current results: `README.md` and `EXPERIMENT_LOG.md`. Current plan: `NEXT_STEPS.md`.

**Project**: Protein–RNA Binding Prediction
**Last updated**: 2026-11-26
**Current best in-distribution model**: V4 concat_bi on `generalized_v3a` — test AUROC=**0.829 ± 0.009** AUPRC=**0.732 ± 0.011** (3 seeds, P100)
**V2 anchor on v3a**: test AUROC=**0.813** AUPRC=**0.713** (single seed)
**Phase 2 baseline (v2 data)**: test AUROC=0.690 AUPRC=0.580 (169 proteins)
**Cross-protocol transfer (k=4 LR, in-vitro)**: mean AUROC **0.791** vs within **0.974** — see `results/phase3b_summary.json`
**Benchmark target**: ZHMolGraph — AUROC=0.798 AUPRC=0.820 (hard split, unseen proteins AND RNAs).
**Note on comparability**: our protein-disjoint test is not equivalent to ZHMolGraph's hard split.

---

## 1. What The Experiments Have Actually Proven

### 1.1 One-hot CNN beats frozen ESM-2 (three independent experiments)

| Model | Test AUROC | Test AUPRC | vs V2 |
|-------|-----------|------------|-------|
| V2 CNN (one-hot) | **0.690** | **0.580** | — |
| V3c ESM-2 residue CNN | 0.685 | 0.595 | −0.005 |
| V3b CNN + ESM-2 mean-pool | 0.666 | 0.568 | −0.024 |
| V3 ESM-2 mean-pool only | 0.634 | 0.547 | −0.056 |

This is not evidence that ESM-2 is uninformative. It is evidence that:
(a) frozen mean-pooling over all residues dilutes binding-domain signal;
(b) residue-level Conv1D on frozen ESM-2 partially recovers positional selectivity
    but does not yet beat one-hot sequence convolution on 169 proteins;
(c) the gains from ESM-2 (motif-poor proteins: UNK +0.25, TAF15 +0.06) are real
    but outweighed by losses on motif-rich proteins (PUM2 −0.25, LARP7 −0.20).

**Implication drawn in 2026-05**: further frozen mean-pool variants were not pursued, and the options named were either (a) fine-tuning, (b) explicit interaction modeling, or (c) better data.

### 1.2 Validation-test gap (systematic, causes not separated)

All models show val AUROC ≈ test AUROC + 0.05–0.08 (clean runs). The gap is present in all models.
Root causes not yet analyzed:
- HTR-SELEX vs RBNS distribution shift (RNA length, selection protocol)
- Protein family imbalance between val and test splits (not verified)
- Early stopping on val AUPRC may overfit to val split (only one seed used)

Until this gap is explained, val AUPRC is not a reliable optimization target.

### 1.3 SELEX/RBNS negatives are easy negatives

All negatives in the SELEX/RBNS training data are shuffled or non-cognate random sequences.
The model may have learned to distinguish "enriched sequence" from "statistically random
sequence" rather than learning the binding affinity landscape. The eCLIP negatives
(flanking genomic regions, GC/length-matched) are much harder but are currently
restricted to training only and represent a small fraction of the data (4% = 26k/659k pairs).

---

## 2. Bugs Fixed 2026-05-13 (All Checkpoints Retrained)

### Bug 1 — Double class-weighting (CRITICAL)
**Scripts affected**: 06, 08, 09, 10
**Description**: `WeightedRandomSampler` (creates ~50/50 batches) was used simultaneously
with `BCEWithLogitsLoss(pos_weight=n_neg/n_pos ≈ 2.0)`. Combined effect: each positive
gets 2× larger gradient weight despite already appearing at 2× baseline frequency in
batches. Equivalent to pos_weight ≈ 4× in a standard unsampled setting.
**Impact**: inflated val AUROC (0.811 → 0.746 for V2), inflated test AUROC (0.703 → 0.690),
off-calibration, non-reproducible gradients.
**Fix**: `WeightedRandomSampler` removed from all four training scripts. `pos_weight` kept.
**Status**: Fixed and retrained. All checkpoints (V2, V3, V3b, V3c) regenerated. Numbers in
this document and all result JSONs reflect the clean runs.

### Bug 2 — Per-protein dataset column always "unknown" (HIGH)
**Script affected**: 06_train_generalized_v2.py
**Description**: Per-protein results in v2_cnn_results.json show `"dataset": "unknown"`
for all proteins. The column is `dataset_source` in `data/generalized/` splits
but `dataset` in `data/generalized_v2/` splits. The script read the wrong column.
**Fix**: Column normalization added at load time.
**Status**: Fix applied. Re-run script 06 to get correctly labelled breakdowns.

### Bug 3 — Broken pipeline runner (MEDIUM)
**File**: `scripts/archive/run_pipeline.sh` (moved from the repository root)
**Description**: `phase1_htr()` called non-existent `scripts/02b_train_htr_selex_validation.py`.
**Fix**: Corrected to `scripts/02_train_validation_model.py --config configs/htr_selex_validation.yaml`.

### Bug 4 — Missing dependencies (HIGH)
**File**: requirements.txt
**Description**: `transformers`, `sentencepiece`, `openpyxl` were not listed.
All ESM-2 scripts and the external evaluation script would fail on a fresh install.
**Fix**: Dependencies added to requirements.txt.

---

## 3. State of the work (2026-11-26)

- `generalized_v3a` built (2,658,999 pairs, 494 proteins). V2 test AUROC 0.813 (1 seed), V4 concat_bi 0.829 +- 0.009 (3 seeds).
- External curated literature pairs: V2 0.763, V4 0.737 (159 pairs).
- Homology audit (`scripts/55`) and cluster split `generalized_v3b` (`scripts/58`): test proteins with a training protein at >= 50% identity 62% (v3a) and 3% (v3b).
- Cross-protein-negative probe on v3b: nearest-protein baseline 0.627, V2 0.527 and V4 0.559 (seed 42), frozen ESM-2 + RNA-FM bilinear head 0.559 +- 0.010 (3 seeds).
- Zero-shot RNAcompete evaluation applies to the Phase 2 V2 checkpoint only. The Eukarya and RBPZoo panels are part of v3a training.
- Open items: `NEXT_STEPS.md`, section Open checks.

---

## 4. Lessons from the experiments

- Frozen ESM-2 mean-pool variants (V3, V3b) were below V2 on `generalized_v2`, and a frozen residue Conv1D (V3c) was 0.005 AUROC below V2. Frozen ESM-2 and RNA-FM heads on v3b gave 0.531-0.559 probe AUROC against 0.627 for the nearest-protein baseline.
- ZHMolGraph AUROC 0.798 is measured on NPInter5 with unseen proteins and RNAs. The in vitro protein-disjoint tests are different tasks and the numbers are not compared directly.
- External validation AUROC 0.763 is read together with the caveats of section 5 (159 pairs, 72% positives, window-max scoring).
- Hyperparameters are not tuned on the test set. Differences below about 0.02 AUROC lie within the seed spread of 3-seed runs.
- eCLIP data are used for training only in `generalized_v2` because the positive definition differs from in vitro enrichment.

---

## 5. External Validation — Correct Interpretation

V2 CNN (v3a checkpoint) evaluated on literature Excel + expanded TSV from `scripts/31`.

### Curated only (`dataset_without_affinities.xlsx`, 159 pairs)

| Number | What it means |
|--------|--------------|
| AUROC = **0.763** | 114 pos / 45 neg; primary qualitative OOD check |
| AUPRC = 0.915 | Positive rate = 72%. Random baseline = **0.717**. Gain = +0.20 |
| 88% single-class proteins | Per-protein AUROC mostly undefined on curated set alone |
| 57% RNAs > 60 nt | Window-max scoring inflates scores for long RNAs |

### Expanded (`external_benchmark_expanded.tsv`, 540 pairs)

| Number | What it means |
|--------|--------------|
| AUROC = 0.688 | 114 pos + 426 neg (381 generated + 45 curated neg) |
| AUPRC = 0.488 | Positive rate = **21%**. Random baseline = **0.211**. Gain = **+0.28** |
| 0.488 and 0.915 are not comparable | Different class balance; the gain over random is compared or use stratified `curated_pos_vs_generated_neg` |

**Valid use**: sanity check on literature pairs; diagnostic decoy rejection (shuffle / cross-pair).  
**Invalid use**: numeric comparison to SELEX test AUROC (0.813) or ZHMolGraph without caveats.

---

## 6. Status of the evidence for a publication

Measured so far:

1. Splits: protein-disjoint (v3a), plus a cluster split with 3% of test proteins at >= 50% identity to training (v3b). Homology is quantified (`scripts/55`).
2. Negatives: SELEX and RBNS negatives are non-enriched or shuffled sequences, and the cross-protein probe removes RNA composition as a signal.
3. Seeds: 3 seeds for V4 on v3a, the probe and the frozen heads. V2 and V4 on v3b rest on seed 42.
4. External benchmark: 159 curated pairs, small and diagnostic.
5. RNAcompete zero-shot (Phase 2 V2 checkpoint): per-protein median 0.549 on 715 proteins not in training.

Candidate framings in `NEXT_STEPS.md`: a benchmark and audit contribution (homology-stratified evaluation, RNA-only and nearest-protein baselines, cross-protein negatives), and a method contribution if a model exceeds the nearest-protein baseline in the identity bins below 50% on v3b.

---

## 7. RNAcompete Zero-Shot Results — Diagnosis (2026-05-13)

### 7.1 Results

| Metric | Value |
|---|---|
| Overall AUROC | 0.571 |
| Per-protein median, all (742) | 0.551 |
| Per-protein median, truly unseen (715) | **0.549** |
| Per-protein median, seen in training (27) | 0.629 |
| Homo sapiens ucRBP (613 proteins, 1.45M pairs) | 0.554 |

**Verdict**: zero-shot generalization has essentially failed. The model performs at near-random
on unseen proteins across assay and organism boundaries.

### 7.2 Root Cause Analysis

This is **not primarily a bug problem**. The double class-weighting bug affects calibration
and absolute AUROC, but not the fundamental capacity to generalize. The root causes are:

1. **Training set too small**: 169 unique proteins cannot cover the motif space of all RBPs.
   The 0.08 gap between seen/unseen proteins shows minimal memorization — the model is not
   overfitting to protein identity, it just lacks generalizable signal.

2. **Assay and RNA length distribution mismatch**: V2 was trained on HTR-SELEX sequences
   (30–60 nt enriched) and evaluated on RNAcompete probes (35–41 nt synthetic covering all
   7-mers). These are different distributions; the CNN's learned motif filters may be tuned
   to the specific frequency spectrum of SELEX-enriched sequences.

3. **No organism-invariant features**: organisms like Leishmania (0.534) and C. elegans (0.597)
   have distinct codon usage and RNA composition. One-hot CNN has no mechanism to abstract
   over these differences.

4. **Architecture ceiling**: one-hot CNN with global max pooling is the right inductive bias
   for motif detection but has no interaction layer — it scores protein and RNA independently
   and combines them only in the MLP head. This limits the model to protein-specific and
   RNA-specific features rather than joint interaction features.

### 7.3 What Would Improve Zero-Shot Performance

In order of expected impact:

1. **More diverse training proteins** — adding RNAcompete to training (with homology-aware
   split to exclude proteins appearing in the test set) would provide 1,087 proteins spanning
   26 organisms. This is the highest-leverage intervention.

2. **Interaction layer** — bilinear product or cross-attention between RNA and protein
   representations forces the model to learn joint features that may generalize better
   than independent branch representations combined only at the head.

3. **Fine-tuned protein encoder** — LoRA fine-tuned ESM-2 on the training proteins may
   learn binding-domain-specific representations that transfer to unseen proteins in the
   same family. Frozen ESM-2 failed; fine-tuned may not.

4. **RNA structural features** — RNAcompete probes are short enough for RNAfold;
   adding MFE/accessibility features may help the model generalize beyond
   sequence-level motifs.

---

## 8. Steps listed in 2026-05 to 2026-09 and their outcome

| Step | Outcome |
|---|---|
| Fix benchmark organism names (`scripts/archive/17_prepare_rnacompete_benchmark.py`) | Done |
| Retrain V2 after the weighting fix | Done, test AUROC 0.690 |
| RNAcompete zero-shot of V2 (`scripts/archive/20_evaluate_benchmark.py`) | Done, per-protein median 0.549 (715 unseen proteins) |
| Homology audit | Done on v3a (`scripts/55`) and applied in v3b (`scripts/58`) |
| Interaction layer V4 | Done: 0.829 +- 0.009 on v3a (3 seeds), probe on v3a 0.754 +- 0.001 |
| RNAcompete in training with homology-aware split | Done as `generalized_v3a` (name split, then audited) |
| Multi-seed V2 on v3a | Not run (V4: 3 seeds) |
| iCLIP acquisition (TDP-43, FUS), affinity regression head | Not started. Affinity-style test on RNAcompete intensities was run (`scripts/72`) |

Later phases (affinity, localization, variants, in vivo bridge) are in `docs/archive/RESEARCH_ROADMAP.md`. The current plan is `NEXT_STEPS.md`.
