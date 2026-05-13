# Research Strategy

**Project**: Protein–RNA Binding Prediction
**Last updated**: 2026-05-13
**Current best model**: V2 CNN — test AUROC=0.703 AUPRC=0.599
**Known limitation of current best**: trained with double class-weighting bug (see §3).
**Benchmark target**: ZHMolGraph — AUROC=0.798 AUPRC=0.820 (hard split, unseen proteins AND RNAs).
**Note on comparability**: our protein-aware-only test is NOT equivalent to ZHMolGraph's hard split.

---

## 1. What The Experiments Have Actually Proven

### 1.1 One-hot CNN beats frozen ESM-2 (three independent experiments)

| Model | Test AUROC | vs V2 |
|-------|-----------|-------|
| V2 CNN (one-hot) | **0.703** | — |
| V3c ESM-2 residue CNN | 0.685 | −0.018 |
| V3b CNN + ESM-2 mean-pool | 0.666 | −0.037 |
| V3 ESM-2 mean-pool only | 0.634 | −0.069 |

This is not evidence that ESM-2 is uninformative. It is evidence that:
(a) frozen mean-pooling over all residues dilutes binding-domain signal;
(b) residue-level Conv1D on frozen ESM-2 partially recovers positional selectivity
    but does not yet beat one-hot sequence convolution on 169 proteins;
(c) the gains from ESM-2 (motif-poor proteins: UNK +0.25, TAF15 +0.06) are real
    but outweighed by losses on motif-rich proteins (PUM2 −0.25, LARP7 −0.20).

**Implication**: the next gain will NOT come from yet another frozen ESM-2 pooling variant.
It requires either (a) fine-tuning, (b) explicit interaction modeling, or (c) better data.

### 1.2 Val/test gap is systematic and unexplained

All models show val AUROC ≈ test AUROC + 0.08–0.11. This is NOT model-specific noise.
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

## 2. Known Bugs in Existing Results (Fixed 2026-05-13)

### Bug 1 — Double class-weighting (CRITICAL)
**Scripts affected**: 06, 08, 09, 10
**Description**: `WeightedRandomSampler` (creates ~50/50 batches) was used simultaneously
with `BCEWithLogitsLoss(pos_weight=n_neg/n_pos ≈ 2.0)`. Combined effect: each positive
gets 2× larger gradient weight despite already appearing at 2× baseline frequency in
batches. Equivalent to pos_weight ≈ 4× in a standard unsampled setting.
**Impact**: inflates model bias toward recall; calibration is off; results are not
reproducible with a clean implementation.
**Fix**: `WeightedRandomSampler` removed from all four training scripts. `pos_weight` kept.
**Status**: Fix applied. All existing V2/V3/V3b/V3c checkpoints were trained with the bug.
Retrain V2 with fixed script before treating current numbers as authoritative.

### Bug 2 — Per-protein dataset column always "unknown" (HIGH)
**Script affected**: 06_train_generalized_v2.py
**Description**: Per-protein results in v2_cnn_results.json show `"dataset": "unknown"`
for all proteins. The column is `dataset_source` in `data/generalized/` splits
but `dataset` in `data/generalized_v2/` splits. The script read the wrong column.
**Fix**: Column normalization added at load time.
**Status**: Fix applied. Re-run script 06 to get correctly labelled breakdowns.

### Bug 3 — Broken pipeline runner (MEDIUM)
**File**: run_pipeline.sh
**Description**: `phase1_htr()` called non-existent `scripts/02b_train_htr_selex_validation.py`.
**Fix**: Corrected to `scripts/02_train_validation_model.py --config configs/htr_selex_validation.yaml`.

### Bug 4 — Missing dependencies (HIGH)
**File**: requirements.txt
**Description**: `transformers`, `sentencepiece`, `openpyxl` were not listed.
All ESM-2 scripts and the external evaluation script would fail on a fresh install.
**Fix**: Dependencies added to requirements.txt.

---

## 3. Current Decision Tree

```
WHERE WE ARE (2026-05-13):
  V2 CNN is the current best: test AUROC=0.703 (bug-affected, retrain pending)
  V3/V3b/V3c: all frozen ESM-2 variants are worse than V2

IMMEDIATE NEXT STEPS (before any new architecture):

  Step 1 — Retrain V2 with corrected class-weighting
    python scripts/06_train_generalized_v2.py --data_dir data/generalized_v2
    Expected: small change in AUROC (±0.01), measurable improvement in calibration.
    This is the new anchor point for all future comparisons.

  Step 2 — Multi-seed variance
    Run V2 with seeds {0, 1, 2, 42, 123}.
    Report mean ± std for val and test AUROC.
    Required before claiming any improvement is real.

  Step 3 — Homology audit (measurement, not resplit)
    Compute pairwise protein identity for all 169 proteins.
    Flag test proteins with >30% identity homologs in training.
    Report per-protein AUROC split by "clean" vs "homolog-contaminated".

  Step 4 — Hard negatives for SELEX/RBNS
    Generate dinucleotide-matched, length-matched decoys for each protein's pool.
    Retrain V2 on improved negatives.
    Expected: lower AUROC on SELEX/RBNS test (harder task), higher real-world relevance.

IF V2 re-training + variance is clean:
  → Proceed to interaction layer experiments (bilinear or 2-head cross-attention on V2)

IF homology audit reveals major contamination:
  → Perform family-aware resplit before any further experiments

AFTER clean V2 baseline:
  → RNA-FM branch (structure-aware RNA encoding)
  → Cross-attention interaction module
  → eCLIP training data expansion (not for val/test — domain shift documented)
```

---

## 4. What NOT To Do Next

- **Do NOT** run another frozen ESM-2 mean-pool variant. Three experiments confirm this fails.
- **Do NOT** claim V2 AUROC=0.703 is directly comparable to ZHMolGraph AUROC=0.798.
  The task definitions differ (protein-aware only vs protein+RNA held-out).
- **Do NOT** cite external validation AUROC=0.798 without the caveats in §5 below.
- **Do NOT** tune hyperparameters on the test set or add models until multi-seed
  variance on V2 is established.
- **Do NOT** add eCLIP data to val or test splits (domain shift from in vitro labels).

---

## 5. External Validation — Correct Interpretation

V2 CNN evaluated on `dataset without affinities.xlsx` (159 pairs, 96 proteins):

| Number | What it means |
|--------|--------------|
| AUROC = 0.798 | Computed over all 159 pairs. 87/96 proteins are single-class. |
| AUPRC = 0.927 | Positive rate = 72%. Random classifier baseline = 0.717. Gain over random = +0.21. |
| 57% of RNAs > 60 nt | Scored by window-max. Longer RNAs get systematically higher scores. |
| 9 proteins evaluable | Remaining 87 have only pos or only neg examples. |

**Valid use**: qualitative check that the model assigns non-zero probability to known
binding pairs from literature.
**Invalid use**: numeric comparison to SELEX/RBNS test AUROC or to published baselines.

---

## 6. Publication Readiness

Not ready for a main-track ML or bioinformatics paper. Blockers:

1. Current best model (V2) was trained with a known class-weighting bug.
2. Splits are protein-aware but NOT homology-aware. Paralog leakage is unquantified.
3. All negatives in SELEX/RBNS are artificial shuffles.
4. Val/test gap (~0.10 AUROC) is unexplained.
5. Single-seed results only — no variance estimates.
6. External benchmark is statistically too small for strong claims.

Potential path to a short paper (workshop / methods letter):
- Fix bugs, retrain V2 clean.
- Run homology audit; report results honestly.
- Improve negatives; retrain; show AUROC changes.
- This becomes a benchmark/dataset contribution, not a methods paper.

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

## 8. Prioritized Next Steps (Post-Benchmark)

### Immediate (this week — no new training required)

| Task | Action | Output |
|---|---|---|
| Fix benchmark organism names | Re-run `17_prepare_rnacompete_benchmark.py` with fix | Clean per-organism AUROC table |
| Retrain V2-CLEAN (1 run) | `python scripts/06_train_generalized_v2.py` | Bug-free anchor baseline |
| Re-run RNAcompete on V2-CLEAN | `scripts/20_evaluate_benchmark.py` | Corrected zero-shot score |
| Multi-seed V2 (5 seeds) | `scripts/18_run_multiseed.py` | AUROC mean ± std |

### Short-term (after clean baseline confirmed)

1. **Homology audit** — run MMseqs2 between training proteins and RNAcompete proteins at 30%
   identity. Identify how many of the 27 overlaps are true sequence matches vs name matches.
   Determines whether RNAcompete can be used as a training source without leakage.

2. **Interaction layer V4** — add a bilinear interaction layer between V2's 256-d RNA and
   protein branch outputs. No new encoders. Expected to improve both in-distribution test
   AUROC and zero-shot generalization. This is the highest-priority architecture change.

3. **iCLIP data acquisition** — prepare TDP-43 and FUS iCLIP datasets. These two proteins
   appear in RNAcompete training overlap (TARDBP = TDP-43, FUS = EWSR1 family) and provide
   in vivo context for the proteins the model already partially handles.

### Medium-term

4. **RNAcompete → Training (Phase 3)** — after homology audit and interaction layer V4:
   merge RNAcompete sub-datasets into training with homology-aware split. Re-run zero-shot
   benchmark on held-out organisms (non-human organisms absent from any training set).

5. **Affinity regression head** — add a parallel regression output trained on
   `probe_intensity` from RNAcompete and `R_max` from RBNS. Multi-task training on
   both classification and affinity may learn better-calibrated motif representations.

### Do Not Do

- Do not try more frozen ESM-2 variants (three experiments, conclusively worse).
- Do not report RNAcompete median AUROC 0.551 as a success metric; it is a baseline.
- Do not compare to ZHMolGraph AUROC 0.798 — different dataset, different task.
