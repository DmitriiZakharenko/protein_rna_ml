# Domain-aware phase — after cross-protocol (Week 1 complete)

**Status**: **Complete** (2026-09-04, VM `pleasedimpala-8f147`)  
**Headline experiment**: clean V2 baseline vs **domain-class conditioning** — **null result**  
**Deferred**: Table S1 construct-replace ablation (on v3a most matches are already
RNAcompete constructs → full vs construct is nearly a no-op)

**Sources**: `DATA_SOURCES_AND_DOWNLOADS.md`, Table S1, UniProt bulk TSV, sanitized v3a

---

## Goal

Use **domain architecture labels** (RRM / KH / multi / …) as an inductive bias
in the binding model, with a matched negative control (shuffled labels).

## Experiments

| Arm | Script flag | What changes |
|-----|-------------|--------------|
| **A — baseline** | `--mode baseline` | RNABindingCNN, protein seq as in data |
| **B — domain conditioned** | `--mode domain_conditioned` | + learnable `domain_class` embedding in head |
| **C — shuffle control** | `--mode domain_shuffle` | same as B, labels permuted across variants |

All arms: `data/sanitized/generalized_v3a`, **known domain labels only**
(`--known_domains_only`, default on): drop variants with `domain_class=unknown`
so the unknown embedding cannot absorb signal. Same row filter for baseline,
conditioned, and shuffle.

Approx. size after filter: train ~1.58M / val ~0.22M / test ~0.23M rows
(~285 / 36 / 37 proteins). Labels are then all real classes (mostly RRM/KH/multi).

## Commands

```bash
python scripts/38_train_domain_conditioned_v2.py --qc_only --refresh_qc

# A–C on the SAME known-domain cohort (default)
python scripts/38_train_domain_conditioned_v2.py \
  --mode baseline --prot_max 700 --seed 42

python scripts/38_train_domain_conditioned_v2.py \
  --mode domain_conditioned --prot_max 700 --seed 42

python scripts/38_train_domain_conditioned_v2.py \
  --mode domain_shuffle --prot_max 700 --seed 42
```

To train on full v3a including unknown (not recommended for the domain claim):

```bash
python scripts/38_train_domain_conditioned_v2.py --mode baseline --include_unknown
```

Outputs: `results/domain_aware/v2_domain_cond/{mode}/v2_domain_results.json`  
Checkpoints: `models/saved/domain_v2_{mode}/best_model.pt`

## Success criteria

1. Baseline on sanitized v3a is in the ballpark of Phase 3A (~test AUROC 0.81).
2. Domain-conditioned ≥ baseline on global and/or per-protein median AUROC.
3. Shuffle control ≤ domain-conditioned (otherwise capacity/leak, not domains).
4. Report per-`domain_class` median AUROC.

## Results (seed 42, known-domains-only cohort)

| Mode | Test AUROC | Test AUPRC | pp-median | best_epoch |
|------|------------|------------|-----------|------------|
| baseline | 0.8470 | 0.7473 | 0.8594 | 17 |
| domain_conditioned | 0.8418 | 0.7447 | 0.8468 | 17 |
| domain_shuffle | **0.8549** | **0.7528** | **0.8851** | 13 |

Per-protein median AUROC by `domain_class`:

| domain_class | baseline | domain_conditioned | domain_shuffle |
|--------------|----------|--------------------|----------------|
| CCCH | 0.704 | 0.680 | **0.768** |
| KH | 0.887 | **0.898** | 0.883 |
| RRM | 0.884 | 0.865 | **0.898** |
| multi | 0.737 | 0.683 | **0.804** |

**Verdict**: criteria 1 ✅, 2 ❌, 3 ❌. Coarse Table S1 `domain_class` does **not** improve
V2 on protein-disjoint v3a test. Shuffle ≥ baseline suggests extra head capacity, not biological
domain signal. Largest conditioned losses vs baseline: CCCH (−0.024) and multi (−0.054).

JSON: `results/domain_aware/v2_domain_cond/{mode}/v2_domain_results.json` (key `test_metrics`, not `test`).

## Later (not this run)

- Attribution mass inside UniProt/S1 intervals (`scripts/39`)
- UniProt interval mask on full-length SELEX where seq == UniProt
- Skipper eCLIP reprocessing
