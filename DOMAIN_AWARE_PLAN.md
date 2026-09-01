# Domain-aware phase — after cross-protocol (Week 1 complete)

**Status**: **Next** (run `scripts/38` on VM)  
**Headline experiment**: clean V2 baseline vs **domain-class conditioning**  
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

## Later (not this run)

- Attribution mass inside UniProt/S1 intervals (`scripts/39`)
- UniProt interval mask on full-length SELEX where seq == UniProt
- Skipper eCLIP reprocessing
