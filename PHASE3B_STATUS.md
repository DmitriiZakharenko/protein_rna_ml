# Phase 3B — status and VM commands

**Last updated**: 2026-09-04  
**Full metrics**: `results/phase3b_summary.json`

---

## Completed

| Step | Result |
|------|--------|
| V4 train (single) | test AUROC 0.829 |
| V4 multi-seed (42, 0, 1) | test AUROC **0.829 ± 0.009**, AUPRC **0.732 ± 0.011** |
| External V4 curated | AUROC 0.737 (V2 **0.763**) |
| **Cross-protocol in-vitro** (scripts 33–36) | within mean AUROC **0.974**; transfer **0.791** (284 rows) |
| **Domain-conditioned V2** (script 38) | shuffle **0.855** > baseline **0.847** > conditioned **0.842** (null) |

**Headline**: V4 beats V2 on v3a test; V2 better OOD; cross-assay k-mer LR transfers with ~0.18 AUROC drop; coarse `domain_class` does not help V2.

### Cross-protocol outputs (VM: `pleasedimpala`)

```
results/cross_protocol_invitro/
  classifier_summary.json
  within_protocol_metrics.tsv
  transfer_metrics.tsv
  motif_concordance.tsv
  protein_roster.tsv
figures/cross_protocol_*.png
```

---

## Sync results VM → Mac (for git)

Run on **Mac**:

```bash
SSH='ssh -i /Users/zahalae/.ssh/id_ed25519 -p 30021'
HOST='ubuntu@194.94.113.18'
REPO='/Users/zahalae/Desktop/protein_rna_ml'

# All cross-protocol TSVs + JSON (~few MB)
rsync -avP -e "$SSH" \
  $HOST:/vol/space/protein_rna_ml/results/cross_protocol_invitro/ \
  $REPO/results/cross_protocol_invitro/

# Figures
rsync -avP -e "$SSH" \
  $HOST:/vol/space/protein_rna_ml/figures/cross_protocol_*.png \
  $REPO/figures/
```

Then on Mac: `git add results/cross_protocol_invitro/ figures/cross_protocol_*.png results/phase3b_summary.json EXPERIMENT_LOG.md PHASE3B_STATUS.md && git commit && git push`

---

## Phase 3B complete

Domain-conditioned V2 finished on VM. See `DOMAIN_AWARE_PLAN.md` and `results/phase3b_summary.json` → `domain_aware_v2`.

**Optional next**: V4 `--no-source-emb` OOD ablation; RPIembeddor2 eCLIP protein-disjoint eval.
