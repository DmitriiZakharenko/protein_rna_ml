# Phase 3B — status and VM commands

**Last updated**: 2026-09-01  
**Full metrics**: `results/phase3b_summary.json`

---

## Completed

| Step | Result |
|------|--------|
| V4 train (single) | test AUROC 0.829 |
| V4 multi-seed (42, 0, 1) | test AUROC **0.829 ± 0.009**, AUPRC **0.732 ± 0.011** |
| External V4 curated | AUROC 0.737 (V2 **0.763**) |
| **Cross-protocol in-vitro** (scripts 33–36) | within mean AUROC **0.974**; transfer **0.791** (284 rows) |

**Headline**: V4 beats V2 on v3a test; V2 better OOD; cross-assay k-mer LR transfers with ~0.18 AUROC drop vs within-protocol.

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

## Next on VM — domain-conditioned V2 (Step 6)

See `DOMAIN_AWARE_PLAN.md` / `scripts/38_train_domain_conditioned_v2.py`:

```bash
cd /vol/space/protein_rna_ml
source /vol/space/miniconda3/bin/activate prna

# Example — adjust config paths as needed
python scripts/38_train_domain_conditioned_v2.py --mode baseline
python scripts/38_train_domain_conditioned_v2.py --mode domain_conditioned
python scripts/38_train_domain_conditioned_v2.py --mode domain_shuffle
```
