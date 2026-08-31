# Phase 3B — status and VM commands

**Last updated**: 2026-08-31  
**Full metrics**: `results/phase3b_summary.json`

---

## Completed

| Step | Result |
|------|--------|
| V4 train (single) | test AUROC 0.829 |
| V4 multi-seed (42, 0, 1) | test AUROC **0.829 ± 0.009**, AUPRC **0.732 ± 0.011** |
| External V4 curated | AUROC 0.737 (V2 **0.763**) |

**Headline**: V4 stably beats V2 on v3a test; **V2 better on literature OOD**.

---

## Next on VM — cross-protocol (Step 5)

```bash
cd /vol/space/protein_rna_ml
git pull
source /vol/space/miniconda3/bin/activate prna

python scripts/33_build_cross_protocol_roster.py

python scripts/34_cross_protocol_classifiers.py \
  --config configs/cross_protocol.yaml \
  --model logistic_regression

python scripts/35_cross_protocol_motif_concordance.py
python scripts/36_visualize_cross_protocol.py
```

If `34` fails on missing paths:

```bash
ls /vol/space/rbns_analysis/results/ml_dataset_rbns_clean.tsv
# Edit configs/cross_protocol.yaml or clone htr_selex_analysis / rnacompete_analysis under /vol/space/
```

Then domain models (`scripts/38`) — see `DOMAIN_AWARE_PLAN.md`.
