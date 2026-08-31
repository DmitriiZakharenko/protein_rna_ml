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

## Next on VM — cross-protocol in-vitro only (Step 5)

Uses `configs/cross_protocol_invitro.yaml` (HTR-SELEX, RBNS, RNAcompete Eukarya/RBPZoo — no eCLIP).

### Data prerequisites (script 34 needs full TSVs)

```bash
cd /vol/space/protein_rna_ml
ls -lh ../rbns_analysis/results/ml_dataset_rbns_clean.tsv
ls -lh ../htr_selex_analysis/results/ml_dataset_simple_clean.tsv
ls -lh ../rnacompete_analysis/eukarya/results/ml_dataset_eukarya_clean.tsv.gz
ls -lh ../rnacompete_analysis/rbpzoo/results/ml_dataset_rbpzoo_clean.tsv.gz
```

`rbns_analysis` is on `/vol/space/`. Clone or rsync sibling repos if paths are missing.

### Run pipeline

```bash
git pull
source /vol/space/miniconda3/bin/activate prna

python scripts/33_build_cross_protocol_roster.py --config configs/cross_protocol_invitro.yaml

python scripts/34_cross_protocol_classifiers.py \
  --config configs/cross_protocol_invitro.yaml \
  --model logistic_regression

python scripts/35_cross_protocol_motif_concordance.py --config configs/cross_protocol_invitro.yaml
python scripts/36_visualize_cross_protocol.py --config configs/cross_protocol_invitro.yaml
```

Outputs: `results/cross_protocol_invitro/`
