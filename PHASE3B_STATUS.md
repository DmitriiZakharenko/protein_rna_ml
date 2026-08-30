# Phase 3B — status and VM commands

**Last updated**: 2026-08-30  
**Full metrics**: `results/phase3b_summary.json`  
**Detailed runbook**: local only (`PHASE3B_RUNBOOK.md`, gitignored)

---

## Completed

| Step | Result |
|------|--------|
| V4 train on v3a | test AUROC **0.829**, AUPRC **0.735** |
| External V2 curated / expanded | AUROC 0.763 / 0.688 |
| External V4 curated / expanded | AUROC 0.737 / 0.666 |

**Headline**: V4 wins on v3a test (+1.6 pp AUROC); **V2 still better on literature OOD**.

---

## Next on VM (in order)

### 0. Pull

```bash
cd /vol/space/protein_rna_ml
git pull
source /vol/space/miniconda3/bin/activate prna
```

### 1. Multi-seed V4 (~1–2 days, tmux)

```bash
tmux new -s v4_multiseed
```

```bash
cd /vol/space/protein_rna_ml
source /vol/space/miniconda3/bin/activate prna

python scripts/18_run_multiseed.py \
  --script scripts/21_train_generalized_v4_interaction.py \
  --seeds 42 0 1 \
  --output_dir results/multiseed/v4_concat_bi_v3a \
  --extra_args "--data_dir data/generalized_v3a --interaction concat_bi --use_source_emb --prot_max 700 --batch_size 512 --epochs 60"
```

Check: `cat results/multiseed/v4_concat_bi_v3a/summary.json`

### 2. Cross-protocol (scripts 33–36)

```bash
python scripts/33_build_cross_protocol_roster.py

python scripts/34_cross_protocol_classifiers.py \
  --config configs/cross_protocol.yaml \
  --model logistic_regression

python scripts/35_cross_protocol_motif_concordance.py
python scripts/36_visualize_cross_protocol.py
```

If `34` fails — check sibling paths in `configs/cross_protocol.yaml`; `rbns_analysis` is under `/vol/space/`.

### 3. Domain-aware (after cross-protocol)

```bash
python scripts/38_train_domain_conditioned_v2.py --qc_only --refresh_qc

for mode in baseline domain_conditioned domain_shuffle; do
  python scripts/38_train_domain_conditioned_v2.py \
    --mode "$mode" --prot_max 700 --seed 42 --epochs 60 --batch_size 512
done
```

### 4. Optional — V4 without source_emb (OOD ablation)

```bash
tmux new -s v4_no_src

python scripts/21_train_generalized_v4_interaction.py \
  --data_dir data/generalized_v3a \
  --interaction concat_bi \
  --prot_max 700 \
  --batch_size 512 \
  --epochs 60 \
  --model_dir models/saved/generalized_v4_phase3a_nosrc \
  --out_dir results/generalized/v4_phase3a_nosrc \
  2>&1 | tee results/generalized/v4_phase3a_nosrc/train.log

python scripts/21b_evaluate_external_v4.py \
  --checkpoint models/saved/generalized_v4_phase3a_nosrc/best_model.pt \
  --benchmark_tsv data/external/external_benchmark_curated.tsv \
  --prot_max 700 \
  --out_dir results/external/v4_nosrc_curated
```

---

## External eval notes

- Curated subset without xlsx: `data/external/external_benchmark_curated.tsv` (filter from expanded TSV).
- Do **not** report RNAcompete `rnacompete_all` as zero-shot for v3a/v4 models.
