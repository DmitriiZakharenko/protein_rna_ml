#!/bin/bash
# Comparison tables with CIs for the v3b probe and the v3b standard test; picks up whatever score files exist.
# Usage: bash scripts/run_v3b_report.sh   (PYTHON=... N_BOOT=...)
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
N_BOOT=${N_BOOT:-300}

add() { [ -f "${2%%:*}" ] && SC+=(--scorer "$1=$2"); }

SC=()
add nn1        results/knn_baseline/v3b_probe_scores.tsv.gz:nn1_lr
add prior      results/knn_baseline/v3b_probe_scores.tsv.gz:mean_lr
add v2_s42     results/v3b_overnight/probe_after_seed42/scores_cross_neg_probe.tsv.gz:v2_s42__full
add v4_s42     results/v3b_overnight/probe_after_seed42/scores_cross_neg_probe.tsv.gz:v4_s42__full
for h in rna_std protein_std concat_std bilinear_std bilinear_xneg; do
  add frozen_$h results/v3b_frozen/${h}_s42/scores_probe.tsv.gz
done
add hybrid_joint results/v3b_hybrid/hybrid_s42/scores_probe.tsv.gz
add stack        results/v3b_hybrid/stack_s42/scores_probe_stack.tsv.gz
$PYTHON scripts/69_score_table_with_ci.py --name v3b_probe "${SC[@]}" --n_boot "$N_BOOT" | cut -c1-400

SC=()
add nn1        results/knn_baseline/v3b_standard_scores.tsv.gz:nn1_lr
add prior      results/knn_baseline/v3b_standard_scores.tsv.gz:mean_lr
for h in rna_std protein_std concat_std bilinear_std bilinear_xneg; do
  add frozen_$h results/v3b_frozen/${h}_s42/scores_test_standard.tsv.gz
done
add hybrid_joint results/v3b_hybrid/hybrid_s42/scores_test_standard.tsv.gz
add stack        results/v3b_hybrid/stack_s42/scores_test_standard_stack.tsv.gz
$PYTHON scripts/69_score_table_with_ci.py --name v3b_standard "${SC[@]}" --n_boot "$N_BOOT" | cut -c1-400
