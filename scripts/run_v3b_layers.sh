#!/bin/bash
# Frozen bilinear heads (script 67) on v3b with the protein embedding taken from other ESM-2 layers than 33 (script 66 --prot_layer).
# RNA embeddings are shared with data/embeddings_v3b. Skips finished runs.
# Usage: nohup bash scripts/run_v3b_layers.sh > results/v3b_layers_runner.log 2>&1 &
#   LAYERS="0 22 26 32"  SEEDS="42 0 1"  CONFIGS="bilinear:std bilinear:xneg"  PYTHON=...
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
LAYERS=${LAYERS:-"0 22 26 32"}
SEEDS=${SEEDS:-"42 0 1"}
CONFIGS=${CONFIGS:-"bilinear:std bilinear:xneg"}
for L in $LAYERS; do
  e=data/embeddings_v3b_L$L
  if [ ! -f $e/prot_esm2.npz ]; then
    mkdir -p $e && cp data/embeddings_v3b/rna_rnafm.npz $e/
    $PYTHON -u scripts/66_embed_v3b_frozen.py --stage proteins --prot_layer $L --out_dir $e > results/v3b_layers_embed_L$L.log 2>&1
  fi
  for s in $SEEDS; do
    for c in $CONFIGS; do
      h=${c%%:*}; m=${c##*:}
      d=results/v3b_frozen_layers/L${L}_${h}_${m}_s${s}
      [ -f $d/scores_probe.tsv.gz ] && continue
      echo "=== layer $L $h $m seed $s $(date +%H:%M)"
      $PYTHON -u scripts/67_frozen_embedding_baselines.py --head $h --train_mode $m --seed $s --emb_dir $e --out_dir $d > results/v3b_layers_L${L}_${h}_${m}_s${s}.log 2>&1
      tail -n 3 results/v3b_layers_L${L}_${h}_${m}_s${s}.log
    done
  done
done
echo done
