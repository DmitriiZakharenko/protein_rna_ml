#!/bin/bash
# Frozen heads (script 67) on v3b with TF ChIP-seq rows added to training (script 76 table, embeddings from script 66 --tsv).
# Usage: nohup bash scripts/run_v3b_tf.sh > results/v3b_tf_runner.log 2>&1 &
#   SEEDS="42 0 1"  CONFIGS="bilinear:std bilinear:xneg"  PYTHON=...  EXTRA_FRAC=1.0
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
SEEDS=${SEEDS:-"42 0 1"}
CONFIGS=${CONFIGS:-"bilinear:std bilinear:xneg"}
EXTRA_FRAC=${EXTRA_FRAC:-1.0}
for s in $SEEDS; do
  for c in $CONFIGS; do
    h=${c%%:*}; m=${c##*:}
    d=results/v3b_frozen_tf/${h}_${m}_s${s}
    [ -f $d/scores_probe.tsv.gz ] && continue
    echo "=== $h $m +TF seed $s $(date +%H:%M)"
    $PYTHON -u scripts/67_frozen_embedding_baselines.py --head $h --train_mode $m --seed $s --out_dir $d \
      --extra_tsv data/benchmarks/tf_chipseq/pairs.tsv --extra_emb_dir data/embeddings_tf --extra_frac $EXTRA_FRAC \
      > results/v3b_frozen_tf_${h}_${m}_s${s}.log 2>&1
    tail -n 5 results/v3b_frozen_tf_${h}_${m}_s${s}.log
  done
done
echo done
