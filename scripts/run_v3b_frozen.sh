#!/bin/bash
# Frozen-embedding heads (script 67) on v3b with per-row scores saved; skips runs that already have scores.
# Usage: nohup bash scripts/run_v3b_frozen.sh > results/v3b_frozen_runner.log 2>&1 &
#   SEEDS="42 0 1"  CONFIGS="rna:std protein:std concat:std bilinear:std bilinear:xneg"  PYTHON=...
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
SEEDS=${SEEDS:-"42 0 1"}
CONFIGS=${CONFIGS:-"bilinear:xneg bilinear:std rna:std protein:std concat:std"}
for s in $SEEDS; do
  for c in $CONFIGS; do
    h=${c%%:*}; m=${c##*:}
    d=results/v3b_frozen/${h}_${m}_s${s}
    [ -f $d/scores_probe.tsv.gz ] && continue
    echo "=== $h $m seed $s $(date +%H:%M)"
    $PYTHON -u scripts/67_frozen_embedding_baselines.py --head $h --train_mode $m --seed $s --out_dir $d > results/v3b_frozen_${h}_${m}_s${s}.log 2>&1
    tail -n 5 results/v3b_frozen_${h}_${m}_s${s}.log
  done
done
echo done
