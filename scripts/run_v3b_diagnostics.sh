#!/bin/bash
# Diagnostics for "is there really no protein signal on v3b?" (frozen ESM-2 / RNA-FM bilinear head, script 67):
#   1. positive control: seen training proteins, held-out RNAs (cross-protein probe)
#   2. learning curve over the fraction of training clusters
#   3. small hyper-parameter sweep (lr, hidden, dropout, weight decay)
# Usage: PYTHON=... nohup bash scripts/run_v3b_diagnostics.sh > results/v3b_diag_runner.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
run() {  # name, extra args...
  n=$1; shift
  d=results/v3b_diag/$n
  [ -f $d/results.json ] && return
  echo "=== $n $(date +%H:%M)"
  $PYTHON -u scripts/67_frozen_embedding_baselines.py --head bilinear --train_mode xneg --out_dir $d "$@" > results/v3b_diag_$n.log 2>&1
  grep -E "seen-protein probe|^probe:|^test_standard:" results/v3b_diag_$n.log
}
run seen_probe        --seen_probe_frac 0.05
run seen_probe_std    --seen_probe_frac 0.05 --train_mode std
for f in 0.25 0.5 0.75; do run clusters_$f --cluster_frac $f; done
for f in 0.25 0.5 0.75; do run clusters_${f}_s0 --cluster_frac $f --seed 0; done
run lr3e-4        --lr 3e-4
run lr3e-3        --lr 3e-3
run hid128        --hidden 128
run hid512        --hidden 512
run drop0.1       --dropout 0.1
run drop0.5       --dropout 0.5
run wd1e-3        --wd 1e-3
run wd1e-1        --wd 1e-1
echo done
