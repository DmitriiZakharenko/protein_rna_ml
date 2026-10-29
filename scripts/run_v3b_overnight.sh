#!/usr/bin/env bash
# Unattended pipeline on the cluster-based v3b split (run with nohup; safe to disconnect).
#
#   1. build data/generalized_v3b from results/v3b/protein_split_map.tsv (script 58 build)
#   2. build the v3b cross-protein probe (script 53)
#   3. for each seed: train V2, then V4 on v3b (skipped if a checkpoint exists)
#   4. probe every checkpoint that exists, aggregate mean +- std (script 54)
#
# Seeds run one after another (seed 42 first) so partial results are usable early.
# Progress: cat results/v3b_overnight/status.txt     Logs: results/v3b_overnight/<stage>.log
# Overrides: SEEDS (default "42 0 1"), EPOCHS, MAX_POS, PYTHON
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-}"
for cand in "$PYTHON" "$(command -v python || true)" /vol/space/miniconda3/envs/prna/bin/python; do
  if [ -n "$cand" ] && "$cand" -c "import torch, sklearn, pandas" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done
if ! "${PYTHON:-/nonexistent}" -c "import torch, sklearn, pandas" >/dev/null 2>&1; then
  echo "No Python with torch/sklearn/pandas found. Activate the environment or set PYTHON=/path/to/python." >&2
  exit 1
fi
echo "Using PYTHON=$PYTHON"
SEEDS="${SEEDS:-42 0 1}"
EPOCHS="${EPOCHS:-60}"
OUT=results/v3b_overnight
mkdir -p "$OUT"
STATUS="$OUT/status.txt"

stage() {  # stage <name> <command...>
  local name="$1"; shift
  echo "$(date -Is) START  $name" >> "$STATUS"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "$(date -Is) OK     $name" >> "$STATUS"
  else
    echo "$(date -Is) FAILED $name (see $OUT/$name.log)" >> "$STATUS"
  fi
}

train_v2() { "$PYTHON" scripts/06_train_generalized_v2.py --data_dir data/generalized_v3b --prot_max 700 \
    --epochs "$EPOCHS" --seed "$1" --model_dir "models/saved/v2_v3b_seed$1" --out_dir "$OUT/v2_seed$1"; }
train_v4() { "$PYTHON" scripts/21_train_generalized_v4_interaction.py --data_dir data/generalized_v3b \
    --interaction concat_bi --epochs "$EPOCHS" --seed "$1" \
    --model_dir "models/saved/v4_v3b_seed$1" --out_dir "$OUT/v4_seed$1"; }

probe() {  # probe <out_dir>   (only checkpoints that exist)
  local args=()
  for s in $SEEDS; do
    [ -f "models/saved/v2_v3b_seed$s/best_model.pt" ] && args+=(--model "v2_s$s,type=v2,path=models/saved/v2_v3b_seed$s/best_model.pt,rna_max=60,prot_max=700")
    [ -f "models/saved/v4_v3b_seed$s/best_model.pt" ] && args+=(--model "v4_s$s,type=v4,path=models/saved/v4_v3b_seed$s/best_model.pt,rna_max=60,prot_max=700")
  done
  "$PYTHON" scripts/51_eclip_eval_suite.py --bench_dir data/benchmarks/v3b_cross_neg --splits cross_neg_probe \
    --max_train_rows 200000 --n_boot 100 --out_dir "$1" "${args[@]}"
}

echo "$(date -Is) PIPELINE START (seeds: $SEEDS)" >> "$STATUS"
stage build_v3b "$PYTHON" scripts/58_build_v3b_cluster_split.py build
stage build_probe "$PYTHON" scripts/53_build_invitro_cross_protein_probe.py \
  --v3a_dir data/generalized_v3b --out_dir data/benchmarks/v3b_cross_neg --max_pos "${MAX_POS:-40000}"

for s in $SEEDS; do
  for m in v2 v4; do
    if [ -f "models/saved/${m}_v3b_seed$s/best_model.pt" ]; then
      echo "$(date -Is) SKIP   train_${m}_seed$s (checkpoint exists)" >> "$STATUS"
    else
      stage "train_${m}_seed$s" train_$m "$s"
    fi
  done
  stage "probe_after_seed$s" probe "$OUT/probe_after_seed$s"
done

stage probe_final probe "$OUT/probe_final"
stage aggregate "$PYTHON" scripts/54_aggregate_probe.py \
  --summary "$OUT/probe_final/step1_summary.json" --out_json "$OUT/aggregate.json"
echo "$(date -Is) PIPELINE DONE" >> "$STATUS"
