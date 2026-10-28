#!/usr/bin/env bash
# Unattended multi-seed pipeline (run with nohup; safe to disconnect).
#
#   1. build the in vitro cross-protein probe once (script 53)
#   2. probe every existing checkpoint (V2 seed 42, V4 seeds 42/0/1)   -> guaranteed early result
#   3. train V2 on v3a with seeds 0 and 1 (skipped if a checkpoint already exists)
#   4. probe all six checkpoints, aggregate mean +- std (script 54)
#
# Each stage is independent: a failed stage is recorded and the script moves on.
# Progress: cat results/overnight/status.txt      Logs: results/overnight/<stage>.log
# Overrides: MAX_POS, EPOCHS, V2_SEEDS, PYTHON
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
EPOCHS="${EPOCHS:-60}"
V2_SEEDS="${V2_SEEDS:-0 1}"
TRAIN_D=data/generalized_v3a; [ -f "$TRAIN_D/train.tsv" ] || TRAIN_D=data/sanitized/generalized_v3a
PROBE_D=data/sanitized/generalized_v3a; [ -f "$PROBE_D/test.tsv" ] || PROBE_D=data/generalized_v3a
OUT=results/overnight
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

probe() {  # probe <out_dir>
  "$PYTHON" scripts/51_eclip_eval_suite.py \
    --bench_dir data/benchmarks/v3a_cross_neg --splits cross_neg_probe \
    --max_train_rows 200000 --n_boot 100 --out_dir "$1" \
    --model "v2_s42,type=v2,path=models/saved/generalized_v2/best_model.pt,rna_max=60,prot_max=700" \
    --model "v2_s0,type=v2,path=models/saved/v2_v3a_seed0/best_model.pt,rna_max=60,prot_max=700" \
    --model "v2_s1,type=v2,path=models/saved/v2_v3a_seed1/best_model.pt,rna_max=60,prot_max=700" \
    --model "v4_s42,type=v4,path=results/multiseed/v4_concat_bi_v3a/seed_42/checkpoints/best_model.pt,rna_max=60,prot_max=700" \
    --model "v4_s0,type=v4,path=results/multiseed/v4_concat_bi_v3a/seed_0/checkpoints/best_model.pt,rna_max=60,prot_max=700" \
    --model "v4_s1,type=v4,path=results/multiseed/v4_concat_bi_v3a/seed_1/checkpoints/best_model.pt,rna_max=60,prot_max=700"
}

train_v2() {  # train_v2 <seed>
  "$PYTHON" scripts/06_train_generalized_v2.py --data_dir "$TRAIN_D" --prot_max 700 \
    --epochs "$EPOCHS" --seed "$1" \
    --model_dir "models/saved/v2_v3a_seed$1" --out_dir "$OUT/v2_seed$1"
}

echo "$(date -Is) PIPELINE START (train data: $TRAIN_D, probe data: $PROBE_D)" >> "$STATUS"

stage build_probe "$PYTHON" scripts/53_build_invitro_cross_protein_probe.py \
  --v3a_dir "$PROBE_D" --out_dir data/benchmarks/v3a_cross_neg --max_pos "${MAX_POS:-40000}"
stage probe_existing probe "$OUT/probe_existing"

for s in $V2_SEEDS; do
  if [ -f "models/saved/v2_v3a_seed$s/best_model.pt" ]; then
    echo "$(date -Is) SKIP   train_v2_seed$s (checkpoint exists)" >> "$STATUS"
  else
    stage "train_v2_seed$s" train_v2 "$s"
  fi
done

stage probe_final probe "$OUT/probe_final"
stage aggregate "$PYTHON" scripts/54_aggregate_probe.py \
  --summary "$OUT/probe_final/step1_summary.json" --out_json "$OUT/aggregate.json"
echo "$(date -Is) PIPELINE DONE" >> "$STATUS"
