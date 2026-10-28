#!/usr/bin/env bash
# Step 1: rebuild eCLIP benchmarks with real protein sequences, train in-domain
# V2 models, and run the honest evaluation suite (scripts/51).
#
# Stages (run all, or pick with STAGES="build baselines train eval"):
#   build      41 -> 41b -> 41c(protein_and_rna) -> 42 (cross_protein_neg splits)
#   baselines  GC / k-mer LR only (CPU)
#   train      V2 from scratch on cross_protein_neg and protein_and_rna (151 nt)
#   eval       zero-shot v3a V2/V4 + in-domain V2, with protein-shuffle control
#
# Overrides:
#   PYTHON (interpreter with torch/sklearn; default: python, else python3)
#   TRAIN_SPLITS (default: "protein_and_rna cross_protein_neg")
#   V3A_TRAIN_TSV, V2_V3A_CKPT, V4_V3A_CKPT, EPOCHS, BATCH
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
STAGES="${STAGES:-build baselines train eval}"
V3A_TRAIN_TSV="${V3A_TRAIN_TSV:-data/sanitized/generalized_v3a/train.tsv}"
[ -f "$V3A_TRAIN_TSV" ] || V3A_TRAIN_TSV="data/generalized_v3a/train.tsv"
V2_V3A_CKPT="${V2_V3A_CKPT:-models/saved/generalized_v2/best_model.pt}"
V4_V3A_CKPT="${V4_V3A_CKPT:-models/saved/generalized_v4_phase3a/best_model.pt}"
TRAIN_SPLITS="${TRAIN_SPLITS:-protein_and_rna cross_protein_neg}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-256}"
BENCH=data/benchmarks/skipper_eclip
OUT=results/skipper_eclip

has() { [[ " $STAGES " == *" $1 "* ]]; }

if has build; then
  echo "### build"
  "$PYTHON" scripts/41_build_skipper_eclip_benchmark.py --train_tsv "$V3A_TRAIN_TSV"
  "$PYTHON" scripts/41b_split_skipper_eclip_protein_disjoint.py --rebuild
  "$PYTHON" scripts/41c_split_skipper_eclip_rna_disjoint.py --mode protein_and_rna --rebuild
  "$PYTHON" scripts/42_build_skipper_eclip_cross_protein_neg.py --write_splits
fi

if has baselines; then
  echo "### baselines"
  "$PYTHON" scripts/51_eclip_eval_suite.py \
    --splits cross_protein_neg protein_and_rna protein_disjoint \
    --out_dir "$OUT/step1_baselines" --no_cuda
fi

if has train; then
  for split in $TRAIN_SPLITS; do
    echo "### train V2 on $split"
    "$PYTHON" scripts/06_train_generalized_v2.py \
      --data_dir "$BENCH/$split" --rna_max 151 --prot_max 700 \
      --epochs "$EPOCHS" --batch_size "$BATCH" \
      --model_dir "models/saved/skipper_eclip_v2_$split" \
      --out_dir "$OUT/v2_train_$split"
  done
fi

if has eval; then
  echo "### eval suite"
  "$PYTHON" scripts/51_eclip_eval_suite.py \
    --splits cross_protein_neg protein_and_rna \
    --v3a_train_tsv "$V3A_TRAIN_TSV" \
    --model "v2_v3a,type=v2,path=$V2_V3A_CKPT,rna_max=60,prot_max=700" \
    --model "v4_v3a,type=v4,path=$V4_V3A_CKPT,rna_max=60,prot_max=700" \
    --model "v2_eclip,type=v2,path=models/saved/skipper_eclip_v2_cross_protein_neg/best_model.pt,rna_max=151,prot_max=700,splits=cross_protein_neg" \
    --model "v2_eclip_pr,type=v2,path=models/saved/skipper_eclip_v2_protein_and_rna/best_model.pt,rna_max=151,prot_max=700,splits=protein_and_rna" \
    --out_dir "$OUT/step1"
fi
echo "done -> $OUT/step1*/step1_summary.json"
