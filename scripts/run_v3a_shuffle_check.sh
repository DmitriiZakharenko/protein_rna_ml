#!/usr/bin/env bash
# Protein-shuffle control on the in vitro v3a test set: do V2/V4 use the protein
# when scored in-distribution?
set -euxo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
D=data/sanitized/generalized_v3a; [ -f "$D/test.tsv" ] || D=data/generalized_v3a
mkdir -p data/benchmarks/v3a_check
ln -sfn "$(realpath "$D")" data/benchmarks/v3a_check/v3a_test

"$PYTHON" scripts/51_eclip_eval_suite.py \
  --bench_dir data/benchmarks/v3a_check --splits v3a_test \
  --max_train_rows 200000 --max_test_rows "${MAX_TEST_ROWS:-60000}" --n_boot 100 \
  --model "v2_v3a,type=v2,path=${V2_V3A_CKPT:-models/saved/generalized_v2/best_model.pt},rna_max=60,prot_max=700" \
  --model "v4_v3a,type=v4,path=${V4_V3A_CKPT:-models/saved/generalized_v4_phase3a/best_model.pt},rna_max=60,prot_max=700" \
  --out_dir results/skipper_eclip/step1_v3a_shuffle
