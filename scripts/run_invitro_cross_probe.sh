#!/usr/bin/env bash
# In vitro cross-protein-negative probe: do V2/V4 learn the protein x RNA
# interaction, or only "protein -> RNA type"? (scripts 53 + 51)
set -euxo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
D=data/sanitized/generalized_v3a; [ -f "$D/test.tsv" ] || D=data/generalized_v3a

"$PYTHON" scripts/53_build_invitro_cross_protein_probe.py --v3a_dir "$D" \
  --out_dir data/benchmarks/v3a_cross_neg --max_pos "${MAX_POS:-40000}"

"$PYTHON" scripts/51_eclip_eval_suite.py \
  --bench_dir data/benchmarks/v3a_cross_neg --splits cross_neg_probe \
  --max_train_rows 200000 --n_boot 100 \
  --model "v2_v3a,type=v2,path=${V2_V3A_CKPT:-models/saved/generalized_v2/best_model.pt},rna_max=60,prot_max=700" \
  --model "v4_v3a,type=v4,path=${V4_V3A_CKPT:-models/saved/generalized_v4_phase3a/best_model.pt},rna_max=60,prot_max=700" \
  --out_dir results/skipper_eclip/step1_invitro_cross_probe
