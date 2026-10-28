#!/usr/bin/env bash
# Bridge test: fine-tune the in vitro V2 (v3a) on eCLIP and evaluate the
# protein-specific signal with the shuffle control.
#   FT_A  cross_protein_neg  (protein-disjoint, GC-matched cross-protein negatives)
#   FT_B  protein_and_rna    (protein-disjoint, genomic negatives)
# Both at lr 1e-4 starting from V2_V3A_CKPT; compare with from-scratch runs.
set -euxo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
CKPT="${V2_V3A_CKPT:-models/saved/generalized_v2/best_model.pt}"
B=data/benchmarks/skipper_eclip
R=results/skipper_eclip

for split in cross_protein_neg protein_and_rna; do
  "$PYTHON" scripts/06_train_generalized_v2.py --data_dir $B/$split \
    --rna_max 151 --prot_max 700 --epochs 30 --batch_size 256 --lr 1e-4 --patience 10 \
    --init_checkpoint "$CKPT" \
    --model_dir models/saved/skipper_eclip_v2_ft_$split --out_dir $R/v2_ft_$split
done

"$PYTHON" scripts/51_eclip_eval_suite.py \
  --splits cross_protein_neg protein_and_rna --v3a_train_tsv data/sanitized/generalized_v3a/train.tsv \
  --model "v2_ft_cpn,type=v2,path=models/saved/skipper_eclip_v2_ft_cross_protein_neg/best_model.pt,rna_max=151,prot_max=700,splits=cross_protein_neg" \
  --model "v2_ft_pr,type=v2,path=models/saved/skipper_eclip_v2_ft_protein_and_rna/best_model.pt,rna_max=151,prot_max=700,splits=protein_and_rna" \
  --out_dir $R/step1_finetune
