#!/usr/bin/env bash
# Diagnostics for the cross_protein_neg failure of V2 (loss stuck at ln 2):
#   A) same proteins in train and test (RNA-disjoint split) -> can the model fit at all?
#   B) V2 with a lower learning rate on the protein-disjoint split
#   C) V4 (bilinear interaction) on the protein-disjoint split
set -euxo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
B=data/benchmarks/skipper_eclip
R=results/skipper_eclip

"$PYTHON" scripts/41c_split_skipper_eclip_rna_disjoint.py --mode rna \
  --pairs_tsv $B/fixlen_151_cross_protein_neg_all.tsv \
  --out_dir $B/cross_protein_neg_seen_protein \
  --summary_json $R/cross_protein_neg_seen_protein_split_summary.json
"$PYTHON" scripts/06_train_generalized_v2.py --data_dir $B/cross_protein_neg_seen_protein \
  --rna_max 151 --prot_max 700 --epochs 40 --batch_size 256 \
  --model_dir models/saved/skipper_eclip_v2_cpn_seen --out_dir $R/v2_train_cpn_seen

"$PYTHON" scripts/06_train_generalized_v2.py --data_dir $B/cross_protein_neg \
  --rna_max 151 --prot_max 700 --epochs 40 --batch_size 256 --lr 1e-4 --patience 12 \
  --model_dir models/saved/skipper_eclip_v2_cpn_lr1e4 --out_dir $R/v2_train_cpn_lr1e4

"$PYTHON" scripts/21_train_generalized_v4_interaction.py --data_dir $B/cross_protein_neg \
  --interaction concat_bi --rna_max 151 --prot_max 700 --epochs 40 --batch_size 256 \
  --lr 1e-4 --patience 12 \
  --model_dir models/saved/skipper_eclip_v4_cpn --out_dir $R/v4_train_cpn
