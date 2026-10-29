#!/usr/bin/env bash
# Motif recovery + in silico mutagenesis for every v3b checkpoint that exists (run after run_v3b_overnight.sh).
# Needs: results/v3b/protein_split_map.tsv, results/homology_audit/v3b_test_protein_identity.tsv,
#        results/mechanistic/v3b_domain_positions.tsv (compute once locally with hmmscan and commit), mmseqs only for the nn baseline.
# Usage: bash scripts/run_v3b_mechanistic.sh   (SEEDS="42 0 1", PYTHON=/path/to/python)
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/vol/space/miniconda3/envs/prna/bin/python}"
SEEDS="${SEEDS:-42 0 1}"
args=()
for s in $SEEDS; do
  [ -f "models/saved/v2_v3b_seed$s/best_model.pt" ] && args+=(--model "v2_s$s,type=v2,path=models/saved/v2_v3b_seed$s/best_model.pt,rna_max=60,prot_max=700")
  [ -f "models/saved/v4_v3b_seed$s/best_model.pt" ] && args+=(--model "v4_s$s,type=v4,path=models/saved/v4_v3b_seed$s/best_model.pt,rna_max=60,prot_max=700")
done
NN=()
command -v mmseqs >/dev/null 2>&1 && NN=(--train_tsv data/generalized_v3b/train.tsv)
"$PYTHON" scripts/61_motif_recovery_and_mutagenesis.py --name v3b "${args[@]}" \
  --test_tsv data/generalized_v3b/test.tsv "${NN[@]}" \
  --identity_tsv results/homology_audit/v3b_test_protein_identity.tsv \
  --domain_positions results/mechanistic/v3b_domain_positions.tsv
