#!/usr/bin/env bash
# End-to-end cross-protocol Week-1 pipeline.
# Run from protein_rna_ml repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 33 roster ==="
python scripts/33_build_cross_protocol_roster.py

echo "=== 34 classifiers (within + transfer) ==="
# Needs sibling clean TSVs readable from this shell (Desktop paths).
python scripts/34_cross_protocol_classifiers.py \
  --config configs/cross_protocol.yaml \
  --model logistic_regression

echo "=== 35 motif concordance ==="
python scripts/35_cross_protocol_motif_concordance.py

echo "=== 36 figures ==="
python scripts/36_visualize_cross_protocol.py

echo "Done. See results/cross_protocol/ and figures/cross_protocol_*.png"
