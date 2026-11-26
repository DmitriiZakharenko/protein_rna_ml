#!/bin/bash
# ============================================================
# Protein-RNA Binding Prediction — Pipeline Runner
# Run from inside protein_rna_ml/:
#   cd ~/Desktop/protein_rna_ml
#   bash run_pipeline.sh [step]
#
# Steps:
#   phase1_htr   — complete HTR-SELEX PRJEB25907 Phase 1 (missing results)
#   phase2_cnn   — train generalized CNN V2 (Phase 2)
#   all          — run both in order
# ============================================================

set -e
STEP=${1:-"help"}
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

check_deps() {
    python3 -c "import sklearn, xgboost, torch" 2>/dev/null || {
        echo -e "${YELLOW}Installing missing packages...${NC}"
        pip install scikit-learn xgboost torch --break-system-packages -q
    }
}

phase1_htr() {
    echo -e "\n${GREEN}=== Phase 1: HTR-SELEX PRJEB25907 ===${NC}"
    echo "Training validation models (LR, RF, XGBoost)..."
    python3 scripts/02_train_validation_model.py --config configs/htr_selex_validation.yaml
    echo -e "${GREEN}✅ Phase 1 HTR-SELEX PRJEB25907 complete${NC}"
}

phase2_cnn() {
    echo -e "\n${GREEN}=== Phase 2 V2: Dual-branch CNN ===${NC}"
    echo "Training on 463k samples (raw sequences, length-agnostic)"
    echo "Device: auto-detected (CUDA > MPS > CPU)"
    python3 scripts/06_train_generalized_v2.py
    echo -e "${GREEN}✅ Phase 2 CNN training complete${NC}"
    echo "Results: results/generalized/v2_cnn_results.json"
}

case "$STEP" in
    phase1_htr)
        check_deps
        phase1_htr
        ;;
    phase2_cnn)
        check_deps
        phase2_cnn
        ;;
    all)
        check_deps
        phase1_htr
        phase2_cnn
        ;;
    help|*)
        echo ""
        echo "Usage: bash run_pipeline.sh [phase1_htr | phase2_cnn | all]"
        echo ""
        echo "  phase1_htr  — run HTR-SELEX PRJEB25907 Phase 1 validation (missing)"
        echo "  phase2_cnn  — train generalized CNN V2 on all 3 datasets (~30min MPS)"
        echo "  all         — run phase1_htr then phase2_cnn"
        echo ""
        echo "Current status:"
        python3 -c "
import json, os
datasets = {
    'HTR-SELEX PRJEB25907': 'results/htr_selex/metrics/validation_results.json',
    'RBNS':                 'results/rbns/metrics/validation_results.json',
    'HTR-SELEX PRJEB47428': 'results/htr_selex_prjeb47428/metrics/validation_results.json',
}
for name, path in datasets.items():
    try:
        d = json.load(open(path))
        print(f'  Phase 1 {name}: {d[\"status\"]} (AUROC={d[\"best_val_auroc\"]:.3f})')
    except:
        print(f'  Phase 1 {name}: ❌ MISSING — run phase1_htr')
cnn = 'results/generalized/v2_cnn_results.json'
try:
    d = json.load(open(cnn))
    print(f'  Phase 2 CNN: done (test AUROC={d[\"test_metrics\"][\"auroc\"]:.3f})')
except:
    print(f'  Phase 2 CNN: ❌ NOT TRAINED — run phase2_cnn')
"
        ;;
esac
