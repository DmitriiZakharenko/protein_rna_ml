# Protein–RNA Binding Prediction

Machine learning pipeline for predicting whether an RNA-binding protein (RBP)
binds a given RNA sequence, trained across multiple large-scale in vitro
and (planned) in vivo binding datasets.

## Goal

Build a generalizable model that outperforms the current state of the art —
**ZHMolGraph** (Liu et al., *Commun. Biol.* 2025, AUROC 0.798 on unseen proteins)
— using fine-tuned protein and RNA language model embeddings instead of frozen ones,
and curated experimental negatives instead of network-based negatives.

## Strategy

```
Input: protein amino acid sequence + RNA nucleotide sequence
Output: binding probability ∈ [0, 1]
```

Three-stage development:

**Phase 1 — Dataset validation** (complete)
Each dataset is tested independently with a fast baseline model (XGBoost on
k-mer features, protein-aware train/test split). A dataset passes if AUROC > 0.70
on entirely unseen proteins. This guards against noisy or uninformative data
entering the generalized training pool.

**Phase 2 — Generalized model** (in progress)
All validated datasets are merged into a single training pool (~630k examples,
~170 unique RBPs). Three architectures in sequence:
- V1: MLP on k-mer features — multi-dataset baseline
- V2: Dual-branch CNN on raw sequences — learns positional motif filters
- V3: ESM-2 (protein) + RNA-FM (RNA) with fine-tuning + cross-attention — target model

**Phase 3 — Extension** (planned)
- Add in vivo datasets (eCLIP/ENCODE, iCLIP)
- Multi-task: binding classification + affinity regression (R_max from RBNS)
- Experimental context label (in vivo / in vitro)

## Datasets — Phase 1 results

| Dataset | Source | RBPs | Examples | Best val AUROC | Status |
|---|---|---|---|---|---|
| HTR-SELEX PRJEB25907 | Ray et al., *Genome Res.* 2019 | 93 | 279,000 | 0.844 | ✅ PASS |
| RBNS | Lambert et al., *Nature* 2020 | 96 | 284,642 | 0.758 | ✅ PASS |
| HTR-SELEX PRJEB47428 | Laverty et al., *NAR* 2022 | 23 | 69,000 | 0.817 | ✅ PASS |

All splits are **protein-aware**: no RBP appears in both train and test.
This mirrors the real deployment scenario (predicting binding for a novel RBP).

## Benchmark

| Model | Method | AUROC (hard split) | Ref |
|---|---|---|---|
| ZHMolGraph | Frozen LLM + GNN on interaction network | 0.798 | Liu et al. 2025 |
| **This project V2 (target)** | CNN / fine-tuned ESM-2 + RNA-FM | > 0.85 | — |

## Repository structure

```
protein_rna_ml/
├── configs/          YAML configs for each dataset and model
├── scripts/
│   ├── 01_prepare_dataset.py          Phase 1: encode + split one dataset
│   ├── 02_train_validation_model.py   Phase 1: train baseline classifiers
│   ├── 03_evaluate_validation.py      Phase 1: test-set eval + plots
│   ├── 04_build_generalized_dataset.py  Phase 2: merge all datasets
│   ├── 05_train_generalized_v1.py     Phase 2 V1: MLP
│   └── 06_train_generalized_v2.py     Phase 2 V2: CNN
├── src/
│   ├── data/         Dataset classes, k-mer encoding, splitting
│   ├── models/       MLP, CNN, (planned) pretrained encoder models
│   └── utils/        Metrics, visualization
├── results/          Per-dataset validation metrics and plots (JSON + PNG)
├── METHODS.md        All methodological choices with rationale and references
└── PROJECT_PLAN.md   Full project roadmap
```

## Quick start

```bash
pip install -r requirements.txt

# Phase 1: validate one dataset
python scripts/01_prepare_dataset.py --config configs/rbns_validation.yaml
python scripts/02_train_validation_model.py --config configs/rbns_validation.yaml
python scripts/03_evaluate_validation.py --config configs/rbns_validation.yaml --model xgboost

# Phase 2: train generalized model
python scripts/04_build_generalized_dataset.py
python scripts/05_train_generalized_v1.py   # MLP baseline (~15 min, CPU)
python scripts/06_train_generalized_v2.py   # CNN (~30 min, GPU recommended)
```

## Key design decisions

- **Protein-aware split**: prevents data leakage; see `METHODS.md §3`
- **k-mer encoding (Phase 1)**: RNA 4-mer (256 features) + Protein 3-mer (8000 features)
- **CNN (Phase 2 V2)**: dual-branch 1D conv; handles variable sequence lengths
- **Planned ESM-2 + RNA-FM (Phase 2 V3)**: fine-tuned language model embeddings;
  this is where we expect to surpass ZHMolGraph
