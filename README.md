# Protein–RNA Binding Prediction

Machine learning pipeline for predicting whether an RNA-binding protein (RBP)
binds a given RNA sequence, trained across multiple large-scale in vitro
and (planned) in vivo binding datasets.

## Strategy

```
Input: protein amino acid sequence + RNA nucleotide sequence
Output: binding probability ∈ [0, 1]
```

Three-stage development:

**Phase 1 — Dataset validation** (complete)  
Each dataset is tested independently with fast baseline models (LR / RF / XGBoost on
k-mer features, protein-aware train/test split). A dataset passes if val AUROC > 0.70
on entirely unseen proteins, confirming that binding signal is learnable and data quality
is sufficient before merging into the generalized pool.

**Phase 2 — Generalized model** (in progress)  
All validated datasets are merged (~630k examples, ~170 unique RBPs).
Three architectures in sequence:
- V1: MLP on k-mer features — cross-dataset baseline
- V2: Dual-branch CNN on raw sequences — learns positional motif filters, length-agnostic
- V3: ESM-2 protein embeddings + RNA CNN — pre-trained evolutionary protein language

**Phase 3 — Extension** (planned)  
- Add in vivo datasets (eCLIP/ENCODE, iCLIP)
- Multi-task: binding classification + affinity regression (R_max from RBNS)
- Cross-attention interaction module (RNA tokens × protein residues)
- Experimental context label (in vivo / in vitro)

## Datasets — Phase 1 results

| Dataset | Source | RBPs | Examples | Best val AUROC | Best val AUPRC | Status |
|---|---|---|---|---|---|---|
| HTR-SELEX PRJEB25907 | Ray et al., *Genome Res.* 2019 | 93 | 279,000 | 0.825 (XGB) | 0.742 | ✅ PASS |
| RBNS | Lambert et al., *Nature* 2020 | 96 | 284,642 | 0.758 (RF) | 0.684 | ✅ PASS |
| HTR-SELEX PRJEB47428 | Laverty et al., *NAR* 2022 | 23 | 69,000 | 0.817 (LR) | 0.736 | ✅ PASS |

All splits are **protein-aware**: no RBP appears in both train and test.
This mirrors the real deployment scenario — predicting binding for a novel RBP
with no prior experimental data.

## Phase 2 — Generalized model results

Combined training pool: 632,642 examples · 168 unique proteins · 3 datasets

| Model | Encoding | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|
| MLP (V1) | RNA 4-mer + Prot 3-mer | 0.716 | 0.636 | 0.674 | 0.544 |
| CNN (V2) | One-hot raw sequences | **0.811** | **0.734** | **0.703** | **0.599** |
| ESM-2 + RNA CNN (V3) | ESM-2 1280-d + RNA CNN | in progress | in progress | — | — |

**V2 per-protein test results** (24 unseen proteins):

| Protein | Dataset | AUROC |
|---|---|---|
| ESRP1-construct3 | HTR-SELEX 25907 | 0.981 |
| PUF60 | RBNS | 0.924 |
| KHDRBS3 | HTR-SELEX 25907 | 0.901 |
| PUM2 | HTR-SELEX 25907 | 0.835 |
| RBM28 | HTR-SELEX 25907 | 0.842 |
| … | … | … |
| SRSF8 | RBNS | 0.577 |
| TAF15 | RBNS | 0.468 |
| IGF2BP3 | RBNS | 0.459 |
| UNK | RBNS | 0.429 |

Median per-protein AUROC: **0.718** · Median HTR-SELEX: 0.779 · Median RBNS: 0.611

## Benchmark

| Model | Method | AUROC | AUPRC | Split | Reference |
|---|---|---|---|---|---|
| Our XGBoost (k-mer) | Protein-aware, per-dataset | 0.825 | 0.742 | Protein-aware | This project |
| Our CNN V2 (generalized) | Protein-aware, cross-dataset | 0.703 | 0.599 | Protein-aware | This project |
| Our ESM-2 V3 (generalized) | Protein-aware, cross-dataset | in progress | in progress | Protein-aware | This project |
| ZHMolGraph | Frozen LLM + GNN on interaction network | 0.798 | 0.820 | Hard (unseen prot+RNA) | Liu et al. 2025 |
| RPITER | Iterative feature refinement + RF | 0.727 | 0.774 | Random | Peng et al. 2019 |
| IPMiner | Deep autoencoder + DBN | 0.664 | 0.742 | Random | Pan et al. 2016 |

## Repository structure

```
protein_rna_ml/
├── configs/          YAML configs for each dataset and model
├── scripts/
│   ├── 01_prepare_dataset.py             Phase 1: encode + split one dataset
│   ├── 02_train_validation_model.py      Phase 1: train baseline classifiers
│   ├── 03_evaluate_validation.py         Phase 1: test-set eval + plots
│   ├── 04_build_generalized_dataset.py   Phase 2: merge all datasets
│   ├── 05_train_generalized_v1.py        Phase 2 V1: MLP on k-mer
│   ├── 06_train_generalized_v2.py        Phase 2 V2: Dual-branch CNN
│   ├── 07_extract_esm2_embeddings.py     Phase 2 V3: extract ESM-2 embeddings
│   └── 08_train_generalized_v3.py        Phase 2 V3: ESM-2 + RNA CNN
├── src/
│   ├── data/         Dataset classes, k-mer encoding, splitting
│   ├── models/       MLP, CNN, ESM-2 projection model
│   └── utils/        Metrics, visualization
├── results/
│   ├── phase1_summary.json   All Phase 1 results with interpretation
│   ├── phase2_summary.json   Phase 2 model comparison
│   ├── htr_selex/            Per-dataset metrics + plots
│   ├── rbns/
│   ├── htr_selex_prjeb47428/
│   └── generalized/          Cross-dataset model results
├── tasks/
│   ├── todo.md       Current task tracking
│   └── lessons.md    Lessons learned / known issues
├── METHODS.md        Methodological choices with rationale and references
└── PROJECT_PLAN.md   Full project roadmap
```

## Quick start

```bash
pip install -r requirements.txt

# Phase 1: validate one dataset
python scripts/01_prepare_dataset.py --config configs/rbns_validation.yaml
python scripts/02_train_validation_model.py --config configs/rbns_validation.yaml
python scripts/03_evaluate_validation.py --config configs/rbns_validation.yaml --model xgboost

# Phase 2 V2: CNN (protein-aware generalized model)
python scripts/04_build_generalized_dataset.py
python scripts/06_train_generalized_v2.py          # ~30 min GPU / ~90 min CPU

# Phase 2 V3: ESM-2 + RNA CNN
pip install transformers sentencepiece
python scripts/07_extract_esm2_embeddings.py       # ~10-15 min, runs once
python scripts/08_train_generalized_v3.py          # ~20-40 min GPU
```

## Key design decisions

- **Protein-aware split**: prevents data leakage across all phases; see `METHODS.md §3`
- **Primary metric: AUPRC** — more informative than AUROC under 1:2 class imbalance
- **k-mer encoding (Phase 1)**: RNA 4-mer (256 features) + Protein 3-mer (8000 features)
- **CNN (Phase 2 V2)**: dual-branch 1D conv with global max pooling — length-agnostic,
  resolves the 20 nt (RBNS) vs 40 nt (HTR-SELEX) length mismatch that breaks k-mer MLP
- **ESM-2 (Phase 2 V3)**: frozen mean-pool embeddings replace one-hot protein branch;
  1280-d evolutionary representation vs 8000 sparse k-mer frequencies
