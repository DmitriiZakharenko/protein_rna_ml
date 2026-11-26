# Sequence-only baselines on PRA310 and PRA201 (binding free energy)

Script `scripts/84_pra310_baselines.py`, results in `results/pra310_baselines/` (`summary.tsv`, `per_fold.tsv`). Date: 2026-11-26.

## Data and method

The tables of the HARP-RPA authors (GitHub repository of the paper, 310 complexes in PRA310 and 201 in PRA201, one protein chain and one RNA chain each, measured binding free energy) with their five-fold assignments. The data licence is not stated on the pages that were read, so the tables are kept in `data/external/harp_rpa/` and are not tracked in git. Models are fitted on the training folds and scored on the held-out fold with the metric of the paper: Pearson (PCC) and Spearman (SCC) correlation between predicted and measured free energy, mean and sd over the five folds. Validation folds hold 54 to 72 complexes, so the sd between folds is large (0.1 to 0.2).

Models: mean free energy of the complexes of the k nearest training proteins by MMseqs2 bit score (training mean without a hit), the same with the nearest training RNAs (cosine of 3-mer frequencies), and ridge regression on RNA 3-mer frequencies and log length, on amino-acid composition and log length, and on both. No structure, no language model, no neural network.

## Results (PCC, mean over 5 folds)

| Model | PRA310 benchmark folds (CD-HIT 70%) | PRA310 protein identity below 50% | PRA310 free-energy-stratified folds | PRA201 benchmark folds | PRA201 free-energy-stratified folds |
|---|---|---|---|---|---|
| nearest protein, k = 1 | 0.302 | 0.327 | 0.535 | 0.360 | 0.529 |
| nearest protein, k = 3 | 0.306 | 0.345 | 0.562 | 0.350 | 0.539 |
| nearest RNA, k = 3 | 0.170 | 0.024 | 0.312 | 0.122 | 0.218 |
| ridge, RNA | 0.233 | 0.283 | 0.326 | 0.340 | 0.350 |
| ridge, protein composition | 0.295 | 0.234 | 0.399 | 0.359 | 0.395 |
| ridge, both | 0.295 | 0.275 | 0.371 | 0.421 | 0.417 |

Published values for comparison (HARP-RPA paper, PCC on PRA310): benchmark folds frozen language-model embeddings with an MLP 0.300, IPA 0.495, CoPRA 0.580, HARP-RPA 0.621. Protein sequence-aware split (50%): HARP-RPA 0.587. Label-stratified folds (Table 2): language-model MLP 0.441, F3Affinity 0.559, CoPRA without pretraining 0.575, HARP-RPA 0.661.

## Reading

1. On the benchmark folds the simple sequence baselines (PCC 0.29 to 0.31 on PRA310) match the published frozen-embedding MLP (0.300) and stay far below the structure-based methods (0.58 to 0.62). On PRA201 the ridge model on both molecules reaches 0.421.
2. On the folds that group the free energy in bins, the validation proteins often have a training protein of the same sequence (see the fold audit below). There the nearest-protein baseline gives PCC 0.535 to 0.562, which is at the level of the published F3Affinity (0.559) and CoPRA without pretraining (0.575) and 0.10 below HARP-RPA (0.661). Most of the published correlations on these folds can therefore be obtained by copying the free energy of a similar training protein.
3. The file `PRA310_S.csv` was identified with the label-stratified split because every fold holds 62 validation complexes as described in the paper. That the authors used this file for their Table 2 was not checked.
4. A gap of 0.3 PCC remains between the best sequence-only baselines and the structure-based methods on the benchmark folds and on the 50% identity split. Whether this gap is due to the 3D interface features or to the language-model adaptation is not tested here.

## Audit of the folds (`python scripts/84_pra310_baselines.py audit`, `results/pra310_baselines/fold_audit.tsv`)

Share of validation complexes with a training protein at the given identity (MMseqs2, query coverage of at least 0.5), mean over five folds:

| Split file | 30% identity or more | 50% or more | 70% or more | identical sequence | identical RNA |
|---|---|---|---|---|---|
| PRA310 benchmark folds (CD-HIT 70%) | 40% | 13% | 5% | 0% | 7% |
| PRA310 protein identity below 50% | 42% | 7% | 6% | 1% | 8% |
| PRA310 free-energy-stratified folds | 71% | 58% | 54% | 27% | 15% |
| PRA201 benchmark folds | 51% | 15% | 7% | 0% | 10% |
| PRA201 free-energy-stratified folds | 75% | 58% | 53% | 27% | 16% |


## Limits

Five folds of 54 to 72 complexes, large sd between folds, ridge penalty chosen by internal cross-validation on the training folds, nearest-protein hits with query coverage of at least 0.5 only. The affinity data come from different assays and conditions.
