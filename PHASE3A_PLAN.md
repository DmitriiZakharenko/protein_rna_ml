# Phase 3A Plan — Expand Training with RNAcompete

**Status**: In progress  
**Last updated**: 2026-06-25  
**Goal**: Scale protein diversity in the generalized training set by merging
HTR-SELEX + RBNS with a curated RNAcompete training subset, using homology-aware
protein splits.

---

## 1. Problem Statement

Simple per-protein classifiers achieve AUROC 0.95–0.99 on RNA-only features, but the
best cross-protein model (V2 CNN) reaches only test AUROC 0.69 and zero-shot RNAcompete
median AUROC 0.55. The bottleneck is **training diversity** (~169 proteins), not data
quality within each assay.

Phase 3A addresses this by adding RNAcompete to training while preserving a clean
zero-shot benchmark on held-out proteins.

---

## 2. RNAcompete Inclusion Policy

| Sub-panel | Training use | Rationale |
|-----------|--------------|-----------|
| **Eukarya** | Full panel (~200 RBPs) | Clean negatives, diverse organisms |
| **RBPZoo** | Full panel (~174 RBPs) | Best negative quality (Sasse et al. 2025) |
| **ucRBP** | **23 proteins only** | Ray & Laverty 2023 reproducibility filter; full ucRBP panel (~613 experiments) includes proteins that failed QC |

Whitelist: `configs/ucrbp_23_reproducible.txt`  
Builder script: `scripts/22a_prepare_rnacompete_training.py`  
Output TSV: `data/benchmarks/rnacompete/rnacompete_training_phase3a.tsv`

**Expected RNAcompete contribution**: ~397 unique proteins (minus overlap with SELEX/RBNS),
~6–8M pairs before subsampling.

---

## 3. Pipeline Steps

### Step 0 — Prepare RNAcompete training subset
```bash
python scripts/22a_prepare_rnacompete_training.py
```

### Step 1 — Export FASTA for homology search
```bash
python scripts/22_build_phase3a_dataset.py \
    --export_fasta_only \
    --selex_dir data/generalized_v2 \
    --rnacompete data/benchmarks/rnacompete/rnacompete_training_phase3a.tsv \
    --out_dir data/generalized_v3a
```

### Step 2 — MMseqs2 homology search (30% identity)
```bash
mkdir -p results/homology tmp/mmseqs
mmseqs easy-search \
    data/generalized_v3a/proteins_train.fasta \
    data/generalized_v3a/proteins_rnacompete.fasta \
    results/homology/train_vs_rnacompete.tsv \
    tmp/mmseqs/ \
    --min-seq-id 0.30 -c 0.8 --cov-mode 0
```

### Step 3 — Build merged dataset
```bash
python scripts/22_build_phase3a_dataset.py \
    --selex_dir data/generalized_v2 \
    --rnacompete data/benchmarks/rnacompete/rnacompete_training_phase3a.tsv \
    --homology_tsv results/homology/train_vs_rnacompete.tsv \
    --out_dir data/generalized_v3a \
    --max_rnacompete 2000000
```

### Step 4 — Train baseline on expanded data (compare to V2)
```bash
python scripts/06_train_generalized_v2.py \
    --data_dir data/generalized_v3a \
    --epochs 60 \
    --prot_max 700 \
    --model_dir models/saved/generalized_v3a \
    --out_dir results/generalized/v3a_scale
```

### Step 5 — Evaluate zero-shot benchmark (unchanged protocol)
```bash
python scripts/20_evaluate_benchmark.py \
    --model_path models/saved/generalized_v3a/best_model.pt \
    --benchmark_dir data/benchmarks/rnacompete
```

---

## 4. Split Rules (homology-aware)

1. SELEX/RBNS splits from `generalized_v2` are **frozen** — no resplit.
2. RNAcompete proteins with **exact name match** to existing SELEX protein → same split.
3. RNAcompete proteins **homologous** (>30% ID) to SELEX val/test protein → forced to **train** only.
4. Remaining new RNAcompete proteins → random protein-level assignment (10% val, 10% test).

---

## 5. Success Criteria

| Metric | V2 baseline (v2 data) | Phase 3A target |
|--------|----------------------|-----------------|
| Test AUROC (held-out proteins) | 0.690 | ≥ 0.70 |
| Test AUPRC | 0.580 | ≥ 0.58 |
| RNAcompete zero-shot median (unseen proteins) | 0.549 | ≥ 0.60 |
| Per-protein median test AUROC | 0.714 | ≥ 0.72 |

Report multi-seed variance (5 seeds) before claiming improvement.

---

## 6. Local Machine vs Cloud VM

### Safe on local laptop (Apple Silicon MPS or CPU)

| Task | RAM | Time estimate | Notes |
|------|-----|---------------|-------|
| `22a` RNAcompete subset build | 8–16 GB | 10–30 min | Reads ~5 GB TSVs |
| `22` dataset merge + subsample | 16 GB | 15–45 min | |
| MMseqs2 homology search | 8 GB | 5–15 min | Already installed |
| V2 CNN training (v3a, 60 epochs) | 16 GB | 2–6 h | Same as current V2 run |
| V4 bilinear training | 16 GB | 3–8 h | Slightly larger head |
| Multi-seed V2 (5×) | 16 GB | 10–30 h | Run overnight |
| XGBoost stacking on CNN embeddings | 8 GB | < 1 h | |

**Laptop safety**: sustained GPU/MPS training is normal workload. Modern laptops throttle
under heat but are designed for this. Use a cooling pad, keep vents clear, plug in power.
No permanent damage from training CNNs at batch size 256.

### VM / cloud GPU recommended

| Task | Why |
|------|-----|
| ESM-2 LoRA fine-tuning (650M params) | 24+ GB VRAM, long runs |
| RNA-FM + cross-attention transformer | Large memory, multi-day training |
| Full RNAcompete without subsampling (~8M+ rows) | Faster with GPU + more RAM |
| Hyperparameter sweeps (10+ configs) | Parallel runs |

**Practical cutoff**: everything through **Phase 3A + V4 + hard negatives + multi-task
regression head** fits a powerful laptop. Move to VM when starting **pretrained encoder
fine-tuning** or **transformer interaction layers**.

---

## 7. Data to Collect in Parallel

While Phase 3A runs, gather these (priority order):

### High priority
1. **MMseqs2 paralog map** for all 169 SELEX proteins vs each other (quantify current split leakage).
2. **iCLIP peaks** — TDP-43 (TARDBP) and FUS from ENCODE/GEO (in vivo binding sites).
3. **eCLIP expansion** — 10 → 30+ RBPs from ENCODE portal (GC-matched negatives already in pipeline).
4. **Supplementary Table S1** from Ray & Laverty 2023 — verify all 23 ucRBP names and experiment IDs match our whitelist.

### Medium priority
5. **RBNS R_max / affinity values** — enable multi-task regression head.
6. **RNA secondary structure** — ViennaRNA MFE for training RNAs (batch job, CPU).
7. **UniProt domain annotations** — RRM/KH/Pumilio boundaries for domain-aware encoding.
8. **PAR-CLIP AGO2** (GEO GSE21918) — miRNA seed-pairing logic.

### Lower priority (Phase 4+)
9. **RNAInter** interaction graph (hard negatives after homology filter).
10. **RNA-FM embeddings** precomputed for all training RNAs.
11. **Literature binding pairs** — expand external validation beyond 159 pairs.

---

## 8. Experiment Queue After Phase 3A

| ID | Experiment | Depends on |
|----|------------|------------|
| 3A-1 | V2 on `generalized_v3a` | This plan |
| 3A-2 | Multi-seed variance (5 seeds) | 3A-1 |
| 3B-1 | V4 `concat_bi` on v3a | 3A-1 |
| 3B-2 | Hard negatives mix | New script |
| 3C-1 | CNN embedding → XGBoost stack | 3A-1 checkpoint |
| 3C-2 | Multi-task affinity regression | RBNS R_max + RNAcompete intensity |

---

## 9. What NOT to Do

- Do not merge full ucRBP panel (613 experiments) — noisy proteins hurt training.
- Do not use RNAcompete for both training and zero-shot eval on the same proteins.
- Do not add another frozen ESM-2 mean-pool variant (3 failed experiments).
- Do not tune on test set before multi-seed baseline is established.
