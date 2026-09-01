# Cross-Assay Strategy — Protocol Comparison → Domain-Aware Models

**Status**: Week 1 complete (in-vitro cross-protocol, 2026-09-01); **next** = domain-aware (`scripts/38`)  
**Last updated**: 2026-09-01  
**Repo**: stay in `protein_rna_ml` (do not fork a new project)  
**Related**: `STRATEGY.md`, `RESEARCH_ROADMAP.md`, `DATA_SOURCES_AND_DOWNLOADS.md`,
`scripts/24–27b`, `scripts/33–36`, `scripts/40`

---

## 1. Scientific question

Not: *Can one model predict binding for every RBP and every assay?*

But:

> For the **same protein**, how consistent is RNA specificity across assays
> (HTR-SELEX, RBNS, RNAcompete, eCLIP), and does **domain architecture**
> explain when transfer succeeds or fails?

Binding prediction remains the measurement tool. The deliverable is a
**quantitative map of cross-assay agreement**, stratified by domains.

---

## 2. Why this week first

| Fact | Implication |
|------|-------------|
| RNA-only per-protein AUROC ~0.95–0.99 within assay | Signal is strong *inside* each protocol |
| Zero-shot / cross-protein AUROC ~0.55 historically | Universal generalization is the wrong headline |
| ~84 proteins after name-normalization in ≥2 protocols | Enough for a first figure panel |
| Scripts 24/25/27b + Table S1 already exist | Fast path; almost no new modeling |

Universal binder SOTA is deferred. Cross-protocol comparison is the high-value result
aligned with moving from sequence-only linking toward domains / assay context.

---

## 3. Week-1 deliverables (cross-protocol)

### 3.1 Outputs

| File | Content |
|------|---------|
| `results/cross_protocol/protein_roster.tsv` | Matched proteins, native names per protocol, domain labels |
| `results/cross_protocol/within_protocol_metrics.tsv` | Per protein × protocol AUROC/AUPRC (LR primary) |
| `results/cross_protocol/transfer_metrics.tsv` | Train protocol A → test protocol B |
| `results/cross_protocol/motif_concordance.tsv` | Top-10 7-mer Jaccard (+ optional score correlation) |
| `results/cross_protocol/summary.json` | Counts, combo frequencies, run metadata |
| `figures/cross_protocol_*.png` | Heatmaps / domain-stratified boxplots |

### 3.2 Protocols

| Protocol ID | Data source (default paths) |
|-------------|----------------------------|
| `htr_selex` | `../htr_selex_analysis/results/ml_dataset_simple_clean.tsv` |
| `rbns` | `../rbns_analysis/results/ml_dataset_rbns_clean.tsv` |
| `rnacompete_eukarya` | `../rnacompete_analysis/eukarya/results/ml_dataset_eukarya_clean.tsv.gz` |
| `rnacompete_rbpzoo` | `../rnacompete_analysis/rbpzoo/results/ml_dataset_rbpzoo_clean.tsv.gz` |
| `eclip` | `data/eclip/eclip_all.tsv` |

Paths are configured in `configs/cross_protocol.yaml`.

### 3.3 Matching rules (strict)

1. `base_gene_key()` strips construct suffixes; synonyms are applied only when they
   do **not** collide within the same protocol (`src/data/protein_names.py`).
2. Classifiers use roster `name_in_<protocol>` with **exact** native-name equality —
   constructs are never pooled.
3. Multiple constructs → one representative (prefer no `-constructN`); discarded
   natives listed in `match_ambiguities.tsv` / `all_names_in_*`.
4. Domain from Table S1 joined by gene key; ambiguous architectures flagged
   (`domain_match_status`).

**Do not use** `--within_from_cached` with `*_per_protein_metrics.tsv` best_model
rows (mostly RF). Cached path now requires LR rows from `*_model_comparison.tsv`.

### 3.4 Classifier protocol (must stay simple)

- Features: RNA **4-mer** frequency vectors (same as `scripts/25`).
- Primary model: **LogisticRegression** (`class_weight=balanced`).
- Secondary: RandomForest (ablation only; not the headline).
- **Within-protocol**: honest 60/20/20 stratified split; report **test** AUROC/AUPRC.
- **Transfer**: fit on all usable sequences of protocol A for that protein; score all
  of protocol B; report AUROC/AUPRC. No mixing of B into training.
- RNAcompete: best-experiment + modal-length filters (reuse logic from script 25).
- Skip protein×protocol if either class has too few examples (`min_pos` / `min_neg`).

### 3.5 Motif concordance

From `results/top_bottom_examples/all_protocols_summary.tsv` (positives only):

- Motif concordance: exact top-10 7-mer Jaccard **and** soft **core-5** Jaccard
  (handles register shifts, e.g. AUGCAUG vs GCAUGAA for RBFOX2).
- If enrichment / Z-scores present for shared k-mers → Spearman of scores.

### 3.6 Figures (minimum)

1. Transfer AUROC heatmap (protocol × protocol), median over proteins.
2. Within vs transfer AUROC scatter (per protein).
3. Boxplots of transfer AUROC by coarse domain class.
4. Motif Jaccard vs transfer AUROC (are motif-similar proteins the ones that transfer?).

### 3.7 Commands

```bash
# 1) Roster (no large sibling files required if metrics TSVs exist)
python scripts/33_build_cross_protocol_roster.py

# 2) Within-protocol + transfer classifiers (needs sibling clean TSVs + eCLIP)
python scripts/34_cross_protocol_classifiers.py \
    --config configs/cross_protocol.yaml \
    --model logistic_regression

# Optional: reuse existing honest within-protocol metrics as a quick baseline
python scripts/34_cross_protocol_classifiers.py \
    --config configs/cross_protocol.yaml \
    --within_from_cached \
    --model logistic_regression

# 3) Motif concordance
python scripts/35_cross_protocol_motif_concordance.py

# 4) Figures
python scripts/36_visualize_cross_protocol.py
```

**Timeline**: ~3–7 days on CPU. Step 2 dominates runtime; roster + motifs are minutes.

### 3.8 Success criteria (Week 1)

| Criterion | Pass bar |
|-----------|----------|
| Roster size | ≥60 proteins with ≥2 protocols |
| Transfer matrix | All pairwise protocol directions with ≥5 proteins each (where overlap allows) |
| Domain coverage | ≥50% of roster proteins have a Table S1 architecture |
| Narrative | Clear statement: within-assay high, cross-assay lower/variable; domains associate with gap |
| Honesty | No claim of universal generalization |

---

## 4. Next phase — Domain-aware model (after Week 1)

Depends on Week-1 transfer gaps being real and partly domain-structured.

### 4.1 Goal

Move from full-length sequence binding scores to **domain-conditioned** predictions
that are mechanistically interpretable.

### 4.2 Data to add

| Resource | Use |
|----------|-----|
| Table S1 domain strings + boundaries | Architecture labels; construct sequences |
| UniProt / InterPro for SELEX/RBNS/eCLIP proteins | Domain intervals on full-length sequences |
| Week-1 transfer metrics | Supervision signal: which architecture families transfer |

### 4.3 Experiments (minimal strong set)

1. **Construct-masked input** — protein CNN sees only RBD / construct residues vs full-length.
   Hypothesis: construct-masked matches RNAcompete better; full-length may hurt transfer.
2. **Domain-type conditioning** — learnable embedding of coarse domain class (or architecture
   string) concatenated into the interaction head of V2/V4.
3. **Attribution enrichment** — fraction of gradient/IG mass inside annotated domains vs
   flanking sequence (report median enrichment).
4. **Link to Week 1** — proteins with the same coarse class should show higher transfer
   AUROC than cross-class pairs (test statistically).

### 4.4 Scripts (planned numbering)

| Script | Role |
|--------|------|
| `37_annotate_protein_domains.py` | UniProt/InterPro + Table S1 → `data/domains/protein_domains.tsv` |
| `38_train_domain_conditioned_v2.py` | Masked / conditioned CNN variants |
| `39_domain_attribution_eval.py` | Attribution mass in domains |

Do **not** start these until Week-1 figures exist.

---

## 5. What not to do in parallel

- Do not burn weeks on full multi-seed V4 on CPU for this narrative.
- Do not merge all assays into one label and only report a single AUROC.
- Do not treat eCLIP as interchangeable with SELEX labels without reporting the gap.
- Do not expand frozen ESM-2 variants.

V2/V4 may continue as a side baseline. They are not the Week-1 headline.

---

## 6. Paper framing (working)

**Title direction**: Cross-assay consistency of RNA-binding specificity and its dependence
on domain architecture.

**Contribution stack**:
1. Matched multi-protocol protein panel with honest within- vs cross-assay metrics.
2. Motif concordance vs classifier transfer.
3. Domain stratification explaining part of the gap.
4. (Later) Domain-conditioned model + attribution; (parallel track) Skipper eCLIP resource.

**Venue target**: Bioinformatics / NAR / Communications Biology class — methods + analysis,
not glamour biology.

---

## 7. Checklist

### Week 1
- [ ] Run `33` → inspect roster size and domain coverage
- [ ] Run `34` → within + transfer tables
- [ ] Run `35` → motif Jaccard
- [ ] Run `36` → figures
- [ ] Write 1-page result note: main numbers + failure modes (construct naming, length mismatch)

### After Week 1
- [ ] Decide domain experiment order from observed gaps
- [ ] Annotate domains (`37`)
- [ ] Construct-mask vs full-length pilot on overlap proteins
- [ ] Conditioning + attribution

---

## 8. Data QC blockers (2026-07-20 audit)

### Protein sequence contamination in training TSVs

| Dataset | Issue | Example |
|---------|-------|---------|
| `generalized_v3a` | Embedded residue numbers | **RBM38** `...EE708090100110120AVV...` (104→89 aa) |
| `generalized_v3a` / v2 | Trailing `*` stop codon | HEXIM1/2, many ucRBP entries |

RNA-only cross-protocol classifiers (scripts 33–36) do **not** use protein
sequences, so Week-1 metrics are unaffected by this bug.

**Any protein-aware model (V2/V3/V4, ESM, domain masking) MUST retrain after sanitize.**

```bash
# Audit
python scripts/40_sanitize_training_protein_sequences.py

# Repair (writes data/sanitized/... or --inplace with .bak)
python scripts/40_sanitize_training_protein_sequences.py --apply
# then rebuild/retrain from sanitized TSVs
```

### Bugs fixed in cross-protocol code (same day)

1. Silent `A2BP1→RBFOX1` merge while **both** exist in RNAcompete Eukarya — blocked.
2. Construct pooling via `groupby(protein_key)` — removed; exact native filter only.
3. `--within_from_cached` reading RF `best_model` AUROC as if LR — refused; requires
   `model_comparison` LR `test_auroc`.
4. Dedupe preferred positives (optimistic) — aligned with script 25 majority vote.
5. Transfer now reports exact RNA-string overlap fraction (leakage diagnostic).
