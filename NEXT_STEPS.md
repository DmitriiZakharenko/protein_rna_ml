# Next steps (sequenced plan, 2026-10-29)

Context: results so far are in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). Key facts driving this plan:

- Genomic-negative eCLIP benchmarks are composition-solvable (RNA-only k-mer LR 0.81; 0.68 GC-matched).
- On the in vitro cross-protein-negative probe (same RNA, different protein; RNA-only baselines = 0.500) V4 reaches
  0.754 ± 0.001 and V2 0.686 ± 0.015 on v3a, but transfer tracks sequence similarity to training proteins (no homolog: ~0.5).
- A nearest-training-protein baseline (RNA-only k-mer models of the most similar proteins) matches V4 on the probe
  (0.746 [0.70, 0.79] vs 0.754) and has a higher per-protein median. The neural models have not yet shown more than
  homology transfer.
- v3b (cluster split, ≥50% identity to train: 62% → 3% of test proteins) is the honest test set; V2/V4 on v3b are running.
- Classic RPI benchmarks (RPI369/488/1446/2241) are largely solved by a single molecule (protein-only 0.96 on RPI369,
  RNA-only 0.91 on RPI488, RNA-only 0.83 vs both 0.87 on RPI2241).

Every phase below has a gate: a result that can fail and is recorded in the experiment log either way.

## Phase 0 — Integrity of the finished experiments (run alongside the VM job)

| # | Task | Status |
|---|---|---|
| 0.1 | Split integrity audit (`scripts/62`): sequence/name overlap, RNA overlap, duplicates, balance | done: v3b has 0 shared protein names/sequences; v3a had 1 test name and 5 val sequences also in train; ~70–73% of test rows share an RNA with train (assay probe sets), handled by the probe |
| 0.2 | Bootstrap CIs by identity bin / Pfam family for V2/V4 (3 seeds) on v3b and the probe | after VM job |
| 0.3 | Nearest-protein / PWM baselines on v3a and v3b (`scripts/59`) | done |
| 0.4 | Report V2/V4/kNN together in the same bins (identity, family) | after VM job |
| 0.5 | Reproduce PaRPI (Zenodo 10.5281/zenodo.14878562, GitHub ljquanlab/PaRPI) → add RNA-only k-mer baseline on their windows/split → rebuild with cross-protein negatives | data downloading |
| 0.6 | Per-assay breakdown (RBNS / HTR-SELEX / RNAcompete) of probe and standard test | todo |
| 0.7 | eCLIP: drop the 5 of 24 test proteins ≥90% identical to v3a train (`scripts/55`), re-report | todo |
| 0.8 | Align README/EXPERIMENT_LOG wording: "homology-dependent transfer", not "unseen proteins" | done in README |
| 0.9 | Gate: if V2/V4 on v3b do not beat the nearest-protein baseline (probe 0.63, standard 0.68), stop architecture work and go to Phases 2/4/5 | after VM job |

## Phase 1 — Fair comparison with published method families

1. RPISeq-style RF (RNA 4-mer + protein 3-mer, 7-class alphabet) on v3b standard, probe and eCLIP splits.
2. Frozen ESM-2 / RNA-FM (and ProtTrans) embeddings + MLP, ZHMolGraph/BioLLMNet-style, same splits.
3. Hybrid retrieval model à la NPBIP: Pfam-domain alignment similarity (not full-length identity) + residual learner.
4. Report all rows next to V2/V4/kNN in identity bins and Pfam families.

## Phase 2 — Mechanistic validation (motif recovery and in silico mutagenesis)

Script: `scripts/61_motif_recovery_and_mutagenesis.py`.

- **Motif recovery.** For each held-out protein, the "true" motif is the k-mer enrichment of its positives vs negatives in
  the test split. The predicted motif is the mean model score of every 5-mer inserted into random background contexts.
  Metric: Spearman/top-k overlap per protein, compared with the nearest-protein baseline and an RNA-only prior, stratified by
  identity bin and family.
- **RNA mutagenesis.** Mutate each position of the strongest positive RNAs; the importance profile should concentrate on
  the motif positions (known motif overlap).
- **Protein mutagenesis.** Alanine-substitute the RNA-recognition residues (RNP1/RNP2 for RRM, GXXG loop for KH, Cys/His
  for CCCH) versus an equal number of random surface residues; the predicted binding should drop more for recognition
  residues. Control: shuffled-residue and disordered-tail substitutions.
- Gate: if scores do not respond to recognition-residue mutations more than to random ones, the model is not using the
  domain and the claim "learned protein–RNA interaction" is not supported.

## Phase 3 — Training that forces interaction learning

1. Contrastive / in-batch cross-protein negatives (same RNA, different protein) as a training objective.
2. RNA-binding-domain crop + residue-level cross-attention with ESM-2 residue embeddings (no global mean pooling).
3. Motif-profile regression: predict the protein's k-mer enrichment vector, normalised across RBNS/SELEX/RNAcompete, trained
   with a ranking loss; evaluated by motif recovery (Phase 2).
4. Selective prediction: reliability from identity to training/family and calibration; report AUROC versus coverage and
   state the applicability domain explicitly (e.g. no answer below 30% identity).

## Phase 4 — Biological features

- RNA: RNAplfold unpaired probability per window (cheap; start here), local structure motifs, repeat spacing /
  multivalency, GC/low-complexity; for in vivo: icSHAPE/DMS, region (5'UTR/CDS/3'UTR/intron), splice-site distance, m6A.
- Protein: domain architecture and linker length, recognition-surface residues per family, ortholog conservation /
  specificity-determining residues, electrostatics and accessibility of the groove (AlphaFold structures, APBS),
  disorder / RGG boxes as a "low specificity" flag.
- Interaction: motif databases (ATtRACT, CISBP-RNA, RBPmap) as priors for ortholog-rich proteins, concentration series
  from RBNS (ranking), RBP–motif co-evolution.
- In vivo: build an honest benchmark with expression-matched (TPM) and region-matched negatives plus abundance/competition
  covariates; evaluate within-protein ranking on enrichment (CLIPper/Skipper `enrichment_l2or`, HOMER PWM baselines).

## Phase 5 — Track C: structure-derived recognition prior

C0 feasibility (1–2 weeks, kill criterion): build a PDB table of specific protein–RNA complexes (resolution ≤ 3.5 Å,
rRNA/tRNA removed, protein clusters at 30%), extract residue–base contacts (≤ 4 Å heavy atoms, stacking, H-bonds), and
test whether contact-derived base preferences correlate with the nucleotide composition of top RBNS/SELEX k-mers for
proteins that have both structure and in vitro data. Stop if there is no signal.
C1 residue–base contact model (ESM-2 + local structure → base distribution), cluster-wise cross-validation.
C2 register-aware composition into k-mer scores, end-to-end fine-tuning on v3b with C1 pre-training; evaluate on v3b,
the probe, and unseen families (dsRBD, DEAD, PAZ, YTH from PRBAB/ProNAB).
Risks: apo/holo differences, non-modular recognition in RRM/KH, non-specific PDB complexes, binding-register ambiguity.

## Phase 6 — Affinity, new families, validation, paper

- Phase 3C affinity (probe intensity / R_max / Kd): Spearman targets where baselines sit near 0.
- Eval-only on PRBAB/ProNAB/UTexas families absent from v3a.
- Prospective experimental validation of 3–5 predictions for proteins outside the training families (collaboration).
- Paper framing: (a) benchmark/audit — hidden shortcuts, cross-protein negatives, identity/family-stratified evaluation,
  RNA-only/nearest-protein baselines; (b) method only if it beats the nearest-protein baseline in bins < 50% identity on
  v3b and passes the mutagenesis gate.
- Related work to read in full before writing: NPBIP (2026), "protein diversity" mRNA–protein benchmark (J. Cheminf. 2026),
  PaRPI (2025), ZeRPI (2025), IRIS (2025), RBPsuite 2.0 (2025).

## Data sources to extend coverage (distinct proteins/families matter more than rows)
Current bottleneck: ~500 proteins, ~160 homology clusters, RRM/KH dominated. Candidates (availability and sizes to be verified):
- In vitro: PRBAB / ProNAB (eval-only families: dsRBD, DEAD, PAZ, YTH), more RNAcompete/RBNS/HTR-SELEX proteins outside RRM/KH.
- In vivo: ENCODE eCLIP (done, negative), POSTAR3 / CLIPdb PAR-CLIP/iCLIP/HITS-CLIP for non-RRM RBPs, matched-annotation negatives.
- Motif resources as weak labels or evaluation: ATtRACT, oRNAment, CISBP-RNA (motifs partly inferred by homology: circular for our test).
- Structure: PDB protein-RNA complexes (contact tables) for Track C; AlphaFold/RoseTTAFoldNA models as augmentation.
- Affinity: ProNAB, PRA310, RNAcompete intensities, RBNS enrichment (R-values).
Rule: any new source is first an evaluation-only set, split by cluster, before it is used for training.

## Frozen-embedding baselines (Phase 1, from the RPIEmbeddor comparison)
Mean-pooled ESM-2 + RNA-FM -> logreg/MLP, then token-level cross-attention, trained on v3b and evaluated on the v3b
probe with the nearest-protein gate (nn1 0.627). Report by identity bin and Pfam family.
