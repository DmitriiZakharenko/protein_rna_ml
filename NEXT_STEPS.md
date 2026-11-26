# Next steps (state of 2026-11-26)

Context: results so far are in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) and summarised in [`README.md`](README.md). The plan below lists the measured starting points, the planned work with a stated gate per phase, and the open checks. Each gate is a result that can fail, and the outcome is recorded in the log either way.

## Measured starting points

- Genomic-negative eCLIP benchmarks are solved from RNA composition: RNA-only k-mer LR reaches AUROC 0.808 (0.680 on the GC-matched subset).
- On the in vitro cross-protein-negative probe (RNA-only baselines 0.500), V4 reaches 0.754 +- 0.001 and V2 0.686 +- 0.015 on v3a (3 seeds). Probe AUROC follows the identity to the nearest training protein (V4: 0.525 without hit, 0.884 at 70-90%).
- The nearest-protein baseline (RNA-only k-mer LR of the most similar training proteins) reaches 0.746 [0.70, 0.79] (top-3 neighbours) on the v3a probe and 0.627 [0.57, 0.68] on the v3b probe.
- v3b (cluster split, test proteins at >= 50% identity to training: 62% in v3a, 3% in v3b) is the test set for claims about new proteins. On the v3b probe V2 reaches 0.527 and V4 0.559 (seed 42), and the frozen ESM-2 + RNA-FM bilinear head 0.531 +- 0.027 (ordinary labels) and 0.559 +- 0.010 (cross-protein negatives, 3 seeds).
- Adding 248 TFs of ENCODE ChIP-seq (857,839 rows) changed the probe AUROC of the frozen head to 0.547 +- 0.016 (ordinary labels) and 0.535 +- 0.021 (cross-protein negatives).
- The same head on proteins seen in training reaches 0.827-0.828 probe AUROC on held-out RNAs.
- Classic RPI benchmarks: protein-only AUROC 0.964 on RPI369 (both molecules 0.974), RNA-only 0.909 on RPI488 (both 0.906), RNA-only 0.831 against 0.874 with both on RPI2241 (random 5-fold, RPISeq-style RF). CORAL component-wise folds: gradient boosting on composition features 0.654 (30/50) and 0.710 (60/80).
- ESM-2 650M layers against RNAcompete binding-profile similarity (175 proteins): Spearman 0.07-0.16, no layer separated from layer 0.

## Phase 0: integrity of the finished experiments

| # | Task | Status |
|---|---|---|
| 0.1 | Split integrity audit (`scripts/62`) | Done: v3b has 0 shared protein names and sequences. v3a had 1 test name and 5 val sequences also in train. 69-73% of test rows share an RNA with train (assays reuse probe sets), handled by the probe |
| 0.2 | Bootstrap CIs by identity bin and Pfam family for V2 and V4 (3 seeds) on v3b and the probe | Open: V2 and V4 on v3b rest on seed 42 |
| 0.3 | Nearest-protein and PWM baselines on v3a and v3b (`scripts/59`) | Done |
| 0.4 | V2, V4 and the nearest-protein baseline in the same identity and family bins | Open for V2 and V4 seeds 0 and 1 on v3b |
| 0.5 | PaRPI: RNA-only controls and own CNNs on the 261 datasets (`scripts/63-64`) | Done. PaRPI itself was not run |
| 0.6 | Per-assay breakdown of the probe and the standard test | Done for the nearest-protein baseline (`results/v3b_report/`), open for the neural models |
| 0.7 | eCLIP: remove the 5 of 24 test proteins at >= 90% identity to v3a train (`scripts/55`) and re-report | Open |
| 0.8 | Gate: V2 and V4 on v3b compared with the nearest-protein baseline (probe 0.627, standard test 0.676) | Result for seed 42: below the baseline on the probe |

## Phase 1: comparison with published method families

1. RPISeq-style RF (RNA 4-mer, protein 3-mer in a 7-class alphabet) on the v3b standard, probe and eCLIP splits.
2. Frozen ESM-2 and RNA-FM embeddings with MLP heads on the same splits (done for ESM-2 650M layer 33 and RNA-FM, `scripts/66-67`). ProtTrans embeddings are not done.
3. Hybrid retrieval model after NPBIP: Pfam-domain alignment similarity instead of full-length identity, plus a residual learner.
4. All rows reported next to V2, V4 and the nearest-protein baseline in identity bins and Pfam families.

## Phase 2: mechanistic validation (`scripts/61`, `scripts/68`, `scripts/75`)

- Motif recovery: the measured motif of a held-out protein is the 5-mer enrichment of its positives against negatives in the test split. The predicted motif is the mean model score of every 5-mer inserted into background contexts. Metric: Spearman correlation and top-k overlap per protein, compared with the nearest-protein baseline and an RNA-only prior, by identity bin and family. Measured for V2 seed 42 on v3a: Spearman 0.221 (held-out proteins), 0.259 (training proteins), nearest-protein motif 0.446.
- RNA mutagenesis: mutating each position of the strongest positive RNAs, the importance profile is compared with the known motif positions.
- Protein mutagenesis: alanine substitution of recognition residues (RNP1 and RNP2 of RRM, GXXG loop of KH, Cys and His of CCCH) against an equal number of random residues, with shuffled-residue and disordered-tail controls.
- Gate: a model passes if its scores respond more to recognition-residue mutations than to random ones. V2 on v3a: more than random in-domain residues in 52% of held-out proteins (p = 0.84) and 65% of training proteins (p = 0.14).

## Phase 3: training objectives for the interaction

1. In-batch cross-protein negatives (same RNA, different protein). Done for V4 (`scripts/65`) and the frozen head (`xneg`). Ranking-only supervision (`scripts/67` modes `rank`, `rank_prot`) has no result yet.
2. RNA-binding-domain crop with residue-level cross-attention on ESM-2 residue embeddings.
3. Motif-profile regression: predicting the 5-mer enrichment vector of a protein (`scripts/73`, ESM-2 mean embedding: median Spearman 0.185 for ridge against 0.101 for the mean-profile prior), evaluated by motif recovery.
4. Selective prediction: reliability from identity to training proteins and family, AUROC against coverage.

## Phase 4: biological features

- RNA: RNAplfold unpaired probability per window, local structure motifs, repeat spacing, GC and low complexity. For in vivo data: icSHAPE or DMS, transcript region, splice-site distance, m6A.
- Protein: domain architecture and linker length, recognition-surface residues per family, conservation of specificity-determining residues, groove electrostatics and accessibility (AlphaFold structures), disorder and RGG boxes.
- Interaction: motif databases (ATtRACT, CISBP-RNA, RBPmap) as priors, RBNS concentration series, RBP-motif co-evolution.
- In vivo benchmark with expression-matched (TPM) and region-matched negatives and abundance covariates, evaluated by within-protein ranking on CLIPper or Skipper `enrichment_l2or` with HOMER PWM baselines.

## Phase 5: structure-derived recognition prior (Track C)

- C0 feasibility (1-2 weeks, stop criterion: no signal): PDB table of specific protein-RNA complexes (resolution <= 3.5 A, rRNA and tRNA removed, protein clusters at 30%), residue-base contacts (<= 4 A heavy atoms, stacking, hydrogen bonds), and the correlation of contact-derived base preferences with the composition of the top RBNS and SELEX k-mers for proteins that have both structure and in vitro data.
- C1 residue-base contact model (ESM-2 plus local structure to a base distribution), cluster-wise cross-validation.
- C2 register-aware composition into k-mer scores, fine-tuning on v3b with C1 pre-training, evaluated on v3b, the probe and unseen families (dsRBD, DEAD, PAZ, YTH from PRBAB and ProNAB).
- Risks: apo and holo differences, non-modular recognition in RRM and KH, non-specific PDB complexes, binding-register ambiguity.

## Phase 6: affinity, new families, validation, paper

- Affinity (probe intensity, R_max, Kd): Spearman targets. Measured with `scripts/72` on 52 v3b test proteins: nearest-protein 0.287 mean (median 0.340), protein-independent prior 0.062, own-data ridge 0.561.
- Evaluation-only sets for PRBAB, ProNAB and UTexas families absent from v3a.
- Prospective experimental validation of 3-5 predictions for proteins outside the training families (collaboration).
- Two paper framings: an audit and benchmark (homology and family-stratified evaluation, cross-protein negatives, RNA-only and nearest-protein baselines), and a method if it exceeds the nearest-protein baseline in the identity bins below 50% on v3b and passes the mutagenesis gate.
- Related work to read in full: NPBIP (2026), the protein-diversity mRNA-protein benchmark (J. Cheminformatics 2026), PaRPI (2025), ZeRPI (2025), IRIS (2025), RBPsuite 2.0 (2025).

## Data sources to extend coverage

The training set has about 500 proteins in 160 homology clusters, dominated by RRM and KH (v3a train: RRM 279, KH 37, CCCH 17). Candidates, with availability and sizes still to be verified:

- In vitro: PRBAB and ProNAB (evaluation-only families dsRBD, DEAD, PAZ, YTH, 3-6 proteins each), more RNAcompete, RBNS and HTR-SELEX proteins outside RRM and KH.
- In vivo: ENCODE eCLIP (done), POSTAR3 and CLIPdb (PAR-CLIP, iCLIP, HITS-CLIP) for non-RRM RBPs.
- Motif resources as weak labels or evaluation: ATtRACT, oRNAment, CISBP-RNA (motifs partly inferred by homology, which overlaps with the homology controls of v3b).
- Structure: PDB protein-RNA complexes for Track C, AlphaFold and RoseTTAFoldNA models.
- Affinity: ProNAB, PRA310, RNAcompete intensities, RBNS R-values.

Rule: a new source is first an evaluation-only set split by cluster, and is used for training afterwards.

## Publication structure

**Part A, audit and benchmark (new proteins).** Evidence in hand: homology-stratified audit, cluster split v3b, nearest-protein and RNA-only baselines, composition-neutral cross-protein probe, seen-protein control (0.827) against unseen clusters (0.53-0.63), frozen-embedding heads, hybrid and stacking, motif-profile regression, shortcut audit of RPI benchmarks and CORAL folds. Remaining: second and third seeds of V2 and V4 on v3b, report assembly, optional Track C pilot.

**Part B, known RBPs: binding prediction, variant effects, disease.** Per-protein in vivo models work without cross-protein transfer (protein-free CNN 0.888 median AUROC on 17 PaRPI HeLa datasets, 0.791 on all 261). Steps with gates:

- B0: recompute the CNN with the epoch selection of the published PaRPI code to measure how much of the 0.035-0.045 gap (HeLa) is protocol.
- B1: label quality (replicate concordance, continuous log2 enrichment target, eCLIP against PAR-CLIP and iCLIP covariate).
- B2: features (RNAplfold, icSHAPE, transcript region with region-matched negatives, positional k-mers, sliding windows). The reverse-transfer test showed that 1-4-mer frequencies miss motifs such as the PUM1 8-mer.
- B3: variant effect as in silico mutagenesis delta score per RBP, compared with k-mer and PWM delta-LLR and published tools on independent sets (allele-specific binding, RBP-QTL, UTR and splicing MPRAs, ClinVar non-coding variants). Gate: exceeds PWM delta-LLR with CI on at least one independent set.
- B4: disease link by enrichment of predicted RBP-disrupting variants in RBP-disease gene sets, with controls and without clinical claims.
- B5: protein-side variants (RBP domain missense) after a model passes the attribution gates of Part A.

Reverse transfer (`scripts/74`): k-mer LR trained on eCLIP of a protein scores the same protein in vitro at mean AUROC 0.670 (RNA prior 0.579, own-data ceiling 0.947, 50 proteins).

## Open checks

| Item | State | Needed |
|---|---|---|
| RBNS protein count | `results/phase1_summary.json` gives 52 proteins for RBNS, while `results/rna_only_per_protein_honest/`, `results/cross_protocol_invitro/roster_summary.json` and `docs/METHODS.md` give 96 (split 72 / 11 / 13) | Number of unique `protein_name` in `../rbns_analysis/results/ml_dataset_rbns_clean.tsv`, then correction of the JSON |
| Domain criterion | The Pfam filter is the conditional E-value of `--domE 1e-3`, the stored column is the i-Evalue (`docs/METHODS.md` section 10). On the 175 RBPZoo constructs an extra i-Evalue filter keeps 174 proteins and lowers median coverage from 0.641 to 0.545 | Sensitivity of the 448-protein family table (`scripts/56`) and of the v3b stratification to the filter |
| Training copy of the v3a checkpoints | Not recorded whether V2 (0.813) and V4 (0.829) used `data/generalized_v3a/` or `data/sanitized/generalized_v3a/` | Training command or log of the VM runs |
| 168 against 169 proteins in `generalized_v2` | 632,642 rows and 168 proteins (three datasets) against 659,004 rows and 169 proteins (with eCLIP) | Row count and protein list of `data/generalized_v2/` |
| V1 MLP | Result file dated 2026-04-27, before the sampler fix of `scripts/05` | Rerun |
| V2 and V4 on v3b | Seed 42 only, seeds 0 and 1 not completed | VM run |
| Per-layer correlations of script 80 | Point estimates for 24 of 34 layers and `layer_correlation_domain.tsv`, `layer_correlation_outside.tsv` are not in the repository | Files from the VM |
| Scripts 80 and 81 input paths | Defaults are `rnacompete_analysis/data/raw/...` relative to the project root, which matches the present layout (upstream folders inside the project) | After moving the folders one level up, pass `--zscores` and `--scores` explicitly |
| Share of RNAs above 60 nt in the curated external set | 51%, 56% and 57% are quoted in different files | Count on `data/external/external_benchmark_curated.tsv` |
| Bootstrap CI of the nearest-protein baseline on the v3b probe | [0.57, 0.68] (script 59) and [0.581, 0.679] (`results/v3b_report/v3b_probe.json`, script 69) | Choice of the reported interval |
