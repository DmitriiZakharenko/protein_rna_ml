# ENCODE protein map: ESM-2 embeddings of TF and RBP proteins and 5-mer binding profiles

Script `scripts/82_encode_protein_map.py`, results in `results/encode_map/`, interactive page `results/encode_map/encode_protein_map.html` (served locally with `python -m http.server`). Date of the analysis: 2026-11-26.

## Data and method

- Proteins: 1,051 human proteins. 873 are ENCODE TF ChIP-seq targets (TF-annotated list, GRCh38), 168 are ENCODE eCLIP targets (GRCh38). Class from the broad ChIP-seq list: TF only 869, RBP only 143, both assays 39 (7 with the TF-annotated list).
- Embedding: ESM-2 650M, mean over residues, 34 layers, windows of 1,022 residues with stride 511 for longer proteins. Similarity: cosine after centring over proteins.
- 5-mer profile: log2 of the 5-mer frequency in positive windows over the frequency in GC-matched negative windows (bins of 0.05), 1,024 values, centred over proteins within each assay. Sources: TF ChIP-seq windows (248 TFs, up to 1,000 peaks per TF and cell line) and Skipper eCLIP windows (148 proteins with at least 100 positive windows). Similarity: Pearson r of two profiles (r5) and Jaccard index of the 50 most enriched 5-mers.
- Annotations: UniProt keywords, Pfam domains and names (1,050 of 1,051 proteins found), sequence features, number of ENCODE experiments and cell lines.
- Statistics: permutation nulls (proteins permuted), bootstrap over proteins for AUROC intervals. P-values are not corrected for the 34 layers, the pair groups and the two similarity targets, and best layers were chosen after the fact.

## Results

| Question | Result |
|---|---|
| Does the nearest neighbour of a protein have the same class? (layers 0, 10, 18, 22 and 33, chance in brackets) | TF 0.93, 0.97, 0.97, 0.97, 0.95 (0.83). RBP 0.55, 0.65, 0.74, 0.72, 0.66 (0.14). Both assays 0.31, 0.56, 0.51, 0.49, 0.41 (0.04) |
| Does the nearest neighbour have the same domain family (v3b labelled proteins)? | 0.60 at layer 0, 0.80 at layer 18, 0.67 at layer 33, null 0.37 |
| Score "RBP-likeness" (mean cosine to RBP-only proteins minus mean cosine to TF-only proteins), AUROC for proteins with eCLIP and ChIP-seq against TF-only proteins | layer 0 0.818, best layer 21 0.973 [0.948, 0.990], layer 33 0.882. Baselines: UniProt RNA-binding keyword 0.853, fraction of G,S,Y,Q,N 0.647, fraction of K,R 0.609, length 0.504 |
| Can the embedding tell proteins with ChIP-seq and eCLIP from other RBPs? (TF-likeness, AUROC) | layer 0 0.710, best layer 4 0.726 [0.640, 0.811], layer 33 0.551. Baselines: fraction of G,S,Y,Q,N 0.745, UniProt DNA-binding keyword 0.688, transcription-regulation keyword 0.684 |
| Does the 5-mer profile recover known consensus sequences? | 21 of 24 TFs have a 5-mer of the consensus (either strand) in the top 20. Rank 1 for GATA1, MAX, USF1, FOSL1, CEBPB, NFYA, NFYB, ELK1, GABPA, YY1, NRF1, CREB1. Misses: SP1 (rank 21), ZBTB33 (54), TCF7L2 (67) |
| Spearman correlation between centred cosine and profile similarity, TF-TF pairs (30,628) | r5: 0.024 at layer 0, 0.064 at layer 12. Jaccard of top 50: 0.052 at layer 0, 0.098 at layer 5 |
| The same, RBP-RBP pairs (10,878) | r5: 0.048 at layer 0, the best layer is layer 0. Jaccard: 0.059 at layer 0, 6 of 34 layers with p below 0.05 |
| The same, TF-RBP pairs (36,700) | r5 about 0 (0 of 34 layers with p below 0.05), Jaccard 1 of 34 layers |
| Profile similarity inside and between Pfam domain groups | mean r5 0.018 inside a group against -0.001 between groups (permutation p 0.002). Spearman of cosine against r5: 0.169 inside groups and 0.001 between groups at layer 21 (0.088 and 0.000 at layer 0) |
| Ridge prediction of a protein profile from its embedding (per-protein Pearson r) | ChIP-seq: random 5-fold 0.041 at layer 0, best 0.147. Leave one domain group out: 0.028 at layer 0, 0.066 at layer 9 (null -0.009, 95th percentile 0.012, p 0.02, 50 permutations). eCLIP: random 0.035 to 0.077. Leave group out: 0.014 to 0.049 at layer 18 (null -0.017) |
| Same protein, ChIP-seq profile against eCLIP profile | 4 proteins with both profiles, mean r5 0.057 against 0.000 for random pairs (p 0.085) |

## Conclusions supported by these numbers

1. Middle layers of ESM-2 (around layers 18 to 22) separate eCLIP proteins from TFs (AUROC 0.973) and group proteins by domain family better than the composition layer 0 (0.82, 0.60) and better than the last layer (0.88, 0.67). The UniProt RNA-binding keyword gives 0.853. The classes are defined by the ENCODE target lists, so this describes which proteins ENCODE assayed with which method.
2. The embedding does not identify proteins with both assays among the RBPs beyond composition (0.73 against 0.71 at layer 0 and 0.745 for the fraction of G,S,Y,Q,N).
3. Similarity of embeddings explains little of the similarity of binding profiles (Spearman at most about 0.10) and mostly inside Pfam domain groups. This agrees with the RNAcompete test (0.07 to 0.16). For RBPs the best layer is the composition layer.
4. Ridge prediction of the profile of a protein from a held-out domain group gives per-protein r of 0.05 to 0.07, above the permutation null, below any practical use.
5. The 5-mer profiles are valid: the consensus control passes for 21 of 24 TFs. The weak association is therefore not caused by empty profiles.

## Genomic overlap of ChIP-seq and eCLIP peaks (script 83)

Files: ENCODE GRCh38 ChIP-seq "IDR thresholded peaks" (89 experiments) and eCLIP "peaks" (CLIPper, 68 experiments) of the 39 proteins that have both assays, 156 files of 157 experiments (one experiment had no file of this type), 149 MB, not tracked in git. Intervals were merged without strand. Jaccard index (base pairs) of every ChIP-seq set against every eCLIP set of the same cell line. Own-protein pair against the other proteins of the same cell line as null, because a random shift of intervals ignores that eCLIP peaks lie in expressed genes. The Jaccard index depends on the number and width of the peaks, so log Jaccard was also centred over rows and columns (size-adjusted).

| Question | HepG2 (26 ChIP sets, 33 eCLIP sets, 22 own pairs) | K562 (34 ChIP sets, 32 eCLIP sets, 28 own pairs) |
|---|---|---|
| Own-protein Jaccard against the Jaccard with other proteins (median) | 0.0083 against 0.0109 | 0.0034 against 0.0057 |
| Own pair ranks first (raw, size-adjusted) | 1 of 22, 1 of 22 | 0 of 28, 1 of 28 |
| Median rank of the own pair among all eCLIP sets (size-adjusted) | 18 of 33 | 13 of 32 |
| Spearman of Jaccard with centred cosine, layer 21 (raw, size-adjusted) | -0.073 (p 0.09), -0.027 (p 0.41) | -0.202 (p 0.003), -0.069 (p 0.037) |
| Highest size-adjusted own-pair overlap | TARDBP z 3.5 (rank 1 of 33), SRSF1 1.7, HNRNPK 1.6 | KHSRP 1.8 (rank 1), HNRNPK 1.6, TARDBP 1.5 |

Reading: the ChIP-seq peaks of an RBP do not overlap its own eCLIP peaks more than the eCLIP peaks of other RBPs in the same cell line (median rank in the middle). The negative raw correlation with the embedding cosine in K562 is a set-size effect and falls to -0.06 to -0.07 after adjustment. TARDBP (HepG2), HNRNPK, KHSRP, SRSF1 and U2AF2 stand out as single cases with 22 and 28 own pairs and no correction for multiple tests, so they are hypotheses and not results. The eCLIP files are CLIPper peaks as deposited and were not filtered by reproducibility or enrichment, and the 156 files are the preferred or largest file per experiment. Tables: `results/encode_peaks/`.

### All TF and RBP pairs of a cell line (script 83, stage overlap_all)

Files: 1,413 ENCODE GRCh38 peak files (TF-annotated ChIP-seq IDR-thresholded peaks and eCLIP peaks of K562 and HepG2, 728 MB, not tracked in git), one file per experiment, sets of the same protein, cell line and assay merged. Pairs of the same protein were excluded. Statistic: Spearman correlation between the centred cosine of the ESM-2 embeddings and the Jaccard index of two peak sets, raw and after removing the row and column effect of log Jaccard (size-adjusted). Null: protein labels permuted (200 permutations, smallest possible p 0.005).

| Cell line | Pairs | Layer 0 | Layer 18 | Layer 21 | Layer 33 |
|---|---|---|---|---|---|
| K562, 352 ChIP-seq sets x 139 eCLIP sets | 48,926 | raw +0.047, adjusted +0.007 (p 0.11) | +0.028, +0.018 (p 0.005) | +0.035, +0.013 (p 0.010) | +0.073, +0.027 (p 0.005) |
| HepG2, 552 x 105 | 57,955 | raw -0.019, adjusted +0.018 (p 0.005) | -0.021, +0.031 (p 0.005) | -0.043, +0.020 (p 0.005) | -0.009, +0.025 (p 0.005) |

Reading: the size-adjusted correlations are between +0.007 and +0.031. They differ from zero statistically because of the number of pairs, but explain under 0.1% of the rank variance. Similarity of ESM-2 embeddings of a TF and an RBP does not predict how much their peaks overlap in the genome. Layer 0 (composition) is not clearly lower than the contextual layers. Raw correlations change sign between the cell lines, as expected from a set-size effect. Pairs of proteins from the same domain group give inconsistent means (K562 168 pairs, residual -0.10 against -0.002 for other pairs, HepG2 301 pairs, +0.046 against -0.001). A first version of the script also compared the overlap of TFs with the RNA-binding keyword and the RBP-likeness score, but this was computed on row means that are zero after the adjustment and was removed. Tables: `results/encode_peaks/jaccard_all_*.tsv.gz`, `overlap_all_summary.json`.

### Reproducibility of peaks of one protein between K562 and HepG2 (script 83, stage cross_cell)

Jaccard of the K562 peak set of a protein against the HepG2 sets of all proteins, own protein against the others. eCLIP: 76 RBPs in both cell lines, 105 candidate sets, own set first in 34 cases (45%) raw and 28 (37%) after the size adjustment, in the top 5 in 71% and 68%, median Jaccard 0.086 against 0.035 for other proteins. ChIP-seq: 205 TFs in both cell lines, 552 candidate sets, own set first in 49 cases (24%) raw and 29 (14%) adjusted, in the top 5 in 38% and 35%, median Jaccard 0.060 against 0.038. Chance for the first place is 1% and 0.2%. The binding sites of an RBP are therefore partly shared between the two cell lines, those of a TF much less, and the absolute overlap is small (Jaccard about 0.09 for eCLIP). This bounds any evaluation that transfers an in vivo model between cell lines. Tables: `results/encode_peaks/cross_cell_*.tsv`.

## Literature check of the flagged proteins (search snippets only)

The check was made with a web search and reading of snippets, not of full texts. Part of the identifiers were recalled and are not verified. Statements must be checked against the primary papers before use.

| Protein | RNA binding | DNA or chromatin | Reading |
|---|---|---|---|
| TARDBP | strong (CLIP-seq) | locus-level ChIP and single-stranded TG repeats, no genome-wide ChIP-seq paper found | the HepG2 own-pair overlap (z 3.5) is a hypothesis, not a documented effect |
| HNRNPK | strong | single-stranded C-rich element of c-myc, ChIP-seq at 3' ends of immediate-early genes reported | documented dual binder |
| KHSRP | strong | no evidence found | no support for chromatin binding |
| SRSF1 | strong | ChIP-seq at promoters reported, linked to nascent RNA and Pol II | chromatin association mainly through RNA |
| U2AF2 | strong | ChIP signal depends on nascent RNA, no direct DNA binding found | same |
| FUBP1 | medium (KH domains) | strong (single-stranded FUSE upstream of c-myc, structure solved) | flagged RBP-like TF, consistent with the literature |
| YBX1 | strong | claimed at promoters, specificity contested | flagged, supported for RNA |
| CDC5L | medium (splicing complex) | medium-low (Myb-like domain) | flagged, partly supported |
| AGO2 | strong | contested in human cells, antibody artefacts reported | flagged, RNA supported |
| CEBPZ | medium (rRNA processing, no direct RNA assay) | medium-low | flagged, partly supported |
| ZGPAT, GPBP1L1, PREB, CC2D1A, EED | no experimental evidence, or weak and disputed (EED) | annotated or single promoters | flagged, not supported |

Of the 10 TF-labelled proteins among the 25 highest RBP-likeness scores that were checked here, 5 have published RNA-binding evidence of at least medium strength (YBX1, FUBP1, AGO2, CDC5L, CEBPZ) and 5 have none or weak evidence. The ENCODE-wide study of chromatin interactions of RBPs by Xiao and colleagues (2019, Cell, identifier recalled and not verified) is the natural reference for the finding that RBP ChIP-seq peaks do not coincide with the eCLIP sites of the same protein.

## Checks prepared and not yet run

`scripts/82_encode_protein_map.py kmer --mask_repeats` removes windows with a tandem repeat of a 1 to 6 nt unit that covers 16 nt or more, and `kmer --by_cell` writes one profile per protein, assay and cell line. The stages `kmerstats --kmer_file ... --suffix ...` and `reproducibility` evaluate them. Both runs need the window tables on the GPU VM (`docs/VM_RUNBOOK.md`, section 5a).

## Findings that limit the interpretation

- Pairs of TFs with the highest profile similarity and distant embeddings (ARNT, SOX6, ZNF282, IKZF1, PHB2, TRIM22, YBX1, r5 0.75 to 0.86, cosine -0.31 to -0.10) share the repeats CACAC, GUGUG and AUGGA, UCCAU. Their peaks are probably in repeats and microsatellites. Windows were not repeat-masked and not filtered with the ENCODE blacklist. Before centring over proteins the similarity of unrelated promoter TFs was even higher because of a shared CpG-rich context.
- Only 4 proteins have both a ChIP-seq and an eCLIP profile, so statements about dual-function proteins rest on 4 proteins. ChIP-seq of an RBP can reflect indirect chromatin association.
- Candidate list: the 25 TF-only proteins with the highest RBP-likeness (layer 21) contain 3 with the UniProt RNA-binding keyword (CDC5L, YBX1, AGO2, base rate 10 of 869). Others (ZGPAT, CEBPZ, PREB, CC2D1A, EED, GPN1, FUBP1, ZMAT5, GLYR1) are not verified in the literature.

## Open checks

Repeat masking and blacklist filtering of the windows, profiles per cell line, comparison with JASPAR matrices, peak-overlap Jaccard for all TF and RBP pairs of a cell line (done here for the 39 proteins with both assays), control by sequence identity (MMseqs2) next to Pfam groups, held-out evaluation by protein clusters instead of Pfam groups, correction for the number of layers.

Tables: `results/encode_map/layer_stats_broad.tsv`, `kmer_layer_stats.tsv`, `kmer_dual_test.json`, `patterns_*.tsv`, `patterns_summary.json`, `predict_profiles.tsv`, `motif_controls.tsv`.
