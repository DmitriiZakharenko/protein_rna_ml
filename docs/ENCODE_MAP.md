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

## Findings that limit the interpretation

- Pairs of TFs with the highest profile similarity and distant embeddings (ARNT, SOX6, ZNF282, IKZF1, PHB2, TRIM22, YBX1, r5 0.75 to 0.86, cosine -0.31 to -0.10) share the repeats CACAC, GUGUG and AUGGA, UCCAU. Their peaks are probably in repeats and microsatellites. Windows were not repeat-masked and not filtered with the ENCODE blacklist. Before centring over proteins the similarity of unrelated promoter TFs was even higher because of a shared CpG-rich context.
- Only 4 proteins have both a ChIP-seq and an eCLIP profile, so statements about dual-function proteins rest on 4 proteins. ChIP-seq of an RBP can reflect indirect chromatin association.
- Candidate list: the 25 TF-only proteins with the highest RBP-likeness (layer 21) contain 3 with the UniProt RNA-binding keyword (CDC5L, YBX1, AGO2, base rate 10 of 869). Others (ZGPAT, CEBPZ, PREB, CC2D1A, EED, GPN1, FUBP1, ZMAT5, GLYR1) are not verified in the literature.

## Open checks

Repeat masking and blacklist filtering of the windows, profiles per cell line, comparison with JASPAR matrices, peak-overlap Jaccard with a null that keeps the genomic region (needs the narrowPeak files), control by sequence identity (MMseqs2) next to Pfam groups, held-out evaluation by protein clusters instead of Pfam groups, correction for the number of layers.

Tables: `results/encode_map/layer_stats_broad.tsv`, `kmer_layer_stats.tsv`, `kmer_dual_test.json`, `patterns_*.tsv`, `patterns_summary.json`, `predict_profiles.tsv`, `motif_controls.tsv`.
