# ucRBP proteins that failed reproducibility QC (corrected)

**Paper**: Ray & Laverty et al., *Sci. Rep.* 13:5238 (2023)  
**Table S1**: `data/raw/ucrbp_paper/S1.RNAcompete_Experiment_Info.xlsx`  
(from http://datah.ccbr.utoronto.ca/ucRBP/)

## Definition
- Start from Supplementary Table S1 rows with `Experiment Category == ucRBP`
- Exclude the **23 Fig. 2 sequence-specific successes** (`configs/ucrbp_23_reproducible.txt`)
- Export **one full-length construct sequence per gene** (sanitized)

## Counts
- Paper text: **492** ucRBPs assayed → **23** successes
- S1 unique `ucRBP` genes: **485**
- Failed exported: **469** (= panel genes − Fig.2 overlaps present in panel)
- Fig.2 23 sequences also exported for reference

## Sequence cleaning
- Trailing stop `*` removed
- Digits / non-standard AA removed (strict 20 AA alphabet)
- All failed sequences now `valid_strict=True`
- Paper `Amino Acid Length` matches cleaned length for all failed rows: **True**

## Caveats
- Failed reproducibility ≠ proven non-binder
- S1 has 485 unique ucRBP gene names vs 492 in the text (small discrepancy)
- ILF2 is Fig.2 success but S1 Manual Call = Fail for its ucRBP row


## Negative tiers

| Tier | n | Meaning | File |
|------|---|--------|------|
| **Hard negatives** | 422 | No replicate / no Pass signal in S1 | `ucrbp_failed_hard_negatives.fasta` |
| **Soft / weak signal** | 47 | Had replicate or Pass/Uncertain but **not** Fig.2 reproducible | `ucrbp_failed_soft_signal.fasta` |
| All failed | 469 | Panel − Fig.2 | `ucrbp_failed_reproducibility_proteins.fasta` |

Soft ≠ proven binders; they had *some* assay signal that did not become a Fig.2 success.

## Domain / ortholog / truncation constructs

Not main negatives. Follow-up constructs from the paper.

| File | Content |
|------|---------|
| `ucrbp_domain_ortholog_constructs.fasta` | sanitized sequences; header has parent / Fig.2 anchors |
| `ucrbp_domain_ortholog_constructs.tsv` | full metadata |
| `ucrbp_construct_to_parent_links.tsv` | gene ↔ parent_gene ↔ fig2_anchors |

**How linkage works**
- **Truncation test**: `parent_gene` = same `GeneName` (domain slice of that protein)
- **Domain test**: tested gene = `GeneName`; `family` from category; `related_fig2_anchors` = Fig.2 proteins in that domain family (e.g. CCHC-zf → CNBP)
- **Ortholog test**: `parent_gene` parsed from insert note (“identity to hSERBP1”) or family default
- **Control Protein**: self

Unique constructs: **105** (Pass=25, Fail=80).
