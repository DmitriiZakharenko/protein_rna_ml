# Domain-aware phase — kickoff after Week-1 cross-protocol

**Status**: Active  
**Prerequisite**: Cross-protocol results in `results/cross_protocol/` (complete)  
**Sources**: `DATA_SOURCES_AND_DOWNLOADS.md`, Table S1, UniProt (optional fetch)

---

## Goal

Explain (and then reduce) the cross-assay transfer gap using **domain architecture**:
construct-masked protein input, domain-type conditioning, attribution inside RBDs.

## Immediate next scripts

| Step | Script | Output |
|------|--------|--------|
| 1 | `37_annotate_protein_domains.py` | `data/domains/protein_domains.tsv` |
| 2 | Pilot: same-domain vs transfer stats | `results/domain_aware/transfer_by_domain_stats.json` |
| 3 | `38_…` construct-mask / condition V2 | checkpoints + metrics |
| 4 | `39_…` attribution enrichment | domain mass enrichment table |

## Commands (step 1)

```bash
# Table S1 only (offline, fast)
python scripts/37_annotate_protein_domains.py

# Optional: also query UniProt for roster proteins missing Table S1
python scripts/37_annotate_protein_domains.py --fetch_uniprot
```

## Success for this phase (minimal)

1. Every roster protein has domain_class or explicit `unknown`.
2. Construct intervals available for RNAcompete-overlapping proteins (Table S1).
3. One controlled experiment: full-length vs construct-masked protein encoder
   on the cross-protocol overlap set.
