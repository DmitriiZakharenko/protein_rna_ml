# Skipper / eCLIP benchmark data (Dropbox + Figshare)

Processed ENCODE eCLIP pos/neg sets and protein domain annotations for
train-in-vitro → test-eCLIP evaluation and domain masking / attribution (scripts 39–41).

**Sources**
- Dropbox (lab): pos/neg FASTA/BED, RBD annotations on eCLIP protein sequences
- [Figshare 21206009](https://doi.org/10.6084/m9.figshare.21206009): Skipper ENCODE outputs (Boyle et al., *Cell Genomics* 2023)

## Layout

```
data/raw/eclip_skipper/
  README.md
  manifests/              # small files tracked in git
  archives/dropbox/       # large tar.xz (gitignored)
  archives/figshare/      # Figshare download (~1.3 GB, gitignored)
  extracted/              # optional unpacked subsets (gitignored)
```

## Manifests (git)

| File | Role |
|------|------|
| `eclip_various_pos_neg_sets.hg38.info.tsv` | Per-experiment stats: pos/neg BED paths, RNA length modes |
| `encode_eclip_rbp_id_best_acc_seq.added_domain_annot.tsv` | **185 RBPs**: UniProt seq + InterPro domain intervals (`hit_pos`) |
| `encode_eclip_rbp_id_best_acc_seq.tsv` | Same roster without domain columns (older) |
| `encode3_manifest_{H,K}.csv` | Skipper sample manifest (Figshare) |
| `encode3_*_reference.tsv` | Skipper cross-RBP summaries (Figshare) |
| `dataset_summary.json` | Machine-readable index |

## Archives (not in git)

### Dropbox (`archives/dropbox/`)

| Archive | Contents |
|---------|----------|
| `eclip_various_pos_neg_sets.hg38.tar.xz` | Main bundle: `pos_neg_fasta_out/*.{fa,bed}` for 4 RNA-length modes |
| `new_neg_pos_renamed.tar.xz` | Updated naming; pos in `pos_renamed_region_ids_bed_fasta_out/`, negs in `new_neg_from_pos_bed_fasta_out/` |
| `new_neg_from_rep12_pos.tar.xz` | Alternative negative strategy from rep1/2 positives |
| `rmsk.no_unknown.hg38.sorted.bed.xz` | RepeatMasker hg38 (negative generation mask) |
| `ENCODE_finemapped_windows.HepG2.tar` | Skipper 75 nt finemapped windows (HepG2) |

### Figshare (`archives/figshare/`)

Full Skipper ENCODE3 panel: `reproducible_enriched_windows`, `background_windows`, finemapped, homer, GFF, etc. Use when rebuilding from genomic coordinates instead of Dropbox FASTA.

## RNA length modes (from `info.tsv`)

| Mode suffix | Approx. length | Notes |
|-------------|----------------|-------|
| `varlen_ext20` | median ~102 nt | variable, max 300 |
| `fixlen_101` | 101 nt | |
| **`fixlen_151`** | **151 nt** | **recommended** — matches RPIembeddor2 / Jose eval |
| `fixlen_201` | 201 nt | |

264 eCLIP experiments × 4 modes = 1056 rows in `info.tsv`.

## Domain annotations

`encode_eclip_rbp_id_best_acc_seq.added_domain_annot.tsv`:
- `symbol` — join to `eclip_id` prefix (`CSTF2_K562_…` → `CSTF2`)
- `protein_sequence` — canonical UniProt sequence for the benchmark
- `hit_pos` — e.g. `IPR000504:17-90` (1-based inclusive on `protein_sequence`)
- 149/185 proteins have at least one domain interval

## Extracted subset

`extracted/fixlen_151_fasta/` — 528 FASTA files (pos + neg per experiment) unpacked from
`eclip_various_pos_neg_sets.hg38.tar.xz` for fast iteration without full tar extract.

## Build benchmark TSV (script 41)

```bash
python scripts/41_build_skipper_eclip_benchmark.py \\
  --max_per_class_per_experiment 200 \\
  --train_tsv data/sanitized/generalized_v3a/train.tsv
```

Outputs (gitignored large TSVs): `data/benchmarks/skipper_eclip/`  
Summary: `results/skipper_eclip/build_summary.json`

Evaluate V2 on protein-disjoint holdout (`rna_max=151`):

```bash
python scripts/11_evaluate_external.py \\
  --benchmark_tsv data/benchmarks/skipper_eclip/fixlen_151_protein_disjoint_v3a.tsv \\
  --v2_dir models/saved/generalized_v2 --rna_max 151 --prot_max 700
```

## Unpack commands

```bash
BASE=data/raw/eclip_skipper

# All pos/neg from main Dropbox bundle
tar -xJf $BASE/archives/dropbox/eclip_various_pos_neg_sets.hg38.tar.xz

# Renamed bundle only
tar -xJf $BASE/archives/dropbox/new_neg_pos_renamed.tar.xz

# Skipper enriched windows (HepG2)
tar -xf $BASE/archives/figshare/ENCODE_reproducible_enriched_windows.HepG2.tar
```
