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

## Scripts 41 / 41b / 41c / 41d

### 41 — pair TSV + cross-assay holdout (in vitro train → eCLIP test)

```bash
python scripts/41_build_skipper_eclip_benchmark.py \\
  --max_per_class_per_experiment 200 \\
  --train_tsv data/sanitized/generalized_v3a/train.tsv

python scripts/11_evaluate_external.py \\
  --benchmark_tsv data/benchmarks/skipper_eclip/fixlen_151_protein_disjoint_v3a.tsv \\
  --v2_dir models/saved/generalized_v2 --rna_max 151 --prot_max 700
```

### 41b — Jose-style split (train eCLIP → test eCLIP, protein-disjoint)

```bash
python scripts/41b_split_skipper_eclip_jose_style.py

python scripts/06_train_generalized_v2.py \\
  --data_dir data/benchmarks/skipper_eclip/jose_style \\
  --rna_max 151 --prot_max 700 \\
  --model_dir models/saved/skipper_eclip_v2_rna151 \\
  --out_dir results/skipper_eclip/jose_style_v2_train
```

Outputs: `data/benchmarks/skipper_eclip/jose_style/{train,val,test}.tsv`  
Summaries: `results/skipper_eclip/build_summary.json`, `jose_style_split_summary.json`

### 41c — RNA-disjoint / protein+RNA splits (unseen RNA eval)

```bash
# RNA unseen in test (proteins may repeat across splits)
python scripts/41c_split_skipper_eclip_rna_disjoint.py --mode rna

# Protein-disjoint + drop test RNAs seen in train
python scripts/41c_split_skipper_eclip_rna_disjoint.py --mode protein_and_rna

python scripts/06_train_generalized_v2.py \\
  --data_dir data/benchmarks/skipper_eclip/rna_disjoint \\
  --rna_max 151 --prot_max 700 \\
  --model_dir models/saved/skipper_eclip_v2_rna151_rna_disjoint \\
  --out_dir results/skipper_eclip/rna_disjoint_v2_train
```

### 41d — diagnostics (GC baseline, RNA-unseen subset, GC-matched neg)

```bash
# No GPU — composition baselines only
python scripts/41d_eval_eclip_diagnostics.py \\
  --train_tsv data/benchmarks/skipper_eclip/jose_style/train.tsv \\
  --test_tsv data/benchmarks/skipper_eclip/jose_style/test.tsv \\
  --out_dir results/skipper_eclip/jose_style_diagnostics

# With trained Jose-style checkpoint
python scripts/41d_eval_eclip_diagnostics.py \\
  --train_tsv data/benchmarks/skipper_eclip/jose_style/train.tsv \\
  --test_tsv data/benchmarks/skipper_eclip/jose_style/test.tsv \\
  --checkpoint models/saved/skipper_eclip_v2_rna151/best_model.pt \\
  --rna_max 151 --prot_max 700 \\
  --out_dir results/skipper_eclip/jose_style_diagnostics
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
