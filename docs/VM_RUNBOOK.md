# VM runbook

Reference for running the analyses on the GPU VM. Checked against the scripts in the repository on 2026-11-26.
Commands marked "not verified on the VM" were read in the code and the logs but not executed for this document.
Output of the VM is copied back into the chat by hand, so each block names the file or lines to paste.

## 1. Variables

| Variable | Meaning | Value |
|---|---|---|
| `HOST` | Login string of the VM (user, at sign, address). | filled in locally, not stored in the repository |
| `PORT` | SSH port. | `30021` (the earlier port 30047 is obsolete) |
| `KEY` | Path of the private SSH key on the local machine. | filled in locally, not stored in the repository |
| `VMDIR` | Project directory on the VM. | `/vol/space/protein_rna_ml` |
| `LOCAL` | Path of this repository on the local machine. | filled in locally |
| `SSH` | Prefix for ssh and rsync. | `ssh -i $KEY -p $PORT` |

```bash
# local shell, once per session
HOST=<login>
PORT=30021
KEY=<path-to-private-key>
LOCAL=<path-to-repository>
VMDIR=/vol/space/protein_rna_ml
SSH="ssh -i $KEY -p $PORT"
ssh-add "$KEY"
```

## 2. Sync

### 2.1 Code, local to VM

The VM project is a git clone. Code reaches it through `git pull`. Commits and pushes are made locally and only on request.

```bash
$SSH $HOST
cd /vol/space/protein_rna_ml && git pull
git log --oneline -3          # the top commit should be the one pushed last
git status -sb | head -5      # untracked results folders are expected
```

A single file without a commit:

```bash
scp -i $KEY -P $PORT "$LOCAL/scripts/<file>" $HOST:$VMDIR/scripts/
```

`git pull` stops if an untracked VM file has the same path as a file in the pulled commit. In that case the VM file is copied aside by hand before pulling.

### 2.2 Results, VM to local

A dry run (`-n`) first. It lists files and copies nothing. The second command without `-n` copies.

```bash
rsync -avPn -e "$SSH" $HOST:$VMDIR/results/<folder>/ "$LOCAL/results/<folder>/"
rsync -avP  -e "$SSH" $HOST:$VMDIR/results/<folder>/ "$LOCAL/results/<folder>/"
scp -i $KEY -P $PORT $HOST:$VMDIR/results/<folder>/<file> "$LOCAL/results/<folder>/"
```

Per-run `results.json` files of a runner (a few kB each):

```bash
rsync -avPn --include='*/' --include='results.json' --exclude='*' -e "$SSH" \
  $HOST:$VMDIR/results/v3b_frozen_layers/ "$LOCAL/results/v3b_frozen_layers/"
```

Notes:
- Embeddings (`data/embeddings_*`), checkpoints (`models/saved/`) and `scores_*.tsv.gz` stay on the VM. They are not tracked by git.
- `results/esm_layer_profile/*.npz` is not git-ignored. Copying the folder into the repository and running `git add` on it would add large arrays.
- An scp from the local machine to the VM timed out once. `git pull` is the usual path for code.
- Some local input folders (`rbns_analysis/`, `rnacompete_analysis/`, parts of `data/`) can hold iCloud placeholders that hang on `cat` or `scp`.

## 3. Environment and checks (read-only)

The current environment is `/vol/space/venv_emb` (Python 3.11, torch 2.6 with CUDA, transformers, multimolecule, fair-esm, scipy, matplotlib). The conda environment `prna` is older and cannot import multimolecule.

```bash
source /vol/space/venv_emb/bin/activate
which python
python -c "import torch,esm,multimolecule,scipy,matplotlib;print(torch.__version__,torch.cuda.is_available())"   # expected: 2.6.x True
nvidia-smi | head -12                                                           # expected: Tesla P100
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv                    # process holding GPU memory
df -h /vol/space | tail -1
du -sh data/embeddings_v3b* 2>/dev/null
pgrep -fa "run_v3b|66_embed|67_frozen|80_esm|81_rnafm|06_train|21_train|51_eclip" | cut -c1-160
tail -n 5 results/<name>.log
tail -f results/<name>_runner.log          # Ctrl-C ends tail only, a nohup job continues
cat results/v3b_overnight/status.txt       # runner with a status file
```

Starting a background job:

```bash
PYTHON=python nohup bash scripts/<runner>.sh > results/<name>_runner.log 2>&1 &
setsid nohup bash scripts/run_v3b_overnight.sh > results_v3b_nohup.log 2>&1 < /dev/null &    # survives a disconnect
```

A running job that holds GPU memory blocks an embedding job (12.5 GB of the 16 GB P100 were held by one V2 training on 2026-10-30). The memory query above is run before every GPU job.

## 4. Runners in `scripts/`

All runners start with `cd "$(dirname "$0")/.."`, so they run from any folder. Paths are relative to `$VMDIR`. Data files are not in git, so their presence on the VM is "not verified on the VM" in every row. Archived runners (`run_cross_protocol.sh`, `run_overnight_multiseed.sh`, `run_v3a_shuffle_check.sh`, `run_pipeline.sh`) are in `scripts/archive/` and are not listed.

| Runner | Purpose | Inputs | Variables read | Outputs | Resume |
|---|---|---|---|---|---|
| `run_v3b_frozen.sh` | Frozen ESM-2 and RNA-FM heads (script 67) on v3b, per-row scores saved. | `data/generalized_v3b/{train,val,test}.tsv`, `data/embeddings_v3b/{prot_esm2,rna_rnafm}.npz`, `data/benchmarks/v3b_cross_neg/cross_neg_probe/test.tsv`, `results/v3b/protein_split_map.tsv`, `results/homology_audit/v3b_test_protein_identity.tsv` | `PYTHON` (python3), `SEEDS` ("42 0 1"), `CONFIGS` ("bilinear:xneg bilinear:std rna:std protein:std concat:std", format head:mode) | `results/v3b_frozen/<head>_<mode>_s<seed>/` (results.json, scores_probe.tsv.gz, scores_test_standard.tsv.gz, head.pt), logs `results/v3b_frozen_<head>_<mode>_s<seed>.log` | Skips a run whose `results.json` exists |
| `run_v3b_layers.sh` | Bilinear heads with the protein embedding from other ESM-2 layers (script 66 `--prot_layer`). | as above, plus `data/embeddings_v3b/rna_rnafm.npz` (copied into each layer folder) and `fair-esm` | `PYTHON`, `LAYERS` ("0 22 26 32"), `SEEDS` ("42 0 1"), `CONFIGS` ("bilinear:std bilinear:xneg") | `data/embeddings_v3b_L<L>/prot_esm2.npz`, `results/v3b_frozen_layers/L<L>_<head>_<mode>_s<seed>/`, logs `results/v3b_layers_embed_L<L>.log`, `results/v3b_layers_L<L>_<head>_<mode>_s<seed>.log` | Skips the embedding if `prot_esm2.npz` exists and a run if its `results.json` exists |
| `run_v3b_tf.sh` | Frozen heads with TF ChIP-seq rows added to training. | v3b inputs, `data/benchmarks/tf_chipseq/pairs.tsv` (script 76), `data/embeddings_tf/` (script 66 `--tsv`) | `PYTHON`, `SEEDS`, `CONFIGS` ("bilinear:std bilinear:xneg"), `EXTRA_FRAC` (1.0) | `results/v3b_frozen_tf/<head>_<mode>_s<seed>/`, logs `results/v3b_frozen_tf_<head>_<mode>_s<seed>.log` | Skips a run whose `results.json` exists |
| `run_v3b_diagnostics.sh` | Seen-protein control, learning curve over training clusters, hyper-parameter sweep (bilinear, xneg unless named). | as `run_v3b_frozen.sh` | `PYTHON` | `results/v3b_diag/<name>/results.json`, logs `results/v3b_diag_<name>.log`. Names: seen_probe, seen_probe_std, clusters_0.25/0.5/0.75 (and `_s0`), lr3e-4, lr3e-3, hid128, hid512, drop0.1, drop0.5, wd1e-3, wd1e-1 | Skips a run whose `results.json` exists |
| `run_v3b_overnight.sh` | Builds v3b data (script 58 `build`) and the probe (script 53), trains V2 then V4 per seed, probes (script 51), aggregates (script 54). | `results/v3b/protein_split_map.tsv` (the runner exits if it is missing and never recreates it), a Python with torch, sklearn, pandas | `PYTHON`, `SEEDS` ("42 0 1"), `EPOCHS` (60), `MAX_POS` (40000) | `data/generalized_v3b/`, `data/benchmarks/v3b_cross_neg/`, `models/saved/v{2,4}_v3b_seed<s>/best_model.pt`, `results/v3b_overnight/{status.txt,<stage>.log,probe_after_seed<s>/,probe_final/,aggregate.json}` | Skips training if the checkpoint exists. Build, probe and aggregate stages rerun. A failed stage is logged FAILED and the pipeline continues |
| `run_v3b_mechanistic.sh` | Motif recovery and in silico mutagenesis (script 61) for existing v3b checkpoints. | `results/v3b/protein_split_map.tsv`, `results/homology_audit/v3b_test_protein_identity.tsv`, `results/mechanistic/v3b_domain_positions.tsv`, `data/generalized_v3b/test.tsv`, `mmseqs` on PATH for the nearest-neighbour baseline | `PYTHON` (default is the old `prna` interpreter, so `PYTHON=python` is passed), `SEEDS` ("42 0 1") | Written by script 61 with `--name v3b` (exact files not verified) | None |
| `run_v3b_report.sh` | Comparison tables with bootstrap intervals (script 69) for the v3b probe and standard test, from whatever score files exist. | `results/knn_baseline/v3b_*_scores.tsv.gz`, `results/v3b_overnight/probe_after_seed42/scores_cross_neg_probe.tsv.gz`, `results/v3b_frozen/<head>_<mode>_s42/scores_*.tsv.gz`, optional `results/v3b_hybrid/` | `PYTHON`, `N_BOOT` (300), `OUT_PREFIX` (empty, prefix of the output file names) | Tables on stdout (cut to 400 characters), files in `results/v3b_report/` | None. A different `OUT_PREFIX` keeps an earlier table |
| `run_eclip_step1.sh` | Rebuilds the eCLIP benchmarks, baselines, trains V2 on two splits, evaluation suite (script 51). | v3a train table, `models/saved/generalized_v2/best_model.pt`, `models/saved/generalized_v4_phase3a/best_model.pt` | `PYTHON`, `STAGES` ("build baselines train eval"), `TRAIN_SPLITS`, `V3A_TRAIN_TSV`, `V2_V3A_CKPT`, `V4_V3A_CKPT`, `EPOCHS` (40), `BATCH` (256) | `data/benchmarks/skipper_eclip/`, `models/saved/skipper_eclip_v2_<split>/`, `results/skipper_eclip/{step1_baselines,step1,v2_train_<split>}/` | By stage only. The `build` stage rebuilds the splits (`--rebuild`) and counts as a write |
| `run_eclip_finetune.sh` | Fine-tunes the v3a V2 on two eCLIP splits at lr 1e-4, then evaluates with the shuffle control. | eCLIP splits from `run_eclip_step1.sh`, the V2 checkpoint, `data/sanitized/generalized_v3a/train.tsv` | `PYTHON`, `V2_V3A_CKPT` | `models/saved/skipper_eclip_v2_ft_<split>/`, `results/skipper_eclip/{v2_ft_<split>,step1_finetune}/` | None |
| `run_cross_neg_diag.sh` | Diagnostics of the cross-protein-negative failure (seen-protein split, V2 at lr 1e-4, V4 `concat_bi`). | eCLIP splits | `PYTHON` | `data/benchmarks/skipper_eclip/cross_protein_neg_seen_protein/`, `models/saved/skipper_eclip_{v2_cpn_seen,v2_cpn_lr1e4,v4_cpn}/`, `results/skipper_eclip/{v2_train_cpn_seen,v2_train_cpn_lr1e4,v4_train_cpn}/` | None |
| `run_invitro_cross_probe.sh` | Builds the in vitro cross-protein probe (script 53) and evaluates V2 and V4 (script 51). | v3a data, V2 and V4 checkpoints | `PYTHON`, `MAX_POS` (40000), `V2_V3A_CKPT`, `V4_V3A_CKPT` | `data/benchmarks/v3a_cross_neg/`, `results/skipper_eclip/step1_invitro_cross_probe/` | None |

Run times recorded in the log: ESM-2 embedding of the 175 RNAcompete constructs 31 s (script 80). One frozen head run about 3 min (one pair of timestamps, 16:10 and 16:13). RNA-FM embedding of 867,522 RNAs 1,092 s and of 559,239 TF-table RNAs about 2,840 s (script 66). V4 with cross-protein negatives 381 s per epoch. No run time is recorded for the other runners.

The `.sh` runners that use `set -euxo pipefail` (`run_eclip_finetune.sh`, `run_cross_neg_diag.sh`, `run_invitro_cross_probe.sh`) print each command, so `grep -E "^\+ .*python" <log>` lists the invocations.

## 5. Pending experiments

Before each block: `cd /vol/space/protein_rna_ml && git pull && source /vol/space/venv_emb/bin/activate`. The top commit should include scripts 80 and 81 (committed in 6a4e248).

### 5.1 ESM-2 pooling over RNA-binding domains and outside them (script 80)

Verified in the script: `embed` reads `results/esm_layer_profile/experiments.tsv` and, when present, `domains.tsv`, and writes `esm2_layers.npz` with the arrays `emb`, `emb_domain`, `emb_outside`, `seqs`. `analyze --pool domain` and `--pool outside` read `emb_domain` and `emb_outside`. The file from the earlier full-sequence run does not have them, so `embed` is rerun first. `domains.tsv` is tracked in git (hmmscan was run locally), so `hmmscan` is not needed on the VM.

Read-only checks:

```bash
ls -la results/esm_layer_profile/
wc -l results/esm_layer_profile/domains.tsv
python -c "import numpy as np; z=np.load('results/esm_layer_profile/esm2_layers.npz',allow_pickle=True); print(z.files, z['emb'].shape)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Expected: `experiments.tsv`, `domains.tsv`, `Zscores_RNAcompete2025.txt.gz`, `layer_bootstrap.tsv` present. The old npz lists `['emb', 'seqs']` with shape `(175, 34, 1280)`. GPU memory used is a few MiB.

Writing steps (the old file is replaced, so it is copied first):

```bash
cp -n results/esm_layer_profile/esm2_layers.npz results/esm_layer_profile/esm2_layers_full_only.npz
nohup python -u scripts/80_esm_layer_profile_similarity.py embed > results/esm_layer_embed2.log 2>&1 &
tail -n 3 results/esm_layer_embed2.log        # wait for: saved 175 proteins x 34 layers
```

After that line appears:

```bash
python scripts/80_esm_layer_profile_similarity.py analyze --pool domain  --n_perm 200 --zscores results/esm_layer_profile/Zscores_RNAcompete2025.txt.gz > results/esm_layer_domain.log 2>&1
python scripts/80_esm_layer_profile_similarity.py analyze --pool outside --n_perm 200 --zscores results/esm_layer_profile/Zscores_RNAcompete2025.txt.gz > results/esm_layer_outside.log 2>&1
tail -n 3 results/esm_layer_domain.log results/esm_layer_outside.log
```

Argparse names of script 80: `stage` (meta, domains, embed, analyze, bootstrap), `--pairs`, `--zscores`, `--out_dir` (results/esm_layer_profile), `--n_perm` (500), `--n_boot` (200), `--pool` (full, domain, outside), `--pfam`, `--seed` (42). The earlier full-pool table used 500 permutations, so `--n_perm 500` gives the same resolution (smallest p about 0.002) and `--n_perm 200` about 0.005. The `bootstrap` stage ignores `--pool`.

Paste back: `cat results/esm_layer_profile/layer_correlation_domain.tsv`, `cat results/esm_layer_profile/layer_correlation_outside.tsv`, and the line "domain pooling: N of 175 proteins usable" from each log. Output files: `layer_correlation_domain.tsv`, `layer_correlation_outside.tsv`, `fig_layer_correlation_domain.png`, `fig_layer_correlation_outside.png`.

### 5.2 RNA-FM layers (script 81)

Verified in the script: `subset` was run locally and `results/rna_layer_profile/probes.tsv` and `profiles.npz` are tracked in git. The VM runs `embed` (RNA-FM `multimolecule/rnafm` from HuggingFace, hidden states of all layers, mean over nucleotides) and `analyze`. `analyze` reads `profiles.npz`, `rnafm_layers.npz` and `probes.tsv`.

Read-only checks:

```bash
ls -la results/rna_layer_profile/
python -c "import multimolecule; print('multimolecule ok')"
ls ~/.cache/huggingface/hub 2>/dev/null | grep -i rnafm
```

Expected: `probes.tsv` (56,088 bytes) and `profiles.npz` (751,067 bytes). An empty cache listing means the first run downloads the weights (network needed).

Writing steps:

```bash
nohup python -u scripts/81_rnafm_layer_profile_similarity.py embed > results/rna_layer_embed.log 2>&1 &
tail -n 3 results/rna_layer_embed.log         # expected end: saved 1000 probes x 13 hidden states
python scripts/81_rnafm_layer_profile_similarity.py analyze > results/rna_layer_analyze.log 2>&1
tail -n 20 results/rna_layer_analyze.log
```

Argparse names of script 81: `stage` (subset, embed, analyze), `--scores`, `--experiments`, `--out_dir` (results/rna_layer_profile), `--n_probes` (1000), `--batch` (64), `--n_boot` (100), `--seed` (42).
Paste back: `cat results/rna_layer_profile/layer_correlation.tsv` and `tail -n 20 results/rna_layer_analyze.log`. The column names of that table were not read, so they are not verified. The end line of the embed log is taken from the session notes and not from the script text.

### 5.3 Frozen heads with ESM-2 layers 0, 22, 26, 32 (`run_v3b_layers.sh`)

The runner was started once (log lines `=== layer 0 bilinear std seed 42` and `=== layer 0 bilinear xneg seed 42`). Whether it finished is not verified on the VM.

Read-only checks:

```bash
pgrep -fa "run_v3b_layers|66_embed|67_frozen" | cut -c1-140      # expected: nothing if no job is running
tail -n 8 results/v3b_layers_runner.log                          # expected: "done" at the end if finished
ls -d data/embeddings_v3b_L*
ls results/v3b_frozen_layers 2>/dev/null | wc -l                 # expected: 24 when complete
ls data/embeddings_v3b/                                          # expected: prot_esm2.npz and rna_rnafm.npz
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Start (safe to repeat, finished runs are skipped):

```bash
PYTHON=python LAYERS="0 22 26 32" SEEDS="42 0 1" CONFIGS="bilinear:std bilinear:xneg" nohup bash scripts/run_v3b_layers.sh > results/v3b_layers_runner.log 2>&1 &
```

With these values the runner makes 4 embeddings and 24 head runs. Script 66 argparse names used by it: `--stage`, `--prot_layer`, `--out_dir` (required together with `--tsv`, default data/embeddings_v3b otherwise), `--data_dir`, `--probe_tsv`, `--rna_batch`, `--rna_backend` (hf, fm), `--tsv`, `--limit`. Script 67 names used: `--head` (rna, protein, concat, bilinear), `--train_mode` (std, xneg, rank, rank_prot), `--seed`, `--emb_dir`, `--out_dir`, plus `--epochs`, `--steps_per_epoch`, `--batch`, `--lr`, `--val_pos`, `--patience`, `--hidden`, `--dropout`, `--wd`, `--split_map`, `--identity_tsv`, `--data_dir`, `--probe_tsv`, `--cluster_frac`, `--seen_probe_frac`, `--rank_weight`, `--extra_tsv`, `--extra_emb_dir`, `--extra_frac`.
The runner has no `set -e`. A failed embedding shows up as failed head runs, so `results/v3b_layers_embed_L<L>.log` is read first.
References at layer 33 (probe AUROC, mean +- sd over seeds 42, 0, 1): std 0.531 +- 0.027, xneg 0.559 +- 0.009. The decision rule in EXPERIMENT_LOG.md is a mean above the layer-33 value by more than the seed spread.

## 5a. ENCODE protein map (script 82)

Embeddings and 5-mer profiles are computed on the VM, statistics and the interactive page locally. The protein table `results/encode_map/proteins.tsv` (1,051 proteins) is in the repository.

```bash
cd /vol/space/protein_rna_ml && git pull && source /vol/space/venv_emb/bin/activate
python scripts/82_encode_protein_map.py embed      # ESM-2 650M, 34 layers, about 1,051 proteins, writes results/encode_map/esm2_layers.npz (about 90 MB)
python scripts/82_encode_protein_map.py kmer       # 5-mer profiles from data/benchmarks/tf_chipseq/pairs.tsv and data/benchmarks/skipper_eclip/fixlen_151_all.tsv
ls -la results/encode_map/                         # expected: esm2_layers.npz and kmer_profiles.npz
```

Copy both files to the local repository (dry run with `-n` first), then run locally:

```bash
rsync -avn -e "$SSH" "$HOST:$VMDIR/results/encode_map/esm2_layers.npz" "$HOST:$VMDIR/results/encode_map/kmer_profiles.npz" "$LOCAL/results/encode_map/"
python scripts/82_encode_protein_map.py stats --chip_def broad
python scripts/82_encode_protein_map.py kmerstats --chip_def broad
python scripts/82_encode_protein_map.py map --chip_def broad
python -m http.server 8765 --bind 127.0.0.1 --directory results/encode_map   # then open http://127.0.0.1:8765/encode_protein_map.html
```

Run times of `embed` and `kmer` are not recorded. Expected end of `embed`: a line `saved N proteins x 34 layers`. The columns of the two input tables (`protein_name`, `rna_sequence`, `binding_label`, `neg_type`) are read from the code and not verified on the VM.

## 6. Collecting results

Table of probe and standard-test AUROC (reads `results.json` only, keys `probe.auroc`, `test_standard.auroc`, `best_epoch`). Not verified on the VM.

```bash
python3 -I - <<'EOF'
import glob, json, re, statistics as st
rows = {}
for f in sorted(glob.glob('results/v3b_frozen_layers/*/results.json')):
    m = re.match(r'(L\d+)_(\w+?)_(std|xneg)_s(\d+)$', f.split('/')[-2])
    r = json.load(open(f))
    rows.setdefault((m.group(1), m.group(3)), []).append((int(m.group(4)), r['probe']['auroc'], r['test_standard']['auroc']))
sd = lambda a: st.stdev(a) if len(a) > 1 else float('nan')
for k, v in sorted(rows.items(), key=lambda x: (int(x[0][0][1:]), x[0][1])):
    p = [x[1] for x in v]
    t = [x[2] for x in v]
    print(k, 'n=%d' % len(v), 'probe %.3f +- %.3f' % (st.mean(p), sd(p)), 'standard %.3f +- %.3f' % (st.mean(t), sd(t)), 'seeds', [x[0] for x in v])
EOF
```

The layer-33 reference uses the same snippet with the glob `results/v3b_frozen/bilinear_*_s*/results.json` and the regex `(bilinear)_(std|xneg)_s(\d+)$` (group numbers adjusted).

Completeness and consistency checks:

```bash
for L in 0 22 26 32; do for s in 42 0 1; do for m in std xneg; do echo L${L}_bilinear_${m}_s${s}; done; done; done | sort > /tmp/want.txt
ls results/v3b_frozen_layers | sort > /tmp/have.txt
comm -13 /tmp/have.txt /tmp/want.txt                              # runs still missing, empty when complete
for d in results/v3b_frozen_layers/*/; do [ -f $d/results.json ] || echo "no results.json: $d"; done
grep -il "error\|traceback" results/v3b_layers_*.log
python3 -I -c "import numpy as np; a=np.load('data/embeddings_v3b/prot_esm2.npz'); b=np.load('data/embeddings_v3b_L22/prot_esm2.npz'); print(a['emb'].shape, b['emb'].shape, (a['names']==b['names']).all())"   # same shape, True
```

What to paste back:

| Experiment | Paste |
|---|---|
| Domain and outside pooling | the two `layer_correlation_*.tsv` files, the "usable" line of each log |
| RNA-FM layers | `layer_correlation.tsv`, `tail -n 20 results/rna_layer_analyze.log` |
| Frozen heads with layers | output of the table snippet, `tail -n 8 results/v3b_layers_runner.log` |
| v3b overnight | `cat results/v3b_overnight/status.txt`, `python scripts/54_aggregate_probe.py --summary results/v3b_overnight/probe_final/step1_summary.json` |
| eCLIP runners | `results/skipper_eclip/step1*/step1_summary.json` |
| Report runner | stdout of `run_v3b_report.sh` |

To bring small result files to the local machine, use the dry-run pattern in section 2.2.

## 7. Common mistakes

- Starting `analyze` before `embed` has finished gives FileNotFoundError. The embed log is checked for its final "saved" line first.
- Rerunning script 80 `embed` replaces `esm2_layers.npz`. The file is copied first (section 5.1).
- Embedding and training jobs compete for the 16 GB GPU. `nvidia-smi` and `pgrep` are run before a start.
- The runners with `set -u` or without `set -e` continue after a failed step. Per-run logs are read, not only the runner log.
- `run_v3b_mechanistic.sh` defaults to the `prna` interpreter. `PYTHON=python` is passed with the venv active.
- RNA-FM needs the venv `venv_emb`. The original RNA-FM weight URL returned 403, so script 66 uses `--rna_backend hf` (the default).
- `fair-esm` is imported as `esm` and is missing from older environments (`pip install fair-esm`, run only after approval).
- conda channels were unreachable from the VM and pip worked (log, 2026-10-30).
- The VM host key changed once, and the disk content may have differed. The checks in section 8 are run after a reconnect.
- The `build` stage of `run_eclip_step1.sh` rebuilds splits. eCLIP tables built before the "validate reason" fix carry `protein_sequence == "ok"` and are rejected by scripts 41 to 42.
- `scripts/58_build_v3b_cluster_split.py split` refuses to overwrite `results/v3b/protein_split_map.tsv` without `--force`. `run_v3b_overnight.sh` never calls `split` and exits when the map is missing.
- `scripts/66_embed_v3b_frozen.py` with `--tsv` requires `--out_dir`.
- Test sets are evaluated with the model chosen on validation (lessons.md, script 03 case). One seed is not a basis for a layer decision.
- Mantel p-values have a floor of 1/(n_perm + 1).
- Commits and pushes are made locally on request. The VM copy is updated by `git pull`.

## 8. Checks on the VM before the first run (read-only)

```bash
cd /vol/space/protein_rna_ml && git status -sb | head -3 && git log --oneline -3
ls -la data/generalized_v3b/ data/embeddings_v3b/ data/benchmarks/v3b_cross_neg/cross_neg_probe/
ls -la results/v3b/protein_split_map.tsv results/homology_audit/v3b_test_protein_identity.tsv
ls -la models/saved/generalized_v2/best_model.pt models/saved/generalized_v4_phase3a/best_model.pt models/saved/v2_v3b_seed42/best_model.pt models/saved/v4_v3b_seed42/best_model.pt
ls ~/.cache/torch/hub/checkpoints 2>/dev/null | grep -i esm2
ls ~/.cache/huggingface/hub 2>/dev/null | grep -i rnafm
```

Expected: `train.tsv`, `val.tsv`, `test.tsv` in `data/generalized_v3b/`, `prot_esm2.npz` (1280 values per protein) and `rna_rnafm.npz` (640 values per RNA) in `data/embeddings_v3b/`, a `test.tsv` in the probe folder, and the listed files. An empty cache listing means the weights are downloaded on first use.

Not verified on the VM:
- presence and content of all data folders and checkpoints above (they are not in git)
- whether `run_v3b_layers.sh` finished and how long it ran
- the state of `results/esm_layer_profile/esm2_layers.npz` (old file without `emb_domain` expected)
- the column names of `results/rna_layer_profile/layer_correlation.tsv`
- the output files of `run_v3b_mechanistic.sh`
- the table snippet in section 6
- whether the weight caches exist, and network access for first downloads

## Appendix. Lint

Rule: every script or command handed to the cluster passes `python3 /<wiki>/.claude/scripts/preflight_lint.py <file>` (usage: `preflight_lint.py <file-or-folder> [...]`, exit status 1 if an ERROR is found). The linter was run with `python3 -I` on each `scripts/run_*.sh` on 2026-11-26. It was not run on anything else. Its rules (P001 to P010) were written for the PMCN cluster. Only P006 (no `set -e`), P009 (hard-coded local path) and P010 (no date) apply to these files. It does not check Python commands, VM paths or GPU use. The inline blocks of this document are not files and were not linted.

Result: 11 files, 0 ERROR, 7 WARN (P006), 10 INFO (P010). All exit statuses 0.

```
### scripts/run_cross_neg_diag.sh
-- 1 files, 0 errors, 0 warnings, 1 info
scripts/run_cross_neg_diag.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_eclip_finetune.sh
-- 1 files, 0 errors, 0 warnings, 1 info
scripts/run_eclip_finetune.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_eclip_step1.sh
-- 1 files, 0 errors, 0 warnings, 1 info
scripts/run_eclip_step1.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_invitro_cross_probe.sh
-- 1 files, 0 errors, 0 warnings, 1 info
scripts/run_invitro_cross_probe.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_v3b_diagnostics.sh
-- 1 files, 0 errors, 1 warnings, 1 info
scripts/run_v3b_diagnostics.sh:1: WARN P006 no 'set -e' / 'set -euo pipefail'
scripts/run_v3b_diagnostics.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_v3b_frozen.sh
-- 1 files, 0 errors, 1 warnings, 1 info
scripts/run_v3b_frozen.sh:1: WARN P006 no 'set -e' / 'set -euo pipefail'
scripts/run_v3b_frozen.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_v3b_layers.sh
-- 1 files, 0 errors, 1 warnings, 0 info
scripts/run_v3b_layers.sh:1: WARN P006 no 'set -e' / 'set -euo pipefail'
### scripts/run_v3b_mechanistic.sh
-- 1 files, 0 errors, 1 warnings, 1 info
scripts/run_v3b_mechanistic.sh:1: WARN P006 no 'set -e' / 'set -euo pipefail'
scripts/run_v3b_mechanistic.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_v3b_overnight.sh
-- 1 files, 0 errors, 1 warnings, 1 info
scripts/run_v3b_overnight.sh:1: WARN P006 no 'set -e' / 'set -euo pipefail'
scripts/run_v3b_overnight.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_v3b_report.sh
-- 1 files, 0 errors, 1 warnings, 1 info
scripts/run_v3b_report.sh:1: WARN P006 no 'set -e' / 'set -euo pipefail'
scripts/run_v3b_report.sh:1: INFO P010 no date in the first 5 lines
### scripts/run_v3b_tf.sh
-- 1 files, 0 errors, 1 warnings, 1 info
scripts/run_v3b_tf.sh:1: WARN P006 no 'set -e' / 'set -euo pipefail'
scripts/run_v3b_tf.sh:1: INFO P010 no date in the first 5 lines
```
