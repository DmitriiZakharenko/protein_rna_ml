# Project inventory

Inventory of the repository as of 2026-11-26 (HEAD `6a4e248`, 138 commits between 2026-04-29 and 2026-11-19). The inventory was made by reading the file tree, the documents, `git ls-files`, a `py_compile` run over all scripts and a check of relative links in the Markdown files. Nothing was moved or deleted. Section 6 lists the proposed archive and deletion steps, which wait for approval.

## 1. What the project is

Prediction of RNA-protein binding from the protein sequence and the RNA sequence, evaluated on proteins that are not in the training set. Training and evaluation data come from in vitro assays (HTR-SELEX, RBNS, RNAcompete), in vivo eCLIP (Skipper) and, as an extension, ENCODE TF ChIP-seq. Current state of the science: [`README.md`](README.md) (status table) and [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (every run with its decision rule).

## 2. Layout and size

| Path | Tracked in git | Size on disk | Content |
|---|---|---|---|
| `scripts/` | yes (109 files) | 1.4 MB | numbered analysis and training scripts 01-81, runner `.sh` files, report builders |
| `src/` | yes (22 files) | 0.2 MB | data loaders, splits, negative sampling, models (MLP, CNN V2, interaction V4) |
| `configs/` | yes (8) | small | YAML for validation, cross-protocol and benchmark runs |
| `results/` | 1,202 files (945 in `top_bottom_examples/`) | 24 MB | metrics (JSON/TSV), logs, small score tables |
| `figures/` | yes (22) | small | figures used in the README |
| `reports/` | yes (2) | small | `results.html`, `esm_layer_similarity.html` |
| `data/` | 21 small files | 2.7 GB | raw, sanitized and benchmark data (benchmarks 1.1 GB, raw 0.8 GB, sanitized 0.7 GB, external 0.1 GB); large parts are ignored by `.gitignore` and are rebuilt by scripts |
| `models/` | no | small | local checkpoints (ignored) |
| `tasks/` | no (ignored) | small | local notes: `todo.md`, `lessons.md`, `TASKS_AND_DATASETS.md`, `NEXT_SESSION_PROMPT.md` (all last updated 2026-07/08, outdated) |
| `tmp/` | no (ignored) | 0.1 MB | two folders with RNAcompete spectra for a collaborator |
| `htr_selex_analysis/`, `htr_selex_prjeb47428/`, `rbns_analysis/`, `rnacompete_analysis/` | no, each is its own git repository (nested `.git`) | 0.4 MB | upstream data-processing repositories; README and `DATA_SOURCES_AND_DOWNLOADS.md` call them "sibling repos" at `../`, but here they sit inside this folder |
| `share_tf/` | no | 20 KB | ENCODE TF ChIP-seq manifest, FASTA and letter to a collaborator (see its `final/`) |
| `reference_letters/` | no | 64 KB | evidence file and drafts of reference letters, contain personal data |
| `.git` | | 14 MB (pack 18.9 MiB) | |

The GPU VM keeps its own copy of the data and the checkpoints (`/vol/space/protein_rna_ml`, environment `/vol/space/venv_emb`); the Mac copy is the code and the results.

## 3. Scripts by group

All 110 Python files compile (`py_compile`, Python 3). Many scripts have no docstring; the purpose below comes from `EXPERIMENT_LOG.md` and the file contents.

| Group | Scripts | Purpose |
|---|---|---|
| Phase 1 validation | 01-03, `01_prepare_htr_selex`, `02b` | per-dataset k-mer baselines with protein-aware splits |
| Phase 2 models | 04-10, 15, 16, 19 | generalized dataset build, V1 MLP, V2 CNN, V3/V3b/V3c with ESM-2, analysis of training and predictions |
| Data download | 12, 13, 14, 42b, 43 | eCLIP, RNAInter, new assays |
| Phase 3A/3B | 17, 18, 20-22a, 23, 24-32 | RNAcompete benchmark and training set, multi-seed runner, V4 interaction model, v3a build, external benchmark, figures, RNA-only baselines |
| Cross-protocol and domains | 33-38, 40 | cross-protocol roster, classifiers, motif concordance, Pfam annotation, domain-conditioned V2 (null), sequence sanitation |
| Skipper eCLIP | 41-41d, 42, 44-52 | benchmark build, protein-disjoint and RNA-disjoint splits, cross-protein negatives, enrichment and motif QC, evaluation suite |
| In vitro probe and homology | 53, 54, 55, 56, 57, 58, 62 | cross-protein-negative probe, MMseqs2 homology audit, family inventory, v3b cluster split, split integrity check |
| Baselines and heads on v3b | 59, 65-67, 69-71 | nearest-protein baseline, V4 with cross-protein negatives, frozen ESM-2/RNA-FM embeddings and heads, scores with CI, hybrid |
| Mechanistic | 61, 68, 75 | motif recovery and in silico mutagenesis, attribution maps and figures |
| Benchmark audits | 60, 63, 64, 78, 79 | shortcut audit of RPI benchmarks, PaRPI audit, CORAL fold audit and converter |
| Affinity and profiles | 72, 73, 77, 80, 81 | affinity-style probe, motif-profile regression, profile similarity, ESM-2 and RNA-FM layer tests |
| Extensions | 74, 76 | eCLIP to in vitro reverse transfer, ENCODE TF ChIP-seq pairs |
| Runners | `run_*.sh` (13) | unattended VM pipelines (multi-seed, v3b overnight, mechanistic, report, TF, layers) |
| Reports | `build_report_draft.py`, `build_esm_layer_report.py`, `summarize_v3b_runs.py` | HTML reports and run summaries |

Script number 39 does not exist; the numbering is chronological, not by topic.

## 4. Documents

| Document | Last changed | Status | Action proposed |
|---|---|---|---|
| `README.md` | 2026-10-29, updated 2026-11-26 | current after this update | keep |
| `EXPERIMENT_LOG.md` | 2026-11-19 | current, 1,266 lines | keep; three headings (TF, CORAL, ESM-2 layers) still say "planned/built, not yet run" although results follow below |
| `METHODS.md` | header says 2026-06-12 | covers phases 1-3A only; v3b, probe, frozen heads are in the log only | update or mark partial |
| `DATA.md` | 2026-07-11 | v1-v3a only; line 173 refers to a script that does not exist | update for v3b, eCLIP, TF |
| `DATA_QUALITY_DATASETS.md`, `DATA_CLIP_PRIM_INVENTORY.md` | 2026-10 | current inventories; the first also names the non-existent `15_generate_hard_negatives.py` | keep |
| `DATA_SOURCES_AND_DOWNLOADS.md` | 2026-09-01 | describes sibling repos at `../`; paths differ from this layout | update paths |
| `NEXT_STEPS.md` | 2026-10-30 | plan written before the TF, CORAL and layer work | replace by the new plan after the planning step |
| `STRATEGY.md` | 2026-09-01 | header numbers (V4 0.829) predate the homology audit and v3b; the "lessons" part remains valid | mark as superseded in the header, keep the lessons |
| `PROJECT_PLAN.md` | 2026-04-29 | original plan, historical | archived 2026-11-26 to `docs/archive/` |
| `RESEARCH_ROADMAP.md` | header 2026-06-25 | planning document before v3b, historical | archived to `docs/archive/` |
| `PHASE3A_PLAN.md`, `PHASE3B_STATUS.md`, `PHASE3B_RUNBOOK.md` (ignored), `CROSS_ASSAY_PLAN.md`, `DOMAIN_AWARE_PLAN.md` | 2026-09 | each states "Complete" | archived to `docs/archive/` (`PHASE3B_RUNBOOK.md` is ignored and stays local) |
| `PAPERS_TRACKER.md/.html` (ignored) | 2026-10 | literature tracker, local only | keep |
| `dashboard.html` | 2026-10-28 | README calls it legacy (phases 1-2 only) | archived to `docs/archive/` |
| `strategy_dashboard.html` | 2026-07-11 | architecture and queue, outdated | archived to `docs/archive/` |
| `MEETING_UPDATE_2026-09.html` (ignored) | 2026-10 | meeting update | keep locally |
| `tasks/*.md` (ignored) | 2026-07/08 | outdated todo and session prompt | archive or delete after approval |

## 5. Problems found

1. `README.md` was outdated: no v3b, no scripts 55-81, no TF, CORAL or layer results, a repository map that stops at script 51. Fixed in this update.
2. `reference_letters/` (personal data) and `share_tf/` were not covered by `.gitignore`, and the four nested repositories could be added by accident with `git add .`. All six paths are now ignored.
3. `requirements.txt` lacks packages that scripts import (`requests`, `fair-esm` for `import esm`, `multimolecule` for RNA-FM) and external tools used by 11 scripts (MMseqs2, HMMER). Added to the file with comments. Versions of these packages were not pinned because the VM environment was not read.
4. Files tracked in git although the patterns in `.gitignore` match them: `.Rhistory` (empty) and `results/multiseed/v2_cnn/seed_0/checkpoints/best_model.pt` (4.8 MB, committed 2026-06-25). Tracked large inputs in `results/`: `Zscores_RNAcompete2025.txt.gz` (24 MB, a published table), `knn_baseline/*.tsv.gz`, `hybrid/nn_models.npz`, `rna_layer_profile/profiles.npz`.
5. Untracked results of finished runs that are not committed: `results/coral_audit/` and its log, `results/v3b_report/v3b_*.{json,txt}`, `results/reverse_transfer_run.log`, `results/mechanistic/figures_run.log`.
6. `results/phase3b_summary.json` is cited as "canonical metrics" in the README but contains Phase 3B only. A summary for v3b does not exist; `results/v3b/summary.json` holds the split only.
7. `EXPERIMENT_LOG.md`: three headings still say "planned/built, not yet run" for experiments whose results follow below (TF ChIP-seq, CORAL audit, ESM-2 layers). Entries without a result in the log: ranking-only supervision (script 67 `rank`), profile similarity (script 77), ESM-2 domain/outside pooling, RNA-FM layers (script 81), frozen heads with layers 0/22/26/32.
8. Local copies of some inputs are iCloud placeholders and hang when read (recorded in the log for the Z-score file); `du` reports 0 B for several folders of the project. Large data should not live in the iCloud-synced folder.
9. The `__pycache__` directories were removed during this check (generated files, ignored by git).
10. The repository has 3 commits under other identities: 1 under a personal e-mail address and 2 as `Ubuntu` from the de.NBI cloud VM (not an error; `git config` on the VM can set the name).

## 6. Reorganisation

Done on 2026-11-26 (approved): steps 1 and 3, and the commit of new files (step 6). Open: steps 2, 4, 5, 7, 8 and the script candidates in section 7.

1. Create `docs/archive/` and move there with `git mv`: `PROJECT_PLAN.md`, `RESEARCH_ROADMAP.md`, `PHASE3A_PLAN.md`, `PHASE3B_STATUS.md`, `CROSS_ASSAY_PLAN.md`, `DOMAIN_AWARE_PLAN.md`, `dashboard.html`, `strategy_dashboard.html`. Links in `README.md` are updated at the same time.
2. Create `docs/` for the living documents (`METHODS.md`, `DATA*.md`, `STRATEGY.md`) and keep `README.md`, `EXPERIMENT_LOG.md`, `INVENTORY.md`, `NEXT_STEPS.md` in the root.
3. `git rm --cached` for `.Rhistory` and the tracked checkpoint; the files stay on disk.
4. Move `tasks/` and `tmp/` into an `archive_local/` folder outside git (or delete `tasks/` after reading it once more).
5. Move the four nested upstream repositories out of this folder to `../` (where the documents already say they are), or keep them and rewrite the paths in the documents.
6. Commit `results/coral_audit/`, the v3b report files and the small logs.
7. Rewrite the three outdated headings of the log and add a result block for each run in problem 7 once its output exists.
8. Add a `results/v3b_summary.json` generated from the score tables, so that the README has one canonical file for the current phase.

## 7. Script review (2026-11-26)

Method: for each script, references in other scripts, runners, Markdown files and the log, and use of `src/` modules (text search, so a reference in a comment counts as a reference). Nothing was moved.

Candidates for `scripts/archive/` (no caller, no mention in README, log or other documents, except as noted):
| Script | Reason |
|---|---|
| `23_generate_readme_figures.py` | no reference anywhere; README figures are produced by 32 and 19 |
| `27b_select_domain_diverse_proteins.py` | mentioned only in the archived `CROSS_ASSAY_PLAN.md` |
| `29b_validate_protein_fasta.py` | no reference anywhere |
| `01_prepare_htr_selex.py` | superseded by the config-driven `01_prepare_dataset.py`; mentioned only in the archived `PROJECT_PLAN.md` |
| `src/data/loaders.py` | not imported by any script or module |

Keep, but historical (referenced by README, log or other scripts): 05, 07-10 (Phase 2 models V1-V3c, results in the README table), 12-16 (downloads and Phase 2 analyses), 25-26, 34-37, 45-50 (documented in `data/raw/eclip_skipper/README.md`).

Not verifiable by text search: whether the scripts above were used once for a file that is still in `results/`. Their outputs were not checked.
