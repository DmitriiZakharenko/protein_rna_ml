#!/usr/bin/env python3
"""
build_report_draft.py
---------------------
Builds reports/draft_report.html: a hypothesis -> test -> result -> verdict report of everything done so far.
Numbers come from result files where they exist locally (results/knn_baseline, v3b_report, v3b_affinity, reverse_transfer, parpi_audit,
homology_audit); results that were produced on the VM and only exist as logged output are typed in here and cite their EXPERIMENT_LOG section.

Usage:  python scripts/build_report_draft.py
"""
from __future__ import annotations

import datetime
import html
import io
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 10
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
C = {"v2": "#7a7a7a", "v4": "#0072B2", "nn": "#D55E00", "emb": "#009E73", "other": "#CC79A7", "ref": "#E69F00"}


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2)


def table(head, rows, note="") -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>' + (f'<p class="note">{note}</p>' if note else "")


def fig(svg_s, caption) -> str:
    return f'<figure>{svg_s}<figcaption>{caption}</figcaption></figure>'


VERDICT = {"S": ("supported", "ok"), "N": ("not supported", "no"), "P": ("partly supported", "mid"), "I": ("inconclusive", "mid"),
           "R": ("rejected (control passes)", "ok")}


def section(sid, title, hyp, test, body, verdict, caveat) -> str:
    v, cls = VERDICT[verdict]
    return (f'<section id="{sid}"><h2>{sid.upper()}. {html.escape(title)} <span class="chip {cls}">{v}</span></h2>'
            f'<dl><dt>Hypothesis</dt><dd>{hyp}</dd><dt>Test</dt><dd>{test}</dd></dl>{body}'
            f'<p class="cav"><b>Caveats.</b> {caveat}</p></section>')


def knn(v, e):
    d = json.load(open(R / f"knn_baseline/{v}_summary.json"))["evals"][e]
    return d


# ------------------------------------------------------------------ figures
def fig_bins():
    bins = ["<30% / no hit", "30-50%", "50-70%", "70-90%", ">=90%"]
    n = [6, 15, 14, 13, 7]
    v2, v4, nn = [0.475, 0.613, 0.667, 0.789, 0.850], [0.525, 0.699, 0.695, 0.884, 0.874], [0.45, 0.71, 0.76, 0.84, 0.85]
    f, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(5)
    for i, (y, lab, c) in enumerate(((v2, "V2 (3 seeds)", C["v2"]), (v4, "V4 (3 seeds)", C["v4"]), (nn, "nearest protein (nn1)", C["nn"]))):
        ax.bar(x + (i - 1) * 0.27, y, 0.27, label=lab, color=c)
    ax.axhline(0.5, color="black", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={k})" for b, k in zip(bins, n)])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("cross-protein probe AUROC")
    ax.set_xlabel("best MMseqs2 identity of the test protein to any training protein (v3a)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    style_ax(ax)
    return svg(f)


def fig_v3a_v3b():
    f, ax = plt.subplots(figsize=(5.2, 3.2))
    names = ["V2", "V4", "nearest protein (nn1)"]
    a, b = [0.686, 0.754, 0.724], [0.527, 0.559, 0.627]
    x = np.arange(3)
    ax.bar(x - 0.2, a, 0.4, label="v3a (name-level split)", color="#9ecae1")
    ax.bar(x + 0.2, b, 0.4, label="v3b (cluster split)", color="#08519c")
    for i in range(3):
        ax.text(x[i] - 0.2, a[i] + 0.01, f"{a[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, b[i] + 0.01, f"{b[i]:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="black", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("cross-protein probe AUROC")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax)
    return svg(f)


V3B_PROBE = [  # name, probe AUROC, group
    ("RNA prior / RNA-only head", 0.500, "other"), ("protein-only head (ESM-2)", 0.507, "emb"), ("V2 one-hot (seed 42)", 0.527, "v2"),
    ("concat head (ESM-2 + RNA-FM)", 0.528, "emb"), ("V4 + cross-protein negatives (val probe, early stop)", 0.522, "v4"),
    ("bilinear head, ordinary labels", 0.551, "emb"), ("joint hybrid (nn + head)", 0.557, "other"), ("V4 one-hot (seed 42)", 0.559, "v4"),
    ("bilinear head + cross-protein negatives", 0.570, "emb"), ("nearest protein (nn1, k-mer LR)", 0.627, "nn"),
    ("stack: nn1 + frozen head (weights fitted on val)", 0.630, "other")]


def fig_v3b():
    d = sorted(V3B_PROBE, key=lambda r: r[1])
    f, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.barh(range(len(d)), [r[1] for r in d], color=[C[r[2]] for r in d])
    ax.axvspan(0.5688, 0.6823, color=C["nn"], alpha=0.12, lw=0)
    ax.axvline(0.5, color="black", lw=0.7, ls="--")
    for i, r in enumerate(d):
        ax.text(r[1] + 0.003, i, f"{r[1]:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([r[0] for r in d], fontsize=8)
    ax.set_xlim(0.45, 0.70)
    ax.set_xlabel("v3b cross-protein probe AUROC (shaded: 95% CI of nn1, protein bootstrap)")
    style_ax(ax)
    return svg(f)


def fig_curve():
    f, ax = plt.subplots(figsize=(5.6, 3.3))
    pts = [(0.25, 0.521), (0.5, 0.541), (0.75, 0.549), (0.75, 0.538), (1.0, 0.570)]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=C["emb"], zorder=3)
    ax.axhline(0.827, color=C["v4"], lw=1)
    ax.text(0.26, 0.835, "seen training proteins, new RNAs: 0.827", fontsize=8, color=C["v4"])
    ax.axhline(0.627, color=C["nn"], lw=1)
    ax.text(0.26, 0.635, "nearest protein (nn1): 0.627", fontsize=8, color=C["nn"])
    ax.axhline(0.5, color="black", lw=0.7, ls="--")
    ax.set_xlim(0.2, 1.05)
    ax.set_ylim(0.45, 0.9)
    ax.set_xlabel("fraction of training clusters used")
    ax.set_ylabel("v3b probe AUROC (bilinear head)")
    style_ax(ax)
    return svg(f)


def parpi_table():
    rows = []
    for cell, fn in (("HeLa", "our_models_Hela.tsv"), ("H9", "our_cnn_H9.tsv"), ("HEK293T", "our_cnn_HEK293T.tsv"), ("HEK293", "our_cnn_HEK293.tsv"),
                     ("HepG2", "our_cnn_HepG2.tsv"), ("K562", "our_cnn_K562.tsv")):
        p = R / "parpi_audit" / fn
        if p.exists():
            d = pd.read_csv(p, sep="\t")
            rows.append((cell, len(d), *(float(d[c].median()) for c in ("rna_cnn", "rna_shape_cnn", "deepbind", "iDeep", "Prismnet", "HDRNet", "PaRPI"))))
    return rows


def fig_parpi(rows):
    f, ax = plt.subplots(figsize=(8.0, 3.6))
    labs = ["our CNN (no protein)", "our CNN + icSHAPE", "DeepBind", "iDeep", "PrismNet", "HDRNet", "PaRPI"]
    cols = [C["v4"], "#56B4E9", "#bbbbbb", "#999999", "#777777", "#555555", C["nn"]]
    x = np.arange(len(rows))
    w = 0.12
    for j, (lab, c) in enumerate(zip(labs, cols)):
        ax.bar(x + (j - 3) * w, [r[2 + j] for r in rows], w, label=lab, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\n({r[1]} sets)" for r in rows])
    ax.set_ylim(0.6, 0.97)
    ax.set_ylabel("median AUROC over datasets")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_ax(ax)
    return svg(f)


def fig_reverse():
    d = pd.read_csv(R / "reverse_transfer/per_protein.tsv", sep="\t").sort_values("crossrbp")
    f, ax = plt.subplots(figsize=(7.2, 7.4))
    y = np.arange(len(d))
    ax.scatter(d.invitro_own_cv, y, marker="|", s=60, color="#999999", label="in vitro own data, 5-fold CV")
    ax.scatter(d.prior, y, marker="x", s=14, color=C["other"], label="protein-independent RNA prior")
    ax.scatter(d.crossrbp, y, s=16, color=C["v4"], label="trained on eCLIP of the same protein (cross-RBP negatives)", zorder=3)
    ax.axvline(0.5, color="black", lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d.protein, fontsize=6)
    ax.set_xlabel("AUROC on the in vitro rows of the protein")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style_ax(ax)
    return svg(f)


# ------------------------------------------------------------------ build
def build() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    k3a_s, k3a_p, k3b_s, k3b_p = knn("v3a", "standard"), knn("v3a", "probe"), knn("v3b", "standard"), knn("v3b", "probe")

    def kc(e, s):
        o = e["overall"][s]
        return f"{o['auroc']:.3f} [{o['ci'][0]:.2f}, {o['ci'][1]:.2f}]"

    nnp = json.load(open(R / "v3b_report/nn_probe.json"))["strata"]
    aff = json.load(open(R / "v3b_affinity/summary.json"))["scorers"]
    rev = pd.read_csv(R / "reverse_transfer/per_protein.tsv", sep="\t")
    parpi_rows = parpi_table()
    S = []

    # ---- summary
    hyp = [
        ("H1", "Reported generalisation to unseen proteins is inflated by homology.", "S", "62% of v3a test proteins are >=50% identical to training; probe AUROC rises with identity (0.48 -> 0.85)."),
        ("H2", "A bilinear interaction layer (V4) learns more protein x RNA interaction than concatenation (V2).", "P", "V4 > V2 on the v3a probe in every seed (0.754 vs 0.686); on the cluster split both are near chance (0.559 vs 0.527)."),
        ("H3", "Neural models beat a nearest-protein baseline (no protein model).", "N", "v3a: nn3 0.746 vs V4 0.754; v3b: nn1 0.627 vs V4 0.559, V2 0.527."),
        ("H4", "A cluster split (<=40% identity) removes homology leakage.", "S", "Share of test proteins >=50% identical: 62% -> 3%; 53 of 64 stay at 30-50% (domain-level similarity)."),
        ("H5", "The standard test is largely solvable from the RNA alone.", "S", "RNA-only head: 0.652 on the standard v3b test, 0.500 on the probe; RNA prior 0.59-0.61."),
        ("H6", "Frozen ESM-2 / RNA-FM embeddings improve transfer to new clusters.", "N", "Probe 0.507-0.570 for all heads; stacking with nn1 adds +0.004."),
        ("H7", "Composition-neutral cross-protein negatives in training enable transfer.", "N", "One-hot V4 memorises (loss 1.36 -> 0.77, probe flat); frozen head 0.570 vs 0.551 with ordinary labels."),
        ("H8", "A pipeline defect explains the null result on new clusters.", "R", "Seen proteins / new RNAs: 0.827; nn1 reproduced inside the hybrid pipeline (0.6265 vs 0.627); shuffled protein lowers the score."),
        ("H9", "Protein sequence -> RNA motif profile is learnable beyond the prior.", "N", "ESM-2 ridge Spearman 0.179 vs prior 0.155 (CIs overlap); probe 0.509."),
        ("H10", "V2 uses the RNA-binding domain (alanine mutagenesis, motif recovery).", "N", "Domain-contact residues not more important than outside (p = 0.17-0.24); motif Spearman 0.22 vs 0.26 on seen proteins."),
        ("H11", "The number of training clusters limits transfer.", "P", "Probe 0.52 -> 0.57 from 25% to 100% of clusters; noisy, far from 0.627."),
        ("H12", "In vitro-trained models transfer to eCLIP.", "N", "Zero-shot 0.50-0.56; cross-protein eCLIP negatives 0.50-0.53 for all models (Skipper audit)."),
        ("H13", "In vivo benchmarks with random negatives measure peak-likeness, not interaction.", "S", "RNA-only k-mer LR median 0.739 on 261 PaRPI datasets; our pooled protein model is a dataset identifier (shuffled protein -0.007)."),
        ("H14", "A protein-free CNN is competitive with published protein-free baselines.", "P", "Beats DeepBind, ~iDeep; PaRPI is 0.036 (HeLa) to 0.08 (HEK293, HepG2) above our best protein-free model."),
        ("H15", "The same protein shows concordant eCLIP and in vitro specificity.", "P", "eCLIP-trained k-mer model scores in vitro rows 0.67 vs prior 0.58 (+0.09); protein dependent; ceiling 0.95."),
        ("H16", "Homology transfer carries affinity information (RNAcompete intensities).", "S", "Spearman 0.29 (nn1) vs 0.06 (prior); own-data ceiling 0.56."),
        ("H17", "Classic RPI benchmarks can be solved by one molecule.", "S", "RPI369 protein-only 0.96 vs both 0.97; RPI488 RNA-only 0.91 vs both 0.91."),
    ]
    rows = [(f'<a href="#{h[0].lower()}">{h[0]}</a>', h[1], f'<span class="chip {VERDICT[h[2]][1]}">{VERDICT[h[2]][0]}</span>', h[3]) for h in hyp]
    S.append('<section id="summary"><h2>Summary of hypotheses and outcomes</h2>' + table(["", "Hypothesis", "Verdict", "Key number"], rows,
             "Verdicts refer to the tests listed in each section; every test has stated caveats. AUROC = area under the ROC curve. "
             "Probe = cross-protein probe (same RNA paired with another protein of the same assay; RNA-only and composition baselines give 0.500).") + "</section>")

    S.append('<section id="data"><h2>Data, splits and evaluation protocol</h2>'
             + table(["Item", "Value", "Source"], [
                 ("Dataset v3a", "494 proteins, 2.66 M pairs (HTR-SELEX, RBNS, RNAcompete); split by protein name", "EXPERIMENT_LOG, Phase 3A"),
                 ("Dataset v3b", "same pairs re-split by MMseqs2 single-linkage clusters (identity >=40% at coverage >=50%, or >=70% at >=10%): 160 clusters, "
                  "train 375 proteins / 2.06 M rows, val 55, test 64 (RRM 43, KH 8, CCCH 3, other 10)", "script 58"),
                 ("Cross-protein probe", "positive (A, r) paired with the same RNA r and a different test protein B of the same assay for which r is not a positive; "
                  "RNA marginals identical, GC and k-mer LR give 0.500", "script 53"),
                 ("Nearest-protein baseline", "per-training-protein k-mer logistic regression (RNA only); a test pair is scored by the model of the most similar training protein (MMseqs2)", "script 59"),
                 ("Confidence intervals", "protein-level bootstrap (resampling proteins with replacement)", "scripts 59, 69"),
                 ("Split integrity", "v3b: 0 shared protein names or sequences, 0 exact pair overlaps; 73% of test rows share an RNA with train (assay probe-set reuse), so the standard test is partly RNA-solvable", "script 62")])
             + "</section>")

    # H1
    S.append(section("h1", "Homology inflates generalisation to unseen proteins",
                     "Probe AUROC on the v3a test set depends on how similar the test protein is to a training protein.",
                     "MMseqs2 best identity of each of the 55 v3a test proteins to the 393 training proteins (script 55); probe AUROC by identity bin, 3 seeds each.",
                     fig(fig_bins(), "Fig. 1. v3a probe AUROC by identity bin. Both neural models and the nearest-protein baseline rise with identity; with no detectable homolog "
                                     "all are at chance. Sources: EXPERIMENT_LOG 'MMseqs2 homology audit', 'Nearest-protein baseline on v3a'.")
                     + table(["Best identity to train", "test proteins", "V2", "V4", "nn1"], [
                         ("no hit / <30%", 6, "0.475", "0.525", "0.45"), ("30-50%", 15, "0.613", "0.699", "0.71"), ("50-70%", 14, "0.667", "0.695", "0.76"),
                         ("70-90%", 13, "0.789", "0.884", "0.84"), (">=90%", 7, "0.850", "0.874", "0.85"), ("all", 55, "0.686 +/- 0.015", "0.754 +/- 0.001", "0.724 [0.66, 0.77]")]),
                     "S", "Only 6 proteins without a hit; 7 test proteins are >=90% identical to training ones (e.g. ESRP1 constructs, IGF2BP1), so the name-level split leaks. "
                          "Bin definitions differ slightly between scripts 55 and 59."))

    # H2 + H3
    S.append(section("h2", "V4 (bilinear) vs V2 (concatenation)",
                     "A bilinear interaction layer learns protein x RNA interaction better than concatenation.",
                     "3 seeds each on the v3a probe; per-family breakdown; the same models on v3b (1 seed).",
                     table(["Model", "v3a probe AUROC", "pp-median", "v3b probe AUROC (seed 42)", "v3b standard test"], [
                         ("V2", "0.686 +/- 0.015", "0.696", "0.527", "0.677"), ("V4 concat_bi", "0.754 +/- 0.001", "0.759", "0.559", "0.676")],
                           "Paired V4 - V2 per seed (0, 1, 42): +0.061, +0.085, +0.058. On the standard v3a test the gap is only ~0.016.")
                     + table(["Pfam family (v3a test proteins)", "V2", "V4"], [
                         ("RRM (33)", "0.761", "0.815"), ("KH (5)", "0.731", "0.813"), ("multi (5)", "0.601", "0.668"), ("other (6)", "0.482", "0.536"),
                         ("CCCH (2)", "0.366", "0.343"), ("all (55)", "0.686", "0.754")], "Families with many training proteins (RRM 279, KH 37) transfer; the rest sit at chance."),
                     "P", "On v3a V4 wins in every seed; on v3b (cluster split) the difference is 0.03 in a single seed and both are near chance. Non-RRM families have <=6 test proteins."))
    S.append(section("h3", "Neural models vs a nearest-protein baseline",
                     "V2/V4 learn something beyond homology transfer, i.e. they beat a baseline that uses only the sequence similarity to training proteins.",
                     "Nearest-protein baseline (RNA k-mer LR of the most similar training protein) on v3a and v3b, standard test and probe.",
                     fig(fig_v3a_v3b(), "Fig. 2. Probe AUROC on the name-level split (v3a) and on the cluster split (v3b). Removing homologs costs V2 0.16, V4 0.19, the nearest-protein baseline 0.10.")
                     + table(["Scorer", "v3a standard", "v3a probe", "v3b standard", "v3b probe"], [
                         ("RNA prior (mean of training models)", kc(k3a_s, "mean_lr"), kc(k3a_p, "mean_lr"), kc(k3b_s, "mean_lr"), kc(k3b_p, "mean_lr")),
                         ("nearest protein, k-mer LR (nn1)", kc(k3a_s, "nn1_lr"), kc(k3a_p, "nn1_lr"), kc(k3b_s, "nn1_lr"), kc(k3b_p, "nn1_lr")),
                         ("top-3 neighbours (nn3)", kc(k3a_s, "nn3_lr"), kc(k3a_p, "nn3_lr"), kc(k3b_s, "nn3_lr"), kc(k3b_p, "nn3_lr")),
                         ("nearest protein, PWM (nn1)", kc(k3a_s, "nn1_pwm"), kc(k3a_p, "nn1_pwm"), kc(k3b_s, "nn1_pwm"), kc(k3b_p, "nn1_pwm")),
                         ("V2 (v3a: 3 seeds; v3b: seed 42)", "~0.81", "0.686", "0.677", "0.527"), ("V4 (v3a: 3 seeds; v3b: seed 42)", "~0.81", "0.754", "0.676", "0.559")],
                           "Intervals: protein bootstrap over test proteins (v3a 55, v3b 64 / 63)."),
                     "N", "Neural v3b numbers are one seed and have no interval yet (script 69 prepared, run pending). pp-median (per-protein median AUROC) of nn1 on the v3b probe is 0.703."))

    # H4 + H5
    S.append(section("h4", "Cluster split removes homology leakage",
                     "A single-linkage cluster split gives test proteins with no close training homolog.", "Script 58 (split) and script 55 (identity audit of the result).",
                     table(["Best identity to train", "v3a test (55)", "v3b test (64)"], [("no hit / <30%", "6 / 0", "6 / 3"), ("30-50%", 15, 53), ("50-70%", 14, 2), ("70-90%", 13, 0), (">=90%", 7, 0), (">=50% share", "62%", "3%")]),
                     "S", "Domain-level similarity at 30-50% remains for most test proteins and cannot be removed inside RRM (a threshold of 30% merges all RRM proteins into one cluster of 400+)."))
    S.append(section("h5", "The standard test is largely solvable from the RNA alone",
                     "A model without protein input already scores well on the standard test but at chance on the probe.",
                     "RNA-FM embedding -> MLP without protein input (script 67 head 'rna'); RNA prior; shuffled-protein control.",
                     table(["Scorer (v3b)", "standard test", "probe"], [("RNA-only head (RNA-FM)", "0.652", "0.500"), ("RNA prior (k-mer LR mean)", "0.591", "0.500"), ("protein-only head (ESM-2)", "0.506", "0.507"),
                                                                     ("nearest protein (nn1)", "0.676", "0.627")], "Standard-test AUROC of the RNA-only head: by identity bin <30% 0.69, 30-50% 0.64, 50-70% 0.67, no hit 0.81."),
                     "S", "73% of test rows share an RNA with training rows (shared probe sets of the assays). One seed."))

    # H6, H7, H8, H11 (v3b models)
    S.append(section("h6", "Frozen foundation-model embeddings and hybrids on the cluster split",
                     "ESM-2 (protein) and RNA-FM (RNA) embeddings with a small trained head transfer to unseen clusters; combined with nn1 they add to it.",
                     "Mean-pooled frozen embeddings, heads rna / protein / concat / bilinear (script 67); joint hybrid and held-out stacking with nn1 (script 71); all seed 42.",
                     fig(fig_v3b(), "Fig. 3. v3b cross-protein probe AUROC of every model tested. Colours: grey = one-hot V2, blue = one-hot V4, green = frozen-embedding heads, orange = nearest protein, pink = other.")
                     + table(["Scorer", "probe", "standard test", "shuffled-protein probe"], [
                         ("bilinear head, ordinary labels", "0.551", "0.659", "0.483"), ("bilinear head + cross-protein negatives", "0.570", "0.548", "0.497"),
                         ("concat head", "0.528", "-", "0.488"), ("nn1 only", "0.6265", "0.6755", "-"), ("frozen head only (held-out stacking run)", "0.570", "0.548", "-"),
                         ("stack: logistic combination of nn1 and head, weights fitted on val probe", "0.630", "0.676", "-"),
                         ("joint hybrid (head trained together with nn weight)", "0.557", "0.533", "-")],
                           "Stacking weights: nn1 0.137, head 0.012. In the joint hybrid the nn weight collapsed from 1.0 to 0.14 while the head memorised the training proteins."),
                     "N", "Seed 42 only; one head configuration plus a hyper-parameter sweep still running on the VM. Mean pooling over the whole protein dilutes domain signal (RBD-cropped pooling not tested)."))
    S.append(section("h7", "Cross-protein negatives in training",
                     "Training with the same RNA paired with another protein (composition-neutral negatives) forces the model to learn the interaction.",
                     "V4 with this objective (script 65, early stopping on a held-out cross-protein probe of val proteins); frozen bilinear head with the same objective.",
                     "<p>One-hot V4: training loss falls from 1.36 to 0.77 while the validation probe stays at 0.49-0.52 (best 0.522 at epoch 2, early stop at epoch 10): the model memorises training pairs. "
                     "Frozen head: probe 0.570 with this objective vs 0.551 with ordinary labels.</p>",
                     "N", "One seed, one set of hyper-parameters for the one-hot model."))
    S.append(section("h8", "Is the null result on new clusters a pipeline defect?",
                     "If the pipeline were defective, the same head would also fail on proteins it has seen.",
                     "Positive control: 5% of the training RNAs (all rows containing them) are removed from training; a cross-protein probe is built from them for the seen training proteins. "
                     "Also: nn1 reproduced in the hybrid pipeline; protein-shuffle control.",
                     fig(fig_curve(), "Fig. 4. Learning curve of the bilinear head with cross-protein negatives over the fraction of training clusters (points: single runs; 100% = seed 42; 25-75% mixed seeds, "
                                      "seed-42 runs at 25% and 50% failed on a sampling bug, fixed, pending on the VM).")
                     + table(["Check", "Result"], [
                         ("seen training proteins, new RNAs (ordinary labels / cross-protein negatives)", "0.828 / 0.827"),
                         ("unseen clusters, same heads", "0.550 / 0.542 (this run); 0.551 / 0.570 (full data)"),
                         ("nn1 recomputed inside the hybrid pipeline", "probe 0.6265, standard 0.6755 (script 59: 0.627 / 0.676)"),
                         ("protein shuffled at test (bilinear heads)", "0.55 -> 0.48, 0.57 -> 0.50")]),
                     "R", "The positive control shows the head can learn protein-specific RNA preferences; it does not prove that no better architecture exists. Hyper-parameter sweep pending."))
    S.append(section("h11", "Number of training clusters",
                     "Transfer to unseen clusters grows with the number of training clusters.", "Bilinear head with cross-protein negatives trained on 25/50/75/100% of the 160 clusters.",
                     "<p>Probe AUROC 0.521 (25%, seed 0), 0.541 (50%, seed 0), 0.549 / 0.538 (75%, seeds 42 / 0), 0.570 (100%, seed 42): +0.05 over a fourfold increase, differences between seeds 0.01-0.02. "
                     "Extrapolation to the nn1 level (0.627) cannot be made from these points.</p>", "P",
                     "Cluster sampling is dominated by the one large cluster (220 RRM proteins), so the fractions are not comparable in protein count; seed-42 runs for 25% and 50% pending."))

    # H9, H10
    S.append(section("h9", "Protein sequence -> motif profile",
                     "The 5-mer enrichment profile of a protein can be predicted from its sequence on held-out clusters.",
                     "Per-protein 5-mer log2 enrichment (positives vs negatives) as target; ESM-2 mean embedding; ridge and kNN; Spearman against the true profile on 64 test proteins; probe AUROC of RNAs scored with the predicted profile (script 73).",
                     table(["Predictor", "Spearman median", "mean [95% CI]", "top-20 overlap", "probe AUROC"], [
                         ("prior (mean training profile)", "0.101", "0.155 [0.070, 0.236]", "0.048", "0.500"), ("kNN1 (ESM-2 cosine)", "0.116", "0.114 [0.037, 0.190]", "0.091", "0.552"),
                         ("kNN5", "0.227", "0.183 [0.107, 0.262]", "0.089", "0.541"), ("ridge (alpha 1e4 selected on val)", "0.185", "0.179 [0.107, 0.249]", "0.080", "0.509"),
                         ("oracle: the protein's own true profile", "-", "-", "-", "0.902")],
                           "The oracle shows that a 5-mer profile is a sufficient RNA scorer if it were known."),
                     "N", "Mean-pooled ESM-2 only (no RBD crop, no residue-level model, no structure); 375 training proteins."))
    S.append(section("h10", "Mechanistic checks of V2: motif recovery and alanine mutagenesis",
                     "A model that has learned binding recognises the protein's motif in silico and depends on domain contact residues.",
                     "Script 61 on V2 seed 42 (v3a): predicted 5-mer motif vs the measured enrichment profile (real-RNA insertion contexts), alanine replacement of domain-contact residues vs outside-contact residues; seen proteins as positive control.",
                     table(["", "held-out proteins (55)", "training proteins (55)"], [
                         ("motif Spearman, V2", "0.221", "0.259"), ("motif Spearman, shuffled protein", "0.126", "0.139"), ("protein-agnostic prior", "0.171", "-"),
                         ("nearest-protein motif", "0.446", "-"), ("domain-contact > outside-contact residues (share of proteins, p)", "48% (0.24)", "35% (0.17)")]),
                     "N", "V2 only; the positive control is not clearly better than held-out proteins, so the test is weakly sensitive. These tests are kept as gates for any model that passes the probe."))

    # H12, H13, H14
    S.append(section("h12", "In vitro models and eCLIP",
                     "Models trained on in vitro data transfer to eCLIP with composition-neutral cross-protein negatives.",
                     "Skipper eCLIP rebuilt with real protein sequences; V2/V4 zero-shot, fine-tuned and trained on eCLIP; cross_protein_neg benchmark (scripts 41-53).",
                     table(["Model", "protein_and_rna split", "cross_protein_neg"], [
                         ("V2 trained on eCLIP", "full 0.858, GC-matched 0.772", "0.485"), ("V2 fine-tuned from v3a", "GC-matched 0.711", "0.527 (GC-matched)"),
                         ("V2 / V4 v3a zero-shot", "0.56-0.60 / 0.52-0.54 (GC-matched)", "0.528 / 0.50-0.51"), ("RNA-only k-mer LR on Skipper eCLIP", "0.81", "-")]),
                     "N", "Single seed, one 60k sample; name-level protein novelty for eCLIP (5 of 24 test proteins >=90% identical to training). Re-report without them pending; a null result cannot be created by removing leakage."))
    S.append(section("h13", "In vivo benchmarks: composition and dataset identity",
                     "On PaRPI-type data (top peaks vs random transcript windows) a protein-free model already explains most of the signal, and a pooled protein model acts as a dataset identifier.",
                     "261 PaRPI datasets: RNA-only k-mer LR (scripts 63); HeLa: our CNN variants and pooled V2 with protein permutation and cross-RBP negatives (script 64).",
                     table(["Scenario", "median AUROC"], [
                         ("RNA-only k-mer LR, original random split (261 datasets)", "0.739"), ("... held-out transcripts", "0.741"), ("... negatives = other RBPs' positives of the cell line", "0.674"),
                         ("GC content alone", "0.397"), ("HeLa: our pooled V2 (RNA + protein one-hot)", "0.881"), ("... protein sequences permuted at test", "0.874"),
                         ("... negatives = positives of the other 16 RBPs", "0.644")]),
                     "S", "PaRPI itself was not run; published per-dataset numbers are the authors'. Their released code selects the epoch on the test split (code reading, not executed)."))
    S.append(section("h14", "Protein-free CNN vs published methods on PaRPI data",
                     "A per-dataset CNN on RNA sequence (optionally with in vivo icSHAPE) matches published protein-free baselines.",
                     "Per-dataset CNN (one-hot, 101 nt) and CNN + icSHAPE, random stratified 80/20 split with a separate validation split for epoch selection (script 64, one seed). Published values from PaRPI Supplementary Data 1.",
                     fig(fig_parpi(parpi_rows), "Fig. 5. Median test AUROC per cell line. Our models: one seed, validation-based epoch selection. Published models: the authors' numbers for the same datasets.")
                     + table(["Cell line", "datasets", "our CNN", "our CNN + icSHAPE", "DeepBind", "iDeep", "PrismNet", "HDRNet", "PaRPI"],
                             [(r[0], r[1], *(f"{v:.3f}" for v in r[2:])) for r in parpi_rows],
                             "H9 has a single dataset. K562 (112 datasets) is running and not yet included." if not any(r[0] == "K562" for r in parpi_rows) else ""),
                     "P", "Gap to PaRPI: HeLa 0.036, HEK293 0.076, HepG2 0.081 (our best protein-free model). Part of it may come from test-based epoch selection in their code; the size of that effect is not measured (planned: B0)."))

    # H15
    S.append(section("h15", "Same protein: eCLIP vs in vitro specificity",
                     "A model trained only on the eCLIP data of a protein predicts that protein's in vitro positives.",
                     "50 proteins present in PaRPI eCLIP data and in v3b (in vitro rows only); k-mer LR (k=1..4 frequencies) on eCLIP vs genomic negatives or vs other RBPs' peaks; AUROC on the protein's in vitro rows (script 74).",
                     fig(fig_reverse(), "Fig. 6. Per-protein AUROC on in vitro rows. Blue: model trained on eCLIP of the same protein; x: protein-independent RNA prior; bar: in vitro own data (5-fold CV).")
                     + table(["Scorer", "mean [95% CI over proteins]", "median", "share > 0.6"], [
                         ("eCLIP model, genomic negatives", "0.670 [0.611, 0.723]", "0.695", "0.72"), ("eCLIP model, cross-RBP negatives", "0.671 [0.613, 0.726]", "0.677", "0.64"),
                         ("protein-independent in vitro prior", "0.579 [0.517, 0.636]", "0.642", "0.58"), ("in vitro own data (5-fold CV)", "0.947 [0.929, 0.961]", "0.958", "1.00")],
                           f"Paired difference to the prior: +0.091 [0.028, 0.159] (genomic) and +0.092 [0.016, 0.163] (cross-RBP). By assay: RNAcompete Eukarya (24 proteins) 0.73-0.75, SELEX/RBNS (21) 0.58-0.59. "
                           f"Highest: PTBP1, ELAVL1, TRA2A, HNRNPA1, TIA1, PCBP2; lowest five: TAF15, HNRNPF, SAFB2, SF3B4, PUM1 (all below 0.4; the 8-mer UGUAHAUA of PUM1 is not visible to 1-4-mer frequencies)."),
                     "P", "k-mer frequencies only; some proteins also occur in the v3b training set; window lengths differ (101 nt vs 20-60 nt)."))

    # H16
    S.append(section("h16", "Affinity-style test on RNAcompete intensities",
                     "Homology transfer carries information about probe intensity, not only the binary label.",
                     "52 v3b test proteins with RNAcompete data, ~2000 probes each: per-protein Spearman between score and probe intensity (script 72).",
                     table(["Scorer", "Spearman median", "mean [95% CI over proteins]"], [
                         ("protein-independent RNA prior", f"{aff['prior']['median']:.3f}", f"{aff['prior']['mean']:.3f} [{aff['prior']['ci'][0]:.3f}, {aff['prior']['ci'][1]:.3f}]"),
                         ("nearest training protein (nn1, k-mer LR from binary labels)", f"{aff['nn1']['median']:.3f}", f"{aff['nn1']['mean']:.3f} [{aff['nn1']['ci'][0]:.3f}, {aff['nn1']['ci'][1]:.3f}]"),
                         ("own data, ridge, 5-fold by probe (needs the protein's measurements)", f"{aff['own_cv']['median']:.3f}", f"{aff['own_cv']['mean']:.3f} [{aff['own_cv']['ci'][0]:.3f}, {aff['own_cv']['ci'][1]:.3f}]")],
                           "For orientation, NPBIP (2026) reports Pearson 0.32-0.41 on RNAcompete with a different protocol and metric; not a like-for-like comparison."),
                     "S", "Only baselines scored; V2/V4/frozen heads not run on the same probes; intensity-trained transfer models not tried."))

    # H17
    S.append(section("h17", "Shortcut audit of classic RPI benchmarks",
                     "Published benchmark numbers can be reached by a model that sees only the RNA or only the protein.",
                     "RPISeq-style random forest (RNA 4-mer, protein 3-mer in a 7-letter alphabet) with single-molecule controls under random, protein-disjoint and RNA-disjoint folds (script 60).",
                     table(["Dataset (pairs / proteins / RNAs)", "Folds", "RNA only", "protein only", "both"], [
                         ("RPI369 (447 / 280 / 229)", "random", "0.854", "0.964", "0.974"), ("RPI369", "protein-disjoint", "0.857", "0.909", "0.932"),
                         ("RPI488 (486 / 247 / 25)", "random", "0.909", "0.422", "0.906"), ("RPI1446 (2684 / 635 / 99)", "random", "0.794", "0.609", "0.922"),
                         ("RPI1446", "protein-disjoint", "0.794", "0.507", "0.844"), ("RPI2241 (2884 / 923 / 416)", "random", "0.831", "0.594", "0.874"), ("RPI2241", "protein-disjoint", "0.832", "0.556", "0.847")]),
                     "S", "Our re-implementation does not reproduce published numbers (e.g. RPI369 random 0.974 vs 0.85 reported), so the claim is limited to dataset properties; de-duplicated pairs, untuned random forest."))

    # per-assay / family breakdown of nn1
    def srow(k, label):
        s = nnp[k]
        return (label, s["n_proteins"], f"{s['auroc']['nn1'][0]:.3f} [{s['auroc']['nn1'][1]:.2f}, {s['auroc']['nn1'][2]:.2f}]")
    S.append('<section id="strata"><h2>Nearest-protein baseline on the v3b probe by stratum</h2>'
             + table(["Stratum", "proteins", "nn1 AUROC [95% CI]"], [srow("all", "all"), srow("identity:30-50%", "identity 30-50%"), srow("identity:no_hit", "identity: no hit (equals the RNA prior)"),
                                                                    srow("family:RRM", "family RRM"), srow("family:KH", "family KH"), srow("family:other", "family other"),
                                                                    srow("source:rnacompete_eukarya", "assay RNAcompete Eukarya"), srow("source:rnacompete_rbpzoo", "assay RNAcompete RBPZoo"), srow("source:selex_rbns", "assay SELEX / RBNS")],
                    "Script 69 (protein bootstrap, 300 draws). The same table for V2/V4/frozen heads needs their score files (pending).") + "</section>")

    S.append('<section id="status"><h2>Open items and run status</h2>' + table(["Item", "Status"], [
        ("Seeds 0 and 1 for the frozen heads, hyper-parameter sweep, cluster learning curve (25% / 50% with seed 42)", "running on the VM (script run_v3b_frozen.sh / run_v3b_diagnostics.sh)"),
        ("Table with confidence intervals and paired differences for all v3b scorers (V2, V4, heads, hybrid)", "script 69 ready; needs score files from the VM"),
        ("PaRPI-type comparison: K562 (112 datasets)", "running locally"),
        ("Gap to published PaRPI numbers: recompute with test-based epoch selection (B0)", "planned"),
        ("eCLIP re-report without proteins >=90% identical to training; per-assay breakdown of V2/V4", "needs VM score files"),
        ("Attribution maps (script 68) and domain-enrichment tests", "tool ready; to be run on a model that passes the probe gate; none does yet"),
        ("Variant-effect and disease part (B3-B4), PU-style negatives, structure-derived priors (Track C)", "not started"),
        ("Not reproduced: PaRPI and RPIEmbeddor 2.0 themselves (code without data/checkpoints)", "comparison via published numbers and our own models")]) + "</section>")

    nav = '<nav><a href="#summary">Summary</a><a href="#data">Data</a>' + "".join(f'<a href="#{h[0].lower()}">{h[0]}</a>' for h in hyp) + '<a href="#strata">Strata</a><a href="#status">Status</a></nav>'
    css = """
:root{--bg:#fff;--fg:#1c1f23;--mut:#5b6470;--line:#d9dee4;--card:#f6f8fa;--ok:#0b6e4f;--no:#a4262c;--mid:#8a5a00;--link:#0b57d0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171a;--fg:#e6e8ea;--mut:#9aa4af;--line:#2c3238;--card:#1b1f23;--ok:#4cc38a;--no:#ff7b72;--mid:#e3b341;--link:#79b8ff}}
:root[data-theme="dark"]{--bg:#14171a;--fg:#e6e8ea;--mut:#9aa4af;--line:#2c3238;--card:#1b1f23;--ok:#4cc38a;--no:#ff7b72;--mid:#e3b341;--link:#79b8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px 16px 80px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 8px;border-top:1px solid var(--line);padding-top:16px}
.meta{color:var(--mut);margin:0 0 14px}a{color:var(--link)}nav{display:flex;flex-wrap:wrap;gap:6px 12px;margin:12px 0 4px;font-size:13px}
dl{margin:0 0 10px;display:grid;grid-template-columns:90px 1fr;gap:4px 10px}dt{color:var(--mut);font-weight:600}dd{margin:0}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--card);font-weight:600}.note,.cav{color:var(--mut);font-size:13px;margin:6px 0}figure{margin:12px 0}figure svg{max-width:100%;height:auto}svg text{fill:var(--fg)!important}svg [style*="stroke: #000000"]{stroke:var(--fg)!important}
figcaption{color:var(--mut);font-size:13px}.chip{font-size:12px;font-weight:600;padding:1px 8px;border-radius:10px;border:1px solid currentColor;white-space:nowrap;vertical-align:middle}
.chip.ok{color:var(--ok)}.chip.no{color:var(--no)}.chip.mid{color:var(--mid)}
@media (max-width:600px){dl{grid-template-columns:1fr}}
"""
    head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Transfer Audit Draft</title><style>{css}</style></head><body><main>'
            f'<h1>Protein-RNA transfer: hypotheses and tests (draft)</h1>'
            f'<p class="meta">Generated {today} from repository commit {commit}. Every number is traceable to EXPERIMENT_LOG.md or a results file; '
            f'single-seed results are marked as such. Draft: open items are listed at the end.</p>{nav}')
    return head + "".join(S) + "</main></body></html>"


if __name__ == "__main__":
    out = ROOT / "reports" / "draft_report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
