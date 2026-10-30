#!/usr/bin/env python3
"""
build_report_draft.py
---------------------
Builds reports/draft_report.html: a sequential report (A data and baselines, B does the evaluation measure interaction, C transfer to new
proteins, D attempts to improve and controls, E in vivo data, F affinity). Every block is question -> test -> result -> verdict.

Numbers come from result files where they exist locally (results/knn_baseline, v3b_report, v3b_affinity, reverse_transfer, parpi_audit,
rna_only_per_protein_honest, mechanistic/figures); results produced on the VM that exist only as logged output are typed in here and cite EXPERIMENT_LOG.

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

matplotlib.rcParams.update({"svg.fonttype": "none", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
                            "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5})
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
C = {"v2": "#8a8a8a", "v4": "#0072B2", "nn": "#D55E00", "emb": "#009E73", "other": "#CC79A7", "own": "#222222"}


def svg(f) -> str:
    buf = io.StringIO()
    f.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(f)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def table(head, rows, note="") -> str:
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>' + (f'<p class="note">{note}</p>' if note else "")


def figure(s, cap) -> str:
    return f"<figure>{s}<figcaption>{cap}</figcaption></figure>"


VERDICT = {"S": ("supported", "ok"), "N": ("not supported", "no"), "P": ("partly", "mid"), "R": ("ruled out", "ok"), "I": ("illustration", "mid")}


# ----------------------------------------------------------------------------- figures
def fig_ladder():
    rows = [("seen protein, own data (standard test)", 0.95, C["own"]), ("seen protein, new RNAs (cross-protein probe)", 0.83, C["own"]),
            ("homolog >=70% identical (V4, v3a probe)", 0.88, C["v4"]), ("homolog 50-70% (V4, v3a probe)", 0.695, C["v4"]),
            ("homolog 30-50% (V4, v3a probe)", 0.699, C["v4"]), ("new cluster, nearest protein (v3b probe)", 0.627, C["nn"]),
            ("new cluster, best neural model (v3b probe)", 0.559, C["v4"]), ("no detectable homolog (V4, v3a probe)", 0.525, C["v4"])]
    f, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
    for yy, r in zip(y, rows):
        ax.text(r[1] + 0.008, yy, f"{r[1]:.2f}", va="center", fontsize=11)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.text(0.502, y.max() + 0.62, "chance", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("AUROC")
    clean(ax)
    return svg(f)


def fig_bins():
    bins = ["no hit /\n<30%", "30-50%", "50-70%", "70-90%", ">=90%"]
    n = [6, 15, 14, 13, 7]
    v2, v4, nn = [0.475, 0.613, 0.667, 0.789, 0.850], [0.525, 0.699, 0.695, 0.884, 0.874], [0.45, 0.71, 0.76, 0.84, 0.85]
    f, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(5)
    for i, (y, lab, c) in enumerate(((v2, "V2", C["v2"]), (v4, "V4", C["v4"]), (nn, "nearest protein (nn1)", C["nn"]))):
        ax.bar(x + (i - 1) * 0.27, y, 0.27, label=lab, color=c)
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={k})" for b, k in zip(bins, n)])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("v3a probe AUROC")
    ax.set_xlabel("best identity of the test protein to any training protein")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean(ax)
    return svg(f)


def fig_scatter():
    pts = [("RNA prior", 0.591, 0.500, C["other"]), ("RNA-only head", 0.651, 0.500, C["other"]), ("protein-only head", 0.507, 0.504, C["emb"]),
           ("concat head", 0.618, 0.528, C["emb"]), ("bilinear head", 0.623, 0.531, C["emb"]), ("bilinear + cross-protein neg.", 0.531, 0.559, C["emb"]),
           ("V2", 0.677, 0.527, C["v2"]), ("V4", 0.676, 0.559, C["v4"]), ("nearest protein", 0.676, 0.627, C["nn"]), ("nn1 + head (stack)", 0.676, 0.630, C["other"])]
    offs = {"RNA prior": (-6, -14), "RNA-only head": (6, -4), "protein-only head": (6, 4), "concat head": (-8, -16), "bilinear head": (6, -10),
            "bilinear + cross-protein neg.": (6, 3), "V2": (6, -12), "V4": (6, 2), "nearest protein": (-6, 8), "nn1 + head (stack)": (6, 8)}
    f, ax = plt.subplots(figsize=(8.6, 5.2))
    for name, x, y, c in pts:
        ax.scatter(x, y, s=70, color=c, zorder=3)
        ax.annotate(name, (x, y), xytext=offs[name], textcoords="offset points", fontsize=10.5, ha="left" if offs[name][0] > 0 else "right")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xlim(0.48, 0.72)
    ax.set_ylim(0.47, 0.66)
    ax.set_xlabel("standard v3b test AUROC (RNA features are enough)")
    ax.set_ylabel("cross-protein probe AUROC (protein is required)")
    clean(ax)
    return svg(f)


def fig_forest():
    rows = [("RNA prior / RNA-only head (3 seeds)", 0.500, 0.500, 0.500, C["other"]), ("protein-only head (3 seeds)", 0.504, 0.500, 0.507, C["emb"]),
            ("V2 one-hot (1 seed)", 0.527, None, None, C["v2"]), ("concat head (3 seeds)", 0.528, 0.522, 0.533, C["emb"]),
            ("bilinear head, ordinary labels (3 seeds)", 0.531, 0.501, 0.551, C["emb"]), ("joint hybrid nn + head (1 seed)", 0.557, None, None, C["other"]),
            ("V4 one-hot (1 seed)", 0.559, None, None, C["v4"]), ("bilinear head + cross-protein negatives (3 seeds)", 0.559, 0.553, 0.570, C["emb"]),
            ("nearest protein nn1 (95% CI, protein bootstrap)", 0.627, 0.569, 0.682, C["nn"]), ("stack nn1 + head (1 seed)", 0.630, None, None, C["other"])]
    f, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(rows))
    for yy, (n, m, lo, hi, c) in zip(y, rows):
        ax.scatter(m, yy, s=70, color=c, zorder=3)
        if lo is not None and hi != lo:
            ax.plot([lo, hi], [yy, yy], color=c, lw=3, alpha=0.6)
        ax.text(0.713, yy, f"{m:.3f}", va="center", fontsize=10.5)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.axvline(0.627, color=C["nn"], lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.48, 0.73)
    ax.set_xlabel("v3b cross-protein probe AUROC (dot = mean or single run, line = seed range or CI)")
    clean(ax)
    return svg(f)


def fig_curve():
    pts = [(0.25, 0.535), (0.25, 0.521), (0.5, 0.554), (0.5, 0.541), (0.75, 0.549), (0.75, 0.537), (1.0, 0.570), (1.0, 0.553), (1.0, 0.554)]
    means = [(0.25, 0.528), (0.5, 0.548), (0.75, 0.543), (1.0, 0.559)]
    f, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=C["emb"], s=36, alpha=0.7, label="single run")
    ax.plot([m[0] for m in means], [m[1] for m in means], color=C["emb"], lw=2, label="mean over seeds")
    ax.axhline(0.827, color=C["own"], lw=1.2)
    ax.text(0.205, 0.835, "same head on training proteins, new RNAs: 0.83", fontsize=10.5)
    ax.axhline(0.627, color=C["nn"], lw=1.2)
    ax.text(0.205, 0.635, "nearest protein: 0.627", fontsize=10.5, color=C["nn"])
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xlim(0.2, 1.05)
    ax.set_ylim(0.45, 0.9)
    ax.set_xlabel("fraction of training clusters used")
    ax.set_ylabel("v3b probe AUROC")
    ax.legend(frameon=False, loc="center right")
    clean(ax)
    return svg(f)


def fig_bars(labels, vals, colors, ylabel, ylim, err=None, w=7.6, h=3.8, rot=0):
    f, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=0.62, yerr=None if err is None else np.array(err).T, capsize=4, error_kw={"lw": 1.2})
    for i, v in enumerate(vals):
        ax.text(i, v + (0.01 if err is None else err[i][1] + 0.01), f"{v:.3f}", ha="center", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    clean(ax)
    return svg(f)


def fig_rpi():
    d = {"RPI369": (0.854, 0.964, 0.974), "RPI488": (0.909, 0.422, 0.906), "RPI1446": (0.794, 0.609, 0.922), "RPI2241": (0.831, 0.594, 0.874)}
    f, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(4)
    for i, (lab, c) in enumerate((("RNA only", C["other"]), ("protein only", C["emb"]), ("both", C["v4"]))):
        ax.bar(x + (i - 1) * 0.27, [v[i] for v in d.values()], 0.27, label=lab, color=c)
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(list(d))
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("AUROC, random 5-fold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    clean(ax)
    return svg(f)



def fig_design():
    from matplotlib.patches import FancyBboxPatch
    f, axs = plt.subplots(3, 1, figsize=(10.5, 7.8))

    def box(ax, x, y, w, h, t, fc, fs=11, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#444444", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(ax, x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1.3))

    PA, PB, RN, OK, NO = "#cfe3f5", "#f5dcc4", "#e7e7e7", "#cfeedd", "#f6d3d3"
    ax = axs[0]
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.1); ax.axis("off")
    ax.set_title("Standard test: each pair is labelled by the assay (RNA that binds the protein = 1, other RNA of the same assay = 0)", loc="left", fontsize=11.5)
    box(ax, 0.1, 1.15, 2.2, 0.6, "protein A", PA); box(ax, 2.6, 1.15, 2.6, 0.6, "RNA 1", RN); arrow(ax, 5.3, 6.2, 1.45); box(ax, 6.3, 1.15, 2.2, 0.6, "binds (1)", OK)
    box(ax, 0.1, 0.2, 2.2, 0.6, "protein A", PA); box(ax, 2.6, 0.2, 2.6, 0.6, "RNA 2", RN); arrow(ax, 5.3, 6.2, 0.5); box(ax, 6.3, 0.2, 2.2, 0.6, "does not bind (0)", NO)
    ax.text(8.9, 0.85, "RNA 1 and RNA 2 differ in\nsequence and composition:\nthe label can often be\nread from the RNA alone", fontsize=10.5, va="center")
    ax = axs[1]
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.1); ax.axis("off")
    ax.set_title("Cross-protein probe: the same RNA is paired with a different protein of the same assay", loc="left", fontsize=11.5)
    box(ax, 0.1, 1.15, 2.2, 0.6, "protein A", PA); box(ax, 2.6, 1.15, 2.6, 0.6, "RNA 1", RN); arrow(ax, 5.3, 6.2, 1.45); box(ax, 6.3, 1.15, 2.2, 0.6, "binds (1)", OK)
    box(ax, 0.1, 0.2, 2.2, 0.6, "protein B", PB); box(ax, 2.6, 0.2, 2.6, 0.6, "RNA 1", RN); arrow(ax, 5.3, 6.2, 0.5); box(ax, 6.3, 0.2, 2.2, 0.6, "does not bind (0)", NO)
    ax.text(8.9, 0.85, "the RNA is identical in both\nclasses: a model that sees only\nthe RNA scores 0.5 (chance);\nthe protein must be used", fontsize=10.5, va="center")
    ax = axs[2]
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.1); ax.axis("off")
    ax.set_title("Data splits: how test proteins relate to training proteins", loc="left", fontsize=11.5)
    box(ax, 0.1, 1.15, 2.6, 0.6, "training: protein A'", PA); box(ax, 3.0, 1.15, 2.6, 0.6, "test: protein A", PA); ax.text(5.9, 1.45, "A is a close relative of A'  ->  v3a (split by protein name)", va="center", fontsize=10.5)
    box(ax, 0.1, 0.2, 2.6, 0.6, "training: clusters 1..k", PB); box(ax, 3.0, 0.2, 2.6, 0.6, "test: cluster k+1", PA); ax.text(5.9, 0.5, "no close relative in training  ->  v3b (split by homology cluster)", va="center", fontsize=10.5)
    f.tight_layout()
    return svg(f)

def parpi_frames():
    out = []
    for cell, fn in (("HeLa", "our_models_Hela.tsv"), ("H9", "our_cnn_H9.tsv"), ("HEK293T", "our_cnn_HEK293T.tsv"), ("HEK293", "our_cnn_HEK293.tsv"),
                     ("HepG2", "our_cnn_HepG2.tsv"), ("K562", "our_cnn_K562.tsv")):
        p = R / "parpi_audit" / fn
        if p.exists():
            d = pd.read_csv(p, sep="\t")
            d["cell"] = cell
            out.append(d)
    return out


def fig_parpi_scatter(a):
    f, ax = plt.subplots(figsize=(6.4, 5.6))
    cols = {"HeLa": "#0072B2", "H9": "#999999", "HEK293T": "#56B4E9", "HEK293": "#009E73", "HepG2": "#E69F00", "K562": "#D55E00"}
    for c, g in a.groupby("cell"):
        ax.scatter(g.rna_shape_cnn, g.PaRPI, s=14, alpha=0.75, color=cols[c], label=f"{c} ({len(g)})")
    ax.plot([0.6, 1], [0.6, 1], color="black", lw=0.8, ls="--")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("our CNN + icSHAPE (no protein), AUROC")
    ax.set_ylabel("published PaRPI, AUROC")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    clean(ax)
    return svg(f)


def fig_parpi_methods(a):
    meds = [("DeepBind", a.deepbind.median(), "#bbbbbb"), ("our CNN", a.rna_cnn.median(), C["v4"]), ("iDeep", a.iDeep.median(), "#999999"),
            ("our CNN + icSHAPE", a.rna_shape_cnn.median(), "#56B4E9"), ("PrismNet", a.Prismnet.median(), "#777777"), ("HDRNet", a.HDRNet.median(), "#555555"),
            ("PaRPI", a.PaRPI.median(), C["nn"])]
    return fig_bars([m[0] for m in meds], [m[1] for m in meds], [m[2] for m in meds], "median AUROC, 261 datasets", (0.6, 0.97), w=6.6, h=4.2, rot=30)


def fig_reverse():
    d = pd.read_csv(R / "reverse_transfer/per_protein.tsv", sep="\t").sort_values("crossrbp")
    f, ax = plt.subplots(figsize=(9, 9.2))
    y = np.arange(len(d))
    ax.scatter(d.invitro_own_cv, y, marker="|", s=90, color="#555555", label="in vitro own data, 5-fold CV")
    ax.scatter(d.prior, y, marker="x", s=20, color=C["other"], label="protein-independent RNA prior")
    ax.scatter(d.crossrbp, y, s=26, color=C["v4"], label="model trained on eCLIP of the same protein", zorder=3)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d.protein, fontsize=8.5)
    ax.set_xlabel("AUROC on the in vitro rows of the protein")
    ax.legend(frameon=False, loc="lower right")
    clean(ax)
    return svg(f)


def gallery() -> str:
    """Motif logos and saliency maps from scripts/75 (results/mechanistic/figures); empty if not generated."""
    d = R / "mechanistic/figures"
    if not (d / "stats.json").exists():
        return ""
    st = json.load(open(d / "stats.json"))
    import base64
    out, rows = "", []

    def img(fp, cap):
        b = base64.b64encode(fp.read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(cap[:60])}"><figcaption>{cap}</figcaption></figure>'

    for n, s_ in st["proteins"].items():
        f = d / f"motif_{n}.png"
        if f.exists():
            out += img(f, f"{n}: measured 5-mer enrichment vs the model's prediction (left; labelled: top measured 5-mers), measured motif logo (middle), model-predicted motif logo (right).")
        rows.append((n, f"{s_['spearman_pred_vs_true']:.2f}", f"{s_.get('domain_fraction', float('nan')):.2f}", f"{s_.get('ig_share_in_domain', float('nan')):.2f}"))
    for n in st["proteins"]:
        f = d / f"saliency_{n}.png"
        if f.exists():
            out += img(f, f"{n}: saliency map of one positive RNA. Top to bottom: sequence, in-silico mutations (signed logo), sliding-window deletion, integrated gradients on the RNA, "
                          f"integrated gradients on the protein with Pfam domains shaded. Positive letters point up, negative letters are flipped.")
    out += table(["Protein", "Spearman, predicted vs measured 5-mer profile", "share of residues in domain", "share of |IG| in domain"], rows,
                 "If the model used the domain, the share of |IG| in the domain would clearly exceed the share of residues in it.")
    return out



def terms_section() -> str:
    gl = [
        ("Data", [
            ("In vitro data", "Binding experiments HTR-SELEX and RBNS (RNAs selected from random pools) and RNAcompete (fixed RNA probes). 494 proteins, 2.66 million (protein, RNA) pairs, label: binds = 1, does not bind = 0."),
            ("In vivo data", "eCLIP / CLIP: protein-RNA crosslinking in cells, 101-nt windows (section 5). PaRPI: published benchmark of 261 such datasets."),
            ("v3a", "In vitro data split into train / validation / test by protein name. Relatives (homologs) of a test protein can be in the training set."),
            ("v3b", "Same data split by homology cluster: proteins with >= 40% sequence identity over >= 50% of their length (MMseqs2) are always on the same side. "
                    "160 clusters; 375 / 55 / 64 proteins in train / validation / test; test proteins have no close relative in training."),
            ("Seen / unseen protein", "Seen: in the training data. Unseen: not in it (in v3b also without a close relative)."),
            ("Identity bin", "Best sequence identity of a test protein to any training protein, grouped: <30% or no hit, 30-50%, 50-70%, 70-90%, >=90%."),
        ]),
        ("Tests", [
            ("Standard test", "Pairs labelled by the assay; negatives are other RNAs of the same experiment, so RNA composition alone is informative."),
            ("Cross-protein probe", "The same RNA is a positive for protein A and a negative for another protein B of the same assay (schematic above). RNA-only models score 0.5."),
        ]),
        ("Models and baselines", [
            ("V2", "Convolutional network (CNN) with one branch for the RNA and one for the protein sequence (one-hot encoded); the two vectors are concatenated and classified."),
            ("V4", "V2 plus a bilinear (multiplicative) layer on the RNA and protein vectors, so the network can model their interaction."),
            ("Frozen-embedding head", "Small network on fixed vectors from pretrained models ESM-2 (protein) and RNA-FM (RNA). Variants: RNA only, protein only, concatenation, bilinear."),
            ("Nearest-protein baseline (nn1)", "No protein model: an RNA-only model (logistic regression on k-mer frequencies) is fitted per training protein; a test pair is scored with the model of the training protein most similar in sequence to the test protein."),
            ("RNA prior", "Mean of the RNA-only models of all training proteins; ignores the test protein."),
            ("Seed", "Random initialisation of training; several seeds show the run-to-run spread."),
        ]),
        ("Metrics", [
            ("AUROC", "Probability that a random positive is scored above a random negative: 0.5 = chance, 1.0 = perfect."),
            ("95% interval", "Bracket after a value: resampling of proteins with replacement (protein-level bootstrap)."),
            ("Spearman", "Rank correlation of two lists (1 = same ranking, 0 = unrelated)."),
        ]),
        ("Biology", [
            ("Pfam, RRM, KH", "Pfam: database of protein domain families; RRM and KH: the two most common RNA-binding domain families."),
            ("k-mer, motif", "k-mer: sequence of k nucleotides. Motif: short sequence pattern preferred by a protein (e.g. GCAUG for RBFOX2)."),
            ("icSHAPE", "Probing of RNA structure inside cells; used as an extra input channel in section 5."),
        ]),
    ]
    rows = []
    for grp, items in gl:
        rows.append(f'<tr class="grp"><td colspan="2">{grp}</td></tr>')
        rows += [f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in items]
    return ('<h2 class="part">0. Task, design and terms</h2>'
            '<p class="plain">Question of this report: do models that receive the protein sequence predict RNA binding for proteins that were not used for training, '
            'or do they use only RNA composition and the similarity of the protein to training proteins? All results are measured on in vitro data unless section 5 says otherwise.</p>'
            + figure(fig_design(), "Construction of the two tests and of the two data splits used throughout the report.")
            + '<div class="tw"><table class="gl"><thead><tr><th>Term</th><th>Meaning</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


# ----------------------------------------------------------------------------- content
def knn(v, e):
    return json.load(open(R / f"knn_baseline/{v}_summary.json"))["evals"][e]


def build() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    k3a_s, k3a_p, k3b_s, k3b_p = knn("v3a", "standard"), knn("v3a", "probe"), knn("v3b", "standard"), knn("v3b", "probe")
    kc = lambda e, s: f"{e['overall'][s]['auroc']:.3f} [{e['overall'][s]['ci'][0]:.2f}, {e['overall'][s]['ci'][1]:.2f}]"  # noqa: E731
    aff = json.load(open(R / "v3b_affinity/summary.json"))["scorers"]
    frames = parpi_frames()
    allp = pd.concat(frames)
    own = {a: json.load(open(R / f"rna_only_per_protein_honest/{a}_stats.json"))["median_test_auroc"] for a in ("htr_selex", "rbns", "rnacompete_eukarya", "rnacompete_rbpzoo")}
    blocks = []

    def add(part, question, key, verdict, test, body, caveat=""):
        blocks.append((part, question, key, verdict, test, body, caveat))

    P_A, P_B, P_C, P_D, P_E, P_F = ("1. Data and baselines", "2. Evaluation design", "3. Transfer to new proteins",
                                    "4. Attempts to improve transfer; controls", "5. In vivo data (eCLIP, PaRPI)", "6. Affinity")

    # ---------------- A
    add(P_A, "Data and splits", "Test proteins >=50% identical to training: 62% (v3a) -> 3% (v3b).", None,
        "MMseqs2 best identity of every test protein to the training proteins (script 55); v3b = clusters by single linkage (identity >=40% at coverage >=50%, or >=70% at >=10%).",
        table(["", "v3a (split by protein name)", "v3b (split by homology cluster)"], [
            ("proteins / pairs", "494 / 2.66 M (HTR-SELEX, RBNS, RNAcompete)", "same pairs, re-split: 160 clusters; train 375, val 55, test 64 proteins"),
            ("test proteins >=50% identical to training", "62% (7 are >=90%)", "3% (none >=70%)"),
            ("test proteins at 30-50% identity", "15 of 55", "53 of 64"),
            ("shared protein names / sequences with training", "1 / 1 (test)", "0 / 0"),
            ("test rows whose RNA also occurs in training", "69%", "73% (assay probe sets are reused across proteins)")],
              "Domain-level similarity at 30-50% remains for most v3b test proteins and cannot be removed inside RRM (at 30% identity all RRM proteins merge into one cluster of 400+)."),
        "Greedy clustering left near-identical proteins on both sides (e.g. PABPC1 / PABPC4), hence single linkage.")
    add(P_A, "Baselines", "Nearest-protein baseline 0.627 on the v3b probe; no baseline without protein information exceeds 0.500 there.", None,
        "Each baseline uses a different piece of information. The nearest-protein baseline fits a k-mer logistic regression on the RNAs of every training protein and scores a test pair with the model of the most similar training protein (MMseqs2).",
        table(["Baseline", "Information used", "v3a probe", "v3b probe", "v3b standard test"], [
            ("RNA prior", "RNA only (mean of all training protein models)", kc(k3a_p, "mean_lr"), kc(k3b_p, "mean_lr"), kc(k3b_s, "mean_lr")),
            ("RNA-only neural head (RNA-FM)", "RNA only", "-", "0.500", "0.651"),
            ("protein-only neural head (ESM-2)", "protein only", "-", "0.504", "0.507"),
            ("nearest protein, k-mer LR (nn1)", "sequence similarity to training proteins + their RNAs", kc(k3a_p, "nn1_lr"), kc(k3b_p, "nn1_lr"), kc(k3b_s, "nn1_lr")),
            ("top-3 neighbours (nn3)", "same, 3 neighbours", kc(k3a_p, "nn3_lr"), kc(k3b_p, "nn3_lr"), kc(k3b_s, "nn3_lr")),
            ("per-protein model on its own data (seen protein)", "the protein's own labelled RNAs, honest 60/20/20 split",
             "0.83 (seen protein, new RNAs, frozen head)", "-", f"median {own['htr_selex']:.2f} (HTR-SELEX), {own['rbns']:.2f} (RBNS), {own['rnacompete_eukarya']:.2f} (RNAcompete)")],
              "Brackets: 95% interval, protein-level bootstrap. The seen-protein row is the ceiling for a protein with its own data."),
        "'Cross-protein probe': the same RNA paired with another protein of the same assay where it is not a positive; RNA composition is identical in positives and negatives, so RNA-only and GC baselines give 0.500.")

    # ---------------- B
    add(P_B, "The standard test can be solved from the RNA alone", "RNA-only head: 0.651 on the standard test, 0.500 on the probe.", "S",
        "Neural heads on frozen embeddings with RNA only, protein only, or both (script 67, 3 seeds); same models on the standard test and on the probe.",
        figure(fig_scatter(), "Each dot is one model on v3b. A model that needs only RNA features sits far right and at 0.5 on the vertical axis; the probe is the only axis on which the protein matters."),
        "73% of test rows share an RNA with training rows. Single v3b split.")
    add(P_B, "Classic RPI benchmarks can be solved from one molecule", "RPI369: protein only 0.96 vs both 0.97. RPI488: RNA only 0.91 vs both 0.91.", "S",
        "RPISeq-style random forest with RNA-only, protein-only and combined features on RPI369/488/1446/2241 (script 60).",
        figure(fig_rpi(), "Random 5-fold. Under protein-disjoint folds RPI1446 and RPI2241 still give 0.79-0.83 with RNA only."),
        "Our re-implementation does not reproduce the published numbers (RPI369 random 0.974 vs 0.85 reported); the claim concerns dataset properties only. De-duplicated pairs, untuned forest.")

    # ---------------- C
    add(P_C, "Transfer to unseen proteins depends on sequence identity to training proteins", "V4 probe AUROC rises from 0.52 (no hit) to 0.88 (>=70% identity).", "S",
        "Probe AUROC of V2 and V4 (3 seeds each) and of the nearest-protein baseline, by identity bin of the test protein (v3a).",
        figure(fig_bins(), "v3a probe. Every model rises with identity and is at chance without a homolog; the nearest-protein baseline matches V4 from 30% identity on.")
        + table(["Pfam family (test proteins)", "V2", "V4"], [("RRM (33)", "0.761", "0.815"), ("KH (5)", "0.731", "0.813"), ("multi (5)", "0.601", "0.668"), ("other (6)", "0.482", "0.536"),
                                                            ("CCCH (2)", "0.366", "0.343"), ("all (55)", "0.686 +/- 0.015", "0.754 +/- 0.001")],
                "Families with many training proteins transfer (RRM 279, KH 37); the others are at chance. Within RRM: 30-50% identity V4 0.75, >=90% 0.92."),
        "Only 6 test proteins without a hit; non-RRM families have <=6 test proteins each.")
    add(P_C, "V2 and V4 exceed the nearest-protein baseline on the name-level split (v3a)", "V4 0.754 vs nn3 0.746 (probe); V2 0.686.", "N",
        "3 seeds each; paired V4 - V2 per seed +0.061, +0.085, +0.058 (V4 wins every seed).",
        table(["Scorer", "v3a standard", "v3a probe", "probe, per-protein median"], [
            ("RNA prior", kc(k3a_s, "mean_lr"), kc(k3a_p, "mean_lr"), "0.547"), ("nearest protein nn1", kc(k3a_s, "nn1_lr"), kc(k3a_p, "nn1_lr"), "0.831"),
            ("nearest protein nn3", kc(k3a_s, "nn3_lr"), kc(k3a_p, "nn3_lr"), "0.831"), ("V2 (3 seeds)", "~0.81", "0.686 +/- 0.015", "0.696"), ("V4 (3 seeds)", "~0.81", "0.754 +/- 0.001", "0.759")]),
        "V4 > V2 is robust; V4 > baseline is not (intervals overlap, baseline per-protein median is higher).")
    add(P_C, "V2, V4 and frozen-embedding heads exceed the nearest-protein baseline on the cluster split (v3b)", "v3b probe: nn1 0.627, V4 0.559, V2 0.527.", "N",
        "V2 and V4 retrained on v3b (seed 42), frozen-embedding heads (3 seeds), hybrids; all scored on the same probe.",
        figure(fig_forest(), "v3b probe AUROC. Only the nearest protein and the stack that contains it exceed 0.6; neural heads stay at 0.50-0.56.")
        + table(["Model", "v3b probe AUROC [95% interval]", "difference to nearest protein [95% interval]", "v3b standard test [95% interval]"], [
            ("nearest protein nn1", "0.627 [0.566, 0.689]", "-", "0.676 [0.610, 0.754]"),
            ("RNA prior", "0.500 [0.457, 0.538]", "-0.127 [-0.202, -0.052]", "0.591 [0.546, 0.640]"),
            ("V2 (seed 42)", "0.527 [0.498, 0.557]", "-0.100 [-0.176, -0.029]", "0.677"),
            ("V4 (seed 42)", "0.559 [0.527, 0.587]", "-0.068 [-0.147, +0.000]", "0.676"),
            ("RNA-only head (seed 42)", "0.500 [0.464, 0.537]", "-0.127 [-0.203, -0.047]", "0.652 [0.613, 0.698]"),
            ("protein-only head (seed 42)", "0.507 [0.478, 0.533]", "-0.120 [-0.197, -0.045]", "0.506 [0.499, 0.519]"),
            ("concat head (seed 42)", "0.528 [0.485, 0.571]", "-", "0.610 [0.567, 0.666]")],
                "Intervals: resampling of proteins, the same draws for all models (paired differences). A difference whose interval excludes 0 is a difference between models, not noise in the choice of test proteins. "
                "Intervals for the bilinear heads, the hybrid and the stack were cut off in the pasted run output.")
        + table(["Subset of test proteins (number)", "nearest protein", "V2", "V4"], [
            ("all (63)", "0.627", "0.527", "0.559"), ("identity 30-50% (53)", "0.635", "0.525", "0.564"), ("no hit (5)", "0.628 (= RNA prior)", "0.609", "0.564"),
            ("family RRM (43)", "0.651", "0.522", "0.556"), ("family KH (8)", "0.585", "0.501", "0.534"), ("family other (9)", "0.549", "0.597", "0.590"),
            ("assay RNAcompete Eukarya (22)", "0.579", "0.529", "0.561"), ("assay RNAcompete RBPZoo (28)", "0.677", "0.515", "0.557"), ("assay SELEX / RBNS (13)", "0.623", "0.550", "0.557")],
                "v3b probe AUROC by subset. V2 and V4 are above the nearest protein only in the 9 proteins of the family 'other' (differences +0.048 and +0.041, intervals include 0) and in the 5 proteins without a hit, "
                "where the nearest-protein score equals the RNA prior; per-subset intervals are wide (3-9 proteins)."),
        "V2 / V4 on v3b: one seed (seed 0 of V2 was stopped, V4 seeds 0 and 1 not started).")

    # ---------------- D
    add(P_D, "The null result on new clusters is caused by a pipeline defect", "Same head on training proteins with new RNAs: 0.83; on new clusters: 0.55.", "R",
        "Positive control: 5% of the training RNAs (all rows containing them) are held out; a cross-protein probe is built from them for the seen proteins.",
        table(["Check", "Result"], [("seen proteins, new RNAs (ordinary labels / cross-protein negatives)", "0.828 / 0.827"), ("unseen clusters, same runs", "0.550 / 0.542"),
                                    ("nearest protein recomputed inside the hybrid pipeline", "probe 0.6265, standard 0.6755 (script 59: 0.627 / 0.676)"),
                                    ("protein shuffled at test (bilinear heads)", "0.55 -> 0.48 and 0.57 -> 0.50")]),
        "Shows the head can learn protein-specific preferences; does not prove that no architecture can generalise.")
    add(P_D, "Frozen ESM-2 / RNA-FM embeddings, combination with the baseline and tuning improve transfer to new clusters", "Probe 0.50-0.56 (3-seed means); stack with nn1 +0.004; tuning 0.53-0.56 vs default 0.57.", "N",
        "Heads rna / protein / concat / bilinear (3 seeds), joint hybrid, held-out stacking, hyper-parameter sweep (scripts 67, 71).",
        table(["Scorer", "probe (mean +/- sd, 3 seeds)", "standard test"], [
            ("RNA-only head", "0.500 +/- 0.000", "0.651 +/- 0.001"), ("protein-only head", "0.504 +/- 0.004", "0.507 +/- 0.001"), ("concat head", "0.528 +/- 0.006", "0.618 +/- 0.009"),
            ("bilinear head, ordinary labels", "0.531 +/- 0.027", "0.623 +/- 0.032"), ("bilinear head + cross-protein negatives", "0.559 +/- 0.010", "0.531 +/- 0.015"),
            ("nn1 + head, weights fitted on val (seed 42)", "0.630", "0.676"), ("joint hybrid, trained together (seed 42)", "0.557", "0.533")],
              "Stack weights: nn1 0.137, head 0.012. In the joint hybrid the nn weight collapsed 1.0 -> 0.14 while the head memorised training proteins. "
              "Sweep (probe, seed 42; default 0.570): lr 3e-4 / 3e-3 0.544 / 0.557, hidden 128 / 512 0.539 / 0.549, dropout 0.1 / 0.5 0.529 / 0.556; weight-decay variants pending."),
        "Mean pooling over the whole protein dilutes domain signal (RBD-cropped pooling not tested). Best epochs of the heads are 1-5.")
    add(P_D, "Training with composition-neutral cross-protein negatives yields transfer to new clusters", "Frozen head 0.559 vs 0.531 with ordinary labels; one-hot V4 memorises.", "N",
        "Each training positive is paired with the same RNA and another training protein of the same assay (script 65 for V4, script 67 for frozen heads).",
        "<p class='plain'>One-hot V4: training loss 1.36 -> 0.77 while the validation probe stays at 0.49-0.52 (early stop at epoch 10, best 0.522). "
        "Frozen head: +0.03 on the probe, and the standard-test AUROC drops from 0.623 to 0.531 because the RNA shortcut is removed.</p>",
        "One seed for V4 and one set of hyper-parameters for the one-hot model.")
    add(P_D, "Transfer increases with the number of training clusters", "Probe 0.528 -> 0.559 from 25% to 100% of clusters.", "P",
        "Bilinear head with cross-protein negatives on 25/50/75/100% of the 160 clusters (two seeds per fraction, three at 100%).",
        figure(fig_curve(), "Learning curve. +0.03 for a fourfold increase of clusters; the curve is far from the nearest-protein baseline and from the seen-protein control."),
        "The one large cluster (220 RRM proteins) dominates sampling, so fractions are not comparable in protein count.")
    add(P_D, "The RNA motif of a new protein can be predicted from its sequence", "ESM-2 ridge: Spearman 0.179 vs 0.155 for the prior (CIs overlap).", "N",
        "Target: per-protein 5-mer log2 enrichment (1024 values); ESM-2 mean embedding; ridge and kNN; Spearman on 64 test proteins; probe AUROC when RNAs are scored with the predicted profile (script 73).",
        fig_bars(["prior", "kNN1", "kNN5", "ridge"], [0.155, 0.114, 0.183, 0.179], ["#aaaaaa", C["emb"], C["emb"], C["emb"]], "mean Spearman vs true profile", (0, 0.35),
                 err=[(0.155 - 0.070, 0.236 - 0.155), (0.114 - 0.037, 0.190 - 0.114), (0.183 - 0.107, 0.262 - 0.183), (0.179 - 0.107, 0.249 - 0.179)], w=6.6, h=3.6)
        + table(["Predictor", "Spearman mean [95% CI]", "top-20 overlap", "probe AUROC"], [("prior", "0.155 [0.070, 0.236]", "0.048", "0.500"), ("kNN1 (ESM-2 cosine)", "0.114 [0.037, 0.190]", "0.091", "0.552"),
                                                                                          ("kNN5", "0.183 [0.107, 0.262]", "0.089", "0.541"), ("ridge", "0.179 [0.107, 0.249]", "0.080", "0.509"),
                                                                                          ("oracle: the protein's own true profile", "-", "-", "0.902")],
                "The oracle shows a 5-mer profile is a sufficient RNA scorer if it were known: the bottleneck is predicting the profile of a new protein."),
        "Mean-pooled ESM-2 only; no RBD crop, residue-level model or structure; 375 training proteins.")
    add(P_D, "V2 uses the RNA-binding domain (motif recovery, alanine mutagenesis)", "Motif Spearman 0.22 on held-out vs 0.26 on training proteins; domain effect p = 0.17-0.24.", "N",
        "Script 61 on V2 seed 42 (v3a): predicted 5-mer motif vs measured enrichment (real-RNA insertion contexts); alanine replacement of domain-contact vs outside-contact residues; training proteins as positive control.",
        fig_bars(["V2, held-out", "V2, shuffled protein", "protein-agnostic prior", "nearest-protein motif", "V2, training proteins"], [0.221, 0.126, 0.171, 0.446, 0.259],
                 [C["v2"], "#bbbbbb", "#aaaaaa", C["nn"], C["v2"]], "motif Spearman", (0, 0.55), w=8.4, h=3.8, rot=20)
        + table(["Alanine mutagenesis", "held-out proteins", "training proteins"], [("domain-contact > outside-contact residues (share of proteins, p)", "48% (0.24)", "35% (0.17)")]),
        "V2 only. The positive control is not clearly better than held-out proteins, so the test is weakly sensitive; kept as a gate for any model that passes the probe.")
    g = gallery()
    if g:
        add(P_D, "Motif logos and saliency maps of V2 for proteins seen in training", "Per-protein Spearman and domain shares: table below the maps.", "I",
            "Script 75 on V2 trained on SELEX/RBNS data (generalized_v2); proteins with a known motif that were in its training data; measured vs predicted 5-mer profile, and four-view saliency maps with Pfam domains.",
            g, "Illustration of the method on a model trained on an earlier data version; these maps do not show transfer. Gate for any claim: the cross-protein probe, then these maps on a model that passes it.")

    # ---------------- E
    add(P_E, "Models trained on in vitro data transfer to eCLIP", "Zero-shot cross-protein eCLIP AUROC 0.50-0.53 for all models.", "N",
        "Skipper eCLIP rebuilt with real protein sequences; V2/V4 zero-shot, fine-tuned and trained on eCLIP; cross-protein negatives (scripts 41-53).",
        table(["Model", "protein_and_rna split", "cross_protein_neg"], [("V2 trained on eCLIP", "full 0.858, GC-matched 0.772", "0.485"), ("V2 fine-tuned from v3a", "GC-matched 0.711", "0.527"),
                                                                  ("V2 / V4 v3a zero-shot", "0.56-0.60 / 0.52-0.54", "0.528 / 0.50-0.51"), ("RNA-only k-mer LR on Skipper eCLIP", "0.81", "-")]),
        "Single seed, 60k sample. 5 of 24 eCLIP test proteins are >=90% identical to training; a null result cannot be created by removing them (re-report pending).")
    add(P_E, "In vivo benchmarks with random negatives measure RNA composition rather than interaction", "RNA-only k-mer LR: median 0.739 on 261 PaRPI datasets; pooled protein model = dataset identifier.", "S",
        "261 PaRPI datasets: protein-free k-mer LR (script 63); HeLa: pooled V2 with protein permutation and cross-RBP negatives (script 64).",
        fig_bars(["GC content\nalone", "RNA-only\nvs other RBPs' peaks", "RNA-only,\nrandom split", "HeLa pooled V2,\nprotein permuted", "HeLa pooled V2"], [0.397, 0.674, 0.739, 0.874, 0.881],
                 ["#bbbbbb", C["other"], C["other"], C["v2"], C["v2"]], "median AUROC", (0.3, 0.95), w=9.0, h=3.8)
        + table(["Scenario", "median AUROC"], [("held-out transcripts (RNA-only k-mer LR)", "0.741"), ("HeLa pooled V2, negatives = positives of the other 16 RBPs", "0.644")]),
        "PaRPI itself was not run. Its released code selects the epoch on the test split (code reading, not executed).")
    add(P_E, "A protein-free CNN matches published protein-free methods on PaRPI data", f"Median over 261 datasets: our CNN {allp.rna_cnn.median():.3f}, +icSHAPE {allp.rna_shape_cnn.median():.3f}, PaRPI {allp.PaRPI.median():.3f}.", "P",
        "Per-dataset CNN (one-hot, 101 nt) with and without in vivo icSHAPE, random 80/20 split with a separate validation split for epoch selection (script 64, one seed). Published values: PaRPI Supplementary Data 1.",
        '<div class="two">' + figure(fig_parpi_methods(allp), "Median AUROC over all 261 datasets.") + figure(fig_parpi_scatter(allp), "One dot per dataset; below the diagonal PaRPI is better.") + "</div>"
        + table(["Cell line (datasets)", "our CNN", "our CNN + icSHAPE", "DeepBind", "iDeep", "PrismNet", "HDRNet", "PaRPI"],
                [(f"{g_.cell.iloc[0]} ({len(g_)})", *(f"{g_[c].median():.3f}" for c in ("rna_cnn", "rna_shape_cnn", "deepbind", "iDeep", "Prismnet", "HDRNet", "PaRPI"))) for g_ in frames]
                + [(f"all ({len(allp)})", *(f"{allp[c].median():.3f}" for c in ("rna_cnn", "rna_shape_cnn", "deepbind", "iDeep", "Prismnet", "HDRNet", "PaRPI")))],
                f"Our CNN beats DeepBind on {(allp.rna_cnn >= allp.deepbind).mean():.0%} of the datasets and iDeep on {(allp.rna_cnn >= allp.iDeep).mean():.0%}. "
                f"Gap PaRPI minus our CNN + icSHAPE, all datasets: {allp.PaRPI.median() - allp.rna_shape_cnn.median():.3f}."),
        "One seed. Part of the gap may come from test-based epoch selection in the published code; not measured yet (planned: B0).")
    add(P_E, "The same protein has concordant eCLIP and in vitro specificity", "eCLIP-trained k-mer model: 0.67 on the protein's in vitro rows vs 0.58 prior (+0.09); own data 0.95.", "P",
        "50 proteins present in PaRPI eCLIP data and in v3b (in vitro rows only); k-mer LR on eCLIP vs genomic negatives or vs other RBPs' peaks (script 74).",
        figure(fig_reverse(), "Per-protein AUROC on in vitro rows, sorted by the eCLIP-trained model. Highest: PTBP1, ELAVL1, TRA2A, HNRNPA1, TIA1, PCBP2. Lowest five: TAF15, HNRNPF, SAFB2, SF3B4, PUM1.")
        + table(["Scorer", "mean [95% CI over proteins]", "median"], [("eCLIP model, genomic negatives", "0.670 [0.611, 0.723]", "0.695"), ("eCLIP model, cross-RBP negatives", "0.671 [0.613, 0.726]", "0.677"),
                                                                    ("protein-independent in vitro prior", "0.579 [0.517, 0.636]", "0.642"), ("in vitro own data, 5-fold CV", "0.947 [0.929, 0.961]", "0.958")],
                "Paired difference to the prior: +0.091 [0.028, 0.159] (genomic), +0.092 [0.016, 0.163] (cross-RBP). By assay: RNAcompete Eukarya (24 proteins) 0.73-0.75, SELEX/RBNS (21) 0.58-0.59."),
        "k-mer frequencies only (the PUM1 8-mer is invisible to them); some proteins are also in the v3b training set; window lengths differ (101 nt vs 20-60 nt).")

    # ---------------- F
    add(P_F, "Homology transfer predicts RNAcompete probe intensity", f"Spearman {aff['nn1']['mean']:.2f} (nn1) vs {aff['prior']['mean']:.2f} (prior); own data {aff['own_cv']['mean']:.2f}.", "S",
        "52 v3b test proteins with RNAcompete data (~2000 probes each); per-protein Spearman between score and probe intensity (script 72).",
        fig_bars(["RNA prior", "nearest protein (nn1)", "own data (5-fold)"], [aff["prior"]["mean"], aff["nn1"]["mean"], aff["own_cv"]["mean"]], ["#aaaaaa", C["nn"], C["own"]],
                 "mean Spearman with probe intensity", (0, 0.7), err=[(aff[k]["mean"] - aff[k]["ci"][0], aff[k]["ci"][1] - aff[k]["mean"]) for k in ("prior", "nn1", "own_cv")], w=6.6, h=3.8)
        + table(["Scorer", "median", "mean [95% CI over proteins]"], [(n, f"{aff[k]['median']:.3f}", f"{aff[k]['mean']:.3f} [{aff[k]['ci'][0]:.3f}, {aff[k]['ci'][1]:.3f}]")
                                                                    for n, k in (("RNA prior", "prior"), ("nearest protein (binary-label k-mer LR)", "nn1"), ("own data, ridge, 5-fold by probe", "own_cv"))],
                "For orientation, NPBIP (2026) reports Pearson 0.32-0.41 on RNAcompete with a different protocol and metric; not a like-for-like comparison."),
        "Only baselines scored; neural models not run on these probes; transfer models trained on intensity not tried.")

    # ----------------------------------------------------------------------------- html
    cnt = {"M": 0, "H": 0, "I": 0}
    glance, secs, cur = [], [], None
    for part, q, key, v, test, body, cav in blocks:
        kind = "M" if v is None else "I" if v == "I" else "H"
        cnt[kind] += 1
        bid = f"{kind}{cnt[kind]}"
        if part != cur:
            cur = part
            glance.append(f'<tr class="part"><td colspan="4">{html.escape(part)}</td></tr>')
            secs.append(f'<h2 class="part">{html.escape(part)}</h2>')
        label, cls = VERDICT[v] if v else ("", "")
        chip = f'<span class="chip {cls}">{label}</span>' if v else ""
        glance.append(f'<tr><td><a href="#{bid}">{bid}</a></td><td>{html.escape(q)}</td><td>{chip}</td><td>{html.escape(key)}</td></tr>')
        secs.append(f'<section id="{bid}"><h3><span class="qn">{bid}</span> {html.escape(q)} {chip}</h3>'
                    f'<p class="test"><b>{"Method" if v is None else "Test"}.</b> {html.escape(test)}</p>{body}'
                    + (f'<details><summary>Caveats</summary><p>{cav}</p></details>' if cav else "") + "</section>")

    findings = [
        "On the name-level split (v3a), V4 (CNN with a bilinear protein-RNA layer) reaches AUROC 0.754 on the cross-protein probe. This tracks the sequence identity of the test protein to training proteins "
        "(0.52 without a homolog, 0.88 at >= 70% identity), and the nearest-protein baseline, which contains no protein model, reaches 0.746.",
        "After removing homologs (v3b), the nearest-protein baseline scores 0.627; V2 / V4 (CNNs trained on one-hot sequences) score 0.527 / 0.559 and heads on frozen ESM-2 + RNA-FM embeddings 0.50-0.56 (3 seeds). "
        "Adding the heads to the baseline raises the score by 0.004.",
        "The same frozen-embedding heads reach 0.83 on proteins seen in training with new RNAs: training works, transfer to unseen homology clusters does not.",
        "The standard test is solvable from the RNA alone (RNA-only head 0.651 on the standard test, 0.500 on the probe). Classic RPI benchmarks behave similarly.",
        "Predicting a protein's RNA motif from its sequence is not better than the average motif (ridge regression on ESM-2: Spearman 0.179 vs 0.155); the measured motif would give 0.90.",
        "For proteins with in vivo data, k-mer models trained on eCLIP score the same protein's in vitro RNAs at 0.67 (RNA prior 0.58, in vitro own data 0.95).",
    ]
    css = """
:root{color-scheme:light;--bg:#fff;--fg:#17191c;--mut:#555d68;--line:#d6dbe1;--card:#f4f6f8;--ok:#0b6e4f;--no:#a4262c;--mid:#8a5a00;--link:#0b57d0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 20px 90px}h1{font-size:30px;margin:0 0 6px;line-height:1.2}
h2.part{font-size:24px;margin:56px 0 4px;padding-top:18px;border-top:3px solid var(--fg)}h3{font-size:20px;margin:34px 0 10px;line-height:1.3}
.qn{display:inline-block;background:var(--fg);color:var(--bg);border-radius:6px;padding:0 8px;margin-right:6px;font-size:16px}
.meta{color:var(--mut);margin:0 0 18px;font-size:15px}a{color:var(--link)}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 22px 12px;margin:18px 0}.box h2{font-size:20px;margin:12px 0 4px}.box li{margin:7px 0}
.test{margin:6px 0 10px}.plain{margin:10px 0}
.tw{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:15.5px}th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--card);font-weight:600}tr.grp td{background:var(--card);font-weight:700;border-top:2px solid var(--line)}table.gl td:first-child{width:24%;white-space:nowrap}tr.part td{background:var(--fg);color:var(--bg);font-weight:600;font-size:16px}
.note{color:var(--mut);font-size:15px;margin:6px 0 10px}figure{margin:14px 0}figure svg,figure img{max-width:100%;height:auto}svg text{fill:var(--fg)!important}
svg [style*="stroke: #000000"]{stroke:var(--fg)!important}figcaption{color:var(--mut);font-size:15px;margin-top:4px}
.chip{font-size:14px;font-weight:600;padding:1px 10px;border-radius:12px;border:1.5px solid currentColor;white-space:nowrap;vertical-align:middle}
.chip.ok{color:var(--ok)}.chip.no{color:var(--no)}.chip.mid{color:var(--mid)}
details{margin:8px 0;color:var(--mut);font-size:15px}summary{cursor:pointer;font-weight:600}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}
@media (max-width:760px){.two{grid-template-columns:1fr}body{font-size:16px}}
"""
    open_items = [("Intervals and subsets for the bilinear heads, hybrid and stack on the v3b probe", "computed on the VM (script 69); columns cut off in the pasted output"),
                  ("Weight-decay variants of the hyper-parameter sweep", "running on the VM"),
                  ("V2 / V4 on v3b: seeds 0 and 1", "not run (V2 seed 0 stopped, V4 not started)"),
                  ("Gap to the published PaRPI numbers with test-based epoch selection (B0)", "planned"),
                  ("eCLIP re-report without proteins >=90% identical to training; per-assay breakdown of V2 / V4", "needs VM score files"),
                  ("Saliency maps for a model that passes the probe gate", "tools ready (scripts 68, 75); none passes the gate yet"),
                  ("Variant effects and disease link (B3-B4), PU-style negatives, structure-derived priors (Track C)", "not started"),
                  ("PaRPI and RPIEmbeddor 2.0 themselves", "not reproduced (code without data / checkpoints); compared via published numbers and our own models")]
    head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Transfer Audit Draft</title><style>{css}</style></head><body><main>'
            f'<h1>Protein-RNA binding models: what transfers to new proteins</h1>'
            f'<p class="meta">Draft, {today}, repository commit {commit}. Every number is traceable to EXPERIMENT_LOG.md or a results file; single-seed results are marked. '
            f'Sections: 1 data and baselines, 2 evaluation design, 3 transfer to new proteins, 4 attempts and controls, 5 in vivo data, 6 affinity, 7 not done. M = method, H = hypothesis, I = illustration.</p>'
            f'{terms_section()}<div class="box"><h2>Main results</h2><ol>' + "".join(f"<li>{html.escape(x)}</li>" for x in findings) + "</ol></div>"
            f'{figure(fig_ladder(), "AUROC as the test protein becomes less known. Dark bars: seen proteins; blue: neural models; orange: nearest-protein baseline. Values are taken from sections 1-4.")}'
            f'<h2 class="part">Table 1. Hypotheses and outcomes</h2><div class="tw"><table><thead><tr><th>No.</th><th>Statement</th><th>Outcome</th><th>Key number</th></tr></thead><tbody>'
            + "".join(glance) + "</tbody></table></div>")
    tail = '<h2 class="part">7. Not done or pending</h2>' + table(["Item", "Status"], open_items)
    return head + "".join(secs) + tail + "</main></body></html>"


if __name__ == "__main__":
    out = ROOT / "reports" / "draft_report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
