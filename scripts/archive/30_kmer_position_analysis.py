#!/usr/bin/env python3
"""
30_kmer_position_analysis.py
----------------------------
Per-experiment kmer_position summary and plots for top/bottom exemplars.

Outputs (default: results/top_bottom_examples/kmer_position/):
  kmer_position_by_experiment.tsv   — per protein/experiment stats (all datasets)
  htr_position_summary.tsv          — HTR-SELEX detail + example flags
  htr_motif_at_start_proteins.tsv   — all 93 HTR proteins: motif-at-read-start vs pos0
  htr_pos0_reconciliation.tsv       — how 9.7% pos0 breaks down by protein group
  *.png                             — position distributions and HTR examples

Usage:
    python scripts/30_kmer_position_analysis.py

    python scripts/30_kmer_position_analysis.py \\
        --summary_tsv results/top_bottom_examples/all_protocols_summary.tsv \\
        --out_dir results/top_bottom_examples/kmer_position
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Proteins highlighted in the HTR position-0 analysis
HTR_EXAMPLE_PROTEINS = ("CSTF2", "CSTF2T", "DAZ3", "DAZL", "RBMY1F", "BOLL", "SNRNP70", "HNRNPC")

DATASET_ORDER = [
    "RNAcompete_Eukarya",
    "RNAcompete_RBPZoo",
    "RNAcompete_ucRBP23",
    "HTR-SELEX",
    "RBNS",
]

DATASET_COLORS = {
    "RNAcompete_Eukarya": "#4e79a7",
    "RNAcompete_RBPZoo": "#59a14f",
    "RNAcompete_ucRBP23": "#76b7b2",
    "HTR-SELEX": "#f28e2b",
    "RBNS": "#e15759",
}


def summarize_group(g: pd.DataFrame) -> dict:
    pos = g[g["split"] == "positive"]
    n_pos = len(pos)
    n_pos0 = int((pos["kmer_position"] == 0).sum()) if n_pos else 0
    starts3 = pos["rna_sequence"].str[:3]
    ag_start = int(pos["rna_sequence"].str.startswith("AG").sum()) if n_pos else 0
    return {
        "protocol": g["protocol"].iloc[0] if len(g) else "",
        "dataset": g["dataset"].iloc[0] if len(g) else "",
        "protein_name": g["protein_name"].iloc[0] if len(g) else "",
        "experiment_id": g["experiment_id"].iloc[0] if len(g) else "",
        "n_positive": n_pos,
        "n_negative": int((g["split"] == "negative").sum()),
        "n_pos0": n_pos0,
        "pct_pos0": round(100 * n_pos0 / n_pos, 1) if n_pos else np.nan,
        "median_kmer_position": float(pos["kmer_position"].median()) if n_pos else np.nan,
        "mean_kmer_position": round(float(pos["kmer_position"].mean()), 1) if n_pos else np.nan,
        "top_matched_kmer": pos["matched_kmer"].mode().iloc[0] if n_pos else "",
        "common_read_start3": starts3.mode().iloc[0] if n_pos and not starts3.empty else "",
        "pct_read_start_AG": round(100 * ag_start / n_pos, 1) if n_pos else np.nan,
        "n_unique_read_start3": int(starts3.nunique()) if n_pos else 0,
        "probe_length": int(pos["probe_length"].mode().iloc[0]) if n_pos else np.nan,
    }


def motif_at_read_start(seq: str, kmer: str) -> bool:
    if not kmer or kmer == "-":
        return False
    return seq.upper().replace("T", "U").startswith(kmer.upper().replace("T", "U"))


def build_htr_motif_at_start_table(htr_pos: pd.DataFrame) -> pd.DataFrame:
    """Per-protein table: reads where top 7-mer starts the 40 nt window (= position 0)."""
    rows = []
    for (protein, eid), g in htr_pos.groupby(["protein_name", "experiment_id"], sort=True):
        g = g.copy()
        g["motif_at_read_start"] = [
            motif_at_read_start(s, k) for s, k in zip(g["rna_sequence"], g["matched_kmer"])
        ]
        n = len(g)
        n_start = int(g["motif_at_read_start"].sum())
        n_pos0 = int((g["kmer_position"] == 0).sum())
        rows.append(
            {
                "protein_name": protein,
                "experiment_id": eid,
                "top_matched_kmer": g["matched_kmer"].mode().iloc[0],
                "n_positive_exemplars": n,
                "n_reads_motif_at_start": n_start,
                "pct_reads_motif_at_start": round(100 * n_start / n, 1),
                "n_reads_kmer_position_0": n_pos0,
                "pct_reads_kmer_position_0": round(100 * n_pos0 / n, 1),
                "all_exemplars_motif_at_start": n_start == n,
                "any_exemplar_motif_at_start": n_start > 0,
                "pos0_equals_motif_at_start": n_start == n_pos0,
            }
        )
    tbl = pd.DataFrame(rows)
    return tbl.sort_values(
        ["any_exemplar_motif_at_start", "n_reads_motif_at_start", "protein_name"],
        ascending=[False, False, True],
    )


def build_htr_pos0_reconciliation(htr_pos: pd.DataFrame, protein_tbl: pd.DataFrame) -> pd.DataFrame:
    htr_pos = htr_pos.copy()
    htr_pos["motif_at_read_start"] = [
        motif_at_read_start(s, k) for s, k in zip(htr_pos["rna_sequence"], htr_pos["matched_kmer"])
    ]
    n_total = len(htr_pos)
    n_pos0 = int((htr_pos["kmer_position"] == 0).sum())
    n_start = int(htr_pos["motif_at_read_start"].sum())

    any_proteins = set(protein_tbl.loc[protein_tbl["any_exemplar_motif_at_start"], "protein_name"])
    reads_any = htr_pos[htr_pos["protein_name"].isin(any_proteins)]
    reads_none = htr_pos[~htr_pos["protein_name"].isin(any_proteins)]

    rows = [
        {
            "category": "all_HTR_positive_exemplars",
            "n_reads": n_total,
            "n_position_0": n_pos0,
            "pct_position_0": round(100 * n_pos0 / n_total, 2),
            "n_motif_at_read_start": n_start,
            "note": "position 0 == read starts with matched 7-mer (identical for all 462 reads)",
        },
        {
            "category": "proteins_with_any_motif_at_start_read",
            "n_reads": len(reads_any),
            "n_position_0": int((reads_any["kmer_position"] == 0).sum()),
            "pct_position_0": round(100 * (reads_any["kmer_position"] == 0).mean(), 2),
            "n_motif_at_read_start": int(reads_any["motif_at_read_start"].sum()),
            "note": f"{len(any_proteins)} proteins; enriched motif sometimes at 5' of read",
        },
        {
            "category": "proteins_with_no_motif_at_start_read",
            "n_reads": len(reads_none),
            "n_position_0": int((reads_none["kmer_position"] == 0).sum()),
            "pct_position_0": round(100 * (reads_none["kmer_position"] == 0).mean(), 2),
            "n_motif_at_read_start": int(reads_none["motif_at_read_start"].sum()),
            "note": f"{htr_pos['protein_name'].nunique() - len(any_proteins)} proteins; 0% position 0",
        },
        {
            "category": "proteins_all_5_exemplars_motif_at_start",
            "n_reads": int(protein_tbl.loc[protein_tbl["all_exemplars_motif_at_start"], "n_positive_exemplars"].sum()),
            "n_position_0": int(protein_tbl.loc[protein_tbl["all_exemplars_motif_at_start"], "n_reads_kmer_position_0"].sum()),
            "pct_position_0": 100.0,
            "n_motif_at_read_start": int(
                protein_tbl.loc[protein_tbl["all_exemplars_motif_at_start"], "n_reads_motif_at_start"].sum()
            ),
            "note": f"{protein_tbl['all_exemplars_motif_at_start'].sum()} proteins (DAZ3, RBMY1F)",
        },
    ]
    return pd.DataFrame(rows)


def plot_htr_pos0_reconciliation(htr_pos: pd.DataFrame, protein_tbl: pd.DataFrame, out_dir: Path) -> None:
    htr_pos = htr_pos.copy()
    htr_pos["motif_at_read_start"] = [
        motif_at_read_start(s, k) for s, k in zip(htr_pos["rna_sequence"], htr_pos["matched_kmer"])
    ]
    any_proteins = set(protein_tbl.loc[protein_tbl["any_exemplar_motif_at_start"], "protein_name"])
    reads_any = htr_pos[htr_pos["protein_name"].isin(any_proteins)]
    reads_none = htr_pos[~htr_pos["protein_name"].isin(any_proteins)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: overall 462 reads
    ax = axes[0]
    n_pos0 = (htr_pos["kmer_position"] == 0).sum()
    n_other = len(htr_pos) - n_pos0
    ax.bar(
        ["motif at read start\n(position 0)", "motif elsewhere\n(position > 0)"],
        [n_pos0, n_other],
        color=["#d62728", "#aec7e8"],
        edgecolor="white",
    )
    ax.set_ylabel("Positive exemplar reads")
    ax.set_title(f"All HTR-SELEX (n={len(htr_pos)})\n{n_pos0}/{len(htr_pos)} = {100*n_pos0/len(htr_pos):.1f}% at position 0")
    for i, v in enumerate([n_pos0, n_other]):
        ax.text(i, v + 5, str(v), ha="center", fontsize=10)

    # Panel B: split by protein group
    ax = axes[1]
    groups = [
        f"20 proteins\n(any motif-at-start)\n(n={len(reads_any)} reads)",
        f"73 proteins\n(no motif-at-start)\n(n={len(reads_none)} reads)",
    ]
    pos0_counts = [
        (reads_any["kmer_position"] == 0).sum(),
        (reads_none["kmer_position"] == 0).sum(),
    ]
    other_counts = [len(reads_any) - pos0_counts[0], len(reads_none) - pos0_counts[1]]
    x = np.arange(2)
    w = 0.55
    ax.bar(x, pos0_counts, w, label="position 0", color="#d62728", edgecolor="white")
    ax.bar(x, other_counts, w, bottom=pos0_counts, label="position > 0", color="#aec7e8", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("Positive exemplar reads")
    ax.set_title("By protein group\n(all position-0 reads come from the 20-protein subset)")
    ax.legend(fontsize=8, loc="upper right")
    for i, (p, t) in enumerate(zip(pos0_counts, [len(reads_any), len(reads_none)])):
        if p:
            ax.text(i, p / 2, f"{p}\n({100*p/t:.0f}%)", ha="center", va="center", fontsize=9, color="white")

    fig.suptitle(
        "HTR-SELEX: position 0 = read starts with top 7-mer (not a fixed library prefix)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "htr_pos0_reconciliation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_group(g) for _, g in summary.groupby(["dataset", "protein_name", "experiment_id"], sort=True)]
    tbl = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DATASET_ORDER)}
    tbl["_ord"] = tbl["dataset"].map(order).fillna(99)
    return tbl.sort_values(["_ord", "pct_pos0", "protein_name"], ascending=[True, False, True]).drop(
        columns="_ord"
    )


def plot_position_by_dataset(summary: pd.DataFrame, out_dir: Path) -> None:
    pos = summary[summary["split"] == "positive"].copy()
    datasets = [d for d in DATASET_ORDER if d in pos["dataset"].unique()]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 2.2 * len(datasets)), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = pos[pos["dataset"] == ds]
        max_pos = int(sub["kmer_position"].max())
        bins = np.arange(-0.5, max(max_pos, 20) + 1.5, 1)
        ax.hist(
            sub["kmer_position"],
            bins=bins,
            color=DATASET_COLORS.get(ds, "#888888"),
            edgecolor="white",
            linewidth=0.4,
        )
        n = len(sub)
        n0 = (sub["kmer_position"] == 0).sum()
        med = sub["kmer_position"].median()
        ax.set_ylabel(ds.replace("RNAcompete_", "RC\n"), fontsize=8)
        ax.text(
            0.98,
            0.85,
            f"n={n}, pos0={n0} ({100*n0/n:.1f}%), median={med:.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
    axes[-1].set_xlabel("kmer_position (0-based)")
    fig.suptitle("K-mer position distribution — positive exemplars", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "position_distribution_by_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_htr_pos0_by_protein(htr_tbl: pd.DataFrame, out_dir: Path) -> None:
    sub = htr_tbl.sort_values(["n_pos0", "pct_pos0", "protein_name"], ascending=[False, False, True])
    fig_h = max(6, 0.22 * len(sub))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(sub))
    colors = ["#d62728" if p in HTR_EXAMPLE_PROTEINS else "#f28e2b" for p in sub["protein_name"]]
    ax.barh(y, sub["n_pos0"], color=colors, edgecolor="white", height=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["protein_name"], fontsize=7)
    ax.set_xlabel("Number of positive exemplars at position 0 (of 5)")
    ax.set_xlim(0, 5.4)
    ax.set_xticks(range(6))
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_title("HTR-SELEX: position-0 exemplars per protein (red = discussed examples)")
    fig.tight_layout()
    fig.savefig(out_dir / "htr_pos0_by_protein.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_htr_read_starts(htr_pos: pd.DataFrame, out_dir: Path) -> None:
    starts = htr_pos["rna_sequence"].str[:3].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(starts.index, starts.values, color="#f28e2b", edgecolor="white")
    ax.set_xlabel("Read 5′ 3-mer")
    ax.set_ylabel("Count (positive exemplars)")
    pct = 100 * starts.iloc[0] / len(htr_pos)
    ax.set_title(f"HTR-SELEX read starts — no fixed prefix (top = {starts.index[0]}, {pct:.1f}%)")
    fig.tight_layout()
    fig.savefig(out_dir / "htr_read_start_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rnacompete_starts(rc_pos: pd.DataFrame, out_dir: Path) -> None:
    starts = rc_pos["rna_sequence"].str[:3].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(starts.index, starts.values, color="#4e79a7", edgecolor="white")
    ax.set_xlabel("Probe 5′ 3-mer")
    ax.set_ylabel("Count")
    ax.set_title("RNAcompete (all panels): fixed AGA/AGG 5′ context")
    fig.tight_layout()
    fig.savefig(out_dir / "rnacompete_read_start_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_htr_examples(summary: pd.DataFrame, out_dir: Path) -> None:
    htr = summary[(summary["dataset"] == "HTR-SELEX") & (summary["split"] == "positive")]
    proteins = [p for p in HTR_EXAMPLE_PROTEINS if p in htr["protein_name"].values]
    n = len(proteins)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.35 * n))
    if n == 1:
        axes = [axes]
    for ax, protein in zip(axes, proteins):
        sub = htr[htr["protein_name"] == protein].sort_values("rank")
        ylabels = []
        for _, row in sub.iterrows():
            seq = row["rna_sequence"]
            kmer = row["matched_kmer"]
            pos = int(row["kmer_position"])
            rank = int(row["rank"])
            display = list("." * len(seq))
            if kmer and kmer != "-" and 0 <= pos <= len(seq) - len(kmer):
                for i in range(pos, pos + len(kmer)):
                    display[i] = seq[i]
            ylabels.append(f"rank {rank}  pos={pos}")
            ax.text(
                0.01,
                rank - 1,
                "".join(display),
                fontfamily="monospace",
                fontsize=7,
                va="center",
                transform=ax.get_yaxis_transform(),
            )
        kmer = sub["matched_kmer"].iloc[0]
        n0 = (sub["kmer_position"] == 0).sum()
        ax.set_yticks(range(1, len(sub) + 1))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_ylim(0.3, len(sub) + 0.7)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(f"{protein} — top kmer {kmer} — {n0}/{len(sub)} at position 0", fontsize=9, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("HTR-SELEX: motif position in 40 nt reads (highlighted = matched 7-mer)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "htr_motif_position_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_htr_pos0_vs_median_scatter(htr_tbl: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = htr_tbl["median_kmer_position"]
    y = htr_tbl["pct_pos0"]
    colors = ["#d62728" if p in HTR_EXAMPLE_PROTEINS else "#4e79a7" for p in htr_tbl["protein_name"]]
    ax.scatter(x, y, c=colors, s=40, alpha=0.85, edgecolors="white", linewidths=0.5)
    for _, row in htr_tbl[htr_tbl["protein_name"].isin(HTR_EXAMPLE_PROTEINS)].iterrows():
        ax.annotate(
            row["protein_name"],
            (row["median_kmer_position"], row["pct_pos0"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Median kmer_position")
    ax.set_ylabel("% exemplars at position 0")
    ax.set_title("HTR-SELEX per experiment: position preference (red = examples)")
    ax.set_ylim(-2, 105)
    fig.tight_layout()
    fig.savefig(out_dir / "htr_position_preference_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--summary_tsv",
        type=Path,
        default=Path("results/top_bottom_examples/all_protocols_summary.tsv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/top_bottom_examples/kmer_position"),
    )
    args = parser.parse_args()

    if not args.summary_tsv.exists():
        print(f"ERROR: summary not found: {args.summary_tsv}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_tsv, sep="\t")

    tbl = build_summary_table(summary)
    tbl.to_csv(args.out_dir / "kmer_position_by_experiment.tsv", sep="\t", index=False)

    htr_tbl = tbl[tbl["dataset"] == "HTR-SELEX"].copy()
    htr_tbl["is_discussed_example"] = htr_tbl["protein_name"].isin(HTR_EXAMPLE_PROTEINS)
    htr_tbl.to_csv(args.out_dir / "htr_position_summary.tsv", sep="\t", index=False)

    pos = summary[summary["split"] == "positive"]
    htr_pos = pos[pos["dataset"] == "HTR-SELEX"]
    motif_tbl = build_htr_motif_at_start_table(htr_pos)
    motif_tbl.to_csv(args.out_dir / "htr_motif_at_start_proteins.tsv", sep="\t", index=False)
    recon = build_htr_pos0_reconciliation(htr_pos, motif_tbl)
    recon.to_csv(args.out_dir / "htr_pos0_reconciliation.tsv", sep="\t", index=False)
    plot_htr_pos0_reconciliation(htr_pos, motif_tbl, args.out_dir)
    plot_position_by_dataset(summary, args.out_dir)
    plot_htr_pos0_by_protein(htr_tbl, args.out_dir)
    plot_htr_read_starts(pos[pos["dataset"] == "HTR-SELEX"], args.out_dir)
    rc_pos = pos[pos["dataset"].str.startswith("RNAcompete")]
    if len(rc_pos):
        plot_rnacompete_starts(rc_pos, args.out_dir)
    plot_htr_examples(summary, args.out_dir)
    plot_htr_pos0_vs_median_scatter(htr_tbl, args.out_dir)

    n_any = int(motif_tbl["any_exemplar_motif_at_start"].sum())
    n_pos0 = int((htr_pos["kmer_position"] == 0).sum())
    print(f"Wrote tables and plots to {args.out_dir}/")
    print(f"  kmer_position_by_experiment.tsv ({len(tbl)} experiments)")
    print(f"  htr_position_summary.tsv ({len(htr_tbl)} HTR experiments)")
    print(f"  htr_motif_at_start_proteins.tsv ({n_any}/93 proteins with any motif-at-start read)")
    print(f"  htr_pos0_reconciliation.tsv (overall {n_pos0}/{len(htr_pos)} = {100*n_pos0/len(htr_pos):.1f}% pos0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
