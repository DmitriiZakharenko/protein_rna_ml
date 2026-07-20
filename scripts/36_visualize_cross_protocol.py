#!/usr/bin/env python3
"""
36_visualize_cross_protocol.py
------------------------------
Figures for cross-protocol comparison outputs from scripts 33–35.

Usage:
    python scripts/36_visualize_cross_protocol.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str | Path, base: Path = ROOT) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def protocol_order(cols: list[str]) -> list[str]:
    preferred = [
        "htr_selex",
        "rbns",
        "rnacompete_eukarya",
        "rnacompete_rbpzoo",
        "rnacompete_ucrbp",
        "eclip",
    ]
    rest = sorted(c for c in cols if c not in preferred)
    return [c for c in preferred if c in cols] + rest


def fig_transfer_heatmap(transfer: pd.DataFrame, out: Path) -> None:
    if transfer.empty:
        return
    g = (
        transfer.groupby(["train_protocol", "test_protocol"])["auroc"]
        .median()
        .reset_index()
    )
    protocols = protocol_order(
        sorted(set(g["train_protocol"]) | set(g["test_protocol"]))
    )
    mat = (
        g.pivot(index="train_protocol", columns="test_protocol", values="auroc")
        .reindex(index=protocols, columns=protocols)
    )
    counts = (
        transfer.groupby(["train_protocol", "test_protocol"]).size().reset_index(name="n")
    )
    nmat = (
        counts.pivot(index="train_protocol", columns="test_protocol", values="n")
        .reindex(index=protocols, columns=protocols)
    )

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": "Median AUROC"},
    )
    # Overlay sample sizes as secondary annotation in title note
    ax.set_title("Cross-protocol transfer (median AUROC)\ntrain → test")
    ax.set_xlabel("Test protocol")
    ax.set_ylabel("Train protocol")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # companion count heatmap
    out_n = out.with_name(out.stem + "_n.png")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(nmat.fillna(0).astype(int), annot=True, fmt="d", cmap="Greys", ax=ax)
    ax.set_title("Number of proteins per transfer direction")
    ax.set_xlabel("Test protocol")
    ax.set_ylabel("Train protocol")
    fig.tight_layout()
    fig.savefig(out_n, dpi=150)
    plt.close(fig)


def fig_within_vs_transfer(within: pd.DataFrame, transfer: pd.DataFrame, out: Path) -> None:
    if within.empty or transfer.empty:
        return
    # For each transfer row, attach within-AUROC of train and test protocols
    w = within.set_index(["protein_key", "protocol"])["auroc"].to_dict()
    rows = []
    for _, r in transfer.iterrows():
        w_train = w.get((r["protein_key"], r["train_protocol"]), np.nan)
        w_test = w.get((r["protein_key"], r["test_protocol"]), np.nan)
        rows.append(
            {
                "protein_key": r["protein_key"],
                "transfer_auroc": r["auroc"],
                "within_train_auroc": w_train,
                "within_test_auroc": w_test,
                "within_mean": np.nanmean([w_train, w_test]),
                "domain_class": r.get("domain_class", "unknown"),
                "pair": f"{r['train_protocol']}→{r['test_protocol']}",
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["transfer_auroc", "within_mean"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(df["within_mean"], df["transfer_auroc"], alpha=0.55, s=28, edgecolors="none")
    lims = [0.45, 1.02]
    ax.plot(lims, lims, ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Mean within-protocol AUROC")
    ax.set_ylabel("Transfer AUROC")
    ax.set_title("Within-assay skill vs cross-assay transfer")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_transfer_by_domain(transfer: pd.DataFrame, out: Path) -> None:
    if transfer.empty or "domain_class" not in transfer.columns:
        return
    df = transfer.copy()
    df = df[df["domain_class"].fillna("unknown") != ""]
    # Keep classes with enough points
    counts = df["domain_class"].value_counts()
    keep = counts[counts >= 5].index
    df = df[df["domain_class"].isin(keep)]
    if df.empty:
        return
    order = (
        df.groupby("domain_class")["auroc"].median().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="domain_class",
        y="auroc",
        order=order,
        ax=ax,
        color="#4C78A8",
        fliersize=2,
    )
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("Domain class (Table S1)")
    ax.set_ylabel("Transfer AUROC")
    ax.set_title("Cross-protocol transfer by domain class")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_motif_vs_transfer(path: Path, out: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path, sep="\t")
    xcol = "jaccard_core5" if "jaccard_core5" in df.columns else "jaccard_top_kmers"
    df = df.dropna(subset=[xcol, "mean_transfer_auroc"])
    if len(df) < 5:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        df[xcol],
        df["mean_transfer_auroc"],
        alpha=0.6,
        s=32,
        edgecolors="none",
    )
    ax.set_xlabel("Motif concordance (core 5-mer Jaccard)" if xcol == "jaccard_core5" else "Top-10 7-mer Jaccard")
    ax.set_ylabel("Mean bidirectional transfer AUROC")
    ax.set_title("Motif concordance vs classifier transfer")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_roster_overview(roster: pd.DataFrame, out: Path) -> None:
    if roster.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # protocol count histogram
    axes[0].hist(roster["n_protocols"], bins=range(2, int(roster["n_protocols"].max()) + 2),
                 color="#4C78A8", edgecolor="white")
    axes[0].set_xlabel("# protocols per protein")
    axes[0].set_ylabel("Proteins")
    axes[0].set_title("Matched panel size")

    if "domain_class" in roster.columns:
        vc = roster["domain_class"].value_counts()
        axes[1].barh(vc.index.astype(str)[::-1], vc.values[::-1], color="#F58518")
        axes[1].set_xlabel("Proteins")
        axes[1].set_title("Domain class coverage")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_protocol.yaml")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--figures_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    out_dir = resolve(args.out_dir or cfg["paths"]["out_dir"])
    fig_dir = resolve(args.figures_dir or cfg["paths"].get("figures_dir", "figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    roster_path = out_dir / "protein_roster.tsv"
    within_path = out_dir / "within_protocol_metrics.tsv"
    transfer_path = out_dir / "transfer_metrics.tsv"
    motif_vs = out_dir / "motif_vs_transfer.tsv"

    if roster_path.exists():
        roster = pd.read_csv(roster_path, sep="\t")
        fig_roster_overview(roster, fig_dir / "cross_protocol_roster.png")
        print("Wrote cross_protocol_roster.png")

    within = pd.read_csv(within_path, sep="\t") if within_path.exists() else pd.DataFrame()
    transfer = (
        pd.read_csv(transfer_path, sep="\t") if transfer_path.exists() else pd.DataFrame()
    )

    if not transfer.empty:
        fig_transfer_heatmap(transfer, fig_dir / "cross_protocol_transfer_heatmap.png")
        print("Wrote cross_protocol_transfer_heatmap.png")
        fig_transfer_by_domain(transfer, fig_dir / "cross_protocol_transfer_by_domain.png")
        print("Wrote cross_protocol_transfer_by_domain.png")

    if not within.empty and not transfer.empty:
        fig_within_vs_transfer(
            within, transfer, fig_dir / "cross_protocol_within_vs_transfer.png"
        )
        print("Wrote cross_protocol_within_vs_transfer.png")

    fig_motif_vs_transfer(motif_vs, fig_dir / "cross_protocol_motif_vs_transfer.png")
    if motif_vs.exists():
        print("Wrote cross_protocol_motif_vs_transfer.png")

    print(f"Figures in {fig_dir}")


if __name__ == "__main__":
    main()
