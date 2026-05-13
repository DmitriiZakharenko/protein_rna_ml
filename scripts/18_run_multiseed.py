#!/usr/bin/env python3
"""
18_run_multiseed.py
Multi-seed evaluation runner for any training script.

Runs a training script N times with different seeds, then aggregates metrics
(mean ± std for AUROC, AUPRC, per-protein median AUROC) and produces a
variance report. Designed for post-Phase-1 clean retrains.

Usage:
  # Run V2 CNN with 5 seeds (dry-run first):
  python scripts/18_run_multiseed.py \
      --script scripts/06_train_generalized_v2.py \
      --seeds 42 0 1 2 3 \
      --output_dir results/multiseed/v2_cnn \
      --extra_args "--data_dir data/generalized_v2 --epochs 50" \
      [--dry_run]

  # Run V2 CNN with automatic seed range:
  python scripts/18_run_multiseed.py \
      --script scripts/06_train_generalized_v2.py \
      --n_seeds 5 \
      --output_dir results/multiseed/v2_cnn

Output:
  results/multiseed/<model>/
    seed_{N}/         ← per-seed result JSON (moved here automatically)
    summary.json      ← mean ± std across seeds
    summary.tsv       ← human-readable table
    variance_plot.png ← AUROC / AUPRC box plot
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_seed(script: str, seed: int, seed_dir: str, extra_args: list[str],
             python: str, dry_run: bool) -> tuple[int, float]:
    os.makedirs(seed_dir, exist_ok=True)
    log_path = os.path.join(seed_dir, "train.log")

    cmd = [python, script] + extra_args + ["--seed", str(seed),
                                            "--out_dir", seed_dir]
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n  [seed={seed}] {cmd_str}")
    print(f"  log → {log_path}")

    if dry_run:
        print("  [DRY-RUN] skipping execution")
        return 0, 0.0

    t0 = time.time()
    with open(log_path, "w") as log_fh:
        result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return result.returncode, elapsed


def find_result_json(seed_dir: str) -> dict | None:
    """Find the most recently modified JSON result file in seed_dir."""
    jsons = sorted(Path(seed_dir).glob("*.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not jsons:
        return None
    with open(jsons[0]) as fh:
        return json.load(fh)


def extract_metrics(result: dict) -> dict:
    test = result.get("test_metrics", {})
    pp   = result.get("per_protein", [])
    pp_aurocs = [p["auroc"] for p in pp if "auroc" in p]
    return {
        "test_auroc":         test.get("auroc"),
        "test_auprc":         test.get("auprc"),
        "best_val_auroc":     result.get("best_val_auroc"),
        "best_val_auprc":     result.get("best_val_auprc"),
        "best_epoch":         result.get("best_epoch"),
        "per_protein_median": float(np.median(pp_aurocs)) if pp_aurocs else None,
        "per_protein_min":    float(np.min(pp_aurocs))    if pp_aurocs else None,
        "n_proteins":         len(pp_aurocs),
    }


def aggregate(seed_metrics: dict[int, dict]) -> dict:
    keys = [k for k in next(iter(seed_metrics.values())).keys()
            if k not in ("best_epoch", "n_proteins")]
    agg = {}
    for k in keys:
        vals = [v[k] for v in seed_metrics.values()
                if v.get(k) is not None]
        if not vals:
            continue
        agg[k] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
            "values": [round(v, 5) for v in vals],
            "seeds":  list(seed_metrics.keys()),
        }
    return agg


def print_summary(agg: dict, seeds: list[int]):
    print(f"\n{'─'*60}")
    print(f"  Multi-seed summary ({len(seeds)} seeds: {seeds})")
    print(f"{'─'*60}")
    for metric in ["test_auroc", "test_auprc", "best_val_auroc", "per_protein_median"]:
        if metric not in agg:
            continue
        a = agg[metric]
        print(f"  {metric:<25} {a['mean']:.4f} ± {a['std']:.4f}"
              f"  [{a['min']:.4f}, {a['max']:.4f}]")
    print(f"{'─'*60}")


def plot_variance(agg: dict, output_dir: str, model_name: str):
    if not HAS_MATPLOTLIB:
        return
    metrics_to_plot = ["test_auroc", "test_auprc", "per_protein_median", "best_val_auroc"]
    present = [m for m in metrics_to_plot if m in agg and agg[m].get("values")]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(max(6, len(present) * 3), 5))
    if len(present) == 1:
        axes = [axes]

    for ax, metric in zip(axes, present):
        vals = agg[metric]["values"]
        mean = agg[metric]["mean"]
        std  = agg[metric]["std"]
        seeds = agg[metric]["seeds"]

        ax.boxplot([vals], positions=[0], widths=0.5,
                   medianprops={"color": "#e15759", "lw": 2})
        ax.scatter([0] * len(vals), vals, color="#4e79a7", s=50, zorder=5, alpha=0.8)
        for i, (s, v) in enumerate(zip(seeds, vals)):
            ax.annotate(f"s{s}", (0, v), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, color="gray")

        ax.set_xticks([0])
        ax.set_xticklabels([metric.replace("_", "\n")], fontsize=9)
        ax.set_title(f"{mean:.4f} ± {std:.4f}")
        ax.set_ylabel("Score")

        # ZHMolGraph reference
        if "auroc" in metric:
            ax.axhline(0.798, color="black", ls=":", lw=1, alpha=0.7, label="ZHMolGraph")
        if "auprc" in metric:
            ax.axhline(0.820, color="black", ls=":", lw=1, alpha=0.7)

    fig.suptitle(f"Multi-seed variance — {model_name}\n(n={len(vals)} seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "variance_plot.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script",     required=True,
                        help="Training script to run (e.g. scripts/06_train_generalized_v2.py)")
    parser.add_argument("--seeds",      nargs="+", type=int,
                        help="Explicit list of seeds (e.g. 42 0 1 2 3)")
    parser.add_argument("--n_seeds",    type=int, default=5,
                        help="Number of seeds if --seeds not provided (default: 5)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for per-seed dirs + summary")
    parser.add_argument("--extra_args", default="",
                        help="Extra arguments to forward to the training script (quoted string)")
    parser.add_argument("--python",     default=sys.executable,
                        help="Python interpreter to use")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_failed",action="store_true",
                        help="Skip seeds that fail instead of aborting")
    args = parser.parse_args()

    seeds = args.seeds or list(range(args.n_seeds))
    extra = shlex.split(args.extra_args) if args.extra_args else []
    model_name = os.path.splitext(os.path.basename(args.script))[0]

    print(f"\n{'='*60}")
    print(f"  Multi-seed runner")
    print(f"{'='*60}")
    print(f"  script     : {args.script}")
    print(f"  seeds      : {seeds}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  extra_args : {extra}")
    print(f"  dry_run    : {args.dry_run}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    seed_metrics: dict[int, dict] = {}
    failed_seeds: list[int] = []

    for seed in seeds:
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        retcode, elapsed = run_seed(
            args.script, seed, seed_dir, extra, args.python, args.dry_run)

        if args.dry_run:
            continue

        if retcode != 0:
            print(f"  [FAIL] seed={seed} exited with code {retcode}")
            failed_seeds.append(seed)
            if not args.skip_failed:
                sys.exit(f"Aborting. Re-run with --skip_failed to ignore failures.")
            continue

        result = find_result_json(seed_dir)
        if result is None:
            print(f"  [WARN] seed={seed}: no result JSON found in {seed_dir}")
            failed_seeds.append(seed)
            continue

        metrics = extract_metrics(result)
        seed_metrics[seed] = metrics
        print(f"  [seed={seed}] done in {elapsed:.0f}s | "
              f"test_auroc={metrics.get('test_auroc', '?'):.4f} | "
              f"test_auprc={metrics.get('test_auprc', '?'):.4f}")

    if args.dry_run:
        print("\nDry run complete.")
        return

    if not seed_metrics:
        print("No seed metrics collected.")
        return

    # ── Aggregate ──────────────────────────────────────────────────────────
    agg = aggregate(seed_metrics)
    print_summary(agg, list(seed_metrics.keys()))

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "model":         model_name,
        "script":        args.script,
        "seeds":         list(seed_metrics.keys()),
        "failed_seeds":  failed_seeds,
        "n_runs":        len(seed_metrics),
        "per_seed":      {str(s): m for s, m in seed_metrics.items()},
        "aggregate":     agg,
    }
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  → saved {json_path}")

    # TSV summary
    import pandas as pd
    rows = []
    for seed, m in seed_metrics.items():
        row = {"seed": seed}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    tsv_path = os.path.join(args.output_dir, "summary.tsv")
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.5f")
    print(f"  → saved {tsv_path}")

    # ── Variance plot ──────────────────────────────────────────────────────
    plot_variance(agg, args.output_dir, model_name)

    print(f"\nDone. {len(seed_metrics)}/{len(seeds)} seeds completed.")
    if failed_seeds:
        print(f"  WARNING: failed seeds: {failed_seeds}")


if __name__ == "__main__":
    main()
