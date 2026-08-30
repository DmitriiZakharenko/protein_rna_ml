#!/usr/bin/env python3
"""
21b_evaluate_external_v4.py
---------------------------
External validation for V4 (RNABindingV4) checkpoints on literature / expanded
benchmark pairs. Reuses data loading and sliding-window scoring from script 11.

Usage:
    python scripts/21b_evaluate_external_v4.py \\
        --checkpoint models/saved/generalized_v4_phase3a/best_model.pt \\
        --prot_max 700 \\
        --out_dir results/external/v4_phase3a_curated

    python scripts/21b_evaluate_external_v4.py \\
        --checkpoint models/saved/generalized_v4_phase3a/best_model.pt \\
        --benchmark_tsv data/external/external_benchmark_expanded.tsv \\
        --prot_max 700 \\
        --out_dir results/external/v4_phase3a_expanded
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.external_benchmark import load_benchmark_tsv
from src.models.interaction_model import RNABindingV4


def _load_script11():
    path = os.path.join(ROOT, "scripts", "11_evaluate_external.py")
    spec = importlib.util.spec_from_file_location("evaluate_external_11", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = _load_script11()


def load_v4_model(checkpoint_path: str, device: torch.device) -> tuple[RNABindingV4, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    train_args = ckpt.get("args", {})

    interaction = arch.get("interaction", train_args.get("interaction", "concat_bi"))
    use_source_emb = arch.get("use_source_emb", train_args.get("use_source_emb", False))
    inter_dim = arch.get("inter_dim", train_args.get("inter_dim", 256))
    dropout = train_args.get("dropout", 0.3)
    source_emb_dim = train_args.get("source_emb_dim", 16)

    model = RNABindingV4(
        interaction=interaction,
        inter_dim=inter_dim,
        dropout=dropout,
        use_source_emb=use_source_emb,
        source_emb_dim=source_emb_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "interaction": interaction,
        "use_source_emb": use_source_emb,
        "epoch": ckpt.get("epoch"),
        "val_metrics": ckpt.get("val_metrics"),
    }
    return model, meta


@torch.no_grad()
def score_pair_v4(
    model: RNABindingV4,
    rna_seq: str,
    prot_seq: str,
    rna_max: int,
    prot_max: int,
    device: torch.device,
    win_step: int,
) -> float:
    windows = ext.rna_windows(rna_seq, rna_max, win_step)
    prot_oh = ext.one_hot_prot(prot_seq, prot_max).unsqueeze(0).to(device)
    scores = []
    for w in windows:
        rna_oh = ext.one_hot_rna(w, rna_max).unsqueeze(0).to(device)
        logit = model(rna_oh, prot_oh, source_tags=None)
        scores.append(torch.sigmoid(logit).item())
    return max(scores)


def _is_valid_xlsx(path: str) -> bool:
    """xlsx files are zip archives; reject empty/corrupt uploads."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as fh:
            return fh.read(2) == b"PK"
    except OSError:
        return False


def _resolve_curated_xlsx(explicit: str | None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend([
        "data/external/dataset without affinities.xlsx",
        "data/external/dataset_without_affinities.xlsx",
        "dataset without affinities.xlsx",
    ])
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if os.path.exists(c) and _is_valid_xlsx(c):
            return c
    return None


def load_pairs(args) -> tuple[pd.DataFrame, str]:
    if args.benchmark_tsv:
        print(f"\n=== Loading benchmark TSV: {args.benchmark_tsv} ===")
        df = load_benchmark_tsv(args.benchmark_tsv)
        return df, args.benchmark_tsv

    xlsx = _resolve_curated_xlsx(args.xlsx)
    if xlsx is None:
        print("ERROR: no valid curated xlsx found (need PK zip header, ~86 KB).")
        print("  Fix: re-upload from Mac, or use expanded TSV:")
        print("    --benchmark_tsv data/external/external_benchmark_expanded.tsv")
        print("  Curated-only subset (~159 rows):")
        print("    python -c \"import pandas as pd; df=pd.read_csv('data/external/external_benchmark_expanded.tsv', sep='\\t'); df[df.example_class.str.startswith('curated_')].to_csv('data/external/external_benchmark_curated.tsv', sep='\\t', index=False)\"")
        sys.exit(1)

    print(f"\n=== Loading curated external xlsx: {xlsx} ({os.path.getsize(xlsx):,} bytes) ===")
    df_raw = ext.load_external_dataset(xlsx)

    def find_col(df, *candidates):
        for c in df.columns:
            if c and any(k.lower() in c.lower() for k in candidates):
                return c
        return None

    col_protein = find_col(df_raw, "protein") or "Protein"
    col_rna_seq = find_col(df_raw, "rna sequence", "rna seq") or "RNA sequence"
    col_label = find_col(df_raw, "interaction") or "Interaction (yes/no)"
    col_prot_seq = None
    for c in df_raw.columns:
        if c and "protein sequence" in c.lower():
            col_prot_seq = c
            break
    col_prot_seq = col_prot_seq or "Protein Sequence"
    col_domain_seq = None
    for c in df_raw.columns:
        if c and "domain" in c.lower() and ("sequence" in c.lower() or "mutation" in c.lower()):
            col_domain_seq = c
            break
    col_domain_seq = col_domain_seq or col_prot_seq

    records = []
    for _, row in df_raw.iterrows():
        label = ext.parse_label(row.get(col_label))
        if label is None:
            continue
        rna_seq = str(row.get(col_rna_seq, "") or "").strip().upper().replace("T", "U")
        prot_seq = str(row.get(col_domain_seq, "") or "").strip()
        if len(prot_seq) < 10:
            prot_seq = str(row.get(col_prot_seq, "") or "").strip()
        prot_name = str(row.get(col_protein, "") or "").strip()
        if len(rna_seq) < 4 or len(prot_seq) < 10:
            continue
        records.append({
            "pair_id": "",
            "protein_name": prot_name,
            "rna_seq": rna_seq,
            "prot_seq": prot_seq,
            "label": label,
            "rna_len": len(rna_seq),
            "prot_len": len(prot_seq),
            "example_class": "",
            "neg_strategy": "",
        })

    return pd.DataFrame(records), xlsx


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to V4 best_model.pt")
    parser.add_argument("--benchmark_tsv", default=None)
    parser.add_argument("--xlsx", default=None)
    parser.add_argument("--out_dir", default="results/external/v4_external")
    parser.add_argument("--rna_max", type=int, default=60)
    parser.add_argument("--prot_max", type=int, default=700)
    parser.add_argument("--win_step", type=int, default=30)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.no_cuda and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    model, meta = load_v4_model(args.checkpoint, device)
    print(f"  Loaded V4 | interaction={meta['interaction']} "
          f"| source_emb={meta['use_source_emb']} | epoch={meta.get('epoch')}")

    df, dataset_source = load_pairs(args)
    print(f"  Pairs: {len(df)}  pos={int(df['label'].sum())}  "
          f"neg={int((df['label'] == 0).sum())}")

    probs = []
    for i, row in df.iterrows():
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(df)}] scoring...", end="\r")
        probs.append(score_pair_v4(
            model, row["rna_seq"], row["prot_seq"],
            args.rna_max, args.prot_max, device, args.win_step,
        ))
    print(f"\n  Done scoring {len(df)} pairs.")
    df["prob_v4"] = probs

    labels = df["label"].values
    if len(set(labels)) < 2:
        print("  Only one class — cannot compute AUROC/AUPRC.")
        sys.exit(1)

    auroc = float(roc_auc_score(labels, probs))
    auprc = float(average_precision_score(labels, probs))
    pos_rate = float(labels.mean())
    print(f"\n  V4 external → AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}")
    print(f"  Random AUPRC baseline: {pos_rate:.4f}")

    out_tsv = os.path.join(args.out_dir, "external_pairs_scored.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    results = {
        "checkpoint": args.checkpoint,
        "dataset": dataset_source,
        "model": f"V4_{meta['interaction']}",
        "use_source_emb": meta["use_source_emb"],
        "n_pairs": int(len(df)),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "pos_rate": round(pos_rate, 4),
        "random_auprc_baseline": round(pos_rate, 4),
        "v4": {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "auprc_gain_over_random": round(auprc - pos_rate, 4),
        },
        "sliding_window": {"rna_max": args.rna_max, "step": args.win_step},
        "prot_max": args.prot_max,
    }
    out_json = os.path.join(args.out_dir, "external_validation_v4.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Saved {out_tsv}")
    print(f"  Saved {out_json}")


if __name__ == "__main__":
    main()
