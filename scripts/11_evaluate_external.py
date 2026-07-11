"""
Script 11: External validation on manually curated protein–RNA interaction dataset.

Evaluates our trained models (V2 CNN, V3b if available) on real, experimentally
validated protein–RNA interactions collected from literature — a hard out-of-distribution
test:
  - Our training data: short synthetic RNAs (20–40 nt SELEX/RBNS, 168 proteins)
  - This dataset: natural lncRNAs and mRNA fragments (median 150 nt, up to 17918 nt)
    + diverse in vitro binding pairs (165 rows, 117 positives / 45 negatives)

Long RNA strategy — sliding window:
  RNAs longer than rna_max are split into overlapping 60 nt windows (step = 30 nt).
  The model scores each window; the pair's final score is the MAXIMUM window score.
  Rationale: if ANY region of the RNA binds the protein, the pair should be positive.
  This is analogous to the global max pooling the CNN already does within a sequence.

Usage (from protein_rna_ml/):
    python scripts/11_evaluate_external.py
    python scripts/11_evaluate_external.py --xlsx "path/to/dataset without affinities.xlsx"
    python scripts/11_evaluate_external.py --benchmark_tsv data/external/external_benchmark_expanded.tsv
    python scripts/11_evaluate_external.py --model_dir models/saved/generalized_v2 --model_type v2
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.cnn_model import RNABindingCNN
from src.data.external_benchmark import load_benchmark_tsv

try:
    import openpyxl
except ImportError:
    print("Installing openpyxl...")
    os.system("pip install openpyxl --break-system-packages -q")
    import openpyxl


# ── One-hot encoding ──────────────────────────────────────────────────────────
RNA_ALPHA   = "AUGC"
AA_ALPHA    = "ACDEFGHIKLMNPQRSTVWY"
RNA_TO_IDX  = {c: i for i, c in enumerate(RNA_ALPHA)}
AA_TO_IDX   = {c: i for i, c in enumerate(AA_ALPHA)}

# Also map DNA-like thymine → uracil for RNA
RNA_TO_IDX["T"] = RNA_TO_IDX["U"]
RNA_TO_IDX["t"] = RNA_TO_IDX["U"]


def one_hot_rna(seq: str, max_len: int) -> torch.Tensor:
    t = torch.zeros(max_len, 4)
    for i, c in enumerate(str(seq).upper()[:max_len]):
        idx = RNA_TO_IDX.get(c)
        if idx is not None:
            t[i, idx] = 1.0
    return t


def one_hot_prot(seq: str, max_len: int) -> torch.Tensor:
    t = torch.zeros(max_len, 20)
    for i, c in enumerate(str(seq).upper()[:max_len]):
        idx = AA_TO_IDX.get(c)
        if idx is not None:
            t[i, idx] = 1.0
    return t


# ── Sliding window for long RNAs ──────────────────────────────────────────────
def rna_windows(seq: str, win: int, step: int) -> list[str]:
    """Split a long RNA sequence into overlapping windows of length `win`."""
    seq = str(seq).upper()
    if len(seq) <= win:
        return [seq]
    windows = []
    for start in range(0, len(seq) - win + 1, step):
        windows.append(seq[start:start + win])
    # Make sure the last window reaches the end
    if (len(seq) - win) % step != 0:
        windows.append(seq[-win:])
    return windows


# ── Label normalisation ───────────────────────────────────────────────────────
def parse_label(val) -> int | None:
    """
    Returns 1 (binding), 0 (non-binding), or None (ambiguous → skip).
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("yes", "1", "true"):
        return 1
    if s in ("no", "0", "false"):
        return 0
    # Ambiguous: "yes (weak)", "no?", "yes?" — skip
    return None


# ── Load Excel dataset ────────────────────────────────────────────────────────
def load_external_dataset(xlsx_path: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = list(rows[0])
    # Normalise column names (strip whitespace and newlines)
    clean_header = [str(h).strip().replace("\n", " ") if h else None for h in header]
    df = pd.DataFrame(rows[1:], columns=clean_header)
    return df


# ── Score a single pair (with sliding window for long RNAs) ──────────────────
@torch.no_grad()
def score_pair(model, rna_seq: str, prot_seq: str,
               rna_max: int, prot_max: int, device,
               win_step: int = 30) -> float:
    windows = rna_windows(rna_seq, rna_max, win_step)
    prot_oh = one_hot_prot(prot_seq, prot_max).unsqueeze(0).to(device)

    scores = []
    for w in windows:
        rna_oh = one_hot_rna(w, rna_max).unsqueeze(0).to(device)
        logit  = model(rna_oh, prot_oh)
        scores.append(torch.sigmoid(logit).item())

    return max(scores)  # max over windows = "any region binds"


# ── V3b model (only loaded if checkpoint exists) ──────────────────────────────
class ConvBranchV3b(nn.Module):
    def __init__(self, in_ch, filters, kernels, dropout=0.3):
        super().__init__()
        layers, ch = [], in_ch
        for f, k in zip(filters, kernels):
            layers += [nn.Conv1d(ch,f,k,padding=k//2), nn.BatchNorm1d(f), nn.GELU(), nn.Dropout(dropout)]
            ch = f
        self.net = nn.Sequential(*layers)
        self.out_dim = filters[-1]
    def forward(self, x):
        return self.net(x.transpose(1,2)).max(dim=-1).values

class V3bModel(nn.Module):
    def __init__(self, esm_dim=1280, esm_proj_dim=128, dropout=0.3):
        super().__init__()
        self.rna_branch  = ConvBranchV3b(4,  [128,256,256], [7,5,3],   dropout)
        self.prot_branch = ConvBranchV3b(20, [128,256,256], [11,7,5],  dropout)
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, esm_proj_dim), nn.LayerNorm(esm_proj_dim),
            nn.GELU(), nn.Dropout(dropout))
        in_dim = 256 + 256 + esm_proj_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),    nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    def forward(self, rna_oh, prot_oh, prot_emb):
        return self.head(torch.cat([
            self.rna_branch(rna_oh),
            self.prot_branch(prot_oh),
            self.esm_proj(prot_emb)], dim=-1)).squeeze(-1)


@torch.no_grad()
def score_pair_v3b(model, emb_lookup, rna_seq, prot_seq, prot_name,
                   rna_max, prot_max, device, win_step=30):
    if prot_name not in emb_lookup:
        return None  # no ESM-2 embedding → skip for V3b
    prot_emb = torch.tensor(emb_lookup[prot_name], dtype=torch.float32).unsqueeze(0).to(device)
    prot_oh  = one_hot_prot(prot_seq, prot_max).unsqueeze(0).to(device)
    windows  = rna_windows(rna_seq, rna_max, win_step)
    scores   = []
    for w in windows:
        rna_oh = one_hot_rna(w, rna_max).unsqueeze(0).to(device)
        logit  = model(rna_oh, prot_oh, prot_emb)
        scores.append(torch.sigmoid(logit).item())
    return max(scores)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default=None,
                        help="Path to 'dataset without affinities.xlsx'. "
                             "Defaults to auto-detected path in uploads/.")
    parser.add_argument("--benchmark_tsv", default=None,
                        help="Expanded benchmark TSV from scripts/31_build_external_benchmark.py. "
                             "Takes precedence over --xlsx when set or auto-detected.")
    parser.add_argument("--v2_dir",  default="models/saved/generalized_v2")
    parser.add_argument("--v3b_dir", default="models/saved/generalized_v3b")
    parser.add_argument("--emb_path", default="data/embeddings/esm2_protein_embeddings.npz")
    parser.add_argument("--out_dir", default="results/external")
    parser.add_argument("--rna_max",  type=int, default=60)
    parser.add_argument("--prot_max", type=int, default=300)
    parser.add_argument("--win_step", type=int, default=30,
                        help="Sliding window step size for long RNAs")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-detect benchmark TSV / xlsx path (only when neither is explicitly set)
    if args.benchmark_tsv is None and args.xlsx is None:
        tsv_candidates = ["data/external/external_benchmark_expanded.tsv"]
        xlsx_candidates = [
            "dataset without affinities.xlsx",
            "data/external/dataset_without_affinities.xlsx",
        ]
        for c in tsv_candidates:
            if os.path.exists(c):
                args.benchmark_tsv = c
                break
        if args.benchmark_tsv is None:
            for c in xlsx_candidates:
                if os.path.exists(c):
                    args.xlsx = c
                    break
        if args.benchmark_tsv is None and args.xlsx is None:
            print("ERROR: Could not find external benchmark.")
            print("Pass --benchmark_tsv or --xlsx explicitly.")
            sys.exit(1)

    # Device
    if not args.no_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    if args.benchmark_tsv:
        print(f"\n=== Loading expanded benchmark TSV: {args.benchmark_tsv} ===")
        df = load_benchmark_tsv(args.benchmark_tsv)
        dataset_source = args.benchmark_tsv
        print(f"  Pairs: {len(df)}  (pos={df['label'].sum()}, neg={(df['label']==0).sum()})")
        if "example_class" in df.columns:
            print(f"  Example classes: {df['example_class'].value_counts().to_dict()}")
        if "neg_strategy" in df.columns and (df["neg_strategy"] != "").any():
            print(f"  Neg strategies:  {df[df['neg_strategy']!='']['neg_strategy'].value_counts().to_dict()}")
    else:
        print(f"\n=== Loading external dataset: {args.xlsx} ===")
        df_raw = load_external_dataset(args.xlsx)
        dataset_source = args.xlsx
        print(f"  Raw rows: {len(df_raw)}")

        # Find relevant columns (flexible name matching)
        def find_col(df, *candidates):
            for c in df.columns:
                if c and any(k.lower() in c.lower() for k in candidates):
                    return c
            return None

        col_protein    = find_col(df_raw, "protein") or "Protein"
        col_rna_seq    = find_col(df_raw, "rna sequence", "rna seq") or "RNA sequence"
        col_label      = find_col(df_raw, "interaction") or "Interaction (yes/no)"
        # Protein sequences: full sequence column and domain/cropped sequence column.
        # Be specific to avoid matching length columns like "Protein lenght (part, if isolated domains were used)".
        col_prot_seq = None
        for c in df_raw.columns:
            if c and "protein sequence" in c.lower():
                col_prot_seq = c
                break
        col_prot_seq = col_prot_seq or "Protein Sequence"
        # Domain/cropped sequence — must contain "domain" AND "sequence" (or "mutations sequence")
        col_domain_seq = None
        for c in df_raw.columns:
            if c and "domain" in c.lower() and ("sequence" in c.lower() or "mutation" in c.lower()):
                col_domain_seq = c
                break
        col_domain_seq = col_domain_seq or col_prot_seq

        print(f"  Columns used:")
        print(f"    protein name : {col_protein}")
        print(f"    RNA sequence : {col_rna_seq}")
        print(f"    protein seq  : {col_domain_seq} (fallback: {col_prot_seq})")
        print(f"    label        : {col_label}")

        # Parse and filter
        records = []
        skipped_label = 0
        skipped_seq   = 0
        for _, row in df_raw.iterrows():
            label = parse_label(row.get(col_label))
            if label is None:
                skipped_label += 1
                continue

            rna_seq = str(row.get(col_rna_seq, "") or "").strip()
            # Prefer domain/cropped sequence for protein (more specific binding context)
            prot_seq = str(row.get(col_domain_seq, "") or "").strip()
            if len(prot_seq) < 10:
                prot_seq = str(row.get(col_prot_seq, "") or "").strip()
            prot_name = str(row.get(col_protein, "") or "").strip()

            if len(rna_seq) < 4 or len(prot_seq) < 10:
                skipped_seq += 1
                continue

            # Normalize RNA sequence (T→U)
            rna_seq = rna_seq.upper().replace("T", "U")

            records.append({
                "pair_id": "",
                "protein_name": prot_name,
                "rna_seq":      rna_seq,
                "prot_seq":     prot_seq,
                "label":        label,
                "rna_len":      len(rna_seq),
                "prot_len":     len(prot_seq),
                "example_class": "",
                "neg_strategy": "",
            })

        df = pd.DataFrame(records)
        print(f"  Skipped (ambiguous label): {skipped_label}")
        print(f"  Skipped (missing seq):     {skipped_seq}")

    print(f"  Usable pairs: {len(df)}  (pos={df['label'].sum()}, neg={(df['label']==0).sum()})")
    print(f"  Unique proteins: {df['protein_name'].nunique()}")
    print(f"  RNA length: min={df['rna_len'].min()}, median={df['rna_len'].median():.0f}, "
          f"max={df['rna_len'].max()}")
    print(f"  Protein length: min={df['prot_len'].min()}, max={df['prot_len'].max()}")
    print(f"  RNAs requiring sliding window (>{args.rna_max} nt): "
          f"{(df['rna_len'] > args.rna_max).sum()} / {len(df)}")

    # ── Load V2 CNN ───────────────────────────────────────────────────────────
    v2_ckpt = os.path.join(args.v2_dir, "best_model.pt")
    print(f"\n=== Loading V2 CNN: {v2_ckpt} ===")
    model_v2 = RNABindingCNN(
        rna_filters=[128, 256, 256], rna_kernels=[7, 5, 3],
        prot_filters=[128, 256, 256], prot_kernels=[11, 7, 5],
        head_dims=[256, 64], dropout=0.3,
    ).to(device)
    ckpt = torch.load(v2_ckpt, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model_v2.load_state_dict(state)
    model_v2.eval()
    print("  V2 CNN loaded OK")

    # ── Load V3b (optional) ───────────────────────────────────────────────────
    v3b_ckpt = os.path.join(args.v3b_dir, "best_model.pt")
    model_v3b  = None
    emb_lookup = {}
    if os.path.exists(v3b_ckpt):
        print(f"\n=== Loading V3b: {v3b_ckpt} ===")
        model_v3b = V3bModel().to(device)
        ckpt3b = torch.load(v3b_ckpt, map_location=device, weights_only=False)
        state3b = ckpt3b["model_state"] if isinstance(ckpt3b, dict) and "model_state" in ckpt3b else ckpt3b
        model_v3b.load_state_dict(state3b)
        model_v3b.eval()
        # Load ESM-2 embeddings
        if os.path.exists(args.emb_path):
            data = np.load(args.emb_path)
            emb_lookup = {k: data[k] for k in data.files}
            print(f"  ESM-2 embeddings: {len(emb_lookup)} proteins")
        print("  V3b loaded OK")
    else:
        print(f"\n  V3b checkpoint not found ({v3b_ckpt}) — skipping V3b evaluation.")
        print("  Run script 09 first to train V3b.")

    # ── Score all pairs ───────────────────────────────────────────────────────
    print(f"\n=== Scoring {len(df)} pairs (sliding window step={args.win_step} nt) ===")
    probs_v2   = []
    probs_v3b  = []

    for i, row in df.iterrows():
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}] ...", end="\r")

        p_v2 = score_pair(
            model_v2, row["rna_seq"], row["prot_seq"],
            args.rna_max, args.prot_max, device, args.win_step)
        probs_v2.append(p_v2)

        if model_v3b is not None:
            p_v3b = score_pair_v3b(
                model_v3b, emb_lookup, row["rna_seq"], row["prot_seq"],
                row["protein_name"], args.rna_max, args.prot_max, device, args.win_step)
            probs_v3b.append(p_v3b)

    df["prob_v2"] = probs_v2
    print(f"\n  Done scoring.")

    # ── Integrity checks before reporting any metric ─────────────────────────
    n_pairs    = len(df)
    n_pos      = int(df["label"].sum())
    n_neg      = int((df["label"] == 0).sum())
    n_proteins = int(df["protein_name"].nunique())
    n_single_class = int(df.groupby("protein_name")["label"].nunique().eq(1).sum())
    n_long_rna = int((df["rna_len"] > args.rna_max).sum())
    pos_rate   = n_pos / max(n_pairs, 1)
    random_auprc_baseline = pos_rate  # AUPRC of a random classifier = class prevalence

    WARN = []
    BLOCK = []

    if n_pos < 10 or n_neg < 10:
        BLOCK.append(
            f"INSUFFICIENT CLASSES: only {n_pos} positives and {n_neg} negatives. "
            "AUROC/AUPRC are statistically meaningless.")

    if n_single_class / n_proteins > 0.50:
        WARN.append(
            f"SINGLE-CLASS PROTEINS: {n_single_class}/{n_proteins} proteins ({n_single_class/n_proteins:.0%}) "
            "have only positive OR only negative examples. "
            "Aggregate AUROC is computed across all examples regardless — it is NOT a per-protein metric. "
            "Reported AUROC/AUPRC does NOT reflect generalisation to individual proteins.")

    if n_long_rna / n_pairs > 0.30:
        WARN.append(
            f"WINDOW-MAX INFLATION: {n_long_rna}/{n_pairs} RNAs ({n_long_rna/n_pairs:.0%}) "
            f"exceed {args.rna_max} nt and are scored as max over sliding windows. "
            "Max-pool scoring systematically assigns higher probabilities to longer RNAs, "
            "independently of actual binding. This inflates AUPRC when positives tend to be longer. "
            "Do NOT compare these numbers to SELEX/RBNS results without stratifying by RNA length.")

    if pos_rate > 0.60:
        WARN.append(
            f"HIGH POSITIVE RATE: {pos_rate:.0%} of pairs are positive. "
            f"Random-classifier AUPRC baseline = {random_auprc_baseline:.3f}. "
            "Subtract this baseline before interpreting reported AUPRC.")

    if WARN or BLOCK:
        print(f"\n{'!'*60}")
        print("  EVALUATION INTEGRITY WARNINGS")
        print(f"{'!'*60}")
        for w in WARN:
            print(f"  ⚠️  {w}")
        for b in BLOCK:
            print(f"  ❌  {b}")
        print(f"{'!'*60}")

    if BLOCK:
        print("\n  BLOCKED: aggregate AUROC/AUPRC not reported due to above critical issue.")
        auroc_v2, auprc_v2 = None, None
    else:
        # ── Overall metrics (V2) ─────────────────────────────────────────────
        labels = df["label"].values
        pv2    = df["prob_v2"].values

        def safe_metrics(labs, probs, label=""):
            if len(set(labs)) < 2:
                print(f"  [{label}] Only one class present — cannot compute AUROC/AUPRC")
                return None, None
            auroc = float(roc_auc_score(labs, probs))
            auprc = float(average_precision_score(labs, probs))
            return auroc, auprc

        print(f"\n{'='*55}")
        print(f"  EXTERNAL VALIDATION RESULTS  [read warnings above before citing]")
        print(f"{'='*55}")
        auroc_v2, auprc_v2 = safe_metrics(labels, pv2, "V2")
        if auroc_v2 is not None:
            print(f"  V2 CNN  →  AUROC: {auroc_v2:.4f}  |  AUPRC: {auprc_v2:.4f}")
            print(f"  Random AUPRC baseline: {random_auprc_baseline:.4f}  "
                  f"(effective gain over random: {auprc_v2 - random_auprc_baseline:+.4f})")

    labels = df["label"].values
    pv2    = df["prob_v2"].values

    def safe_metrics(labs, probs, label=""):
        if len(set(labs)) < 2:
            print(f"  [{label}] Only one class present — cannot compute AUROC/AUPRC")
            return None, None
        return float(roc_auc_score(labs, probs)), float(average_precision_score(labs, probs))

    # Re-assign in case BLOCK path skipped computation above
    if auroc_v2 is None and not BLOCK:
        auroc_v2, auprc_v2 = safe_metrics(labels, pv2, "V2")

    results = {
        "dataset": dataset_source,
        "n_pairs": int(len(df)),
        "n_pos":   int(df["label"].sum()),
        "n_neg":   int((df["label"]==0).sum()),
        "n_proteins": int(df["protein_name"].nunique()),
        "n_single_class_proteins": int(n_single_class),
        "pos_rate": round(pos_rate, 4),
        "random_auprc_baseline": round(random_auprc_baseline, 4),
        "rna_len_stats": {
            "min": int(df["rna_len"].min()),
            "median": int(df["rna_len"].median()),
            "max": int(df["rna_len"].max()),
            "pct_long": float((df["rna_len"] > args.rna_max).mean()),
        },
        "sliding_window": {"rna_max": args.rna_max, "step": args.win_step},
        "integrity_warnings": WARN,
        "integrity_blocks": BLOCK,
        "v2_cnn": {
            "auroc": auroc_v2,
            "auprc": auprc_v2,
            "auprc_gain_over_random": round(auprc_v2 - random_auprc_baseline, 4)
                                      if auprc_v2 is not None else None,
        },
    }

    # V3b overall
    if model_v3b is not None and probs_v3b:
        df["prob_v3b"] = probs_v3b
        # Some rows may have None (missing ESM-2 embedding)
        mask = df["prob_v3b"].notna()
        if mask.sum() > 0 and len(set(labels[mask])) > 1:
            auroc_v3b, auprc_v3b = safe_metrics(labels[mask], df.loc[mask, "prob_v3b"].values, "V3b")
            print(f"  V3b     →  AUROC: {auroc_v3b:.4f}  |  AUPRC: {auprc_v3b:.4f}  "
                  f"(n={mask.sum()}, ESM-2 coverage)")
            results["v3b"] = {"auroc": auroc_v3b, "auprc": auprc_v3b, "n_scored": int(mask.sum())}

    # ── Stratified metrics (expanded benchmark) ───────────────────────────────
    if "example_class" in df.columns and (df["example_class"] != "").any():
        print(f"\n  Stratified metrics (V2 CNN):")
        print(f"  (Neg-type rows always include all curated positives as contrast set.)")

        def subset_metrics(name: str, mask: pd.Series, *, quiet_single: bool = False) -> dict | None:
            sub = df[mask]
            if len(sub) == 0:
                return None
            labs = sub["label"].values
            probs = sub["prob_v2"].values
            if len(set(labs)) < 2:
                if not quiet_single:
                    print(f"    {name:<32} n={len(sub):4d}  [single class — skipped]")
                return {"n": int(len(sub)), "auroc": None, "auprc": None, "note": "single_class"}
            auroc_s, auprc_s = safe_metrics(labs, probs, name)
            pos_rate_s = float(labs.mean())
            print(
                f"    {name:<32} n={len(sub):4d}  "
                f"pos={int(labs.sum())}/{len(labs)} ({pos_rate_s:.0%})  "
                f"AUROC={auroc_s:.4f}  AUPRC={auprc_s:.4f}"
            )
            return {
                "n": int(len(sub)),
                "n_pos": int(labs.sum()),
                "n_neg": int((labs == 0).sum()),
                "pos_rate": round(pos_rate_s, 4),
                "auroc": round(auroc_s, 4),
                "auprc": round(auprc_s, 4),
            }

        stratified: dict[str, dict] = {}
        pos_mask = df["label"] == 1
        gen_neg_mask = df["example_class"] == "generated_negative"
        curated_mask = df["example_class"].isin(["curated_positive", "curated_negative"])

        if curated_mask.any():
            m = subset_metrics("curated_only", curated_mask)
            if m:
                stratified["curated_only"] = m

        if pos_mask.any() and gen_neg_mask.any():
            m = subset_metrics(
                "curated_pos_vs_generated_neg",
                pos_mask | gen_neg_mask,
            )
            if m:
                stratified["curated_pos_vs_generated_neg"] = m

        shuffle_mask = df["neg_strategy"].isin(["shuffle_uniform", "shuffle_dinucleotide"])
        if pos_mask.any() and shuffle_mask.any():
            m = subset_metrics("curated_pos_vs_shuffle_negs", pos_mask | shuffle_mask)
            if m:
                stratified["curated_pos_vs_shuffle_negs"] = m

        cross_mask = df["neg_strategy"].isin(["cross_protein", "cross_rna"])
        if pos_mask.any() and cross_mask.any():
            m = subset_metrics("curated_pos_vs_cross_negs", pos_mask | cross_mask)
            if m:
                stratified["curated_pos_vs_cross_negs"] = m

        if "neg_strategy" in df.columns:
            neg_df = df[df["label"] == 0].copy()
            neg_df["neg_strategy"] = neg_df["neg_strategy"].fillna("").astype(str)
            for strategy in sorted(s for s in neg_df["neg_strategy"].unique() if s):
                m = subset_metrics(
                    f"pos_vs_{strategy}",
                    pos_mask | (df["neg_strategy"].fillna("").astype(str) == strategy),
                )
                if m:
                    stratified[f"pos_vs_{strategy}"] = m

        results["stratified_v2"] = stratified

    # ── Per-protein breakdown (V2) ────────────────────────────────────────────
    print(f"\n  Per-protein breakdown (V2 CNN):")
    per_prot = []
    for prot, grp in df.groupby("protein_name"):
        labs  = grp["label"].values
        probs = grp["prob_v2"].values
        n_pos = int(labs.sum())
        n_neg = int((labs==0).sum())
        if len(set(labs)) < 2:
            note = "single_class"
            auroc_p, auprc_p = None, None
        else:
            try:
                auroc_p, auprc_p = safe_metrics(labs, probs)
            except ValueError:
                auroc_p, auprc_p = None, None
                note = "constant_scores"
            else:
                note = ""
        per_prot.append({
            "protein": prot,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": round(auroc_p, 4) if auroc_p is not None else None,
            "auprc": round(auprc_p, 4) if auprc_p is not None else None,
            "note": note,
        })
        if auroc_p is not None:
            marker = f"  AUROC={auroc_p:.3f} AUPRC={auprc_p:.3f}"
        else:
            marker = f"  [{note}]"
        print(f"    {prot:<30} n={n_pos}+/{n_neg}-{marker}")

    results["per_protein_v2"] = per_prot

    scored_prots = [p for p in per_prot if p["auroc"] is not None]
    if scored_prots:
        median_auroc = float(np.median([p["auroc"] for p in scored_prots]))
        median_auprc = float(np.median([p["auprc"] for p in scored_prots]))
        print(
            f"\n  Per-protein summary: {len(scored_prots)}/{len(per_prot)} proteins "
            f"with both classes → median AUROC={median_auroc:.3f}, median AUPRC={median_auprc:.3f}"
        )
        results["per_protein_summary_v2"] = {
            "n_proteins_scored": len(scored_prots),
            "n_proteins_total": len(per_prot),
            "median_auroc": round(median_auroc, 4),
            "median_auprc": round(median_auprc, 4),
        }

    # Comparison to in-distribution test performance
    print(f"\n  In-distribution (SELEX/RBNS, 24 proteins) → V2: AUROC=0.703  AUPRC=0.599")
    print(f"  External (lncRNA/literature, this eval)   → V2: AUROC={auroc_v2:.3f}  AUPRC={auprc_v2:.3f}")
    if auroc_v2 is not None:
        delta_auroc = auroc_v2 - 0.7028
        delta_auprc = auprc_v2 - 0.5987
        direction = "better" if delta_auroc > 0 else "worse"
        print(f"  Δ AUROC: {delta_auroc:+.3f}  Δ AUPRC: {delta_auprc:+.3f}  ({direction} than in-distribution)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = os.path.join(args.out_dir, "external_validation_v2.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_json}")

    # Save scored pairs TSV
    out_tsv = os.path.join(args.out_dir, "external_pairs_scored.tsv")
    df.drop(columns=["rna_seq", "prot_seq"]).to_csv(out_tsv, sep="\t", index=False)
    print(f"  Scored pairs → {out_tsv}")

    print(f"\n{'='*55}")
    print(f"  Done. Check results/external/ for full output.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
