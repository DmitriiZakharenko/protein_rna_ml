#!/usr/bin/env python3
"""
34_cross_protocol_classifiers.py
--------------------------------
Within-protocol and cross-protocol RNA-only classifiers for matched proteins.

STRICT DATA RULES
-----------------
- Use roster column name_in_<protocol> (representative native string) with EXACT
  equality filter. Never groupby protein_key (that would merge constructs).
- Dedupe matches scripts/25: majority vote; drop label ties.
- RNAcompete best-experiment selection matches scripts/25 (mean positive intensity).
- Cached within-metrics MUST come from model_comparison LR rows (test_auroc),
  never from best_model summaries (mostly RF).
- Invalid RNA letters are dropped and counted.

Usage:
    python scripts/34_cross_protocol_classifiers.py --config configs/cross_protocol.yaml
    python scripts/34_cross_protocol_classifiers.py --within_from_cached
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RNA_ALPHABET = "AUGC"
RNA_VALID_RE = re.compile(r"^[AUGC]+$", re.IGNORECASE)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p: str | Path, base: Path = ROOT) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def build_kmer_index(alphabet: str, k: int) -> dict[str, int]:
    return {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}


def kmer_freq_vector(seq: str, kmer_index: dict[str, int], k: int) -> np.ndarray:
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    seq = str(seq).upper().replace("T", "U")
    n = 0
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k]
        if km in kmer_index:
            vec[kmer_index[km]] += 1
            n += 1
    if n > 0:
        vec /= n
    return vec


def encode_rna_matrix(seqs: pd.Series, kmer_index: dict[str, int], k: int) -> np.ndarray:
    X = np.zeros((len(seqs), len(kmer_index)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        X[i] = kmer_freq_vector(seq, kmer_index, k)
    return X


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def get_model(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def select_best_rnacompete_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one experiment per protein (highest mean positive probe_intensity).

    Matches scripts/25_train_rna_only_per_protein.py exactly.
    """
    id_col = next((c for c in ("experiment_id", "hyb_id") if c in df.columns), None)
    if id_col is None:
        return df

    parts = []
    for prot, sub in df.groupby("protein_name"):
        if sub[id_col].nunique() <= 1:
            parts.append(sub)
            continue
        sub = sub.copy()
        sub["probe_intensity"] = pd.to_numeric(sub["probe_intensity"], errors="coerce")
        pos = sub[sub["binding_label"] == 1]
        if pos.empty or pos["probe_intensity"].isna().all():
            parts.append(sub)
            continue
        best_eid = pos.groupby(id_col)["probe_intensity"].mean().idxmax()
        parts.append(sub[sub[id_col] == best_eid])
    return pd.concat(parts, ignore_index=True) if parts else df


def filter_modal_length(df: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    lengths = df["rna_sequence"].astype(str).str.len()
    if lengths.nunique() <= 1:
        return df.reset_index(drop=True), (int(lengths.iloc[0]) if len(lengths) else None)
    mode_len = Counter(lengths).most_common(1)[0][0]
    return df.loc[lengths == mode_len].reset_index(drop=True), int(mode_len)


def dedupe_by_sequence(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    One row per unique rna_sequence. Conflicting labels → majority vote; ties dropped.
    Matches scripts/25.
    """
    n_before = len(df)
    rows: list[dict] = []
    for seq, grp in df.groupby("rna_sequence", sort=False):
        labels = grp["binding_label"].values
        counts = Counter(labels)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue  # ambiguous — drop
        label = int(top[0][0])
        row = grp.iloc[0].to_dict()
        row["binding_label"] = label
        rows.append(row)
    out = pd.DataFrame(rows) if rows else df.iloc[0:0].copy()
    return out, int(n_before - len(out))


def filter_valid_rna(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    seqs = df["rna_sequence"].astype(str).str.upper().str.replace("T", "U", regex=False)
    ok = seqs.map(lambda s: bool(RNA_VALID_RE.match(s)))
    n_drop = int((~ok).sum())
    out = df.loc[ok].copy()
    out["rna_sequence"] = seqs.loc[ok].values
    return out, n_drop


def load_protocol_frame(
    path: Path,
    protocol_id: str,
    protein_col: str | None,
    is_rnacompete: bool,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if protein_col and protein_col in df.columns:
        df = df.rename(columns={protein_col: "protein_name"})
    elif "target_name" in df.columns and "protein_name" not in df.columns:
        df = df.rename(columns={"target_name": "protein_name"})
    required = {"protein_name", "rna_sequence", "binding_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{protocol_id}: missing columns {missing} in {path}")
    n_raw = len(df)
    df = df.dropna(subset=["protein_name", "rna_sequence", "binding_label"]).copy()
    df["protein_name"] = df["protein_name"].astype(str).str.strip()
    df["binding_label"] = pd.to_numeric(df["binding_label"], errors="coerce")
    df = df.dropna(subset=["binding_label"])
    df["binding_label"] = df["binding_label"].astype(int)
    df = df[df["binding_label"].isin([0, 1])]
    df, n_bad_rna = filter_valid_rna(df)
    df["protocol"] = protocol_id
    if is_rnacompete:
        df = select_best_rnacompete_experiment(df)
        parts = []
        for _, g in df.groupby("protein_name", sort=False):
            g2, _ = filter_modal_length(g)
            parts.append(g2)
        df = pd.concat(parts, ignore_index=True) if parts else df
    stats = {
        "n_raw_rows": n_raw,
        "n_after_filters": len(df),
        "n_invalid_rna_dropped": n_bad_rna,
        "n_proteins": int(df["protein_name"].nunique()),
    }
    return df, stats


def stratified_three_way(
    n: int, y: np.ndarray, test_frac: float, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    idx_tv, idx_te = train_test_split(
        idx, test_size=test_frac, stratify=y, random_state=seed
    )
    y_tv = y[idx_tv]
    val_ratio = val_frac / (1.0 - test_frac)
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=val_ratio, stratify=y_tv, random_state=seed
    )
    return idx_tr, idx_va, idx_te


def within_protocol_eval(
    sub: pd.DataFrame,
    model_name: str,
    kmer_index: dict[str, int],
    k: int,
    seed: int,
    test_frac: float,
    val_frac: float,
    min_pos: int,
    min_neg: int,
    min_examples: int,
) -> dict | None:
    sub, n_deduped = dedupe_by_sequence(sub)
    n_pos = int((sub["binding_label"] == 1).sum())
    n_neg = int((sub["binding_label"] == 0).sum())
    if n_pos < min_pos or n_neg < min_neg or len(sub) < min_examples:
        return None
    y = sub["binding_label"].values
    try:
        idx_tr, idx_va, idx_te = stratified_three_way(
            len(sub), y, test_frac, val_frac, seed
        )
    except ValueError:
        return None
    # Evaluate on held-out test; train on train+val (same reporting spirit as script 25
    # model-selection on val then refit — here single model so train+val → test).
    train = sub.iloc[np.concatenate([idx_tr, idx_va])]
    test = sub.iloc[idx_te]
    if train["binding_label"].nunique() < 2 or test["binding_label"].nunique() < 2:
        return None
    if len(test) < 20:
        return None

    X_tr = encode_rna_matrix(train["rna_sequence"], kmer_index, k)
    X_te = encode_rna_matrix(test["rna_sequence"], kmer_index, k)
    y_tr = train["binding_label"].values
    y_te = test["binding_label"].values
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = clone(get_model(model_name, seed))
    model.fit(X_tr_s, y_tr)
    prob = model.predict_proba(X_te_s)[:, 1]
    return {
        "auroc": safe_auroc(y_te, prob),
        "auprc": safe_auprc(y_te, prob),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_deduped_removed": n_deduped,
        "split_mode": "honest_trainval_test",
    }


def transfer_eval(
    src: pd.DataFrame,
    tgt: pd.DataFrame,
    model_name: str,
    kmer_index: dict[str, int],
    k: int,
    seed: int,
    min_pos: int,
    min_neg: int,
    min_examples: int,
) -> dict | None:
    src, n_ded_s = dedupe_by_sequence(src)
    tgt, n_ded_t = dedupe_by_sequence(tgt)
    n_pos_s = int((src["binding_label"] == 1).sum())
    n_neg_s = int((src["binding_label"] == 0).sum())
    n_pos_t = int((tgt["binding_label"] == 1).sum())
    n_neg_t = int((tgt["binding_label"] == 0).sum())
    if (
        n_pos_s < min_pos
        or n_neg_s < min_neg
        or n_pos_t < min_pos
        or n_neg_t < min_neg
        or len(src) < min_examples
        or len(tgt) < min_examples
    ):
        return None
    if src["binding_label"].nunique() < 2 or tgt["binding_label"].nunique() < 2:
        return None

    # Exact sequence overlap leakage check (same RNA string in train and test)
    overlap = set(src["rna_sequence"]) & set(tgt["rna_sequence"])
    n_overlap = len(overlap)

    X_tr = encode_rna_matrix(src["rna_sequence"], kmer_index, k)
    X_te = encode_rna_matrix(tgt["rna_sequence"], kmer_index, k)
    y_tr = src["binding_label"].values
    y_te = tgt["binding_label"].values
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = clone(get_model(model_name, seed))
    model.fit(X_tr_s, y_tr)
    prob = model.predict_proba(X_te_s)[:, 1]
    return {
        "auroc": safe_auroc(y_te, prob),
        "auprc": safe_auprc(y_te, prob),
        "n_train": int(len(src)),
        "n_test": int(len(tgt)),
        "n_pos_train": n_pos_s,
        "n_neg_train": n_neg_s,
        "n_pos_test": n_pos_t,
        "n_neg_test": n_neg_t,
        "n_deduped_removed_train": n_ded_s,
        "n_deduped_removed_test": n_ded_t,
        "n_exact_rna_overlap": n_overlap,
        "exact_rna_overlap_fraction_of_test": float(n_overlap / max(len(tgt), 1)),
    }


def load_cached_within_lr(cfg: dict, roster: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Load within-protocol metrics ONLY for the requested model from model_comparison TSVs.

    Falls back to per_protein_metrics only if it contains a model column matching
    model_name. Refuses best_model aggregates (those mix RF/LR).
    """
    if model_name != "logistic_regression":
        raise SystemExit(
            "--within_from_cached currently supports only logistic_regression "
            "(use model_comparison files). Recompute with --model for RF."
        )
    paths = cfg.get("paths", {}).get("cached_metrics", {})
    rows = []
    for protocol, rel in paths.items():
        metrics_path = resolve(rel)
        # Prefer sibling model_comparison next to per_protein_metrics
        comparison = metrics_path.with_name(
            metrics_path.name.replace("_per_protein_metrics.tsv", "_model_comparison.tsv")
        )
        src_path = comparison if comparison.exists() else metrics_path
        if not src_path.exists():
            print(f"[warn] cached metrics missing: {src_path}")
            continue
        df = pd.read_csv(src_path, sep="\t")
        if "model" in df.columns:
            df = df[df["model"] == "logistic_regression"].copy()
            auroc_col = "test_auroc" if "test_auroc" in df.columns else "auroc"
            auprc_col = "test_auprc" if "test_auprc" in df.columns else "auprc"
        elif "best_model" in df.columns:
            raise SystemExit(
                f"{src_path} is a best_model summary (mostly RF). "
                f"Provide model_comparison.tsv with logistic_regression rows, "
                f"or omit --within_from_cached."
            )
        else:
            raise SystemExit(f"Unrecognized metrics schema: {src_path}")

        if df.empty:
            print(f"[warn] no logistic_regression rows in {src_path}")
            continue
        df = df.rename(columns={auroc_col: "auroc", auprc_col: "auprc"})
        df["protocol"] = protocol
        df["source_file"] = str(src_path.relative_to(ROOT)) if src_path.is_relative_to(ROOT) else str(src_path)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    # Restrict to roster representatives: match native name to name_in_<protocol>
    keep_idx = []
    for i, r in out.iterrows():
        col = f"name_in_{r['protocol']}"
        if col not in roster.columns:
            continue
        allowed = set(roster[col].dropna().astype(str))
        if str(r["protein_name"]) in allowed:
            keep_idx.append(i)
    out = out.loc[keep_idx].copy()
    # Attach protein_key from roster
    key_maps = {}
    for protocol in out["protocol"].unique():
        col = f"name_in_{protocol}"
        sub = roster[roster[col].astype(str).str.len() > 0][["protein_key", col]]
        key_maps[protocol] = dict(zip(sub[col].astype(str), sub["protein_key"]))
    out["protein_key"] = [
        key_maps.get(p, {}).get(str(n), "")
        for p, n in zip(out["protocol"], out["protein_name"])
    ]
    out = out[out["protein_key"] != ""]
    out = out.drop_duplicates(["protein_key", "protocol"], keep="first")
    out["model"] = "logistic_regression"
    out["split_mode"] = "honest_cached_lr_test"
    return out


def slice_protein(df: pd.DataFrame, native_name: str) -> pd.DataFrame:
    """Exact native name match — no construct merging."""
    return df[df["protein_name"].astype(str) == str(native_name)].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_protocol.yaml")
    ap.add_argument("--roster", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--within_from_cached", action="store_true")
    ap.add_argument("--skip_transfer", action="store_true")
    ap.add_argument("--skip_within", action="store_true")
    ap.add_argument("--max_proteins", type=int, default=None)
    ap.add_argument("--protocols", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    paths = cfg["paths"]
    out_dir = resolve(args.out_dir or paths["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    roster_path = resolve(args.roster) if args.roster else out_dir / "protein_roster.tsv"
    if not roster_path.exists():
        raise SystemExit(
            f"Roster not found: {roster_path}\nRun scripts/33_build_cross_protocol_roster.py first."
        )
    roster = pd.read_csv(roster_path, sep="\t")
    if args.max_proteins:
        roster = roster.head(args.max_proteins).copy()

    # Require representative native columns
    proto_cfg = {p["id"]: p for p in cfg["protocols"]}
    protocol_ids = list(proto_cfg.keys())
    if args.protocols:
        protocol_ids = [p.strip() for p in args.protocols.split(",") if p.strip()]
    for pid in protocol_ids:
        col = f"name_in_{pid}"
        if col not in roster.columns:
            raise SystemExit(
                f"Roster missing {col}. Re-run scripts/33_build_cross_protocol_roster.py "
                f"(updated matching policy)."
            )

    model_name = args.model or cfg.get("model", "logistic_regression")
    seed = int(cfg.get("seed", 42))
    k = int(cfg.get("k", 4))
    min_pos = int(cfg.get("min_pos", 30))
    min_neg = int(cfg.get("min_neg", 30))
    min_examples = int(cfg.get("min_examples", 80))
    test_frac = float(cfg.get("test_frac", 0.2))
    val_frac = float(cfg.get("val_frac", 0.2))
    kmer_index = build_kmer_index(RNA_ALPHABET, k)

    # Needed native names per protocol
    needed_natives: dict[str, set[str]] = {}
    for pid in protocol_ids:
        col = f"name_in_{pid}"
        needed_natives[pid] = {
            str(x) for x in roster[col].dropna().astype(str) if str(x).strip()
        }

    data_by_protocol: dict[str, pd.DataFrame] = {}
    load_stats: dict[str, dict] = {}
    t0 = time.time()
    for pid in protocol_ids:
        pinfo = proto_cfg[pid]
        path_key = pinfo.get("path_key", pid)
        path = resolve(paths[path_key])
        if not path.exists():
            print(f"[warn] missing data for {pid}: {path}")
            continue
        print(f"Loading {pid} from {path} ...")
        df, stats = load_protocol_frame(
            path,
            pid,
            pinfo.get("protein_col"),
            bool(pinfo.get("rnacompete", False)),
        )
        before = len(df)
        df = df[df["protein_name"].isin(needed_natives[pid])].copy()
        stats["n_rows_roster_natives"] = len(df)
        stats["n_roster_natives_present"] = int(df["protein_name"].nunique())
        stats["n_roster_natives_expected"] = len(needed_natives[pid])
        stats["n_rows_before_native_filter"] = before
        data_by_protocol[pid] = df
        load_stats[pid] = stats
        print(
            f"  kept {len(df):,} rows / {df['protein_name'].nunique()} natives "
            f"(expected {len(needed_natives[pid])}); invalid RNA dropped={stats['n_invalid_rna_dropped']}"
        )
    print(f"Load time: {time.time() - t0:.1f}s")

    dom_map = dict(zip(roster["protein_key"], roster.get("domain_class", pd.Series(dtype=str))))
    arch_map = dict(
        zip(roster["protein_key"], roster.get("domain_architecture", pd.Series(dtype=str)))
    )
    # native -> key per protocol
    native_to_key: dict[str, dict[str, str]] = {}
    for pid in protocol_ids:
        col = f"name_in_{pid}"
        native_to_key[pid] = {
            str(n): k
            for n, k in zip(roster[col].astype(str), roster["protein_key"])
            if str(n).strip()
        }

    within_rows: list[dict] = []
    if args.within_from_cached and not args.skip_within:
        cached = load_cached_within_lr(cfg, roster, model_name)
        if cached.empty:
            print("[warn] cached LR within-metrics empty; will compute")
        else:
            for _, r in cached.iterrows():
                within_rows.append(
                    {
                        "protein_key": r["protein_key"],
                        "protein_name": r["protein_name"],
                        "protocol": r["protocol"],
                        "model": "logistic_regression",
                        "auroc": float(r["auroc"]),
                        "auprc": float(r["auprc"]) if pd.notna(r.get("auprc")) else np.nan,
                        "n_train": r.get("n_train", np.nan),
                        "n_test": r.get("n_test", np.nan),
                        "n_pos": r.get("n_pos", np.nan),
                        "n_neg": r.get("n_neg", np.nan),
                        "domain_class": dom_map.get(r["protein_key"], "unknown"),
                        "domain_architecture": arch_map.get(r["protein_key"], ""),
                        "source": "cached_model_comparison_lr",
                        "split_mode": "honest_cached_lr_test",
                        "source_file": r.get("source_file", ""),
                    }
                )
            print(f"Loaded {len(within_rows)} cached LR within-protocol rows")

    if not args.skip_within and not within_rows:
        print("Computing within-protocol metrics on representative natives only...")
        for pid, df in data_by_protocol.items():
            n_ok = 0
            for native in sorted(needed_natives[pid]):
                g = slice_protein(df, native)
                if g.empty:
                    continue
                key = native_to_key[pid].get(native, "")
                metrics = within_protocol_eval(
                    g,
                    model_name,
                    kmer_index,
                    k,
                    seed,
                    test_frac,
                    val_frac,
                    min_pos,
                    min_neg,
                    min_examples,
                )
                if metrics is None:
                    continue
                n_ok += 1
                within_rows.append(
                    {
                        "protein_key": key,
                        "protein_name": native,
                        "protocol": pid,
                        "model": model_name,
                        "domain_class": dom_map.get(key, "unknown"),
                        "domain_architecture": arch_map.get(key, ""),
                        "source": "computed",
                        **metrics,
                    }
                )
            print(f"  {pid}: {n_ok} proteins")

    within_df = pd.DataFrame(within_rows)
    within_path = out_dir / "within_protocol_metrics.tsv"
    within_df.to_csv(within_path, sep="\t", index=False)
    print(f"Wrote {within_path} ({len(within_df)} rows)")

    transfer_rows: list[dict] = []
    if not args.skip_transfer:
        print("Computing transfer metrics (representative natives only)...")
        loaded = list(data_by_protocol.keys())
        for src_id in loaded:
            for tgt_id in loaded:
                if src_id == tgt_id:
                    continue
                # proteins present in both with non-empty representatives
                pairs = roster[
                    (roster[f"name_in_{src_id}"].astype(str).str.len() > 0)
                    & (roster[f"name_in_{tgt_id}"].astype(str).str.len() > 0)
                ]
                n_ok = 0
                for _, row in pairs.iterrows():
                    src_native = str(row[f"name_in_{src_id}"])
                    tgt_native = str(row[f"name_in_{tgt_id}"])
                    key = row["protein_key"]
                    s = slice_protein(data_by_protocol[src_id], src_native)
                    t = slice_protein(data_by_protocol[tgt_id], tgt_native)
                    if s.empty or t.empty:
                        continue
                    metrics = transfer_eval(
                        s,
                        t,
                        model_name,
                        kmer_index,
                        k,
                        seed,
                        min_pos,
                        min_neg,
                        min_examples,
                    )
                    if metrics is None:
                        continue
                    n_ok += 1
                    transfer_rows.append(
                        {
                            "protein_key": key,
                            "train_protein_name": src_native,
                            "test_protein_name": tgt_native,
                            "train_protocol": src_id,
                            "test_protocol": tgt_id,
                            "model": model_name,
                            "domain_class": dom_map.get(key, "unknown"),
                            "domain_architecture": arch_map.get(key, ""),
                            **metrics,
                        }
                    )
                print(f"  {src_id} → {tgt_id}: {n_ok}/{len(pairs)} proteins")

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_path = out_dir / "transfer_metrics.tsv"
    transfer_df.to_csv(transfer_path, sep="\t", index=False)
    print(f"Wrote {transfer_path} ({len(transfer_df)} rows)")

    summary = {
        "model": model_name,
        "k": k,
        "seed": seed,
        "matching_policy": "representative_native_exact_match_only",
        "load_stats": load_stats,
        "n_within_rows": int(len(within_df)),
        "n_transfer_rows": int(len(transfer_df)),
        "protocols_loaded": list(data_by_protocol.keys()),
        "within_median_auroc_by_protocol": (
            within_df.groupby("protocol")["auroc"].median().dropna().to_dict()
            if not within_df.empty
            else {}
        ),
        "transfer_median_auroc_by_pair": {},
        "transfer_median_exact_rna_overlap_fraction": (
            float(transfer_df["exact_rna_overlap_fraction_of_test"].median())
            if not transfer_df.empty and "exact_rna_overlap_fraction_of_test" in transfer_df
            else None
        ),
        "transfer_median_auroc_by_domain_class": (
            transfer_df.groupby("domain_class")["auroc"].median().dropna().to_dict()
            if not transfer_df.empty and "domain_class" in transfer_df.columns
            else {}
        ),
    }
    if not transfer_df.empty:
        for (a, b), g in transfer_df.groupby(["train_protocol", "test_protocol"]):
            summary["transfer_median_auroc_by_pair"][f"{a}->{b}"] = {
                "median_auroc": float(g["auroc"].median()),
                "n": int(len(g)),
                "median_rna_overlap_frac": float(
                    g["exact_rna_overlap_fraction_of_test"].median()
                ),
            }
    summary_path = out_dir / "classifier_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
