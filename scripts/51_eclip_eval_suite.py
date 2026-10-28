#!/usr/bin/env python3
"""
51_eclip_eval_suite.py
----------------------
One honest evaluation table for Skipper eCLIP benchmarks (step 1 of the
post-Phase-3B plan).

For every split directory (train/val/test TSV) it scores:

  * composition baselines  — GC fraction, RNA k-mer (k<=4) logistic regression
                             trained on the split's own train set (protein-agnostic)
  * neural checkpoints     — V2 CNN / V4 bilinear, batched, with several window
                             aggregations (center / mean / max / full-length)
  * protein-shuffle control — same checkpoint with protein sequences permuted
                             across test rows; if AUROC does not drop, the model
                             is not using the protein

on four subsets: full test, GC-matched negatives, RNA-unseen (RNA absent from
train) and, optionally, proteins absent from a v3a train TSV (name-level only;
sequence-level homology audit is a separate step).

Every TSV is checked for real protein sequences first — a stale build where
``protein_sequence == "ok"`` aborts the run.

Usage
-----
  # baselines only (CPU)
  python scripts/51_eclip_eval_suite.py \\
      --splits cross_protein_neg protein_and_rna protein_disjoint

  # + zero-shot in vitro checkpoints and an in-domain model
  python scripts/51_eclip_eval_suite.py \\
      --splits cross_protein_neg protein_and_rna \\
      --v3a_train_tsv data/sanitized/generalized_v3a/train.tsv \\
      --model "v2_v3a,type=v2,path=models/saved/generalized_v2/best_model.pt,rna_max=60,prot_max=700" \\
      --model "v4_v3a,type=v4,path=models/saved/generalized_v4_phase3a/best_model.pt,rna_max=60,prot_max=700" \\
      --model "v2_eclip,type=v2,path=models/saved/skipper_eclip_v2_cross_protein_neg/best_model.pt,rna_max=151,prot_max=700,splits=cross_protein_neg"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cnn_model import RNABindingCNN  # noqa: E402

BENCH_DIR = ROOT / "data" / "benchmarks" / "skipper_eclip"
RNA_ALPHA = "AUGC"  # same channel order as scripts/11_evaluate_external.py
AA_ALPHA = "ACDEFGHIKLMNPQRSTVWY"
AGGS = ("center", "mean", "max", "full")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_diag41d = _load_module("eclip_diag_41d", ROOT / "scripts" / "41d_eval_eclip_diagnostics.py")
_ext11 = _load_module("evaluate_external_11", ROOT / "scripts" / "11_evaluate_external.py")
gc_matched_subset = _diag41d.gc_matched_subset
rna_windows = _ext11.rna_windows


def resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


# ── data ─────────────────────────────────────────────────────────────────────
def assert_protein_sequences(df: pd.DataFrame, context: str) -> None:
    lens = df.groupby("protein_name")["protein_sequence"].first().astype(str).str.len()
    n_ok = int((df["protein_sequence"].astype(str) == "ok").sum())
    if n_ok or df["protein_sequence"].nunique() < 2 or float(lens.min()) < 10:
        raise SystemExit(
            f"Protein sequence QC failed for {context}: rows_eq_ok={n_ok}, "
            f"min_len={int(lens.min())}. Rebuild with scripts 41/41b/41c/42 "
            "(stale build from before the validate-reason fix)."
        )


def gc_frac(seq: str) -> float:
    s = str(seq).upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def load_split(split_dir: Path, max_train_rows: int = 0, seed: int = 42) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "test"):
        path = split_dir / f"{name}.tsv"
        if not path.exists():
            raise SystemExit(f"Missing {path} — run the build scripts first")
        df = pd.read_csv(path, sep="\t", low_memory=False)
        assert_protein_sequences(df, str(path))
        out[name] = df
    out["test"]["gc"] = out["test"]["rna_sequence"].map(gc_frac)
    out["test"]["rna_in_train"] = out["test"]["rna_sequence"].isin(set(out["train"]["rna_sequence"]))
    if max_train_rows and len(out["train"]) > max_train_rows:  # only for the k-mer baseline
        out["train"] = out["train"].sample(max_train_rows, random_state=seed)
    return out


def make_subsets(test: pd.DataFrame, seed: int, v3a_proteins: set[str] | None) -> dict[str, pd.DataFrame]:
    subsets = {
        "full": test,
        "gc_matched": gc_matched_subset(test, seed=seed),
        "rna_unseen": test[~test["rna_in_train"]],
    }
    if v3a_proteins is not None:
        subsets["protein_not_in_v3a"] = test[~test["protein_name"].isin(v3a_proteins)]
    return subsets


# ── metrics ──────────────────────────────────────────────────────────────────
def _auc(y: np.ndarray, s: np.ndarray) -> tuple[float | None, float | None]:
    if len(y) == 0 or len(set(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def per_protein_auroc(df: pd.DataFrame, col: str) -> dict[str, float]:
    res = {}
    for prot, g in df.groupby("protein_name"):
        a, _ = _auc(g["binding_label"].values, g[col].values)
        if a is not None:
            res[prot] = a
    return res


def metrics(df: pd.DataFrame, col: str, n_boot: int, rng: np.random.Generator) -> dict:
    y, s = df["binding_label"].values, df[col].values
    auroc, auprc = _auc(y, s)
    if auroc is None:
        return {"n": int(len(df)), "auroc": None}
    pp = per_protein_auroc(df, col)
    prots = df["protein_name"].unique()
    by_prot = {p: g for p, g in df.groupby("protein_name")}
    boots, boots_med = [], []
    for _ in range(n_boot):
        pick = rng.choice(prots, size=len(prots), replace=True)
        samp = pd.concat([by_prot[p] for p in pick], ignore_index=True)
        a, _ = _auc(samp["binding_label"].values, samp[col].values)
        if a is not None:
            boots.append(a)
        vals = [pp[p] for p in pick if p in pp]
        if vals:
            boots_med.append(float(np.median(vals)))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if v else None
    return {
        "n": int(len(df)),
        "n_proteins": int(len(prots)),
        "auroc": round(auroc, 4),
        "auroc_ci95_protein_bootstrap": ci(boots),
        "auprc": round(auprc, 4),
        "pp_median_auroc": round(float(np.median(list(pp.values()))), 4) if pp else None,
        "pp_median_ci95": ci(boots_med),
        "pp_frac_auroc_gt_0.5": round(float(np.mean([v > 0.5 for v in pp.values()])), 3) if pp else None,
    }


# ── baselines ────────────────────────────────────────────────────────────────
def kmer_lr_scores(train: pd.DataFrame, test: pd.DataFrame, k_max: int = 4) -> np.ndarray:
    vec = CountVectorizer(analyzer="char", ngram_range=(1, k_max), lowercase=False)
    to_rna = lambda s: s.str.upper().str.replace("T", "U")
    xtr = vec.fit_transform(to_rna(train["rna_sequence"]))
    xte = vec.transform(to_rna(test["rna_sequence"]))
    lens_tr = train["rna_sequence"].str.len().values[:, None]
    lens_te = test["rna_sequence"].str.len().values[:, None]
    xtr = xtr.multiply(1.0 / lens_tr).tocsr()
    xte = xte.multiply(1.0 / lens_te).tocsr()
    scaler = StandardScaler(with_mean=False).fit(xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(xtr), train["binding_label"].values)
    return clf.predict_proba(scaler.transform(xte))[:, 1]


# ── neural scoring ───────────────────────────────────────────────────────────
def _lut(alpha: str, extra: dict[str, str] | None = None) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int16)
    for i, c in enumerate(alpha):
        lut[ord(c)] = i
    for src, dst in (extra or {}).items():
        lut[ord(src)] = lut[ord(dst)]
    return lut


RNA_LUT = _lut(RNA_ALPHA, {"T": "U", "t": "U", "a": "A", "u": "U", "g": "G", "c": "C"})
AA_LUT = _lut(AA_ALPHA)


def one_hot(seqs: list[str], length: int, lut: np.ndarray, n_ch: int) -> np.ndarray:
    arr = np.zeros((len(seqs), length, n_ch), dtype=np.float32)
    for i, s in enumerate(seqs):
        idx = lut[np.frombuffer(str(s).upper().encode()[:length], dtype=np.uint8)]
        pos = np.nonzero(idx >= 0)[0]
        arr[i, pos, idx[pos]] = 1.0
    return arr


def load_model(kind: str, path: Path, device: torch.device):
    if kind == "v2":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = RNABindingCNN().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, lambda m, r, p: m(r, p)
    if kind == "v4":
        v4 = _load_module("eval_v4_21b", ROOT / "scripts" / "21b_evaluate_external_v4.py")
        model, _ = v4.load_v4_model(str(path), device)
        return model, lambda m, r, p: m(r, p, source_tags=None)
    raise SystemExit(f"unknown model type {kind!r} (use v2 or v4)")


@torch.no_grad()
def score_model(
    model,
    fwd,
    df: pd.DataFrame,
    prot_seqs: list[str],
    *,
    rna_max: int,
    prot_max: int,
    win_step: int,
    device: torch.device,
    batch: int,
) -> dict[str, np.ndarray]:
    """Return {agg: probs} for agg in center/mean/max/full."""
    rnas = df["rna_sequence"].astype(str).tolist()
    uniq = sorted(set(prot_seqs))
    pidx_map = {s: i for i, s in enumerate(uniq)}
    prot_t = torch.from_numpy(one_hot(uniq, prot_max, AA_LUT, 20)).to(device)
    row_p = np.array([pidx_map[s] for s in prot_seqs])

    def run(windows: list[str], owners: np.ndarray, length: int) -> np.ndarray:
        probs = np.zeros(len(windows), dtype=np.float32)
        for i in range(0, len(windows), batch):
            r = torch.from_numpy(one_hot(windows[i:i + batch], length, RNA_LUT, 4)).to(device)
            p = prot_t[torch.from_numpy(row_p[owners[i:i + batch]]).to(device)]
            probs[i:i + batch] = torch.sigmoid(fwd(model, r, p)).float().cpu().numpy()
        return probs

    n = len(rnas)
    out: dict[str, np.ndarray] = {}

    win, own = [], []
    for j, s in enumerate(rnas):
        ws = rna_windows(s, rna_max, win_step)
        win += ws
        own += [j] * len(ws)
    own_a = np.array(own)
    wp = run(win, own_a, rna_max)
    mx = np.full(n, -np.inf)
    np.maximum.at(mx, own_a, wp)
    sm = np.zeros(n)
    np.add.at(sm, own_a, wp)
    out["max"] = mx
    out["mean"] = sm / np.bincount(own_a, minlength=n)

    cen = []
    for s in rnas:
        s = s.upper()
        a = max((len(s) - rna_max) // 2, 0)
        cen.append(s[a:a + rna_max])
    out["center"] = run(cen, np.arange(n), rna_max)

    full_len = max(len(s) for s in rnas)
    out["full"] = run(rnas, np.arange(n), full_len)
    return out


def parse_model_spec(spec: str) -> dict:
    parts = [p.strip() for p in spec.split(",")]
    cfg = {"name": parts[0], "rna_max": 60, "prot_max": 700, "win_step": 30, "splits": None}
    for kv in parts[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = v
    for k in ("rna_max", "prot_max", "win_step"):
        cfg[k] = int(cfg[k])
    if cfg["splits"]:
        cfg["splits"] = cfg["splits"].split("|")
    if "type" not in cfg or "path" not in cfg:
        raise SystemExit(f"--model needs type= and path=: {spec}")
    return cfg


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench_dir", default=str(BENCH_DIR.relative_to(ROOT)),
                    help="Directory containing the split directories (default: eCLIP benchmarks)")
    ap.add_argument("--max_train_rows", type=int, default=0, help="Subsample train for the k-mer baseline")
    ap.add_argument("--splits", nargs="+", default=["cross_protein_neg", "protein_and_rna", "protein_disjoint"],
                    help="Split directory names under data/benchmarks/skipper_eclip/")
    ap.add_argument("--model", action="append", default=[], help="name,type=v2|v4,path=...,rna_max=,prot_max=,splits=a|b")
    ap.add_argument("--aggs", nargs="+", default=list(AGGS), choices=AGGS)
    ap.add_argument("--v3a_train_tsv", default=None)
    ap.add_argument("--out_dir", default="results/skipper_eclip/step1")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_shuffle_control", action="store_true")
    ap.add_argument("--max_test_rows", type=int, default=0, help="Subsample test rows (smoke tests)")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and not args.no_cuda:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    out_root = resolve(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    models = [parse_model_spec(s) for s in args.model]

    v3a_proteins = None
    if args.v3a_train_tsv:
        v3a_proteins = set(pd.read_csv(resolve(args.v3a_train_tsv), sep="\t", usecols=["protein_name"])["protein_name"])
        print(f"v3a train proteins (name-level): {len(v3a_proteins)}")

    loaded = {}
    summary: dict = {"script": "51_eclip_eval_suite.py", "seed": args.seed, "splits": {}}

    for split in args.splits:
        sdir = resolve(args.bench_dir) / split
        print(f"\n=== split: {split} ===")
        data = load_split(sdir, args.max_train_rows, args.seed)
        train, test = data["train"], data["test"].copy()
        if args.max_test_rows and len(test) > args.max_test_rows:
            test = test.sample(args.max_test_rows, random_state=args.seed).reset_index(drop=True)

        # protein-sequence permutation for the shuffle control (fixed per split)
        perm = np.random.default_rng(args.seed).permutation(len(test))
        shuf_prot = test["protein_sequence"].values[perm].tolist()

        score_cols: list[str] = []
        test["gc_baseline"] = test["gc"]
        score_cols.append("gc_baseline")
        test["kmer_lr_baseline"] = kmer_lr_scores(train, test)
        score_cols.append("kmer_lr_baseline")

        for cfg in models:
            if cfg["splits"] and split not in cfg["splits"]:
                continue
            ck = resolve(cfg["path"])
            if not ck.exists():
                print(f"  [skip] {cfg['name']}: checkpoint not found at {ck}")
                continue
            key = (cfg["type"], str(ck))
            if key not in loaded:
                loaded[key] = load_model(cfg["type"], ck, device)
            model, fwd = loaded[key]
            kw = dict(rna_max=cfg["rna_max"], prot_max=cfg["prot_max"], win_step=cfg["win_step"],
                      device=device, batch=args.batch_size)
            print(f"  scoring {cfg['name']} ...")
            real = score_model(model, fwd, test, test["protein_sequence"].tolist(), **kw)
            shuf = None if args.skip_shuffle_control else score_model(model, fwd, test, shuf_prot, **kw)
            for agg in args.aggs:
                col = f"{cfg['name']}__{agg}"
                test[col] = real[agg]
                score_cols.append(col)
                if shuf is not None:
                    test[f"{col}__protshuf"] = shuf[agg]
                    score_cols.append(f"{col}__protshuf")

        subsets = make_subsets(test, args.seed, v3a_proteins)
        table = {}
        for col in score_cols:
            table[col] = {name: metrics(sub, col, args.n_boot, rng) for name, sub in subsets.items()}
        for col in list(table):
            if col.endswith("__protshuf"):
                continue
            shuf_col = f"{col}__protshuf"
            if shuf_col in table:
                for name in subsets:
                    a, b = table[col][name].get("auroc"), table[shuf_col][name].get("auroc")
                    table[col][name]["protein_dependence_dAUROC"] = None if a is None or b is None else round(a - b, 4)
                    pa, pb = table[col][name].get("pp_median_auroc"), table[shuf_col][name].get("pp_median_auroc")
                    table[col][name]["protein_dependence_pp_median_dAUROC"] = None if pa is None or pb is None else round(pa - pb, 4)

        summary["splits"][split] = {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_test_proteins": int(test["protein_name"].nunique()),
            "gc_pos_mean": round(float(test.loc[test.binding_label == 1, "gc"].mean()), 4),
            "gc_neg_mean": round(float(test.loc[test.binding_label == 0, "gc"].mean()), 4),
            "pct_test_rna_in_train": round(100 * float(test["rna_in_train"].mean()), 2),
            "subset_sizes": {k: int(len(v)) for k, v in subsets.items()},
            "scores": table,
        }
        keep = ["protein_name", "rna_sequence", "binding_label"] + score_cols
        test[keep].to_csv(out_root / f"scores_{split}.tsv.gz", sep="\t", index=False)

        print(f"\n{'scorer':<34}{'full':>8}{'gc_match':>10}{'rna_unseen':>12}")
        for col in score_cols:
            if col.endswith("__protshuf"):
                continue
            g = lambda s: table[col][s].get("auroc")
            print(f"{col:<34}{str(g('full')):>8}{str(g('gc_matched')):>10}{str(g('rna_unseen')):>12}")

    with open(out_root / "step1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_root / 'step1_summary.json'}")


if __name__ == "__main__":
    main()
