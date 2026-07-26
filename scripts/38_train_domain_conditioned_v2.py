#!/usr/bin/env python3
"""
38_train_domain_conditioned_v2.py
---------------------------------
Main domain-aware experiment (replaces the abandoned construct-replace ablation
as the headline design):

  A) baseline            — RNABindingCNN on sanitized generalized_v3a
                           (protein sequence unchanged; no domain input)
  B) domain_conditioned  — same encoders + learnable embedding of coarse
                           domain_class (RRM / KH / multi / … / unknown)
                           concatenated into the MLP head
  C) domain_shuffle      — same as B but domain labels shuffled across
                           variants (negative control: architecture capacity
                           without real domain signal)

Why not construct-replace?
  On v3a most Table S1 matches are RNAcompete rows where train seq already
  equals the construct, so full vs construct is nearly a no-op. Domain-class
  conditioning uses labels on ~75% of variants and keeps all assays.

All arms use the same row filter by default: drop domain_class=unknown
(~78% of train rows kept). Pass --include_unknown only for a full-v3a sanity run.

Usage
-----
  python scripts/38_train_domain_conditioned_v2.py --qc_only --refresh_qc

  python scripts/38_train_domain_conditioned_v2.py \\
    --mode baseline --prot_max 700 --seed 42

  python scripts/38_train_domain_conditioned_v2.py \\
    --mode domain_conditioned --prot_max 700 --seed 42

  python scripts/38_train_domain_conditioned_v2.py \\
    --mode domain_shuffle --prot_max 700 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import one_hot_encode, _AA_LUT, _RNA_LUT  # noqa: E402
from src.data.domain_constructs import (  # noqa: E402
    DOMAIN_CLASS_VOCAB,
    alignment_summary,
    build_alignment_table,
    build_domain_label_maps,
    index_constructs_by_key,
    load_table_s1_constructs,
    variant_key,
)
from src.data.protein_names import base_gene_key  # noqa: E402
from src.models.cnn_model import RNABindingCNN, RNABindingCNNDomainCond  # noqa: E402

MODES = ("baseline", "domain_conditioned", "domain_shuffle")


def resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def assert_sanitized_data_dir(data_dir: Path, allow_unsanitized: bool) -> None:
    if "sanitized" in str(data_dir.resolve()):
        return
    if allow_unsanitized:
        print("WARNING: --allow_unsanitized set")
        return
    raise SystemExit(
        f"Refusing data_dir={data_dir}. Use data/sanitized/generalized_v3a "
        f"or pass --allow_unsanitized."
    )


def unique_variants_from_splits(data_dir: Path) -> pd.DataFrame:
    cols_wanted = {"protein_name", "protein_sequence", "dataset_source", "dataset"}
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(
            data_dir / f"{split}.tsv", sep="\t", usecols=lambda c: c in cols_wanted
        )
        if "dataset_source" not in df.columns and "dataset" in df.columns:
            df = df.rename(columns={"dataset": "dataset_source"})
        if "dataset_source" not in df.columns:
            df["dataset_source"] = "unknown"
        u = df.groupby(["protein_name", "protein_sequence"], as_index=False).agg(
            dataset_source=("dataset_source", "first"),
            n_rows=("dataset_source", "size"),
        )
        u["split_first_seen"] = split
        frames.append(u)
    all_u = pd.concat(frames, ignore_index=True)
    variants = (
        all_u.sort_values("split_first_seen")
        .drop_duplicates(["protein_name", "protein_sequence"])
        .reset_index(drop=True)
    )
    multi = variants.groupby("protein_name")["protein_sequence"].nunique()
    multi = multi[multi > 1]
    if len(multi):
        print(
            f"NOTE: {len(multi)} protein_name(s) have multiple sequences; "
            f"labels keyed by variant_id=name||seq."
        )
    return variants


class DomainSeqDataset(Dataset):
    """One-hot RNA/protein + optional domain class id per row."""

    def __init__(
        self,
        df: pd.DataFrame,
        domain_id_by_variant: dict[str, int],
        *,
        rna_max_len: int,
        prot_max_len: int,
        use_domain: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.rna_max = rna_max_len
        self.prot_max = prot_max_len
        self.use_domain = use_domain
        names = self.df["protein_name"].astype(str).to_numpy()
        seqs = self.df["protein_sequence"].astype(str).to_numpy()
        self._rna = self.df["rna_sequence"].astype(str).str.upper().to_numpy()
        self._prot = np.array([str(s).upper().rstrip("*") for s in seqs], dtype=object)
        self._y = self.df["binding_label"].astype(np.float32).to_numpy()
        self._dom = np.array(
            [
                int(domain_id_by_variant.get(variant_key(n, s), 0))
                for n, s in zip(names, seqs)
            ],
            dtype=np.int64,
        )
        self._names = names

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int):
        rna = one_hot_encode(self._rna[idx], self.rna_max, _RNA_LUT, 4)
        prot = one_hot_encode(self._prot[idx], self.prot_max, _AA_LUT, 20)
        y = torch.tensor(self._y[idx], dtype=torch.float32)
        if self.use_domain:
            d = torch.tensor(self._dom[idx], dtype=torch.long)
            return rna, prot, d, y
        return rna, prot, y


def build_or_load_qc(args) -> pd.DataFrame:
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_path = out_dir / "construct_alignment_qc.tsv"
    data_dir = resolve(args.data_dir)
    table_s1 = resolve(args.table_s1)

    if qc_path.exists() and not args.refresh_qc:
        print(f"Reusing QC {qc_path}")
        return pd.read_csv(qc_path, sep="\t", low_memory=False)

    print("Building alignment QC + domain label table…")
    records = load_table_s1_constructs(table_s1)
    by_key = index_constructs_by_key(records)
    uniq = unique_variants_from_splits(data_dir)
    qc = build_alignment_table(uniq, by_key)
    qc.to_csv(qc_path, sep="\t", index=False)

    class_by_vid, id_by_vid, lab_sum = build_domain_label_maps(
        qc, resolve(args.domains_tsv)
    )
    label_rows = []
    for vid, c in class_by_vid.items():
        label_rows.append(
            {
                "variant_id": vid,
                "domain_class": c,
                "domain_id": id_by_vid[vid],
            }
        )
    lab_path = out_dir / "domain_labels_by_variant.tsv"
    pd.DataFrame(label_rows).to_csv(lab_path, sep="\t", index=False)

    summary = alignment_summary(qc)
    summary["domain_labels"] = lab_sum
    summary["table_s1"] = str(table_s1)
    summary["data_dir"] = str(data_dir)
    with open(out_dir / "construct_alignment_qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(lab_sum, indent=2))
    print(f"Wrote {qc_path}")
    print(f"Wrote {lab_path}")
    return qc


def shuffle_domain_ids(
    id_by_vid: dict[str, int], seed: int
) -> dict[str, int]:
    """Permute domain ids across variants (fixed seed)."""
    vids = sorted(id_by_vid.keys())
    ids = [id_by_vid[v] for v in vids]
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    return dict(zip(vids, shuffled))


def evaluate(model, loader, device, use_domain: bool) -> dict:
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            if use_domain:
                rna, prot, dom, y = batch
                rna, prot, dom = rna.to(device), prot.to(device), dom.to(device)
                p = torch.sigmoid(model(rna, prot, dom)).cpu().numpy()
            else:
                rna, prot, y = batch
                rna, prot = rna.to(device), prot.to(device)
                p = torch.sigmoid(model(rna, prot)).cpu().numpy()
            probs_all.append(p)
            labels_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "probs": probs,
    }


def per_protein_metrics(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    class_by_vid: dict[str, str],
) -> list[dict]:
    df = test_df.copy()
    df["prob"] = probs
    if "dataset" not in df.columns and "dataset_source" in df.columns:
        df = df.rename(columns={"dataset_source": "dataset"})
    df["_vid"] = [
        variant_key(str(n), str(s))
        for n, s in zip(df["protein_name"], df["protein_sequence"])
    ]
    rows = []
    for vid, grp in df.groupby("_vid"):
        if grp["binding_label"].nunique() < 2:
            continue
        name = str(grp["protein_name"].iloc[0])
        rows.append(
            {
                "protein": name,
                "protein_key": base_gene_key(name),
                "domain_class": class_by_vid.get(str(vid), "unknown"),
                "dataset": grp["dataset"].iloc[0] if "dataset" in grp.columns else "unknown",
                "auroc": float(roc_auc_score(grp["binding_label"], grp["prob"])),
                "n": int(len(grp)),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baseline vs domain-conditioned V2 CNN"
    )
    ap.add_argument("--mode", choices=MODES, default="baseline")
    ap.add_argument("--data_dir", default="data/sanitized/generalized_v3a")
    ap.add_argument("--table_s1", default="data/raw/rbpzoo/TableS1.xlsx")
    ap.add_argument("--domains_tsv", default="data/domains/protein_domains.tsv")
    ap.add_argument(
        "--out_dir",
        default="results/domain_aware/v2_domain_cond",
        help="Shared QC/labels + per-mode metrics parent",
    )
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--rna_max", type=int, default=60)
    ap.add_argument("--prot_max", type=int, default=700)
    ap.add_argument("--domain_emb_dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--qc_only", action="store_true")
    ap.add_argument("--refresh_qc", action="store_true")
    ap.add_argument(
        "--include_unknown",
        action="store_true",
        help="Keep variants with domain_class=unknown (default: drop them so "
        "baseline/conditioned/shuffle share a known-domain-only cohort)",
    )
    ap.add_argument("--allow_unsanitized", action="store_true")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    known_domains_only = not args.include_unknown

    data_dir = resolve(args.data_dir)
    assert_sanitized_data_dir(data_dir, args.allow_unsanitized)
    parent_out = resolve(args.out_dir)
    parent_out.mkdir(parents=True, exist_ok=True)

    qc = build_or_load_qc(args)
    if args.qc_only:
        # ensure labels written even when reusing QC
        class_by_vid, id_by_vid, lab_sum = build_domain_label_maps(
            qc, resolve(args.domains_tsv)
        )
        pd.DataFrame(
            [
                {"variant_id": v, "domain_class": c, "domain_id": id_by_vid[v]}
                for v, c in class_by_vid.items()
            ]
        ).to_csv(parent_out / "domain_labels_by_variant.tsv", sep="\t", index=False)
        with open(parent_out / "domain_label_summary.json", "w") as f:
            json.dump(lab_sum, f, indent=2)
        print(json.dumps(lab_sum, indent=2))
        return

    class_by_vid, id_by_vid, lab_sum = build_domain_label_maps(
        qc, resolve(args.domains_tsv)
    )
    if known_domains_only:
        known_vids = {v for v, c in class_by_vid.items() if c != "unknown"}
        class_by_vid = {v: class_by_vid[v] for v in known_vids}
        id_by_vid = {v: id_by_vid[v] for v in known_vids}
        lab_sum = {
            **lab_sum,
            "known_domains_only": True,
            "n_variants_kept": len(known_vids),
            "n_unknown_dropped": int(lab_sum["n_unknown"]),
            "domain_class_counts": pd.Series(list(class_by_vid.values()))
            .value_counts()
            .to_dict(),
        }
        print(
            f"  known_domains_only: kept {len(known_vids)} variants "
            f"(dropped {lab_sum['n_unknown_dropped']} unknown)"
        )
    else:
        lab_sum = {**lab_sum, "known_domains_only": False}
        known_vids = set(class_by_vid.keys())
        print("  include_unknown: training on full v3a (unknown kept)")

    if args.mode == "domain_shuffle":
        id_by_vid = shuffle_domain_ids(id_by_vid, seed=args.seed + 17)
        print(f"Shuffled domain labels (seed={args.seed + 17})")

    use_domain = args.mode in {"domain_conditioned", "domain_shuffle"}
    mode_out = parent_out / args.mode
    model_dir = resolve(
        args.model_dir if args.model_dir else f"models/saved/domain_v2_{args.mode}"
    )
    if not args.dry_run:
        mode_out.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else (0 if sys.platform == "darwin" else 2)
    )
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
    print(f"  mode={args.mode}  seed={args.seed}  prot_max={args.prot_max}")
    print(f"  domain labels: {lab_sum['domain_class_counts']}")
    if not known_domains_only:
        print(f"  unknown variants: {lab_sum['n_unknown']}/{lab_sum['n_variants']}")

    def make_loader(split: str, shuffle: bool):
        df = pd.read_csv(data_dir / f"{split}.tsv", sep="\t", low_memory=False)
        if known_domains_only:
            vids = [
                variant_key(str(n), str(s))
                for n, s in zip(df["protein_name"], df["protein_sequence"])
            ]
            df = df.assign(_vid=vids)
            df = df[df["_vid"].isin(known_vids)].drop(columns=["_vid"]).reset_index(
                drop=True
            )
            if len(df) == 0:
                raise SystemExit(f"No rows left in {split} after known_domains_only")
        ds = DomainSeqDataset(
            df,
            id_by_vid,
            rna_max_len=args.rna_max,
            prot_max_len=args.prot_max,
            use_domain=use_domain,
        )
        gen = None
        if shuffle:
            gen = torch.Generator()
            gen.manual_seed(args.seed)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size if shuffle else args.batch_size * 2,
            shuffle=shuffle,
            generator=gen,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        return ds, loader, df

    train_ds, train_loader, train_df = make_loader("train", True)
    val_ds, val_loader, _ = make_loader("val", False)
    test_ds, test_loader, test_df = make_loader("test", False)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    if use_domain:
        model = RNABindingCNNDomainCond(
            n_domain_classes=len(DOMAIN_CLASS_VOCAB),
            domain_emb_dim=args.domain_emb_dim,
            rna_filters=[128, 256, 256],
            rna_kernels=[7, 5, 3],
            prot_filters=[128, 256, 256],
            prot_kernels=[11, 7, 5],
            head_dims=[256, 64],
            dropout=args.dropout,
        ).to(device)
    else:
        model = RNABindingCNN(
            rna_filters=[128, 256, 256],
            rna_kernels=[7, 5, 3],
            prot_filters=[128, 256, 256],
            prot_kernels=[11, 7, 5],
            head_dims=[256, 64],
            dropout=args.dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    n_pos = int((train_ds._y == 1).sum())
    n_neg = int((train_ds._y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    run_meta = {
        "mode": args.mode,
        "data_dir": str(data_dir),
        "seed": args.seed,
        "rna_max": args.rna_max,
        "prot_max": args.prot_max,
        "known_domains_only": known_domains_only,
        "domain_emb_dim": args.domain_emb_dim if use_domain else None,
        "domain_vocab": list(DOMAIN_CLASS_VOCAB),
        "domain_label_summary": lab_sum,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "device": str(device),
        "n_params": n_params,
    }

    def train_step_batch(batch):
        if use_domain:
            rna, prot, dom, y = batch
            rna, prot, dom, y = (
                rna.to(device),
                prot.to(device),
                dom.to(device),
                y.to(device),
            )
            return criterion(model(rna, prot, dom), y), len(y)
        rna, prot, y = batch
        rna, prot, y = rna.to(device), prot.to(device), y.to(device)
        return criterion(model(rna, prot), y), len(y)

    if args.dry_run:
        print("\n=== DRY RUN ===")
        model.train()
        batch = next(iter(train_loader))
        optimizer.zero_grad()
        loss, _ = train_step_batch(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mode_out.mkdir(parents=True, exist_ok=True)
        with open(mode_out / "dry_run_meta.json", "w") as f:
            json.dump(run_meta, f, indent=2)
        print(f"  OK loss={loss.item():.4f}")
        return

    print(
        f"\n=== Training ({args.mode}) max {args.epochs} epochs, "
        f"early stop on val AUPRC, patience={args.patience} ==="
    )
    best_auprc, best_auroc, best_epoch, no_improve = 0.0, 0.0, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, bs = train_step_batch(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * bs
            n_seen += bs
        train_loss /= max(n_seen, 1)

        val_m = evaluate(model, val_loader, device, use_domain)
        scheduler.step()
        elapsed = time.time() - t0
        is_best = val_m["auprc"] > best_auprc
        marker = "★" if is_best else ""
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_auroc": round(val_m["auroc"], 4),
                "val_auprc": round(val_m["auprc"], 4),
            }
        )
        print(
            f"  {epoch:>5}  {train_loss:>8.4f}  {val_m['auroc']:>9.4f}  "
            f"{val_m['auprc']:>9.4f}  {elapsed:>5.1f}s  {marker}"
        )
        if is_best:
            best_auprc, best_auroc, best_epoch, no_improve = (
                val_m["auprc"],
                val_m["auroc"],
                epoch,
                0,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": {"auroc": val_m["auroc"], "auprc": val_m["auprc"]},
                    "args": vars(args),
                    "run_meta": run_meta,
                },
                model_dir / "best_model.pt",
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device, use_domain)
    print(f"\n=== Test === AUROC={test_m['auroc']:.4f}  AUPRC={test_m['auprc']:.4f}")

    per_protein = per_protein_metrics(test_df, test_m["probs"], class_by_vid)
    pp = [p["auroc"] for p in per_protein]
    by_dom: dict[str, list[float]] = {}
    for p in per_protein:
        by_dom.setdefault(p["domain_class"], []).append(p["auroc"])

    def _med(xs):
        return float(np.median(xs)) if xs else None

    results = {
        **run_meta,
        "best_val_auroc": best_auroc,
        "best_val_auprc": best_auprc,
        "best_epoch": best_epoch,
        "test_metrics": {"auroc": test_m["auroc"], "auprc": test_m["auprc"]},
        "per_protein_summary": {
            "n": len(pp),
            "median": _med(pp),
            "min": float(np.min(pp)) if pp else None,
            "by_domain_class_median": {k: _med(v) for k, v in sorted(by_dom.items())},
        },
        "per_protein": per_protein,
        "history": history,
        "checkpoint": str(model_dir / "best_model.pt"),
    }
    out_json = mode_out / "v2_domain_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(per_protein).to_csv(mode_out / "per_protein_test.tsv", sep="\t", index=False)
    print(f"\n✅ Wrote {out_json}")
    print(
        f"   Per-protein median={results['per_protein_summary']['median']}  "
        f"by domain={results['per_protein_summary']['by_domain_class_median']}"
    )


if __name__ == "__main__":
    main()
