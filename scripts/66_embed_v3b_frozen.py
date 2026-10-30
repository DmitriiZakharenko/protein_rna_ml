#!/usr/bin/env python3
"""
66_embed_v3b_frozen.py
----------------------
Mean-pooled frozen embeddings for every unique protein and RNA of the v3b split (and of the v3b cross-protein probe):
proteins with ESM-2 (650M, layer 33, sequences cut to 1022 residues), RNAs with RNA-FM (layer 12).

Outputs in --out_dir:
  prot_esm2.npz   names (n,), emb (n, 1280) float16
  rna_rnafm.npz   seqs  (m,), emb (m, 640)  float16

Usage
-----
  python scripts/66_embed_v3b_frozen.py --stage proteins
  python scripts/66_embed_v3b_frozen.py --stage rna --rna_batch 256
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_tables(data_dir: Path, probe: Path | None):
    cols = ["protein_name", "rna_sequence", "protein_sequence"]
    dfs = [pd.read_csv(data_dir / f"{s}.tsv", sep="\t", usecols=cols, low_memory=False) for s in ("train", "val", "test")]
    if probe is not None and probe.exists():
        dfs.append(pd.read_csv(probe, sep="\t", usecols=cols, low_memory=False))
    return pd.concat(dfs, ignore_index=True)


def clean(seq: str, alphabet: str) -> str:
    return "".join(c if c in alphabet else "X" for c in str(seq).upper().replace("T", "U" if alphabet == "ACGU" else "T"))


@torch.no_grad()
def embed_proteins(df: pd.DataFrame, out: Path, device) -> None:
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.half().to(device).eval()
    conv = alphabet.get_batch_converter()
    prots = df.drop_duplicates("protein_name").set_index("protein_name")["protein_sequence"]
    names = sorted(prots.index, key=lambda n: len(prots[n]))
    embs = []
    for i, n in enumerate(names):
        seq = clean(prots[n], "ACDEFGHIKLMNPQRSTVWYX")[:1022]
        _, _, tok = conv([(n, seq)])
        rep = model(tok.to(device), repr_layers=[33])["representations"][33][0, 1:len(seq) + 1]
        embs.append(rep.float().mean(0).cpu().numpy().astype(np.float16))
        if (i + 1) % 50 == 0:
            print(f"  proteins {i + 1}/{len(names)}", flush=True)
    np.savez(out / "prot_esm2.npz", names=np.array(names), emb=np.stack(embs))
    print(f"wrote {out / 'prot_esm2.npz'}")


@torch.no_grad()
def embed_rna(df: pd.DataFrame, out: Path, device, batch: int) -> None:
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.half().to(device).eval()
    conv = alphabet.get_batch_converter()
    seqs = sorted(df.rna_sequence.unique(), key=len)
    embs = np.zeros((len(seqs), 640), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(seqs), batch):
        chunk = [clean(s, "ACGU") for s in seqs[i:i + batch]]
        _, _, tok = conv([(str(j), s) for j, s in enumerate(chunk)])
        rep = model(tok.to(device), repr_layers=[12])["representations"][12]
        for j, s in enumerate(chunk):
            embs[i + j] = rep[j, 1:len(s) + 1].float().mean(0).cpu().numpy()
        if (i // batch) % 200 == 0:
            print(f"  rna {i + len(chunk)}/{len(seqs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "rna_rnafm.npz", seqs=np.array(seqs), emb=embs)
    print(f"wrote {out / 'rna_rnafm.npz'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["proteins", "rna"], required=True)
    ap.add_argument("--data_dir", default="data/generalized_v3b")
    ap.add_argument("--probe_tsv", default="data/benchmarks/v3b_cross_neg/cross_neg_probe/test.tsv")
    ap.add_argument("--out_dir", default="data/embeddings_v3b")
    ap.add_argument("--rna_batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N unique sequences")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_tables(resolve(args.data_dir), resolve(args.probe_tsv))
    if args.limit:
        keep = df.drop_duplicates("protein_name" if args.stage == "proteins" else "rna_sequence").head(args.limit)
        df = keep
    print(f"device={device} proteins={df.protein_name.nunique():,} rnas={df.rna_sequence.nunique():,}")
    if args.stage == "proteins":
        embed_proteins(df, out, device)
    else:
        embed_rna(df, out, device, args.rna_batch)


if __name__ == "__main__":
    main()
