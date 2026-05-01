"""
Script 07: Extract ESM-2 protein embeddings for all unique proteins.

ESM-2 (esm2_t33_650M_UR50D) encodes each amino acid into a 1280-dim context vector.
We mean-pool over residues → fixed 1280-d protein representation.

Why ESM-2 over one-hot:
  - Pre-trained on 250M+ protein sequences — encodes evolutionary co-variation
  - Each residue embedding reflects structural and functional context
  - Captures long-range dependencies (full attention), unlike CNN's local filters
  - Expected AUROC gain: +0.05-0.10 over one-hot CNN (from literature)

Output:
  data/embeddings/esm2_protein_embeddings.npz
    protein_ids : (N_proteins,)  — protein_name strings
    embeddings  : (N_proteins, 1280) — mean-pooled ESM-2 embeddings

Usage (from protein_rna_ml/):
    python scripts/07_extract_esm2_embeddings.py
    python scripts/07_extract_esm2_embeddings.py --model esm2_t12_35M_UR50D  # faster/smaller
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/generalized")
    parser.add_argument("--out_dir",    default="data/embeddings")
    parser.add_argument("--model",      default="esm2_t33_650M_UR50D",
                        help="ESM-2 variant. Options:\n"
                             "  esm2_t6_8M_UR50D     (8M,  fast, ~0.1 GB)\n"
                             "  esm2_t12_35M_UR50D   (35M, good tradeoff)\n"
                             "  esm2_t30_150M_UR50D  (150M)\n"
                             "  esm2_t33_650M_UR50D  (650M, best, default)\n"
                             "  esm2_t36_3B_UR50D    (3B,  needs GPU)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Proteins per batch. Reduce if OOM.")
    parser.add_argument("--max_len",    type=int, default=1022,
                        help="Truncate protein sequences to this length (ESM-2 limit=1022)")
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "esm2_protein_embeddings.npz")
    if os.path.exists(out_path):
        print(f"Embeddings already exist: {out_path}")
        print("Delete the file to recompute. Exiting.")
        return

    # ── Device ───────────────────────────────────────────────────────────────
    if not args.no_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # ── Load unique proteins ──────────────────────────────────────────────────
    print(f"\nLoading proteins from {args.data_dir}/train.tsv ...")
    all_dfs = []
    for split in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"{split}.tsv")
        df = pd.read_csv(path, sep="\t", usecols=["protein_name", "protein_sequence"])
        all_dfs.append(df)
    combined = pd.concat(all_dfs).drop_duplicates("protein_name")
    protein_ids = combined["protein_name"].tolist()
    protein_seqs = combined["protein_sequence"].tolist()
    print(f"  Unique proteins: {len(protein_ids)}")
    print(f"  Sequence lengths: min={min(len(s) for s in protein_seqs)}  "
          f"max={max(len(s) for s in protein_seqs)}  "
          f"median={int(np.median([len(s) for s in protein_seqs]))}")

    # ── Load ESM-2 ────────────────────────────────────────────────────────────
    print(f"\nLoading ESM-2 ({args.model}) from HuggingFace...")
    print("  (First run: downloads ~2.5 GB — may take a few minutes)")
    try:
        from transformers import EsmModel, EsmTokenizer
    except ImportError:
        print("\n❌  transformers not installed.")
        print("   Run: pip install transformers sentencepiece")
        sys.exit(1)

    model_name = f"facebook/{args.model}"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    esm = EsmModel.from_pretrained(model_name).to(device)
    esm.eval()
    n_params = sum(p.numel() for p in esm.parameters()) / 1e6
    print(f"  Model loaded. Parameters: {n_params:.0f}M")

    # ── Extract embeddings ────────────────────────────────────────────────────
    print(f"\nExtracting embeddings (batch_size={args.batch_size})...")
    all_embeddings = []
    n_batches = (len(protein_ids) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in range(0, len(protein_ids), args.batch_size):
            batch_seqs = [s[:args.max_len] for s in protein_seqs[i:i + args.batch_size]]
            batch_names = protein_ids[i:i + args.batch_size]
            t0 = time.time()

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len + 2,  # +2 for BOS/EOS tokens
            ).to(device)

            outputs = esm(**inputs)
            # outputs.last_hidden_state: (batch, seq_len, 1280)
            # Mean-pool over actual residues (exclude padding and BOS/EOS)
            hidden = outputs.last_hidden_state          # (B, L, 1280)
            mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            # Exclude BOS (pos 0) and EOS from pooling
            # Simple mean over all non-pad positions (BOS/EOS contribute <1% for long seqs)
            emb = (hidden * mask).sum(1) / mask.sum(1)  # (B, 1280)
            all_embeddings.append(emb.cpu().float().numpy())

            batch_i = i // args.batch_size + 1
            elapsed = time.time() - t0
            print(f"  Batch {batch_i}/{n_batches}: "
                  f"{[n[:15] for n in batch_names]}  {elapsed:.1f}s")

    embeddings = np.vstack(all_embeddings)   # (N_proteins, 1280)
    print(f"\nEmbedding matrix: {embeddings.shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        protein_ids=np.array(protein_ids),
        embeddings=embeddings.astype(np.float16),  # float16 saves ~half space
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n✅ Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"   Shape: {embeddings.shape}  dtype: float16")
    print(f"\n   Next: python scripts/08_train_generalized_v3.py")


if __name__ == "__main__":
    main()
