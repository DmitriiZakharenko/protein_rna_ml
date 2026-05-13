"""
Script 07b: Extract ESM-2 per-residue embeddings for V3c.

Unlike script 07 (which mean-pools → 1280-d per protein), this script
saves the full per-residue representation: (L × 1280) for each protein,
padded to prot_max=300 positions.

V3c uses these as input to a Conv1D branch that learns which residue
positions are relevant to RNA binding — avoiding the mean-pool dilution
that caused V3 and V3b to underperform V2.

Output:
  data/embeddings/esm2_residue_embeddings.npz
    protein_ids    : (N,)               — protein name strings
    embeddings     : (N, prot_max, 1280) — per-residue, padded, float16
    lengths        : (N,)               — actual sequence lengths (for masking)

File size: 168 × 300 × 1280 × 2 bytes ≈ 129 MB

Usage (from protein_rna_ml/):
    python scripts/07b_extract_esm2_residues.py
    python scripts/07b_extract_esm2_residues.py --model esm2_t12_35M_UR50D  # faster
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
                             "  esm2_t12_35M_UR50D   (35M, recommended for speed)\n"
                             "  esm2_t30_150M_UR50D  (150M)\n"
                             "  esm2_t33_650M_UR50D  (650M, best quality, default)\n"
                             "  esm2_t36_3B_UR50D    (3B,  needs GPU)")
    parser.add_argument("--prot_max",   type=int, default=300,
                        help="Pad/truncate protein sequences to this length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Proteins per batch. Reduce if OOM (residue embeddings are large).")
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "esm2_residue_embeddings.npz")
    if os.path.exists(out_path):
        print(f"Residue embeddings already exist: {out_path}")
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
    print(f"\nLoading proteins from {args.data_dir}/ ...")
    all_dfs = []
    for split in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"{split}.tsv")
        df = pd.read_csv(path, sep="\t", usecols=["protein_name", "protein_sequence"])
        all_dfs.append(df)
    combined = pd.concat(all_dfs).drop_duplicates("protein_name")
    protein_ids   = combined["protein_name"].tolist()
    protein_seqs  = combined["protein_sequence"].tolist()
    n_proteins    = len(protein_ids)
    actual_lengths = [min(len(s), args.prot_max) for s in protein_seqs]

    print(f"  Unique proteins: {n_proteins}")
    print(f"  Sequence lengths: min={min(len(s) for s in protein_seqs)}  "
          f"max={max(len(s) for s in protein_seqs)}  "
          f"median={int(np.median([len(s) for s in protein_seqs]))}")
    print(f"  Padding all to prot_max={args.prot_max}")

    est_mb = n_proteins * args.prot_max * 1280 * 2 / 1e6
    print(f"  Estimated output size: {est_mb:.0f} MB (fp16)")

    # ── Load ESM-2 ────────────────────────────────────────────────────────────
    print(f"\nLoading ESM-2 ({args.model}) ...")
    print("  (First run: downloads model from HuggingFace — may take a few minutes)")
    try:
        from transformers import EsmModel, EsmTokenizer
    except ImportError:
        print("\n❌  transformers not installed. Run: pip install transformers")
        sys.exit(1)

    model_name = f"facebook/{args.model}"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    esm = EsmModel.from_pretrained(model_name).to(device)
    esm.eval()
    n_params = sum(p.numel() for p in esm.parameters()) / 1e6
    print(f"  Model loaded. Parameters: {n_params:.0f}M")

    # ── Pre-allocate output array ─────────────────────────────────────────────
    # Use float16 to keep memory manageable
    all_embeddings = np.zeros((n_proteins, args.prot_max, 1280), dtype=np.float16)
    print(f"\n  Pre-allocated output: {all_embeddings.nbytes / 1e6:.0f} MB in RAM")

    # ── Extract per-residue embeddings ────────────────────────────────────────
    print(f"\nExtracting per-residue embeddings (batch_size={args.batch_size})...")
    n_batches = (n_proteins + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    with torch.no_grad():
        for i in range(0, n_proteins, args.batch_size):
            batch_idx   = list(range(i, min(i + args.batch_size, n_proteins)))
            batch_seqs  = [protein_seqs[j][:args.prot_max] for j in batch_idx]
            batch_names = [protein_ids[j][:15] for j in batch_idx]
            t0 = time.time()

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.prot_max + 2,   # +2 for BOS/EOS tokens
            ).to(device)

            outputs = esm(**inputs)
            # last_hidden_state: (B, seq_len_padded+2, 1280)
            # Position 0 = BOS token, positions 1..L = residues, L+1 = EOS
            hidden = outputs.last_hidden_state.cpu().float().numpy()  # (B, L+2, 1280)

            for local_j, global_j in enumerate(batch_idx):
                actual_len = actual_lengths[global_j]
                # Extract residue positions 1..actual_len (skip BOS at 0, skip EOS+padding)
                residues = hidden[local_j, 1:actual_len + 1, :]  # (actual_len, 1280)
                # Write into pre-allocated array (padded with zeros beyond actual_len)
                all_embeddings[global_j, :actual_len, :] = residues.astype(np.float16)

            elapsed = time.time() - t0
            batch_i = i // args.batch_size + 1
            total_elapsed = time.time() - t_start
            eta = total_elapsed / batch_i * (n_batches - batch_i)
            print(f"  Batch {batch_i}/{n_batches}: {batch_names}  "
                  f"{elapsed:.1f}s  ETA: {eta/60:.1f}min")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving {out_path} ...")
    np.savez_compressed(
        out_path,
        protein_ids=np.array(protein_ids),
        embeddings=all_embeddings,          # (N, prot_max, 1280) float16
        lengths=np.array(actual_lengths),   # (N,) actual sequence lengths
    )
    size_mb = os.path.getsize(out_path) / 1e6
    total_time = (time.time() - t_start) / 60
    print(f"\n✅  Saved: {out_path}")
    print(f"   Shape: {all_embeddings.shape}  dtype: float16")
    print(f"   File size: {size_mb:.1f} MB")
    print(f"   Total time: {total_time:.1f} min")
    print(f"\n   Next: python scripts/10_train_generalized_v3c.py")


if __name__ == "__main__":
    main()
