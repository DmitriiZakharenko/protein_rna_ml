"""
Sequence encoding utilities for protein-RNA binding prediction.

Supported encodings:
  - k-mer frequency vectors (fast, for baseline models)
  - one-hot encoding (for CNN/Transformer models)
"""

from itertools import product
from typing import Optional

import numpy as np


RNA_ALPHABET  = "AUGC"
AA_ALPHABET   = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids

RNA_TO_IDX  = {c: i for i, c in enumerate(RNA_ALPHABET)}
AA_TO_IDX   = {c: i for i, c in enumerate(AA_ALPHABET)}


# ─────────────────────────────────────────────────────────────────────────────
# k-mer encoding
# ─────────────────────────────────────────────────────────────────────────────

def build_kmer_index(alphabet: str, k: int) -> dict[str, int]:
    """Return {kmer: index} for all k-mers over alphabet."""
    return {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}


def kmer_freq_vector(seq: str, kmer_index: dict, k: int, normalize: bool = True) -> np.ndarray:
    """Count k-mers in seq and return normalized frequency vector."""
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k]
        if kmer in kmer_index:
            vec[kmer_index[kmer]] += 1
            n += 1
    if normalize and n > 0:
        vec /= n
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# One-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

def one_hot_rna(seq: str, max_len: int = 50) -> np.ndarray:
    """
    One-hot encode an RNA sequence.
    Returns: (max_len, 4) array. Positions beyond seq length are zero-padded.
    Unknown chars → all zeros.
    """
    arr = np.zeros((max_len, 4), dtype=np.float32)
    for i, c in enumerate(seq[:max_len]):
        if c in RNA_TO_IDX:
            arr[i, RNA_TO_IDX[c]] = 1.0
    return arr


def one_hot_protein(seq: str, max_len: int = 500) -> np.ndarray:
    """
    One-hot encode a protein sequence.
    Returns: (max_len, 20) array. Pads/truncates to max_len.
    """
    arr = np.zeros((max_len, 20), dtype=np.float32)
    for i, c in enumerate(seq[:max_len]):
        if c in AA_TO_IDX:
            arr[i, AA_TO_IDX[c]] = 1.0
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Sequence validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_rna(seq: str) -> bool:
    """Return True if seq contains only A/U/G/C."""
    return all(c in RNA_ALPHABET for c in seq.upper())


def validate_protein(seq: str) -> bool:
    """Return True if seq contains only standard amino acids."""
    return all(c in AA_ALPHABET for c in seq.upper())


def clean_sequence(seq: str, alphabet: str) -> str:
    """Remove characters not in alphabet, convert to uppercase."""
    seq = seq.upper()
    return "".join(c for c in seq if c in alphabet)
