"""
PyTorch Dataset classes for Phase 2.

Two dataset types:
  KmerDataset   — loads pre-encoded .npz k-mer features (fast, for MLP)
  SeqDataset    — loads raw sequences and one-hot encodes on the fly (for CNN)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RNA_ALPHA  = "AUGC"
AA_ALPHA   = "ACDEFGHIKLMNPQRSTVWY"
RNA_TO_IDX  = {c: i for i, c in enumerate(RNA_ALPHA)}
AA_TO_IDX   = {c: i for i, c in enumerate(AA_ALPHA)}


def _build_char_lut(char_to_idx: dict[str, int], size: int = 128) -> np.ndarray:
    """Map ASCII code → channel index, or -1 for unknown / padding."""
    lut = np.full(size, -1, dtype=np.int16)
    for ch, idx in char_to_idx.items():
        lut[ord(ch)] = idx
    return lut


_RNA_LUT = _build_char_lut(RNA_TO_IDX)
_AA_LUT  = _build_char_lut(AA_TO_IDX)


def one_hot_encode(seq: str, max_len: int, lut: np.ndarray, n_channels: int) -> torch.Tensor:
    """
    Vectorized one-hot encoder (numpy LUT). Output matches the legacy per-char loop.

    Returns (max_len, n_channels) float32 tensor.
    """
    arr = np.zeros((max_len, n_channels), dtype=np.float32)
    if not seq or max_len <= 0:
        return torch.from_numpy(arr)

    raw = str(seq).upper()[:max_len]
    if not raw:
        return torch.from_numpy(arr)

    codes = np.frombuffer(raw.encode("ascii", errors="ignore"), dtype=np.uint8)
    n = min(len(codes), max_len)
    if n == 0:
        return torch.from_numpy(arr)

    codes = codes[:n]
    ch_idx = lut[codes]
    valid = ch_idx >= 0
    if valid.any():
        pos = np.flatnonzero(valid)
        arr[pos, ch_idx[valid]] = 1.0

    return torch.from_numpy(arr)


# ── k-mer Dataset (for MLP) ───────────────────────────────────────────────────

class KmerDataset(Dataset):
    """
    Loads pre-encoded k-mer feature arrays from .npz files.
    Used with RNABindingMLP (Phase 2 V1).

    Args:
        npz_path : path to .npz file with keys 'X' (features) and 'y' (labels)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Sequence Dataset (for CNN) ────────────────────────────────────────────────

class SeqDataset(Dataset):
    """
    Loads raw sequences from a TSV and one-hot encodes them on the fly.
    Used with RNABindingCNN (Phase 2 V2).

    Args:
        tsv_path      : path to TSV with columns
                        [protein_name, protein_sequence, rna_sequence, binding_label, ...]
        rna_max_len   : pad/truncate RNA to this length (default 60)
        prot_max_len  : pad/truncate protein to this length (default 800)
        protein_col   : column name for protein identifier
        rna_col       : column name for RNA sequence
        prot_col      : column name for protein sequence
        label_col     : column name for binding label
    """

    def __init__(
        self,
        tsv_path: str,
        rna_max_len:  int = 60,
        prot_max_len: int = 800,
        protein_col:  str = "protein_name",
        rna_col:      str = "rna_sequence",
        prot_col:     str = "protein_sequence",
        label_col:    str = "binding_label",
    ):
        self.df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        self.rna_max  = rna_max_len
        self.prot_max = prot_max_len
        self.rna_col  = rna_col
        self.prot_col = prot_col
        self.label_col = label_col

        # Pre-extract columns — avoids pandas iloc in __getitem__ (hot path).
        self._rna_seqs  = self.df[rna_col].astype(str).str.upper().to_numpy()
        self._prot_seqs = self.df[prot_col].astype(str).str.upper().to_numpy()
        self._labels    = self.df[label_col].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self._labels)

    def _one_hot_rna(self, seq: str) -> torch.Tensor:
        """Returns (rna_max_len, 4) float tensor."""
        return one_hot_encode(seq, self.rna_max, _RNA_LUT, 4)

    def _one_hot_prot(self, seq: str) -> torch.Tensor:
        """Returns (prot_max_len, 20) float tensor."""
        return one_hot_encode(seq, self.prot_max, _AA_LUT, 20)

    def __getitem__(self, idx):
        rna_oh  = self._one_hot_rna(self._rna_seqs[idx])
        prot_oh = self._one_hot_prot(self._prot_seqs[idx])
        label   = torch.tensor(self._labels[idx], dtype=torch.float32)
        return rna_oh, prot_oh, label
