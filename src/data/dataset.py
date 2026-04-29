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
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.rna_max  = rna_max_len
        self.prot_max = prot_max_len
        self.rna_col  = rna_col
        self.prot_col = prot_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def _one_hot_rna(self, seq: str) -> torch.Tensor:
        """Returns (rna_max_len, 4) float tensor."""
        arr = torch.zeros(self.rna_max, 4)
        for i, c in enumerate(seq[:self.rna_max]):
            if c in RNA_TO_IDX:
                arr[i, RNA_TO_IDX[c]] = 1.0
        return arr

    def _one_hot_prot(self, seq: str) -> torch.Tensor:
        """Returns (prot_max_len, 20) float tensor."""
        arr = torch.zeros(self.prot_max, 20)
        for i, c in enumerate(seq[:self.prot_max]):
            if c in AA_TO_IDX:
                arr[i, AA_TO_IDX[c]] = 1.0
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rna_oh  = self._one_hot_rna(str(row[self.rna_col]).upper())
        prot_oh = self._one_hot_prot(str(row[self.prot_col]).upper())
        label   = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return rna_oh, prot_oh, label
