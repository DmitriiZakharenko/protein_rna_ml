"""
Phase 2 Model V2: Dual-branch CNN on one-hot encoded sequences.

Architecture:
    RNA sequence  (L_rna  × 4)  →  RNA  CNN branch  →  rna_emb  (256-d)
    Prot sequence (L_prot × 20) →  Prot CNN branch  →  prot_emb (256-d)
                                   concat (512-d)
                                   MLP head → binding score

Why CNN over k-mer MLP:
    - Learns position-sensitive motif filters (like MEME but end-to-end)
    - Captures local sequence context (flanking nucleotides around a motif)
    - Separate branches for RNA and protein → modality-specific inductive bias
    - First convolutional layer ≈ learned position weight matrix (PWM)

Reference:
    Alipanahi et al. (2015) DeepBind, Nature Biotechnology
    Zeng et al. (2016) Convolutional neural network architectures for predicting
    DNA/RNA sequence motifs, Briefings in Bioinformatics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBranch(nn.Module):
    """
    1D convolutional encoder for a single sequence modality.

    Input : (batch, length, alphabet_size)  [N × L × C]
    Output: (batch, out_channels)  — global max-pooled representation

    Three convolutional layers with increasing filter counts,
    each followed by BatchNorm + GELU. Global max pooling collapses
    the length dimension → fixed-size embedding regardless of sequence length.
    """

    def __init__(
        self,
        in_channels: int,       # 4 for RNA, 20 for protein
        filters: list[int] = [128, 256, 256],
        kernel_sizes: list[int] = [7, 5, 3],
        dropout: float = 0.2,
    ):
        super().__init__()
        assert len(filters) == len(kernel_sizes)

        convs = []
        ch = in_channels
        for f, k in zip(filters, kernel_sizes):
            convs += [
                nn.Conv1d(ch, f, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(f),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            ch = f

        self.convs   = nn.Sequential(*convs)
        self.out_dim = filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, length, channels) → transpose to (batch, channels, length) for Conv1d
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = x.max(dim=-1).values   # global max pooling over length
        return x                   # (batch, out_channels)


class RNABindingCNN(nn.Module):
    """
    Dual-branch CNN for protein–RNA binding prediction.

    Inputs:
        rna_onehot  : (batch, L_rna,  4)  — one-hot RNA
        prot_onehot : (batch, L_prot, 20) — one-hot protein

    Output:
        logits      : (batch,) — binding logit (sigmoid → probability)
    """

    def __init__(
        self,
        rna_filters:  list[int] = [128, 256, 256],
        prot_filters: list[int] = [128, 256, 256],
        rna_kernels:  list[int] = [7, 5, 3],
        prot_kernels: list[int] = [11, 7, 5],  # larger kernels for protein (longer motifs)
        head_dims:    list[int] = [256, 64],
        dropout: float = 0.3,
    ):
        super().__init__()

        self.rna_branch  = ConvBranch(4,  rna_filters,  rna_kernels,  dropout)
        self.prot_branch = ConvBranch(20, prot_filters, prot_kernels, dropout)

        in_dim = rna_filters[-1] + prot_filters[-1]   # 512 by default
        head = []
        for h in head_dims:
            head += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        head.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head)

    def forward(
        self,
        rna_onehot: torch.Tensor,
        prot_onehot: torch.Tensor,
    ) -> torch.Tensor:
        rna_emb  = self.rna_branch(rna_onehot)    # (batch, 256)
        prot_emb = self.prot_branch(prot_onehot)  # (batch, 256)
        combined = torch.cat([rna_emb, prot_emb], dim=-1)  # (batch, 512)
        return self.head(combined).squeeze(-1)              # (batch,)

    def predict_proba(self, rna_onehot, prot_onehot):
        return torch.sigmoid(self.forward(rna_onehot, prot_onehot))
