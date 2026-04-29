"""
Phase 2 Model V1: Multi-layer Perceptron on k-mer features.

Architecture:
    [RNA 4-mer (256) | Protein 3-mer (8000) | metadata (2)]
                            ↓
                    BatchNorm → Dropout
                     Linear(8258 → 512) → GELU
                     BatchNorm → Dropout
                     Linear(512 → 256) → GELU
                     BatchNorm → Dropout
                     Linear(256 → 128) → GELU
                     Dropout
                     Linear(128 → 1) → Sigmoid
                            ↓
                     binding_score ∈ [0, 1]
"""

import torch
import torch.nn as nn


class RNABindingMLP(nn.Module):
    """
    MLP classifier for protein–RNA binding prediction on k-mer features.

    Args:
        input_dim   : total feature dimension (default: 8258 = 256+8000+2)
        hidden_dims : list of hidden layer sizes
        dropout     : dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_dim: int = 8258,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # output logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) float tensor
        Returns:
            logits: (batch_size,) — pass through sigmoid for probabilities
        """
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0, 1]."""
        return torch.sigmoid(self.forward(x))


class RNABindingMLPWithAffinity(nn.Module):
    """
    Multi-task MLP: binary binding classification + affinity regression.
    The regression head is only activated when affinity labels are available
    (masked loss in trainer).

    Future extension — not used in Phase 2 V1.
    """

    def __init__(
        self,
        input_dim: int = 8258,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        self.shared = nn.Sequential(*layers)
        self.cls_head      = nn.Linear(in_dim, 1)   # binding classification
        self.affinity_head = nn.Linear(in_dim, 1)   # log(R_max) regression

    def forward(self, x):
        h = self.shared(x)
        return self.cls_head(h).squeeze(-1), self.affinity_head(h).squeeze(-1)
