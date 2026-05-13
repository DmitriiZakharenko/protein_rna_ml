"""
Phase 3B Model V4: Dual-branch CNN + Bilinear Interaction Layer.

Architecture upgrade over V2:
  V2: RNA_CNN(256) + Prot_CNN(256) → concat(512) → MLP → logit
  V4: RNA_CNN(256) + Prot_CNN(256) → BilinearInteraction(256)
                                    + residual concat(256+256+256=768)
                                    → MLP → logit

Why bilinear interaction beats simple concatenation:
  Concatenation lets the MLP head implicitly learn pairwise features,
  but only through additive combinations of independent RNA and protein
  features. A bilinear layer explicitly models multiplicative interactions:

      interaction[k] = Σ_ij  W[k,i,j] * rna[i] * prot[j]

  In practice: hadamard product in projected space (low-rank bilinear pooling):

      interaction = Linear(rna) ⊙ Linear(prot)    (⊙ = element-wise product)

  This is equivalent to a rank-1 Tucker decomposition of the full bilinear
  tensor, capturing the most important pairwise motif co-occurrences without
  the O(d²) parameter cost of the full bilinear form.

  Reference:
    Kim et al. (2017) "Hadamard Product for Low-rank Bilinear Pooling"
    ICLR 2017 — originally for VQA, directly applicable here.

Optional: dataset_source embedding
  An additional learned embedding for the data source
  (selex_rbns, rnacompete, eclip) is concatenated before the MLP head.
  This lets the model learn source-specific calibration without leaking
  source identity as a shortcut feature.

Model variants (controlled by --interaction flag in training script):
  concat     — V2 baseline (no interaction layer, backward compatible)
  bilinear   — hadamard bilinear pooling (recommended, V4)
  concat_bi  — bilinear + residual concat (full V4)
"""

import torch
import torch.nn as nn

from src.models.cnn_model import ConvBranch


# ── Interaction modules ───────────────────────────────────────────────────────

class BilinearInteraction(nn.Module):
    """
    Hadamard bilinear interaction between two d-dimensional vectors.

    interaction = LayerNorm(Linear(a) ⊙ Linear(b))

    Parameters
    ----------
    in_dim  : dimension of both input vectors (must be equal)
    out_dim : output dimension (default: same as in_dim)
    dropout : dropout on the interaction vector

    Input:  a (batch, in_dim), b (batch, in_dim)
    Output: (batch, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int | None = None, dropout: float = 0.3):
        super().__init__()
        out_dim = out_dim or in_dim
        self.proj_a  = nn.Linear(in_dim, out_dim, bias=True)
        self.proj_b  = nn.Linear(in_dim, out_dim, bias=True)
        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.proj_a(a) * self.proj_b(b)))


class DatasetSourceEmbedding(nn.Module):
    """
    Learned embedding for dataset source tag.
    Maps a source index (int) to a dense vector concatenated to the MLP input.

    Sources are indexed by DATASET_SOURCES dict.
    Unknown sources map to index 0 (a learnable 'unknown' embedding).
    """

    DATASET_SOURCES = {
        "unknown":      0,
        "selex_rbns":   1,
        "htr_selex":    1,
        "rbns":         1,
        "rnacompete":   2,
        "rnacompete_rbpzoo":   2,
        "rnacompete_eukarya":  2,
        "rnacompete_ucrbp":    2,
        "eclip":        3,
        "iclip":        4,
        "par_clip":     5,
    }
    N_SOURCES = 6

    def __init__(self, emb_dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(self.N_SOURCES, emb_dim, padding_idx=0)
        self.emb_dim = emb_dim

    def forward(self, source_tags: list[str] | None, batch_size: int,
                device: torch.device) -> torch.Tensor:
        """Returns (batch, emb_dim) tensor. All-zeros if source_tags is None."""
        if source_tags is None:
            return torch.zeros(batch_size, self.emb_dim, device=device)
        indices = torch.tensor(
            [self.DATASET_SOURCES.get(str(t).lower(), 0) for t in source_tags],
            dtype=torch.long, device=device)
        return self.emb(indices)

    @classmethod
    def source_to_idx(cls, source: str) -> int:
        return cls.DATASET_SOURCES.get(str(source).lower(), 0)


# ── Full V4 model ─────────────────────────────────────────────────────────────

class RNABindingV4(nn.Module):
    """
    V4: Dual-branch CNN + Bilinear Interaction + optional dataset source embedding.

    Inputs:
        rna_onehot   : (batch, L_rna,  4)
        prot_onehot  : (batch, L_prot, 20)
        source_tags  : list[str] of length batch (optional)

    Output:
        logits : (batch,) — binding logit

    Interaction modes
    -----------------
    'concat'    : V2 baseline — concat(rna, prot) → MLP
    'bilinear'  : bilinear(rna, prot) → MLP  (no concat residual)
    'concat_bi' : concat(rna, prot, bilinear(rna,prot)) → MLP  [DEFAULT]
    """

    INTERACTION_MODES = ("concat", "bilinear", "concat_bi")

    def __init__(
        self,
        rna_filters:   list[int] = [128, 256, 256],
        prot_filters:  list[int] = [128, 256, 256],
        rna_kernels:   list[int] = [7, 5, 3],
        prot_kernels:  list[int] = [11, 7, 5],
        interaction:   str       = "concat_bi",
        inter_dim:     int       = 256,
        head_dims:     list[int] = [512, 128, 32],
        dropout:       float     = 0.3,
        use_source_emb: bool     = False,
        source_emb_dim: int      = 16,
    ):
        super().__init__()
        assert interaction in self.INTERACTION_MODES, \
            f"interaction must be one of {self.INTERACTION_MODES}"

        self.interaction = interaction
        self.rna_dim     = rna_filters[-1]
        self.prot_dim    = prot_filters[-1]

        self.rna_branch  = ConvBranch(4,  rna_filters,  rna_kernels,  dropout)
        self.prot_branch = ConvBranch(20, prot_filters, prot_kernels, dropout)

        # Bilinear interaction layer
        if interaction in ("bilinear", "concat_bi"):
            assert rna_filters[-1] == prot_filters[-1], \
                "BilinearInteraction requires rna_dim == prot_dim"
            self.inter_layer = BilinearInteraction(rna_filters[-1], inter_dim, dropout)
        else:
            self.inter_layer = None

        # Dataset source embedding (optional)
        self.use_source_emb = use_source_emb
        if use_source_emb:
            self.src_emb = DatasetSourceEmbedding(source_emb_dim)
        else:
            self.src_emb = None
            source_emb_dim = 0

        # MLP head input dimension
        if interaction == "concat":
            head_in = rna_filters[-1] + prot_filters[-1]
        elif interaction == "bilinear":
            head_in = inter_dim
        else:  # concat_bi
            head_in = rna_filters[-1] + prot_filters[-1] + inter_dim
        head_in += source_emb_dim

        head = []
        in_dim = head_in
        for h in head_dims:
            head += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        head.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  V4 RNABindingV4 | interaction={interaction} | "
              f"head_in={head_in} | params={n_params:,}")

    def forward(
        self,
        rna_onehot:  torch.Tensor,
        prot_onehot: torch.Tensor,
        source_tags: list[str] | None = None,
    ) -> torch.Tensor:
        rna_emb  = self.rna_branch(rna_onehot)    # (B, rna_dim)
        prot_emb = self.prot_branch(prot_onehot)  # (B, prot_dim)

        parts = []
        if self.interaction in ("concat", "concat_bi"):
            parts.extend([rna_emb, prot_emb])
        if self.inter_layer is not None:
            parts.append(self.inter_layer(rna_emb, prot_emb))

        combined = torch.cat(parts, dim=-1)        # (B, head_in - src_dim)

        if self.use_source_emb and source_tags is not None:
            src_vec = self.src_emb(source_tags, len(rna_emb), rna_emb.device)
            combined = torch.cat([combined, src_vec], dim=-1)
        elif self.use_source_emb:
            combined = torch.cat(
                [combined, torch.zeros(len(rna_emb), self.src_emb.emb_dim,
                                       device=rna_emb.device)], dim=-1)

        return self.head(combined).squeeze(-1)     # (B,)

    def predict_proba(self, rna_onehot, prot_onehot, source_tags=None):
        return torch.sigmoid(self.forward(rna_onehot, prot_onehot, source_tags))
