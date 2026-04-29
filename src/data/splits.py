"""
Protein-aware dataset splitting.

Key principle: a protein must NOT appear in more than one split.
This ensures the model is evaluated on truly unseen proteins.
"""

import numpy as np
import pandas as pd
from typing import Optional


def protein_aware_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
    val_frac: float   = 0.11,
    seed: int         = 42,
    protein_col: str  = "protein_name",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset by protein so no protein is shared between splits.

    Args:
        df:           Full dataset DataFrame.
        train_frac:   Fraction of proteins assigned to train.
        val_frac:     Fraction of proteins assigned to val.
        seed:         Random seed for reproducibility.
        protein_col:  Column containing protein identifiers.

    Returns:
        (train_df, val_df, test_df, split_map_df)
        split_map_df has columns [protein_col, "split"]
    """
    rng      = np.random.default_rng(seed)
    proteins = np.array(df[protein_col].unique())
    n        = len(proteins)

    shuffled = rng.permutation(proteins)
    n_train  = int(np.round(n * train_frac))
    n_val    = int(np.round(n * val_frac))

    train_proteins = set(shuffled[:n_train])
    val_proteins   = set(shuffled[n_train: n_train + n_val])
    test_proteins  = set(shuffled[n_train + n_val:])

    # Sanity check
    assert not (train_proteins & val_proteins),  "Protein overlap: train ∩ val"
    assert not (train_proteins & test_proteins), "Protein overlap: train ∩ test"
    assert not (val_proteins   & test_proteins), "Protein overlap: val ∩ test"

    def assign_split(protein: str) -> str:
        if protein in train_proteins: return "train"
        if protein in val_proteins:   return "val"
        return "test"

    df = df.copy()
    df["split"] = df[protein_col].map(assign_split)

    train_df = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df   = df[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test_df  = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    split_map_df = pd.DataFrame([
        {protein_col: p, "split": assign_split(p)}
        for p in proteins
    ])

    return train_df, val_df, test_df, split_map_df


def stratified_protein_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
    val_frac: float   = 0.11,
    seed: int         = 42,
    protein_col: str  = "protein_name",
    stratify_col: Optional[str] = "source",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Protein-aware split that additionally tries to balance a stratification
    column (e.g., 'source' = in vitro / in vivo) across splits.
    Falls back to plain protein_aware_split if stratify_col is None.
    """
    if stratify_col is None or stratify_col not in df.columns:
        return protein_aware_split(df, train_frac, val_frac, seed, protein_col)

    # Get per-protein majority source
    protein_source = (
        df.groupby(protein_col)[stratify_col]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    train_proteins, val_proteins, test_proteins = set(), set(), set()

    for source, group in protein_source.groupby(stratify_col):
        proteins = group[protein_col].values
        n = len(proteins)
        shuffled = rng.permutation(proteins)
        n_train = max(1, int(np.round(n * train_frac)))
        n_val   = max(1, int(np.round(n * val_frac)))
        train_proteins.update(shuffled[:n_train])
        val_proteins.update(shuffled[n_train: n_train + n_val])
        test_proteins.update(shuffled[n_train + n_val:])

    # If a protein landed in multiple sets due to rounding, resolve by priority
    val_proteins   -= train_proteins
    test_proteins  -= train_proteins
    test_proteins  -= val_proteins

    def assign_split(protein: str) -> str:
        if protein in train_proteins: return "train"
        if protein in val_proteins:   return "val"
        return "test"

    df = df.copy()
    df["split"] = df[protein_col].map(assign_split)

    train_df = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df   = df[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test_df  = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    all_proteins = df[protein_col].unique()
    split_map_df = pd.DataFrame([
        {protein_col: p, "split": assign_split(p)}
        for p in all_proteins
    ])

    return train_df, val_df, test_df, split_map_df
