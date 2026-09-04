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


def rna_aware_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
    val_frac: float = 0.11,
    seed: int = 42,
    rna_col: str = "rna_sequence",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique RNA sequence so no RNA is shared between splits.

    Proteins may appear in multiple splits (with different RNAs). Use this to
    measure generalization to unseen RNA sequences (cf. ZHMolGraph hard split).
    """
    rng = np.random.default_rng(seed)
    rnas = np.array(df[rna_col].unique())
    shuffled = rng.permutation(rnas)
    n_train = int(np.round(len(rnas) * train_frac))
    n_val = int(np.round(len(rnas) * val_frac))

    train_rnas = set(shuffled[:n_train])
    val_rnas = set(shuffled[n_train: n_train + n_val])
    test_rnas = set(shuffled[n_train + n_val:])

    assert not (train_rnas & val_rnas)
    assert not (train_rnas & test_rnas)
    assert not (val_rnas & test_rnas)

    def assign_split(rna: str) -> str:
        if rna in train_rnas:
            return "train"
        if rna in val_rnas:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out[rna_col].map(assign_split)

    train_df = out[out["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df = out[out["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test_df = out[out["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    split_map_df = pd.DataFrame([
        {rna_col: r, "split": assign_split(r)}
        for r in rnas
    ])
    return train_df, val_df, test_df, split_map_df


def pair_aware_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
    val_frac: float = 0.11,
    seed: int = 42,
    protein_col: str = "protein_name",
    rna_col: str = "rna_sequence",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique (protein, RNA) pairs."""
    rng = np.random.default_rng(seed)
    pairs = df[[protein_col, rna_col]].drop_duplicates()
    pair_keys = list(map(tuple, pairs.values))
    shuffled = rng.permutation(pair_keys)
    n_train = int(np.round(len(shuffled) * train_frac))
    n_val = int(np.round(len(shuffled) * val_frac))

    train_pairs = set(map(tuple, shuffled[:n_train]))
    val_pairs = set(map(tuple, shuffled[n_train: n_train + n_val]))
    test_pairs = set(map(tuple, shuffled[n_train + n_val:]))

    def assign_split(row) -> str:
        key = (row[protein_col], row[rna_col])
        if key in train_pairs:
            return "train"
        if key in val_pairs:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out.apply(assign_split, axis=1)

    train_df = out[out["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df = out[out["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test_df = out[out["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    rows = []
    for key in pair_keys:
        if key in train_pairs:
            split = "train"
        elif key in val_pairs:
            split = "val"
        else:
            split = "test"
        rows.append({protein_col: key[0], rna_col: key[1], "split": split})
    split_map_df = pd.DataFrame(rows)
    return train_df, val_df, test_df, split_map_df


def drop_eval_rnas_seen_in_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rna_col: str = "rna_sequence",
    *,
    drop_val_rnas_from_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove val/test rows whose RNA appeared in train (protein+RNA strict eval)."""
    train_rnas = set(train_df[rna_col])
    val_out = val_df[~val_df[rna_col].isin(train_rnas)].reset_index(drop=True)
    forbidden = train_rnas
    if drop_val_rnas_from_test:
        forbidden = forbidden | set(val_df[rna_col])
    test_out = test_df[~test_df[rna_col].isin(forbidden)].reset_index(drop=True)
    return val_out, test_out


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
