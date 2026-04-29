"""
Evaluation metrics for protein-RNA binding prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, brier_score_loss, precision_score, recall_score
)
from typing import Optional


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict:
    """
    Compute a standard set of binary classification metrics.

    Args:
        y_true:    Binary ground-truth labels (0 or 1).
        y_prob:    Predicted probabilities for class 1.
        threshold: Decision threshold for class predictions.
        prefix:    Optional prefix added to all metric keys.

    Returns:
        Dictionary of metric_name → value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auroc":       float(roc_auc_score(y_true, y_prob)),
        "auprc":       float(average_precision_score(y_true, y_prob)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "n_samples":   int(len(y_true)),
        "n_positive":  int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }

    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

    return metrics


def compute_per_protein_metrics(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    protein_col: str = "protein_name",
    label_col: str   = "binding_label",
    min_examples: int = 10,
) -> pd.DataFrame:
    """
    Compute AUROC and AUPRC per protein.

    Args:
        df:           DataFrame with protein_col and label_col.
        y_prob:       Predicted probabilities (same order as df).
        protein_col:  Column with protein names.
        label_col:    Column with binary labels.
        min_examples: Minimum examples per protein to compute metrics.

    Returns:
        DataFrame with columns [protein_name, auroc, auprc, n_examples, n_positive, note]
    """
    df = df.copy()
    df["__y_prob"] = y_prob
    df["__y_true"] = df[label_col].values

    rows = []
    for protein, group in df.groupby(protein_col):
        n_pos = int(group["__y_true"].sum())
        n_neg = len(group) - n_pos
        n_total = len(group)

        if n_total < min_examples:
            rows.append({
                "protein_name": protein,
                "auroc":       None,
                "auprc":       None,
                "n_examples":  n_total,
                "n_positive":  n_pos,
                "note":        "too_few_examples",
            })
            continue

        if group["__y_true"].nunique() < 2:
            rows.append({
                "protein_name": protein,
                "auroc":       None,
                "auprc":       None,
                "n_examples":  n_total,
                "n_positive":  n_pos,
                "note":        "single_class",
            })
            continue

        rows.append({
            "protein_name": protein,
            "auroc":       float(roc_auc_score(group["__y_true"], group["__y_prob"])),
            "auprc":       float(average_precision_score(group["__y_true"], group["__y_prob"])),
            "n_examples":  n_total,
            "n_positive":  n_pos,
            "note":        "ok",
        })

    return pd.DataFrame(rows).sort_values("auroc", ascending=False, na_position="last")


def validation_decision(
    overall_auroc: float,
    per_protein_median_auroc: float,
    pass_threshold: float = 0.70,
    warn_threshold: float = 0.60,
    per_protein_pass: float = 0.65,
) -> dict:
    """
    Return a validation decision dict for a dataset.
    """
    if overall_auroc >= pass_threshold and per_protein_median_auroc >= per_protein_pass:
        status  = "PASS"
        message = "Dataset validated — genuine signal confirmed. Ready for Phase 2."
    elif overall_auroc >= warn_threshold:
        status  = "WARN"
        message = "Marginal performance — investigate flagged proteins before Phase 2."
    else:
        status  = "FAIL"
        message = "Validation failed — do NOT include in Phase 2 training."

    return {
        "status":                   status,
        "message":                  message,
        "overall_auroc":            overall_auroc,
        "per_protein_median_auroc": per_protein_median_auroc,
    }
