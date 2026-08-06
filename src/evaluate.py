"""Evaluation harness. Takes scores, not models — so trivial baselines and CNNs
go through exactly the same code path."""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred  = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prevalence = y_true.mean()

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:                       # only one class present
        auc = np.nan

    return {
        'n': len(y_true),
        'prevalence': round(prevalence, 3),
        'majority_floor': round(max(prevalence, 1 - prevalence), 3),
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'specificity': round(tn / (tn + fp), 4) if (tn + fp) else np.nan,
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'roc_auc': round(auc, 4) if auc == auc else np.nan,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def evaluate_all(y_true, y_score, src, model_name, threshold=0.5):
    """Metrics on every evaluation tier. `src` is the per-image source column."""
    y_true, y_score, src = np.asarray(y_true), np.asarray(y_score), np.asarray(src)

    tiers = {
        'full test': np.ones(len(y_true), dtype=bool),
        'source-matched (flickr)': src == 'flickr',
        'unsplash only': src == 'unsplash',
    }

    rows = []
    for name, mask in tiers.items():
        if mask.sum() == 0:
            continue
        rows.append({'model': model_name, 'slice': name,
                     **compute_metrics(y_true[mask], y_score[mask], threshold)})
    return pd.DataFrame(rows)
