"""Scikit-learn-free classification metrics.

These are deliberately dependency-light (NumPy only) so the evaluation
harness runs identically everywhere -- including CI hosts that can't load
sklearn's compiled extensions. Models may use sklearn for training, but
reporting/selection never depends on it.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def accuracy(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def per_class_metrics(
    y_true: np.ndarray | list, y_pred: np.ndarray | list
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1/support, keyed by label.

    For a class absent from the ground truth the F1 is 0 by convention;
    never raises on empty denominators.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    report: dict[str, dict[str, float]] = {}
    for label in classes:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))

        precision = tp / (tp + fp + EPS) if (tp + fp) else 0.0
        recall = tp / (tp + fn + EPS) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall + EPS) if (precision + recall) else 0.0
        report[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return report


def macro_f1(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """Unweighted mean of per-class F1 (sklearn-compatible semantics).

    Returns ``nan`` on empty input; classes absent from the ground truth
    contribute 0 without raising.
    """
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return float("nan")
    report = per_class_metrics(y_true, y_pred)
    scores = [row["f1"] for row in report.values()]
    return float(np.mean(scores))


__all__ = ["accuracy", "macro_f1", "per_class_metrics"]
