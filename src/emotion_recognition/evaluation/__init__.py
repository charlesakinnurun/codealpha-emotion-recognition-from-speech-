"""Evaluation metrics (NumPy-only; safe on hosts where sklearn is blocked)."""

from .metrics import accuracy, macro_f1, per_class_metrics

__all__ = ["accuracy", "macro_f1", "per_class_metrics"]
