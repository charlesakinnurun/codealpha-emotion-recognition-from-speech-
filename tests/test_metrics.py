"""Tests for the NumPy-only evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from emotion_recognition.evaluation.metrics import (
    accuracy,
    macro_f1,
    per_class_metrics,
)


def test_accuracy_perfect_and_partial() -> None:
    y_true = ["a", "b", "c", "c"]
    assert accuracy(y_true, y_true) == 1.0
    assert accuracy(y_true, ["a", "b", "c", "b"]) == 0.75


def test_macro_f1_hand_computed() -> None:
    y_true = np.asarray(["a", "a", "b", "b"])
    y_pred = np.asarray(["a", "b", "b", "b"])
    report = per_class_metrics(y_true, y_pred)
    # class a: tp=1, fp=0, fn=1 -> p=1.0, r=0.5, f1=2/3
    # class b: tp=2, fp=1, fn=0 -> p=2/3, r=1.0, f1=0.8
    assert round(report["a"]["f1"], 6) == round(2.0 / 3.0, 6)
    assert round(report["b"]["f1"], 6) == 0.8
    expected_macro = (report["a"]["f1"] + report["b"]["f1"]) / 2.0
    assert macro_f1(y_true, y_pred) == expected_macro
    assert accuracy(y_true, y_pred) == 0.75


def test_all_wrong_single_class_no_division_errors() -> None:
    y_true = ["a"]
    y_pred = ["b"]
    report = per_class_metrics(y_true, y_pred)
    assert report["a"]["f1"] == 0.0
    assert report["b"]["f1"] == 0.0
    assert macro_f1(y_true, y_pred) == 0.0
    assert accuracy(y_true, y_pred) == 0.0


def test_class_present_in_pred_but_not_true_is_zero() -> None:
    y_true = ["a", "a"]
    y_pred = ["a", "z"]  # 'z' never in ground truth -> contributes 0, no crash
    report = per_class_metrics(y_true, y_pred)
    assert report["z"]["f1"] == 0.0
    assert macro_f1(y_true, y_pred) > 0.0  # a perfect, z zero -> mean > 0


def test_empty_input_returns_nan() -> None:
    assert np.isnan(accuracy([], []))
    assert np.isnan(macro_f1([], []))


def test_deterministic_and_numeric_dtypes() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    y_pred = np.clip(y_true + rng.integers(-1, 2, size=200), 0, 2)
    assert macro_f1(y_true, y_pred) == macro_f1(y_true, y_pred)
    assert 0.0 <= macro_f1(y_true, y_pred) <= 1.0


def test_macro_f1_matches_sklearn() -> None:
    try:
        from sklearn.metrics import f1_score
    except ImportError:  # blocked host (App Control) -> skip comparison
        pytest.skip("scikit-learn unavailable on this host")

    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 4, size=150)
    y_pred = rng.integers(0, 4, size=150)
    assert abs(macro_f1(y_true, y_pred) - f1_score(y_true, y_pred, average="macro")) < 1e-9
