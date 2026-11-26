"""Accuracy and calibration metrics for multi-class probability predictions."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


def brier(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error between probabilities and one-hot labels."""
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, classes, n_bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    confidence = proba.max(axis=1)
    correct = np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def reliability_table(y_true, proba, classes, n_bins: int = 10):
    """Per-bin mean confidence, accuracy and count (for reliability diagrams)."""
    confidence = proba.max(axis=1)
    correct = np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in itertools.pairwise(edges):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            rows.append({"bin_low": lo, "bin_high": hi, "confidence": confidence[in_bin].mean(),
                         "accuracy": correct[in_bin].mean(), "n": int(in_bin.sum())})
    return rows


def score(y_true, proba, classes) -> dict[str, float]:
    """All metrics for one set of predictions. Probabilities are clipped for log loss."""
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    predicted = np.asarray(classes)[proba.argmax(axis=1)]
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "macro_f1": f1_score(y_true, predicted, average="macro", labels=classes, zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=classes),
        "brier": brier(y_true, proba, classes),
        "ece": expected_calibration_error(y_true, proba, classes),
    }
