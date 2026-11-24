"""Evaluation metrics and plots for imbalanced binary classification.

Accuracy is not reported on purpose: with 0.17% fraud, predicting "normal"
for everything already gives 99.8% accuracy.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


def metrics_at_threshold(y_true, scores, threshold: float = 0.5) -> dict:
    """Precision, recall, F1 and confusion-matrix counts at one threshold,
    plus threshold-free PR-AUC (average precision) and ROC-AUC."""
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def best_f1_threshold(y_true, scores) -> float:
    """Threshold that gives the highest F1 (use on validation data, not test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # the last precision/recall point has no threshold
    return float(thresholds[np.argmax(f1[:-1])])


def threshold_for_recall(y_true, scores, target_recall: float = 0.80) -> float:
    """Highest threshold that still reaches the target recall (best precision at that recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds[0])
    return float(thresholds[ok[-1]])


def results_table(rows: dict[str, dict]) -> pd.DataFrame:
    """Turn {model name: metrics dict} into a rounded table."""
    table = pd.DataFrame(rows).T
    float_cols = ["threshold", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    table[float_cols] = table[float_cols].astype(float).round(4)
    count_cols = ["tp", "fp", "fn", "tn"]
    table[count_cols] = table[count_cols].astype(int)
    return table


def save_table(table: pd.DataFrame, name: str) -> None:
    """Save a results table as CSV in results/tables."""
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES / f"{name}.csv")


def plot_pr_curves(y_true, scores_by_model: dict, title: str, name: str | None = None):
    """Precision-recall curves for several models on the same axes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})")
    base = np.mean(y_true)
    ax.axhline(base, color="grey", linestyle="--", label=f"Random (AP = {base:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left")
    _save(fig, name)
    return fig


def plot_confusion(y_true, scores, threshold: float, title: str, name: str | None = None):
    """Confusion matrix at a given threshold."""
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Normal", "Fraud"], cmap="Blues",
        colorbar=False, ax=ax, values_format=",d",
    )
    ax.set_title(f"{title}\n(threshold = {threshold:.3f})")
    _save(fig, name)
    return fig


def _save(fig, name: str | None) -> None:
    if name:
        config.FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
