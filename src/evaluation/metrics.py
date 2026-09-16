# src/evaluation/metrics.py
"""
Evaluation metrics focused on low false-positive rate
for the PhreshPhish classifier.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def compute_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average Precision (area under the Precision-Recall curve)."""
    return float(average_precision_score(y_true, y_prob))


def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
) -> Tuple[float, float, float]:
    """
    Return (precision, actual_recall, threshold) at the point
    closest to the desired recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Find index closest to target_recall
    idx = int(np.argmin(np.abs(recall - target_recall)))

    # thresholds array is one element shorter than precision/recall
    thr = float(thresholds[idx]) if idx < len(thresholds) else 1.0

    return float(precision[idx]), float(recall[idx]), thr


def find_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.92,
) -> float:
    """Convenience wrapper that returns only the threshold."""
    _, _, thr = precision_at_recall(y_true, y_prob, target_recall)
    return thr


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    name: str = "Model",
    target_recalls: Tuple[float, ...] = (0.90, 0.95),
) -> Dict[str, Any]:
    """
    Full evaluation report focused on low-FP metrics.

    Returns a dictionary with AP, ROC-AUC, Precision@Recall, etc.
    """
    y_pred = (y_prob >= threshold).astype(int)

    ap = compute_average_precision(y_true, y_prob)
    roc = float(roc_auc_score(y_true, y_prob))

    print(f"\n===== {name} =====")
    print(f"Average Precision : {ap:.4f}")
    print(f"ROC-AUC           : {roc:.4f}")
    print(f"\nClassification Report (threshold={threshold:.3f}):")
    print(classification_report(
        y_true, y_pred,
        target_names=["benign", "phish"],
        digits=4
    ))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    prec_at_rec = {}
    for r in target_recalls:
        p, actual_r, thr = precision_at_recall(y_true, y_prob, target_recall=r)
        print(f"Precision @ Recall={r:.2f}: {p:.4f}  (actual recall={actual_r:.3f}, thr≈{thr:.3f})")
        prec_at_rec[f"precision_at_{int(r*100)}_recall"] = p

    return {
        "name": name,
        "average_precision": ap,
        "roc_auc": roc,
        "threshold": threshold,
        "confusion_matrix": cm,
        **prec_at_rec,
        "y_prob": y_prob,
    }


def fpr_at_tpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_tpr: float = 0.90,
) -> float:
    """
    False Positive Rate at a given True Positive Rate.
    Useful for low-FP operating points.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return float(fpr[idx])