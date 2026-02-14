"""Evaluation metrics for galaxy morphology regression.

Computes Zoobot-aligned metrics: per-question and aggregate, with optional
bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data.dataset import MorphologySchema


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    schema: MorphologySchema | None = None,
) -> dict[str, float]:
    """Compute aggregate metrics across all valid outputs.

    Parameters
    ----------
    predictions : [N, num_outputs] predicted vote fractions
    targets : [N, num_outputs] true vote fractions
    masks : [N, num_outputs] validity mask

    Returns
    -------
    Dict of metric names to values.
    """
    # Flatten to only valid entries
    valid = masks.astype(bool).flatten()
    pred_flat = predictions.flatten()[valid]
    true_flat = targets.flatten()[valid]

    results = {
        "mse": mean_squared_error(true_flat, pred_flat),
        "mae": mean_absolute_error(true_flat, pred_flat),
        "r2": r2_score(true_flat, pred_flat),
    }

    # Correlation
    if len(pred_flat) > 2:
        results["pearson_r"], results["pearson_p"] = stats.pearsonr(true_flat, pred_flat)
        results["spearman_r"], results["spearman_p"] = stats.spearmanr(true_flat, pred_flat)

    # Classification metrics (via argmax per question)
    if schema is not None:
        acc_list, f1_list = [], []
        for question, (start, end) in schema.question_slices.items():
            q_mask = masks[:, start] > 0
            if q_mask.sum() == 0:
                continue
            pred_cls = predictions[q_mask, start:end].argmax(axis=1)
            true_cls = targets[q_mask, start:end].argmax(axis=1)
            acc_list.append(accuracy_score(true_cls, pred_cls))
            f1_list.append(f1_score(true_cls, pred_cls, average="weighted", zero_division=0))

        if acc_list:
            results["accuracy_mean"] = np.mean(acc_list)
            results["f1_weighted_mean"] = np.mean(f1_list)

    return results


def compute_per_question_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    schema: MorphologySchema,
) -> dict[str, dict[str, float]]:
    """Compute metrics for each morphology question separately.

    Returns
    -------
    Dict mapping question names to their metric dicts.
    """
    results = {}

    for question, (start, end) in schema.question_slices.items():
        q_mask = masks[:, start] > 0
        n_valid = q_mask.sum()

        if n_valid == 0:
            results[question] = {"n_valid": 0}
            continue

        q_pred = predictions[q_mask, start:end]
        q_true = targets[q_mask, start:end]

        # Regression metrics on vote fractions
        pred_flat = q_pred.flatten()
        true_flat = q_true.flatten()

        q_results = {
            "n_valid": int(n_valid),
            "mse": mean_squared_error(true_flat, pred_flat),
            "mae": mean_absolute_error(true_flat, pred_flat),
            "r2": r2_score(true_flat, pred_flat),
        }

        # Correlation
        if len(pred_flat) > 2:
            q_results["pearson_r"], _ = stats.pearsonr(true_flat, pred_flat)
            q_results["spearman_r"], _ = stats.spearmanr(true_flat, pred_flat)

        # Classification (argmax discretization)
        pred_cls = q_pred.argmax(axis=1)
        true_cls = q_true.argmax(axis=1)
        q_results["accuracy"] = accuracy_score(true_cls, pred_cls)
        q_results["f1_weighted"] = f1_score(
            true_cls, pred_cls, average="weighted", zero_division=0,
        )

        results[question] = q_results

    return results


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    metric_fn: callable,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for a metric.

    Parameters
    ----------
    metric_fn : callable(pred, true, mask) -> float

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)

    point = metric_fn(predictions, targets, masks)
    boot_values = []

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        val = metric_fn(predictions[idx], targets[idx], masks[idx])
        boot_values.append(val)

    boot_values = np.array(boot_values)
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(boot_values, alpha * 100)
    ci_upper = np.percentile(boot_values, (1 - alpha) * 100)

    return point, ci_lower, ci_upper
