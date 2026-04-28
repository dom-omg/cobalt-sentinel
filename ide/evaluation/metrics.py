"""Evaluation metrics for drift detection experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist


def detection_rate(trial_results: list[dict]) -> float:
    """Fraction of trials where drift was detected.

    Args:
        trial_results: List of dicts with key 'detected' (bool).

    Returns:
        Detection rate in [0, 1].
    """
    if not trial_results:
        return 0.0
    return float(sum(1 for r in trial_results if r["detected"]) / len(trial_results))


def mean_detection_time(trial_results: list[dict]) -> float:
    """Mean detection time over trials that detected drift.

    Args:
        trial_results: List of dicts with keys 'detected' (bool) and
                       'detection_time' (int, actions post-calibration).

    Returns:
        Mean detection time, or inf if no trial detected.
    """
    times = [r["detection_time"] for r in trial_results if r["detected"]]
    return float(np.mean(times)) if times else float("inf")


def clopper_pearson_ci(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for a binomial proportion.

    Args:
        successes: Number of successes.
        trials: Total number of trials.
        alpha: Significance level (1 - confidence level).

    Returns:
        (lower, upper) bounds of the (1-alpha) CI.
    """
    if trials == 0:
        return (0.0, 1.0)
    lo = beta_dist.ppf(alpha / 2.0, successes, trials - successes + 1) if successes > 0 else 0.0
    hi = beta_dist.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes) if successes < trials else 1.0
    return (float(lo), float(hi))


def roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ROC curve from continuous scores and binary labels.

    Args:
        scores: Continuous detection scores (higher = more likely drift).
        labels: Binary labels (1 = drift present).

    Returns:
        (fpr_array, tpr_array) sorted by ascending threshold.
    """
    thresholds = np.sort(np.unique(scores))[::-1]
    fprs, tprs = [1.0], [1.0]
    neg = (labels == 0).sum()
    pos = (labels == 1).sum()
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fprs.append(float(fp / neg) if neg > 0 else 0.0)
        tprs.append(float(tp / pos) if pos > 0 else 0.0)
    fprs.append(0.0)
    tprs.append(0.0)
    return np.array(fprs), np.array(tprs)


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve via trapezoidal rule.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.

    Returns:
        AUC value in [0, 1].
    """
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        values: Sample values.
        n_bootstrap: Number of bootstrap replicates.
        alpha: Significance level.
        seed: RNG seed.

    Returns:
        (lower, upper) percentile bootstrap CI.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    lo = float(np.percentile(means, 100 * alpha / 2.0))
    hi = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))
    return (lo, hi)
