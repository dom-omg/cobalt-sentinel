"""Chi-squared goodness-of-fit baseline detector."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import chi2 as chi2_dist

EPS = 1e-9


class Chi2Detector:
    """Pearson chi-squared test on a sliding window.

    Args:
        vocab: List of action specs or names.
        alpha: Significance level.
        window_size: Sliding window size.
    """

    def __init__(self, vocab: list, alpha: float = 0.05, window_size: int = 20) -> None:
        self.vocab = [v.name if hasattr(v, "name") else v for v in vocab]
        self._n = len(self.vocab)
        self._name_to_idx = {v: i for i, v in enumerate(self.vocab)}
        self.alpha = alpha
        self.window_size = window_size
        self._baseline: np.ndarray | None = None
        self._window: deque[str] = deque(maxlen=window_size)
        self._critical = chi2_dist.ppf(1.0 - alpha, df=max(self._n - 1, 1))

    def calibrate(self, action_names: list[str]) -> None:
        """Estimate baseline with Laplace smoothing."""
        counts = np.ones(self._n, dtype=np.float64)
        for name in action_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0
        self._baseline = counts / counts.sum()
        self._window.clear()

    def observe(self, action_name: str, **_kwargs) -> dict:
        """Pearson chi-squared test: Σ (c_i - N·b_i)² / (N·b_i)."""
        if self._baseline is None:
            return {"alert": False, "stat": 0.0}

        self._window.append(action_name)
        if len(self._window) < self.window_size:
            return {"alert": False, "stat": 0.0}

        counts = np.zeros(self._n, dtype=np.float64)
        for name in self._window:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0

        N = counts.sum()
        expected = N * self._baseline
        stat = float(np.sum((counts - expected) ** 2 / (expected + EPS)))
        alert = stat > self._critical
        return {"alert": alert, "stat": stat, "critical": self._critical}
