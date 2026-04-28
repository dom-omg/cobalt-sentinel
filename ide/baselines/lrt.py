"""Likelihood-Ratio Test (LRT) baseline detector.

Canonical G² statistic on sliding window with Laplace smoothing.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.stats import chi2


class LRTDetector:
    """G²-based Likelihood-Ratio Test on a sliding window.

    Args:
        vocab: List of action names.
        alpha: Significance level for chi-squared critical value.
        window_size: Number of recent actions used per test.
    """

    def __init__(
        self,
        vocab: list,
        alpha: float = 0.05,
        window_size: int = 20,
    ) -> None:
        self.vocab = [v.name if hasattr(v, "name") else v for v in vocab]
        self.alpha = alpha
        self.window_size = window_size
        self._n = len(self.vocab)
        self._name_to_idx: dict[str, int] = {v: i for i, v in enumerate(self.vocab)}
        self._baseline: np.ndarray | None = None
        self._window: deque[str] = deque(maxlen=window_size)
        self._critical: float = chi2.ppf(1.0 - alpha, df=max(self._n - 1, 1))

    def calibrate(self, action_names: list[str]) -> None:
        """Estimate baseline distribution with Laplace smoothing.

        Args:
            action_names: Calibration trace.
        """
        counts = np.ones(self._n, dtype=np.float64)  # Laplace prior
        for name in action_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0
        self._baseline = counts / counts.sum()
        self._window.clear()

    def observe(self, action_name: str, **_kwargs) -> dict:
        """Process one action and test for drift.

        Args:
            action_name: Observed action name.

        Returns:
            Dict with keys: alert (bool), g2 (float), critical (float).
        """
        if self._baseline is None:
            return {"alert": False, "g2": 0.0}

        self._window.append(action_name)
        if len(self._window) < self.window_size:
            return {"alert": False, "g2": 0.0}

        counts = np.zeros(self._n, dtype=np.float64)
        for name in self._window:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0

        N = counts.sum()
        if N == 0:
            return {"alert": False, "g2": 0.0}

        observed = counts / N
        g2 = 0.0
        for i in range(self._n):
            if counts[i] > 0:
                g2 += 2.0 * counts[i] * math.log(observed[i] / self._baseline[i])

        alert = g2 > self._critical
        return {"alert": alert, "g2": g2, "critical": self._critical}
