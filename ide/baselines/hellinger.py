"""Hellinger Distance baseline detector with empirical threshold calibration."""

from __future__ import annotations

from collections import deque

import numpy as np

EPS = 1e-9
CALIBRATION_TRIALS = 500
FPR_QUANTILE = 0.97


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """H(p, q) = (1/√2) · ‖√p − √q‖₂."""
    return float((1.0 / np.sqrt(2.0)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q)))


class HellingerDetector:
    """Hellinger distance on sliding window with empirical FPR=5% threshold.

    Args:
        vocab: List of action specs or names.
        alpha: Target FPR for empirical threshold calibration.
        window_size: Sliding window size.
    """

    def __init__(self, vocab: list, alpha: float = 0.05, window_size: int = 20) -> None:
        self.vocab = [v.name if hasattr(v, "name") else v for v in vocab]
        self._n = len(self.vocab)
        self._name_to_idx = {v: i for i, v in enumerate(self.vocab)}
        self.alpha = alpha
        self.window_size = window_size
        self._baseline: np.ndarray | None = None
        self._threshold: float = 0.2
        self._window: deque[str] = deque(maxlen=window_size)

    def calibrate(self, action_names: list[str], seed: int = 0) -> None:
        """Estimate baseline and calibrate threshold via no-drift simulation."""
        counts = np.ones(self._n, dtype=np.float64)
        for name in action_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0
        self._baseline = counts / counts.sum()
        self._window.clear()

        rng = np.random.default_rng(seed)
        stats = []
        for _ in range(CALIBRATION_TRIALS):
            sample = rng.choice(self._n, size=self.window_size, p=self._baseline)
            sample_counts = np.bincount(sample, minlength=self._n).astype(np.float64)
            p = (sample_counts + EPS) / (sample_counts.sum() + EPS * self._n)
            stats.append(_hellinger(p, self._baseline))
        self._threshold = float(np.quantile(stats, FPR_QUANTILE))

    def observe(self, action_name: str, **_kwargs) -> dict:
        """Compute Hellinger distance and compare to empirical threshold."""
        if self._baseline is None:
            return {"alert": False, "hellinger": 0.0}

        self._window.append(action_name)
        if len(self._window) < self.window_size:
            return {"alert": False, "hellinger": 0.0}

        counts = np.zeros(self._n, dtype=np.float64)
        for name in self._window:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0

        p = (counts + EPS) / (counts.sum() + EPS * self._n)
        h = _hellinger(p, self._baseline)
        return {"alert": h > self._threshold, "hellinger": h, "threshold": self._threshold}
