"""Jensen-Shannon Divergence baseline detector with empirical threshold calibration."""

from __future__ import annotations

from collections import deque

import numpy as np

EPS = 1e-9
CALIBRATION_TRIALS = 500
FPR_QUANTILE = 0.97


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS(p || q) = 0.5 * [KL(p||m) + KL(q||m)], m = (p+q)/2."""
    m = 0.5 * (p + q) + EPS
    kl_pm = np.sum(p * np.log(p / m + EPS))
    kl_qm = np.sum(q * np.log(q / m + EPS))
    return float(0.5 * (kl_pm + kl_qm))


class JSDivergenceDetector:
    """JS divergence on sliding window with empirical FPR=5% threshold.

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
        self._threshold: float = 0.1
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
            stats.append(_js_divergence(p, self._baseline))
        self._threshold = float(np.quantile(stats, FPR_QUANTILE))

    def observe(self, action_name: str, **_kwargs) -> dict:
        """Compute JS divergence and compare to empirical threshold."""
        if self._baseline is None:
            return {"alert": False, "js": 0.0}

        self._window.append(action_name)
        if len(self._window) < self.window_size:
            return {"alert": False, "js": 0.0}

        counts = np.zeros(self._n, dtype=np.float64)
        for name in self._window:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0

        p = (counts + EPS) / (counts.sum() + EPS * self._n)
        js = _js_divergence(p, self._baseline)
        return {"alert": js > self._threshold, "js": js, "threshold": self._threshold}
