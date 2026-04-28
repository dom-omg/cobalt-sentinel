"""CBD baseline: Cosine Behavioral Distance without SPRT or semantics."""

from __future__ import annotations

import numpy as np

WARNING_THRESHOLD = 0.35
CRITICAL_THRESHOLD = 0.65
EPS = 1e-9


class CBDDetector:
    """Cosine Behavioral Distance with fixed thresholds (paper baseline method).

    Args:
        vocab: List of action specs or names.
        alpha: Not used (thresholds are fixed at 0.35/0.65).
        window_size: Sliding window size for empirical distribution.
    """

    def __init__(self, vocab: list, alpha: float = 0.05, window_size: int = 20) -> None:
        from collections import deque

        self.vocab = [v.name if hasattr(v, "name") else v for v in vocab]
        self._n = len(self.vocab)
        self._name_to_idx = {v: i for i, v in enumerate(self.vocab)}
        self.window_size = window_size
        self._baseline: np.ndarray | None = None
        self._window: deque[str] = deque(maxlen=window_size)

    def calibrate(self, action_names: list[str]) -> None:
        """Estimate baseline frequency distribution."""
        counts = np.zeros(self._n, dtype=np.float64)
        for name in action_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0
        total = counts.sum()
        self._baseline = counts / total if total > 0 else np.ones(self._n) / self._n
        self._window.clear()

    def observe(self, action_name: str, **_kwargs) -> dict:
        """Compute CBD on sliding window."""
        if self._baseline is None:
            return {"alert": False, "cbd": 0.0}

        self._window.append(action_name)
        if len(self._window) < self.window_size:
            return {"alert": False, "cbd": 0.0}

        counts = np.zeros(self._n, dtype=np.float64)
        for name in self._window:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0

        total = counts.sum()
        p = counts / total if total > 0 else np.ones(self._n) / self._n

        denom = (np.linalg.norm(p) * np.linalg.norm(self._baseline)) + EPS
        cbd = float(1.0 - np.dot(p, self._baseline) / denom)

        alert = cbd >= CRITICAL_THRESHOLD
        level = "CRITICAL" if cbd >= CRITICAL_THRESHOLD else ("WARNING" if cbd >= WARNING_THRESHOLD else "CLEAN")
        return {"alert": alert, "cbd": cbd, "level": level}
