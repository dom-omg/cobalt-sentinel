"""Page's CUSUM baseline detector on the dominant action's frequency."""

from __future__ import annotations

import numpy as np

DEFAULT_K = 0.5   # allowable slack (half the expected shift)
DEFAULT_H = 5.0   # decision threshold


class CUSUMDetector:
    """Page's CUSUM on the dominant action's indicator stream.

    Args:
        vocab: List of action specs or names.
        alpha: Maps to CUSUM threshold h = -log(alpha) * 2 (heuristic).
        window_size: Not used (CUSUM is inherently sequential).
    """

    def __init__(self, vocab: list, alpha: float = 0.05, window_size: int = 20) -> None:
        self.vocab = [v.name if hasattr(v, "name") else v for v in vocab]
        self._n = len(self.vocab)
        self._name_to_idx = {v: i for i, v in enumerate(self.vocab)}
        self.alpha = alpha
        self._h = float(-2.0 * np.log(alpha))
        self._dominant_idx: int = 0
        self._mu0: float = 0.0
        self._C_pos: float = 0.0
        self._C_neg: float = 0.0

    def calibrate(self, action_names: list[str]) -> None:
        """Identify dominant action and estimate its baseline rate."""
        counts = np.zeros(self._n, dtype=np.float64)
        for name in action_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0
        N = len(action_names) if action_names else 1
        self._dominant_idx = int(np.argmax(counts))
        self._mu0 = float(counts[self._dominant_idx] / N)
        self._C_pos = 0.0
        self._C_neg = 0.0

    def observe(self, action_name: str, **_kwargs) -> dict:
        """Update CUSUM and check threshold."""
        idx = self._name_to_idx.get(action_name)
        x = 1.0 if idx == self._dominant_idx else 0.0

        k = DEFAULT_K * self._mu0
        self._C_pos = max(0.0, self._C_pos + x - self._mu0 - k)
        self._C_neg = max(0.0, self._C_neg - x + self._mu0 - k)

        alert = self._C_pos > self._h or self._C_neg > self._h
        if alert:
            self._C_pos = 0.0
            self._C_neg = 0.0
        return {"alert": alert, "C_pos": self._C_pos, "C_neg": self._C_neg}
