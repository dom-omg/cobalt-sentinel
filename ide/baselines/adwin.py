"""ADWIN baseline detector wrapping river.drift.ADWIN."""

from __future__ import annotations

import numpy as np


class ADWINDetector:
    """Per-action ADWIN drift detector; alerts if any action's rate drifts.

    Args:
        vocab: List of action specs or names.
        alpha: ADWIN delta parameter (sensitivity).
        window_size: Not used (ADWIN is self-windowing).
    """

    def __init__(self, vocab: list, alpha: float = 0.05, window_size: int = 20) -> None:
        self.vocab = [v.name if hasattr(v, "name") else v for v in vocab]
        self._n = len(self.vocab)
        self._name_to_idx = {v: i for i, v in enumerate(self.vocab)}
        self.alpha = alpha
        self._detectors: list | None = None

    def calibrate(self, action_names: list[str]) -> None:
        """Initialize one ADWIN per action and warm up with calibration trace."""
        try:
            from river.drift import ADWIN  # type: ignore
        except ImportError:
            self._detectors = None
            return

        self._detectors = [ADWIN(delta=self.alpha) for _ in range(self._n)]
        for name in action_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                for j, det in enumerate(self._detectors):
                    det.update(1.0 if j == idx else 0.0)
                    det.drift_detected  # consume flag

    def observe(self, action_name: str, **_kwargs) -> dict:
        """Feed binary indicator to each ADWIN; alert if any detects drift."""
        if self._detectors is None:
            return {"alert": False}

        idx = self._name_to_idx.get(action_name)
        alert = False
        for j, det in enumerate(self._detectors):
            det.update(1.0 if j == idx else 0.0)
            if det.drift_detected:
                alert = True
        return {"alert": alert}
