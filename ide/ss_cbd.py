"""SS-CBD: Sequential Semantic Cosine Behavioral Distance detector.

Flagship method combining:
  1. Sequential testing via Wald's SPRT
  2. Semantic action embeddings
  3. Tier-weighted operational risk
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .embeddings import ActionEmbedder, ActionSpec
from .sequential import SPRTDecision, SequentialTester

EPS = 1e-9
DEFAULT_TIER_WEIGHTS: dict[str, float] = {"SAFE": 1.0, "UNSAFE": 2.0, "CRITICAL": 5.0}


class SSCBDDetector:
    """SS-CBD drift detector.

    Combines frequency log-likelihood ratio with semantic distance, weighted
    by operational tier, then feeds the composite signal into a SPRT engine.

    Args:
        vocab: List of known ActionSpecs for this agent.
        embedder: ActionEmbedder instance (shared across detectors is fine).
        alpha: SPRT Type-I error bound.
        beta: SPRT Type-II error bound.
        tier_weights: Override default tier weights (SAFE=1, UNSAFE=2, CRITICAL=5).
        semantic_lambda: Blend weight between frequency (0) and semantic (1) signal.
        n_min: Number of calibration observations required before locking baseline.
    """

    def __init__(
        self,
        vocab: list[ActionSpec],
        embedder: ActionEmbedder,
        alpha: float = 0.05,
        beta: float = 0.05,
        tier_weights: dict[str, float] | None = None,
        semantic_lambda: float = 0.5,
        n_min: int = 20,
    ) -> None:
        self.vocab = vocab
        self.embedder = embedder
        self.alpha = alpha
        self.beta = beta
        self.tier_weights = tier_weights or DEFAULT_TIER_WEIGHTS
        self.semantic_lambda = semantic_lambda
        self.n_min = n_min

        self._name_to_idx: dict[str, int] = {s.name: i for i, s in enumerate(vocab)}
        self._tier_w: np.ndarray = np.array(
            [self.tier_weights.get(s.tier, 1.0) for s in vocab], dtype=np.float64
        )
        self._E: np.ndarray = embedder.embed_vocabulary(vocab)  # (n, d)

        self._baseline: np.ndarray | None = None
        self._centroid: np.ndarray | None = None
        self._locked: bool = False
        self._calibration_counts: np.ndarray = np.zeros(len(vocab), dtype=np.float64)
        self._calibration_n: int = 0

        self._sprt = SequentialTester(alpha=alpha, beta=beta)

    # ── Calibration ────────────────────────────────────────────────────────────

    def calibrate(self, action_names: list[str]) -> None:
        """Feed calibration window and lock baseline.

        Args:
            action_names: Sequence of action names from the baseline period.
        """
        counts = np.zeros(len(self.vocab), dtype=np.float64)
        for name in action_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                counts[idx] += 1
        self._calibration_counts = counts
        self._calibration_n = len(action_names)
        self._lock_baseline(counts)

    def _lock_baseline(self, counts: np.ndarray) -> None:
        n = counts.sum()
        if n == 0:
            b = np.ones(len(self.vocab), dtype=np.float64) / len(self.vocab)
        else:
            b = (counts + EPS) / (n + EPS * len(self.vocab))

        b_w = b * self._tier_w
        b_w /= b_w.sum()

        centroid = (b_w[:, None] * self._E).sum(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > EPS:
            centroid /= norm

        self._baseline = b
        self._centroid = centroid
        self._locked = True
        self._sprt.reset()

    # ── Observation ────────────────────────────────────────────────────────────

    def observe(
        self,
        action_name: str,
        alt_dist: np.ndarray | None = None,
    ) -> dict:
        """Process one action observation.

        Args:
            action_name: Name of the observed action.
            alt_dist: Alternative distribution (H1) for log-LR computation.
                      If None, a uniform shift is assumed.

        Returns:
            Dict with keys: decision, sprt_S, n, log_lr, semantic_dist, tier_factor.
        """
        if not self._locked:
            return {"decision": "CALIBRATING"}

        idx = self._name_to_idx.get(action_name)
        if idx is None:
            return {"decision": "DRIFT_NEW_ACTION", "action": action_name}

        b_i = float(self._baseline[idx])  # type: ignore[index]
        tier_factor = float(self._tier_w[idx]) / float(self._tier_w.mean())

        if alt_dist is not None and idx < len(alt_dist):
            alt_i = float(alt_dist[idx]) + EPS
        else:
            alt_i = (1.0 / len(self.vocab)) + EPS

        freq_lr = math.log((alt_i) / (b_i + EPS))

        e_i = self._E[idx]
        semantic_dist = float(1.0 - np.dot(e_i, self._centroid))  # type: ignore[arg-type]

        log_lr = tier_factor * (
            (1.0 - self.semantic_lambda) * freq_lr
            + self.semantic_lambda * semantic_dist * math.copysign(1.0, freq_lr)
        )

        decision_enum = self._sprt.update(log_lr)
        if decision_enum == SPRTDecision.H1_ACCEPTED:
            decision: str = "DRIFT_DETECTED"
        elif decision_enum == SPRTDecision.H0_ACCEPTED:
            decision = "NO_DRIFT"
        else:
            decision = "CONTINUE"

        return {
            "decision": decision,
            "sprt_S": self._sprt.S,
            "n": self._sprt.n,
            "log_lr": log_lr,
            "semantic_dist": semantic_dist,
            "tier_factor": tier_factor,
        }

    @property
    def is_locked(self) -> bool:
        """Whether the baseline has been locked."""
        return self._locked

    def reset(self) -> None:
        """Reset SPRT state without clearing the baseline."""
        self._sprt.reset()
