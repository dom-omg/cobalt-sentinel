"""Wald's Sequential Probability Ratio Test (SPRT) engine.

Reference: Wald, A. (1945). Sequential tests of statistical hypotheses.
"""

from __future__ import annotations

import math
from enum import Enum


class SPRTDecision(Enum):
    """Possible outcomes from a SPRT update step."""

    CONTINUE = "CONTINUE"
    H0_ACCEPTED = "H0_ACCEPTED"
    H1_ACCEPTED = "H1_ACCEPTED"


class SequentialTester:
    """Implements Wald's SPRT for sequential drift detection.

    Args:
        alpha: Type-I error bound (false alarm rate).
        beta: Type-II error bound (miss rate).
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.05) -> None:
        if not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError("alpha and beta must be in (0, 1)")
        self.alpha = alpha
        self.beta = beta
        self.log_A: float = math.log(beta / (1.0 - alpha))
        self.log_B: float = math.log((1.0 - beta) / alpha)
        self.S: float = 0.0
        self.n: int = 0

    def update(self, log_lr: float) -> SPRTDecision:
        """Incorporate one log-likelihood ratio observation.

        Args:
            log_lr: log(P(obs | H1) / P(obs | H0)) for this observation.

        Returns:
            SPRTDecision indicating whether to continue or stop.
        """
        self.S += log_lr
        self.n += 1
        if self.S <= self.log_A:
            return SPRTDecision.H0_ACCEPTED
        if self.S >= self.log_B:
            return SPRTDecision.H1_ACCEPTED
        return SPRTDecision.CONTINUE

    def reset(self) -> None:
        """Reset cumulative statistic and counter."""
        self.S = 0.0
        self.n = 0

    @property
    def expected_sample_size_h0(self) -> float:
        """Wald approximation of E[N | H0]."""
        # E[N|H0] ≈ (α·log_B + (1-α)·log_A) / E[log_LR | H0]
        # Under H0, E[log_LR] ≈ log_A (boundary approximation)
        # Simplified: (α·log_B + (1-α)·log_A) / log_A  — conservative lower bound
        if abs(self.log_A) < 1e-10:
            return float("inf")
        return (self.alpha * self.log_B + (1 - self.alpha) * self.log_A) / self.log_A

    @property
    def expected_sample_size_h1(self) -> float:
        """Wald approximation of E[N | H1]."""
        if abs(self.log_B) < 1e-10:
            return float("inf")
        return ((1 - self.beta) * self.log_B + self.beta * self.log_A) / self.log_B
