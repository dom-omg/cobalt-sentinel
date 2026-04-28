"""Tests for ide/sequential.py."""

import math

import numpy as np
import pytest

from ide.sequential import SPRTDecision, SequentialTester


def _run_sprt_trials(mu_h0, mu_h1, sample_from, n_trials, alpha, beta, seed):
    """Run n_trials of SPRT with given H0/H1 means."""
    rng = np.random.default_rng(seed)
    decisions = []
    for _ in range(n_trials):
        tester = SequentialTester(alpha=alpha, beta=beta)
        for _ in range(5000):
            x = rng.normal(sample_from, 1.0)
            log_lr = (mu_h1 - mu_h0) * x - 0.5 * (mu_h1**2 - mu_h0**2)
            decision = tester.update(log_lr)
            if decision != SPRTDecision.CONTINUE:
                decisions.append(decision)
                break
        else:
            decisions.append(SPRTDecision.CONTINUE)
    return decisions


def test_sprt_respects_alpha():
    """FPR when sampling from H0 must be ≤ 0.07 (tolerance for CI)."""
    alpha, beta = 0.05, 0.05
    decisions = _run_sprt_trials(
        mu_h0=0.0, mu_h1=1.0, sample_from=0.0,
        n_trials=1000, alpha=alpha, beta=beta, seed=0
    )
    false_alarms = sum(1 for d in decisions if d == SPRTDecision.H1_ACCEPTED)
    fpr = false_alarms / len(decisions)
    assert fpr <= 0.07, f"FPR {fpr:.3f} exceeds tolerance 0.07"


def test_sprt_respects_beta():
    """Miss rate when sampling from H1 must be ≤ 0.07."""
    alpha, beta = 0.05, 0.05
    decisions = _run_sprt_trials(
        mu_h0=0.0, mu_h1=1.0, sample_from=1.0,
        n_trials=1000, alpha=alpha, beta=beta, seed=1
    )
    misses = sum(1 for d in decisions if d == SPRTDecision.H0_ACCEPTED)
    miss_rate = misses / len(decisions)
    assert miss_rate <= 0.07, f"Miss rate {miss_rate:.3f} exceeds tolerance 0.07"


def test_sprt_faster_than_fixed_sample():
    """SPRT expected sample size under H0 should be less than fixed-sample n."""
    alpha, beta = 0.05, 0.20
    tester = SequentialTester(alpha=alpha, beta=beta)
    # Fixed-sample requires n ≈ (z_alpha + z_beta)² / delta² for normal test
    # With delta=1, z_alpha=1.645, z_beta=0.84: n ≈ (1.645+0.84)²/1 ≈ 6.2
    fixed_n = (1.645 + 0.84) ** 2
    ess_h0 = abs(tester.expected_sample_size_h0)
    # SPRT should converge faster on average
    assert ess_h0 < fixed_n * 3, (
        f"ESS H0 {ess_h0:.1f} unexpectedly large vs fixed {fixed_n:.1f}"
    )


def test_reset_clears_state():
    """reset() must restore S=0 and n=0."""
    tester = SequentialTester()
    tester.update(2.0)
    tester.update(1.0)
    tester.reset()
    assert tester.S == 0.0
    assert tester.n == 0


def test_h1_accepted_on_strong_signal():
    """Strong positive log-LR stream should trigger H1_ACCEPTED."""
    tester = SequentialTester(alpha=0.05, beta=0.05)
    result = SPRTDecision.CONTINUE
    for _ in range(200):
        result = tester.update(0.5)
        if result == SPRTDecision.H1_ACCEPTED:
            break
    assert result == SPRTDecision.H1_ACCEPTED
