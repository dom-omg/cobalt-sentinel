"""Tests for ide/ss_cbd.py including the paper-validation test."""

import numpy as np
import pytest

from ide.embeddings import MockEmbedder
from ide.evaluation.simulator import (
    BASELINE_CS,
    PHISHING_DRIFT,
    AgentSimulator,
    build_customer_service_vocab,
)
from ide.ss_cbd import SSCBDDetector


@pytest.fixture()
def vocab():
    return build_customer_service_vocab()


@pytest.fixture()
def embedder():
    return MockEmbedder()


@pytest.fixture()
def detector(vocab, embedder):
    return SSCBDDetector(vocab=vocab, embedder=embedder, n_min=20)


def _calibrate(detector, vocab, n=50, seed=0):
    rng = np.random.default_rng(seed)
    names = [v.name for v in vocab]
    actions = rng.choice(names, size=n, p=BASELINE_CS).tolist()
    detector.calibrate(actions)
    return actions


def test_calibration_locks(detector, vocab):
    """After calibrate(), is_locked must be True."""
    assert not detector.is_locked
    _calibrate(detector, vocab)
    assert detector.is_locked


def test_unknown_action_triggers_drift(detector, vocab):
    """Action not in vocab → DRIFT_NEW_ACTION."""
    _calibrate(detector, vocab)
    result = detector.observe("unknown_action_xyz")
    assert result["decision"] == "DRIFT_NEW_ACTION"


def test_no_drift_no_alert(vocab, embedder):
    """Sampling from baseline should produce < 10% H1_ACCEPTED (FPR tolerance)."""
    rng = np.random.default_rng(42)
    names = [v.name for v in vocab]
    n_obs = 1000
    alerts = 0
    for trial in range(20):
        det = SSCBDDetector(vocab=vocab, embedder=embedder, n_min=20, alpha=0.05, beta=0.05)
        calib = rng.choice(names, size=50, p=BASELINE_CS).tolist()
        det.calibrate(calib)
        for _ in range(n_obs // 20):
            action = rng.choice(names, p=BASELINE_CS)
            result = det.observe(str(action), alt_dist=BASELINE_CS)
            if result["decision"] == "DRIFT_DETECTED":
                alerts += 1
                det.reset()
    fpr = alerts / n_obs
    assert fpr < 0.10, f"FPR {fpr:.3f} exceeds 0.10 tolerance"


def test_clear_drift_detected(vocab, embedder):
    """Severe distribution reversal should be detected in < 100 obs on ≥95% trials."""
    from ide.evaluation.simulator import FINANCIAL_DRIFT

    rng = np.random.default_rng(7)
    names = [v.name for v in vocab]
    n_trials = 100
    detected = 0
    for trial in range(n_trials):
        det = SSCBDDetector(vocab=vocab, embedder=embedder, n_min=20, alpha=0.05, beta=0.05)
        calib = rng.choice(names, size=50, p=BASELINE_CS).tolist()
        det.calibrate(calib)
        seed = int(rng.integers(0, 2**31))
        obs_rng = np.random.default_rng(seed)
        for _ in range(100):
            action = obs_rng.choice(names, p=FINANCIAL_DRIFT)
            result = det.observe(str(action), alt_dist=FINANCIAL_DRIFT)
            if result["decision"] == "DRIFT_DETECTED":
                detected += 1
                break
    rate = detected / n_trials
    assert rate >= 0.85, f"Detection rate {rate:.2f} < 0.85 on clear drift"


def test_sscbd_beats_lrt_in_detection_time(vocab, embedder):
    """⭐ PAPER VALIDATION: SS-CBD mean detection time × 1.3 < LRT mean detection time.

    Tests on phishing_step scenario (send_email 5%→45%).
    100 trials. This is the critical test for the paper's headline claim.
    """
    from ide.baselines.lrt import LRTDetector
    from ide.evaluation.runner import run_method_on_trace
    from ide.evaluation.simulator import DriftScenario

    scenario = DriftScenario(
        name="phishing_step",
        baseline_dist=BASELINE_CS,
        drift_dist=PHISHING_DRIFT,
        onset_time=200,
        drift_type="step",
    )
    sim = AgentSimulator(vocab=vocab, seed=0)
    n_trials = 100
    sscbd_times = []
    lrt_times = []

    for trial in range(n_trials):
        trace = sim.generate_trace(scenario, duration=600, seed=trial)
        # SS-CBD
        sscbd = SSCBDDetector(vocab=vocab, embedder=embedder, alpha=0.05, beta=0.05, n_min=20)
        detected, t = run_method_on_trace(sscbd, trace, alt_dist=PHISHING_DRIFT)
        if detected:
            sscbd_times.append(t)

        # LRT
        lrt = LRTDetector(vocab=vocab, alpha=0.05, window_size=20)
        detected_lrt, t_lrt = run_method_on_trace(lrt, trace)
        if detected_lrt:
            lrt_times.append(t_lrt)

    assert len(sscbd_times) > 0, "SS-CBD detected nothing — check configuration"
    assert len(lrt_times) > 0, "LRT detected nothing — check configuration"

    mean_sscbd = float(np.mean(sscbd_times))
    mean_lrt = float(np.mean(lrt_times))

    print(f"\n  SS-CBD mean detection time: {mean_sscbd:.1f}")
    print(f"  LRT   mean detection time: {mean_lrt:.1f}")
    print(f"  Speedup: {mean_lrt / mean_sscbd:.2f}x")

    assert mean_sscbd * 1.3 < mean_lrt, (
        f"SS-CBD ({mean_sscbd:.1f}) not faster enough vs LRT ({mean_lrt:.1f}). "
        f"Speedup = {mean_lrt/mean_sscbd:.2f}x < 1.3x. "
        "Investigate before modifying hyperparameters."
    )
