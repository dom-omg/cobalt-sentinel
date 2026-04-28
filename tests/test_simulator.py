"""Tests for ide/evaluation/simulator.py."""

import numpy as np
import pytest

from ide.evaluation.simulator import (
    BASELINE_CS,
    AgentSimulator,
    DriftScenario,
    build_customer_service_vocab,
)


@pytest.fixture()
def vocab():
    return build_customer_service_vocab()


@pytest.fixture()
def sim(vocab):
    return AgentSimulator(vocab=vocab, seed=0)


def test_no_drift_produces_baseline_dist(sim, vocab):
    """Control scenario: empirical KL(p_empirical || baseline) < 0.05."""
    scenario = DriftScenario(
        name="control",
        baseline_dist=BASELINE_CS,
        drift_dist=BASELINE_CS,
        onset_time=999999,
        drift_type="step",
    )
    names = [v.name for v in vocab]
    name_to_idx = {n: i for i, n in enumerate(names)}
    trace = sim.generate_trace(scenario, duration=10000, seed=42)

    counts = np.zeros(len(vocab), dtype=np.float64)
    for action in trace:
        idx = name_to_idx.get(action)
        if idx is not None:
            counts[idx] += 1.0
    p_emp = counts / counts.sum()

    eps = 1e-9
    kl = float(np.sum(p_emp * np.log((p_emp + eps) / (BASELINE_CS + eps))))
    assert kl < 0.05, f"KL divergence {kl:.4f} ≥ 0.05"


def test_step_drift_visible(sim, vocab):
    """Post-onset distribution should differ from pre-onset."""
    from ide.evaluation.simulator import PHISHING_DRIFT

    scenario = DriftScenario(
        name="phishing_step",
        baseline_dist=BASELINE_CS,
        drift_dist=PHISHING_DRIFT,
        onset_time=500,
        drift_type="step",
    )
    names = [v.name for v in vocab]
    name_to_idx = {n: i for i, n in enumerate(names)}
    trace = sim.generate_trace(scenario, duration=2000, seed=1)

    def freq(t_slice):
        c = np.zeros(len(vocab), dtype=np.float64)
        for a in t_slice:
            i = name_to_idx.get(a)
            if i is not None:
                c[i] += 1
        return c / c.sum()

    pre = freq(trace[:500])
    post = freq(trace[600:])

    eps = 1e-9
    kl = float(np.sum(post * np.log((post + eps) / (pre + eps))))
    assert kl > 0.01, f"KL divergence {kl:.4f} too small — drift not visible"


def test_seed_reproducibility(sim, vocab):
    """Same seed must produce identical trace."""
    from ide.evaluation.simulator import build_standard_scenarios

    scenario = build_standard_scenarios()[0]
    t1 = sim.generate_trace(scenario, duration=200, seed=99)
    t2 = sim.generate_trace(scenario, duration=200, seed=99)
    assert t1 == t2, "Traces differ with same seed"


def test_trace_length(sim, vocab):
    """Trace length must match requested duration."""
    from ide.evaluation.simulator import build_standard_scenarios

    scenario = build_standard_scenarios()[0]
    for dur in [10, 100, 500]:
        trace = sim.generate_trace(scenario, duration=dur, seed=0)
        assert len(trace) == dur


def test_all_actions_in_vocab(sim, vocab):
    """All generated actions must be in vocab."""
    from ide.evaluation.simulator import build_standard_scenarios

    names = {v.name for v in vocab}
    scenario = build_standard_scenarios()[0]
    trace = sim.generate_trace(scenario, duration=1000, seed=5)
    for action in trace:
        assert action in names, f"Unknown action in trace: {action}"
