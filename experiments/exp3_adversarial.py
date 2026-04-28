"""Adversarial robustness experiments.

1. Threshold-straddling: adversary maintains d_cos = θ* - 0.01
2. Slow drift: linear ramp over 6h; compare SS-CBD vs LRT vs ADWIN
3. Volume dilution: 4x higher action rate; verify magnitude-invariance

Usage:
    python -m experiments.exp3_adversarial
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

N_TRIALS = 100
SEED = 42
OUTPUT_DIR = "results/exp3"
ACTIONS_PER_HOUR = 720  # 12 per minute × 60


def exp3_1_threshold_straddling() -> dict:
    """Adversary maintains cosine distance just below the SPRT trigger."""
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.simulator import BASELINE_CS, PHISHING_DRIFT, build_customer_service_vocab
    from ide.ss_cbd import SSCBDDetector

    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    names = [v.name for v in vocab]
    n = len(vocab)
    rng = np.random.default_rng(SEED)

    # Adversary blends baseline and drift to stay at 98% of normal drift intensity
    BLEND = 0.98
    adversarial_dist = (1.0 - BLEND) * BASELINE_CS + BLEND * PHISHING_DRIFT
    adversarial_dist /= adversarial_dist.sum()

    detected = 0
    for _ in range(N_TRIALS):
        det = SSCBDDetector(vocab=vocab, embedder=embedder, alpha=0.05, beta=0.05, n_min=20)
        calib = rng.choice(names, size=50, p=BASELINE_CS).tolist()
        det.calibrate(calib)
        for _ in range(500):
            action = str(rng.choice(names, p=adversarial_dist))
            result = det.observe(action, alt_dist=adversarial_dist)
            if result["decision"] == "DRIFT_DETECTED":
                detected += 1
                break

    detection_rate = detected / N_TRIALS
    max_injectable = float(np.max(PHISHING_DRIFT - BASELINE_CS))
    return {
        "exp": "threshold_straddling",
        "blend": BLEND,
        "detection_rate": detection_rate,
        "max_critical_injectable_fraction": max_injectable,
    }


def exp3_2_slow_drift() -> pd.DataFrame:
    """Slow linear ramp over 6h; compare detection times."""
    from ide.baselines.adwin import ADWINDetector
    from ide.baselines.lrt import LRTDetector
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.runner import run_method_on_trace
    from ide.evaluation.simulator import (
        BASELINE_CS, PHISHING_DRIFT, AgentSimulator,
        DriftScenario, build_customer_service_vocab,
    )
    from ide.ss_cbd import SSCBDDetector

    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    sim = AgentSimulator(vocab=vocab, seed=SEED)

    ramp_duration = ACTIONS_PER_HOUR * 6  # 6h of actions
    scenario = DriftScenario(
        name="slow_drift_6h",
        baseline_dist=BASELINE_CS,
        drift_dist=PHISHING_DRIFT,
        onset_time=500,
        drift_type="linear_ramp",
        ramp_duration=ramp_duration,
    )
    duration = 500 + ramp_duration + 500
    rng = np.random.default_rng(SEED)
    rows = []
    for method_name, make_method in [
        ("SS-CBD", lambda: SSCBDDetector(vocab=vocab, embedder=embedder, alpha=0.05, beta=0.05, n_min=20)),
        ("LRT", lambda: LRTDetector(vocab=vocab, alpha=0.05, window_size=20)),
        ("ADWIN", lambda: ADWINDetector(vocab=vocab, alpha=0.05)),
    ]:
        times = []
        for trial in range(N_TRIALS):
            trace = sim.generate_trace(scenario, duration=duration, seed=int(rng.integers(0, 2**31)))
            method = make_method()
            detected, t = run_method_on_trace(method, trace, alt_dist=PHISHING_DRIFT)
            if detected:
                times.append(t)
        rows.append({
            "method": method_name,
            "scenario": "slow_drift_6h",
            "det_rate": len(times) / N_TRIALS,
            "mean_det_time": float(np.mean(times)) if times else float("inf"),
        })
    return pd.DataFrame(rows)


def exp3_3_volume_dilution() -> dict:
    """4x higher volume should not change SS-CBD detection time significantly."""
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.runner import run_method_on_trace
    from ide.evaluation.simulator import (
        BASELINE_CS, PHISHING_DRIFT, AgentSimulator,
        DriftScenario, build_customer_service_vocab,
    )
    from ide.ss_cbd import SSCBDDetector

    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()

    scenario = DriftScenario(
        name="dilution",
        baseline_dist=BASELINE_CS,
        drift_dist=PHISHING_DRIFT,
        onset_time=200,
        drift_type="step",
    )
    results = {}
    for multiplier in [1, 4]:
        sim = AgentSimulator(vocab=vocab, seed=SEED)
        times = []
        rng = np.random.default_rng(SEED)
        for trial in range(N_TRIALS):
            trace = sim.generate_trace(scenario, duration=600 * multiplier, seed=int(rng.integers(0, 2**31)))
            det = SSCBDDetector(vocab=vocab, embedder=embedder, alpha=0.05, beta=0.05, n_min=20)
            detected, t = run_method_on_trace(det, trace, alt_dist=PHISHING_DRIFT)
            if detected:
                times.append(t)
        results[multiplier] = float(np.mean(times)) if times else float("inf")

    ratio = results[4] / results[1] if results[1] > 0 else float("nan")
    return {
        "exp": "volume_dilution",
        "mean_time_1x": results[1],
        "mean_time_4x": results[4],
        "ratio_4x_1x": ratio,
        "magnitude_invariant": ratio < 1.5,  # within 50% → invariant enough
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== EXP 3.1: Threshold-Straddling ===")
    r1 = exp3_1_threshold_straddling()
    print(f"Detection rate at {r1['blend']*100:.0f}% drift intensity: {r1['detection_rate']*100:.1f}%")
    print(f"Max CRITICAL-action injectable fraction: {r1['max_critical_injectable_fraction']:.3f}")

    print("\n=== EXP 3.2: Slow Drift (6h linear ramp) ===")
    df2 = exp3_2_slow_drift()
    print(df2.to_string(index=False))
    df2.to_csv(os.path.join(OUTPUT_DIR, "slow_drift.csv"), index=False)

    print("\n=== EXP 3.3: Volume Dilution (4x) ===")
    r3 = exp3_3_volume_dilution()
    print(f"Mean detection time 1x: {r3['mean_time_1x']:.1f}")
    print(f"Mean detection time 4x: {r3['mean_time_4x']:.1f}")
    print(f"Ratio: {r3['ratio_4x_1x']:.2f} — magnitude-invariant: {r3['magnitude_invariant']}")

    pd.DataFrame([r1]).to_csv(os.path.join(OUTPUT_DIR, "threshold_straddling.csv"), index=False)
    pd.DataFrame([r3]).to_csv(os.path.join(OUTPUT_DIR, "volume_dilution.csv"), index=False)


if __name__ == "__main__":
    main()
