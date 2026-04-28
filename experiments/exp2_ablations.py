"""Ablation study: semantic_lambda and tier_weights sweep.

Usage:
    python -m experiments.exp2_ablations
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

LAMBDA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
WEIGHT_SWEEP = [
    {"SAFE": 1.0, "UNSAFE": 1.0, "CRITICAL": 1.0},
    {"SAFE": 1.0, "UNSAFE": 2.0, "CRITICAL": 5.0},
    {"SAFE": 1.0, "UNSAFE": 5.0, "CRITICAL": 10.0},
]
N_TRIALS = 100
SEED = 42
OUTPUT_DIR = "results/exp2"


def run_ablation(n_trials: int = N_TRIALS, seed: int = SEED) -> pd.DataFrame:
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.runner import run_method_on_trace
    from ide.evaluation.simulator import (
        AgentSimulator,
        DriftScenario,
        BASELINE_CS,
        PHISHING_DRIFT,
        build_customer_service_vocab,
    )
    from ide.ss_cbd import SSCBDDetector

    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    sim = AgentSimulator(vocab=vocab, seed=seed)
    scenario = DriftScenario(
        name="phishing_step",
        baseline_dist=BASELINE_CS,
        drift_dist=PHISHING_DRIFT,
        onset_time=200,
        drift_type="step",
    )

    rows = []
    rng = np.random.default_rng(seed)
    for lam in LAMBDA_SWEEP:
        for weights in WEIGHT_SWEEP:
            times = []
            for trial in range(n_trials):
                trace = sim.generate_trace(scenario, duration=600, seed=int(rng.integers(0, 2**31)))
                det = SSCBDDetector(
                    vocab=vocab, embedder=embedder,
                    semantic_lambda=lam, tier_weights=weights,
                    alpha=0.05, beta=0.05, n_min=20,
                )
                detected, t = run_method_on_trace(det, trace, alt_dist=PHISHING_DRIFT)
                if detected:
                    times.append(t)
            rows.append({
                "lambda": lam,
                "w_unsafe": weights["UNSAFE"],
                "w_critical": weights["CRITICAL"],
                "det_rate": len(times) / n_trials,
                "mean_det_time": float(np.mean(times)) if times else float("inf"),
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running ablation study: λ ∈ {LAMBDA_SWEEP}, N={N_TRIALS} trials\n")
    df = run_ablation()
    df.to_csv(os.path.join(OUTPUT_DIR, "ablations.csv"), index=False)

    print(f"{'λ':>6} {'w_unsafe':>9} {'w_crit':>8} {'det%':>8} {'mean_time':>11}")
    print("─" * 50)
    for _, row in df.sort_values("mean_det_time").iterrows():
        pct = f"{row['det_rate']*100:.1f}"
        t = f"{row['mean_det_time']:.1f}" if row["mean_det_time"] != float("inf") else "∞"
        print(f"{row['lambda']:>6.2f} {row['w_unsafe']:>9.0f} {row['w_critical']:>8.0f} {pct:>8} {t:>11}")

    best = df.loc[df["mean_det_time"].idxmin()]
    print(f"\nBest config: λ={best['lambda']}, w_unsafe={best['w_unsafe']}, w_critical={best['w_critical']}")
    if best["lambda"] == 0.5 and best["w_unsafe"] == 2.0 and best["w_critical"] == 5.0:
        print("✓ Default (λ=0.5, w=(2,5)) is optimal.")
    else:
        print(f"⚠ Best config differs from default — update hyperparameters honestly.")


if __name__ == "__main__":
    main()
