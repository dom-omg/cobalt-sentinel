"""SS-CBD ablation study — 6 hyperparameter sensitivity dimensions.

Sweeps (one at a time; all others fixed at default):
  1. semantic_lambda ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}     N=300
  2. tier_weights: 9 (w_unsafe, w_critical) combinations            N=300
  3. alpha=beta ∈ {0.01, 0.05, 0.10, 0.20, 0.30}                  N=100
  4. calib_size ∈ {5, 10, 20, 50, 100}                             N=100
  5. vocab_size ∈ {5, 8, 10, 12, 15}                               N=100
  6. drift_slope ramp_duration ∈ {0, 50, 100, 200, 400}            N=100

Default config: λ=0.5, w=(1,2,5), α=β=0.05, calib=20, vocab=15, step drift.

Usage:
    python -m experiments.exp2_ablations
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Sweep parameters ───────────────────────────────────────────────────────────

LAMBDA_SWEEP = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

WEIGHT_SWEEP = [
    {"SAFE": 1.0, "UNSAFE": 1.0, "CRITICAL": 1.0},
    {"SAFE": 1.0, "UNSAFE": 1.0, "CRITICAL": 3.0},
    {"SAFE": 1.0, "UNSAFE": 1.0, "CRITICAL": 5.0},
    {"SAFE": 1.0, "UNSAFE": 2.0, "CRITICAL": 2.0},
    {"SAFE": 1.0, "UNSAFE": 2.0, "CRITICAL": 5.0},   # default
    {"SAFE": 1.0, "UNSAFE": 2.0, "CRITICAL": 10.0},
    {"SAFE": 1.0, "UNSAFE": 5.0, "CRITICAL": 5.0},
    {"SAFE": 1.0, "UNSAFE": 5.0, "CRITICAL": 10.0},
    {"SAFE": 1.0, "UNSAFE": 10.0, "CRITICAL": 10.0},
]

ALPHA_BETA_SWEEP = [0.01, 0.05, 0.10, 0.20, 0.30]
CALIB_SIZES = [5, 10, 20, 50, 100]
VOCAB_SIZES = [5, 8, 10, 12, 15]
RAMP_DURATIONS = [0, 50, 100, 200, 400]  # 0 → step change

DEFAULT_LAMBDA = 0.5
DEFAULT_WEIGHTS = {"SAFE": 1.0, "UNSAFE": 2.0, "CRITICAL": 5.0}
DEFAULT_ALPHA = 0.05
DEFAULT_CALIB = 20
DEFAULT_VOCAB = 15

N_MAIN = 300
N_AUX = 100
SEED = 42
OUTPUT_DIR = "results/exp2"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_truncated_vocab(k: int):
    from ide.evaluation.simulator import (
        BASELINE_CS, PHISHING_DRIFT, build_customer_service_vocab,
    )
    vocab = build_customer_service_vocab()[:k]
    b = BASELINE_CS[:k].copy()
    b /= b.sum()
    d = PHISHING_DRIFT[:k].copy()
    d /= d.sum()
    return vocab, b, d


def _run_trials(
    n_trials: int,
    rng: np.random.Generator,
    vocab,
    baseline_dist: np.ndarray,
    drift_dist: np.ndarray,
    onset: int,
    drift_type: str,
    ramp_duration: int,
    duration: int,
    embedder,
    lambda_: float,
    weights: dict,
    alpha: float,
    calib_size: int,
) -> tuple[float, float]:
    from ide.evaluation.runner import run_method_on_trace
    from ide.evaluation.simulator import AgentSimulator, DriftScenario
    from ide.ss_cbd import SSCBDDetector

    sim = AgentSimulator(vocab=vocab, seed=int(rng.integers(0, 2**31)))
    scenario = DriftScenario(
        name="sweep",
        baseline_dist=baseline_dist,
        drift_dist=drift_dist,
        onset_time=onset,
        drift_type=drift_type,
        ramp_duration=ramp_duration,
    )
    times = []
    for _ in range(n_trials):
        trace = sim.generate_trace(scenario, duration=duration, seed=int(rng.integers(0, 2**31)))
        det = SSCBDDetector(
            vocab=vocab, embedder=embedder,
            semantic_lambda=lambda_, tier_weights=weights,
            alpha=alpha, beta=alpha, n_min=calib_size,
        )
        detected, t = run_method_on_trace(
            det, trace, alt_dist=drift_dist, calibration_size=calib_size,
        )
        if detected:
            times.append(t)

    det_rate = len(times) / n_trials
    mean_t = float(np.mean(times)) if times else float("inf")
    return det_rate, mean_t


# ── Individual sweep functions ─────────────────────────────────────────────────

def sweep_lambda(seed: int = SEED) -> pd.DataFrame:
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.simulator import BASELINE_CS, PHISHING_DRIFT, build_customer_service_vocab

    print(f"  [1/6] lambda sweep (N={N_MAIN}) ...", flush=True)
    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    rng = np.random.default_rng(seed)
    rows = []
    for lam in LAMBDA_SWEEP:
        dr, mt = _run_trials(
            N_MAIN, rng, vocab, BASELINE_CS, PHISHING_DRIFT,
            onset=200, drift_type="step", ramp_duration=0, duration=600,
            embedder=embedder, lambda_=lam, weights=DEFAULT_WEIGHTS,
            alpha=DEFAULT_ALPHA, calib_size=DEFAULT_CALIB,
        )
        rows.append({"dim": "lambda", "value": lam, "det_rate": round(dr, 4), "mean_det_time": round(mt, 1)})
        print(f"    λ={lam:.2f}  det={dr*100:.1f}%  mean_t={mt:.1f}", flush=True)
    return pd.DataFrame(rows)


def sweep_tier_weights(seed: int = SEED) -> pd.DataFrame:
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.simulator import BASELINE_CS, PHISHING_DRIFT, build_customer_service_vocab

    print(f"  [2/6] tier_weights sweep (N={N_MAIN}) ...", flush=True)
    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for w in WEIGHT_SWEEP:
        label = f"u{w['UNSAFE']:.0f}_c{w['CRITICAL']:.0f}"
        dr, mt = _run_trials(
            N_MAIN, rng, vocab, BASELINE_CS, PHISHING_DRIFT,
            onset=200, drift_type="step", ramp_duration=0, duration=600,
            embedder=embedder, lambda_=DEFAULT_LAMBDA, weights=w,
            alpha=DEFAULT_ALPHA, calib_size=DEFAULT_CALIB,
        )
        rows.append({
            "dim": "tier_weights", "value": label,
            "w_unsafe": w["UNSAFE"], "w_critical": w["CRITICAL"],
            "det_rate": round(dr, 4), "mean_det_time": round(mt, 1),
        })
        print(f"    {label}  det={dr*100:.1f}%  mean_t={mt:.1f}", flush=True)
    return pd.DataFrame(rows)


def sweep_alpha_beta(seed: int = SEED) -> pd.DataFrame:
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.simulator import BASELINE_CS, PHISHING_DRIFT, build_customer_service_vocab

    print(f"  [3/6] alpha/beta sweep (N={N_AUX}) ...", flush=True)
    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    rng = np.random.default_rng(seed + 2000)
    rows = []
    for ab in ALPHA_BETA_SWEEP:
        dr, mt = _run_trials(
            N_AUX, rng, vocab, BASELINE_CS, PHISHING_DRIFT,
            onset=200, drift_type="step", ramp_duration=0, duration=600,
            embedder=embedder, lambda_=DEFAULT_LAMBDA, weights=DEFAULT_WEIGHTS,
            alpha=ab, calib_size=DEFAULT_CALIB,
        )
        rows.append({"dim": "alpha_beta", "value": ab, "det_rate": round(dr, 4), "mean_det_time": round(mt, 1)})
        print(f"    α=β={ab:.2f}  det={dr*100:.1f}%  mean_t={mt:.1f}", flush=True)
    return pd.DataFrame(rows)


def sweep_calib_size(seed: int = SEED) -> pd.DataFrame:
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.simulator import BASELINE_CS, PHISHING_DRIFT, build_customer_service_vocab

    print(f"  [4/6] calibration size sweep (N={N_AUX}) ...", flush=True)
    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    rng = np.random.default_rng(seed + 3000)
    rows = []
    for cs in CALIB_SIZES:
        dr, mt = _run_trials(
            N_AUX, rng, vocab, BASELINE_CS, PHISHING_DRIFT,
            onset=max(cs + 50, 200), drift_type="step", ramp_duration=0,
            duration=max(cs + 50, 200) + 400,
            embedder=embedder, lambda_=DEFAULT_LAMBDA, weights=DEFAULT_WEIGHTS,
            alpha=DEFAULT_ALPHA, calib_size=cs,
        )
        rows.append({"dim": "calib_size", "value": cs, "det_rate": round(dr, 4), "mean_det_time": round(mt, 1)})
        print(f"    calib={cs}  det={dr*100:.1f}%  mean_t={mt:.1f}", flush=True)
    return pd.DataFrame(rows)


def sweep_vocab_size(seed: int = SEED) -> pd.DataFrame:
    from ide.embeddings import ActionEmbedder

    print(f"  [5/6] vocab size sweep (N={N_AUX}) ...", flush=True)
    embedder = ActionEmbedder()
    rng = np.random.default_rng(seed + 4000)
    rows = []
    for k in VOCAB_SIZES:
        vocab, b, d = _build_truncated_vocab(k)
        dr, mt = _run_trials(
            N_AUX, rng, vocab, b, d,
            onset=200, drift_type="step", ramp_duration=0, duration=600,
            embedder=embedder, lambda_=DEFAULT_LAMBDA, weights=DEFAULT_WEIGHTS,
            alpha=DEFAULT_ALPHA, calib_size=DEFAULT_CALIB,
        )
        rows.append({"dim": "vocab_size", "value": k, "det_rate": round(dr, 4), "mean_det_time": round(mt, 1)})
        print(f"    vocab={k}  det={dr*100:.1f}%  mean_t={mt:.1f}", flush=True)
    return pd.DataFrame(rows)


def sweep_drift_slope(seed: int = SEED) -> pd.DataFrame:
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.simulator import BASELINE_CS, PHISHING_DRIFT, build_customer_service_vocab

    print(f"  [6/6] drift slope sweep (N={N_AUX}) ...", flush=True)
    vocab = build_customer_service_vocab()
    embedder = ActionEmbedder()
    rng = np.random.default_rng(seed + 5000)
    rows = []
    for ramp in RAMP_DURATIONS:
        drift_type = "step" if ramp == 0 else "linear_ramp"
        duration = 600 if ramp == 0 else max(600, 200 + ramp + 200)
        dr, mt = _run_trials(
            N_AUX, rng, vocab, BASELINE_CS, PHISHING_DRIFT,
            onset=200, drift_type=drift_type, ramp_duration=ramp, duration=duration,
            embedder=embedder, lambda_=DEFAULT_LAMBDA, weights=DEFAULT_WEIGHTS,
            alpha=DEFAULT_ALPHA, calib_size=DEFAULT_CALIB,
        )
        label = "step" if ramp == 0 else f"ramp_{ramp}"
        rows.append({
            "dim": "drift_slope", "value": label, "ramp_duration": ramp,
            "det_rate": round(dr, 4), "mean_det_time": round(mt, 1),
        })
        print(f"    {label}  det={dr*100:.1f}%  mean_t={mt:.1f}", flush=True)
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== EXP 2: SS-CBD Ablation Study — 6 Dimensions ===\n")

    df_lam = sweep_lambda()
    df_lam.to_csv(os.path.join(OUTPUT_DIR, "sweep_lambda.csv"), index=False)

    df_w = sweep_tier_weights()
    df_w.to_csv(os.path.join(OUTPUT_DIR, "sweep_tier_weights.csv"), index=False)

    df_ab = sweep_alpha_beta()
    df_ab.to_csv(os.path.join(OUTPUT_DIR, "sweep_alpha_beta.csv"), index=False)

    df_cs = sweep_calib_size()
    df_cs.to_csv(os.path.join(OUTPUT_DIR, "sweep_calib_size.csv"), index=False)

    df_vs = sweep_vocab_size()
    df_vs.to_csv(os.path.join(OUTPUT_DIR, "sweep_vocab_size.csv"), index=False)

    df_sl = sweep_drift_slope()
    df_sl.to_csv(os.path.join(OUTPUT_DIR, "sweep_drift_slope.csv"), index=False)

    combined = pd.concat([df_lam, df_w, df_ab, df_cs, df_vs, df_sl], ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "ablations_sscbd.csv"), index=False)

    # ── Summary report ────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Dim 1: semantic_lambda ---")
    print(df_lam[["value", "det_rate", "mean_det_time"]].to_string(index=False))

    print("\n--- Dim 2: tier_weights ---")
    print(df_w[["value", "w_unsafe", "w_critical", "det_rate", "mean_det_time"]].to_string(index=False))

    print("\n--- Dim 3: alpha/beta ---")
    print(df_ab[["value", "det_rate", "mean_det_time"]].to_string(index=False))

    print("\n--- Dim 4: calibration size ---")
    print(df_cs[["value", "det_rate", "mean_det_time"]].to_string(index=False))

    print("\n--- Dim 5: vocab size ---")
    print(df_vs[["value", "det_rate", "mean_det_time"]].to_string(index=False))

    print("\n--- Dim 6: drift slope (ramp duration) ---")
    print(df_sl[["value", "det_rate", "mean_det_time"]].to_string(index=False))

    # ── Validation check ─────────────────────────────────────────────────────

    best_lam_row = df_lam.loc[df_lam["mean_det_time"].idxmin()]
    best_w_row = df_w.loc[df_w["mean_det_time"].idxmin()]

    print(f"\nBest λ: {best_lam_row['value']} (mean_t={best_lam_row['mean_det_time']:.1f})")
    print(f"Best weights: {best_w_row['value']} (mean_t={best_w_row['mean_det_time']:.1f})")

    if str(best_lam_row["value"]) == "0.5":
        print("✓ Default λ=0.5 confirmed optimal.")
    else:
        print(f"⚠ Optimal λ={best_lam_row['value']} differs from default 0.5.")

    if best_w_row["value"] == "u2_c5":
        print("✓ Default weights (u=2, c=5) confirmed optimal.")
    else:
        print(f"⚠ Optimal weights differ from default: {best_w_row['value']}")


if __name__ == "__main__":
    main()
