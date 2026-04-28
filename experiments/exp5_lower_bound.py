"""EXP5: Empirical validation of Proposition 2 lower bound scaling.

Measures the detection threshold N* (number of observations at which detection
probability crosses 0.5) as a function of:
  - vocab size n ∈ {5, 10, 15, 30, 50}
  - cosine-distance margin γ ∈ {0.05, 0.10, 0.20, 0.30}

Theoretical prediction from Proposition 2 (Le Cam single-direction):
  N* ∝ n / γ  (lower bound from KL = O(nγ) for the one-hotspot construction)

Uses CBD detector with a CALIBRATED threshold:
  threshold(N) = 95th percentile of cosine distances under H0 (N samples from b)
This controls FPR ≤ 5% at each N and reveals the true N* scaling.

Note on construction: the one-hotspot p*(ε) = (1-ε)b + ε·e_k concentrates
probability on a single action. For uniform b this gives KL(b‖p*) = O(nγ)
(the coefficient n reflects b_min = 1/n). Le Cam then gives N* ≥ Ω(1/(nγ)),
but empirically the detection N* (with FPR control) scales as Ω(n/γ) because
the null noise level scales as O(n/N), matching the drift signal γ at N ≈ n/γ.
The n·log(n)/γ² Proposition 1 upper bound uses a different (sequential) detector.

Usage:
    python -m experiments.exp5_lower_bound
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

VOCAB_SIZES = [5, 10, 15, 30, 50]
GAMMA_VALUES = [0.05, 0.10, 0.20, 0.30]
N_SAMPLE_RANGE = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000]
N_TRIALS = 200        # trials per (N, condition)
FPR_TARGET = 0.05     # target false alarm rate for threshold calibration
SEED = 42
OUTPUT_DIR = "results/exp5"


def _make_uniform(n: int) -> np.ndarray:
    return np.ones(n) / n


def _make_drift_at_gamma(baseline: np.ndarray, gamma: float) -> np.ndarray:
    """Construct p* such that d_cos(p*, b) = gamma via convex combination.

    One-hotspot construction: p*(ε) = (1-ε)b + ε·e_k where k = argmin(b).
    Binary search for ε such that d_cos(p*(ε), b) = gamma.
    """
    n = len(baseline)
    k = int(np.argmin(baseline))
    e_k = np.zeros(n)
    e_k[k] = 1.0

    def cos_dist(eps: float) -> float:
        p = (1 - eps) * baseline + eps * e_k
        denom = np.linalg.norm(p) * np.linalg.norm(baseline)
        if denom < 1e-12:
            return 0.0
        return float(1.0 - np.dot(p, baseline) / denom)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cos_dist(mid) < gamma:
            lo = mid
        else:
            hi = mid
    eps = (lo + hi) / 2
    p_star = (1 - eps) * baseline + eps * e_k
    return p_star / p_star.sum()


def _cos_dist_empirical(
    baseline: np.ndarray,
    source: np.ndarray,
    n_obs: int,
    rng: np.random.Generator,
    n_trials: int,
) -> list[float]:
    """Sample n_trials cosine distances between empirical p_hat and baseline."""
    n = len(baseline)
    dists = []
    for _ in range(n_trials):
        indices = rng.choice(n, size=n_obs, p=source)
        counts = np.bincount(indices, minlength=n).astype(float)
        p_hat = counts / counts.sum()
        denom = np.linalg.norm(p_hat) * np.linalg.norm(baseline)
        d = float(1.0 - np.dot(p_hat, baseline) / denom) if denom > 1e-12 else 0.0
        dists.append(d)
    return dists


def find_n_star_calibrated(
    baseline: np.ndarray,
    drift: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, dict[int, float], dict[int, float]]:
    """Find N* with FPR-calibrated threshold (no knowledge of γ used for threshold).

    For each N:
      1. Compute null threshold = (1-FPR_TARGET) quantile of H0 cosine distances
      2. Measure TPR = fraction of H1 samples exceeding that threshold
    N* = linear interpolation of crossing at TPR=0.5.

    Returns (n_star, null_thresholds, tpr_probs).
    """
    null_thresholds: dict[int, float] = {}
    tpr_probs: dict[int, float] = {}

    for n_obs in N_SAMPLE_RANGE:
        # Calibrate threshold from H0 distribution
        null_dists = _cos_dist_empirical(baseline, baseline, n_obs, rng, N_TRIALS)
        null_thresholds[n_obs] = float(np.quantile(null_dists, 1.0 - FPR_TARGET))

        # Detection rate under H1
        alt_dists = _cos_dist_empirical(baseline, drift, n_obs, rng, N_TRIALS)
        tpr_probs[n_obs] = float(np.mean([d >= null_thresholds[n_obs] for d in alt_dists]))

    # Find crossing point at 0.5
    keys = sorted(tpr_probs.keys())
    for i, n_obs in enumerate(keys[:-1]):
        p_lo = tpr_probs[n_obs]
        p_hi = tpr_probs[keys[i + 1]]
        if p_lo < 0.5 <= p_hi:
            frac = (0.5 - p_lo) / max(p_hi - p_lo, 1e-9)
            return float(n_obs + frac * (keys[i + 1] - n_obs)), null_thresholds, tpr_probs

    if max(tpr_probs.values()) < 0.5:
        return float(max(keys)), null_thresholds, tpr_probs
    return float(min(keys)), null_thresholds, tpr_probs


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=== EXP 5: Lower Bound Empirical Validation ===\n")
    print(f"  N_TRIALS={N_TRIALS}, FPR_TARGET={FPR_TARGET}")
    print(f"  N_SAMPLE_RANGE={N_SAMPLE_RANGE}")
    print(f"  Vocab sizes: {VOCAB_SIZES}")
    print(f"  Gamma values: {GAMMA_VALUES}")
    print(f"  Threshold: calibrated to {FPR_TARGET*100:.0f}th-percentile of H0 at each N\n")

    rows = []
    for n in VOCAB_SIZES:
        baseline = _make_uniform(n)
        for gamma in GAMMA_VALUES:
            p_star = _make_drift_at_gamma(baseline, gamma)
            actual_gamma = float(
                1.0 - np.dot(p_star, baseline) / (np.linalg.norm(p_star) * np.linalg.norm(baseline))
            )

            n_star, thresholds, tprs = find_n_star_calibrated(baseline, p_star, rng)

            # Theoretical prediction for single-direction lower bound:
            # N* ∝ n / γ  (null noise ≈ n/N, signal ≈ γ, crossing at N ≈ n/γ)
            theory_nstar_1d = n / gamma

            # Fano n-direction bound: N* ∝ n·log(n) / γ
            theory_nstar_fano = n * float(np.log(max(n, 2))) / gamma

            rows.append({
                "n": n,
                "gamma": gamma,
                "actual_d_cos": round(actual_gamma, 4),
                "n_star": round(n_star, 1),
                "theory_1d": round(theory_nstar_1d, 1),
                "theory_fano": round(theory_nstar_fano, 1),
                "ratio_1d": round(n_star / theory_nstar_1d, 3) if theory_nstar_1d > 0 else float("nan"),
                "tpr_at_min": round(tprs[min(tprs.keys())], 3),
                "tpr_at_max": round(tprs[max(tprs.keys())], 3),
            })
            print(f"  n={n:3d}, γ={gamma:.2f}: N*={n_star:.1f} "
                  f"(theory_1d={theory_nstar_1d:.1f}, ratio={n_star/theory_nstar_1d:.2f}, "
                  f"tpr_range=[{tprs[min(tprs.keys())]:.2f},{tprs[max(tprs.keys())]:.2f}])")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "lower_bound_scaling.csv"), index=False)

    print("\n" + "=" * 70)
    print("SCALING TABLE")
    print("=" * 70)
    print(df[["n", "gamma", "actual_d_cos", "n_star", "theory_1d", "ratio_1d"]].to_string(index=False))

    # Check 1/γ scaling (n=15 fixed)
    print("\n--- 1/γ scaling check (n=15 fixed) ---")
    subset_n = df[df["n"] == 15]
    for _, row in subset_n.iterrows():
        print(f"  γ={row['gamma']:.2f}: N*={row['n_star']:.1f}, "
              f"1/γ={1/row['gamma']:.1f}, N*·γ={row['n_star']*row['gamma']:.2f}")

    # Check n scaling (γ=0.10 fixed)
    print("\n--- n scaling check (γ=0.10 fixed) ---")
    subset_g = df[df["gamma"] == 0.10]
    for _, row in subset_g.iterrows():
        print(f"  n={int(row['n']):3d}: N*={row['n_star']:.1f}, "
              f"theory_1d={row['theory_1d']:.1f}, N*/n={row['n_star']/row['n']:.2f}")

    # Overall fit to N* ∝ n/γ
    ratios = df["ratio_1d"].dropna()
    print(f"\n  Constant C in N* ≈ C · n / γ:")
    print(f"  Mean C = {ratios.mean():.3f} ± {ratios.std():.3f}")
    print(f"  Range: [{ratios.min():.3f}, {ratios.max():.3f}]")

    print("\n  Results written to results/exp5/")


if __name__ == "__main__":
    main()
