"""EXP6: Statistical significance analysis — bootstrap CIs on speedup ratios,
McNemar's test (SS-CBD vs LRT), and N=500 control/drift evaluation for Section 7.3.

Usage:
    python -m experiments.exp6_significance
    python -m experiments.exp6_significance --n_trials 500 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

N_TRIALS = 500
SEED = 42
OUTPUT_DIR = "results/exp6"
DRIFT_SCENARIOS = ["phishing_step", "phishing_gradual", "financial_step", "financial_gradual"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument(
        "--reuse_exp1",
        action="store_true",
        default=False,
        help="Reuse exp1 raw_results.csv instead of re-running",
    )
    return parser.parse_args()


def _bootstrap_speedup_ci(
    sscbd_times: np.ndarray,
    lrt_times: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on the speedup = mean(LRT_times) / mean(SS-CBD_times).

    Uses the ratio-of-means estimator (consistent with the paper's reported
    speedup statistic). Bootstrap resamples paired observations and recomputes
    the ratio of means on each resample.

    Returns (speedup, ci_lo, ci_hi).
    """
    assert len(sscbd_times) == len(lrt_times), "Paired trials required"
    n = len(sscbd_times)
    rng = np.random.default_rng(seed)
    # Point estimate
    speedup = float(np.mean(lrt_times)) / float(np.mean(sscbd_times) + 1e-9)
    # Bootstrap replicates of the ratio-of-means
    boot_speedups = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_mean = float(np.mean(sscbd_times[idx]) + 1e-9)
        l_mean = float(np.mean(lrt_times[idx]))
        boot_speedups.append(l_mean / s_mean)
    ci_lo = float(np.percentile(boot_speedups, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_speedups, 100 * (1 - alpha / 2)))
    return speedup, ci_lo, ci_hi


def _mcnemar_test(
    sscbd_detected: np.ndarray,
    lrt_detected: np.ndarray,
) -> tuple[float, float]:
    """McNemar's test for paired proportions.

    H0: P(SS-CBD detects) = P(LRT detects)

    Returns (chi2, p_value). Uses continuity-corrected version.
    """
    a = int(((sscbd_detected == 1) & (lrt_detected == 1)).sum())  # both detect
    b = int(((sscbd_detected == 1) & (lrt_detected == 0)).sum())  # SS-CBD only
    c = int(((sscbd_detected == 0) & (lrt_detected == 1)).sum())  # LRT only
    d = int(((sscbd_detected == 0) & (lrt_detected == 0)).sum())  # neither
    _ = a, d  # not used in statistic
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p = float(1 - chi2_dist.cdf(chi2, df=1))
    return float(chi2), p


def _wilcoxon_detection_times(
    sscbd_times: np.ndarray,
    lrt_times: np.ndarray,
) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired detection times (detected trials only).

    H0: no difference in detection time distribution.
    Returns (statistic, p_value).
    """
    mask = (sscbd_times < np.inf) & (lrt_times < np.inf)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(sscbd_times[mask], lrt_times[mask], alternative="less")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def run_significance_analysis(df: pd.DataFrame) -> dict:
    """Compute all significance statistics from a per-trial DataFrame."""
    results = {}

    for scenario in DRIFT_SCENARIOS:
        sub = df[df["scenario"] == scenario].copy()
        sscbd = sub[sub["method"] == "SS-CBD"].sort_values("trial")
        lrt = sub[sub["method"] == "LRT"].sort_values("trial")

        if len(sscbd) == 0 or len(lrt) == 0:
            continue

        n = min(len(sscbd), len(lrt))
        sscbd = sscbd.iloc[:n]
        lrt = lrt.iloc[:n]

        sscbd_det = sscbd["detected"].values.astype(int)
        lrt_det = lrt["detected"].values.astype(int)
        sscbd_t = np.where(sscbd["detected"].values, sscbd["detection_time"].values.astype(float), np.inf)
        lrt_t = np.where(lrt["detected"].values, lrt["detection_time"].values.astype(float), np.inf)

        mean_sp, ci_lo, ci_hi = _bootstrap_speedup_ci(
            np.where(sscbd_t < np.inf, sscbd_t, 9999.0),
            np.where(lrt_t < np.inf, lrt_t, 9999.0),
            seed=SEED,
        )
        chi2, p_mcnemar = _mcnemar_test(sscbd_det, lrt_det)
        wilcox_stat, p_wilcox = _wilcoxon_detection_times(sscbd_t, lrt_t)

        results[scenario] = {
            "n": n,
            "sscbd_det_rate": float(sscbd_det.mean()),
            "lrt_det_rate": float(lrt_det.mean()),
            "mean_speedup": mean_sp,
            "speedup_ci_lo": ci_lo,
            "speedup_ci_hi": ci_hi,
            "mcnemar_chi2": chi2,
            "mcnemar_p": p_mcnemar,
            "wilcoxon_stat": wilcox_stat,
            "wilcoxon_p": p_wilcox,
        }

    return results


def run_n500_control_eval(
    n_trials: int,
    seed: int,
    output_dir: str,
) -> pd.DataFrame:
    """Run N=500 control + drift scenarios (analogous to paper Section 7.3 scenarios A/B/C)
    using the full method suite. Produces Table 1 replacement with proper CIs."""
    from ide.evaluation.runner import _fresh_method, run_method_on_trace
    from ide.evaluation.simulator import AgentSimulator, build_customer_service_vocab, build_standard_scenarios
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.metrics import clopper_pearson_ci

    vocab = build_customer_service_vocab()
    scenarios = build_standard_scenarios()
    embedder = ActionEmbedder()
    simulator = AgentSimulator(vocab=vocab, seed=seed)
    rng = np.random.default_rng(seed)

    METHODS_7_3 = ["SS-CBD", "LRT", "CBD", "Hellinger"]
    rows = []

    for scenario in scenarios:
        print(f"  EXP6 Sec7.3 scenario: {scenario.name} ...", flush=True)
        trial_seeds = rng.integers(0, 2**31, size=n_trials)
        for trial_idx, trial_seed in enumerate(trial_seeds):
            trace = simulator.generate_trace(scenario, duration=4000, seed=int(trial_seed))
            for mname in METHODS_7_3:
                method = _fresh_method(mname, vocab, embedder)
                detected, det_time = run_method_on_trace(method, trace, alt_dist=scenario.drift_dist)
                rows.append({
                    "method": mname,
                    "scenario": scenario.name,
                    "trial": trial_idx,
                    "detected": detected,
                    "detection_time": det_time,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "sec73_n500_raw.csv"), index=False)

    records = []
    for (method, scenario), grp in df.groupby(["method", "scenario"]):
        n = len(grp)
        detected = int(grp["detected"].sum())
        det_rate = detected / n
        ci_lo, ci_hi = clopper_pearson_ci(detected, n)
        detected_times = grp.loc[grp["detected"], "detection_time"]
        mdt = float(detected_times.mean()) if len(detected_times) > 0 else float("inf")
        mdt_std = float(detected_times.std()) if len(detected_times) > 1 else 0.0
        records.append({
            "method": method,
            "scenario": scenario,
            "n_trials": n,
            "det_rate": round(det_rate, 4),
            "det_rate_ci_lo": round(ci_lo, 4),
            "det_rate_ci_hi": round(ci_hi, 4),
            "mean_det_time": round(mdt, 1),
            "std_det_time": round(mdt_std, 1),
        })

    summary = pd.DataFrame(records)
    summary.to_csv(os.path.join(output_dir, "sec73_n500_summary.csv"), index=False)
    return summary


def _write_results_md(
    sig: dict,
    summary_73: pd.DataFrame,
    output_dir: str,
) -> None:
    lines = ["# EXP6: Statistical Significance Analysis\n"]
    lines.append(f"**N={N_TRIALS} trials, seed={SEED}. All tests two-sided unless noted.**\n")

    lines.append("\n## Table A: Bootstrap 95% CIs on Speedup Ratio (SS-CBD / LRT)\n")
    lines.append("| Scenario | Speedup | 95% CI | McNemar χ² | p (McNemar) | Wilcoxon p |")
    lines.append("|----------|---------|--------|------------|-------------|-----------|")
    for sc, r in sig.items():
        ci = f"[{r['speedup_ci_lo']:.2f}, {r['speedup_ci_hi']:.2f}]"
        p_mc = f"{r['mcnemar_p']:.3f}" if r["mcnemar_p"] < 1 else ">0.99"
        p_wc = f"{r['wilcoxon_p']:.3f}" if not np.isnan(r["wilcoxon_p"]) else "N/A"
        sig_mc = " ✓" if r["mcnemar_p"] < 0.05 else ""
        lines.append(
            f"| {sc} | {r['mean_speedup']:.2f}× | {ci} | {r['mcnemar_chi2']:.2f}{sig_mc} | {p_mc} | {p_wc} |"
        )

    lines.append("\n**Interpretation:** ✓ = significant difference at α=0.05.")
    lines.append("McNemar tests H0: P(SS-CBD detects) = P(LRT detects).")
    lines.append("Wilcoxon tests H0: no difference in detection time (one-sided, SS-CBD faster).\n")

    lines.append("\n## Table B: N=500 Detection Performance (Section 7.3 Replacement)\n")
    lines.append("| Scenario | Method | Det% (95% CI) | Mean Det Time (±std) |")
    lines.append("|----------|--------|---------------|----------------------|")
    for _, row in summary_73.sort_values(["scenario", "method"]).iterrows():
        if row["scenario"] == "control":
            ci_str = f"{row['det_rate']*100:.1f}% [{row['det_rate_ci_lo']*100:.1f}%, {row['det_rate_ci_hi']*100:.1f}%]"
            lines.append(f"| {row['scenario']} | {row['method']} | FPR {ci_str} | — |")
        else:
            ci_str = f"{row['det_rate']*100:.1f}% [{row['det_rate_ci_lo']*100:.1f}%, {row['det_rate_ci_hi']*100:.1f}%]"
            t_str = f"{row['mean_det_time']:.1f} ± {row['std_det_time']:.1f}"
            lines.append(f"| {row['scenario']} | {row['method']} | {ci_str} | {t_str} |")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "RESULTS_EXP6.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {output_dir}/RESULTS_EXP6.md")


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    exp1_csv = "results/exp1/raw_results.csv"
    if args.reuse_exp1 and os.path.exists(exp1_csv):
        print(f"Reusing {exp1_csv} for significance analysis ...")
        df_exp1 = pd.read_csv(exp1_csv)
    else:
        print(f"Running EXP1-style N={args.n_trials} trials for significance analysis ...")
        from ide.evaluation.runner import headline_experiment
        df_exp1 = headline_experiment(
            n_trials=args.n_trials, seed=args.seed, output_dir="results/exp1"
        )

    print("Computing significance statistics ...")
    sig = run_significance_analysis(df_exp1)

    print("Printing significance table:\n")
    print(f"{'Scenario':<25} {'Speedup':>8} {'95% CI':<20} {'McNemar p':>12} {'Wilcoxon p':>12}")
    print("─" * 80)
    for sc, r in sig.items():
        ci = f"[{r['speedup_ci_lo']:.2f}, {r['speedup_ci_hi']:.2f}]"
        p_mc = f"{r['mcnemar_p']:.3f}"
        p_wc = f"{r['wilcoxon_p']:.3f}" if not np.isnan(r["wilcoxon_p"]) else "N/A"
        print(f"{sc:<25} {r['mean_speedup']:>7.2f}× {ci:<20} {p_mc:>12} {p_wc:>12}")

    print("\nRunning N=500 Section-7.3 replacement evaluation ...")
    summary_73 = run_n500_control_eval(
        n_trials=args.n_trials, seed=args.seed, output_dir=args.output_dir
    )

    _write_results_md(sig, summary_73, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
