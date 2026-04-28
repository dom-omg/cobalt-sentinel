"""Headline experiment: SS-CBD vs all baselines on all drift scenarios.

Usage:
    python -m experiments.exp1_headline
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

N_TRIALS = 500
SEED = 42
OUTPUT_DIR = "results/exp1"
DRIFT_SCENARIOS = ["phishing_step", "phishing_gradual", "financial_step", "financial_gradual"]


def _print_table(summary: pd.DataFrame, lrt_times: dict[str, float]) -> bool:
    """Print formatted results table. Returns True if paper validation passes."""
    print("\n=== HEADLINE EXPERIMENT (N=500 trials, seed=42) ===\n")
    header = f"{'Method':<12} {'Scenario':<25} {'Det%':>6} {'MeanTime':>10} {'Speedup':>10} {'FPR (95% CI)':<20}"
    print(header)
    print("─" * len(header))

    drift_only = summary[summary["scenario"] != "control"].copy()
    control = summary[summary["scenario"] == "control"].copy()

    for _, row in drift_only.sort_values(["scenario", "method"]).iterrows():
        ref_time = lrt_times.get(str(row["scenario"]), float("inf"))
        speedup = ref_time / row["mean_det_time"] if row["mean_det_time"] > 0 else float("nan")
        det_pct = f"{row['det_rate'] * 100:.1f}"
        mean_t = f"{row['mean_det_time']:.1f}" if row["mean_det_time"] != float("inf") else "∞"
        sp_str = f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A"
        print(f"{row['method']:<12} {row['scenario']:<25} {det_pct:>6} {mean_t:>10} {sp_str:>10}")

    print()
    if not control.empty:
        print("FPR (control scenario):")
        for _, row in control.iterrows():
            ci_lo = row["ci_lo"]
            ci_hi = row["ci_hi"]
            fpr_pct = row["fpr"] * 100 if not np.isnan(row["fpr"]) else float("nan")
            ci_lo_pct = ci_lo * 100
            ci_hi_pct = ci_hi * 100
            print(f"  {row['method']:<12} {fpr_pct:5.1f}% [{ci_lo_pct:.1f}–{ci_hi_pct:.1f}]")

    # Calculate speedup across drift scenarios
    sscbd_times = drift_only[drift_only["method"] == "SS-CBD"]["mean_det_time"].values
    lrt_mean_times = [
        lrt_times.get(s, float("inf"))
        for s in drift_only[drift_only["method"] == "SS-CBD"]["scenario"].values
    ]
    speedups = [
        lt / st for lt, st in zip(lrt_mean_times, sscbd_times)
        if st > 0 and lt < float("inf")
    ]
    avg_speedup = float(np.mean(speedups)) if speedups else 0.0

    sscbd_wins = sum(
        1 for lt, st in zip(lrt_mean_times, sscbd_times)
        if st > 0 and lt > st
    )
    n_drift = len(DRIFT_SCENARIOS)

    print(f"\n>>> SS-CBD speedup vs LRT (averaged across drift scenarios): {avg_speedup:.2f}x")
    print(f">>> SS-CBD wins on {sscbd_wins}/{n_drift} drift scenarios at matched FPR")

    passes = avg_speedup >= 1.5 and sscbd_wins >= 3
    if passes:
        print(">>> PAPER VALIDATION: PASS")
    else:
        print(">>> PAPER VALIDATION: FAIL — investigate before re-running.")
        print("    DO NOT modify hyperparameters to force success.")

    return passes


def main():
    from ide.evaluation.runner import headline_experiment

    print(f"Running headline experiment: N={N_TRIALS} trials, seed={SEED}")
    print("This will take a few minutes...\n")

    df = headline_experiment(n_trials=N_TRIALS, seed=SEED, output_dir=OUTPUT_DIR)

    summary_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
    summary = pd.read_csv(summary_path)

    lrt_rows = summary[(summary["method"] == "LRT") & (summary["scenario"] != "control")]
    lrt_times = dict(zip(lrt_rows["scenario"], lrt_rows["mean_det_time"]))

    passes = _print_table(summary, lrt_times)

    _plot_figures(df, OUTPUT_DIR)

    _write_results_md(summary, lrt_times, passes, OUTPUT_DIR)

    return 0 if passes else 1


def _plot_figures(df: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Detection time distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        drift_df = df[df["scenario"] == "phishing_step"]
        for method in df["method"].unique():
            times = drift_df[drift_df["method"] == method]["detection_time"].dropna()
            axes[0].hist(times, alpha=0.5, label=method, bins=30)
        axes[0].set_xlabel("Detection Time (actions post-calibration)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Detection Time Distribution — phishing_step")
        axes[0].legend(fontsize=7)

        # Speedup bar chart
        methods = [m for m in df["method"].unique() if m != "LRT"]
        scenarios = ["phishing_step", "phishing_gradual", "financial_step", "financial_gradual"]
        speedups_mean = []
        for m in methods:
            sp = []
            for s in scenarios:
                lrt_t = df[(df["method"] == "LRT") & (df["scenario"] == s)]["detection_time"].mean()
                m_t = df[(df["method"] == m) & (df["scenario"] == s)]["detection_time"].mean()
                if m_t > 0:
                    sp.append(lrt_t / m_t)
            speedups_mean.append(np.mean(sp) if sp else 0)
        axes[1].bar(methods, speedups_mean)
        axes[1].axhline(1.0, color="red", linestyle="--", label="LRT baseline")
        axes[1].axhline(1.5, color="green", linestyle="--", label="1.5x target")
        axes[1].set_ylabel("Speedup vs LRT")
        axes[1].set_title("Mean Speedup vs LRT (averaged across drift scenarios)")
        axes[1].legend()
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_detection_time_dist.png"), dpi=150)
        plt.close()

    except Exception as e:
        print(f"  [warn] Figures skipped: {e}")


def _write_results_md(
    summary: pd.DataFrame,
    lrt_times: dict[str, float],
    passes: bool,
    output_dir: str,
) -> None:
    lines = ["# SS-CBD Headline Experiment Results\n"]
    lines.append(f"**Paper Validation: {'PASS ✅' if passes else 'FAIL ❌'}**\n")

    drift = summary[summary["scenario"] != "control"]
    sscbd = drift[drift["method"] == "SS-CBD"]
    lrt = drift[drift["method"] == "LRT"]

    lines.append("\n## Detection Time Speedup (SS-CBD vs LRT)\n")
    lines.append("| Scenario | SS-CBD MeanTime | LRT MeanTime | Speedup |")
    lines.append("|---|---|---|---|")
    for _, row in sscbd.iterrows():
        lrt_t = lrt[lrt["scenario"] == row["scenario"]]["mean_det_time"].values
        lrt_t = float(lrt_t[0]) if len(lrt_t) > 0 else float("inf")
        sp = lrt_t / row["mean_det_time"] if row["mean_det_time"] > 0 else float("nan")
        lines.append(f"| {row['scenario']} | {row['mean_det_time']:.1f} | {lrt_t:.1f} | {sp:.2f}x |")

    lines.append("\n## Key Numbers for Paper\n")
    speedups = []
    for _, row in sscbd.iterrows():
        lrt_t = lrt_times.get(row["scenario"], float("inf"))
        if row["mean_det_time"] > 0:
            speedups.append(lrt_t / row["mean_det_time"])
    if speedups:
        lines.append(f"- Mean speedup across scenarios: **{np.mean(speedups):.2f}x**")
        lines.append(f"- Min speedup: **{min(speedups):.2f}x**")
        lines.append(f"- Max speedup: **{max(speedups):.2f}x**")

    lines.append("\n## Recommendation\n")
    if passes:
        lines.append("SS-CBD beats LRT at the required 1.5x threshold. Default hyperparameters (λ=0.5, w_unsafe=2, w_critical=5) are recommended.")
    else:
        lines.append("SS-CBD does NOT meet the 1.5x threshold. Do NOT tune hyperparameters to force success. Investigate signal quality and report honest results.")

    with open(os.path.join(output_dir, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
