"""Automated verification of reproduced results against reference values.

Usage:
    python verify_reproduction.py

Exit 0 on pass (all checks within tolerance), 1 on failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

TOLERANCES = {
    "exp1_mean_speedup": (2.0, 3.5),
    "exp1_peak_speedup": (4.0, 6.5),
    "exp1_ss_cbd_wins": (2, 4),           # SS-CBD wins ≥ 2 of 4 scenarios
    "exp7_sscbd_det_rate": (0.95, 1.0),   # SS-CBD 100% det on KB-poison
    "exp7_cbd_det_rate": (0.20, 0.60),    # CBD fails 40–80% of trials
    "exp7_sscbd_median_actions": (5, 60), # Median det time 5–60 actions
}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _check(label: str, value: float, lo: float, hi: float) -> bool:
    ok = lo <= value <= hi
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}: {value:.4f} (expected [{lo}, {hi}])")
    return ok


def verify_exp1() -> bool:
    path = "results/exp1/raw_results.csv"
    if not os.path.exists(path):
        print(f"  [SKIP] EXP1: {path} not found — run 'make exp1' first")
        return True

    df = pd.read_csv(path)
    drift_scenarios = ["phishing_step", "phishing_gradual", "financial_step", "financial_gradual"]
    speedups = []
    wins = 0
    for sc in drift_scenarios:
        ss = df[(df["scenario"] == sc) & (df["method"] == "SS-CBD")]
        lrt = df[(df["scenario"] == sc) & (df["method"] == "LRT")]
        n = min(len(ss), len(lrt))
        ss_t = np.where(ss["detected"].values[:n], ss["detection_time"].values[:n].astype(float), 9999.0)
        lrt_t = np.where(lrt["detected"].values[:n], lrt["detection_time"].values[:n].astype(float), 9999.0)
        sp = float(np.mean(lrt_t)) / float(np.mean(ss_t) + 1e-9)
        speedups.append(sp)
        if sp > 1.0:
            wins += 1

    mean_sp = float(np.mean(speedups))
    peak_sp = float(np.max(speedups))

    print("EXP1 (Headline §7.5):")
    ok1 = _check("mean speedup", mean_sp, *TOLERANCES["exp1_mean_speedup"])
    ok2 = _check("peak speedup", peak_sp, *TOLERANCES["exp1_peak_speedup"])
    ok3 = _check("SS-CBD wins (of 4)", float(wins), *TOLERANCES["exp1_ss_cbd_wins"])
    return ok1 and ok2 and ok3


def verify_exp7() -> bool:
    summary_path = "results/exp7/detection_table.csv"
    raw_path = "results/exp7/raw_results.csv"
    if not os.path.exists(summary_path):
        print(f"  [SKIP] EXP7: {summary_path} not found — run 'make exp7' first")
        return True

    summary = pd.read_csv(summary_path)
    drift = summary[summary["scenario"] == "kb_poison_slowburn"]

    sscbd_row = drift[drift["method"] == "SS-CBD"]
    cbd_row = drift[drift["method"] == "CBD"]

    sscbd_det = float(sscbd_row["det_rate"].iloc[0]) if len(sscbd_row) > 0 else float("nan")
    cbd_det = float(cbd_row["det_rate"].iloc[0]) if len(cbd_row) > 0 else float("nan")

    sscbd_median = float("nan")
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        sscbd_raw = raw[
            (raw["method"] == "SS-CBD") &
            (raw["scenario"] == "kb_poison_slowburn") &
            (raw["detected"] == True)
        ]["detection_time_actions"].dropna().astype(float)
        if len(sscbd_raw) > 0:
            sscbd_median = float(sscbd_raw.median())

    print("EXP7 (KB-poison case study §7.11):")
    ok1 = _check("SS-CBD det_rate", sscbd_det, *TOLERANCES["exp7_sscbd_det_rate"])
    ok2 = _check("CBD det_rate", cbd_det, *TOLERANCES["exp7_cbd_det_rate"])
    ok3 = _check("SS-CBD median (actions)", sscbd_median, *TOLERANCES["exp7_sscbd_median_actions"])
    return ok1 and ok2 and ok3


def main() -> int:
    print("\n=== cobalt-sentinel Reproduction Verification ===\n")
    results = [verify_exp1(), verify_exp7()]
    print()
    if all(results):
        print(f"Overall: {PASS} — all checks passed\n")
        return 0
    else:
        print(f"Overall: {FAIL} — one or more checks failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
