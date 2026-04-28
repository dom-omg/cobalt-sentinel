"""Experiment runner and orchestrator."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from ide.embeddings import ActionEmbedder, ActionSpec, MockEmbedder
from ide.evaluation.simulator import (
    AgentSimulator,
    build_customer_service_vocab,
    build_standard_scenarios,
)

CALIBRATION_SIZE = 50


def _accepts_alt(method: Any) -> bool:
    """Check if method.observe accepts an alt_dist kwarg."""
    import inspect

    sig = inspect.signature(method.observe)
    return "alt_dist" in sig.parameters


def _is_alert(result: dict) -> bool:
    """Extract alert/detection signal from any method's observe result."""
    decision = result.get("decision", "")
    if decision in ("DRIFT_DETECTED", "DRIFT_NEW_ACTION"):
        return True
    return bool(result.get("alert", False))


def run_method_on_trace(
    method: Any,
    trace: list[str],
    alt_dist: np.ndarray | None = None,
    calibration_size: int = CALIBRATION_SIZE,
) -> tuple[bool, int]:
    """Run a detector on a pre-generated trace.

    Args:
        method: Detector instance with calibrate() and observe() methods.
        trace: Full action trace (calibration + observation).
        alt_dist: Alternative distribution for methods that accept it (SS-CBD).
        calibration_size: Number of actions used for calibration.

    Returns:
        (detected, detection_time) where detection_time is actions post-calibration.
    """
    method.calibrate(trace[:calibration_size])
    uses_alt = _accepts_alt(method)
    for t, action in enumerate(trace[calibration_size:]):
        if uses_alt and alt_dist is not None:
            result = method.observe(action, alt_dist=alt_dist)
        else:
            result = method.observe(action)
        if _is_alert(result):
            return True, t
    return False, len(trace) - calibration_size


def _build_methods(vocab: list[ActionSpec]) -> dict[str, Any]:
    """Instantiate all methods with shared embedder."""
    from ide.baselines.adwin import ADWINDetector
    from ide.baselines.cbd import CBDDetector
    from ide.baselines.chi2 import Chi2Detector
    from ide.baselines.cusum import CUSUMDetector
    from ide.baselines.hellinger import HellingerDetector
    from ide.baselines.js_divergence import JSDivergenceDetector
    from ide.baselines.lrt import LRTDetector
    from ide.ss_cbd import SSCBDDetector

    embedder = ActionEmbedder()

    return {
        "SS-CBD": SSCBDDetector(vocab=vocab, embedder=embedder, alpha=0.05, beta=0.05),
        "LRT": LRTDetector(vocab=vocab, alpha=0.05, window_size=20),
        "CBD": CBDDetector(vocab=vocab, window_size=20),
        "Chi2": Chi2Detector(vocab=vocab, alpha=0.05, window_size=20),
        "JS": JSDivergenceDetector(vocab=vocab, alpha=0.05, window_size=20),
        "Hellinger": HellingerDetector(vocab=vocab, alpha=0.05, window_size=20),
        "ADWIN": ADWINDetector(vocab=vocab, alpha=0.05),
        "CUSUM": CUSUMDetector(vocab=vocab, alpha=0.05),
    }


def _fresh_method(name: str, vocab: list[ActionSpec], embedder: ActionEmbedder) -> Any:
    """Instantiate a single fresh method by name."""
    from ide.baselines.adwin import ADWINDetector
    from ide.baselines.cbd import CBDDetector
    from ide.baselines.chi2 import Chi2Detector
    from ide.baselines.cusum import CUSUMDetector
    from ide.baselines.hellinger import HellingerDetector
    from ide.baselines.js_divergence import JSDivergenceDetector
    from ide.baselines.lrt import LRTDetector
    from ide.ss_cbd import SSCBDDetector

    if name == "SS-CBD":
        return SSCBDDetector(vocab=vocab, embedder=embedder, alpha=0.05, beta=0.05)
    if name == "LRT":
        return LRTDetector(vocab=vocab, alpha=0.05, window_size=20)
    if name == "CBD":
        return CBDDetector(vocab=vocab, window_size=20)
    if name == "Chi2":
        return Chi2Detector(vocab=vocab, alpha=0.05, window_size=20)
    if name == "JS":
        return JSDivergenceDetector(vocab=vocab, alpha=0.05, window_size=20)
    if name == "Hellinger":
        return HellingerDetector(vocab=vocab, alpha=0.05, window_size=20)
    if name == "ADWIN":
        return ADWINDetector(vocab=vocab, alpha=0.05)
    if name == "CUSUM":
        return CUSUMDetector(vocab=vocab, alpha=0.05)
    raise ValueError(f"Unknown method: {name}")


METHOD_NAMES = ["SS-CBD", "LRT", "CBD", "Chi2", "JS", "Hellinger", "ADWIN", "CUSUM"]
TRACE_DURATION = 4000


def headline_experiment(
    n_trials: int = 500,
    seed: int = 42,
    output_dir: str = "results/exp1",
) -> pd.DataFrame:
    """Run the main benchmark: all methods × all scenarios × n_trials.

    Args:
        n_trials: Number of independent trials per (method, scenario) cell.
        seed: Master RNG seed for reproducibility.
        output_dir: Directory to write CSV outputs.

    Returns:
        DataFrame with per-trial results.
    """
    vocab = build_customer_service_vocab()
    scenarios = build_standard_scenarios()
    embedder = ActionEmbedder()
    simulator = AgentSimulator(vocab=vocab, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for scenario in scenarios:
        print(f"  scenario: {scenario.name} ...", flush=True)
        trial_seeds = rng.integers(0, 2**31, size=n_trials)

        for trial_idx, trial_seed in enumerate(trial_seeds):
            trace = simulator.generate_trace(
                scenario, duration=TRACE_DURATION, seed=int(trial_seed)
            )
            for method_name in METHOD_NAMES:
                method = _fresh_method(method_name, vocab, embedder)
                detected, det_time = run_method_on_trace(
                    method, trace, alt_dist=scenario.drift_dist
                )
                fpr_in_run = 1 if (detected and scenario.name == "control") else 0
                rows.append(
                    {
                        "method": method_name,
                        "scenario": scenario.name,
                        "trial": trial_idx,
                        "seed": int(trial_seed),
                        "detected": detected,
                        "detection_time": det_time,
                        "fpr_in_run": fpr_in_run,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)

    summary = _summarize(df)
    summary.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)

    return df


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    from ide.evaluation.metrics import clopper_pearson_ci, mean_detection_time

    records = []
    for (method, scenario), grp in df.groupby(["method", "scenario"]):
        n = len(grp)
        detected = grp["detected"].sum()
        det_rate = float(detected / n)
        ci_lo, ci_hi = clopper_pearson_ci(int(detected), n)
        mdt = mean_detection_time(grp.to_dict("records"))
        fpr = float(grp["fpr_in_run"].mean()) if scenario == "control" else float("nan")
        records.append(
            {
                "method": method,
                "scenario": scenario,
                "n_trials": n,
                "det_rate": round(det_rate, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "mean_det_time": round(mdt, 1) if mdt != float("inf") else float("inf"),
                "fpr": round(fpr, 4) if not np.isnan(fpr) else float("nan"),
            }
        )
    return pd.DataFrame(records)
