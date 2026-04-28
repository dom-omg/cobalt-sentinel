"""EXP4: Quasi-real agent trace evaluation.

Evaluates SS-CBD and baselines on traces with realistic statistical structure:
  - Markov-chain temporal autocorrelation
  - Multi-modal behavioral regime switching
  - Lognormal action probability noise

Two agent archetypes:
  - AutoGPT-style: web search, file IO, code execution
  - Open Interpreter-style: code-heavy, file IO, shell control

Drift injection follows the Section 6.2 protocol:
  - Injection fraction α ∈ {0.0, 0.1, 0.3, 0.5, 1.0}  (0.0 = FPR measure)
  - Injection onset at 0.5 × trace length

Honest note: This evaluation uses quasi-real traces (synthetic with realistic
statistical properties), not actual production agent logs. Real-trace evaluation
following the Section 6.2 protocol requires labeled operator-consented logs;
that evaluation is pending. These results validate SS-CBD beyond pure i.i.d.
assumptions and assess robustness to autocorrelation and regime switching.

Usage:
    python -m experiments.exp4_real_traces
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

N_TRIALS = 200
SEED = 42
OUTPUT_DIR = "results/exp4"
TRACE_DURATION = 800
CALIB_SIZE = 50
INJECTION_FRACTIONS = [0.0, 0.1, 0.3, 0.5, 1.0]  # 0.0 = no drift (FPR measure)


def run_evaluation(
    archetype: str,
    n_trials: int = N_TRIALS,
    seed: int = SEED,
) -> pd.DataFrame:
    from ide.baselines.cbd import CBDDetector
    from ide.baselines.hellinger import HellingerDetector
    from ide.baselines.lrt import LRTDetector
    from ide.embeddings import ActionEmbedder
    from ide.evaluation.runner import run_method_on_trace
    from ide.real_traces.quasi_real import build_autogpt_generator, build_oi_generator
    from ide.ss_cbd import SSCBDDetector

    if archetype == "autogpt":
        gen, baseline, drift_dist = build_autogpt_generator(seed=seed)
    elif archetype == "open_interpreter":
        gen, baseline, drift_dist = build_oi_generator(seed=seed)
    else:
        raise ValueError(f"Unknown archetype: {archetype}")

    vocab = gen.vocab
    embedder = ActionEmbedder()
    rng = np.random.default_rng(seed)
    onset = TRACE_DURATION // 2

    rows = []
    for inj_frac in INJECTION_FRACTIONS:
        label = f"α={inj_frac:.1f}"
        print(f"    {label} ...", end="", flush=True)
        for method_name, make_method in [
            ("SS-CBD",    lambda: SSCBDDetector(vocab=vocab, embedder=embedder, alpha=0.05, beta=0.05, n_min=CALIB_SIZE)),
            ("LRT",       lambda: LRTDetector(vocab=vocab, alpha=0.05, window_size=20)),
            ("CBD",       lambda: CBDDetector(vocab=vocab, window_size=20)),
            ("Hellinger", lambda: HellingerDetector(vocab=vocab, alpha=0.05, window_size=20)),
        ]:
            times: list[int] = []
            detections = 0
            for _ in range(n_trials):
                trace = gen.generate_trace(
                    duration=TRACE_DURATION,
                    onset_time=onset,
                    drift_dist=drift_dist,
                    drift_fraction=inj_frac,
                    seed=int(rng.integers(0, 2**31)),
                )
                method = make_method()
                detected, t = run_method_on_trace(
                    method, trace, alt_dist=drift_dist, calibration_size=CALIB_SIZE,
                )
                if detected:
                    detections += 1
                    times.append(t)

            det_rate = detections / n_trials
            mean_t = float(np.mean(times)) if times else float("inf")
            rows.append({
                "archetype": archetype,
                "method": method_name,
                "inj_frac": inj_frac,
                "det_rate": round(det_rate, 4),
                "mean_det_time": round(mean_t, 1) if mean_t != float("inf") else float("inf"),
                "n_trials": n_trials,
            })
        print(" done", flush=True)

    return pd.DataFrame(rows)


def compute_speedup_vs_lrt(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, frac), grp in df.groupby(["archetype", "inj_frac"]):
        ss_rows = grp[grp["method"] == "SS-CBD"]
        lrt_rows = grp[grp["method"] == "LRT"]
        if ss_rows.empty or lrt_rows.empty:
            continue
        ss = ss_rows["mean_det_time"].iloc[0]
        lrt = lrt_rows["mean_det_time"].iloc[0]
        if lrt > 0 and ss > 0 and lrt != float("inf") and ss != float("inf"):
            speedup = round(lrt / ss, 2)
        else:
            speedup = float("nan")
        rows.append({
            "archetype": arch, "inj_frac": frac,
            "ss_cbd_time": ss, "lrt_time": lrt, "speedup": speedup,
        })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== EXP 4: Quasi-Real Agent Trace Evaluation ===\n")
    print(f"N={N_TRIALS} trials × {len(INJECTION_FRACTIONS)} injection fractions")
    print(f"Duration={TRACE_DURATION}, calib={CALIB_SIZE}, onset={TRACE_DURATION//2}\n")

    dfs = []
    for arch in ["autogpt", "open_interpreter"]:
        print(f"--- {arch} ---", flush=True)
        df = run_evaluation(arch, n_trials=N_TRIALS, seed=SEED)
        dfs.append(df)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(OUTPUT_DIR, f"results_{arch}.csv"), index=False)
        print()

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "results_combined.csv"), index=False)

    print("=" * 60)
    print("SS-CBD SPEEDUP OVER LRT")
    print("=" * 60)
    speedup_df = compute_speedup_vs_lrt(combined)
    print(speedup_df.to_string(index=False))
    speedup_df.to_csv(os.path.join(OUTPUT_DIR, "speedup_vs_lrt.csv"), index=False)

    print("\nKEY FINDINGS:")
    fpr_df = combined[(combined["inj_frac"] == 0.0) & (combined["method"] == "SS-CBD")]
    for _, row in fpr_df.iterrows():
        print(f"  FPR [{row['archetype']}]: SS-CBD = {row['det_rate']*100:.1f}%")

    full_df = combined[(combined["inj_frac"] == 1.0) & (combined["method"] == "SS-CBD")]
    for _, row in full_df.iterrows():
        print(f"  Full drift [{row['archetype']}]: SS-CBD {row['det_rate']*100:.1f}% "
              f"(mean_t={row['mean_det_time']:.1f})")

    best = speedup_df[speedup_df["inj_frac"] > 0].dropna(subset=["speedup"]).sort_values("speedup", ascending=False).head(1)
    if not best.empty:
        r = best.iloc[0]
        print(f"  Best speedup: {r['speedup']:.2f}x [{r['archetype']}, α={r['inj_frac']}]")


if __name__ == "__main__":
    main()
