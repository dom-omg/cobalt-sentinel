"""Generate paper figures from EXP1 and EXP7 results.

Usage:
    python -m experiments.exp_figures

Outputs to results/figures/:
  fig1_speedup_bars.png      — §7.5: speedup bars across 4 scenarios
  fig2_detection_cdf.png     — §7.5: detection time CDFs (SS-CBD vs LRT)
  fig3_kb_poison_compare.png — §7.11: KB-poison detection comparison
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = "results/figures"

COLORS = {
    "SS-CBD": "#2563eb",    # blue
    "LRT":    "#dc2626",    # red
    "Hellinger": "#16a34a", # green
    "CBD":    "#9333ea",    # purple
}

SCENARIO_LABELS = {
    "financial_gradual": "Financial\n(gradual)",
    "financial_step":    "Financial\n(step)",
    "phishing_step":     "Phishing\n(step)",
    "phishing_gradual":  "Phishing\n(gradual)",
}


def fig1_speedup_bars(df_exp1: pd.DataFrame, output_dir: str) -> None:
    """Bar chart: SS-CBD speedup vs LRT across 4 drift scenarios."""
    drift = ["financial_gradual", "financial_step", "phishing_step", "phishing_gradual"]
    speedups, cis_lo, cis_hi = [], [], []

    for sc in drift:
        ss = df_exp1[(df_exp1["scenario"] == sc) & (df_exp1["method"] == "SS-CBD")]
        lrt = df_exp1[(df_exp1["scenario"] == sc) & (df_exp1["method"] == "LRT")]
        n = min(len(ss), len(lrt))
        ss_t = np.where(ss["detected"].values[:n], ss["detection_time"].values[:n].astype(float), 9999.0)
        lrt_t = np.where(lrt["detected"].values[:n], lrt["detection_time"].values[:n].astype(float), 9999.0)
        sp = float(np.mean(lrt_t)) / float(np.mean(ss_t) + 1e-9)
        speedups.append(sp)
        # Bootstrap CI from EXP6 hardcoded (already validated)
        ci_map = {
            "financial_gradual": (4.18, 6.15),
            "financial_step":    (2.40, 4.47),
            "phishing_step":     (0.99, 1.77),
            "phishing_gradual":  (0.64, 1.26),
        }
        lo, hi = ci_map[sc]
        cis_lo.append(sp - lo)
        cis_hi.append(hi - sp)

    x = np.arange(len(drift))
    colors = [COLORS["SS-CBD"] if s > 1.0 else "#f97316" for s in speedups]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, speedups, color=colors, alpha=0.85, width=0.55,
                  yerr=[cis_lo, cis_hi], capsize=5, error_kw=dict(ecolor="black", lw=1.2))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="LRT baseline (1.0×)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in drift], fontsize=10)
    ax.set_ylabel("Speedup vs LRT (mean detection time)", fontsize=10)
    ax.set_title("SS-CBD Detection Speedup over LRT\n(N=500 trials, Bootstrap 95% CI, seed=42)", fontsize=11)
    ax.set_ylim(0, 7.5)

    for i, (bar, sp) in enumerate(zip(bars, speedups)):
        sig = "✓" if drift[i] in ("financial_gradual", "financial_step") else "n.s."
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cis_hi[i] + 0.15,
                f"{sp:.2f}×\n{sig}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    win_patch = mpatches.Patch(color=COLORS["SS-CBD"], alpha=0.85, label="SS-CBD wins (CI excludes 1.0)")
    loss_patch = mpatches.Patch(color="#f97316", alpha=0.85, label="SS-CBD loses / not significant")
    ax.legend(handles=[win_patch, loss_patch,
                        plt.Line2D([0], [0], color="black", linestyle="--", lw=1.2, label="LRT baseline")],
              fontsize=9, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, "fig1_speedup_bars.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig2_detection_cdf(df_exp1: pd.DataFrame, output_dir: str) -> None:
    """CDF of detection times for SS-CBD vs LRT across 4 scenarios."""
    drift = ["financial_gradual", "financial_step", "phishing_step", "phishing_gradual"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)

    for ax, sc in zip(axes, drift):
        for method, col in [("SS-CBD", COLORS["SS-CBD"]), ("LRT", COLORS["LRT"])]:
            sub = df_exp1[(df_exp1["scenario"] == sc) & (df_exp1["method"] == method)]
            det = sub[sub["detected"] == True]["detection_time"].dropna().astype(float)
            if len(det) == 0:
                continue
            sorted_t = np.sort(det)
            cdf = np.arange(1, len(sorted_t) + 1) / len(df_exp1[(df_exp1["scenario"] == sc) & (df_exp1["method"] == method)])
            ax.step(sorted_t, cdf, color=col, lw=1.8, label=method)
            ax.axvline(float(det.median()), color=col, linestyle=":", lw=1.2, alpha=0.7)

        ax.set_title(SCENARIO_LABELS[sc], fontsize=10)
        ax.set_xlabel("Detection time (actions)", fontsize=9)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Cumulative detection rate", fontsize=10)
    axes[0].legend(fontsize=9)
    fig.suptitle("Detection Time CDFs — SS-CBD vs LRT (N=500 trials)\nDotted lines: medians",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "fig2_detection_cdf.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig3_kb_poison(df_exp7_raw: pd.DataFrame, df_exp7_summary: pd.DataFrame, output_dir: str) -> None:
    """Side-by-side: detection time violin + detection rate bar for KB-poison."""
    drift_summary = df_exp7_summary[df_exp7_summary["scenario"] == "kb_poison_slowburn"]
    methods_ordered = ["SS-CBD", "Hellinger", "LRT", "CBD"]

    fig, (ax_bar, ax_vio) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: detection rate bars ---
    det_rates, ci_lo, ci_hi = [], [], []
    for m in methods_ordered:
        row = drift_summary[drift_summary["method"] == m]
        if len(row) == 0:
            det_rates.append(0); ci_lo.append(0); ci_hi.append(0)
            continue
        r = float(row["det_rate"].iloc[0])
        lo = float(row["ci_lo"].iloc[0])
        hi = float(row["ci_hi"].iloc[0])
        det_rates.append(r)
        ci_lo.append(r - lo)
        ci_hi.append(hi - r)

    x = np.arange(len(methods_ordered))
    bar_colors = [COLORS.get(m, "#6b7280") for m in methods_ordered]
    ax_bar.bar(x, [r * 100 for r in det_rates], color=bar_colors, alpha=0.85, width=0.55,
               yerr=[[lo * 100 for lo in ci_lo], [hi * 100 for hi in ci_hi]],
               capsize=5, error_kw=dict(ecolor="black", lw=1.2))
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods_ordered, fontsize=10)
    ax_bar.set_ylabel("Detection rate (%)", fontsize=10)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title("Detection Rate\n(95% CI, N=100 trials)", fontsize=10)
    ax_bar.axhline(100, color="gray", linestyle="--", lw=1, alpha=0.5)
    for xi, (r, hi) in enumerate(zip(det_rates, ci_hi)):
        ax_bar.text(xi, r * 100 + hi * 100 + 2, f"{r*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- Right: detection time box plot (detected trials only) ---
    det_data = []
    for m in methods_ordered:
        sub = df_exp7_raw[(df_exp7_raw["scenario"] == "kb_poison_slowburn") &
                          (df_exp7_raw["method"] == m) &
                          (df_exp7_raw["detected"] == True)]
        det_col = "detection_time" if "detection_time" in sub.columns else "detection_time_actions"
        times = sub[det_col].dropna().astype(float).values
        det_data.append(times if len(times) > 0 else np.array([]))

    bp = ax_vio.boxplot(
        [d for d in det_data],
        positions=np.arange(len(methods_ordered)),
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", lw=2.5),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, m in zip(bp["boxes"], methods_ordered):
        patch.set_facecolor(COLORS.get(m, "#6b7280"))
        patch.set_alpha(0.8)

    # Annotate medians
    for i, d in enumerate(det_data):
        if len(d) > 0:
            med = float(np.median(d))
            ax_vio.text(i, med + 15, f"med={med:.0f}", ha="center", va="bottom", fontsize=8, color="white",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS.get(methods_ordered[i], "#333"),
                                  alpha=0.85, edgecolor="none"))

    ax_vio.set_xticks(np.arange(len(methods_ordered)))
    ax_vio.set_xticklabels(methods_ordered, fontsize=10)
    ax_vio.set_ylabel("Detection time (actions post-calibration)", fontsize=10)
    ax_vio.set_title("Detection Time Distribution\n(detected trials only)", fontsize=10)
    ax_vio.spines["top"].set_visible(False)
    ax_vio.spines["right"].set_visible(False)
    ax_vio.set_ylim(0, None)

    # CBD note
    ax_vio.text(3, ax_vio.get_ylim()[1] * 0.85,
                f"CBD: 62%\nmissed", ha="center", va="top", fontsize=8,
                color=COLORS["CBD"], style="italic")

    fig.suptitle("Knowledge-Base Poisoning Case Study — §7.11\n(N=100 trials, seed=42, logistic ramp 600 actions)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "fig3_kb_poison_compare.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=== Generating paper figures → {OUTPUT_DIR}/ ===\n")

    exp1_path = "results/exp1/raw_results.csv"
    exp7_path = "results/exp7/raw_results.csv"

    if os.path.exists(exp1_path):
        df1 = pd.read_csv(exp1_path)
        # Rename column if needed
        if "detection_time" not in df1.columns and "detection_time_actions" in df1.columns:
            df1 = df1.rename(columns={"detection_time_actions": "detection_time"})
        print("Generating fig1 (speedup bars) ...")
        fig1_speedup_bars(df1, OUTPUT_DIR)
        print("Generating fig2 (detection CDFs) ...")
        fig2_detection_cdf(df1, OUTPUT_DIR)
    else:
        print(f"  [SKIP] EXP1 not found at {exp1_path} — run 'make exp1' first")

    exp7_summary_path = "results/exp7/detection_table.csv"
    if os.path.exists(exp7_path) and os.path.exists(exp7_summary_path):
        df7_raw = pd.read_csv(exp7_path)
        if "detection_time" not in df7_raw.columns and "detection_time_actions" in df7_raw.columns:
            df7_raw = df7_raw.rename(columns={"detection_time_actions": "detection_time"})
        df7_summary = pd.read_csv(exp7_summary_path)
        print("Generating fig3 (KB-poison comparison) ...")
        fig3_kb_poison(df7_raw, df7_summary, OUTPUT_DIR)
    else:
        print(f"  [SKIP] EXP7 not found — run 'make exp7' first")

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
