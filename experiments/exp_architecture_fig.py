"""Generate Figure 0: SS-CBD / Identity Drift Engine architecture diagram.

Usage:
    python -m experiments.exp_architecture_fig

Outputs to results/figures/:
  fig0_architecture.pdf  — vector, for paper submission
  fig0_architecture.png  — 300 DPI raster
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = "results/figures"

# ---------------------------------------------------------------------------
# Color palette (NeurIPS-friendly, print-safe)
# ---------------------------------------------------------------------------
C_AGENT     = "#DBEAFE"   # light blue
C_AGENT_ED  = "#2563EB"   # blue edge
C_IDE       = "#F3F4F6"   # light gray
C_IDE_ED    = "#6B7280"   # gray edge
C_SUB       = "#EEF2FF"   # indigo tint — sub-components of SS-CBD
C_SUB_ED    = "#4338CA"   # indigo edge
C_DEC       = "#DCFCE7"   # light green
C_DEC_ED    = "#16A34A"   # green edge
C_ALERT     = "#FEE2E2"   # light red
C_ALERT_ED  = "#DC2626"   # red edge
C_ARROW     = "#374151"   # near-black arrows
C_TEXT      = "#111827"   # body text


def rounded_box(ax, x, y, w, h, facecolor, edgecolor, label_lines,
                fontsize=9.5, bold_first=False, pad=0.015):
    """Draw a rounded FancyBboxPatch and centered multi-line label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)

    cx = x + w / 2
    cy = y + h / 2

    if isinstance(label_lines, str):
        label_lines = [label_lines]

    n = len(label_lines)
    line_gap = 0.028  # axes fraction
    offsets = [(i - (n - 1) / 2) * line_gap for i in range(n)]

    for i, (line, offset) in enumerate(zip(label_lines, offsets)):
        weight = "bold" if (bold_first and i == 0) else "normal"
        style  = "italic" if (not bold_first and i > 0) else "normal"
        ax.text(cx, cy - offset, line,
                ha="center", va="center",
                fontsize=fontsize, fontweight=weight, fontstyle=style,
                color=C_TEXT, zorder=4)


def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW, lw=1.2,
          connectionstyle="arc3,rad=0.0", fontsize=8.5):
    """Draw a directed arrow between two points."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
            mutation_scale=10,
        ),
        zorder=5,
    )
    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(mx, my + 0.015, label,
                ha="center", va="bottom",
                fontsize=fontsize, color=color, zorder=6)


def dashed_rect(ax, x, y, w, h, edgecolor=C_IDE_ED, label=""):
    """Draw a dashed bounding rectangle for a group."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=0.9,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + 0.01, y + h + 0.005, label,
                ha="left", va="bottom",
                fontsize=8, color=edgecolor, fontstyle="italic", zorder=4)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Layout constants  (all in axes [0,1] coordinates)
    # ------------------------------------------------------------------
    BH  = 0.10   # standard box height
    BW  = 0.13   # standard box width

    # Row y-centers (top → bottom)
    Y_TOP   = 0.82   # Agent
    Y_OBS   = 0.64   # IDE Observe()
    Y_LOCK  = 0.46   # Baseline locker  |  Sliding window
    Y_CBD   = 0.24   # SS-CBD detector group
    Y_SUB   = 0.22   # sub-components y-bottom
    Y_DEC   = 0.05   # Decision
    Y_ALERT = 0.05   # Alert (same row, right side)

    # Column x-centers
    X_LEFT  = 0.12
    X_MID   = 0.50
    X_RIGHT = 0.88

    # Sub-component columns inside SS-CBD group
    X_SPRT  = 0.28
    X_EMBD  = 0.50
    X_TIER  = 0.72

    SUB_W   = 0.16
    SUB_H   = 0.13

    # ------------------------------------------------------------------
    # 1. Agent box
    # ------------------------------------------------------------------
    aw, ah = 0.18, BH
    ax_ag = X_MID - aw / 2
    ay_ag = Y_TOP - ah / 2
    rounded_box(ax, ax_ag, ay_ag, aw, ah,
                C_AGENT, C_AGENT_ED,
                ["Agent", "action stream"],
                fontsize=9.5, bold_first=True)

    # Action-tier legend below agent box
    ax.text(X_MID, ay_ag - 0.018,
            "tier:  SAFE  /  UNSAFE  /  CRITICAL",
            ha="center", va="top", fontsize=7.5,
            color="#6B7280", zorder=4)

    # ------------------------------------------------------------------
    # 2. IDE Observe() box
    # ------------------------------------------------------------------
    ow, oh = 0.22, BH
    ax_ob = X_MID - ow / 2
    ay_ob = Y_OBS - oh / 2
    rounded_box(ax, ax_ob, ay_ob, ow, oh,
                C_IDE, C_IDE_ED,
                ["IDE  Observe()", "action → event"],
                fontsize=9.5, bold_first=True)

    # Arrow: Agent → Observe
    arrow(ax, X_MID, ay_ag, X_MID, ay_ob + oh,
          label="aₜ", fontsize=8)

    # ------------------------------------------------------------------
    # 3. Baseline locker  (left branch)
    # ------------------------------------------------------------------
    bw, bh = 0.18, BH
    X_BLOC = 0.23
    ax_bl = X_BLOC - bw / 2
    ay_bl = Y_LOCK - bh / 2
    rounded_box(ax, ax_bl, ay_bl, bw, bh,
                C_IDE, C_IDE_ED,
                ["Baseline locker", "N_min = 20"],
                fontsize=9.5, bold_first=True)

    # Arrow: Observe → Baseline locker  (diagonal)
    arrow(ax, X_MID - ow / 2, Y_OBS,
          X_BLOC + bw / 2, Y_LOCK,
          label="calibrate", fontsize=7.5,
          connectionstyle="arc3,rad=-0.15")

    # "locks b" annotation below baseline locker box
    ax.text(X_BLOC, ay_bl - 0.016,
            "→ locks baseline  b",
            ha="center", va="top", fontsize=7.5,
            color="#6B7280", zorder=4)

    # ------------------------------------------------------------------
    # 4. Sliding window  (right branch)
    # ------------------------------------------------------------------
    sw, sh = 0.18, BH
    X_SWIN = 0.77
    ax_sw = X_SWIN - sw / 2
    ay_sw = Y_LOCK - sh / 2
    rounded_box(ax, ax_sw, ay_sw, sw, sh,
                C_IDE, C_IDE_ED,
                ["Sliding window", "Δt = 1 h"],
                fontsize=9.5, bold_first=True)

    # Arrow: Observe → Sliding window  (diagonal)
    arrow(ax, X_MID + ow / 2, Y_OBS,
          X_SWIN - sw / 2, Y_LOCK,
          label="stream", fontsize=7.5,
          connectionstyle="arc3,rad=0.15")

    # ------------------------------------------------------------------
    # 5. SS-CBD detector group bounding box
    # ------------------------------------------------------------------
    grp_x  = 0.15
    grp_y  = 0.06
    grp_w  = 0.70
    grp_h  = 0.26
    dashed_rect(ax, grp_x, grp_y, grp_w, grp_h,
                edgecolor="#4338CA",
                label="SS-CBD detector")

    # Arrow: Baseline locker → SS-CBD group
    arrow(ax, X_BLOC, ay_bl,
          X_BLOC, grp_y + grp_h,
          label="b", fontsize=8)

    # Arrow: Sliding window → SS-CBD group
    arrow(ax, X_SWIN, ay_sw,
          X_SWIN, grp_y + grp_h,
          label="Wₜ", fontsize=8)

    # ------------------------------------------------------------------
    # 5a. SPRT accumulator
    # ------------------------------------------------------------------
    sub_y = grp_y + (grp_h - SUB_H) / 2
    rounded_box(ax, X_SPRT - SUB_W / 2, sub_y, SUB_W, SUB_H,
                C_SUB, C_SUB_ED,
                ["SPRT", "accumulator", "Sₙ += log-LR"],
                fontsize=8.5, bold_first=True)

    # ------------------------------------------------------------------
    # 5b. Semantic embedder
    # ------------------------------------------------------------------
    rounded_box(ax, X_EMBD - SUB_W / 2, sub_y, SUB_W, SUB_H,
                C_SUB, C_SUB_ED,
                ["Semantic", "embedder", "E5-large → sᵢ"],
                fontsize=8.5, bold_first=True)

    # ------------------------------------------------------------------
    # 5c. Tier weighter
    # ------------------------------------------------------------------
    rounded_box(ax, X_TIER - SUB_W / 2, sub_y, SUB_W, SUB_H,
                C_SUB, C_SUB_ED,
                ["Tier weighter", "wc=5 / wu=2", "wsafe=1"],
                fontsize=8.5, bold_first=True)

    # ------------------------------------------------------------------
    # 6. Decision box
    # ------------------------------------------------------------------
    dw, dh = 0.22, BH
    X_DEC = 0.35
    ax_dec = X_DEC - dw / 2
    ay_dec = grp_y - dh - 0.045
    rounded_box(ax, ax_dec, ay_dec, dw, dh,
                C_DEC, C_DEC_ED,
                ["Decision", "NORMAL · DRIFT_DETECTED", "DRIFT_NEW_ACTION"],
                fontsize=8.5, bold_first=True)

    # Arrow: SS-CBD group → Decision
    arrow(ax, X_DEC, grp_y,
          X_DEC, ay_dec + dh,
          label="score", fontsize=7.5)

    # ------------------------------------------------------------------
    # 7. Alert box
    # ------------------------------------------------------------------
    alw, alh = 0.22, BH
    X_ALE = 0.72
    ax_ale = X_ALE - alw / 2
    ay_ale = ay_dec  # same row as Decision
    rounded_box(ax, ax_ale, ay_ale, alw, alh,
                C_ALERT, C_ALERT_ED,
                ["DriftAlert", "level · distance", "top_shift → operator"],
                fontsize=8.5, bold_first=True)

    # Arrow: Decision → Alert
    arrow(ax, ax_dec + dw, ay_dec + dh / 2,
          ax_ale, ay_ale + alh / 2,
          label="DRIFT_*", fontsize=7.5, color=C_ALERT_ED)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title(
        "Figure 1 — Identity Drift Engine (IDE): SS-CBD Architecture",
        fontsize=10.5, fontweight="bold", color=C_TEXT,
        pad=6,
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_patches = [
        mpatches.Patch(facecolor=C_AGENT,  edgecolor=C_AGENT_ED,  label="External agent"),
        mpatches.Patch(facecolor=C_IDE,    edgecolor=C_IDE_ED,    label="IDE core"),
        mpatches.Patch(facecolor=C_SUB,    edgecolor=C_SUB_ED,    label="SS-CBD sub-components"),
        mpatches.Patch(facecolor=C_DEC,    edgecolor=C_DEC_ED,    label="Decision output"),
        mpatches.Patch(facecolor=C_ALERT,  edgecolor=C_ALERT_ED,  label="Alert / operator"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#D1D5DB",
        handlelength=1.2,
        handleheight=0.9,
        borderpad=0.5,
        columnspacing=1.0,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pdf_path = os.path.join(OUTPUT_DIR, "fig0_architecture.pdf")
    png_path = os.path.join(OUTPUT_DIR, "fig0_architecture.png")

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)

    print(f"[fig0] saved → {pdf_path}")
    print(f"[fig0] saved → {png_path}")


if __name__ == "__main__":
    main()
