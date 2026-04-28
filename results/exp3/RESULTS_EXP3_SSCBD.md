# EXP3: SS-CBD Adversarial Robustness — 4 Scenarios

**N=100 trials per experiment. Seed=42.**

Comparative evaluation: SS-CBD vs LRT vs CBD (and ADWIN for Exp 3.2).

---

## Exp 3.1: Threshold-Straddling (98% blend)

Adversary blends baseline and phishing distributions at 98% intensity, targeting evasion below the detection boundary.

| Method | Detection Rate | Max injectable CRITICAL fraction |
|---|---|---|
| **SS-CBD** | **100.0%** | 39.6% |
| LRT | **100.0%** | 39.6% |
| CBD | 92.0% | 39.6% |

**Finding:** SS-CBD and LRT both achieve 100% detection even against a 98%-blend adversary — the sequential accumulation of evidence across many steps makes threshold-straddling ineffective. CBD achieves 92% detection (8% miss rate), demonstrating the advantage of sequential testing (SPRT) over fixed-threshold batch testing. The "max injectable fraction" of 39.6% is the maximum excess probability mass of the drift distribution over the baseline; it measures the theoretical adversarial capability, not actual injection rate.

---

## Exp 3.2: Slow Drift (6-hour linear ramp)

Adversary gradually increases `send_email::UNSAFE` over 4320 actions (720 act/h × 6h). Compare detection rates and times.

| Method | Detection Rate | Mean Detection Time (actions) |
|---|---|---|
| **SS-CBD** | 100.0% | 140.8 |
| LRT | 100.0% | 135.8 |
| **ADWIN** | **0.0%** | ∞ |
| CBD | 95.0% | 3157.2 |

**Finding:**
- SS-CBD (140.8) and LRT (135.8) perform comparably on the 6h ramp — no significant speedup, consistent with the phishing-adjacent semantic limitation discussed in §6.5.
- **ADWIN achieves 0% detection** on this scenario. ADWIN's adaptive windowing tracks the slowly changing distribution as it evolves, so consecutive windows remain statistically similar throughout the 4320-action ramp. This is a known limitation of the ADWIN algorithm for very slow, sub-window drift. The claim in earlier paper versions that "ADWIN detects earlier" does NOT hold at this time scale.
- CBD achieves 95% detection but with mean detection time of 3157 actions — approximately 4.4 hours post-onset, vs ~12 minutes for SS-CBD/LRT.

**Correction to Section 6.8:** The earlier claim that "ADWIN detects slow drift at approximately 2.1 hours" is not reproducible under the 6h ramp parameters (N=100, seed=42, alpha=0.05).

---

## Exp 3.3: Volume Dilution (1x vs 4x)

Adversary increases total action volume by 4× while maintaining action ratios — tests whether higher volume can dilute detection.

| Method | Mean time (1x) | Mean time (4x) | Ratio 4x/1x | Magnitude-invariant? |
|---|---|---|---|---|
| **SS-CBD** | 41.0 | 42.2 | 1.03 | ✓ |
| LRT | 91.9 | 94.3 | 1.03 | ✓ |
| CBD | 225.3 | 275.1 | 1.22 | ✓ |

**Finding:** All three methods are magnitude-invariant (ratios < 1.5). Volume dilution provides no detection evasion benefit. CBD's ratio of 1.22 reflects slightly higher variance due to its fixed-window design, but it remains within the invariant threshold. SS-CBD and LRT are most robust with ratios of 1.03.

---

## Exp 3.4: Adaptive Evasion (White-Box SPRT Attack)

An adversary with full knowledge of the SS-CBD SPRT state monitors the cumulative statistic S and injects CRITICAL actions only when S < 75% of the upper boundary (log_B). Compared to naive-blend adversaries against LRT and CBD.

| Method | Strategy | Detection Rate | Mean CRITICAL injected | Injection fraction |
|---|---|---|---|---|
| **SS-CBD (white-box)** | Adaptive (boundary_frac=0.75) | 97.0% | 32.0 | 100.0% |
| LRT | Naive blend (90%) | 100.0% | 0.7 | 0.1% |
| CBD | Naive blend (90%) | 91.0% | 6.3 | 0.6% |

**Finding:** Even a white-box adversary who reads the SS-CBD SPRT state is detected 97% of the time after injecting 32 CRITICAL actions. The detection mechanism is robust for an unexpected reason: CRITICAL actions (call_api, process_refund, export_data) that did not appear in the 50-action calibration window receive a near-zero baseline estimate, causing their log-LR to be large and positive regardless of the adversary's strategy. An adversary exploiting SPRT state must also defeat the sparse calibration baseline — a two-factor defense.

**Comparison note:** LRT and CBD results use a different adversary strategy (naive 90%-blend injection), so the CRITICAL injection counts are not directly comparable. The LRT/CBD columns demonstrate that naive-blend evasion is also largely defeated (100% and 91% detection rates, respectively).

**Mitigation summary:**
- White-box SPRT evasion: partially mitigated by sparse calibration; 97% detection rate
- Recommended complementary defenses: randomized SPRT boundaries per deployment; increased calibration window size for rare-action types

---

## Combined Adversarial Summary

| Attack | SS-CBD | LRT | CBD | Winner |
|---|---|---|---|---|
| Threshold-straddling (98%) | 100% ✓ | 100% ✓ | 92% | SS-CBD = LRT |
| Slow drift (6h ramp) | 100% (140.8 act) | 100% (135.8 act) | 95% (3157 act) | LRT marginally |
| Volume dilution | ✓ invariant (ratio=1.03) | ✓ (1.03) | ✓ (1.22) | SS-CBD = LRT |
| Adaptive evasion (white-box) | 97% | N/A | N/A | SS-CBD robust |

**Overall:** SS-CBD matches or exceeds LRT on 3 of 4 adversarial scenarios. CBD is weakest across all four. ADWIN is unsuitable for slow linear-ramp drift at the 6-hour time scale.
