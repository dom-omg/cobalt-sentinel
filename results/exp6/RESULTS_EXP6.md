# EXP6: Statistical Significance Analysis

**N=500 trials, seed=42. All tests two-sided unless noted.**


## Table A: Bootstrap 95% CIs on Speedup Ratio (SS-CBD / LRT)

| Scenario | Speedup | 95% CI | McNemar χ² | p (McNemar) | Wilcoxon p |
|----------|---------|--------|------------|-------------|-----------|
| phishing_step | 1.30× | [0.99, 1.77] | 0.00 | >0.99 | 0.000 |
| phishing_gradual | 0.87× | [0.64, 1.26] | 0.00 | >0.99 | 0.000 |
| financial_step | 3.21× | [2.40, 4.47] | 0.00 | >0.99 | 0.000 |
| financial_gradual | 5.05× | [4.18, 6.15] | 0.00 | >0.99 | 0.000 |

**Interpretation:** ✓ = significant difference at α=0.05.
McNemar tests H0: P(SS-CBD detects) = P(LRT detects).
Wilcoxon tests H0: no difference in detection time (one-sided, SS-CBD faster).


## Table B: N=500 Detection Performance (Section 7.3 Replacement)

| Scenario | Method | Det% (95% CI) | Mean Det Time (±std) |
|----------|--------|---------------|----------------------|
| control | CBD | FPR 0.0% [0.0%, 0.7%] | — |
| control | Hellinger | FPR 100.0% [99.3%, 100.0%] | — |
| control | LRT | FPR 98.2% [96.6%, 99.2%] | — |
| control | SS-CBD | FPR 100.0% [99.3%, 100.0%] | — |
| financial_gradual | CBD | 0.0% [0.0%, 0.7%] | inf ± 0.0 |
| financial_gradual | Hellinger | 100.0% [99.3%, 100.0%] | 68.3 ± 59.4 |
| financial_gradual | LRT | 97.4% [95.6%, 98.6%] | 107.3 ± 96.1 |
| financial_gradual | SS-CBD | 99.8% [98.9%, 100.0%] | 23.5 ± 44.1 |
| financial_step | CBD | 0.0% [0.0%, 0.7%] | inf ± 0.0 |
| financial_step | Hellinger | 99.8% [98.9%, 100.0%] | 63.0 ± 53.8 |
| financial_step | LRT | 97.8% [96.1%, 98.9%] | 103.3 ± 93.4 |
| financial_step | SS-CBD | 99.0% [97.7%, 99.7%] | 24.4 ± 37.7 |
| phishing_gradual | CBD | 0.0% [0.0%, 0.7%] | inf ± 0.0 |
| phishing_gradual | Hellinger | 99.8% [98.9%, 100.0%] | 65.8 ± 57.0 |
| phishing_gradual | LRT | 95.6% [93.4%, 97.2%] | 109.3 ± 93.1 |
| phishing_gradual | SS-CBD | 94.2% [91.8%, 96.1%] | 33.0 ± 51.0 |
| phishing_step | CBD | 0.4% [0.1%, 1.4%] | 164.0 ± 97.6 |
| phishing_step | Hellinger | 99.8% [98.9%, 100.0%] | 63.2 ± 50.8 |
| phishing_step | LRT | 97.0% [95.1%, 98.3%] | 99.7 ± 86.9 |
| phishing_step | SS-CBD | 96.2% [94.1%, 97.7%] | 40.8 ± 67.5 |
