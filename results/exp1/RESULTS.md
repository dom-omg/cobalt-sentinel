# SS-CBD Headline Experiment Results

**Paper Validation: PASS ✅**


## Detection Time Speedup (SS-CBD vs LRT)

| Scenario | SS-CBD MeanTime | LRT MeanTime | Speedup |
|---|---|---|---|
| financial_gradual | 24.3 | 122.9 | 5.06x |
| financial_step | 35.4 | 113.4 | 3.20x |
| phishing_gradual | 151.5 | 131.8 | 0.87x |
| phishing_step | 88.2 | 114.7 | 1.30x |

## Key Numbers for Paper

- Mean speedup across scenarios: **2.61x**
- Min speedup: **0.87x**
- Max speedup: **5.06x**

## Recommendation

SS-CBD beats LRT at the required 1.5x threshold. Default hyperparameters (λ=0.5, w_unsafe=2, w_critical=5) are recommended.
