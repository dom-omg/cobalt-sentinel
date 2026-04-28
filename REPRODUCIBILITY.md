# Reproducibility Guide — cobalt-sentinel

**Paper:** "Identity Drift Detection in Autonomous AI Agents via Cosine Behavioral Distance"  
**Author:** Dominik Blain, QreativeLab Inc.

All experiments are seeded and fully reproducible. Reference results for
automated verification are in `reference_results/`.

---

## Quick Start (< 5 minutes)

```bash
pip install -r requirements.txt
make exp1      # Headline experiment (N=500, seed=42) — §7.5
make exp7      # KB-poison case study (N=100, seed=42) — §7.11
python verify_reproduction.py   # Check results match reference
```

## Docker (isolated, zero-dependency)

```bash
docker build -t cobalt-sentinel .
docker run --rm cobalt-sentinel make exp1
docker run --rm cobalt-sentinel make exp7
docker run --rm cobalt-sentinel python verify_reproduction.py
```

## Full Experiment Suite

```bash
make all       # Tests + exp1 through exp7 (~45 min)
```

Individual experiments:

| Target | Section | N | Runtime |
|--------|---------|---|---------|
| `make exp1` | §7.5 Headline | 500 | ~8 min |
| `make exp2` | §7.7 Ablations | 300 | ~12 min |
| `make exp3` | §7.8 Adversarial | 100 | ~5 min |
| `make exp4` | §7.2 Quasi-real | 200 | ~10 min |
| `make exp5` | §5.8 Lower bound | 200 | ~6 min |
| `make exp6` | §7.5 EXP6 CIs | reuses exp1 | ~3 min |
| `make exp7` | §7.11 Case study | 100 | ~4 min |

## Offline Mode (default)

`TRANSFORMERS_OFFLINE=1` is set in all Makefile targets and in the
Dockerfile. When the `intfloat/e5-large-v2` sentence-transformer is not
downloaded, the `MockEmbedder` (deterministic hash-projection into R^768)
is used automatically. All paper results were produced with `MockEmbedder`.

To reproduce with the real sentence-transformer (requires ~1.3 GB download):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-large-v2')"
# Then run without TRANSFORMERS_OFFLINE=1
```

## Key Paper Numbers and Where to Find Them

| Claim | Source | File |
|-------|--------|------|
| 2.61× mean speedup SS-CBD vs LRT | EXP1 | `results/exp1/summary_table.csv` |
| 5.05× peak speedup (financial_gradual) | EXP1 | `results/exp1/summary_table.csv` |
| Bootstrap 95% CI [4.18, 6.15] (financial_gradual) | EXP6 | `results/exp6/RESULTS_EXP6.md` |
| CBD fails 62% slow-burn trials | EXP7 | `results/exp7/detection_table.csv` |
| SS-CBD median 20 actions (9 min) | EXP7 | `results/exp7/raw_results.csv` |
| SS-CBD 2.03× LRT (KB-poison) | EXP7 | `results/exp7/detection_table.csv` |

## Automated Verification

```bash
python verify_reproduction.py
```

Checks that reproduced results are within tolerance of reference values:
- EXP1: mean speedup ∈ [2.0, 3.5], peak speedup ∈ [4.0, 6.5]
- EXP7: SS-CBD det_rate = 1.0, CBD det_rate ∈ [0.25, 0.55]

Exit 0 on pass, 1 on failure.

## Environment

Tested with Python 3.11. Key dependencies:
- numpy ≥ 1.24
- scipy ≥ 1.10
- pandas ≥ 2.0
- sentence-transformers ≥ 2.2 (optional; MockEmbedder used when offline)
- river ≥ 0.21 (for ADWIN baseline)
- matplotlib ≥ 3.7 (for figures)
