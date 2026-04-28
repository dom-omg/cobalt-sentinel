# cobalt-sentinel — SS-CBD Identity Drift Detector

Sequential Semantic Cosine Behavioral Distance (SS-CBD): detects behavioral drift in autonomous AI agents by combining Wald's SPRT, semantic action embeddings, and tier-weighted operational risk.

## Quick Start

```bash
pip install -r requirements.txt
make test      # run all tests (must pass before experiments)
make exp1      # headline experiment — produces RESULTS.md
make exp2      # ablation sweep
make exp3      # adversarial robustness
```

## Paper Validation

`make exp1` outputs:

```
>>> SS-CBD speedup vs LRT (averaged across drift scenarios): X.XXx
>>> PAPER VALIDATION: PASS / FAIL
```

If FAIL: investigate before modifying hyperparameters. Do not tune to force success.

## Project Structure

```
ide/
  embeddings.py      — ActionSpec + ActionEmbedder (E5 / MockEmbedder fallback)
  sequential.py      — Wald SPRT engine
  ss_cbd.py          — SS-CBD flagship detector
  baselines/         — LRT, Chi2, JS, Hellinger, CBD, ADWIN, CUSUM
  evaluation/
    simulator.py     — synthetic trace generator
    metrics.py       — detection_rate, clopper_pearson_ci, bootstrap_ci, AUC
    runner.py        — experiment orchestrator
experiments/
  exp1_headline.py   — main benchmark
  exp2_ablations.py  — lambda / tier weight sweep
  exp3_adversarial.py — threshold-straddling, slow drift, volume dilution
```
