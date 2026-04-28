# cobalt-sentinel — SS-CBD Identity Drift Detector

**Paper:** [Identity Drift Detection in Autonomous AI Agents via Cosine Behavioral Distance](https://arxiv.org/abs/PLACEHOLDER) — Dominik Blain, QreativeLab Inc., arXiv April 2026

Sequential Semantic Cosine Behavioral Distance (SS-CBD): detects behavioral drift in autonomous AI agents by combining Wald's SPRT, semantic action embeddings, and tier-weighted operational risk. Mean **2.61× detection-time speedup** over LRT on N=500 synthetic trials; validated on three real agent corpora (AgentBench KG, SWE-agent CTF, SWE-bench coding).

## Quick Start

```bash
git clone https://github.com/dom-omg/cobalt-sentinel
cd cobalt-sentinel
pip install -r requirements.txt
make test      # run all tests (26/26 must pass)
make exp1      # headline experiment — §7.5, seed=42
make exp7      # KB-poisoning case study — §7.11
make exp9      # real agent trace evaluation — §7.13
```

## Paper Validation

`make exp1` outputs:

```
>>> SS-CBD speedup vs LRT (averaged across drift scenarios): X.XXx
>>> PAPER VALIDATION: PASS / FAIL
```

If FAIL: investigate before modifying hyperparameters. Do not tune to force success.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full table-to-experiment mapping.

## Citation

```bibtex
@article{blain2026identitydrift,
  title   = {Identity Drift Detection in Autonomous {AI} Agents via Cosine Behavioral Distance},
  author  = {Blain, Dominik},
  journal = {arXiv preprint arXiv:PLACEHOLDER},
  year    = {2026},
  url     = {https://arxiv.org/abs/PLACEHOLDER}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

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
  exp1_headline.py   — §7.5 main benchmark
  exp2_ablations.py  — §7.7 lambda / tier weight sweep
  exp3_adversarial.py — §7.8 threshold-straddling, slow drift, volume dilution
  exp7_case_study.py — §7.11 KB-poisoning
  exp8_industry_baselines.py — §7.12 LangSmith/Arize/naive comparison
  exp9_real_traces.py — §7.13 AgentBench KG + SWE-agent + SWE-bench
data/real_traces/    — AgentBench KG, SWE-agent CTF, SWE-bench coding corpora
```
