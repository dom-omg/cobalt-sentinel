"""FPR tests for all baseline detectors."""

import numpy as np
import pytest

from ide.evaluation.simulator import BASELINE_CS, build_customer_service_vocab

FPR_TOLERANCE = 0.07
N_TRIALS = 500


def _run_fpr(detector_cls, vocab, n_trials=N_TRIALS, seed=0, **kwargs):
    """Measure FPR by running detector on no-drift traces."""
    names = [v.name for v in vocab]
    rng = np.random.default_rng(seed)
    alerts = 0
    n_obs = 0
    for _ in range(n_trials):
        det = detector_cls(vocab=vocab, **kwargs)
        calib = rng.choice(names, size=50, p=BASELINE_CS).tolist()
        det.calibrate(calib)
        for _ in range(50):
            action = str(rng.choice(names, p=BASELINE_CS))
            result = det.observe(action)
            n_obs += 1
            if result.get("alert"):
                alerts += 1
    return alerts / n_obs if n_obs > 0 else 0.0


@pytest.fixture()
def vocab():
    return build_customer_service_vocab()


def test_lrt_respects_fpr(vocab):
    from ide.baselines.lrt import LRTDetector
    fpr = _run_fpr(LRTDetector, vocab, alpha=0.05, window_size=20)
    assert fpr <= FPR_TOLERANCE, f"LRT FPR {fpr:.3f} > {FPR_TOLERANCE}"


def test_chi2_respects_fpr(vocab):
    from ide.baselines.chi2 import Chi2Detector
    fpr = _run_fpr(Chi2Detector, vocab, alpha=0.05, window_size=20)
    assert fpr <= FPR_TOLERANCE, f"Chi2 FPR {fpr:.3f} > {FPR_TOLERANCE}"


def test_cbd_respects_fpr(vocab):
    from ide.baselines.cbd import CBDDetector
    fpr = _run_fpr(CBDDetector, vocab, window_size=20)
    # CBD uses fixed threshold 0.65 which is conservative
    assert fpr <= FPR_TOLERANCE, f"CBD FPR {fpr:.3f} > {FPR_TOLERANCE}"


def test_js_respects_fpr(vocab):
    from ide.baselines.js_divergence import JSDivergenceDetector
    fpr = _run_fpr(JSDivergenceDetector, vocab, alpha=0.05, window_size=20)
    assert fpr <= FPR_TOLERANCE, f"JS FPR {fpr:.3f} > {FPR_TOLERANCE}"


def test_hellinger_respects_fpr(vocab):
    from ide.baselines.hellinger import HellingerDetector
    fpr = _run_fpr(HellingerDetector, vocab, alpha=0.05, window_size=20)
    assert fpr <= FPR_TOLERANCE, f"Hellinger FPR {fpr:.3f} > {FPR_TOLERANCE}"


def test_cusum_respects_fpr(vocab):
    from ide.baselines.cusum import CUSUMDetector
    fpr = _run_fpr(CUSUMDetector, vocab, alpha=0.05)
    assert fpr <= FPR_TOLERANCE, f"CUSUM FPR {fpr:.3f} > {FPR_TOLERANCE}"


def test_adwin_respects_fpr(vocab):
    from ide.baselines.adwin import ADWINDetector
    try:
        import river  # noqa: F401
    except ImportError:
        pytest.skip("river not installed")
    fpr = _run_fpr(ADWINDetector, vocab, alpha=0.05)
    assert fpr <= FPR_TOLERANCE, f"ADWIN FPR {fpr:.3f} > {FPR_TOLERANCE}"
