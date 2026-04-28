"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from ide.embeddings import ActionSpec, MockEmbedder
from ide.evaluation.simulator import build_customer_service_vocab


@pytest.fixture()
def vocab() -> list[ActionSpec]:
    return build_customer_service_vocab()


@pytest.fixture()
def embedder():
    return MockEmbedder()


@pytest.fixture()
def baseline_dist(vocab) -> np.ndarray:
    from ide.evaluation.simulator import BASELINE_CS
    return BASELINE_CS
