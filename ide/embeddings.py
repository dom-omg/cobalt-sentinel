"""Action embedding layer for SS-CBD.

Transforms actions (name + description + tier) into dense vectors where
geometric distance reflects operational similarity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

EMBEDDING_DIM = 768
E5_MODEL = "intfloat/e5-large-v2"


@dataclass
class ActionSpec:
    """Specification of an agent action."""

    name: str
    description: str
    tier: Literal["SAFE", "UNSAFE", "CRITICAL"]
    parameters: dict = field(default_factory=dict)


def _mock_embed(name: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic hash-based embedding for offline/CI use."""
    digest = hashlib.sha256(name.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class MockEmbedder:
    """Offline fallback embedder using deterministic hash projection."""

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, spec: ActionSpec) -> np.ndarray:
        """Embed a single ActionSpec into a unit-norm vector."""
        if spec.name not in self._cache:
            self._cache[spec.name] = _mock_embed(spec.name)
        return self._cache[spec.name]

    def embed_vocabulary(self, specs: list[ActionSpec]) -> np.ndarray:
        """Embed a list of ActionSpecs; returns (n, dim) array."""
        return np.stack([self.embed(s) for s in specs])


class ActionEmbedder:
    """Semantic embedder wrapping sentence-transformers E5.

    Falls back to MockEmbedder if the model cannot be loaded.
    """

    def __init__(self, model_name: str = E5_MODEL) -> None:
        self._cache: dict[str, np.ndarray] = {}
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self._offline = False
        except Exception:
            self._model = None
            self._offline = True
            self._mock = MockEmbedder()

    def _prompt(self, spec: ActionSpec) -> str:
        return f"passage: action {spec.name} of tier {spec.tier}. {spec.description}"

    def embed(self, spec: ActionSpec) -> np.ndarray:
        """Embed a single ActionSpec into a unit-norm vector of shape (dim,)."""
        if spec.name in self._cache:
            return self._cache[spec.name]
        if self._offline:
            vec = self._mock.embed(spec)
        else:
            vec = self._model.encode(
                self._prompt(spec),
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
        self._cache[spec.name] = vec
        return vec

    def embed_vocabulary(self, specs: list[ActionSpec]) -> np.ndarray:
        """Embed full vocabulary; returns (n, dim) array."""
        missing = [s for s in specs if s.name not in self._cache]
        if missing and not self._offline:
            prompts = [self._prompt(s) for s in missing]
            vecs = self._model.encode(
                prompts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            ).astype(np.float32)
            for s, v in zip(missing, vecs):
                self._cache[s.name] = v
        elif missing and self._offline:
            for s in missing:
                self._cache[s.name] = self._mock.embed(s)
        return np.stack([self._cache[s.name] for s in specs])
