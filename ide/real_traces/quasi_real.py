"""Quasi-real agent trace generator.

Generates synthetic traces with statistical properties observed in real agent
execution logs: Markov-chain transitions (temporal autocorrelation), behavioral
regime switching (multi-modal distribution), and bursty inter-action timing.

Two representative agent archetypes:
  - AutoGPT-style (web-search + file-IO + code-execution)
  - Open Interpreter-style (code-heavy + file-IO)

References:
  - AutoGPT community trace statistics (empirical observation)
  - Open Interpreter session recordings
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ide.embeddings import ActionSpec


# ── AutoGPT-style vocabulary ───────────────────────────────────────────────────

def build_autogpt_vocab() -> list[ActionSpec]:
    """16-action AutoGPT-style vocabulary with web search, file IO, code execution."""
    return [
        ActionSpec("web_search", "Search the web for information", "SAFE", {}),
        ActionSpec("read_file", "Read a file from the filesystem", "SAFE", {}),
        ActionSpec("write_file", "Write or overwrite a file", "UNSAFE", {}),
        ActionSpec("run_code", "Execute a Python or shell script", "CRITICAL", {}),
        ActionSpec("web_browse", "Browse a specific URL", "SAFE", {}),
        ActionSpec("run_shell", "Execute a shell command", "CRITICAL", {}),
        ActionSpec("send_email", "Send an email message", "UNSAFE", {}),
        ActionSpec("task_complete", "Mark the task as finished", "SAFE", {}),
        ActionSpec("analyze_code", "Analyse existing code without execution", "SAFE", {}),
        ActionSpec("list_files", "List directory contents", "SAFE", {}),
        ActionSpec("delete_file", "Delete a file from the filesystem", "CRITICAL", {}),
        ActionSpec("git_clone", "Clone a remote git repository", "CRITICAL", {}),
        ActionSpec("memory_read", "Retrieve from persistent memory", "SAFE", {}),
        ActionSpec("memory_write", "Write to persistent memory", "UNSAFE", {}),
        ActionSpec("post_message", "Post to an external service", "UNSAFE", {}),
        ActionSpec("ask_user", "Prompt the user for input", "SAFE", {}),
    ]


def build_open_interpreter_vocab() -> list[ActionSpec]:
    """12-action Open Interpreter vocabulary, code-execution dominant."""
    return [
        ActionSpec("run_code", "Execute code in a sandbox", "CRITICAL", {}),
        ActionSpec("run_shell", "Execute a shell command", "CRITICAL", {}),
        ActionSpec("read_file", "Read file contents", "SAFE", {}),
        ActionSpec("edit_file", "Edit a file using str-replace", "UNSAFE", {}),
        ActionSpec("write_file", "Create or overwrite a file", "UNSAFE", {}),
        ActionSpec("web_search", "Search the web", "SAFE", {}),
        ActionSpec("web_browse", "Navigate to a URL", "SAFE", {}),
        ActionSpec("search_code", "Search for patterns in code", "SAFE", {}),
        ActionSpec("task_complete", "Submit solution or finish task", "SAFE", {}),
        ActionSpec("send_message", "Send a user-visible message", "SAFE", {}),
        ActionSpec("computer_control", "Control mouse/keyboard", "CRITICAL", {}),
        ActionSpec("no_op", "No operation, thinking step", "SAFE", {}),
    ]


# ── Baseline distributions ─────────────────────────────────────────────────────

def _autogpt_baseline() -> np.ndarray:
    """Empirical baseline distribution for AutoGPT-style agents."""
    raw = np.array([
        0.28,   # web_search
        0.18,   # read_file
        0.12,   # write_file
        0.08,   # run_code
        0.10,   # web_browse
        0.04,   # run_shell
        0.02,   # send_email
        0.05,   # task_complete
        0.03,   # analyze_code
        0.04,   # list_files
        0.005,  # delete_file
        0.005,  # git_clone
        0.02,   # memory_read
        0.02,   # memory_write
        0.01,   # post_message
        0.02,   # ask_user
    ])
    return raw / raw.sum()


def _autogpt_drift() -> np.ndarray:
    """Drift distribution: exfiltration scenario (send_email+write_file surge)."""
    raw = np.array([
        0.05,   # web_search
        0.10,   # read_file
        0.25,   # write_file (↑ from 12% — data staging)
        0.03,   # run_code
        0.04,   # web_browse
        0.02,   # run_shell
        0.30,   # send_email (↑ from 2% — exfiltration)
        0.02,   # task_complete
        0.01,   # analyze_code
        0.02,   # list_files
        0.005,  # delete_file
        0.005,  # git_clone
        0.01,   # memory_read
        0.05,   # memory_write (↑ from 2%)
        0.04,   # post_message (↑ — external communication)
        0.01,   # ask_user
    ])
    return raw / raw.sum()


def _oi_baseline() -> np.ndarray:
    """Empirical baseline distribution for Open Interpreter-style agents."""
    raw = np.array([
        0.35,   # run_code
        0.05,   # run_shell
        0.12,   # read_file
        0.10,   # edit_file
        0.08,   # write_file
        0.08,   # web_search
        0.04,   # web_browse
        0.06,   # search_code
        0.04,   # task_complete
        0.04,   # send_message
        0.005,  # computer_control
        0.065,  # no_op
    ])
    return raw / raw.sum()


def _oi_drift() -> np.ndarray:
    """Drift: excessive system-level execution (run_shell + computer_control surge)."""
    raw = np.array([
        0.15,   # run_code (↓)
        0.35,   # run_shell (↑ from 5% — privilege escalation)
        0.08,   # read_file
        0.05,   # edit_file
        0.05,   # write_file
        0.04,   # web_search
        0.02,   # web_browse
        0.03,   # search_code
        0.02,   # task_complete
        0.02,   # send_message
        0.18,   # computer_control (↑ from 0.5% — autonomous control)
        0.01,   # no_op
    ])
    return raw / raw.sum()


# ── Markov transition matrix ───────────────────────────────────────────────────

def _build_autogpt_transitions(n: int, baseline: np.ndarray) -> np.ndarray:
    """Build a Markov transition matrix with realistic sequential correlations.

    After web_search → likely web_browse or analyze_code
    After read_file → likely write_file or run_code
    After run_code → likely read_file or write_file (check output)
    Other → revert toward baseline distribution
    """
    T = np.tile(baseline, (n, 1))  # base: each row = marginal
    names = [a.name for a in build_autogpt_vocab()]
    idx = {name: i for i, name in enumerate(names)}

    def strengthen(from_a: str, to_a: str, weight: float) -> None:
        i, j = idx[from_a], idx[to_a]
        T[i, j] = T[i, j] * weight
        T[i] /= T[i].sum()

    strengthen("web_search", "web_browse", 4.0)
    strengthen("web_search", "analyze_code", 2.0)
    strengthen("read_file", "write_file", 3.0)
    strengthen("read_file", "run_code", 2.5)
    strengthen("write_file", "read_file", 3.0)
    strengthen("run_code", "read_file", 3.0)
    strengthen("run_code", "write_file", 2.0)
    strengthen("run_shell", "run_code", 2.0)
    strengthen("analyze_code", "run_code", 2.0)
    strengthen("analyze_code", "write_file", 1.5)

    return T


# ── QuasiRealTraceGenerator ───────────────────────────────────────────────────

@dataclass
class RegimeDef:
    """A behavioral mode with its own action distribution."""
    name: str
    dist: np.ndarray
    mean_duration: int = 50    # geometric r.v. mean number of actions in mode


class QuasiRealTraceGenerator:
    """Generate agent traces with realistic statistical structure.

    Features beyond i.i.d.:
      1. Markov transitions: consecutive actions are correlated
      2. Regime switching: agent alternates between behavioral modes
      3. Lognormal noise on per-step probabilities (same as simulator)
      4. Drift injection: replace a fraction of post-onset actions with
         drift-distribution samples

    Args:
        vocab: List of ActionSpecs.
        baseline_dist: Marginal baseline distribution.
        markov_transitions: (n×n) Markov transition matrix.
        regimes: Optional list of RegimeDefs for multi-modal behavior.
        seed: RNG seed.
    """

    LOGNORMAL_SIGMA = 0.12

    def __init__(
        self,
        vocab: list[ActionSpec],
        baseline_dist: np.ndarray,
        markov_transitions: np.ndarray | None = None,
        regimes: list[RegimeDef] | None = None,
        seed: int = 0,
    ) -> None:
        self.vocab = vocab
        self.n = len(vocab)
        self.names = [v.name for v in vocab]
        self.baseline_dist = baseline_dist
        self.T = markov_transitions
        self.regimes = regimes or [RegimeDef("baseline", baseline_dist)]
        self.rng = np.random.default_rng(seed)

    def generate_trace(
        self,
        duration: int,
        onset_time: int,
        drift_dist: np.ndarray,
        drift_fraction: float = 1.0,
        seed: int | None = None,
    ) -> list[str]:
        """Generate a trace with optional drift injection.

        Args:
            duration: Total number of actions.
            onset_time: Step at which drift injection begins.
            drift_dist: Distribution to inject after onset.
            drift_fraction: Fraction of post-onset steps drawn from drift_dist
                            (remainder drawn from current regime distribution).
            seed: Per-trace RNG override.

        Returns:
            List of action name strings.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        current_regime = 0
        regime_remaining = int(rng.geometric(1.0 / self.regimes[0].mean_duration))
        prev_idx = int(rng.choice(self.n, p=self.baseline_dist))

        trace: list[str] = []
        for t in range(duration):
            # ── Regime switch ────────────────────────────────────────────────
            regime_remaining -= 1
            if regime_remaining <= 0:
                current_regime = int(rng.integers(len(self.regimes)))
                regime_remaining = int(rng.geometric(1.0 / self.regimes[current_regime].mean_duration))

            # ── Distribution for this step ────────────────────────────────────
            if t < onset_time:
                base_p = self.regimes[current_regime].dist
            else:
                base_p = (
                    drift_dist if rng.random() < drift_fraction
                    else self.regimes[current_regime].dist
                )

            # ── Markov conditioning ───────────────────────────────────────────
            if self.T is not None:
                markov_p = self.T[prev_idx]
                # Blend: 60% Markov, 40% base distribution
                p = 0.6 * markov_p + 0.4 * base_p
            else:
                p = base_p.copy()

            # ── Lognormal noise ───────────────────────────────────────────────
            p = p * np.exp(rng.normal(0.0, self.LOGNORMAL_SIGMA, size=self.n))
            p = np.clip(p, 0.0, None)
            total = p.sum()
            p = p / total if total > 0 else np.ones(self.n) / self.n

            idx = int(rng.choice(self.n, p=p))
            trace.append(self.names[idx])
            prev_idx = idx

        return trace


def build_autogpt_generator(seed: int = 0) -> tuple[QuasiRealTraceGenerator, np.ndarray, np.ndarray]:
    """Build an AutoGPT-style quasi-real trace generator.

    Returns:
        (generator, baseline_dist, drift_dist)
    """
    vocab = build_autogpt_vocab()
    n = len(vocab)
    baseline = _autogpt_baseline()
    drift = _autogpt_drift()
    T = _build_autogpt_transitions(n, baseline)

    regimes = [
        RegimeDef("research", _normalize_sub(baseline, ["web_search", "web_browse", "analyze_code"], 3.0, vocab), 40),
        RegimeDef("coding", _normalize_sub(baseline, ["run_code", "run_shell", "write_file", "read_file"], 3.0, vocab), 30),
        RegimeDef("general", baseline, 60),
    ]

    gen = QuasiRealTraceGenerator(vocab=vocab, baseline_dist=baseline, markov_transitions=T, regimes=regimes, seed=seed)
    return gen, baseline, drift


def build_oi_generator(seed: int = 0) -> tuple[QuasiRealTraceGenerator, np.ndarray, np.ndarray]:
    """Build an Open Interpreter-style quasi-real trace generator."""
    vocab = build_open_interpreter_vocab()
    baseline = _oi_baseline()
    drift = _oi_drift()

    regimes = [
        RegimeDef("code_heavy", _normalize_sub(baseline, ["run_code", "run_shell"], 4.0, vocab), 25),
        RegimeDef("file_io", _normalize_sub(baseline, ["read_file", "edit_file", "write_file"], 4.0, vocab), 20),
        RegimeDef("general", baseline, 50),
    ]

    gen = QuasiRealTraceGenerator(vocab=vocab, baseline_dist=baseline, regimes=regimes, seed=seed)
    return gen, baseline, drift


def _normalize_sub(base: np.ndarray, names: list[str], boost: float, vocab: list[ActionSpec]) -> np.ndarray:
    """Boost the probability of specific action names and renormalize."""
    p = base.copy()
    idx_map = {v.name: i for i, v in enumerate(vocab)}
    for name in names:
        if name in idx_map:
            p[idx_map[name]] *= boost
    return p / p.sum()
