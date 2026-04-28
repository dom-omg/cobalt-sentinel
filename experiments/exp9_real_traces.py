"""EXP9: Real agent trace evaluation.

Uses publicly available LLM agent trace datasets to validate SS-CBD beyond
synthetic simulations:

  - AgentBench KG (Zheng et al., 2023): 1,214 real knowledge-graph traversal
    sessions from GPT-4 agents solving FREEBASE questions.
    Archetype: "knowledge-retrieval agent" (vocab=7: get_relations, get_neighbors,
    intersection, get_attributes, argmax, count, argmin)

  - SWE-agent (Yang et al., 2024): 12 real software-engineering sessions (coding
    + CTF) from Claude/GPT agents solving SWE-bench and CTF challenges.
    Archetype: "coding/security agent" (vocab~10: open, edit, bash, python,
    create, view, str_replace, submit, find, grep)

Protocol (§8.4 of paper):
  1. Parse sessions into ordered action sequences.
  2. Build "long traces" by concatenating consecutive sessions.
     Each long trace = one virtual agent instance doing multiple tasks over time.
  3. Compute baseline from first calib_size actions of each trace.
  4. Inject drift at onset = trace_length // 2 by replacing injection_fraction
     of subsequent actions with samples from a target drift distribution.
     Drift target: shift toward less-frequent action types (plausible role change).
  5. Run SS-CBD (SPRT, λ=0.5), LRT (SPRT, freq-only), Hellinger, CBD on each trial.

N=100 independent trials per archetype, seed=42.

Usage:
    python -m experiments.exp9_real_traces
    make exp9
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = Path("data/real_traces")
OUTPUT_DIR = Path("results/exp9")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 100
SEED = 42
TRACE_LENGTH = 800
CALIB_SIZE = 50
DRIFT_ONSET = 400
DRIFT_FRACTION = 0.50       # fraction of post-onset actions replaced with drift
WINDOW_SIZE = 200           # sliding window for CBD/Hellinger
SPRT_ALPHA = SPRT_BETA = 0.05
LAMBDA_SEM = 0.5            # semantic blend for SS-CBD

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_agentbench_kg() -> list[list[str]]:
    """Returns list of sessions, each a list of action-type strings."""
    sessions: list[list[str]] = []
    for fname in ("kg_rl_all.json", "std.json"):
        path = DATA_DIR / "agentbench_kg" / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data:
            acts = []
            for raw in item.get("actions", []):
                fn = raw.split("(")[0].strip().lower()
                if fn:
                    acts.append(fn)
            if acts:
                sessions.append(acts)
    return sessions


def parse_sailplane_coding() -> list[list[str]]:
    """Parse sailplane/swe-agent-trajs coding sessions (N=400, edit_file dominant)."""
    path = DATA_DIR / "sailplane_swe" / "sessions.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [item["acts"] for item in data if len(item["acts"]) >= 3]


def parse_swe_agent() -> list[list[str]]:
    """Returns list of sessions from .traj files."""
    sessions: list[list[str]] = []
    swe_dir = DATA_DIR / "swe_agent"
    if not swe_dir.exists():
        return sessions
    for traj_file in swe_dir.rglob("*.traj"):
        try:
            with open(traj_file) as f:
                d = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        acts: list[str] = []
        for h in d.get("history", []):
            if h.get("role") != "assistant":
                continue
            content = h.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            # Extract first bash/editor command from code blocks
            blocks = re.findall(r"```(?:\w+)?\n([^`]+)", content)
            for block in blocks:
                first_line = block.strip().split("\n")[0].strip()
                if not first_line:
                    continue
                cmd = first_line.split()[0].lower()
                # Normalize command names
                cmd = _normalize_swe_cmd(cmd)
                acts.append(cmd)
                break
        if acts:
            sessions.append(acts)
    return sessions


def _normalize_swe_cmd(cmd: str) -> str:
    """Map raw SWE-agent commands to normalized action types."""
    mapping = {
        "open": "view_file",
        "view": "view_file",
        "cat": "view_file",
        "head": "view_file",
        "tail": "view_file",
        "grep": "search_code",
        "find": "search_code",
        "ls": "list_files",
        "edit": "edit_file",
        "str_replace": "edit_file",
        "create": "create_file",
        "rm": "delete_file",
        "mkdir": "create_file",
        "python": "run_code",
        "python3": "run_code",
        "pytest": "run_code",
        "bash": "run_shell",
        "sh": "run_shell",
        "/bin/sh": "run_shell",
        "submit": "submit",
        "exit": "submit",
    }
    return mapping.get(cmd, "run_shell")


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------

def build_long_traces(
    sessions: list[list[str]],
    trace_length: int,
    n_trials: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    """
    Build N long traces of `trace_length` actions by concatenating random
    sessions. Each trace represents one virtual agent instance processing
    multiple tasks in sequence.
    """
    flat_actions: list[str] = []
    for sess in sessions:
        flat_actions.extend(sess)

    traces: list[list[str]] = []
    for _ in range(n_trials):
        # Sample with replacement from the flat pool, preserving session order
        trace: list[str] = []
        idx = rng.integers(0, len(flat_actions))
        while len(trace) < trace_length:
            trace.append(flat_actions[idx % len(flat_actions)])
            idx += 1
        traces.append(trace[:trace_length])
    return traces


# ---------------------------------------------------------------------------
# Drift injection
# ---------------------------------------------------------------------------

def build_drift_dist(
    baseline: dict[str, float],
    drift_target_actions: list[str],
    shift_factor: float = 3.0,
) -> dict[str, float]:
    """
    Construct a drift distribution by amplifying `drift_target_actions`
    by `shift_factor` and renormalizing. This models a role change where
    certain action types become disproportionately frequent.
    """
    drift = dict(baseline)
    for act in drift_target_actions:
        if act in drift:
            drift[act] *= shift_factor
        else:
            drift[act] = shift_factor * min(baseline.values(), default=0.01)
    total = sum(drift.values())
    return {k: v / total for k, v in drift.items()}


def inject_drift(
    trace: list[str],
    onset: int,
    drift_dist: dict[str, float],
    fraction: float,
    rng: np.random.Generator,
) -> list[str]:
    """Replace `fraction` of post-onset actions with drift-distribution samples."""
    modified = list(trace)
    vocab = list(drift_dist.keys())
    probs = np.array([drift_dist[v] for v in vocab])
    probs /= probs.sum()
    for i in range(onset, len(trace)):
        if rng.random() < fraction:
            modified[i] = rng.choice(vocab, p=probs)
    return modified


# ---------------------------------------------------------------------------
# Detectors (self-contained, no imports from ide/)
# ---------------------------------------------------------------------------

EPS = 1e-9


def _freq_vector(window: list[str], vocab: list[str]) -> np.ndarray:
    counts = Counter(window)
    vec = np.array([counts.get(a, 0) for a in vocab], dtype=float)
    s = vec.sum()
    return vec / s if s > 0 else vec


def _cosine_dist(p: np.ndarray, q: np.ndarray) -> float:
    n_p = np.linalg.norm(p)
    n_q = np.linalg.norm(q)
    if n_p < EPS or n_q < EPS:
        return 0.0
    return float(1.0 - np.dot(p, q) / (n_p * n_q))


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5) * np.linalg.norm(np.sqrt(p + EPS) - np.sqrt(q + EPS)))


def _mock_embedding(action: str, dim: int = 32) -> np.ndarray:
    """Deterministic hash-projection embedding (matches MockEmbedder in ide/)."""
    vec = np.array(
        [(hash(action + str(j)) % 256) / 128.0 - 1.0 for j in range(dim)],
        dtype=float,
    )
    norm = np.linalg.norm(vec)
    return vec / norm if norm > EPS else vec


def run_cbd(trace: list[str], baseline: dict[str, float], vocab: list[str]) -> int | None:
    """Fixed-threshold CBD. Returns detection step or None."""
    b = np.array([baseline.get(a, EPS) for a in vocab])
    b /= b.sum()
    window: list[str] = []
    for i, action in enumerate(trace):
        window.append(action)
        if len(window) > WINDOW_SIZE:
            window.pop(0)
        if i < CALIB_SIZE:
            continue
        p = _freq_vector(window, vocab)
        if _cosine_dist(p, b) >= 0.35:
            return i
    return None


def run_hellinger_sprt(
    trace: list[str],
    baseline: dict[str, float],
    alt_dist: dict[str, float],
    vocab: list[str],
) -> int | None:
    """Hellinger-distance SPRT sequential test."""
    b = np.array([baseline.get(a, EPS) for a in vocab])
    alt = np.array([alt_dist.get(a, EPS) for a in vocab])
    b /= b.sum(); alt /= alt.sum()
    log_A = np.log(SPRT_BETA / (1 - SPRT_ALPHA))
    log_B = np.log((1 - SPRT_BETA) / SPRT_ALPHA)
    S = 0.0
    window: list[str] = []
    for i, action in enumerate(trace):
        window.append(action)
        if len(window) > WINDOW_SIZE:
            window.pop(0)
        if i < CALIB_SIZE:
            continue
        if action not in vocab:
            return i  # DRIFT_NEW_ACTION
        idx = vocab.index(action)
        h_null = _hellinger(_freq_vector(window, vocab), b)
        h_alt = _hellinger(_freq_vector(window, vocab), alt)
        log_lr = np.log((h_null + EPS) / (h_alt + EPS))
        S += log_lr
        if S >= log_B:
            return i
        if S <= log_A:
            S = 0.0
    return None


def run_lrt(
    trace: list[str],
    baseline: dict[str, float],
    alt_dist: dict[str, float],
    vocab: list[str],
) -> int | None:
    """LRT-SPRT sequential test (frequency-only)."""
    b = np.array([baseline.get(a, EPS) for a in vocab])
    alt = np.array([alt_dist.get(a, EPS) for a in vocab])
    b /= b.sum(); alt /= alt.sum()
    log_A = np.log(SPRT_BETA / (1 - SPRT_ALPHA))
    log_B = np.log((1 - SPRT_BETA) / SPRT_ALPHA)
    S = 0.0
    for i, action in enumerate(trace):
        if i < CALIB_SIZE:
            continue
        if action not in vocab:
            return i
        idx = vocab.index(action)
        log_lr = np.log((alt[idx] + EPS) / (b[idx] + EPS))
        S += log_lr
        if S >= log_B:
            return i
        if S <= log_A:
            S = 0.0
    return None


def run_ss_cbd(
    trace: list[str],
    baseline: dict[str, float],
    alt_dist: dict[str, float],
    vocab: list[str],
    lam: float = LAMBDA_SEM,
) -> int | None:
    """SS-CBD: SPRT + semantic embeddings + tier weighting (simplified, no tiers)."""
    b = np.array([baseline.get(a, EPS) for a in vocab])
    alt = np.array([alt_dist.get(a, EPS) for a in vocab])
    b /= b.sum(); alt /= alt.sum()

    # Baseline semantic centroid
    embeddings = np.array([_mock_embedding(a) for a in vocab])
    centroid_b = embeddings.T @ b
    norm_c = np.linalg.norm(centroid_b)
    if norm_c > EPS:
        centroid_b /= norm_c

    log_A = np.log(SPRT_BETA / (1 - SPRT_ALPHA))
    log_B = np.log((1 - SPRT_BETA) / SPRT_ALPHA)
    S = 0.0

    for i, action in enumerate(trace):
        if i < CALIB_SIZE:
            continue
        if action not in vocab:
            return i  # DRIFT_NEW_ACTION
        idx = vocab.index(action)
        freq_lr = np.log((alt[idx] + EPS) / (b[idx] + EPS))
        sem_dist = float(1.0 - np.dot(embeddings[idx], centroid_b))
        sign_fr = 1.0 if freq_lr >= 0 else -1.0
        log_lr = (1 - lam) * freq_lr + lam * sem_dist * sign_fr
        S += log_lr
        if S >= log_B:
            return i
        if S <= log_A:
            S = 0.0
    return None


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_archetype(
    archetype_name: str,
    sessions: list[str],
    drift_targets: list[str],
    vocab: list[str],
) -> dict:
    """Run all detectors on N_TRIALS drift+control traces. Return summary dict."""
    rng_local = np.random.default_rng(SEED)

    # Build long traces
    drift_traces = build_long_traces(sessions, TRACE_LENGTH, N_TRIALS, rng_local)

    # Compute baseline distribution from natural session distribution
    all_actions = [a for sess in sessions for a in sess]
    counts = Counter(all_actions)
    total = sum(counts.values())
    baseline = {a: counts[a] / total for a in vocab}

    # Drift distribution
    alt_dist = build_drift_dist(baseline, drift_targets, shift_factor=4.0)

    results = {m: {"det": [], "t": [], "fpr_det": []} for m in ["SS-CBD", "LRT", "Hellinger", "CBD"]}

    for trial_idx, base_trace in enumerate(drift_traces):
        # --- Drift trace ---
        drift_trace = inject_drift(base_trace, DRIFT_ONSET, alt_dist, DRIFT_FRACTION, rng_local)
        for method, fn in [
            ("SS-CBD", lambda t: run_ss_cbd(t, baseline, alt_dist, vocab)),
            ("LRT",    lambda t: run_lrt(t, baseline, alt_dist, vocab)),
            ("Hellinger", lambda t: run_hellinger_sprt(t, baseline, alt_dist, vocab)),
            ("CBD",    lambda t: run_cbd(t, baseline, vocab)),
        ]:
            det_step = fn(drift_trace)
            results[method]["det"].append(det_step is not None)
            results[method]["t"].append(det_step - CALIB_SIZE if det_step is not None else None)

        # --- Control trace (no drift) ---
        for method, fn in [
            ("SS-CBD", lambda t: run_ss_cbd(t, baseline, alt_dist, vocab)),
            ("LRT",    lambda t: run_lrt(t, baseline, alt_dist, vocab)),
            ("Hellinger", lambda t: run_hellinger_sprt(t, baseline, alt_dist, vocab)),
            ("CBD",    lambda t: run_cbd(t, baseline, vocab)),
        ]:
            fp = fn(base_trace)
            results[method]["fpr_det"].append(fp is not None)

    # Compute summary stats
    summary = {}
    for method, data in results.items():
        det_arr = np.array(data["det"])
        t_arr = [t for t in data["t"] if t is not None]
        fpr_arr = np.array(data["fpr_det"])
        summary[method] = {
            "det_rate": float(det_arr.mean() * 100),
            "fpr": float(fpr_arr.mean() * 100),
            "mean_det": float(np.mean(t_arr)) if t_arr else float("nan"),
            "median_det": float(np.median(t_arr)) if t_arr else float("nan"),
            "std_det": float(np.std(t_arr)) if t_arr else float("nan"),
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("EXP9: Real Agent Trace Evaluation")
    print(f"N={N_TRIALS} trials, seed={SEED}, trace_length={TRACE_LENGTH}")
    print(f"calib_size={CALIB_SIZE}, drift_onset={DRIFT_ONSET}, drift_fraction={DRIFT_FRACTION}")
    print("=" * 70)

    # --- Archetype 1: Knowledge-Retrieval Agent (AgentBench KG) ---
    print("\n[1/2] Loading AgentBench KG sessions...")
    kg_sessions = parse_agentbench_kg()
    print(f"  Sessions: {len(kg_sessions)}, total actions: {sum(len(s) for s in kg_sessions)}")
    kg_vocab = ["get_relations", "get_neighbors", "intersection",
                "get_attributes", "argmax", "count", "argmin"]
    kg_drift_targets = ["count", "argmax", "argmin"]  # shift toward aggregation ops
    print(f"  Vocab ({len(kg_vocab)}): {kg_vocab}")
    print(f"  Drift targets: {kg_drift_targets}")

    kg_results = evaluate_archetype(
        "KnowledgeAgent", kg_sessions, kg_drift_targets, kg_vocab
    )

    # --- Archetype 2: Coding/Security Agent (SWE-agent CTF, N=10) ---
    print("\n[2/3] Loading SWE-agent CTF sessions (N=10, run_shell dominant)...")
    swe_sessions = parse_swe_agent()
    print(f"  Sessions: {len(swe_sessions)}, total actions: {sum(len(s) for s in swe_sessions)}")
    swe_vocab = ["view_file", "edit_file", "run_code", "run_shell",
                 "create_file", "search_code", "list_files", "delete_file", "submit"]
    swe_drift_targets = ["run_shell", "run_code"]  # shift toward execution (escalation)
    print(f"  Vocab ({len(swe_vocab)}): {swe_vocab}")
    print(f"  Drift targets: {swe_drift_targets}")

    swe_results = evaluate_archetype(
        "CodingAgent", swe_sessions, swe_drift_targets, swe_vocab
    )

    # --- Archetype 3: SWE-bench Coding Agent (sailplane, N=400, edit_file dominant) ---
    print("\n[3/3] Loading sailplane SWE-bench coding sessions (N=400, edit_file dominant)...")
    coding_sessions = parse_sailplane_coding()
    print(f"  Sessions: {len(coding_sessions)}, total actions: {sum(len(s) for s in coding_sessions)}")
    print(f"  Vocab ({len(swe_vocab)}): {swe_vocab}")
    print(f"  Drift targets: {swe_drift_targets}")

    coding_results = evaluate_archetype(
        "CodingAgent_sailplane", coding_sessions, swe_drift_targets, swe_vocab
    )

    # --- Print results ---
    def print_table(name: str, res: dict) -> None:
        print(f"\n{'─'*70}")
        print(f"  Archetype: {name}")
        print(f"{'─'*70}")
        print(f"  {'Method':<14} {'Det%':>7} {'FPR%':>7} {'Mean T':>8} {'Median T':>9} {'Speedup vs LRT':>15}")
        print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*8} {'─'*9} {'─'*15}")
        lrt_mean = res["LRT"]["mean_det"]
        for m in ["SS-CBD", "LRT", "Hellinger", "CBD"]:
            r = res[m]
            speedup = lrt_mean / r["mean_det"] if r["mean_det"] and r["mean_det"] > 0 else float("nan")
            sp_str = f"{speedup:.2f}×" if not (speedup != speedup) else "—"
            print(f"  {m:<14} {r['det_rate']:>6.1f}% {r['fpr']:>6.1f}% "
                  f"{r['mean_det']:>7.1f}  {r['median_det']:>8.1f}  {sp_str:>14}")

    print_table("Knowledge-Retrieval Agent (AgentBench KG)", kg_results)
    print_table("Coding/Security Agent — CTF (SWE-agent, N=10)", swe_results)
    print_table("Coding/Security Agent — SWE-bench (sailplane, N=400)", coding_results)

    # --- Finding #2 validation ---
    print("\n" + "=" * 70)
    print("FINDING #2 VALIDATION (LRT FPR on coding agents)")
    print("=" * 70)
    print(f"  LRT FPR — CTF sessions (run_shell 45%):       {swe_results['LRT']['fpr']:.1f}%")
    print(f"  SS-CBD FPR — CTF sessions:                    {swe_results['SS-CBD']['fpr']:.1f}%")
    print(f"  LRT FPR — SWE-bench coding (run_shell 4%):   {coding_results['LRT']['fpr']:.1f}%")
    print(f"  SS-CBD FPR — SWE-bench coding:                {coding_results['SS-CBD']['fpr']:.1f}%")

    # --- Save results ---
    output = {
        "metadata": {
            "n_trials": N_TRIALS, "seed": SEED, "trace_length": TRACE_LENGTH,
            "calib_size": CALIB_SIZE, "drift_onset": DRIFT_ONSET,
            "drift_fraction": DRIFT_FRACTION,
            "datasets": {
                "agentbench_kg": {
                    "sessions": len(kg_sessions),
                    "total_actions": sum(len(s) for s in kg_sessions),
                    "vocab": kg_vocab, "drift_targets": kg_drift_targets,
                },
                "swe_agent_ctf": {
                    "sessions": len(swe_sessions),
                    "total_actions": sum(len(s) for s in swe_sessions),
                    "vocab": swe_vocab, "drift_targets": swe_drift_targets,
                    "note": "CTF sessions only — run_shell dominant (45%)",
                },
                "swe_coding_sailplane": {
                    "sessions": len(coding_sessions),
                    "total_actions": sum(len(s) for s in coding_sessions),
                    "vocab": swe_vocab, "drift_targets": swe_drift_targets,
                    "source": "sailplane/swe-agent-trajs (HuggingFace)",
                    "note": "SWE-bench coding — edit_file dominant (31%), run_shell 4%",
                },
            },
        },
        "results": {
            "knowledge_agent": kg_results,
            "coding_agent_ctf": swe_results,
            "coding_agent_swebench": coding_results,
        },
    }
    out_path = OUTPUT_DIR / "RESULTS_EXP9.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Markdown report ---
    md_lines = [
        "# EXP9: Real Agent Trace Evaluation (Extended)",
        "",
        f"**N={N_TRIALS} trials, seed={SEED}. Three real agent archetypes.**",
        "",
        "## Datasets",
        "",
        f"- **AgentBench KG** (Liu et al., ICLR 2024): {len(kg_sessions)} sessions, "
        f"{sum(len(s) for s in kg_sessions)} total actions. GPT-4 KG traversal.",
        f"- **SWE-agent CTF** (Yang et al., NeurIPS 2024): {len(swe_sessions)} sessions, "
        f"{sum(len(s) for s in swe_sessions)} total actions. CTF challenges. run_shell dominant (~45%).",
        f"- **SWE-bench Coding** (sailplane/swe-agent-trajs, HuggingFace): {len(coding_sessions)} sessions, "
        f"{sum(len(s) for s in coding_sessions)} total actions. edit_file dominant (~31%), run_shell ~4%.",
        "",
        "## Results",
        "",
    ]

    for arch_name, res in [
        ("Knowledge-Retrieval Agent (AgentBench KG)", kg_results),
        ("Coding/Security CTF (SWE-agent, N=10)", swe_results),
        ("SWE-bench Coding (sailplane, N=400)", coding_results),
    ]:
        lrt_mean = res["LRT"]["mean_det"]
        md_lines += [
            f"### {arch_name}",
            "",
            "| Method | Det% | FPR% | Mean Det (act) | Median Det (act) | Speedup vs LRT |",
            "|--------|------|------|---------------|-----------------|----------------|",
        ]
        for m in ["SS-CBD", "LRT", "Hellinger", "CBD"]:
            r = res[m]
            sp = lrt_mean / r["mean_det"] if r["mean_det"] and r["mean_det"] > 0 else float("nan")
            sp_str = f"{sp:.2f}×" if sp == sp else "—"
            md_lines.append(
                f"| {'**'+m+'**' if m=='SS-CBD' else m} "
                f"| {r['det_rate']:.1f}% "
                f"| {r['fpr']:.1f}% "
                f"| {r['mean_det']:.1f} "
                f"| {r['median_det']:.1f} "
                f"| {sp_str} |"
            )
        md_lines.append("")

    md_lines += [
        "## Notes",
        "",
        "- Long traces built by sequential concatenation of real sessions (§8.4 protocol).",
        "- Drift injection: 50% of post-onset actions replaced with drift-target distribution.",
        f"- Drift onset at action {DRIFT_ONSET} of {TRACE_LENGTH}-action trace.",
        "- FPR measured on unmodified (no drift injection) traces.",
        "- sailplane sessions sourced from sailplane/swe-agent-trajs (HuggingFace, Claude-3.5-Sonnet runs).",
        "",
        f"*Generated by `python -m experiments.exp9_real_traces`, seed={SEED}, N={N_TRIALS}.*",
    ]

    md_path = OUTPUT_DIR / "RESULTS_EXP9.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n✓ Results saved: {out_path}")
    print(f"✓ Report saved: {md_path}")


if __name__ == "__main__":
    main()
