"""EXP9b: Extended real agent trace evaluation — Finding #2 validation.

Extends EXP9 with 400 SWE-bench coding sessions from sailplane/swe-agent-trajs
(HuggingFace) to validate whether LRT 100% FPR is a real effect or a CTF-session
artifact.

Three archetypes evaluated:
  1. AgentBench KG     — knowledge-retrieval (unchanged from EXP9)
  2. SWE-CTF           — local CTF sessions (run_shell dominant, N=10)
  3. SWE-Coding        — sailplane SWE-bench coding sessions (edit_file dominant, N=400)

This allows parametric comparison:
  - CTF archetype: drift targets (run_shell, run_code) = dominant baseline actions
  - Coding archetype: drift targets (run_shell, run_code) = rare baseline actions
  → characterizes LRT FPR collapse condition

Usage:
    python -m experiments.exp9b_real_traces_extended
    make exp9b
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
OUTPUT_DIR = Path("results/exp9b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 100
SEED = 42
TRACE_LENGTH = 800
CALIB_SIZE = 50
DRIFT_ONSET = 400
DRIFT_FRACTION = 0.50
WINDOW_SIZE = 200
SPRT_ALPHA = SPRT_BETA = 0.05
LAMBDA_SEM = 0.5

EPS = 1e-9


# ---------------------------------------------------------------------------
# Parsers (identical to exp9 for AgentBench KG and SWE-CTF)
# ---------------------------------------------------------------------------

def parse_agentbench_kg() -> list[list[str]]:
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


def parse_swe_ctf() -> list[list[str]]:
    """Parse local SWE-agent CTF .traj files (N=10, run_shell dominant)."""
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
            blocks = re.findall(r"```(?:\w+)?\n([^`]+)", content)
            for block in blocks:
                first_line = block.strip().split("\n")[0].strip()
                if not first_line:
                    continue
                cmd = first_line.split()[0].lower()
                cmd = _normalize_swe_cmd(cmd)
                acts.append(cmd)
                break
        if acts:
            sessions.append(acts)
    return sessions


def parse_swe_coding() -> list[list[str]]:
    """Parse sailplane SWE-bench coding sessions (N=400, edit_file dominant)."""
    path = DATA_DIR / "sailplane_swe" / "sessions.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [item["acts"] for item in data if len(item["acts"]) >= 3]


def _normalize_swe_cmd(cmd: str) -> str:
    mapping = {
        "open": "view_file", "view": "view_file", "cat": "view_file",
        "head": "view_file", "tail": "view_file", "grep": "search_code",
        "find": "search_code", "ls": "list_files", "edit": "edit_file",
        "str_replace": "edit_file", "create": "create_file", "rm": "delete_file",
        "mkdir": "create_file", "python": "run_code", "python3": "run_code",
        "pytest": "run_code", "bash": "run_shell", "sh": "run_shell",
        "/bin/sh": "run_shell", "submit": "submit", "exit": "submit",
    }
    return mapping.get(cmd, "run_shell")


# ---------------------------------------------------------------------------
# Trace construction, drift injection, detectors (identical to exp9)
# ---------------------------------------------------------------------------

def build_long_traces(sessions, trace_length, n_trials, rng):
    flat_actions = [a for sess in sessions for a in sess]
    traces = []
    for _ in range(n_trials):
        trace = []
        idx = rng.integers(0, len(flat_actions))
        while len(trace) < trace_length:
            trace.append(flat_actions[idx % len(flat_actions)])
            idx += 1
        traces.append(trace[:trace_length])
    return traces


def build_drift_dist(baseline, drift_targets, shift_factor=3.0):
    drift = dict(baseline)
    for act in drift_targets:
        if act in drift:
            drift[act] *= shift_factor
        else:
            drift[act] = shift_factor * min(baseline.values(), default=0.01)
    total = sum(drift.values())
    return {k: v / total for k, v in drift.items()}


def inject_drift(trace, onset, drift_dist, fraction, rng):
    modified = list(trace)
    vocab = list(drift_dist.keys())
    probs = np.array([drift_dist[v] for v in vocab])
    probs /= probs.sum()
    for i in range(onset, len(trace)):
        if rng.random() < fraction:
            modified[i] = rng.choice(vocab, p=probs)
    return modified


def _freq_vector(window, vocab):
    counts = Counter(window)
    vec = np.array([counts.get(a, 0) for a in vocab], dtype=float)
    s = vec.sum()
    return vec / s if s > 0 else vec


def _cosine_dist(p, q):
    n_p, n_q = np.linalg.norm(p), np.linalg.norm(q)
    if n_p < EPS or n_q < EPS:
        return 0.0
    return float(1.0 - np.dot(p, q) / (n_p * n_q))


def _hellinger(p, q):
    return float(np.sqrt(0.5) * np.linalg.norm(np.sqrt(p + EPS) - np.sqrt(q + EPS)))


def _mock_embedding(action, dim=32):
    vec = np.array(
        [(hash(action + str(j)) % 256) / 128.0 - 1.0 for j in range(dim)], dtype=float
    )
    norm = np.linalg.norm(vec)
    return vec / norm if norm > EPS else vec


def run_cbd(trace, baseline, vocab):
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


def run_lrt(trace, baseline, alt_dist, vocab):
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


def run_ss_cbd(trace, baseline, alt_dist, vocab, lam=LAMBDA_SEM):
    b = np.array([baseline.get(a, EPS) for a in vocab])
    alt = np.array([alt_dist.get(a, EPS) for a in vocab])
    b /= b.sum(); alt /= alt.sum()
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
            return i
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

def compute_kl(baseline, alt_dist, vocab):
    """KL(base||alt) — drives LRT FPR: higher KL = lower FPR."""
    kl = 0.0
    for a in vocab:
        p = baseline.get(a, EPS)
        q = alt_dist.get(a, EPS)
        if p > EPS:
            kl += p * np.log(p / q)
    return kl


def evaluate_archetype(archetype_name, sessions, drift_targets, vocab):
    rng_local = np.random.default_rng(SEED)
    drift_traces = build_long_traces(sessions, TRACE_LENGTH, N_TRIALS, rng_local)
    all_actions = [a for sess in sessions for a in sess]
    counts = Counter(all_actions)
    total = sum(counts.values())
    baseline = {a: counts.get(a, 0) / total for a in vocab}
    # Add EPS for vocab actions with zero count
    baseline = {a: max(baseline.get(a, 0), EPS) for a in vocab}
    total_b = sum(baseline.values())
    baseline = {a: v / total_b for a, v in baseline.items()}
    alt_dist = build_drift_dist(baseline, drift_targets, shift_factor=4.0)
    kl = compute_kl(baseline, alt_dist, vocab)

    results = {m: {"det": [], "t": [], "fpr_det": []} for m in ["SS-CBD", "LRT", "CBD"]}

    for base_trace in drift_traces:
        drift_trace = inject_drift(base_trace, DRIFT_ONSET, alt_dist, DRIFT_FRACTION, rng_local)
        for method, fn in [
            ("SS-CBD", lambda t: run_ss_cbd(t, baseline, alt_dist, vocab)),
            ("LRT",    lambda t: run_lrt(t, baseline, alt_dist, vocab)),
            ("CBD",    lambda t: run_cbd(t, baseline, vocab)),
        ]:
            det_step = fn(drift_trace)
            results[method]["det"].append(det_step is not None)
            results[method]["t"].append(det_step - CALIB_SIZE if det_step is not None else None)
        for method, fn in [
            ("SS-CBD", lambda t: run_ss_cbd(t, baseline, alt_dist, vocab)),
            ("LRT",    lambda t: run_lrt(t, baseline, alt_dist, vocab)),
            ("CBD",    lambda t: run_cbd(t, baseline, vocab)),
        ]:
            fp = fn(base_trace)
            results[method]["fpr_det"].append(fp is not None)

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
        }

    # Natural distribution stats
    dominant_target_freq = sum(baseline.get(a, 0) for a in drift_targets)

    return summary, kl, dominant_target_freq, baseline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXP9b: Extended Real Agent Trace Evaluation — Finding #2 Validation")
    print(f"N={N_TRIALS} trials, seed={SEED}, trace_length={TRACE_LENGTH}")
    print("=" * 70)

    # --- AgentBench KG ---
    print("\n[1/3] Loading AgentBench KG sessions...")
    kg_sessions = parse_agentbench_kg()
    print(f"  Sessions: {len(kg_sessions)}")
    kg_vocab = ["get_relations", "get_neighbors", "intersection",
                "get_attributes", "argmax", "count", "argmin"]
    kg_drift_targets = ["count", "argmax", "argmin"]

    kg_results, kg_kl, kg_dom, kg_base = evaluate_archetype(
        "KG", kg_sessions, kg_drift_targets, kg_vocab
    )

    # --- SWE-CTF (local, run_shell dominant) ---
    print("\n[2/3] Loading SWE-agent CTF sessions (local, N=10)...")
    ctf_sessions = parse_swe_ctf()
    print(f"  Sessions: {len(ctf_sessions)}, total actions: {sum(len(s) for s in ctf_sessions)}")
    swe_vocab = ["view_file", "edit_file", "run_code", "run_shell",
                 "create_file", "search_code", "list_files", "delete_file", "submit"]
    swe_drift_targets = ["run_shell", "run_code"]

    ctf_results, ctf_kl, ctf_dom, ctf_base = evaluate_archetype(
        "CTF", ctf_sessions, swe_drift_targets, swe_vocab
    )

    # --- SWE-Coding (sailplane, edit_file dominant) ---
    print("\n[3/3] Loading sailplane SWE-bench coding sessions...")
    coding_sessions = parse_swe_coding()
    print(f"  Sessions: {len(coding_sessions)}, total actions: {sum(len(s) for s in coding_sessions)}")

    coding_results, coding_kl, coding_dom, coding_base = evaluate_archetype(
        "Coding", coding_sessions, swe_drift_targets, swe_vocab
    )

    # --- Print results ---
    def print_table(name, res, kl, dom_freq):
        print(f"\n{'─'*70}")
        print(f"  Archetype: {name}")
        print(f"  KL(base||alt)={kl:.4f}  dominant_target_freq={dom_freq:.3f}")
        print(f"{'─'*70}")
        print(f"  {'Method':<14} {'Det%':>7} {'FPR%':>7} {'Mean T':>8}")
        print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*8}")
        for m in ["SS-CBD", "LRT", "CBD"]:
            r = res[m]
            print(f"  {m:<14} {r['det_rate']:>6.1f}% {r['fpr']:>6.1f}% {r['mean_det']:>7.1f}")

    print_table("Knowledge-Retrieval (AgentBench KG)", kg_results, kg_kl, kg_dom)
    print_table("SWE-CTF (local, run_shell dominant)", ctf_results, ctf_kl, ctf_dom)
    print_table("SWE-Coding (sailplane, edit_file dominant)", coding_results, coding_kl, coding_dom)

    # --- Summary analysis ---
    print("\n" + "=" * 70)
    print("FINDING #2 VALIDATION")
    print("=" * 70)
    lrt_ctf_fpr = ctf_results["LRT"]["fpr"]
    lrt_coding_fpr = coding_results["LRT"]["fpr"]
    print(f"\nLRT FPR — SWE-CTF (run_shell 45%):     {lrt_ctf_fpr:.1f}%")
    print(f"LRT FPR — SWE-Coding (run_shell 4%):    {lrt_coding_fpr:.1f}%")
    print(f"\nKL(base||alt) — CTF:    {ctf_kl:.4f} nats")
    print(f"KL(base||alt) — Coding: {coding_kl:.4f} nats")
    print(f"\nDrift target dominance — CTF:    {ctf_dom:.3f}")
    print(f"Drift target dominance — Coding: {coding_dom:.3f}")

    if lrt_ctf_fpr >= 80 and lrt_coding_fpr <= 20:
        status = "CONFIRMED_CONDITIONAL"
        print("\n>>> Finding #2: CONFIRMED (CONDITIONAL)")
        print("    LRT collapse is real but contingent on drift-target dominance.")
        print("    CTF agents (high run_shell baseline): LRT unusable.")
        print("    Coding agents (low run_shell baseline): LRT works fine.")
    elif lrt_ctf_fpr >= 80 and lrt_coding_fpr > 20:
        status = "CONFIRMED_PARTIAL"
        print("\n>>> Finding #2: CONFIRMED (PARTIAL) — LRT elevated on coding too")
    elif lrt_ctf_fpr < 80:
        status = "ATTENUATED"
        print("\n>>> Finding #2: ATTENUATED — CTF FPR < 80% with full session pool")
    else:
        status = "UNCERTAIN"

    # --- Save ---
    output = {
        "metadata": {
            "n_trials": N_TRIALS, "seed": SEED, "trace_length": TRACE_LENGTH,
            "calib_size": CALIB_SIZE, "drift_onset": DRIFT_ONSET,
            "finding2_status": status,
        },
        "results": {
            "kg": {"summary": kg_results, "kl": kg_kl, "dominant_target_freq": kg_dom},
            "swe_ctf": {"summary": ctf_results, "kl": ctf_kl, "dominant_target_freq": ctf_dom,
                        "sessions": len(ctf_sessions)},
            "swe_coding": {"summary": coding_results, "kl": coding_kl, "dominant_target_freq": coding_dom,
                           "sessions": len(coding_sessions)},
        },
    }
    out_path = OUTPUT_DIR / "RESULTS_EXP9b.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved: {out_path}")


if __name__ == "__main__":
    main()
