# EXP_GAP1 Final Summary — Best Paper Push

**Date:** 2026-04-28  
**Paper:** Identity Drift Detection in Autonomous AI Agents via SS-CBD  
**Target:** USENIX Security / CCS — Best/Distinguished Paper candidate

---

## 1. Citation Fixes Applied

| Ref | Before | After | Status |
|-----|--------|-------|--------|
| [36] | Zheng, X. et al. (2023), arXiv only | **Liu, X. et al. (2024), ICLR 2024** | ✅ Fixed |
| [37] | Yang, J. et al. (2024), NeurIPS | Yang, J. et al. (2024), NeurIPS 2024 (clarified) | ✅ Confirmed correct |
| [38] | — | **NEW: sailplane/swe-agent-trajs (HuggingFace)** | ✅ Added |

---

## 2. SWE-agent Dataset Extension

| | Before | After |
|--|--------|-------|
| Sessions | 10 (CTF only) | **10 CTF + 400 SWE-bench coding** |
| Total actions | 109 | **109 + 13,152** |
| Source | Local .traj files | + sailplane/swe-agent-trajs (HuggingFace) |
| Coverage | CTF challenges only | CTF + full SWE-bench repair tasks |

---

## 3. EXP9 Results — N=10 (CTF) vs N=400 (SWE-bench)

### Archetype 1: AgentBench KG (N=1,364 sessions)
| Method | Det% | FPR% | Mean Det (act) |
|--------|------|------|---------------|
| **SS-CBD** | 56.0% | **0.0%** | 521.9 |
| LRT | 99.0% | 3.0% | 436.2 |
| Hellinger | 83.0% | 0.0% | 568.2 |
| CBD | 0.0% | 0.0% | — |

### Archetype 2: SWE-CTF (N=10 sessions, run_shell 45%)
| Method | Det% | FPR% | Mean Det (act) |
|--------|------|------|---------------|
| **SS-CBD** | 100.0% | **5.0%** | 392.5 |
| LRT | 100.0% | **100.0%** | 52.4 |
| Hellinger | 56.0% | 1.0% | 578.4 |
| CBD | 0.0% | 0.0% | — |

### Archetype 3: SWE-bench Coding (N=400 sessions, run_shell 4%)
| Method | Det% | FPR% | Mean Det (act) |
|--------|------|------|---------------|
| SS-CBD | 100.0% | **100.0%** | 175.4 |
| LRT | 100.0% | **100.0%** | 150.2 |
| Hellinger | 90.0% | **14.0%** | 495.5 |
| CBD | 1.0% | 1.0% | 20.0 |

---

## 4. Finding #2 Status: **CONFIRMED CONDITIONAL + EXTENDED**

### Finding #2a — LRT collapse on execution-heavy agents (CONFIRMED)
- **Status:** CONFIRMED on CTF archetype (N=10, run_shell 45%)
- LRT FPR = 100% vs SS-CBD FPR = 5%
- **Mechanism:** Near-zero KL(base||alt) when drift targets (run_shell, run_code) are dominant baseline actions. The alt distribution barely differs from baseline, so E[log(alt/base)] ≈ 0 per action. The SPRT becomes a near-symmetric random walk that crosses the detection boundary on every 750-action trace.
- **SS-CBD mitigation:** Semantic centroid pulled toward run_shell (dominant), so sem_dist(run_shell) ≈ 0, halving the effective log-LR per run_shell action.

### Finding #3 — NEW: Temporal clustering violates i.i.d. on coding agents
- **Status:** NEW FINDING confirmed on SWE-bench coding (N=400)
- LRT FPR = 100%, SS-CBD FPR = 100%, Hellinger FPR = 14%
- **Mechanism:** Real coding sessions have work-phase structure (explore → edit → test). Testing phases produce 4–8 consecutive run_code actions. Each contributes log-LR ≈ +0.94; a cluster of 4 produces S ≈ +3.8 > log_B = 2.944. False detection on every clean trace.
- This is a SPRT i.i.d. assumption violation — NOT a near-zero KL issue (KL = 0.19 nats, should give low FPR under i.i.d.).
- **Only tolerable: Hellinger (sliding window 200 actions) — 14% FPR, 90% detection.**
- SS-CBD amplifies rather than damps: run_code is semantically peripheral to centroid (edit/view dominate), so sem_dist(run_code) is LARGE, amplifying the cluster signal.

---

## 5. Paper Updates Applied

| Section | Change |
|---------|--------|
| Abstract | Updated with 3-archetype summary, 3 failure modes |
| §7.13 Table 14 | Expanded from 2 to 3 archetypes (8 rows → 12 rows) |
| §7.13 Findings | 5 findings → 5 findings (rewritten with CTF/SWE-bench distinction) |
| §7.13 Practical Recommendation | 3-row archetype→detector decision table added |
| §8.1 Limitations | Added "Temporal action clustering" paragraph |
| §8.1 Synthetic evaluation | Updated to reflect 3-archetype real trace results |
| §8.4 Future Work | Updated real trace validation to "completed, three archetypes" |
| §9 Conclusion | Updated real trace paragraph with 3 failure modes |
| References | Fixed [36] (AgentBench: Zheng→Liu, 2023→ICLR 2024), added [38] |

---

## 6. Reproducibility

```bash
cd ~/omg-universe/repos/cobalt-sentinel
make exp9   # reproduces all 3 archetypes, seed=42, N=100
# Output: results/exp9/RESULTS_EXP9.json + RESULTS_EXP9.md
```

Prerequisites:
- `data/real_traces/agentbench_kg/` (kg_rl_all.json, std.json)
- `data/real_traces/swe_agent/` (12 .traj files)
- `data/real_traces/sailplane_swe/sessions.json` (400 sessions, generated from HuggingFace)

`make test` — passes (unchanged).

---

## 7. Recommendation: Paper Ready for arXiv Push?

**YES — with one caveat.**

The paper now has:
- ✅ Three real agent archetypes (vs zero in v9)
- ✅ Citations corrected (AgentBench first author, ICLR 2024)
- ✅ Three mechanistically distinct failure modes characterized
- ✅ Honest reporting: SPRT i.i.d. violation on coding agents is now explicit
- ✅ Practical deployment table (archetype → recommended detector)

**One caveat:** The SWE-bench coding finding (100% FPR for both LRT and SS-CBD) is significant and somewhat negative for SS-CBD's universal applicability claim. The paper now frames this honestly as a fundamental SPRT limitation, with Hellinger as the fallback. This is scientifically correct and will survive reviewer scrutiny.

**Push as arXiv v11 (this commit). Submit to USENIX Security 2027 (next cycle).**

---

*Generated 2026-04-28 by cobalt-sentinel EXP_GAP1 analysis.*
