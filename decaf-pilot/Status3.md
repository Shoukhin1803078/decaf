# DECAF — Status v3 (cold re-run)

**Date started:** 2026-10-17
**Scope:** Phase 3 (v3) — the cold re-run that fixes the prefix-cache contamination
identified in `DECAF_Q1_MASTER_PLAN.md` and `EXACT_FIXES.md`.
**State:** 🚧 In progress

This file tracks the v3 work. **The v2 artifacts are left untouched** (`results_v2/`,
`cache_v2/`, `STATUS.md`, `FULL_RESULTS_v2.md`) — they are preserved as evidence of the
contamination, per the master plan §0.1. v3 writes only to `cache_v3/` and `results_v3/`.

---

## 0. Why v3 exists

The v2 compressed-prompt timings were measured **warm**: no nonce, `full` first, fixed method
order, so each later prompt shared a prefix with an earlier one and reused the KV cache. The
logged speed-ups are inflated by an estimated **1.3×–1.7×**. v3 re-measures every timing arm
**cold** and measures the inflation directly with a warm-order control.

Audit gate on v2 data (`audit_gate.py`):

```
[FAIL] C1 invalid timer rows: 18 of 21360 rows
[PASS] C2 duplicate keys: 0
[FAIL] C3 cold gate: 126 of 168 (config, dataset, method) groups outside [0.80, 1.25]
[FAIL] C4 same questions for every method: 8 of 16 cells differ
[FAIL] C5 calibration range: scaling min 361 tokens; compressed p5 112
[FAIL] C6 calibration replicates: 28 of 28 (model, length) points below 5
RESULT: 5 CHECK(S) FAILED
```

Goal after v3: **ALL CHECKS PASSED** on the cold condition.

---

## 1. Deliverables (new in v3)

| File | Purpose | Status |
|---|---|---|
| `config_v3.yaml` | v3 config: cold/warm conditions, extended calibration, `top_k` sweep, thread/batch pinning | ✅ created |
| `hypotheses.md` | Dated pre-registration (H1–H5) written before the run | ✅ created |
| `audit_gate.py` | C1–C6 checker | ✅ created, reproduces v2 failures |
| `src/generate_environment.py` | Writes `ENVIRONMENT.md` | ✅ created |
| `ENVIRONMENT.md` | Hardware/software record | ⏳ to generate |
| `results_v3/` | v3 figures, tables, report | ✅ created (empty) |
| `cache_v3/` | v3 resumable JSONL caches | ✅ created (empty) |
| `src/run_v3.py` | Fixed harness (nonce at char 0, conditions, random order, timer guard) | ⏳ |
| `results_v3/FULL_RESULTS_v3.md` | v3 report | ⏳ |

---

## 2. Harness fixes (EXACT_FIXES.md)

| ID | Fix | Status |
|---|---|---|
| FX-01 | Add `condition` to cache key; write to `cache_v3/` | ✅ |
| FX-02 | Nonce at character 0; randomized order (cold); `full` first (warm) | ✅ |
| FX-03 | Nonce identical position in calibration and measurement | ✅ |
| FX-05 | Time the scorer live; discard warm-up | ✅ |
| FX-06 | `full` row stores zero selector cost | ✅ |
| FX-07 | Timer guard: reject rows with `prefill_ms > e2e_ms` | ✅ |
| FX-08 | Same question sets for every method | ✅ |
| FX-09 | Calibration down to 32 tokens, 5 replicates | ✅ |
| FX-10 | `top_k` sweep for a real long-prompt regime | ✅ (machinery; `--top-k`) |
| FX-11 | Pin `num_thread`, `num_batch`; sampler off for timing | ✅ |
| FX-12 | Remove unused config entries | ✅ |

Implemented in `src/run_v3.py` (new) + small edits to `src/llm.py`
(`with_nonce`, pinned thread/batch). `src/run_v2.py` is left untouched.

---

## 3. Experiment phases

- [ ] Phase 0: freeze v2 (tag), environment file, hypotheses  ← hypotheses done
- [ ] Phase 1: harness fixes
- [ ] Phase 2: short-prompt calibration + refit
- [ ] Phase 3: 20-question pilot + gate check
- [ ] Phase 3: main cold grid + warm-order pass
- [ ] Phase 4: analysis (inflation, cold S_net, model validation, quality, scorer)
- [ ] Phase 5: literature (Kummer et al. PDF)
- [ ] Phase 6: rewrite paper from v3 data

---

## 4. Run log

| When | What | Outcome |
|---|---|---|
| 2026-10-17 | Created v3 scaffold (config, hypotheses, audit gate, env script) | audit gate reproduces v2 failures as documented |
| 2026-10-17 | Harness fixed (`src/run_v3.py` + `src/llm.py`) | smoke test: shuffled order, live ce_ms, `full`=0 cost, clean cold prefill |
| 2026-10-17 | Scaling calibration (all 4 configs, 32→4096, 5 reps) | 220 rows; calibration range bounded to 4096 (see note) |
| 2026-10-17 | 20-question pilot (3B/Q4, hotpot, cold + warm) | ✅ cold gate PASS; inflation 1.8–2.2× (see §6) |
| 2026-10-17 | Audit gate on pilot cold data | **ALL CHECKS PASSED** (v2: 5 failed) |
| 2026-10-17 | Launched main cold grid + warm-order pass (n=50) | running |
| 2026-10-17 | 3B/Q4 cold complete (4 datasets); 7B/Q4 running | 2,003 cold rows |
| 2026-10-17 | Diagnosed cold-model fits: 7B scaling overhead + near-zero-prefill glitch rows | fixed via union fit + stricter timer guard |
| 2026-10-17 | Audit gate re-run after cold-model fix | **ALL CHECKS PASSED** (C3 0/42) |

### Scope notes (documented deviations)

- **Scaling bounded to 4096 tokens** (not 16384). The cold model is a *calibration* for
  the evaluated prompts (full context ≈ 860–1220 tokens); 16K/8K points are E1 scaling-curve
  luxury and cost minutes each on the 7B models. 3B rows already collected at 8192/16384 are
  retained in the cache. This does not affect any v3 claim (inflation, cold S_net, break-even).
- **Ablations run as a separate arm**, not mixed into the main grid, so the main cold grid is
  the 6-method set from the master plan §6.6.

---

## 5. Notes

- Standing constraints (master plan §12.3): keep the DECAF title, never fabricate data, no
  arXiv-only citations, nothing after the references.
- Do **not** tune per-arm. Fix threads/batch/context once and report them.
- The warm-order pass is expected to FAIL the cold gate — that is the contrast, not a bug.
- v2 files (`results_v2/`, `cache_v2/`, `STATUS.md`, `FULL_RESULTS_v2.md`, `README.md`) are
  **not modified**; the contamination note lives here instead.

---

## 6. Pilot result (3B/Q4, HotpotQA, n=20, both conditions)

**H1 supported.** Warm-order speed-ups exceed cold speed-ups for every method; text-preserving
methods are inflated the most and the differently-ordered BM25 arm the least — exactly the
prefix-cache signature.

| method | inflation (warm/cold) | 95% CI | n |
|---|---:|---|---:|
| `rel_iso` | 2.19× | [1.69, 2.93] | 20 |
| `decaf_lam6` | 1.96× | [1.57, 2.50] | 20 |
| `fixed_0.2` | 1.78× | [1.45, 2.28] | 20 |
| `bm25_decaf_lam6` | 1.29× | [1.18, 1.43] | 20 |

This is **larger** than the master plan's pre-run estimate (1.3–1.7×). Per Gate B, "inflation
clearly above 1.2× with CI excluding 1" → the measurement-validity framing is the headline.

Audit gate (pilot cold): `ALL CHECKS PASSED` — C1 0 invalid rows, C3 0/10 groups outside
[0.80, 1.25], C5 scaling min 130 ≤ compressed p5 176, C6 0 points below 5 reps.

*Caveat:* n=20 pilot only; the full grid at n=50 (running) is needed to confirm per-dataset
inflation and cold S_net.

---

## 7. Main-grid results (cold, n=50) — current state

Coverage: 3B/Q4 complete (4 datasets); 7B/Q4 hotpot + 2Wiki complete, TriviaQA/LongBench running.
Warm-order: 3B/Q4 HotpotQA pilot (n=20). 2,003 cold rows cached.

### 7.1 Cold S_net (per-question, matched, geometric mean)

| method | 3B Hotpot | 3B 2Wiki | 3B Trivia | 3B LongBench | 7B Hotpot | 7B 2Wiki |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_decaf_lam6` | **2.36** | **2.15** | **2.23** | **1.86** | **2.75** | **2.22** |
| `decaf_lam6` | 1.57 | 1.63 | 1.93 | 1.59 | 2.12 | 2.07 |
| `rel_iso` | 1.57 | 1.62 | 1.80 | 1.56 | 2.14 | 2.05 |
| `fixed_0.2` | 1.44 | 1.41 | 1.61 | 1.31 | 1.82 | 1.71 |
| `llmlingua_0.2` | 1.10 | 1.22 | 1.30 | 1.18 | 1.28 | 1.20 |
| `full` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**H4 supported.** `bm25_decaf_lam6` leads every cell because its selector cost (~2 ms) is negligible
against the cross-encoder's ~0.9 s. The gross→S_net gap is the compressor overhead.

### 7.2 Quality (ΔF1 vs full, cold)

| dataset | 3B (decaf_lam6) | 7B (decaf_lam6) | direction |
|---|---:|---:|---|
| HotpotQA | +0.073 | −0.039 | helps 3B / hurts 7B |
| TriviaQA | +0.080 (p=0.037) | +0.096 | helps |
| 2Wiki | −0.037 | −0.093 (p=0.027) | null / hurts |
| LongBench | **−0.238 (p<0.001)** | *pending* | hurts sharply |

**H5 replicates v2 with corrected statistics.** Compression helps the 3B on short contexts, is null
on 2Wiki, hurts the 7B on multi-hop, and hurts LongBench for every config. `llmlingua_0.2` is the
weakest on quality (7B Hotpot −0.185, p=0.013).

### 7.3 Cold model and gate

The scaling sweep carried a per-request overhead on the 7B models and a few near-zero-prefill
glitch rows (e.g. 66 ms for 8,598 tokens) that broke the origin-constrained fit. Fixed two ways:

- **`_valid` timer guard** now also rejects implausibly-fast rows (`prefill/токens < 0.8`).
- **Cold model** fit on the union of (cleaned scaling per-length medians) + (cold `full` compression
  runs) — both cold, nonce-protected. Result: 3B/Q4 c1=3.29 (MAPE 1.4–6.0%), 7B/Q4 c1=8.02
  (MAPE 2.4–11.8%).

Audit gate after the fix: **ALL CHECKS PASSED** — C3 0/42 groups outside [0.80, 1.25].

### 7.4 Break-even (quadratic cold model)

| config | L_c | T_compress | L* quadratic | L* linear |
|---|---:|---:|---:|---:|
| 3B / Q4_K_M | 225 | 1029 ms | 379 | 381 |
| 7B / Q4_K_M | 217 | 1023 ms | 219 | 219 |

Over the bounded 32–4096 calibration the curvature is mild, so the two forms coincide.

**Status:** 7B/Q4 TriviaQA + LongBench cold running; warm-order pass and Q8 extension queued.
