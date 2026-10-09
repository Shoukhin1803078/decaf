# DECAF Pilot — Status & Progress

**Updated:** 2026-10-09
**Scope:** Phase 0 (smoke-test pilot) — ✅ complete · Phase 1 (full experimental study) — 🔨 built, validated, **armed but not yet run**
**Location:** `decaf-pilot/`

---

## 0. Where things stand right now

| Phase | State |
|---|---|
| **Phase 0** — smoke-test pilot, 220 runs | ✅ Complete. Results in `results/PRELIMINARY_RESULTS.md` (+ 4 figures). Clean measurements; four post-hoc caveats recorded in its §6. |
| **Phase 1** — full study, 2,904 runs across E1–E8 + §17 + H5 | 🔨 Code complete, pipeline validated end-to-end, launcher armed. **No Phase-1 measurements exist yet.** |

Phase 1 is gated on a quiet machine. `scripts/launch_when_quiet.sh` polls every 20 s and
starts all four phases automatically once the 1-minute load average drops below 2.5 with
>3 GB free. It has not yet tripped.

**Why the gate exists.** This is a latency study on a 4-core i7-8665U. Measured under a
load average of ~13 (Angular dev server + two JVMs + VS Code + Chrome), prefill rose from
~4 ms/token to ~85 ms/token and TPOT from ~37 ms to ~340 ms. That is not merely noisy — at
that speed Phase B alone would take many hours instead of ~55 min. Contention is therefore a
feasibility blocker, not just a quality one.

---

## 1. Thesis

DECAF's claim is that compression's benefit on CPU must be **measured, not assumed**: the
compressor costs time, decode cost depends on context length, and the question is whether
compression is a *net* win. The empirical backbone is the `TPOT(L)` curve and the break-even
context length `L*`.

**Phase-0 bottom line:** on this machine the proposal's central CPU premise is **reversed** —
prefill, not decode, dominates for any realistic context, because TPOT is nearly flat in `L`.
Compression is still a clear net latency win (~4×), but the win is in **prefill/TTFT, not
decode**, and it costs grounding. Phase 1 tests whether that survives a proper
generation-length sweep, a real cross-encoder scorer, and a second quantization.

---

## 2. Phase 1 — what was built

### New and rewritten modules

| File | Status | Purpose |
|---|---|---|
| `src/telemetry.py` | **new** | per-run contention snapshots; preflight guard that *refuses* to measure on a busy machine; KV-cache geometry and achieved-bandwidth maths; STREAM-like DRAM ceiling for the E8 roofline |
| `src/scorer.py` | **new** | pluggable BM25 vs `BAAI/bge-reranker-base` cross-encoder, each returning its own wall-clock cost so `T_compress` is a measurement, not an estimate |
| `src/figures.py` | **new** | 35 figure generators |
| `src/select.py` | rewritten | method-spec grammar (`decaf_ce_nocost_lam6`) so every §17 ablation is a flag; three cost modes (`relative`, `absolute_ms`, `uniform`); reordering and coverage toggles |
| `src/run_experiments.py` | rewritten | model/quantization dimension in every cache key; generation-length sweep; prompt-truncation guard; shared scorer pass per query; KV-reuse study; pinned cost model |
| `src/analyze.py` | rewritten | Table-2 quantities (CE, `O_c`, `S_net`, QE, `L*`); §21 statistics; auto-written `RESULTS.md` |
| `src/llm.py` | patched | `list_models()`; short-answer prompt rule |
| `src/metrics.py` | patched | `contains_gold`, `answer_recall`, `pred_tokens` — verbosity-robust companions to EM/F1 |
| `config.yaml` | rewritten | full matrix (old pilot config kept as `config.pilot.yaml.bak`) |
| `scripts/run_full.sh` | **new** | phased A→D driver, re-analysing after each phase |
| `scripts/launch_when_quiet.sh` | **new** | load-gated auto-launcher |

### Experiment matrix (2,904 runs)

| Phase | Experiments | Runs | Clean estimate |
|---|---|---|---|
| A | E1/E8 context sweep 128→16K × `N_out` ∈ {16,32,64,128} × 3 reps | 76 | ~18 min |
| B | E2/E3/E4/E7 + §17 — 22 methods × 100 questions | 2,200 | ~55 min |
| C | E5 quantization — Q8_0 scaling + 8 methods × 60 questions | 556 | ~35 min |
| D | H5 — prefix/KV-cache reuse + retrieval-prefetch ceiling | 72 | ~5 min |

The 22 methods cover: full context, closed-book, a 5-point fixed-ratio sweep, the iso-budget
relevance baseline, a 5-point λ sweep, the coverage / CPU-cost / reordering ablations, three
absolute-ms λ points, and three cross-encoder variants.

### Figures — 35 generators, pipeline validated

Validated against the smoke caches: **30 rendered without error**, 5 skipped correctly for
absent inputs (2 need Phase C's second quantization, 2 need fields added after the smoke run,
1 needs ≥2 λ points per family).

Groups: scaling/E1/E8 (8 figures incl. a roofline against a measured STREAM ceiling and an
`(L × N_out)` decode-share heatmap) · compression/E2/E7 (8, incl. a 3-panel Pareto and
win/loss vs full) · DECAF mechanism/§17 (5, incl. the BM25-vs-cross-encoder scorer tradeoff)
· break-even/E4/E5 (3) · statistics/§21 (2) · diagnostics (9, incl. contention telemetry and
a summary dashboard).

---

## 3. Measurement hazards found and guarded

Phase 0 found two; Phase 1 adds four more. All six are now guarded in code.

1. **Ollama context cap** (Phase 0) — prompts silently truncated at `num_ctx`. Now a **hard
   error**: a run that truncates aborts the study rather than flattening the scaling curve.
2. **Cross-run prompt-prefix cache** (Phase 0) — a longer prompt sharing a prefix reused
   prefill for free. Every measured run carries a unique nonce; the only exception is the KV
   study, where reuse *is* the object of measurement.
3. **CPU contention** — see §0. Preflight guard + per-run load/RAM telemetry on every record.
4. **Verbosity confound in EM/F1** — correct answers wrapped in sentences scored ~0, and the
   penalty correlates with context length. Short-answer prompt + `contains_gold` /
   `answer_recall`. (A first fix over-corrected: an explicit yes/no clause made the model
   answer "yes" to open questions; removed.)
5. **Generation-length confound** — a method emitting a shorter answer looks faster.
   `e2e_norm_ms` fixes `N_out` at 32 for a controlled comparison.
6. **Variable KV allocation** — `num_ctx` is now constant *within* each family (20480 scaling,
   4096 compression) so allocation cannot confound the `L` sweep.

---

## 4. Open items

### Blocked on the machine being quiet
- [ ] Run Phases A–D and produce `results/RESULTS.md` + ~35 figures.
- [ ] Confirm the short-answer prompt on real data. **Not yet verified under clean
      conditions** — both validation attempts died (one to contention, one to my own careless
      `pkill`). If full-context F1 comes back near 0.03 rather than Phase 0's 0.22, the run
      should be stopped and the prompt revisited before 2,200 measurements build on it.

### Deferred by scope (agreed: 3B only, ~3 h budget)
- [ ] **E6 model size (3B vs 7B)** — dropped. 7B Q4 needs ~4.7 GB against ~5 GB free RAM and
      4.7 GB of a 6.7 GB disk. This leaves proposal RQ4 only half-answered (quantization yes,
      model size no).
- [ ] **E9 generalization** — 2WikiMultihopQA / TriviaQA / LongBench subset not built.
- [ ] **External baselines** — LLMLingua-2 and Perception Compressor not run; the compute
      barrier should be stated qualitatively per proposal §13.
- [ ] Energy / RAPL telemetry; cache-miss counters.

---

## 5. How to run

```bash
PY=/home/tasnia/decaf/.venv/bin/python     # deps already installed here

# recommended: wait for a quiet machine, then run everything
bash scripts/launch_when_quiet.sh

# or run phases directly (refuses to start if the machine is busy)
bash scripts/run_full.sh A B C D
bash scripts/run_full.sh B                 # single phase; resumable
ALLOW_CONTENTION=1 bash scripts/run_full.sh A   # measure anyway, noise logged

$PY -m src.analyze                         # re-render all figures from cache
```

Everything is resumable: runs are cached per `(model, …)` key in `cache/*.jsonl`, so an
interrupted phase re-starts where it stopped and adding a model never invalidates existing
measurements.

Outputs land in `results/`: `RESULTS.md`, ~35 `fig*.png`, `summary.json`, and
`methods.csv` / `scaling.csv` / `stats.csv`.
