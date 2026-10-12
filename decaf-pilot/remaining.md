# DECAF — Remaining Experiments

**Date:** 2026-10-12
**Current state:** ✅ v2 core complete — **4,476 runs**, 41 figures, `results_v2/FULL_RESULTS_V2.md`.
This file tracks what is **not** yet done, so it is easy to pick up.

## Summary

| | Count |
|---|---:|
| Remaining **experiments** | **10** |
| — feasible now (no new hardware) | 8 |
| — blocked (hardware / model) | 2 |
| Remaining **deliverables** (not experiments) | 1 |

Ordered roughly by priority (highest first). "Est. runs" is a data-volume guide, not a time estimate.

---

## A. Feasible now (8)

| # | ID | Experiment | Category | Needs | Est. runs | Priority |
|---:|---|---|---|---|---:|---|
| 1 | E9 | **LongBench** multi-document QA | generalization | HF dataset + loader | 4 cfgs × ~200 q × ~5 methods ≈ 4,000 | high |
| 2 | E9 | **TriviaQA / NQ** single-hop | generalization | HF dataset + loader | 4 cfgs × ~200 q × ~5 methods ≈ 4,000 | high |
| 3 | E2/§17 | **More queries** (>30 → 100–200 / dataset) | statistics power | re-run main compression | 4 cfgs × 2 ds × 200 q × 15 methods ≈ 24,000 | high |
| 4 | base | **Perception Compressor** baseline | external baseline | research code (install/adapt) | 4 cfgs × 2 ds × 30 q × 1–2 rates ≈ 240 | medium |
| 5 | E1 | **Compressed-context `TPOT(L)` curve** (full vs compressed) | scaling | new Phase-A mode | ~60 | medium |
| 6 | §17 | **CPU-cost-term on/off** (explicit) | ablation | new method variant (rel_iso is a proxy) | 4 cfgs × 2 ds × 30 q ≈ 240 | medium |
| 7 | §13 | **Extend retrieval modes to all 4 configs** | retrieval | config change only | +720 (currently 2 of 4 cfgs) | medium |
| 8 | E8 | **Achieved bandwidth + CPU-utilization + roofline fit** | telemetry | new sampler (psutil/instruments) | ~60 | medium |

### Notes

1. **LongBench / TriviaQA (E9).** 2Wiki is the only generalization set so far. A long-context
   multi-doc set also stresses the `L*`/context-length story directly. Needs a dataset loader
   (same pattern as `src/data_v2.py`).
2. **More queries.** n=30 gives only weak power — most method deltas are non-significant after
   Holm. This is the single biggest lever on the statistics; it is also the most expensive.
3. **Compressed-context curve.** Partly derivable from the existing full-context curve evaluated
   at the retained length `L_c` (TPOT depends on actual tokens), but the proposal asks for it
   *measured*. Cheap to add as a `--mode scaling_compressed`.
4. **CPU-cost-term ablation.** `rel_iso` already approximates "cost term off" (iso-budget relevance
   top-k). An explicit `decaf_nocost_lam6` would make the §17 grid complete.

---

## B. Blocked (2)

| # | ID | Experiment | Blocked by | Unblock path |
|---:|---|---|---|---|
| 9 | E1 | **32K context points** (and a true 16K→32K curve) | Qwen2.5 caps at **32,768** tokens — a ~32K *target* truncates | a longer-context model (YaRN / 1M-context variant) or a smaller effective target |
| 10 | E8 | **Energy per query** (RAPL / `powerstat`) | **not exposed on macOS** | Linux box with RAPL, or an external power meter |

---

## C. Deliverables (1) — not an experiment

| # | Task | Notes |
|---:|---|---|
| 11 | **Paper-ready "Preliminary Results" section** | assemble from `results_v2/FULL_RESULTS_V2.md`, `summary_v2.json`, the 41 figures, `STATUS.md` |

---

## Already done (for reference)

- **E1** decode scaling to 16K · **E2** ratio grid 0.2/0.4/0.5/0.6/0.8 · **E3** overhead vs savings ·
  **E4** break-even `L*` per model×quant · **E5/E6** full 2×2 (3B/7B × Q4/Q8) · **E7** grounding ·
  **§17** coverage / reorder / λ / scorer ablation · **§13** BM25 vs dense vs hybrid (2 cfgs) ·
  **H5** prefix/KV reuse · LLMLingua-2 baseline · paired bootstrap + McNemar + Holm.
- See `STATUS.md` §1 for the full list and `results_v2/FULL_RESULTS_V2.md` for the results.
