# DECAF — Remaining Experiments

**Date:** 2026-10-15
**Current state:** ✅ v2 core complete — **23,748 runs**, 42 figures, `results_v2/FULL_RESULTS_V2.md`
(4 datasets — HotpotQA, 2Wiki, TriviaQA, LongBench — all 4 configs at n=200; retrieval on all 4 configs).
This file tracks what is **not** yet done, so it is easy to pick up.

## Summary

| | Count |
|---|---:|
| Remaining **experiments** | **5** |
| — feasible now (no new hardware) | 3 |
| — blocked (hardware / model) | 2 |
| Remaining **deliverables** (not experiments) | 0 (drafted) |

Ordered roughly by priority (highest first). "Est. runs" is a data-volume guide, not a time estimate.

---

## Start here (resume tomorrow)

**Next item — external baseline (Perception Compressor).** Needs its research code, then the same
compression pipeline (`--datasets ...`). Cheaper items in the same batch: the **§17 CPU-cost-term
ablation** and **E8 achieved-bandwidth/CPU-utilization telemetry** (both small, no new data source).

---

## A. Feasible now (3)

| # | ID | Experiment | Category | Needs | Est. runs | Priority |
|---:|---|---|---|---|---:|---|
| 1 | base | **Perception Compressor** baseline | external baseline | research code (install/adapt) | 4 cfgs × ds × q × 1–2 rates | medium |
| 2 | §17 | **CPU-cost-term on/off** (explicit) | ablation | new method variant (rel_iso is a proxy) | ~1,600 | medium |
| 3 | E8 | **Achieved bandwidth + CPU-utilization + roofline fit** | telemetry | new sampler (psutil/instruments) | ~60 | medium |

### Notes

1. **CPU-cost-term ablation.** `rel_iso` already approximates "cost term off".
2. **E8 telemetry.** No energy on macOS (see §B); achieved-bandwidth / CPU-utilization can be sampled.

---

## B. Blocked (2)

| # | ID | Experiment | Blocked by | Unblock path |
|---:|---|---|---|---|
| 4 | E1 | **32K context points** (true 16K→32K curve) | Qwen2.5 caps at **32,768** tokens — a ~32K *target* truncates | a longer-context model (YaRN / 1M-context) or a smaller effective target |
| 5 | E8 | **Energy per query** (RAPL / `powerstat`) | **not exposed on macOS** | Linux box with RAPL, or an external power meter |

---

## Already done (for reference)

- **E1** decode scaling to 16K **+ compressed-context curve** (full vs 25%-compressed, 48 runs) ·
  **E2** ratio grid 0.2/0.4/0.5/0.6/0.8 · **E3** overhead vs savings · **E4** break-even `L*` per
  model×quant · **E5/E6** full 2×2 (3B/7B × Q4/Q8) · **E7** grounding · **E9** 2Wiki + TriviaQA +
  LongBench (all 4 configs) · **§17** coverage / reorder / λ / scorer ablation ·
  **§13** BM25 vs dense vs hybrid (4 configs × 3 datasets) · **H5** prefix/KV reuse · LLMLingua-2
  baseline · paired bootstrap + McNemar + Holm.
- **Statistical power** — all 4 configs at n=200 flipped the central claims: 3B/Q4 compression
  significantly **helps** (HotpotQA DECAF λ6 p=0.009), 7B/Q4 significantly **hurts** (p=0.019).
- **Task dependence** — helps on HotpotQA-3B and TriviaQA-3B, neutral on 2Wiki, hurts on HotpotQA-7B,
  and **hurts long-context LongBench for every config** (−0.14…−0.17, p<0.001). Compression's value is
  strongly task/context-dependent.
- **Generation-length robustness:** N_out swept to 512; decode share plateaus (model hits EOS ~40–64 tok).
- **Paper drafts:** `results_v2/PAPER_{ABSTRACT,INTRO_RELATED,METHODS,RESULTS,DISCUSSION}.md` +
  `CITATION_CHECKLIST.md`.
- See `STATUS.md` §1 for the full list and `results_v2/FULL_RESULTS_V2.md` for the results.
