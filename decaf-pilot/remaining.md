# DECAF — Remaining Experiments

**Date:** 2026-10-14
**Current state:** ✅ v2 core complete — **21,420 runs**, 41 figures, `results_v2/FULL_RESULTS_V2.md`
(4 datasets — HotpotQA, 2Wiki, TriviaQA, LongBench; retrieval on all 4 configs; 3B at n=200).
This file tracks what is **not** yet done, so it is easy to pick up.

## Summary

| | Count |
|---|---:|
| Remaining **experiments** | **7** |
| — feasible now (no new hardware) | 5 |
| — blocked (hardware / model) | 2 |
| Remaining **deliverables** (not experiments) | 0 (drafted) |

Ordered roughly by priority (highest first). "Est. runs" is a data-volume guide, not a time estimate.

---

## Start here (resume tomorrow)

**Top item — finish LongBench for the 7B configs** (3B done at n=200; 7B/Q4 partial ~20, 7B/Q8 not run).
Resumable: only uncached runs execute.

```bash
cd decaf-pilot
.venv/bin/python -m src.run_v2 --config config_v2.yaml --mode compression \
  --datasets longbench --methods "full,fixed_0.5,fixed_0.2,rel_iso,decaf_lam6,decaf_nocov_lam6" \
  && .venv/bin/python -m src.analyze_v2 --config config_v2.yaml
```

Everything else (§A/§B) is listed below with what it needs.

---

## A. Feasible now (5)

| # | ID | Experiment | Category | Needs | Est. runs | Priority |
|---:|---|---|---|---|---:|---|
| 1 | E9 | **Finish LongBench for 7B** (3B done at n=200; 7B/Q4 partial) | generalization | re-run (resumable) | 2 cfgs × ~180 q × 6 methods ≈ 2,200 | high |
| 2 | base | **Perception Compressor** baseline | external baseline | research code (install/adapt) | 4 cfgs × ds × q × 1–2 rates | medium |
| 3 | E1 | **Compressed-context `TPOT(L)` curve** (full vs compressed) | scaling | new Phase-A mode | ~60 | medium |
| 4 | §17 | **CPU-cost-term on/off** (explicit) | ablation | new method variant (rel_iso is a proxy) | ~1,600 | medium |
| 5 | E8 | **Achieved bandwidth + CPU-utilization + roofline fit** | telemetry | new sampler (psutil/instruments) | ~60 | medium |

### Notes

1. **LongBench 7B.** The 3B configs are complete (n=200); 7B/Q4 is partial and 7B/Q8 not run.
   Re-running `--datasets longbench` resumes from cache.
2. **Compressed-context curve.** Partly derivable from the full curve at retained length `L_c`, but the
   proposal asks for it *measured*; cheap as a `--mode scaling_compressed`.
3. **CPU-cost-term ablation.** `rel_iso` already approximates "cost term off".

---

## B. Blocked (2)

| # | ID | Experiment | Blocked by | Unblock path |
|---:|---|---|---|---|
| 6 | E1 | **32K context points** (true 16K→32K curve) | Qwen2.5 caps at **32,768** tokens — a ~32K *target* truncates | a longer-context model (YaRN / 1M-context) or a smaller effective target |
| 7 | E8 | **Energy per query** (RAPL / `powerstat`) | **not exposed on macOS** | Linux box with RAPL, or an external power meter |

---

## Already done (for reference)

- **E1** decode scaling to 16K · **E2** ratio grid 0.2/0.4/0.5/0.6/0.8 · **E3** overhead vs savings ·
  **E4** break-even `L*` per model×quant · **E5/E6** full 2×2 (3B/7B × Q4/Q8) · **E7** grounding ·
  **E9** 2Wiki **+ TriviaQA (all 4 configs) + LongBench (3B)** · **§17** coverage / reorder / λ / scorer ablation ·
  **§13** BM25 vs dense vs hybrid (**4 configs × 3 datasets**) · **H5** prefix/KV reuse ·
  LLMLingua-2 baseline · paired bootstrap + McNemar + Holm.
- **Statistical power** — all 4 configs at n=200 flipped the central claims: 3B/Q4 compression
  significantly **helps** (HotpotQA DECAF λ6 p=0.009), 7B/Q4 significantly **hurts** (p=0.019).
- **Task dependence** — helps on HotpotQA-3B and TriviaQA-3B, neutral on 2Wiki, hurts on HotpotQA-7B,
  and **hurts long-context LongBench (both 3B configs, p<0.001)**. Compression's value is strongly
  task/context-dependent.
- **Generation-length robustness:** N_out swept to 512; decode share plateaus (model hits EOS ~40–64 tok).
- **Paper drafts:** `results_v2/PAPER_{ABSTRACT,INTRO_RELATED,METHODS,RESULTS,DISCUSSION}.md` +
  `CITATION_CHECKLIST.md`.
- See `STATUS.md` §1 for the full list and `results_v2/FULL_RESULTS_V2.md` for the results.
