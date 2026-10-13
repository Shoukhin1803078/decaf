# DECAF — Remaining Experiments

**Date:** 2026-10-13
**Current state:** ✅ v2 core complete — **15,236 runs**, 41 figures, `results_v2/FULL_RESULTS_V2.md`
(3 datasets: HotpotQA, 2Wiki, TriviaQA; 3B configs and HotpotQA/2Wiki at n=200).
This file tracks what is **not** yet done, so it is easy to pick up.

## Summary

| | Count |
|---|---:|
| Remaining **experiments** | **9** |
| — feasible now (no new hardware) | 7 |
| — blocked (hardware / model) | 2 |
| Remaining **deliverables** (not experiments) | 0 (drafted) |

Ordered roughly by priority (highest first). "Est. runs" is a data-volume guide, not a time estimate.

---

## Start here (resume tomorrow)

**Top item — finish TriviaQA for the 7B configs.** Resumable: only the not-yet-cached runs execute.

```bash
cd decaf-pilot
.venv/bin/python -m src.run_v2 --config config_v2.yaml --mode compression \
  --datasets trivia --methods "full,fixed_0.5,fixed_0.2,rel_iso,decaf_lam6,decaf_nocov_lam6" \
  && .venv/bin/python -m src.analyze_v2 --config config_v2.yaml
```

Tips: run it with `run_in_background` and monitor; it survived one interruption already and picks up
from `cache_v2/compression.jsonl`. To bound runtime, drop the slowest config by adding
`--models qwen2.5:7b-instruct-q4_K_M` (7B/Q4 only).

---

## A. Feasible now (7)

| # | ID | Experiment | Category | Needs | Est. runs | Priority |
|---:|---|---|---|---|---:|---|
| 1 | E9 | **Finish TriviaQA for 7B** (3B done at n=200; 7B/Q4 partial) | generalization | re-run (resumable) | 2 cfgs × ~170 q × 6 methods ≈ 2,000 | high |
| 2 | E9 | **LongBench** multi-document QA | generalization | HF dataset + loader | 4 cfgs × ~200 q × ~5 methods ≈ 4,000 | high |
| 3 | base | **Perception Compressor** baseline | external baseline | research code (install/adapt) | 4 cfgs × ds × q × 1–2 rates | medium |
| 4 | E1 | **Compressed-context `TPOT(L)` curve** (full vs compressed) | scaling | new Phase-A mode | ~60 | medium |
| 5 | §17 | **CPU-cost-term on/off** (explicit) | ablation | new method variant (rel_iso is a proxy) | ~1,600 | medium |
| 6 | §13 | **Extend retrieval modes to all 4 configs** | retrieval | config change only | +720 (currently 2 of 4 cfgs) | medium |
| 7 | E8 | **Achieved bandwidth + CPU-utilization + roofline fit** | telemetry | new sampler (psutil/instruments) | ~60 | medium |

### Notes

1. **TriviaQA 7B.** The 3B configs are complete (n=200); 7B/Q4 is partial (~n=29) and 7B/Q8 not run.
   Re-running `--datasets trivia` resumes from cache.
2. **LongBench (E9).** A long-context multi-doc set also stresses the `L*`/context-length story.
   Needs a dataset loader (same pattern as `src/data_v2.py`).
3. **Compressed-context curve.** Partly derivable from the full curve at retained length `L_c`, but the
   proposal asks for it *measured*; cheap as a `--mode scaling_compressed`.
4. **CPU-cost-term ablation.** `rel_iso` already approximates "cost term off".

---

## B. Blocked (2)

| # | ID | Experiment | Blocked by | Unblock path |
|---:|---|---|---|---|
| 8 | E1 | **32K context points** (true 16K→32K curve) | Qwen2.5 caps at **32,768** tokens — a ~32K *target* truncates | a longer-context model (YaRN / 1M-context) or a smaller effective target |
| 9 | E8 | **Energy per query** (RAPL / `powerstat`) | **not exposed on macOS** | Linux box with RAPL, or an external power meter |

---

## Already done (for reference)

- **E1** decode scaling to 16K · **E2** ratio grid 0.2/0.4/0.5/0.6/0.8 · **E3** overhead vs savings ·
  **E4** break-even `L*` per model×quant · **E5/E6** full 2×2 (3B/7B × Q4/Q8) · **E7** grounding ·
  **E9** 2Wiki **+ TriviaQA (3B)** · **§17** coverage / reorder / λ / scorer ablation ·
  **§13** BM25 vs dense vs hybrid (2 cfgs) · **H5** prefix/KV reuse · LLMLingua-2 baseline ·
  paired bootstrap + McNemar + Holm.
- **Statistical power** — all 4 configs at n=200 on HotpotQA/2Wiki flipped the central claims:
  3B/Q4 compression significantly **helps** (DECAF λ6 p=0.009), 7B/Q4 significantly **hurts** (p=0.019).
- **Dataset dependence** — on single-hop TriviaQA, compression **helps even the 7B** (3B/Q4 +0.081,
  p=0.002), so the size interaction is **dataset-dependent**.
- **Generation-length robustness:** N_out swept to 512; decode share plateaus (model hits EOS ~40–64 tok).
- **Paper drafts:** `results_v2/PAPER_{ABSTRACT,INTRO_RELATED,METHODS,RESULTS,DISCUSSION}.md` +
  `CITATION_CHECKLIST.md`.
- See `STATUS.md` §1 for the full list and `results_v2/FULL_RESULTS_V2.md` for the results.
