# DECAF — Preliminary Results (Phase 0 smoke test)

> **Status: superseded in methodology, retained for its numbers.**
> This is the Phase-0 pilot (30 questions, one model, one quantization, BM25 scorer,
> fixed 32-token generations). Its measurements were taken on a quiet machine and the
> curves in §1–2 remain the best clean `TPOT(L)` / `prefill(L)` data in the project.
> **Four caveats discovered after it was written** are recorded in §6 — two of them
> affect how §3 and §4 should be read.
>
> **This is currently the only valid results file in the project.** The Phase-1 full
> study is built and ready but has not yet produced usable measurements (three runs
> were destroyed by memory/CPU pressure; see `STATUS.md` §0). When it succeeds it will
> write `RESULTS.md` here and supersede this file.

CPU-only, `qwen2.5:3b-instruct` via Ollama (llama.cpp) · 10 scaling runs · 210 compression runs

> CPU-only pilot on one 3B quantized model. Absolute numbers are machine-specific; the **shapes and break-even behaviour** are the point.

**Bottom line.** TPOT grows **linearly** with context length (TPOT ≈ 36.5 + 0.725×L ms per 1k tok, r²=0.68); prefill ≈ 4114.4 ms per 1k tok (r²=1.00).

## 1. Decode scaling (RQ1 / E1, E8)

![](fig_scaling.png)

|    L |   prompt_tokens |   prefill_ms |   decode_ms |   tpot_ms |   ttft_ms |   e2e_ms |   total_ms |   decode_share |   prefill_ms_per_tok |
|-----:|----------------:|-------------:|------------:|----------:|----------:|---------:|-----------:|---------------:|---------------------:|
|  256 |           362   |      1091.4  |     1176.68 |     36.77 |   1103.27 |  2281.61 |    2268.08 |           0.52 |                 3.01 |
|  512 |           645   |      2124.17 |     1211.83 |     37.87 |   2136.04 |  3350.32 |    3336    |           0.36 |                 3.29 |
| 1024 |          1178.5 |      4322.26 |     1184.3  |     37.01 |   4345.56 |  5530.22 |    5506.57 |           0.22 |                 3.67 |
| 2048 |          2240.5 |      8447.47 |     1182.01 |     36.94 |   8490.56 |  9680.19 |    9629.48 |           0.12 |                 3.77 |
| 4096 |          4355.5 |     17506.7  |     1286.15 |     40.19 |  17577.4  | 18874    |   18792.8  |           0.07 |                 4.02 |

- With a median generation of **32 tokens**, decode and prefill are equal at **L ≈ 414 tokens**; below that decode dominates, above it **prefill** dominates.
- **H1 verdict: refuted on this setup.** Decode's share of end-to-end latency *decreases* with context length (0.52 → 0.07), because prefill grows ~4 ms/token while TPOT is nearly flat (+0.7 ms per 1k tokens). Compression therefore helps mostly through prefill/TTFT, not TPOT — the opposite of the proposal's CPU hypothesis, which holds only for very short contexts or much longer generations.

## 2. Break-even context length L* (RQ2 / E3, E4)

![](fig_break_even.png)

- compressor overhead T_compress = **1.36 ms**; retained context L_c ≈ **205 tokens**.
- solving the fitted curves gives **L\* ≈ 205 tokens**: below it compression is a net latency loss, above it a net win.
- **H2 verdict: supported, but the threshold is trivial on CPU.** Compressor overhead (~1.4 ms) is negligible next to the per-token prefill cost (~4 ms/token), so L* collapses to ≈ the retained length: compression is a **net latency win for any context longer than the retained budget**. The binding constraint is quality/grounding, not latency.

## 3. Compression: quality, grounding, latency (RQ3 / E2, E7)

![](fig_pareto_grounding.png)

| method                |    F1 |    EM |   ev.recall |   ev.prec |   gold_in_ev |   retained |   compress_ms |   e2e_ms |   net_S |
|:----------------------|------:|------:|------------:|----------:|-------------:|-----------:|--------------:|---------:|--------:|
| DECAF λ=15            | 0.122 | 0.1   |       0.279 |     0.578 |        0.2   |      0.059 |          1.38 |   1146.7 |    3.24 |
| DECAF λ=6             | 0.268 | 0.2   |       0.532 |     0.403 |        0.6   |      0.19  |          1.36 |    987.5 |    3.76 |
| DECAF no-cov (λ=6)    | 0.302 | 0.2   |       0.671 |     0.201 |        0.733 |      0.47  |          1    |   1503.7 |    2.47 |
| Fixed 20%             | 0.265 | 0.233 |       0.562 |     0.315 |        0.6   |      0.249 |          1.03 |    911.9 |    4.07 |
| Fixed 50%             | 0.285 | 0.2   |       0.668 |     0.162 |        0.767 |      0.572 |          1.09 |    949.1 |    3.91 |
| Full context          | 0.221 | 0.1   |       0.751 |     0.097 |        0.833 |      1     |          0.84 |   3712.7 |    1    |
| Relevance top-m (iso) | 0.24  | 0.167 |       0.573 |     0.423 |        0.567 |      0.261 |          1.12 |   1172.3 |    3.17 |

## 4. Component analysis (RQ3 / §17)

![](fig_ablation.png)

- Iso-budget relevance baseline vs DECAF(decaf_lam6): F1 0.240→0.268, evidence recall 0.573→0.532, gold-in-evidence 0.567→0.600 at the same retained count.
- **H3 verdict: not supported.** DECAF does not beat fixed-ratio selection on the quality–latency frontier (fixed-50% F1 0.285 and fixed-20% 0.265 vs DECAF-λ6 0.268) and only edges the iso-budget relevance baseline. On this pilot the CPU-cost and coverage terms add little over a plain relevance ranking.

- **Note (reassuring for the proposal's motivation):** moderate compression *improves* F1 (full 0.221 → fixed-50% 0.285) while cutting latency ~4×, because distractor context hurts the small model; but grounding degrades monotonically with retained evidence (gold-in-evidence 0.833 → 0.60 → 0.20), so accuracy alone would hide the cost.

## 5. Reading for the proposal

- §1 gives the **measured TPOT(L)/prefill(L)** curves that underlie the break-even analysis.
- §2 gives a concrete **L\*** from the fitted curves plus measured compressor overhead.
- §3–4 give the **quality–latency–grounding Pareto** and the CPU-awareness ablation.
- Next: Q4/Q8 and 3B/7B pairs, long-context sets (LongBench), a cross-encoder scorer, and more queries.

---

## 6. Caveats discovered after this pilot (added 2026-10-09)

Four issues were found while building the Phase-1 study. They do not invalidate the
latency curves in §1–2, but two of them change how §3–4 should be read.

**(a) The F1 numbers are depressed by verbosity, and the bias correlates with context length.**
The reader was answering correctly but in full sentences: for gold `flotilla` it produced
*"Iqbal F. Qadir was on a flotilla when he participated in the attack…"*, scoring EM 0 and
F1 0.10 despite containing the right answer. Because a shorter context elicits a more direct
answer, this length penalty is **correlated with the compression ratio**. The headline
"moderate compression *improves* F1 (0.221 → 0.285)" in §3 may therefore be partly a prompt
artifact rather than distractor harm. Phase 1 adds a short-answer instruction plus
`contains_gold` and `answer_recall`, which are invariant to verbosity, and will re-test it.

**(b) The CPU-cost ablation could not have shown an effect.**
`MarginalCPUCost` was `tokens_i / tokens_total`. Since the measured cost curve is *linear* in
tokens, that normalised cost is proportional to token count up to a constant — and the constant
is absorbed by `λ`. "CPU-aware cost" was therefore mathematically indistinguishable from
"token-proportional cost", so the §4 finding that the CPU-cost term adds nothing is a
**tautology, not evidence**. Phase 1 adds an `absolute_ms` cost mode (cost in milliseconds from
the fitted curve for that specific model and quantization, `λ` in quality-per-millisecond),
which is the only form in which the CPU-awareness claim is testable — and which predicts that
the same `λ` selects differently under Q4 vs Q8.

**(c) `T_compress = 1.36 ms` is BM25, not the proposal's specified scorer.**
Proposal §12.1 specifies a `bge-reranker-base` cross-encoder. A first measurement of that
scorer on this CPU gave `T_compress` three orders of magnitude larger than BM25. If that
survives clean measurement, the conclusion in §2 that "compressor overhead is negligible next
to prefill cost" **holds only for BM25** and is reversed for the proposal's own scorer — which
would make the compressor violate the design principle of proposal §5 ("the compressor must be
cheaper than the decoding it saves"). Phase 1 measures both scorers.

**(d) The H1 refutation is conditional on a 32-token generation.**
Decode cost scales with `N_out` while prefill does not, so the crossover `L ≈ 414` is a
property of *this* generation length, not of CPU RAG in general. The fitted curves imply the
crossover moves right as `N_out` grows. Phase 1 sweeps `N_out ∈ {16, 32, 64, 128}` so H1 is
tested as a surface over `(L, N_out)` rather than a single line.

### What stands unchanged

- The `prefill(L)` curve (≈4.1 ms/token, r²=1.00) and the near-flat `TPOT(L)`.
- The qualitative direction of H1 on this hardware: **prefill dominates, not decode.**
- The monotone degradation of grounding with retained evidence (0.833 → 0.60 → 0.20),
  which is measured on retained gold sentences and is unaffected by (a).
