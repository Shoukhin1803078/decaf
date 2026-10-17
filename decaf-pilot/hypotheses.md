# DECAF v3 — Pre-registration (hypotheses)

**Date:** 2026-10-17
**Written before the cold re-run.** Committed to the repo before running any v3 experiments.

These hypotheses follow DECAF_Q1_MASTER_PLAN.md §2.3. They are stated as testable predictions
on the **cold** condition (nonce at character 0, randomized method order, same questions across
methods). The warm-order condition exists only to measure the inflation factor.

## H1: Warm-order speed-ups exceed cold speed-ups

**Prediction:** The geometric-mean inflation factor (warm-order S_net / cold S_net) is above 1.2
for methods that keep original paragraph text (`decaf_lam6`, `fixed_0.2`), near 1.0 for
`bm25_decaf_lam6` (different relevance order, less prefix sharing), and close to 1.0 for
`llmlingua_0.2` (rewritten text, no shared prefix).

**Mechanism:** Prefix-cache reuse in the warm-order condition lets a compressed prompt that is a
prefix-fragment of an earlier prompt reuse prefill tokens for free, deflating the measured
preparation time.

## H2: Cold latency model predicts held-out prompts

**Prediction:** The quadratic, origin-constrained, relative-error-weighted prefill model
(`prefill ~ c1*P + c2*P^2`, calibrated on nonce-protected scaling runs) achieves mean relative
error under 10% on cold full-context and compressed prompts under 300 tokens.

## H3: Cold S_net stays above 1.0 above quadratic break-even L*

**Prediction:** On the cold condition, DECAF S_net remains above 1.0 for prompts above the
quadratic break-even length L* (estimated ~274 tokens for 7B/Q4, ~427 for 3B/Q4 at 32 output
tokens), because the per-token prefill saving exceeds the selector cost.

## H4: BM25 scorer gives higher cold S_net than cross-encoder at similar F1

**Prediction:** At matched retention budgets, the BM25-scored variant achieves higher cold S_net
than the bge-reranker-base cross-encoder, with F1 within a pre-registered equivalence margin.

## H5: Compression helps the 3B reader and hurts the 7B on multi-hop

**Prediction:** Replicating the v2 quality direction but on the cold-timing condition: compression
significantly helps the 3B (HotpotQA ΔF1 > 0) and significantly hurts the 7B (HotpotQA ΔF1 < 0),
with the effect null on single-hop TriviaQA for the 7B and negative for all configs on LongBench.

## Decision rule

If a hypothesis fails, report it and let it steer the framing per the gate tables in
`DECAF_Q1_MASTER_PLAN.md` §7.
