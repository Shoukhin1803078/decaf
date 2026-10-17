# DECAF — Master Plan for a Q1 Submission

Written 2026-10-15 for Mahin. One file, everything in it. Read sections 0 to 2 first, then work through section 4 in order.

## Contents

0. Honest note (read first)
1. Where you stand in 30 seconds
2. What the paper is (questions, contributions, novelty, allowed claims)
3. What is broken (with evidence)
4. What you must do now (step by step, with "done when" tests)
5. Techniques that can give better results
6. Results you must bring back (tables, figures, files, what each outcome means)
7. Q1 gates and decision rules
8. Reviewer risks and your answers
9. Fallback plan and journal shortlist
10. Timeline
11. Submission checklist
12. Appendix (audit numbers, code snippets, constraints)

---

## 0. Honest note (read first)

- **Nobody can ensure Q1.** Acceptance is decided by an editor and reviewers. This plan makes the paper credible and removes the easy reasons to reject it. It does not buy the outcome.
- **Do not tune anything until a number looks good.** Run the experiments as designed and report whatever comes out. If a result is negative, it goes in the paper. That is what makes the paper believable, and a Q1 journal will check.
- **Quartiles change every year.** I have not verified any journal's current JCR or SJR quartile or scope. Check before you choose (section 9).
- **Time and run-count figures are my estimates**, not measurements. Time the 20-question pilot (step 3.1) and rescale.
- Numbers in this file come from my analysis of your raw `cache_v2/` files on 2026-10-15. Items marked **[estimate]** are model-based, not measured.

---

## 1. Where you stand in 30 seconds

**The good**
- You have a large, well-organised experiment: 23,748 logged rows, 2 model sizes x 2 quantizations, 4 datasets, 15 methods, scaling and crossover runs, a cross-encoder scorer, an LLMLingua arm.
- Your answer-quality results are sound and are not affected by the timing problem.
- Your cold latency model (fitted on nonce-protected scaling runs) predicts real full-context prompts within about 3% mean error. That is a strong result.

**The bad**
- The compressed-prompt timings were measured with prefix-cache reuse (no nonce, `full` run first, fixed method order). The speed-ups are inflated by about 1.3x to 1.7x **[estimate]**.
- Baselines are unequal (DECAF n=200, BM25 and LLMLingua n=30 per dataset).
- 18 rows have a corrupted `prefill_ms` (about 1,000,000 ms).
- The latency model was calibrated from 361 tokens up, but selected prompts average about 175 tokens.
- Your ablations show DECAF is not better than a plain top-m selection at the same budget, so "DECAF improves RAG" cannot be the claim.

**The plan**
- Re-run the timing arms cold, fix the harness, widen calibration, then write the paper around a measurement-and-modelling claim.

**The paper in one sentence**
> On CPU-only llama.cpp/Ollama serving, the reported latency gain of context selection is inflated by prefix-cache reuse; we show how to measure it cold, predict it with a validated prefill/decode model, and give an overhead-aware break-even rule.

---

## 2. What the paper is

### 2.1 Problem and gap

Compressing or selecting RAG context is reported to cut latency. On CPU-only local serving (llama.cpp or Ollama), prefix-cache reuse can make a compressed prompt look faster than it would be from a cold start, whenever it shares a leading prefix with a prompt that ran just before.

The closest published study is Kummer et al., "Prompt Compression in the Wild: Measuring Latency, Rate Adherence, and Quality for Faster LLM Inference", ECIR 2026 (LNCS 16483, pp. 257-271). From what I could read (Springer page and arXiv HTML, not the PDF figures): GPUs plus an Apple M1 Pro, LLMLingua variants, time to first token, an empirical break-even profiler, no CPU-only setting, no closed-form latency model, and no discussion of caching. Verify all of this against the PDF.

### 2.2 Research questions

| ID | Question | Type |
|---|---|---|
| RQ1 | How much does prefix-cache reuse inflate the measured speed-up of context selection on CPU-only serving, and can it be detected? | Measurement validity |
| RQ2 | Can a cold-calibrated prefill/decode model predict end-to-end RAG latency on CPU for prompts it was not fitted on? | Prediction |
| RQ3 | Once selector cost is counted, at what prompt length does selection pay off, and how does it depend on model size, quantization and scorer cost? | Break-even |
| RQ4 | What accuracy is paid for the saved time, and does it depend on reader size and task? | Quality cost |

### 2.3 Hypotheses (write them in the repo with a date BEFORE the re-run)

- H1: warm-order speed-ups exceed cold speed-ups by a factor above 1.2 for methods that keep original paragraph text; near 1.0 for LLMLingua.
- H2: held-out mean error of the cold model on short compressed prompts is under 10%.
- H3: cold S_net for DECAF stays above 1.0 for prompts above the quadratic break-even L*.
- H4: a cheap lexical scorer (BM25) gives higher cold S_net than the cross-encoder at similar F1.
- H5: compression helps the weak 3B reader on some tasks and hurts the 7B reader and long-context tasks.

If a hypothesis fails, report it. A failed H1 changes the framing (section 7).

### 2.4 Contributions (claim only what the re-run confirms)

1. **C1: cache-contamination protocol.** A detection test (logged prefill divided by cold-model prefill, plus a no-shared-prefix control) and a cold measurement design (nonce, randomized order).
2. **C2: validated cold latency model** for CPU RAG, with closed-form decode/prefill crossover and break-even, validated out of sample.
3. **C3: overhead-aware break-even.** How much of the gross speed-up the selector consumes, and when a cheap scorer wins.
4. **C4: empirical map** across model sizes, quantizations and four datasets, including negative results.

### 2.5 Core novelty

CPU-only context-compression speed-ups are overstated by prefix-cache reuse, and a cold-calibrated model plus an overhead-aware break-even gives the honest number.

| | Kummer et al. (ECIR 2026) | This work |
|---|---|---|
| Hardware | A100, GTX 1080 Ti, M1 Pro | CPU-only |
| What is compressed | LLMLingua family (token pruning) | Evidence selection plus LLMLingua arm |
| Latency metric | Time to first token | End-to-end including decode |
| Model | Empirical profiler | Closed-form, validated |
| Caching | Not discussed (as far as I could read) | Central finding |
| Quality statistics | None visible | Paired bootstrap, Holm |

### 2.6 What you may claim and what you may not

**May claim:** measured cold timings with confidence intervals; the inflation factor per config; out-of-sample error of the latency model; that selection pays above a stated break-even for the tested setups; quality changes with statistics, including negative ones.

**May not claim:** that DECAF is a better selector (unless the new data supports it at equal budget); that compression improves RAG quality in general; that results hold on GPUs, batched serving or other model families; that cache reuse is the cause if no cache counter was logged (say "consistent with"); guaranteed Q1.

### 2.7 The role of DECAF

DECAF stays as the title and framework name. It is the worked example of a cost-aware selector (admit a paragraph if its gain minus lambda times its cost is non-negative; cost modes relative, ms, uniform). The paper must say that the contribution is the measurement and model, not the selector.

---

## 3. What is broken (evidence)

### 3.1 Cache contamination

Method: fit the cold prefill model on `scaling.jsonl` (nonce-protected), then compare each compression run's logged prefill with the model's prediction at the same prompt length.

| Arm | Median logged / cold prefill |
|---|---|
| `full` prompts | 0.98 to 1.07 (model fits real prompts) |
| DECAF, fixed 20% (keep original paragraph text) | about 0.07 to 0.20; 88% to 100% of runs below 0.5 |
| BM25-scored DECAF | about 0.78 to 0.83 |
| LLMLingua (text rewritten, no shared prefix) | about 0.87 to 0.99 |

Other evidence: in `h5_prefix.jsonl` a warm identical prompt logs 2,241 prompt tokens but only about 37 ms of prefill, so `prompt_tokens` includes cached tokens and cannot detect reuse. This is consistent with prefix-cache reuse, but it is inference because no cache counter was logged.

Root cause in the code: `src/run_v2.py` adds a nonce in scaling/crossover runs only. Compression runs use `build_prompt` with no nonce and run `full` first, then the other methods in a fixed cycle.

### 3.2 Effect on net speed-up [estimate]

S_net = full-context time / (selector time + compressed end-to-end time). "Corrected" swaps the logged prefill for the cold-model prefill at the same prompt length. DECAF lambda=6, n=200 per cell, format logged → corrected:

| Dataset | 3B Q4 | 3B Q8 | 7B Q4 | 7B Q8 |
|---|---|---|---|---|
| HotpotQA | 2.28 → 1.79 | 1.86 → 1.56 | 3.89 → 2.49 | 2.90 → 2.14 |
| 2WikiMultiHopQA | 2.17 → 1.73 | 1.70 → 1.46 | 3.35 → 2.33 | 2.59 → 2.03 |
| TriviaQA | 3.48 → 2.57 | 2.89 → 2.26 | 6.15 → 3.56 | 4.96 → 3.28 |
| LongBench | 2.07 → 1.70 | 1.72 → 1.49 | 4.00 → 2.72 | 3.05 → 2.31 |

Caveat: selected prompts average about 175 tokens, below the 361-token start of the calibration data, so the correction extrapolates.

### 3.3 Unequal baselines

30-question subsets (HotpotQA + 2Wiki, n=60), cold-corrected S_net and F1:

| Config | Method | S_net | F1 | Full-context F1 |
|---|---|---|---|---|
| 3B Q4 | BM25-scored DECAF | 2.55 | 0.275 | 0.206 |
| 3B Q4 | DECAF (cross-encoder) | 1.71 | 0.286 | 0.206 |
| 3B Q4 | Fixed 20% | 1.48 | 0.325 | 0.206 |
| 3B Q4 | LLMLingua 20% | 1.34 | 0.208 | 0.206 |
| 7B Q4 | BM25-scored DECAF | 2.59 | 0.375 | 0.459 |
| 7B Q4 | DECAF (cross-encoder) | 2.44 | 0.398 | 0.459 |
| 7B Q4 | Fixed 20% | 1.97 | 0.379 | 0.459 |
| 7B Q4 | LLMLingua 20% | 1.57 | 0.264 | 0.459 |

The old repo claim "DECAF S_net 2.94 vs LLMLingua-2 1.13" compared a contaminated arm with an uncontaminated one. n=60 only; treat as a lead.

### 3.4 Corrupted timer rows

18 of 21,360 compression rows have a server-side `prefill_ms` near 1,000,000 ms while client times look normal: 15 are `full` rows for 7B Q8 LongBench, plus one each in 3B Q4 2Wiki LLMLingua 50%, 7B Q8 LongBench DECAF, and 7B Q8 LongBench no-coverage DECAF. Any mean of `prefill_ms` including them is wrong.

### 3.5 Calibration range

Scaling runs span 361 to 17,096 prompt tokens. Selected prompts average about 175 tokens (DECAF) to 400 (fixed 20% on LongBench).

### 3.6 Quality (not affected by timing)

Paired bootstrap of DECAF lambda=6 minus full context, F1, n=200 (plain bootstrap, no multiple-comparison correction; your repo uses Holm so p-values will differ):

| Config | Dataset | Delta F1 | 95% CI |
|---|---|---|---|
| 3B Q4 | HotpotQA | +0.077 | +0.036 to +0.119 |
| 3B Q4 | TriviaQA | +0.081 | +0.039 to +0.126 |
| 3B Q4 | 2Wiki | 0.000 | -0.040 to +0.038 |
| 3B Q8 | HotpotQA | +0.027 | -0.007 to +0.061 |
| 3B Q8 | 2Wiki | -0.032 | -0.061 to -0.001 |
| 7B Q4 | HotpotQA | -0.092 | -0.152 to -0.032 |
| 7B Q8 | HotpotQA | -0.064 | -0.116 to -0.014 |
| 7B (both) | TriviaQA | about 0.00 | CIs include 0 |
| 7B (both) | 2Wiki | about -0.02 | CIs include 0 |
| all four configs | LongBench | -0.136 to -0.174 | all exclude 0 |

Full-context F1 on HotpotQA is 0.185 (3B Q4) vs 0.535 (7B Q4): the 3B "gain" mostly means a weak reader is distracted by long context.

DECAF vs plain top-m at the same budget: no difference in the repo table (for example 7B Q4 HotpotQA 0.443 vs 0.443). I read this from `FULL_RESULTS_v2.md` and did not run a formal test.

### 3.7 Break-even

| Config | Linear L* (repo) | Quadratic L* (cold model) |
|---|---|---|
| 3B Q4 | 342 | 427 |
| 7B Q4 | 254 | 274 |

L* uses nonce-protected scaling data and measured selector time, so it is not directly hit by the contamination. Use the quadratic form. Selector (bge-reranker-base on CPU) costs about 0.9 to 1.0 s on short prompts and about 1.34 s on LongBench; the decode/prefill crossover at 32 output tokens is about 584 to 916 prompt tokens depending on config.

---

## 4. What you must do now

Legend: **[MUST]** paper not credible without it. **[SHOULD]** reviewers will expect it. **[NICE]** only if time allows.

### Phase 0 — Prepare

- [ ] **0.1 [MUST] Freeze the current v2 repo state.** Tag the commit (for example `v2-contaminated`) and leave `cache_v2/` untouched. You will report the contamination, so the old data is evidence.
  - Done when: the tag exists on GitHub.
- [ ] **0.2 [MUST] Record the environment** by script: CPU model, physical cores, threads used, RAM, OS, Ollama / llama.cpp version, model files and quantizations, power plan, background load.
  - Done when: `ENVIRONMENT.md` is generated automatically into the repo.
- [ ] **0.3 [SHOULD] Make the machine quiet.** Close other apps, fix thread count, disable sleep, keep power settings constant.
  - Done when: a 10-run idle benchmark has a coefficient of variation under 5%.
- [ ] **0.4 [MUST] Write the hypotheses file** (section 2.3) with a date, commit it before running anything.

### Phase 1 — Fix the harness (`src/run_v2.py`, `src/llm.py`)

- [ ] **1.1 [MUST] Nonce at the very start of every prompt** in the compression runner (your scaling runner already does this). See snippet A.
  - Done when: two prompts with identical paragraphs differ in their first tokens.
- [ ] **1.2 [MUST] Randomize method order per question** with a logged seed (snippet B). Today every question runs `full` first, then a fixed cycle.
- [ ] **1.3 [MUST] Add a warm-order condition** (no nonce, `full` first) so inflation is measured per question rather than estimated.
- [ ] **1.4 [SHOULD] Log cache evidence.** `prompt_tokens` stays at the full count even when the prefix was cached, so it cannot detect reuse. If you run `llama-server`, log its cached-token count (check the exact field name in your version). In Ollama, check whether your version exposes any cache field; if not, say the evidence is timing-based.
- [ ] **1.5 [MUST] Timer guard.** Reject and re-run any row where `prefill_ms > e2e_ms`. Log the count.
- [ ] **1.6 [MUST] Detection gate** after every batch (snippet C).
- [ ] **1.7 [MUST] Same question IDs for every method**, asserted by script.
- [ ] **1.8 [SHOULD] Verify what the "LLMLingua" arm really is** in `src/llmlingua_v2.py` (the repo calls it LLMLingua-2) and name it correctly in the paper.

### Phase 2 — Extend the calibration

- [ ] **2.1 [MUST] Add scaling points at about 50, 100, 150, 200 and 300 prompt tokens**, with nonce, at least 5 repetitions each, for every model config you will use.
- [ ] **2.2 [MUST] Refit** the cold model (snippet D) and the full latency model `T(P,N) = w + c1*P + c2*P^2 + N*(t0 + t1*P)`.
  - Done when: held-out error on real short prompts (under 300 tokens) is reported; target under 10% mean error.
- [ ] **2.3 [SHOULD] Report linear and quadratic L* side by side.**

### Phase 3 — Main cold re-run (scope and details in section 6.6)

- [ ] **3.1 [MUST] Pilot: 20 questions, one config, one dataset, all methods, both conditions.** Check the gate, time it, rescale the plan.
- [ ] **3.2 [MUST] Main grid:** 2 configs (3B Q4, 7B Q4) x 4 datasets x 100 questions x 6 methods cold, then the warm-order pass for 4 compressed methods.
- [ ] **3.3 [SHOULD] Extension:** 3B Q8 and 7B Q8 at n=50 to keep the quantization claim.
- [ ] **3.4 [SHOULD] Deliberate extras from section 5** (cheaper selector, two-stage scoring) as separate, clearly labelled arms.

### Phase 4 — Analysis

- [ ] **4.1 [MUST] Inflation:** per question, warm-order speed-up divided by cold speed-up; geometric mean with bootstrap 95% CI per config and dataset.
- [ ] **4.2 [MUST] Cold S_net** with paired bootstrap CI.
- [ ] **4.3 [MUST] Out-of-sample validation** of the latency model on the new cold timings.
- [ ] **4.4 [MUST] Quality at equal budget:** DECAF vs top-m vs fixed vs BM25 vs LLMLingua; paired bootstrap, Holm correction, effect sizes, equivalence margins for "no difference".
- [ ] **4.5 [SHOULD] Scorer question:** BM25 vs cross-encoder on S_net and F1 at full n.
- [ ] **4.6 [MUST] Run the decision gate (section 7) and choose the framing from the data.**

### Phase 5 — Literature

- [ ] **5.1 [MUST] Read the Kummer et al. PDF in full.** Check what they say about caching, what the profiler does, and whether the abstract and body agree (my extraction hinted at mismatches I could not verify).
- [ ] **5.2 [MUST] Verify every citation has a published venue** (no arXiv-only entries) and correct authors, title, year.
- [ ] **5.3 [SHOULD] Search again for new work** on CPU-only RAG latency, prefix-cache benchmarking pitfalls and compression break-even in the week you submit.

### Phase 6 — Rewrite the paper (`/home/claude/decaf_paper/`)

- [ ] **6.1 [MUST] Abstract and introduction** built on the one-sentence claim. No "DECAF improves RAG".
- [ ] **6.2 [MUST] Methods:** cold measurement protocol (nonce, random order, gate), latency model, break-even, selector as an instance.
- [ ] **6.3 [MUST] Results order:** (1) contamination evidence and inflation; (2) validated model; (3) break-even with selector cost; (4) quality cost and the model-size interaction; (5) scorer comparison.
- [ ] **6.4 [MUST] Limitations:** single CPU, Ollama/llama.cpp only, batch size 1, Qwen2.5 only, 32 output tokens, timing-based cache evidence if no counter.
- [ ] **6.5 [MUST] Rebuild all tables and figures from the new data by script.**
- [ ] **6.6 [MUST] Compare with Kummer et al.** in related work and in a short table.
- [ ] **6.7 [SHOULD] Scripted check** that every number in the text matches a generated table.
- [ ] **6.8 [MUST] Keep the standing constraints** (section 12.3).

### Phase 7 — Release and submit

- [ ] **7.1 [MUST] Reproducibility package** with a one-command rebuild of every table and figure; archive with a DOI (for example Zenodo).
- [ ] **7.2 [MUST] Choose the journal** after checking scope and current quartile (section 9).
- [ ] **7.3 [SHOULD] Supervisor or peer review** of the contribution statement and limitations.
- [ ] **7.4 [MUST] Cover letter** stating the contribution in two sentences and the difference from Kummer et al.
- [ ] **7.5 [MUST] Final checks** (section 11).

### If you only have one day

Do 0.1, 1.1, 1.2, 1.5, then the 20-question pilot (3.1). It tells you whether the cache story holds, and every later decision depends on it.

---

## 5. Techniques that can give better results

None of these is guaranteed to help. "Effect" is what I expect, not what I measured. Try them as labelled extra arms or experiments, never as silent changes to the main grid, and keep the main results free of them.

### 5.1 Measurement techniques (raise credibility; do these first)

| Technique | Why it helps | Notes |
|---|---|---|
| Nonce at the very start of the prompt | Prevents prefix reuse | Already used in your scaling runs |
| Randomized method order with logged seed | Removes order effects | Needed for any paired comparison |
| Cold vs warm-order paired design | Measures inflation directly instead of estimating it | The core experiment |
| Detection gate (logged / cold-model prefill ratio) | Catches leaks automatically | Stop the batch if outside 0.8 to 1.25 |
| Second runtime check (for example `llama-server` with prompt caching switched off per request) | Shows the effect is not a quirk of one tool | Verify the option name in your version |
| Repeated measurements and medians | Reduces noise | Discard a few warm-up runs and say so |
| Fixed threads, fixed batch size, fixed context length, quiet machine | Makes numbers comparable | Record every setting |
| Raw server counters where available | Replaces inference with evidence | Strongest answer to "is this just a bug?" |

### 5.2 Modelling techniques

| Technique | Why it helps | Notes |
|---|---|---|
| Calibrate down to about 50 tokens | Removes the extrapolation caveat | Phase 2 |
| Quadratic prefill term | Fits attention cost growth; gives larger, more honest L* | Compare with linear |
| Relative-error weighting (already used) | Stops long prompts dominating the fit | Keep |
| Leave-one-length-out cross-validation | Honest error estimate across the range | Report alongside held-out real prompts |
| Joint bootstrap over fitted parameters | Intervals that include model uncertainty | Already used in the earlier paper |
| Separate fit per model config | Quantization and size change the constants | Report constants in a table |
| Model selector time as a function of candidate count | Lets you predict the cost of cheaper selectors | Linear in number of pairs is the first guess; check it |

### 5.3 Selector and system techniques (possible improvements; all untested here)

| Technique | Expected effect | Risk |
|---|---|---|
| BM25 as the scorer | Selector time of about 2 ms vs about 0.9 to 1.3 s; early hint of higher S_net at similar F1 | Quality may drop on hard multi-hop questions; confirm at full n |
| Two-stage scoring: BM25 prefilter to top-k, then cross-encoder on the survivors | Cross-encoder cost should shrink roughly in proportion to pairs scored (verify) | Prefilter may drop gold paragraphs |
| Smaller or int8-quantized cross-encoder (for example via an ONNX runtime) | Often reduces CPU latency | Measure both speed and rank quality |
| Score caching for repeated questions or documents | Removes selector cost on repeats | Only useful in repeated-query workloads |
| Tune the budget (lambda or fraction) on a dev split, then evaluate on a separate test split | Gives a fair quality-vs-latency frontier | Do not tune on the test split |
| Report an iso-budget Pareto frontier instead of single points | Fair comparison across selectors | Needs several budgets per method |
| Shorter system/template text | Cuts fixed tokens for every request | Small gain; keep wording constant across arms |
| Cache-aware prompt layout (stable shared prefix first) as a deliberate feature in repeated-document workloads | Turns the effect you found into a real speed-up when documents repeat | Different scenario from your main claim; present separately if at all |
| llama.cpp thread and batch-size tuning | Changes absolute CPU prefill speed | Must be fixed and reported; never tuned per arm |
| Speculative decoding or a smaller draft model | Could cut decode time | Untested here; out of scope for the first paper |

### 5.4 Statistical techniques

- Paired bootstrap over questions (resample questions, not runs).
- Holm correction across the whole family of comparisons.
- Equivalence margins (TOST-style) for any "no difference" statement.
- Power or minimum-detectable-effect statement for each dataset size.
- Report effect sizes with intervals, never only p-values.
- Optional: a mixed-effects model with question as a random effect as a robustness check.
- Report exclusions (corrupted rows, failed runs) with counts.

### 5.5 Writing techniques

- Short sentences; one claim per sentence; numbers in tables, not buried in prose.
- State the contribution in the abstract's first two sentences.
- Put the negative results (LongBench, 7B) in the main text.
- Use hedged wording where evidence is inferential ("consistent with").
- Keep the figures simple: one idea per figure.

### 5.6 Priority order if time is short

1. Cold re-run and inflation measurement (5.1).
2. Short-prompt calibration (5.2).
3. BM25 and two-stage scorer arms (5.3).
4. Equal sample sizes and Holm-corrected statistics (5.4).
5. A second hardware platform or runtime, if you can reach one.

---

## 6. Results you must bring back

This section lists exactly what to produce, what each outcome means and what to send back to me. These are results to **measure and report**, not targets to hit.

### 6.1 Core result tables

| ID | Table | Rows x columns | Why it matters |
|---|---|---|---|
| R1 | **Cache inflation** | config x dataset x method: warm-order S_net, cold S_net, ratio, 95% CI, n | Answers RQ1; the headline finding |
| R2 | **Gate table** | config x method: median logged/cold prefill ratio (cold condition and warm-order condition) | Proves the cold runs are clean and the warm-order runs are contaminated |
| R3 | **Latency model constants** | config: w, c1, c2, t0, t1, fit range, n points | Reproducibility of the model |
| R4 | **Model validation** | config x prompt-length bin (under 300 / 300-1,000 / over 1,000): MAPE, bias, n | Answers RQ2; shows the short-prompt range |
| R5 | **Break-even** | config x scorer (BM25, cross-encoder, LLMLingua): selector time, L_c, linear L*, quadratic L* | Answers RQ3 |
| R6 | **Cold S_net** | config x dataset x method: S_net, gross speed-up, selector share of time, 95% CI | The corrected speed-up numbers |
| R7 | **Quality at equal budget** | config x dataset x method: F1, EM, delta vs full, CI, Holm-adjusted p, equivalence verdict | Answers RQ4 |
| R8 | **Scorer comparison** | BM25 vs cross-encoder vs two-stage: S_net, F1, evidence recall | Tests H4 |
| R9 | **Exclusions** | Count of rejected / re-run rows by reason | Honesty and reproducibility |

### 6.2 Core figures

| ID | Figure | Message |
|---|---|---|
| F1 | Prefill time vs prompt length, cold model curve with cold and warm-order points overlaid | Shows contamination visually |
| F2 | Distribution of logged/cold prefill ratios by method (cold vs warm-order) | The detection test |
| F3 | Inflation factor per config and dataset with CIs | Headline result |
| F4 | Predicted vs measured end-to-end latency (held-out) | Model validity |
| F5 | Break-even curves (net speed-up vs prompt length) per config and scorer | When selection pays off |
| F6 | Latency-vs-F1 Pareto frontier by method at several budgets | The honest trade-off |
| F7 | Decode share vs output length (crossover) | Why prefill dominates on CPU |
| F8 | Framework diagram of the measurement pipeline | Only if it shows the real method |

### 6.3 Headline numbers to bring back (single values, with CIs)

1. Geometric-mean inflation factor, per config, with 95% CI. (My estimate before the re-run: about 1.3 to 1.7.)
2. Cold S_net of DECAF (and of the other methods) per config and dataset.
3. Out-of-sample MAPE of the latency model, per config, short prompts and all prompts.
4. Quadratic break-even L*, per config and scorer.
5. Delta F1 vs full context at equal budget, per config and dataset, with CI.
6. Selector share of total time (selector time / compressed total).
7. BM25 vs cross-encoder: delta S_net and delta F1.

### 6.4 What each outcome means

| Outcome | What it means | What to do |
|---|---|---|
| Inflation above 1.2x, CI excludes 1, model error under 10% | The cache finding is real and the model works | Use the full plan in section 2 |
| Inflation 1.05x to 1.2x | The effect is real but modest | Lead with the CPU break-even model and the scorer finding; keep the cache part as a methods section |
| Inflation under 1.05x | The old contamination was smaller than I estimated, or the correction extrapolated badly | Drop the cache claim; write a CPU break-even and modelling study |
| Model error 10% to 20% on short prompts | Calibration is incomplete | Add more short-prompt points; report the error bins honestly |
| Model error above 20% | Model form fails at short prompts | Try an additional intercept or piecewise form; if it still fails, limit the claim to prompts above the fitted range |
| Gate fails in the cold condition | A prefix is still being shared | Fix the leak before using any number |
| BM25 beats the cross-encoder on S_net with F1 within the equivalence margin | Strong practical finding | Make it a headline in the break-even section |
| BM25 loses clearly on F1 | The cross-encoder is worth its cost for some tasks | Report per-dataset and per-model; map where each wins |
| DECAF beats top-m at equal budget | Selector may deserve a claim | Check with Holm correction before saying so |
| DECAF equals top-m | Selector is not the contribution | Present it as an instance only |

### 6.5 Files to send back to me

So I can check and analyse the new results, bring back these files (zip them):

1. `cache_v3/` raw JSONL for the main grid, warm-order pass, scaling extension and pilot.
2. `ENVIRONMENT.md` (hardware and software versions).
3. `hypotheses.md` (dated, committed before the run).
4. The harness diff (the changes to `run_v2.py` / `llm.py`) or the new runner.
5. A short run log: start and finish times, interruptions, number of rejected rows.
6. Your own `results_v3/` tables if you already generated them.

I will recompute the gate, inflation, validation and statistics independently and tell you where I agree or disagree.

### 6.6 Main cold re-run: design and scope

For each question, in a **random order** drawn with a logged seed:

1. **Cold condition (headline numbers).** Every prompt starts with a unique nonce. Methods: `full`, `decaf_lam6`, `fixed_0.2`, `rel_iso` (top-m at DECAF's budget), `bm25_decaf_lam6`, and the LLMLingua arm (name the variant correctly).
2. **Warm-order condition (the inflation measurement).** No nonce; `full` first, then each compressed method. This reproduces the old design on purpose. Methods: `decaf_lam6`, `fixed_0.2`, `rel_iso`, `bm25_decaf_lam6`.

Same questions, same retrieved contexts, same seed, same generation settings (temperature 0, 32 output tokens, same context window) in both conditions. Run all cold work in a separate pass from the warm-order pass, with the nonce first in the whole prompt.

| Item | Main | Extension |
|---|---|---|
| Configs | 3B Q4_K_M, 7B Q4_K_M | 3B Q8_0, 7B Q8_0 at n=50 |
| Datasets | HotpotQA, 2WikiMultiHopQA, TriviaQA, LongBench | same |
| Questions per dataset | 100 | 50 |
| Cold runs | 2 x 4 x 100 x 6 = 4,800 | +1,200 |
| Warm-order runs | 2 x 4 x 100 x 4 = 3,200 | +800 |
| Total | about 8,000 | about 10,000 |

Note: in an earlier message I said "roughly 4,000 runs" for the reduced scope. That corresponds to n=50 per dataset; the design above at n=100 is about 8,000.

Rough time **[estimate]**: about 6 to 10 hours for the main grid at n=100 (from logged run times). Replace after the pilot.

Fields to log per run: model, quant, dataset, qid, method, condition, run-order index, seed, nonce string, prompt hash, prompt_tokens, output_tokens, prefill_ms, decode_ms, total_ms, ttft_ms (client), e2e_ms (client), compress_ms, retained ratio, answer, F1, EM, evidence recall, peak RSS, cached-token count if available, timestamp.

Common mistakes to avoid:
- Nonce placed after the system prompt or template (shared prefix still cached).
- Cold and warm passes interleaved so cold runs warm the cache.
- Cross-encoder warm-up counted in the first question.
- Different question sets per method.
- Means that include corrupted-timer rows.

---

## 7. Q1 gates and decision rules

### Gate A — Evidence (after the cold re-run)

| Test | Pass |
|---|---|
| Cold gate | Median logged/cold prefill ratio 0.8 to 1.25 for every method |
| Inflation measured | Warm-order vs cold ratio reported with CI for each config |
| Same questions across methods | Asserted by script |
| No corrupted timer rows | Zero rows with `prefill_ms > e2e_ms` in final data |
| Model validated out of sample | Mean error under 10% on short compressed prompts |

### Gate B — Claim (decides the framing)

| Result | Framing | Realistic tier |
|---|---|---|
| Inflation clearly above 1.2x with CI excluding 1, model error under 10% | Measurement-validity plus model paper | Q1 plausible, not assured |
| Inflation 1.05x to 1.2x | Cache effects modest but must be checked; lean on CPU model, break-even and scorer finding | Q2 more likely; Q1 only with extra strength elsewhere |
| Inflation under 1.05x | Drop the cache claim; CPU break-even and modelling study | Q2 or Q3 |

Decide the framing from the data, then write. Do not write first and fit the data to the story. The tiers are my judgement, not a prediction of any journal's decision.

### Gate C — Honesty

- [ ] No claim that DECAF is a better selector unless the new data supports it at equal budget.
- [ ] Every "no difference" claim has an equivalence margin or is worded "no detectable difference".
- [ ] Cache reuse called the cause only if a counter confirms it; otherwise "consistent with".
- [ ] All negative results (LongBench, 7B) are in the main text.
- [ ] Limitations list the single CPU, batch size 1, Qwen2.5 only, short outputs, Ollama/llama.cpp only.
- [ ] Hypotheses were written down (dated) before the re-run.

### Gate D — Novelty

- [ ] Kummer et al. read in full and compared honestly, including where they already cover break-even.
- [ ] A fresh literature search the week you submit.
- [ ] Related work cites only published venues.

### Gate E — Reproducibility

- [ ] One command regenerates every table and figure from raw JSONL.
- [ ] Environment file in the repo and cited.
- [ ] Code and raw data archived with a DOI.
- [ ] Seeds and run orders logged.

### Gate F — Presentation

- [ ] Two-column journal style, short sentences, no thesis-report feel.
- [ ] Every number in the text matches a generated table (scripted check).
- [ ] References serial, complete, correct authors, titles, years.
- [ ] Figures from your own data.
- [ ] AI-use disclosure kept.

### What raises the odds, in order of value

1. A clean cold re-run with a clearly measured inflation.
2. A second hardware platform (another CPU) reproducing the effect.
3. A second runtime (for example `llama-server` with prompt caching off) reproducing it.
4. A third model family or a larger model.
5. A public benchmark script others can run in minutes.

---

## 8. Reviewer risks and your answers

| Reviewer will say | Answer you need |
|---|---|
| "Is this just a bug in your setup?" | Detection test, the LLMLingua no-shared-prefix control, a cache counter if available, the cold re-run on a clean setup, a second runtime if possible |
| "One CPU, one model family." | Add a second CPU or model family if you can; otherwise state it and limit claims |
| "Prior work already did break-even." | Own the overlap; the new parts are CPU-only, the cache finding, the validated closed-form model, selector cost |
| "Your selector is not better than simple baselines." | Agree and say so; the selector is a worked example |
| "Quality conclusions are underpowered." | Power or minimum-detectable-effect, equivalence margins, Holm correction |
| "Short outputs (32 tokens) hide decode." | Keep the crossover analysis over output length (16 to 512) and say what it shows |
| "Qwen2.5 3B/7B are small." | State it; the claim is about CPU-only local serving, where small models are the realistic case |
| "The cross-encoder is an arbitrary choice." | Show BM25 and two-stage alternatives; report selector cost as a variable |

---

## 9. Fallback plan and journal shortlist

### Fallback

If Gate B lands in the lower rows, do not abandon the work:
- Reframe as a CPU break-even and modelling study with the cheap-scorer finding.
- Target a solid Q2 journal, or a systems/IR conference, then extend toward a journal version.
- Keep the measurement protocol as a section; it is still useful to readers.

### Journals to check (scope fit, not guaranteed quartile)

I have not verified current quartiles, scope statements or review times for any of these. Check each journal's aims, recent paper types and present JCR/SJR category before choosing.

- Information Processing & Management
- ACM Transactions on Information Systems
- Journal of Systems Architecture
- Future Generation Computer Systems
- Engineering Applications of Artificial Intelligence

Choose by: does it publish measurement or benchmarking studies, does it accept negative or null findings, how long are reviews, and article processing charges and data/code rules.

---

## 10. Timeline (suggested; durations are estimates)

| When | Work |
|---|---|
| Day 1 | Phase 0 and 1 (tag repo, environment file, hypotheses, harness fixes); 20-question pilot; check gate; rescale plan |
| Day 2 | Phase 2 (short-prompt calibration, refit); start main cold grid |
| Days 3 to 4 | Main grid and warm-order pass finish (machine time); extension arms if time |
| Days 5 to 6 | Phase 4 analysis; decision gate; fix the framing |
| Week 2 | Phase 5 reading (Kummer PDF) and Phase 6 rewrite: methods, results, tables, figures |
| Week 3 | Complete draft; scripted number check; limitations; related work |
| Week 4 | Supervisor review; reproducibility package and DOI; journal choice; cover letter; submit |

If the pilot shows the machine is much slower than estimated, cut the main grid to n=50 per dataset rather than dropping datasets or configs.

---

## 11. Submission checklist

- [ ] Whole paper re-read once for claims beyond the evidence.
- [ ] Every citation's venue rechecked.
- [ ] Zip rebuilt from clean source and compiled without errors.
- [ ] GitHub link and archive DOI work; data/code availability section carries the GitHub link and the `PRELIMINARY_RESULTS.md` link.
- [ ] Cover letter states the contribution and the difference from Kummer et al.
- [ ] Supervisor (A.S.M. Shafi) has seen the final version.
- [ ] Author and affiliation lines added only if the journal's submission system needs them (the LaTeX project keeps them out by your instruction).
- [ ] Nothing after the references.

---

## 12. Appendix

### 12.1 Code snippets (adapt to your code; test before use)

**A. Nonce at the start of every prompt** (the nonce must be the first tokens of the text you send):

```python
import uuid

def with_nonce(prompt: str, tag: str) -> str:
    return f"[run {uuid.uuid4().hex[:12]}-{tag}] " + prompt
```

If your client applies a chat template that puts a fixed system prefix before your text, that fixed prefix can still be cached. It is a few tokens and it is the same for the scaling runs, but the gate (snippet C) is what proves it is harmless.

**B. Randomized method order per question:**

```python
import random

rng = random.Random(SEED)          # log SEED in the run metadata
for qid in question_ids:
    order = methods[:]             # list of method names
    rng.shuffle(order)
    for m in order:
        run_one(qid, m, order_index=order.index(m))
```

**C. Detection gate:**

```python
import numpy as np

def fit_cold(P, y):
    """Origin-constrained, relative-error-weighted prefill model: y ~ c1*P + c2*P^2."""
    P, y = np.asarray(P, float), np.asarray(y, float)
    A = np.c_[P, P**2] / y[:, None]
    return np.linalg.lstsq(A, np.ones(len(y)), rcond=None)[0]

def gate_ratio(rows, coef):
    P = np.array([r["prompt_tokens"] for r in rows], float)
    y = np.array([r["prefill_ms"] for r in rows], float)
    return float(np.median(y / (coef[0] * P + coef[1] * P**2)))

# For every (config, method) in the COLD condition:
#   ratio = gate_ratio(rows, coef[config]);  assert 0.8 <= ratio <= 1.25
```

**D. Timer guard:**

```python
def is_valid(row):
    return row["prefill_ms"] <= row["e2e_ms"] and row["prefill_ms"] > 0
# re-run any row that fails; count the rejections for table R9
```

**E. Cold-corrected end-to-end time** (for the model-based estimates in section 3.2; the re-run replaces these with measurements):

```python
t_cold = row["e2e_ms"] - row["prefill_ms"] \
         + coef[0] * row["prompt_tokens"] + coef[1] * row["prompt_tokens"] ** 2 \
         + row["compress_ms"]
```

### 12.2 Reading list

- Kummer, Jurkschat, Färber, Vahdati. "Prompt Compression in the Wild: Measuring Latency, Rate Adherence, and Quality for Faster LLM Inference." ECIR 2026, LNCS 16483, pp. 257-271. Read the PDF.
- Pan et al. LLMLingua-2 (ACL Findings 2024).
- RAGCache and CacheBlend (KV reuse for RAG): verify venues before citing.
- EdgeRAG (verify publication venue before citing; the page I fetched showed only an arXiv listing).
- Search again before submission for any 2026 work on CPU-only RAG latency or prefix-cache benchmarking pitfalls.

### 12.3 Standing constraints for the paper

- Keep the DECAF title (as the framework name).
- Never fabricate data. No claim of guaranteed Q1.
- Published-venue references only, no arXiv-only citations.
- No author or university lines in the LaTeX project.
- Data and code availability section carries https://github.com/Shoukhin1803078/decaf and the `PRELIMINARY_RESULTS.md` link.
- Nothing after the references.
- Use your own repo figures unchanged unless regenerated from the new data.
- Do not run the watermark-removal tool; the AI-use disclosure stays.
- Deliver a pure LaTeX zip.

### 12.4 What I did not check

- `PAPER_*.md` drafts, `summary_v2.json`, `analyze_v2.py` internals and `retrieval.jsonl` (hybrid retrieval) were not audited line by line.
- I did not verify that the repo's reported p-values reproduce exactly.
- I did not verify the LLMLingua arm's variant in the code.
- I did not read the Kummer et al. figures or profiler description.
- I did not verify any journal's quartile, scope or review time.

### 12.5 Glossary

- **Cold run:** a run whose prompt cannot reuse any cached prefix (unique nonce first).
- **Warm-order run:** a run in the old design: no nonce, `full` first, so later prompts can reuse the cache.
- **Inflation:** warm-order speed-up divided by cold speed-up.
- **S_net:** full-context time divided by (selector time plus compressed end-to-end time).
- **Gross speed-up:** same, without the selector time.
- **L\*:** the prompt length above which selection saves time after paying for the selector.
- **Gate:** automatic test that logged prefill matches the cold model (ratio 0.8 to 1.25).
