# DECAF — Exact Fixes (file, line, change, test)

Written 2026-10-15. Every item cites a real file and line in your uploaded repo (`decaf-main/decaf-pilot/`) or in the LaTeX paper project (`/home/claude/decaf_paper/`). Line numbers are from the files as I read them today; if you edit a file, the numbers shift, so search for the quoted code instead.

Companion files: `DECAF_Q1_MASTER_PLAN.md` (the overall plan) and `audit_gate.py` (a checker script, section 9).

**Severity**
- **S1 Blocker:** the paper's central numbers are wrong or unsupported until this is fixed.
- **S2 Required:** reviewers will reject or ask for major revision without it.
- **S3 Should fix:** weakens the paper or confuses readers.

---

## 1. All fixes at a glance

| ID | Sev | Where | Problem in one line |
|---|---|---|---|
| FX-01 | S1 | `src/run_v2.py` L200 | Re-run into the same cache is silently skipped (key has no `condition`) |
| FX-02 | S1 | `src/run_v2.py` L232-261 | No nonce; `full` runs first; method order fixed; prompts are prefixes of each other |
| FX-03 | S1 | `src/llm.py` L33-38, `src/run_v2.py` L120-122, L154-156 | Nonce sits inside the context, after ~25 shared template tokens; place it first everywhere |
| FX-04 | S2 | `src/select_v2.py` L51-59 | Why contamination is graded by method; relevance-order ablation is confounded |
| FX-05 | S2 | `src/run_v2.py` L181-194 | Selector time measured once, cached, shared by all 4 models; includes warm-up |
| FX-06 | S3 | `src/run_v2.py` L246-283 | `full` row stores a non-zero selector time |
| FX-07 | S1 | `src/run_experiments.py` L64-77 | No guard against the ~1,000,000 ms `prefill_ms` rows |
| FX-08 | S1 | `config_v2.yaml` L69, run commands | 9 of 15 methods have only 30 questions on 2 datasets |
| FX-09 | S2 | `config_v2.yaml` L55-58 | Calibration starts at 361 tokens, 2 replicates; compressed prompts have a median of 158 to 184 tokens |
| FX-10 | S2 | `config_v2.yaml` L33, `src/run_v2.py` L209 | "Full context" is only the top-5 BM25 paragraphs (median 858 to 1,221 tokens); no real long-context regime |
| FX-11 | S3 | `src/llm.py` L61-73, `src/run_v2.py` L50-79 | No thread count/batch pinned; memory sampler runs during timed requests |
| FX-12 | S3 | `config_v2.yaml` L71-72 | `lambda: 1.5` and `fixed_ratios` are unused by the named methods |
| AN-01 | S1 | `src/analyze_v2.py` L74-81 | Latency fit is unweighted linear with intercept, not the model the paper describes |
| AN-02 | S2 | `src/analyze_v2.py` L415-431 | Break-even uses that linear slope; use the quadratic cold model |
| AN-03 | S1 | `src/analyze_v2.py` L478-492 | S_net is a ratio of medians; 30-question arms are divided by the 200-question `full` median |
| AN-04 | S2 | `src/analyze_v2.py` L905-930 | Headline p-values are McNemar on EM, Holm within 14 methods, but claims are about F1 |
| AN-05 | S2 | `src/analyze_v2.py` L291-330, docs | H5 "~99%" is an identical-prompt test, presented as a finding |
| AN-06 | S1 | new | No automatic cold gate in the analysis |
| DOC-01..07 | S1/S2 | repo docs | Specific sentences to retract or reword (section 5) |
| PAP-01..08 | S1/S2 | LaTeX paper | Paper is built on the old 210-request pilot, not v2 (section 6) |

---

## 2. Code fixes — the harness (`src/`)

### FX-01 [S1] Cache key hides the re-run
**File:** `src/run_v2.py` L200 and L233.

```python
cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["compress_cache"]),
                   ["model", "dataset", "qid", "method"])
...
if cache.has(model=model["name"], dataset=dataset_tag, qid=ex.qid, method=method):
    continue
```

**Problem:** if you point a cold re-run at the same file, every (model, dataset, qid, method) already exists, so the loop skips everything and reports "0 new runs". Even with a new file, the warm-order condition needs its own rows.

**Change:**
1. Add `"condition"` to the key list, and pass `condition=condition` in `cache.has(...)` and in the stored record.
2. Write to a new file: add `compress_cache_v3: cache_v3/compression.jsonl` in `config_v2.yaml` L99-106 and use it. Never write into `cache_v2/`.

**Test:** run 2 questions, 2 conditions; the file has 2 × 2 × (number of methods) rows, none skipped.

---

### FX-02 [S1] No nonce, fixed order, `full` first
**File:** `src/run_v2.py` L232-261.

```python
full_ctx = None
for method in methods:                       # L232: config order = full, fixed_0.8, fixed_0.6, ...
    ...
    prompt = build_prompt(ex.question, [ctx], ["evidence"])   # L260: no nonce
    resp = _generate(client, prompt)                          # L261
```

**Problem:**
- `methods` is `[full, fixed_0.8, fixed_0.6, fixed_0.5, fixed_0.4, fixed_0.2, rel_iso, decaf_lam2, decaf_lam6, ...]` (`config_v2.yaml` L69). For every question the server sees `full` first, then a chain of prompts that each start with the same text as the one before.
- In `select_v2.py`, `full` returns `order` (all sentences sorted by relevance) and `fixed_r` returns `order[:m]` (L51-56). `build_context` joins them in that order. So each `fixed_r` prompt is a literal prefix of the `full` prompt, followed only by the question.
- Your own scaling runs add a nonce (L120-122), but this compression loop does not. `STATUS.md` L311-313 describes the nonce as the fix for the prefix-cache problem, but it was applied to scaling only.

**Change (cold condition = headline numbers):**

```python
import random, uuid, zlib

def with_nonce(prompt: str, tag: str) -> str:
    # nonce must be the FIRST characters of the text sent to the server
    return f"[run {uuid.uuid4().hex[:12]}-{tag}]\n" + prompt

# inside run_compression, replace `for method in methods:` (L232) with:
rng = random.Random(cfg["seed"] * 1_000_003 + zlib.crc32(ex.qid.encode()))   # NOT hash(): Python salts str hashes per process
order = list(methods)
if condition == "cold":
    rng.shuffle(order)                        # random order, reproducible
for pos, method in enumerate(order):
    ...
    prompt = build_prompt(ex.question, [ctx], ["evidence"])
    if condition == "cold":
        prompt = with_nonce(prompt, f"{model['label']}-{dataset_tag}-{ex.qid}-{method}")
    resp = _generate(client, prompt)
    ...
    cache.put({..., "condition": condition, "order_pos": pos, "seed": cfg["seed"], ...})
```

**Warm-order condition (measures the inflation):** `condition="warm_order"`, no nonce, `order = list(methods)` (config order, `full` first). This reproduces the old design on purpose. Run it as a separate pass, after the cold pass.

**Test:** `audit_gate.py` check C3 passes for the cold condition (median ratio 0.80 to 1.25 for every method) and the warm-order rows show the old low ratios.

---

### FX-03 [S1] Nonce position must be identical in calibration and measurement
**Files:** `src/llm.py` L33-38 (`build_prompt`), `src/run_v2.py` L120-122 and L154-156, `src/h5_v2.py` (nonce lines).

**Problem:** the calibration nonce is inside the context text: `ctx = nonce + " " + " ".join(filler)`, which `build_prompt` places after the fixed instruction `"Answer the question using the provided context. Give a short, direct answer.\n\nContext:\n[1] reference: "`. Those first ~25 tokens are the same in every request, so they can be cached. For a 361-token prompt that is small; for a 150-token selected prompt it is not.

**Change:** use `with_nonce(...)` (FX-02) at character 0 of the whole prompt in **all** places: scaling, crossover, H5, compression. Then re-run scaling (FX-09), because the fitted constants change slightly.

**Test:** two prompts with identical contexts differ in their first line; the refit cold model reproduces the old fit within a few percent on the overlapping range.

---

### FX-04 [S2] Why contamination is graded, and what it does to your ablations
**File:** `src/select_v2.py` L51-59, L72-78.

**Observed (hotpot, median logged / cold-model prefill, 4 configs):**

| Method | Ratio | Prefix relation to `full` |
|---|---|---|
| full | 1.00 | — |
| fixed_0.8 | 0.98 to 1.02 | prefix of `full`, yet NOT reused (cause not identified) |
| fixed_0.6 / 0.5 / 0.4 | 0.08 to 0.12 | prefix of `full` and of the previous arm |
| fixed_0.2 | 0.17 to 0.20 | prefix |
| rel_iso | 0.27 to 0.31 | prefix (m from DECAF's count) |
| decaf_lam2 / lam15 | 0.35 to 0.51 | relevance-ordered subset, partial prefix |
| decaf_lam6 | 0.08 to 0.14 | relevance-ordered subset |
| decaf_nocov_lam6 | 0.27 to 0.30 | |
| decaf_noreorder_lam6 | 0.68 to 0.77 | document order, shorter shared prefix |
| bm25_decaf_lam6 | 0.76 to 0.82 | different order |
| llmlingua | 0.85 to 0.98 | rewritten text, no shared prefix |

**Consequences you must fix or state:**
1. Contamination differs by method, so even **comparisons between methods** are biased, not only absolute speed-ups.
2. The relevance-reordering ablation (`decaf_noreorder_lam6` vs `decaf_lam6`, for example 7B Q4 HotpotQA E2E 1,577 ms vs 874 ms) is **confounded by the cache**: re-run it cold before claiming that reordering helps latency.
3. `fixed_0.8` behaving cold while `fixed_0.6` and below are not is **unexplained**. Do not write a causal story for it. Note it, and let the cold re-run (with counters if possible) settle it.

**Change:** none in `select_v2.py` itself. Re-run the ablation arms cold (FX-08) and report latency only from cold runs.

---

### FX-05 [S2] Selector time is measured once and shared
**File:** `src/run_v2.py` L181-194 (`_ce_scores`) and L218.

```python
path = os.path.join(ce_dir, f"{dataset}_{ex.qid}.json")
if os.path.exists(path): ... return rec["rel"], rec["ce_ms"]
rel, ce_ms = scorer.score(ex.question, sentences)
```

**Evidence from the data:**
- For every (dataset, qid) the stored `ce_ms` is **identical across all four models** (max 1 distinct value), so the selector time is one measurement reused 4 times.
- The first HotpotQA values are `1270, 965, 1902, 920, 1056` ms against a median of about 900 ms: no warm-up was discarded.
- `LLMLinguaCompressor.compress` (`llmlingua_v2.py` L24-31) is timed live, so DECAF's selector time and LLMLingua's are measured under different protocols.

**Change:**
1. Keep caching the **scores** (they are deterministic) but not the **time**. Time the scorer live in every run: `rel, ce_ms = scorer.score(...)`, or take the median of 3 calls.
2. Before the first timed question of each process, do 5 throw-away scoring calls (and one LLMLingua call).
3. Log `ce_ms` per run, and the number of sentence pairs scored.
4. Report the selector time as a distribution (median, IQR), not a single number.

**Test:** `ce_ms` varies across models for the same question; no first-question spike in the logs.

---

### FX-06 [S3] `full` stores a selector cost
**File:** `src/run_v2.py` L253 and L279-280.

For `method == "full"` the code takes `rel_used, ce_eff = rel, ce_ms`, so the `full` rows carry `ce_ms` ≈ 900 to 1,270 and `compress_ms` > 0. The analysis zeroes this (`analyze_v2.py` L480), but the printed tables show "Full context T_c 904.8 ms", which looks like a bug.

**Change:** `if method == "full": ce_eff = 0.0` and `stats["select_ms"] = 0.0` before `cache.put`.

---

### FX-07 [S1] Corrupted timer rows
**File:** `src/run_experiments.py` L64-77 (`_timing`), used by `run_v2.py` L281.

**Problem:** 18 of 21,360 rows have `prefill_ms` near 1,000,000 ms while the client-side `e2e_ms` is normal (for example `prefill_ms = 1,062,042`, `e2e_ms = 3,194`). 15 are `full` rows for 7B Q8 LongBench. Cause unknown (server counter glitch is my guess; investigate).

**Change:** inside `_generate` (or a wrapper), validate and retry:

```python
def _valid(t):  # t = _timing(resp)
    return 0 < t["prefill_ms"] <= t["e2e_ms"] and t["output_tokens"] > 0

for attempt in range(3):
    resp = _generate(client, prompt)
    if _valid(_timing(resp)):
        break
else:
    log_rejected(...)      # keep a count for the paper's exclusions table
```

**Test:** `audit_gate.py` check C1 reports 0 invalid rows; the rejection log is saved with the data.

---

### FX-08 [S1] Unequal question sets
**File:** `config_v2.yaml` L69 and the commands used to run.

**Observed (3B Q4):** `full`, `fixed_0.2`, `fixed_0.5`, `rel_iso`, `decaf_lam6`, `decaf_nocov_lam6` have 200 questions on all 4 datasets. These nine have **30 questions on HotpotQA and 2Wiki only**: `fixed_0.8`, `fixed_0.6`, `fixed_0.4`, `decaf_lam2`, `decaf_lam15`, `decaf_noreorder_lam6`, `bm25_decaf_lam6`, `llmlingua_0.5`, `llmlingua_0.2`. They have **0 questions on TriviaQA and LongBench**.

**Change:**
1. Choose the final method list (see the master plan: `full`, `decaf_lam6`, `fixed_0.2`, `rel_iso`, `bm25_decaf_lam6`, `llmlingua_0.2`, plus the ablations you want to keep) and run **every** method on the **same** questions on all four datasets.
2. Add an assertion at the end of `main()` that the qid sets are equal across methods per (model, dataset).

**Test:** `audit_gate.py` check C4 passes.

---

### FX-09 [S2] Calibration range and replicates
**File:** `config_v2.yaml` L55-58.

```yaml
scaling:
  lengths: [256, 512, 1024, 2048, 4096, 8192, 16384]
  replicates: 2
```

**Problem:** the smallest prompt is 361 tokens; the 5th percentile of your compressed prompts is 112 tokens, and the DECAF median is 158 tokens (HotpotQA) and 184 (LongBench). Two replicates per length give no usable variance.

**Change:** `lengths: [32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384]`, `replicates: 5` (or more at short lengths, which are cheap). Re-run after FX-03. `contexts.filler_sentences` (`contexts.py` L41-56) adds whole sentences until the target is reached, and the template adds its own tokens, so the real prompt will be longer than the nominal length. Check and report the actual `prompt_tokens` of the shortest points.

**Test:** `audit_gate.py` checks C5 and C6 pass; the refit model is validated on real short prompts.

---

### FX-10 [S2] The "long context" is not long
**Files:** `config_v2.yaml` L33 (`top_k: 5`), `src/run_v2.py` L209: `ranked = rank_paragraphs(ex, cfg["retriever"])[: cfg["top_k"]]`.

**Problem:** every dataset, including LongBench, feeds the five best BM25 paragraphs to the reader. Median full-context prompts are 858 tokens (HotpotQA) and 1,221 (LongBench). The regime where prior work reports its larger gains (several thousand tokens) is covered only by the synthetic filler of the scaling runs, never by real questions.

**Change:** add a `top_k` sweep (for example 5, 10, 20) for HotpotQA and LongBench, and run `full`, `decaf_lam6`, `fixed_0.2`, `bm25_decaf_lam6`, `llmlingua_0.2` at each. Call the setting by its token length, not "long context".

**Test:** a table of S_net against measured full-prompt length with at least one bin above 3,000 tokens.

---

### FX-11 [S3] Hardware settings and sampler overhead
**Files:** `src/llm.py` L61-73 (payload options), `src/run_v2.py` L50-79 (`MemSampler`).

**Changes:**
1. Add `num_thread` and `num_batch` to the options (and record Ollama's version) so the timing conditions are reproducible. Pin them; do not tune per method.
2. `MemSampler` calls `psutil.process_iter` every 0.15 s on the same CPU during timed requests. Either turn it off for the timing runs and measure peak memory in a separate small pass, or measure its effect (run 50 requests with and without) and report it.

---

### FX-12 [S3] Unused config entries
**File:** `config_v2.yaml` L71-72.

`lambda: 1.5` is only a default; every named method (`decaf_lam2`, `decaf_lam6`, `decaf_lam15`) parses its own λ from the name (`select_v2.py` L64-65). `fixed_ratios: [0.5, 0.2]` is not read by any Python file in `src/` (I searched for it). Remove them or document them, so a reader does not think λ=1.5 was used.

---

## 3. Code fixes — the analysis (`src/analyze_v2.py`)

### AN-01 [S1] The latency fit is not the paper's model
**Lines:** L74-81 (`_fit`), used for the scaling fits.

`np.polyfit(x, y, 1)` is an ordinary unweighted linear fit with an intercept, on 14 points per config (7 lengths × 2 replicates). The long prompts dominate it. The paper and the master plan describe an origin-constrained, relative-error-weighted model with a quadratic prefill term.

**Change:** add `fit_cold` (below) and use it for prefill; keep the old linear fit only as a labelled comparison.

```python
def fit_cold(P, y):
    P, y = np.asarray(P, float), np.asarray(y, float)
    A = np.c_[P, P**2] / y[:, None]
    return np.linalg.lstsq(A, np.ones(len(y)), rcond=None)[0]   # (c1, c2)
```

**Test:** held-out error on full-context compression prompts is about 3% (I measured 2.8% for 3B Q4 and 2.9% for 7B Q4); report it per prompt-length bin.

---

### AN-02 [S2] Break-even uses the linear slope
**Lines:** L415-431.

```python
slope = b_pre + N * b_tpot
Lstar = Lc + (Tc / slope)
```

**Change:** solve numerically with the cold model: find the smallest L with `T_full(L) - T_compressed(L_c) >= T_c`, where `T_full(L)` = quadratic prefill + decode at length L. Report both forms. My check with the cold quadratic model gave **427 tokens (3B Q4)** and **274 (7B Q4)** against the repo's linear 342 and 254.

Also state how `L_c` (median prompt tokens of the DECAF arm) and `T_c` (median `compress_ms`) were obtained; after FX-05, `T_c` is a live measurement.

---

### AN-03 [S1] S_net is a ratio of medians over unmatched questions
**Lines:** L463-492.

```python
agg = cp.groupby(keys).agg(..., e2e_ms=("e2e_ms", "median"), compress_ms=("compress_ms", "median"), ...)
...
base = g[g["method"] == "full"]
be0 = float(base["e2e_ms"].iloc[0])             # median over ALL full-context questions (200)
agg.loc[g.index, "net_speedup"] = be0 / g["total_ms_net"]    # method rows may be n=30
```

**Problems:**
1. A ratio of group medians is not the median or mean of per-question speed-ups.
2. For the 30-question arms (LLMLingua, BM25, ablations) the numerator is the median of 200 `full` rows, a different set of questions. That is how "DECAF S_net ≈ 2.94× vs LLMLingua-2 ≈ 1.13×" was produced, on top of the cache contamination.

**Change:** compute per question `s_q = T_full(q) / (T_selector(q) + T_compressed(q))` on the **matched** qid set, then report the geometric mean with a bootstrap 95% CI (resample questions). Keep the median version only as a labelled secondary column.

---

### AN-04 [S2] Which test is behind the p-values
**Lines:** L905-930.

```python
p, n10, n01 = _mcnemar(b.loc[common, "em"].values, a.loc[common, "em"].values)   # exact-match
d_f1 = ...; _, lo, hi = _boot_ci((a.loc[common, "f1"] - b.loc[common, "f1"]).values)
...
adj = _holm(pv)      # within one (model, dataset): 14 comparisons
```

**What this means for your headline numbers:**
- "ΔF1 +0.077, Holm p = 0.009" (`PAPER_ABSTRACT.md` L26-27): the 0.0094 is the **Holm-adjusted McNemar test on exact match** (`mcnemar_p` 0.0009 with only 22 discordant pairs: n10=3, n01=19), not a test on F1. The F1 interval comes from a separate, uncorrected bootstrap.
- For the 7B, `decaf_lam6` has Holm p = 0.019, but the matched-budget `rel_iso` has Holm p = 0.056 and `fixed_0.5` has ΔF1 = −0.019 (p = 1.0). So "compression hurts the 7B" holds at about 12% retention and not at 50%.
- The Holm family covers one (model, dataset) cell, not all comparisons in the paper.

**Change:**
1. Add a paired test on F1 (sign-flip permutation on per-question ΔF1) and report it next to the EM-based McNemar test; name the test in every table.
2. Define the multiple-comparison family before the run (for example all primary comparisons across models and datasets) and apply Holm to that.
3. Report quality across budgets (several retention levels), not only at DECAF λ=6.

---

### AN-05 [S2] H5 is an identical-prompt test
**Lines:** L291-330 and `h5_v2.py`.

The "~99% TTFT reduction" compares a cold request with a repeat of the **identical** prompt (`r_warm = cl.generate(p_cold)`), and "~92 to 96%" with a repeat that shares the whole context. That is the expected behaviour of any prefix cache. It does not show that "KV-cache reuse is a larger, safer win than a cleverer compressor" (`PAPER_ABSTRACT.md` L50), because different RAG questions retrieve different paragraphs and share almost no prefix.

**Change:** keep H5 as the calibration of the detection test (it shows what a cache hit looks like in the logs: full `prompt_tokens`, tiny `prefill_ms`). Remove it as a headline result. If you want a KV-reuse claim, test it on a repeated-document workload.

---

### AN-06 [S1] No automatic cold gate
Add the gate as a first step of the analysis, and stop with an error if it fails. `audit_gate.py` (section 9) already implements it: copy its `fit_cold` and C1 to C6 checks into `analyze_v2.py` (or call the script from the Makefile / run script).

---

## 4. New or changed experiments (checklist)

- [ ] Cold condition: nonce at character 0, shuffled order, all methods, same questions (FX-01 to FX-03, FX-08).
- [ ] Warm-order condition: old design, separate pass (FX-02).
- [ ] Short-prompt calibration, 5+ replicates (FX-09).
- [ ] Live selector timing with warm-up discarded (FX-05).
- [ ] `top_k` sweep for a real long-prompt regime (FX-10).
- [ ] LLMLingua and BM25 arms on all four datasets at the full question count (FX-08).
- [ ] Ablations (`noreorder`, `nocov`, λ sweep) re-run cold (FX-04).
- [ ] Ollama version, thread count, batch size logged (FX-11).
- [ ] Cache counters if the server exposes any; otherwise state the evidence is timing-based.

---

## 5. Repo documentation: statements to retract or reword

These sentences describe results that the cold re-run may change. Reword them now so the repo does not contradict the paper.

| ID | File and line | Current claim | Fix |
|---|---|---|---|
| DOC-01 | `results_v2/PAPER_ABSTRACT.md` L26-27 | "compression significantly helps the 3B (ΔF1 +0.077, Holm p = 0.009) yet significantly hurts the 7B (ΔF1 −0.092, p = 0.019)" | Name the test (McNemar on exact match, Holm within 14 methods), add the F1 test, and state the budget (about 12% retention) |
| DOC-02 | `results_v2/PAPER_ABSTRACT.md` L33 and L50, `PAPER_RESULTS.md` L241-243, `PAPER_DISCUSSION.md` L18, L107, `STATUS.md` L138, `README.md` L43 | "prefix/KV reuse cuts TTFT by ~99%"; "KV-cache reuse … larger, safer wins than a cleverer compressor" | Describe H5 as an identical-prompt calibration; delete the comparison with compressors (AN-05) |
| DOC-03 | `STATUS.md` L311-313 | The nonce "fixed" the prefix-cache problem | Add: "applied to scaling and crossover only; compression runs were not protected" |
| DOC-04 | `results_v2/FULL_RESULTS_v2.md` headline ("Overhead matters … 3.71× falls to S_net 2.00×") and every speed-up and S_net table | Logged speed-ups and S_net | Mark "cache-contaminated, do not cite"; replace after the cold re-run |
| DOC-05 | `results_v2/FULL_RESULTS_v2.md` headline "vs LLMLingua-2 … DECAF ≈2.94× vs ≈1.13×" | Unequal protocols and unmatched questions | Remove until a matched cold comparison exists (AN-03) |
| DOC-06 | `results_v2/FULL_RESULTS_v2.md` "Full context" rows | `T_c` 904.8 ms shown | Show 0 (FX-06) |
| DOC-07 | `results_v2/FULL_RESULTS_v2.md` L9-11 ("H1 refuted on this setup") | H1 refuted at 32 output tokens | Say "at 32 output tokens"; keep the crossover over output length visible |

---

## 6. LaTeX paper (`/home/claude/decaf_paper/`): what to change

The paper I built earlier uses the original **210-request pilot** (30 questions), not the v2 repo. After the cold re-run, rebuild the numbers from v3. Specific text that must change:

| ID | File and line | Problem | Fix |
|---|---|---|---|
| PAP-01 | `sections/00_abstract.tex` L4-8 | DECAF framed as the evaluated method; speed-ups from the pilot | New abstract: lead with the measurement claim (section 2.5 of the master plan), then the cold numbers |
| PAP-02 | `sections/01_introduction.tex` L8, L18 | "We did not find a study that treats these three questions together" | Soften and cite Kummer et al. (ECIR 2026) as the closest work; state that it uses GPUs |
| PAP-03 | `sections/02_related_work.tex` L41 | "none fits a CPU cost model, charges the selector's overhead and states a break-even condition together" | Kummer et al. report break-even with compression overhead on GPUs, so rewrite as "none does this for CPU-only serving with a closed-form model" after you have read the PDF |
| PAP-04 | `sections/02_related_work.tex` L63 | Table row "Measurement validity: rarely discussed" | Replace with a cited statement, or remove if you cannot support it |
| PAP-05 | `sections/05_setup.tex` L9 | Scaling had 10 requests, 362 to 4,356 tokens, no nonce | Describe the v3 calibration (range, replicates, nonce at character 0) |
| PAP-06 | `sections/06_results.tex` L54 | "Compressed prompts keep sentences in document order" | In the v2 code `full` and `fixed_r` are in **relevance order** (`select_v2.py` L51-56), and only `decaf_noreorder` uses document order; re-check against the code that produced the data |
| PAP-07 | `sections/06_results.tex` L156-159 and `sections/07_discussion.tex` L26, L33, `08_conclusion.tex` L8 | Cache analysis on 210 pilot requests with a fitted offset | Replace with the paired cold-vs-warm-order measurement (inflation factor with CI) and the gate figure; keep the fitted-offset model only if still useful |
| PAP-08 | `scripts/make_results.py`, `reanalysis.py`, `reanalysis2.py`, `tables/`, `numbers*.tex` | Built from pilot data | Add a v3 script that regenerates every table, figure and `numbers.tex` macro from `cache_v3/` in one command |

Also: add the Kummer et al. BibTeX entry (take the details from the Springer record: ECIR 2026, LNCS 16483, pp. 257-271) and keep every reference to a published venue.

---

## 7. Order of work

| Step | Fixes | Why in this order |
|---|---|---|
| 1 | Tag the repo; set up a new `cache_v3/` | Keep v2 as evidence |
| 2 | FX-01, FX-02, FX-03, FX-07 | The harness cannot produce clean numbers without them |
| 3 | FX-05, FX-06, FX-11, FX-12 | Small changes in the same files |
| 4 | FX-09, then re-run scaling | The cold model must exist before you judge any compressed run |
| 5 | 20-question pilot + `audit_gate.py` | Check the fix before spending hours |
| 6 | FX-08, FX-10, then the main run | Large machine time |
| 7 | AN-01 to AN-06 | Analysis on clean data |
| 8 | DOC-01 to DOC-07 | Keep the repo honest |
| 9 | PAP-01 to PAP-08 | Rewrite the paper from the final numbers |

---

## 8. What I could not determine

- **Why `fixed_0.8` is not reused** although it is a prefix of `full` (FX-04). Possibly a threshold in the server's cache logic; I did not verify it.
- **The cause of the 18 corrupted `prefill_ms` rows** (FX-07).
- Whether Ollama's version in your setup exposes any cache counter.
- `analyze_v2.py` was read around the lines cited, not end to end; there are 1,204 lines, so other places may carry the same median/linear-fit logic.

---

## 9. The checker script

`audit_gate.py` runs on any scaling and compression JSONL pair and checks C1 to C6 (invalid rows, duplicates, the cold gate, equal question sets, calibration range, calibration replicates). Usage:

```
python3 -I audit_gate.py --scaling cache_v3/scaling.jsonl --compression cache_v3/compression.jsonl
```

Exit code 0 means all checks pass; 1 means at least one failed.

**Result on your current v2 data (so you can see it works):**

```
[FAIL] C1 invalid timer rows: 18 of 21360 rows
[PASS] C2 duplicate keys: 0
[FAIL] C3 cold gate: 126 of 168 (config, dataset, method) groups outside [0.80, 1.25]
        fixed_0.6 0.07 | fixed_0.5 0.08 | fixed_0.4 0.10 | decaf_lam6 0.11 | fixed_0.2 0.14
        decaf_nocov_lam6 0.25 | rel_iso 0.25 | decaf_lam2 0.28 | decaf_lam15 0.31
        decaf_noreorder_lam6 0.50 | bm25_decaf_lam6 0.77
[FAIL] C4 same questions for every method: 9 of 16 cells differ (sizes 30 vs 200)
[FAIL] C5 calibration range: scaling min 361 tokens; compressed 5th percentile 112 tokens
[FAIL] C6 calibration replicates: 28 of 28 (model, length) points below 5
RESULT: 5 CHECK(S) FAILED
```

Your goal after the re-run is `ALL CHECKS PASSED` on the cold condition.

Note: C3 applies only to rows whose `condition` is `cold` (rows without that field are treated as cold). Expect the warm-order rows to fail it, which is the contrast you want to publish.
