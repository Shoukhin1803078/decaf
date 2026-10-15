# DECAF — Status & Progress

**Date:** 2026-10-12
**Scope:** Phase 1 (v2) — **full experiments** for the DECAF proposal
(*"When Does Context Compression Accelerate CPU RAG? A Decode-Centric Study"*).
**State:** ✅ Complete, reproducible end-to-end. 23,748 runs across 4 model×quant configs
(3B/7B × Q4_K_M/Q8_0) × **4 datasets** (HotpotQA, 2Wiki, TriviaQA, LongBench), all at n=200; 42 figures
+ report in `results_v2/`. The Phase-0 pilot is preserved (§3) and superseded.

Location: `decaf-pilot/`

---

## 0. Current status at a glance

| | Phase 0 (pilot) | Phase 1 (v2, **current**) |
|---|---|---|
| Models | 3B Q4_K_M | 3B/7B × Q4_K_M, Q8_0 |
| Datasets | HotpotQA | HotpotQA + 2Wiki + TriviaQA + LongBench |
| Scorer | BM25 (~1.4 ms) | cross-encoder `bge-reranker-base` (~960 ms) |
| Baselines | fixed ratio, relevance | + **LLMLingua-2** |
| Long context | — | up to **16K** tokens |
| Generation sweep | — | N_out ∈ {16, 64, 128, 256, 512} (crossover) |
| Secondary (H5) | — | prefix/KV reuse + prefetch |
| Statistics | — | bootstrap CI · McNemar · Holm |
| Runs | 220 | **23,748** |
| Deliverables | `results/` | `results_v2/` (42 figures, `FULL_RESULTS_v2.md`) |

---

## 1. Phase 1 (v2) — full experiments

### 1.1 What was done

Matrix: **4 model×quant configs × 4 datasets × 15 methods**, plus scaling, crossover, a retrieval-mode
study and an H5 micro-study. All runs are cached (resumable, append-only JSONL); nothing from Phase 0 was modified.

| Family | Runs | Description |
|---|---:|---|
| Scaling (E1/E8) | 56 | TPOT(L)/prefill(L) at L ∈ {256 … 16384}, 2 reps × 4 configs |
| Scaling — compressed (E1) | 48 | full vs 25%-compressed context curve, 6 lengths × 2 reps × 4 configs |
| Crossover (H1) | 60 | decode share vs N_out ∈ {16,64,128,256,512} at L ∈ {512,2048,8192} × 4 configs |
| Compression (E2–E7, E9) | 21,360 | 4 configs × 4 datasets × 200 q × methods (+ n=30 extras) |
| Retrieval modes (§13) | 2,160 | BM25 vs dense vs hybrid, 4 configs × 3 datasets × 3 methods |
| H5 secondary | 64 | cold/warm/partial TTFT prefix-cache study × 4 configs × 2 lengths |
| **Total** | **23,748** | 3B/7B × Q4_K_M/Q8_0 |

**New code (all additive, `*_v2`):**

| File | Purpose |
|---|---|
| `config_v2.yaml` | v2 config (models, datasets, methods, H5, paths) |
| `src/data_v2.py` | unified HotpotQA + 2WikiMultihopQA loaders (schema-agnostic) |
| `src/scorer_v2.py` | cross-encoder relevance scorer (`bge-reranker-base`), times T_compress |
| `src/select_v2.py` | DECAF CPU-aware gate + baselines on cross-encoder relevance |
| `src/run_v2.py` | Phase A driver (scaling / crossover / compression), memory sampling, resumable |
| `src/llmlingua_v2.py` | LLMLingua-2 external baseline wrapper |
| `src/h5_v2.py` | H5 prefix/KV-cache + prefetch micro-study |
| `src/retrieval_v2.py` | BM25 / dense / hybrid retrieval-mode study (§13) |
| `src/analyze_v2.py` | Phase B: 42 figures, stats, `FULL_RESULTS_v2.md`, `summary_v2.json` |
| `scripts/run_v2.sh` | end-to-end v2 runner |

### 1.2 Key results

**E1/E8 — decode scaling (per model, now to 16K).** Including the 16K point exposes mild
prefill super-linearity; prefill, not decode, dominates for realistic contexts.

| model | prefill (ms/tok) | prefill r² | TPOT slope (ms/1k) | decode=prefill at L |
|---|---:|---:|---:|---:|
| 3B / Q4_K_M | 5.51 | 0.990 | +0.85 | ~867 |
| 3B / Q8_0 | 5.28 | 0.994 | +0.81 | ~916 |
| 7B / Q4_K_M | 10.92 | 0.998 | +1.30 | ~584 |
| 7B / Q8_0 | 10.57 | 0.996 | +1.24 | ~867 |

**E4 — break-even L\* (real cross-encoder overhead).** With T_compress ≈ **962 ms**, L\* is no
longer trivial: compression is a net latency loss below it, a net win above.

| model | L_c (retained tok) | T_compress (ms) | **L\*** (tok) |
|---|---:|---:|---:|
| 3B / Q4_K_M | 154 | 962 | **327** |
| 3B / Q8_0 | 154 | 962 | **335** |
| 7B / Q4_K_M | 154 | 962 | **241** |
| 7B / Q8_0 | 154 | 963 | **244** |

**E2/E3/E5/E6/E7 — quality, latency, overhead, grounding.** Example (HotpotQA, DECAF λ=6):

| model | F1 | retained | gold-in-ev | gross speedup | **S_net** (incl. T_compress) |
|---|---:|---:|---:|---:|---:|
| 3B / Q4_K_M | 0.291 | 0.128 | 0.633 | 3.66× | **1.89×** |
| 3B / Q8_0 | 0.330 | 0.128 | 0.633 | 2.75× | 1.69× |
| 7B / Q4_K_M | 0.522 | 0.128 | 0.633 | 9.81× | **4.50×** |
| 7B / Q8_0 | 0.556 | 0.128 | 0.633 | 7.85× | **4.03×** |

- **Overhead matters:** the ~960 ms cross-encoder cuts gross speedup roughly in half — the pilot's
  "always a net win" conclusion was an artifact of the ~1.4 ms BM25 scorer.
- **Model-size / quant interaction (E5/E6, now the full 2×2):** model **size** dominates — L\* ≈ 327–335 (3B)
  vs 241–244 (7B) — while **quantization barely moves L\*** (Q4 vs Q8 within ~1–8 tok at each size).
  Q8 gives higher F1 than Q4 at both sizes (e.g. 7B 0.611 vs 0.595 on full context), and higher precision
  is not slower here (Q8 prefill ≤ Q4 on this CPU, likely Q4_K dequant overhead). Compression *helps* the
  smaller models (distractor harm) but *hurts* the 7B on HotpotQA; grounding falls monotonically with
  retained evidence (0.833 full → 0.633).
- **E9 generalization:** 2Wiki is harder (best F1 0.336); same qualitative ordering.
- **E2 — ratio grid (0.2/0.4/0.5/0.6/0.8):** quality is **non-monotone** — the 3B peaks at ~50%
  retention (F1 0.330 vs 0.199 at full context), the 7B is ~flat from 60% up (0.584–0.595); grounding
  rises monotonically with retention (0.667 → 0.833). At **80% retention S_net ≈ 1.0×** (the
  compressor's ~960 ms overhead cancels the smaller prefill saving).
- **§17 — scorer ablation (cross-encoder vs BM25):** BM25 costs ~1.9 ms vs the cross-encoder's ~989 ms,
  yet the CE buys only small gains (7B Q8: F1 0.556 vs 0.539, gold-in-ev 0.633 vs 0.600) — on CPU the
  CE's ~500× cost rarely pays for itself.
- **§13 — retrieval modes (4 configs × 3 datasets):** **hybrid (BM25 + dense RRF) ≥ dense ≥ BM25** in
  most cells — e.g. 7B/Q8 HotpotQA F1 0.445 → 0.546 (dense) → **0.674 (hybrid)**; 7B/Q4 2Wiki
  0.228 → **0.315**. Retrieval quality often moves the result more than the compression policy.
- **Scale-up to n=200 (all 4 configs) — power flips the central claims:**
  - **3B/Q4:** compression **significantly helps** (DECAF λ6 ΔF1 +0.077, p=0.009; fixed-50% +0.073, p=0.005;
    fixed-20% +0.095, p=0.001) where n=30 showed nothing.
  - **7B/Q4:** compression **significantly hurts** (DECAF λ6 −0.092, p=0.019; rel_iso −0.092, p=0.056;
    no-cov −0.085, p=0.025) — a **significant model-size interaction** (E6/RQ4) that n=30 could not resolve.
- **E9 — long-context multi-doc (LongBench hotpotqa, n=200, all 4 configs):** compression
  **significantly hurts every config** (DECAF λ6 ΔF1 −0.146 (3B/Q4), −0.174 (3B/Q8), −0.145 (7B/Q4),
  −0.136 (7B/Q8); all Holm p<0.001; gold-in-ev 0.64 → 0.38). When the evidence is spread across a long
  context, aggressive selection drops it. This is a **uniform** regime, opposite to the short-context
  cases — compression's value is **strongly task/context-dependent**.
- **E9 — single-hop (TriviaQA, n=200, all 4 configs):** compression **helps 3B/Q4** (DECAF λ6 ΔF1
  **+0.081**, Holm p=0.002) and is **null for the 7B** (7B/Q4 +0.001, 7B/Q8 +0.003, n.s.). So the
  "compression hurts the 7B" effect is **multi-hop-specific**: on single-hop it is *neutral*, not
  harmful — the size interaction is **dataset-dependent**. (At n≈29 the 7B appeared +0.087; the full
  n=200 shows that was small-sample noise.)
- **Generation-length robustness:** extending N_out to 512 shows the decode share **plateaus** (~0.40 at
  short context) instead of rising — because the model hits **EOS at ~40–64 tokens**, so N_out cannot be
  forced higher. The prefill-dominated result therefore holds for *natural* answer lengths; testing true
  long-generation needs EOS-disabled decoding (a follow-up).

**Baseline — LLMLingua-2.** Its own compressor cost (~1.0–2.5 s) exceeds the cross-encoder's, so its
S_net (≈1.0–1.8×) trails DECAF's (≈1.5–4.5×), and DECAF keeps more gold evidence at aggressive ratios.

**§17 ablation + λ.** Coverage-off slightly raises evidence recall (+0.03); reorder-off ≈ neutral;
λ sweep trades retained/grounding for F1 as expected.

**H5 (secondary systems analysis).** llama.cpp prefix/KV-cache reuse cuts TTFT by **~99%** on an
identical prompt and ~92–96% on a shared-context/partial overlap — a *large* end-to-end win here
precisely because prefill dominates (contra the "reuse moves only TTFT" expectation).

**Statistics.** Paired bootstrap CIs, McNemar (EM) and Holm correction (n=30). Most method deltas
are not significant after correction — reported honestly, with the Pareto frontier as the story.

### 1.3 Deliverables (`results_v2/`)

- `FULL_RESULTS_v2.md` — 9 sections (0 headline · 1 scaling · 2 break-even · 3 compression ·
  4 λ/ablation · 5 statistics · 6 H5 · 7 retrieval · 8 figure index).
- `summary_v2.json` — all fitted curves, per-method stats, break-even, H5, retrieval, paired tests.
- `PAPER_RESULTS.md` — paper-ready **Results** (10 sections, tables + figure refs).
- `PAPER_METHODS.md` — paper-ready **Methods** (harness, DECAF algorithm, measurement framework, statistics, reproducibility).
- `PAPER_INTRO_RELATED.md` — paper-ready **Introduction + Related Work + positioning**.
- `PAPER_DISCUSSION.md` — paper-ready **Discussion + Conclusion** (incl. practical guidance).
- `PAPER_ABSTRACT.md` — **title, abstract, keywords, highlights**.
- `CITATION_CHECKLIST.md` — claim→source map + bibliography skeleton (fill verified metadata).
- **42 figures**: scaling (TPOT/prefill/TTFT/E2E/decode-share/RSS/panel), compressed-context curve, crossover,
  break-even curves + L* bar + overhead-vs-savings, Pareto (F1×E2E, ×TTFT, ×retained, frontier),
  TTFT-vs-TPOT, grounding, λ-sweep, ablation, E2 ratio grid, §17 scorer ablation, retrieval modes,
  method×model heatmap, dataset comparison, CIs, stats forest, and H5 (prefix/reuse/prefetch).

### 1.4 Bugs found and fixed (v2)

1. **Broken venv shebang.** `.venv/bin/pip` pointed at a stale venv path (created under
   `experiments/decaf-pilot/`), so `pip install` wrote to the wrong site-packages. Fixed by using
   `.venv/bin/python -m pip`; consider recreating the venv.
2. **Break-even intercept/slope swap.** The prefill fit was unpacked `(slope, intercept)` while
   stored `(intercept, slope)` → `L* = inf`. Fixed → meaningful L\* (241–335 across the 4 configs).
3. **Crossover figures filtered by label, not name.** Both crossover figures came out empty (one
   collapsed to 45 px). Fixed.
4. **S_net ignored compressor cost.** E2E is generation-only, so gross speedup overstated the win;
   added proper `S_net = T_full / (T_compress + T_e2e)` (and don't charge the compressor to `full`).
5. **Dataset API.** `datasets` 5.1.0 removed `trust_remote_code`; script-based 2Wiki repos fail.
   Switched to parquet `framolfese/2WikiMultihopQA`.

### 1.5 What's next / remaining

> Full prioritized list (7 experiments remaining, 2 blocked) in [`remaining.md`](./remaining.md).

- [ ] 32K context points (needs a longer-context model — Qwen2.5 caps at 32,768, so a ~32K target truncates).
- [ ] Perception Compressor baseline (or state the compute barrier).
- [ ] Energy / RAPL-style telemetry (not on macOS) and achieved-bandwidth vs roofline.
- [ ] Citation-correctness grounding where gold citations exist.
- [x] LongBench multi-document QA (E9 breadth) — **all 4 configs, n=200**. Significantly hurts every config.
- [x] More queries — **all 4 configs now n=200**. Statistical power flipped the central claims (3B/Q4
  compression significantly helps, p≤0.02; 7B/Q4 significantly hurts, p=0.019).
- [ ] Paper-ready "Preliminary Results" section from `results_v2/`.

**Completed in the latest batch:**
- [x] **E2** exact compression-ratio grid (0.2 / 0.4 / 0.5 / 0.6 / 0.8).
- [x] **§17** scorer on/off ablation (cross-encoder vs BM25).
- [x] **§13** dense / hybrid retrieval top-k (BM25 vs dense vs hybrid).

---

## 2. How to run (v2)

```bash
cd decaf-pilot
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt   # once
.venv/bin/python -m pip install "sentence-transformers>=3.0" llmlingua psutil  # v2 extras
ollama pull qwen2.5:3b-instruct-q8_0 && ollama pull qwen2.5:7b-instruct-q4_K_M && ollama pull qwen2.5:7b-instruct-q8_0  # once

.venv/bin/python -m src.run_v2 --config config_v2.yaml --mode all   # Phase A (resumable)
.venv/bin/python -m src.h5_v2 --config config_v2.yaml               # H5 micro-study
.venv/bin/python -m src.analyze_v2 --config config_v2.yaml          # Phase B
bash scripts/run_v2.sh                                              # full v2 end-to-end
```

Outputs in `results_v2/`: `FULL_RESULTS_v2.md`, 37 figures, `summary_v2.json`.

---

## 3. Phase 0 (pilot) — historical, superseded by §1

> Preserved verbatim. The v2 results above supersede these single-model numbers; the *shapes*
> (prefill-dominated, grounding cost) hold, but the "compression is always a net win" conclusion
> was an artifact of the ~1.4 ms BM25 scorer and does not survive the real cross-encoder.

**Date:** 2026-10-07
**Scope:** Phase 0 — smoke test / preliminary-results pilot.
**State:** ✅ Complete and runnable end-to-end.

Location: `experiments/decaf-pilot/` (historical path)

### 3.1 Summary

DECAF's thesis is that compression's benefit on CPU must be **measured, not assumed**: the
compressor costs time, decode cost depends on context length, and the question is whether
compression is a *net* win. Its backbone is the empirical `TPOT(L)` curve and the break-even
context length `L*`.

This pilot builds that harness on the project's own CPU/llama.cpp setup and runs the core
experiments E1/E8 (scaling), E2/E3/E4 (compression, overhead, break-even), E7 (grounding) and the
§17 component ablation.

**Bottom line:** on this machine the proposal's central CPU premise is **reversed** — prefill, not
decode, dominates end-to-end latency for any realistic context, because TPOT is nearly flat in `L`.
Compression is nevertheless a clear net latency win (≈4×) above a trivial threshold, but that win
is in **prefill/TTFT, not decode**, and it costs grounding. The CPU-aware selector does not beat a
plain fixed-ratio ranking on quality.

### 3.2 What was done

#### Codebase (under `experiments/decaf-pilot/`)

| File | Purpose |
|---|---|
| `src/contexts.py` | sentence segmentation, token estimation, filler corpus for controlled length |
| `src/select.py` | BM25 relevance, coverage-aware DECAF selector, baselines, `T_compress` |
| `src/run_experiments.py` | **Phase A** — scaling sweep (warmup + prefix-cache nonce) and compression; resumable |
| `src/analyze.py` | **Phase B** — `TPOT(L)`/`prefill(L)` fits, `L*`, net speedup, Pareto, grounding, ablation |
| `src/llm.py` | streaming Ollama client (TTFT) with explicit `num_ctx` |
| `src/data.py`, `src/retriever.py`, `src/metrics.py` | HotpotQA loader (gold supporting facts), BM25, EM/F1 |

#### Runs completed

220 runs (10 scaling + 210 compression), ~6 min CPU. Two measurement bugs found and fixed.

### 3.3 Results

Outputs in `experiments/decaf-pilot/results/`.

#### 3.3.1 Decode scaling (RQ1 / E1, E8) — `fig_scaling.png`

| L (tokens) | prefill ms | TPOT ms/tok | decode share |
|---|---|---|---|
| 362 | 1091 | 36.8 | 0.52 |
| 645 | 2124 | 37.9 | 0.36 |
| 1179 | 4322 | 37.0 | 0.22 |
| 2241 | 8447 | 36.9 | 0.12 |
| 4356 | 17507 | 40.2 | 0.07 |

- `prefill ≈ 4.1 ms/token` (r²=1.00, mildly super-linear); `TPOT ≈ 36.5 + 0.7 ms per 1k tokens` (r²=0.68).
- **Decode/prefill crossover L ≈ 414 tokens** (32-token outputs); **prefill dominates above it.**
- **H1 refuted**: the decode share *falls* with L, and compression's gain is in TTFT, not TPOT.

#### 3.3.2 Break-even L* (RQ2 / E3, E4) — `fig_break_even.png`
- `T_compress = 1.36 ms`, retained `L_c ≈ 205 tokens` → **`L* ≈ 205 tokens`**.
- **H2 supported but trivial on CPU**: because overhead ≪ per-token prefill cost, compression is a
  net latency win for *any* context above the retained budget.

#### 3.3.3 Compression: quality, grounding, latency (RQ3 / E2, E7) — `fig_pareto_grounding.png`

| method | F1 | ev. recall | gold-in-ev | retained | T_compress ms | E2E ms | net speedup |
|---|---|---|---|---|---|---|---|
| Full context | 0.221 | 0.751 | 0.833 | 1.00 | 0.84 | 3713 | 1.00× |
| Fixed 50% | 0.285 | 0.668 | 0.767 | 0.572 | 1.09 | 949 | 3.91× |
| Fixed 20% | 0.265 | 0.562 | 0.600 | 0.249 | 1.03 | 912 | 4.07× |
| Relevance top-m (iso) | 0.240 | 0.573 | 0.567 | 0.261 | 1.12 | 1172 | 3.17× |
| DECAF λ=6 | 0.268 | 0.532 | 0.600 | 0.190 | 1.36 | 988 | 3.76× |
| DECAF λ=15 | 0.122 | 0.279 | 0.200 | 0.059 | 1.38 | 1147 | 3.24× |
| DECAF no-cov (λ=6) | 0.302 | 0.671 | 0.733 | 0.470 | 1.00 | 1504 | 2.47× |

- Compression cuts latency **~4×** and, at moderate ratios, **raises F1** (0.221 → 0.285): distractor
  context hurts the small model.
- **Grounding degrades monotonically** with retained evidence (0.833 → 0.60 → 0.20) — accuracy alone
  hides this.

#### 3.3.4 Component analysis (§17) — `fig_ablation.png`
- Iso-budget relevance vs DECAF(λ6): F1 0.240 → 0.268, gold-in-evidence 0.567 → 0.600 (small gain),
  evidence recall 0.573 → 0.532 (small loss).
- **H3 not supported**: DECAF does not beat fixed-ratio selection (fixed-50% 0.285; fixed-20% 0.265).

### 3.4 Bugs found and honest interpretation

Two measurement bugs were caught and fixed — both matter for a latency study:

1. **Ollama context cap.** With no `num_ctx` set, Ollama truncated prompts (~2048 tokens): a
   4096-token target reported only 2050 tokens. Fixed by setting `num_ctx: 8192`; the token counts
   are now monotonic (345 → 4337).
2. **Cross-run prompt-prefix cache.** Longer prompts sharing a prefix with earlier runs re-used the
   KV prefix, making prefill appear near-free (an impossible 131 ms for 1160 tokens). Fixed with a
   unique per-run leading nonce; the prefill curve is now clean and monotonic.

**Interpretation.** The proposal's CPU hypothesis (decode-dominated; H1) is not supported on this
3B/Q4/CPU setup with ~32-token outputs — prefill dominates above ~400 tokens. The strongest, most
useful result is the honest reframing: **compression always pays off on CPU latency (overhead is
negligible vs prefill cost), and the real decision is how much grounding you are willing to trade**.
DECAF's CPU-aware term does not beat plain relevance ranking here.

**Limits:** one model/quantization, 30 questions, 32-token generations, BM25-based scorer (not the
proposal's cross-encoder), synthetic filler for the scaling sweep.

### 3.5 Phase-0 remaining list (now addressed or re-scoped in §1.5)

- [x] Q4_K_M vs Q8_0 and 3B vs 7B pairs → done in v2 (E5/E6).
- [x] Generation-length sweep (N_out ∈ {16,64,128}) → done in v2 (crossover).
- [x] Cross-encoder scorer (`bge-reranker-base`) → done in v2.
- [x] Break-even by quantization/model table → done in v2 (§1.2).
- [x] Long-context points (8K, 16K) → done in v2 (16K).
- [x] 2WikiMultihopQA generalization → done in v2 (E9).
- [x] LLMLingua-2 baseline → done in v2.
- [x] §17 ablation (coverage/reorder/λ) → done in v2.
- [x] H5 prefix/KV reuse + prefetch → done in v2.
- [x] Bootstrap CIs / McNemar / Holm → done in v2.
- [ ] 32K points · Perception Compressor · energy/bandwidth telemetry · citation correctness ·
  LongBench/TriviaQA · larger n · paper-ready write-up — see §1.5.

### 3.6 How to run (Phase 0)

```bash
cd experiments/decaf-pilot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
ollama pull qwen2.5:3b-instruct                                       # once

.venv/bin/python -m src.run_experiments --mode scaling --lengths 256,512   # quick
.venv/bin/python -m src.analyze
bash scripts/run_pilot.sh                                                  # full pilot
```

Outputs in `results/`: `PRELIMINARY_RESULTS.md`, 4 figures, `summary.json`.
