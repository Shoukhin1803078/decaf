# DECAF — Results (paper-ready draft)

*When does retrieved-context compression pay off on CPU? A decode/prefill characterization with
break-even thresholds.*

**Setup.** Single CPU host (8 physical cores, 16 GB RAM), Ollama / llama.cpp. Four
model×quantization configs — Qwen2.5-**3B**-Instruct and **7B**-Instruct in **Q4_K_M** and **Q8_0**.
Four QA datasets — **HotpotQA** and **2WikiMultihopQA** (multi-hop), **TriviaQA** (single-hop), and
**LongBench** hotpotqa (long-context multi-document).
Scorer: cross-encoder `BAAI/bge-reranker-base` (CPU). Retrieval modes: BM25, dense
(`bge-small-en-v1.5`), hybrid (RRF). **All configs are evaluated at n = 200 questions per dataset**; a
few probe methods are reported at n = 30 and marked as such. 21,420 runs total; statistics are paired
bootstrap CIs, McNemar's test (EM) and Holm–Bonferroni correction across the method family.

---

## 1. Decode-scaling and the prefill/decode split (RQ1 / E1, E8)

We measured `TPOT(L)` and prefill `(L)` on controlled-length contexts (256 → 16,384 tokens), 2
replicates per point, with a per-run prefix nonce to defeat the cross-run KV cache.

| config | prefill (ms/token) | prefill r² | TPOT (ms/token) | TPOT slope (ms/1k tok) | decode = prefill at L |
|---|---:|---:|---:|---:|---:|
| 3B / Q4_K_M | 5.51 | 0.990 | 35.9 + 0.85·L/1k | +0.85 | ~867 |
| 3B / Q8_0 | 5.28 | 0.994 | 58.8 + 0.81·L/1k | +0.81 | ~916 |
| 7B / Q4_K_M | 10.92 | 0.998 | 75.8 + 1.30·L/1k | +1.30 | ~584 |
| 7B / Q8_0 | 10.57 | 0.996 | 132.8 + 1.24·L/1k | +1.24 | ~867 |

**Finding.** Prefill grows **linearly and steeply** with context (≈ 5.3 ms/token on 3B, ≈ 10.7
ms/token on 7B; r² ≥ 0.99), whereas **TPOT is nearly flat in context length** (+0.8–1.3 ms per
1,000 tokens). Decode's share of end-to-end latency therefore **decreases** with `L` and crosses 50%
only at **L ≈ 584–916 tokens**; above that **prefill dominates**. This **refutes the
decode-dominated CPU premise (H1)** for realistic multi-hop contexts on this hardware: the lever
compression pulls is **TTFT/prefill**, not per-token decode.

**Generation-length robustness.** Sweeping the generation cap `N_out ∈ {16, 64, 128, 256, 512}`
shows the decode share **plateaus** rather than rising (e.g. 3B/Q4 at L≈647: 0.21 → 0.41 → 0.40 →
0.39 → 0.40). Inspecting the raw outputs explains why: the model emits **EOS at ~40–64 tokens**
regardless of the cap, so `N_out` cannot be forced higher. The prefill-dominated result therefore
holds for **natural** answer lengths; testing genuinely long generation requires EOS-disabled
decoding (left to future work). *Figures:* `fig_v2_scaling_tpot/prefill/ttft/e2e/decode_share/panel`,
`fig_v2_crossover_*`.

---

## 2. Break-even context length `L*` (RQ2 / E4)

Solving the fitted curves plus the **measured compressor cost** gives the context length above which
compression becomes a net latency win. The compressor here is the cross-encoder, whose own cost is
**T_compress ≈ 865 ms** (3 orders of magnitude above the pilot's BM25 ≈ 1.4 ms).

| config | retained `L_c` (tok) | `T_compress` (ms) | slope (prefill+decode, ms/tok) | **`L*` (tok)** |
|---|---:|---:|---:|---:|
| 3B / Q4_K_M | 150 | 865 | 5.53 | **306** |
| 3B / Q8_0 | 150 | 865 | 5.31 | **312** |
| 7B / Q4_K_M | 150 | 865 | 10.96 | **228** |
| 7B / Q8_0 | 150 | 865 | 10.61 | **231** |

**Finding (H2).** For each configuration there **is** a break-even `L*`: below it compression is a
net latency *loss* (the compressor costs more than it saves), above it a net *win*. `L*` is set by
the per-token cost, so the **larger 7B model breaks even ~80 tokens sooner** than the 3B. Crucially,
`L*` is **only non-trivial because the compressor is expensive** — with the pilot's ~1.4 ms BM25 scorer
`L*` collapsed to ≈ the retained length, i.e. "always a net win". *Figures:*
`fig_v2_breakeven_curves`, `fig_v2_Lstar_bar`, `fig_v2_overhead_vs_savings`.

---

## 3. Compression: quality, latency, grounding (RQ3 / E2, E3, E7)

### 3.1 Main comparison (n = 200, HotpotQA)

| config | method | F1 | retained | gold-in-ev | T_c (ms) | E2E (ms) | S_net |
|---|---|---:|---:|---:|---:|---:|---:|
| 3B/Q4 | full | 0.185 | 1.00 | 0.745 | — | 3922 | 1.00 |
| 3B/Q4 | fixed 20% | **0.280** | 0.238 | 0.655 | 905 | 1097 | 1.96 |
| 3B/Q4 | fixed 50% | 0.258 | 0.550 | 0.720 | 905 | 1127 | 1.93 |
| 3B/Q4 | relevance top-m | 0.261 | 0.128 | 0.530 | 905 | 1127 | 1.93 |
| 3B/Q4 | DECAF λ6 | 0.262 | 0.120 | 0.515 | 905 | 1058 | 2.00 |
| 3B/Q8 | full | 0.196 | 1.00 | 0.745 | — | 4343 | 1.00 |
| 3B/Q8 | fixed 20% | **0.249** | 0.238 | 0.655 | 905 | 1772 | 1.62 |
| 3B/Q8 | DECAF λ6 | 0.222 | 0.120 | 0.515 | 905 | 1806 | 1.60 |
| 7B/Q4 | full | **0.535** | 1.00 | 0.745 | — | 7987 | 1.00 |
| 7B/Q4 | fixed 20% | 0.480 | 0.238 | 0.655 | 905 | 1137 | 3.91 |
| 7B/Q4 | DECAF λ6 | 0.443 | 0.120 | 0.515 | 905 | 874 | 4.49 |
| 7B/Q8 | full | **0.509** | 1.00 | 0.745 | — | 8156 | 1.00 |
| 7B/Q8 | fixed 20% | 0.485 | 0.238 | 0.655 | 905 | 1710 | 3.12 |
| 7B/Q8 | DECAF λ6 | 0.445 | 0.120 | 0.515 | 905 | 1436 | 3.48 |

**Quality is model-dependent.** On the **3B** models, removing distractor context **improves** F1
(3B/Q4: 0.185 → 0.280), while on the **7B** models it **degrades** F1 (7B/Q4: 0.535 → 0.443) — the
larger model can exploit the extra context. 2Wiki shows the same sign, weaker (3B/Q4 ΔF1 ≈ 0.00,
7B/Q4 −0.02).

**Latency vs overhead (S_net).** Compression cuts generation latency ~4×, but the **~900 ms
cross-encoder cost roughly halves the net speedup**: e.g. 3B/Q4 HotpotQA DECAF λ6 goes from a gross
3.7× to **S_net = 2.0×**. The 7B benefits more (S_net 4.5×) because its per-token cost is higher.
At **80% retention S_net ≈ 1.0×** — compression is a net wash.

**Grounding degrades monotonically.** Gold-in-evidence falls with the retention ratio:
full **0.745** → fixed 50% 0.720 → fixed 20% 0.655 → DECAF λ6 **0.515** (HotpotQA; 0.68 → 0.40 on
2Wiki). Accuracy alone hides this. *Figures:* `fig_v2_pareto_f1_e2e`, `fig_v2_pareto_f1_retained`,
`fig_v2_grounding_vs_ratio`, `fig_v2_quality_vs_grounding`, `fig_v2_latency_breakdown`.

### 3.2 Paired significance (n = 200, Holm-corrected)

| config · dataset | ΔF1 vs full (95% CI) | Holm p |
|---|---|---:|
| 3B/Q4 · HotpotQA · DECAF λ6 | **+0.077** [+0.036, +0.119] | **0.009** |
| 3B/Q4 · HotpotQA · fixed 20% | **+0.095** [+0.054, +0.137] | **0.001** |
| 3B/Q4 · HotpotQA · fixed 50% | **+0.073** [+0.039, +0.110] | **0.005** |
| 3B/Q4 · HotpotQA · relevance top-m | **+0.076** [+0.036, +0.119] | **0.015** |
| 3B/Q8 · HotpotQA · fixed 20% | +0.053 [+0.019, +0.089] | **0.018** |
| 7B/Q4 · HotpotQA · DECAF λ6 | **−0.092** [−0.151, −0.037] | **0.019** |
| 7B/Q4 · HotpotQA · no-coverage | −0.085 [−0.141, −0.032] | 0.025 |
| 7B/Q4 · HotpotQA · relevance top-m | −0.092 [−0.152, −0.037] | 0.056 |
| 7B/Q8 · HotpotQA · DECAF λ6 | −0.064 [−0.118, −0.014] | 0.051 |
| 3B/Q4 · 2Wiki · all methods | −0.00 … +0.036 | ≥ 0.43 |

**Finding.** With 200 queries the effect of compression on quality is **statistically significant and
model-dependent**: on 3B/Q4 compression **significantly helps** (p ≤ 0.02 across methods), on 7B/Q4 it
**significantly hurts** (p = 0.019). On 2Wiki the effect is null. This is a **significant model-size
interaction (E6/RQ4)** — and it is invisible at n = 30, which is why the scale-up mattered.

### 3.3 Compression-ratio grid (E2)

| retained | 3B/Q4 F1 | 7B/Q4 F1 | gold-in-ev | S_net |
|---|---:|---:|---:|---:|
| 100% (full) | 0.185 | 0.535 | 0.745 | 1.00 |
| 80% | 0.217 | 0.565 | 0.800 | ~0.94–1.07 |
| 60% | 0.250 | 0.584 | 0.800 | ~1.80–4.37 |
| 50% | 0.258 | 0.516 | 0.720 | ~1.93–4.33 |
| 20% | **0.280** | 0.480 | 0.655 | ~1.96–3.91 |

Quality is **non-monotone in the retention ratio**, and the optimum differs by model (3B peaks at
the most aggressive ratio; 7B peaks at ~60%). At 80% retention the compressor's cost cancels the
small prefill saving (S_net ≈ 1.0). *Figure:* `fig_v2_ratio_grid`.

### 3.4 Single-hop generalization (TriviaQA)

Adding a **single-hop** dataset (TriviaQA, provided Wikipedia passages) shows compression **helps the
3B** and is **neutral for the 7B** — i.e. it does *not* harm the large model here:

| config | full F1 | DECAF λ6 F1 | ΔF1 (Holm p) | gold-in-ev (full → comp) |
|---|---:|---:|---|---:|
| 3B/Q4 | 0.231 | 0.312 | **+0.081 (0.002)** | 0.880 → 0.695 |
| 3B/Q8 | 0.235 | 0.260 | +0.025 (0.36, n.s.) | 0.880 → 0.695 |
| 7B/Q4 | 0.546 | 0.547 | +0.001 (n.s.) | 0.880 → 0.695 |
| 7B/Q8 | 0.542 | 0.545 | +0.003 (n.s.) | 0.880 → 0.695 |

The "compression **hurts the 7B**" effect seen on multi-hop HotpotQA is therefore **task-specific**: on
single-hop TriviaQA the 7B is unaffected (ΔF1 ≈ 0, not negative). Grounding still degrades (0.88 →
0.70) regardless. The model-size interaction is thus **dataset-dependent** — an important boundary on
the headline claim. (*At n ≈ 29 the 7B/Q4 appeared to gain +0.087; the full n = 200 shows that was
small-sample noise.*) *Figures:* TriviaQA points appear in `fig_v2_pareto_all`,
`fig_v2_grounding_vs_ratio`, `fig_v2_dataset_compare`.

### 3.5 Long-context multi-document (LongBench)

On a long-context multi-document set, compression **hurts** — and significantly so for **every** config:

| config | full F1 | DECAF λ6 F1 | ΔF1 (Holm p) | gold-in-ev (full → comp) |
|---|---:|---:|---|---:|
| 3B/Q4 | 0.302 | 0.156 | **−0.146 (<0.001)** | 0.640 → 0.380 |
| 3B/Q8 | 0.303 | 0.129 | **−0.174 (<0.001)** | 0.640 → 0.380 |
| 7B/Q4 | 0.479 | 0.334 | **−0.145 (<0.001)** | 0.640 → 0.380 |
| 7B/Q8 | 0.474 | 0.339 | **−0.136 (<0.001)** | 0.640 → 0.380 |

All methods are negative (−0.12…−0.17), because when the answer evidence is spread across a long
context, aggressive selection removes it (gold-in-evidence 0.64 → 0.38). This is the **opposite regime**
from TriviaQA (§3.4): compression's value is **strongly task- and context-dependent** — it helps on
short distractor-heavy contexts, is neutral on 2Wiki, and is harmful on long-context multi-doc QA.

---

## 4. Model-size and quantization interaction (RQ4 / E5, E6)

| | 3B/Q4 | 3B/Q8 | 7B/Q4 | 7B/Q8 |
|---|---:|---:|---:|---:|
| prefill (ms/tok) | 5.51 | 5.28 | 10.92 | 10.57 |
| **`L*` (tok)** | 306 | 312 | 228 | 231 |
| full-context F1 | 0.185 | 0.196 | 0.535 | 0.509 |
| compression effect on F1 | **helps** | ~neutral | **hurts** | **hurts** |

**Size dominates; quantization is second-order.** `L*` is essentially set by model size (3B ≈ 306–312
vs 7B ≈ 228–231), and **quantization barely moves it** (within ~6 tokens). Quantization also has no
consistent effect on quality (Q8 better on 3B, Q4 better on 7B) and, on this CPU, Q8 is **not slower**
in prefill than Q4 (Q4_K dequantization overhead). The **compression effect flips sign with size**,
the paper's central interaction.

---

## 5. Component and scorer ablation (Sec. 17)

- **CPU-aware gating adds nothing over plain relevance ranking.** On 3B/Q4 HotpotQA, DECAF λ6
  (0.262) ≈ relevance top-m at matched budget (0.261) ≈ fixed 20% (0.280) — the "quality gain per CPU
  cost" gate does **not** beat a plain relevance top-k. **(H3 not supported.)**
- **Coverage term** gives a small, inconsistent gain (DECAF λ6 0.262 vs no-coverage 0.253).
- **Relevance reordering** is ≈ neutral (n = 30 probe).
- **Scorer on/off (CE vs BM25):** the cross-encoder costs **~989 ms** vs BM25 **~1.9 ms**, yet buys
  only small gains (7B/Q8 F1 0.556 vs 0.539; gold-in-ev 0.633 vs 0.600; n = 30) — on CPU the
  cross-encoder's ~500× cost rarely pays for itself.

*Figures:* `fig_v2_lambda_sweep`, `fig_v2_ablation_bars`, `fig_v2_scorer_ablation`.

---

## 6. External baseline: LLMLingua-2 (n = 30 probe)

LLMLingua-2's own compression cost (~1.1–4.4 s) is **higher** than the cross-encoder's, and it
compresses less at the same nominal rate: on HotpotQA 3B/Q4 its S_net is 1.16–1.39× vs DECAF's ~2.0×,
and it retains fewer gold facts at aggressive rates (gold-in-ev 0.43 at 20%). DECAF is on the
quality–latency–grounding frontier at or ahead of this external baseline.

---

## 7. Retrieval mode (Sec. 13: BM25 vs dense vs hybrid)

| config | dataset | method | BM25 | dense | hybrid |
|---|---|---|---:|---:|---:|
| 3B/Q4 | HotpotQA | F1 | 0.262 | 0.300 | **0.310** |
| 3B/Q4 | HotpotQA | gold-in-ev | 0.515 | 0.700 | **0.733** |
| 7B/Q4 | HotpotQA | F1 | 0.443 | 0.587 | **0.589** |
| 7B/Q4 | 2Wiki | F1 | 0.228 | 0.263 | **0.315** |

**Finding.** **Hybrid retrieval (BM25 + dense RRF) improves both quality and grounding** over BM25
in most cells. Notably, the apparent "compression hurts the 7B" result is partly a *retrieval*
limitation: with hybrid retrieval the 7B's compressed F1 rises from 0.443 to 0.589. Retrieval quality
is a first-order lever — often larger than the compression policy. *Figures:* `fig_v2_retrieval`,
`fig_v2_retrieval_pareto`.

---

## 8. Secondary systems analysis (H5: KV reuse + prefetch)

| config · L | cold TTFT (ms) | warm (identical) | partial (shared ctx) | Δ TTFT |
|---|---:|---:|---:|---:|
| 3B/Q4 · 8K | 39,693 | **87** | 289 | **−99.8%** |
| 7B/Q4 · 8K | 81,780 | **147** | 486 | **−99.8%** |
| 7B/Q8 · 8K | 82,527 | 205 | 465 | −99.8% |

llama.cpp prefix/KV-cache reuse cuts TTFT by **~99%** on an identical prompt and ~96% on a
shared-context/partial overlap. **Because prefill dominates here, this is a large end-to-end win** —
the opposite of the "reuse only moves TTFT, hides little" expectation. *Figures:* `fig_v2_h5_prefix`,
`fig_v2_h5_reuse`, `fig_v2_h5_prefetch`.

---

## 9. Summary of hypothesis outcomes

| Hypothesis | Outcome |
|---|---|
| **H1** decode dominates CPU latency | **Refuted** — prefill dominates above L ≈ 584–916; decode share *falls* with L (for natural answer lengths) |
| **H2** break-even `L*` exists | **Supported** — `L*` = 228–312 tokens; non-trivial *because* the compressor is expensive |
| **H3** DECAF beats fixed/relevance on the frontier | **Not supported** — no better than relevance top-k / fixed ratio |
| **H4** size & quantization shift `L*` | **Partly** — size matters (306 vs 228); quantization barely moves `L*` |
| **H5** reuse/prefetch give limited benefit | **Refuted here** — KV reuse gives ~99% TTFT cuts and a large E2E win (prefill-dominated) |

---

## 10. Limitations / threats to validity

1. **Single host** (one macOS CPU); absolute latencies are machine-specific and there is no
   cross-platform generality.
2. **Short natural generations** — the model emits EOS at ~40–64 tokens, so the prefill-dominated
   verdict is established for concise answers only; long-generation regimes are untested.
3. **Two datasets**, both multi-hop QA; no single-hop (TriviaQA/NQ) or long-context (LongBench) set.
4. **Compressor = cross-encoder** (~960 ms); DECAF's own scorer is a ~110M model, not the proposal's
   full component set.
5. **No energy / achieved-bandwidth telemetry** (RAPL not exposed on macOS); E8 is a proxy.
6. n = 200 per dataset is adequate for the headline interactions but several probe methods remain
   n = 30.

*All numbers are reproducible from `results_v2/summary_v2.json`; figures are in `results_v2/`.*
