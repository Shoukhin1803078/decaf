# DECAF — CPU RAG experiments (v2)

A CPU-only harness for the proposal *"When Does Context Compression Accelerate CPU
RAG? A Decode-Centric Study with Adaptive Evidence Selection"*. It measures the
proposal's empirical backbone directly on this machine (Ollama / llama.cpp, CPU):
the `TPOT(L)`/`prefill(L)` curves, the **break-even context length `L*`**, the
quality–latency–grounding Pareto, and a training-free CPU-aware evidence selector.

> **Status:** ✅ v2 complete — **23,772 runs** across 4 model×quant configs and **4 datasets**
> (HotpotQA, 2Wiki, TriviaQA, LongBench), all at **n=200**; **43 figures** + report in `results_v2/`.
> The Phase-0 pilot is preserved in `results/` (historical). See `STATUS.md` for full progress.

## At a glance

| | |
|---|---|
| Models | `qwen2.5:3b-instruct` (Q4_K_M), `qwen2.5:3b-instruct-q8_0`, `qwen2.5:7b-instruct-q4_K_M`, `qwen2.5:7b-instruct-q8_0` |
| Datasets | HotpotQA + 2WikiMultihopQA (multi-hop) · TriviaQA (single-hop) · LongBench hotpotqa (long-context) |
| Scorer | cross-encoder `BAAI/bge-reranker-base` (CPU) |
| Compressors / baselines | DECAF (λ sweep), fixed-ratio, relevance top-m, **LLMLingua-2**, BM25-scorer variant |
| Retrieval | BM25 · dense (`BAAI/bge-small-en-v1.5`) · hybrid (RRF) |
| Runs | scaling 56 · scaling-compressed 48 · e8 24 · crossover 60 · compression 21,360 · retrieval 2,160 · H5 64 = **23,772** |
| Outputs | `results_v2/` — 43 figures, `FULL_RESULTS_v2.md`, `summary_v2.json` |

## Headline results

- **H1 refuted (all four configs):** prefill grows **5.3–10.9 ms/token** while TPOT is
  nearly flat (**+0.8–1.3 ms per 1k tokens**). Decode and prefill cross only at
  **L ≈ 584–916 tokens**; above that **prefill dominates** — the opposite of the
  decode-dominated CPU premise.
- **Break-even `L*` (with the real cross-encoder cost ≈ 960 ms):** **327 / 335 / 241 / 244**
  tokens for 3B-Q4 / 3B-Q8 / 7B-Q4 / 7B-Q8. Model **size** sets `L*`; **quantization barely
  moves it**.
- **Overhead matters:** gross speedup ≈ halved once the compressor is counted — e.g. 3B/Q4
  HotpotQA DECAF λ6 goes **3.66× → S_net 1.89×**.
- **E2 ratio grid is non-monotone:** the 3B peaks at ~50% retention (F1 0.330 vs 0.199 full),
  the 7B is ~flat from 60% up; grounding rises monotonically with retention, and at **80%
  retention S_net ≈ 1.0×** (compression barely breaks even).
- **§17 scorer ablation:** BM25 ≈ **1.9 ms** vs the cross-encoder ≈ **989 ms**, yet the CE buys
  only small gains (7B/Q8 F1 0.556 vs 0.539) → the CE rarely pays for itself on CPU.
- **§13 retrieval modes:** **hybrid (BM25 + dense RRF) wins on both quality and grounding** —
  7B/Q4 HotpotQA F1 0.522 → **0.589**, gold-in-ev 0.633 → **0.733**.
- **H5 (secondary):** llama.cpp prefix/KV-cache reuse cuts TTFT by **~99%** on an identical
  prompt — a *large* end-to-end win precisely because prefill dominates.
- **Statistical power (all configs at n=200):** this flips the central claims — on **3B/Q4 compression
  significantly helps** (DECAF λ6 ΔF1 +0.077, p=0.009; fixed-50% +0.073, p=0.005), while on **7B/Q4 it
  significantly hurts** (DECAF λ6 −0.092, p=0.019). A **significant model-size interaction** that n=30
  could not resolve.
- **Generation length:** extending the N_out sweep to 512 shows the decode share **plateaus** rather than
  rising, because the model hits **EOS at ~40–64 tokens** — so "prefill dominates" holds for *natural*
  answer lengths (testing longer needs EOS-disabled decoding).
- **Dataset dependence (E9):** on single-hop **TriviaQA**, compression **helps the 3B** (3B/Q4 DECAF λ6
  ΔF1 +0.081, Holm p=0.002) but is **null for the 7B** (≈0, n.s.) — so it doesn't *hurt* the 7B there.
  The "compression hurts the 7B" effect is therefore **multi-hop-specific**, not universal.
- **Task dependence — long context (E9):** on **LongBench** (long multi-document context) compression
  **significantly hurts every config** (ΔF1 −0.14…−0.17, all p<0.001) — the evidence is spread across
  the long context and aggressive selection drops it. So compression's value is **strongly
  task/context-dependent** (helps: HotpotQA-3B, TriviaQA-3B · neutral: 2Wiki · hurts: HotpotQA-7B, LongBench).
- **LLMLingua-2** costs more to run than the cross-encoder and its S_net trails DECAF's.

Absolute latencies are machine-specific; the **shapes** (prefill-dominated, grounding cost,
overhead-vs-savings) are the point. 200 questions per dataset.

## Design

- **Scaling sweep:** controlled-length contexts (256 → 16384 target tokens) with a warmup and a
  unique per-run prefix nonce (to defeat the cross-run prompt-prefix cache).
- **Compression:** top-5 paragraphs, sentence-level evidence selection. Methods: `full`,
  `fixed_{0.8,0.6,0.5,0.4,0.2}`, `rel_iso` (iso-budget relevance), `decaf_lam{2,6,15}`,
  `decaf_nocov_lam6`, `decaf_noreorder_lam6`, `bm25_decaf_lam6`, `llmlingua_{0.5,0.2}`.
- **DECAF selector:** relevance × coverage, accepted while `gain ≥ λ · cost` with
  `cost ∝ sentence tokens`. Single-pass, decoder-free; its own cost is measured.
- **Grounding:** evidence recall/precision vs gold supporting facts, plus gold-in-evidence.
- **`L*`:** solved from fitted `prefill(L)`/`TPOT(L)` plus measured compressor overhead.
- **Statistics:** paired bootstrap CIs, McNemar (EM), Holm correction; Pareto frontier as the story.

## Setup / run

```bash
cd decaf-pilot
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt   # once
.venv/bin/python -m pip install "sentence-transformers>=3.0" llmlingua psutil  # v2 extras

ollama pull qwen2.5:3b-instruct-q8_0
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q8_0

.venv/bin/python -m src.run_v2      --config config_v2.yaml --mode all   # Phase A (resumable)
.venv/bin/python -m src.retrieval_v2 --config config_v2.yaml             # §13 retrieval modes
.venv/bin/python -m src.h5_v2       --config config_v2.yaml              # H5 KV-reuse study
.venv/bin/python -m src.analyze_v2  --config config_v2.yaml              # Phase B → results_v2/
```

> Note: `pip` in this venv may have a stale shebang (the venv was created elsewhere);
> use `.venv/bin/python -m pip` as above.

Outputs in `results_v2/`: `FULL_RESULTS_v2.md` (9 sections), 41 figures, `summary_v2.json`.

## Phase 0 (pilot, historical)

The original single-model smoke test (Qwen2.5-3B Q4, BM25 scorer, 220 runs) lives in
`src/run_experiments.py` + `src/analyze.py`, with outputs in `results/`
(`PRELIMINARY_RESULTS.md`, 4 figures). Its *shapes* hold, but its "compression is always a
net win" conclusion was an artifact of the ~1.4 ms BM25 scorer and does not survive the real
cross-encoder.

## Remaining

32K context points (blocked — Qwen2.5 caps at 32,768), Perception Compressor baseline,
energy / achieved-bandwidth telemetry (RAPL not exposed on macOS), citation-correctness
grounding, LongBench / TriviaQA breadth, and more queries (>30) for statistical power.
