# DECAF — Methods (paper-ready draft)

*When does retrieved-context compression pay off on CPU? A decode/prefill characterization with
break-even thresholds.*

This section describes the measurement framework, the system under study, the compressor and
baselines, and the evaluation protocol. All components are released; every measurement is cached to an
append-only JSONL log so results are reproducible.

---

## 3.1 System overview and harness

We study a single-turn, evidence-grounded RAG pipeline on CPU:

```
query → retrieval → sentence-level evidence selection → reader (CPU llama.cpp) → answer
                     ▲ T_retrieve        ▲ T_compress          ▲ TTFT, TPOT, E2E, RSS
```

The harness is deliberately **staged** so that each cost is attributable: retrieval (`T_retrieve`),
compression (`T_compress`), prefill/TTFT, and per-token decode (TPOT) are measured separately. The
reader is served by **Ollama / llama.cpp** with a fixed context window (`num_ctx = 20,480`),
`temperature = 0`, `seed = 0`, and a fixed generation budget (`num_predict = 32` for the main
experiments; swept separately). TTFT is measured as wall-clock time to the first non-empty streamed
chunk; prefill, decode and token counts come from llama.cpp telemetry in the final chunk.

**Hardware.** One CPU host — 8 physical cores, 16 GB RAM, macOS. Absolute latencies are
machine-specific; the shapes and break-even behaviour are the contribution.

## 3.2 Datasets

Two multi-hop QA benchmarks with gold supporting facts, in the **provided-context (distractor)**
setting so retrieval depth and distractor harm are controlled:

- **HotpotQA** (distractor) — 10 candidate paragraphs/question (2 gold + 8 distractors).
- **2WikiMultihopQA** — same structure, harder, 4 gold paragraphs/question (two hops).

We sample **n = 200** questions per dataset with a fixed seed (deterministic shuffle), and hold the
prompt template, retrieval depth and metrics identical across all configurations.

## 3.3 Retrieval and candidate construction

Each question is answered over the **top-5** paragraphs (default BM25). We additionally evaluate two
retrieval modes to test sensitivity to the retrieved set:

- **BM25** (`rank_bm25` Okapi) — the default.
- **Dense** — cosine similarity with `BAAI/bge-small-en-v1.5` sentence embeddings (CPU).
- **Hybrid** — **reciprocal-rank fusion** of BM25 and dense rankings.

Top-5 paragraphs are split into sentences with a punctuation regex; the sentence list is the unit of
evidence selection.

## 3.4 Evidence scoring and the DECAF selector

**Scoring (Sec. 12.1 of the proposal).** Each candidate sentence is scored with a CPU cross-encoder,
`BAAI/bge-reranker-base` (~110M). Its forward-pass time **is** the measured compressor cost
`T_compress`. Relevance is the sigmoid of the cross-encoder logit, min-max normalised to [0, 1].

**Selector (Sec. 12.2).** DECAF is a **single-pass, decoder-free** greedy gate. Sentences are visited
in decreasing relevance order and accepted while a marginal benefit exceeds a marginal cost:

```
gain(s) = Relevance(s) · (0.5 + 0.5 · Coverage(s))          # coverage = new query-term coverage
cost(s) = tokens(s) / Σ tokens                              # CPU decode cost ∝ tokens
accept s  iff   gain(s) ≥ λ · cost(s)                       # single interpretable knob λ
```

Retained sentences are re-ordered by relevance before prompting (a lost-in-the-middle mitigation).
`λ` is calibrated on the held-out split (we report λ ∈ {2, 6, 15}); the retained count at λ = 6 is
also the budget for the iso-budget relevance baseline.

## 3.5 Baselines and ablations

| Family | Methods |
|---|---|
| Full context | `full` (no compression) |
| Fixed ratio | `fixed_{0.8,0.6,0.5,0.4,0.2}` — top-r by relevance |
| Iso-budget relevance | `rel_iso` — top-m by relevance at DECAF's retained count |
| DECAF variants | `decaf_lam{2,6,15}`, coverage-off, reordering-off |
| Scorer ablation | `bm25_decaf` — the same gate driven by BM25 instead of the cross-encoder |
| External baseline | **LLMLingua-2** (training-free token-level prompt compression) |

## 3.6 Generation protocol

For each (config, dataset, question, method) we build the reader prompt from the retained text and
generate a short answer, then score it. The scaling study uses **controlled-length synthetic
contexts** (256 → 16,384 target tokens), with a warmup run and a **unique per-run leading nonce** that
defeats llama.cpp's cross-run prefix/KV cache (otherwise prefill appears near-free). The
**generation-length crossover** repeats this at `N_out ∈ {16, 64, 128, 256, 512}`.

## 3.7 Measurement framework (Sec. 5 of the proposal)

Latency decomposes as

```
T_total = T_retrieve + T_compress + T_prefill(L) + N_out · TPOT(L)
```

We report the fitted `TPOT(L)` and `T_prefill(L)` curves and the derived quantities:

| Quantity | Definition |
|---|---|
| Compression efficiency | `CE = ΔT_decode / ΔL` |
| Compression overhead | `O_c = T_compress / T_baseline` |
| Net speedup | `S_net = T_baseline / (T_compress + T_prefill^c + T_decode^c)` |
| Break-even length | `L* = L_c + T_compress / (b_prefill + N_out · b_TPOT)` |

System telemetry: peak RSS (sampled via `psutil`), tokens/s, and per-stage timings. Energy / achieved
memory bandwidth are not exposed on the host OS and are treated as out of scope (noted in Limitations).

## 3.8 Grounding evaluation

Grounding is scored **decoder-free**, against dataset-provided gold supporting facts:

- **Evidence recall / precision** — overlap of retained sentences with the gold supporting sentences.
- **Gold-in-evidence** — whether the gold answer string survives in the retained context.
- **Answer support** — whether the generated answer is contained in the retained context.

This lets us separate *correctness* (EM/F1) from *citation grounding*, which prior work shows can
diverge sharply under compression.

## 3.9 Statistical analysis plan

Per-query pairing is preserved throughout. For each (config, dataset) we report, versus full context:

- **Accuracy (EM):** **exact McNemar's test** on the paired correct/incorrect vector.
- **Accuracy/grounding (continuous, e.g. F1):** **paired bootstrap** confidence intervals (2,000
  resamples) on the per-query difference.
- **Multiplicity:** **Holm–Bonferroni** across the method family within each (config, dataset).
- Absolute and relative effects are reported together; the **accuracy–latency Pareto frontier**, not a
  single scalar, is the primary summary.

## 3.10 Reproducibility

Exact model checkpoints (Qwen2.5-Instruct, GGUF Q4_K_M / Q8_0 via Ollama), scorer
(`bge-reranker-base`) and dense model (`bge-small-en-v1.5`) are pinned by name; seeds are fixed
(`seed = 42`); prompts, retrieval depth and metrics are identical across configurations. Every run is
written to a resumable append-only cache, and the analysis regenerates all 41 figures and tables from
those caches (`config_v2.yaml`, `src/{run_v2,retrieval_v2,h5_v2,analyze_v2}.py`,
`results_v2/summary_v2.json`).
