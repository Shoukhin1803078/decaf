# DECAF

**Decode-Centric Adaptive evidence selection For CPU-only RAG.**

Research project for the proposal *"When Does Context Compression Accelerate CPU
RAG? A Decode-Centric Study with Adaptive Evidence Selection"* — a controlled study
of how retrieved-context length and training-free compression affect TTFT, per-token
decode latency (TPOT), end-to-end latency, memory and grounding for small quantized
LLMs running **entirely on CPU**.

## Repository layout

| Path | What it is |
|---|---|
| `decaf-pilot/DECAF_CPU_RAG_Proposal_v2.md` | the research proposal (24 sections) |
| `decaf-pilot/` | the CPU experiment harness (Ollama / llama.cpp) |
| `decaf-pilot/src/` | Phase-A experiment drivers + Phase-B analysis |
| `decaf-pilot/results_v2/` | **current** results — 41 figures + `FULL_RESULTS_v2.md` |
| `decaf-pilot/results/` | Phase-0 pilot results (historical) |
| `decaf-pilot/STATUS.md` | status & progress (v2-first; Phase 0 preserved) |

## Status

✅ **Full v2 experiments complete** — 23,700 runs: 3B/7B × Q4_K_M/Q8_0, HotpotQA + 2WikiMultihopQA +
TriviaQA + LongBench (**all at n=200**), cross-encoder scorer (`bge-reranker-base`), LLMLingua-2 baseline,
BM25/dense/hybrid retrieval (all 4 configs), scaling to 16K, and a KV-reuse/prefetch micro-study.

See `decaf-pilot/STATUS.md` for the full status and
`decaf-pilot/results_v2/FULL_RESULTS_v2.md` for the results.

**Headline:** on CPU, **prefill — not decode — dominates** end-to-end latency (TPOT is
nearly flat in context length), so compression's gain is in TTFT. Compression is a net
latency win only above a **break-even context length L\* ≈ 240–340 tokens**, and the real
cost is the compressor (the cross-encoder's own ~960 ms) plus lost grounding.

## Quick start

```bash
cd decaf-pilot
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install "sentence-transformers>=3.0" llmlingua psutil

ollama pull qwen2.5:3b-instruct-q8_0
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q8_0

.venv/bin/python -m src.run_v2 --config config_v2.yaml --mode all   # Phase A (resumable)
.venv/bin/python -m src.retrieval_v2 --config config_v2.yaml        # §13 retrieval modes
.venv/bin/python -m src.h5_v2 --config config_v2.yaml               # H5 KV-reuse study
.venv/bin/python -m src.analyze_v2 --config config_v2.yaml          # Phase B → results_v2/
```

Outputs land in `decaf-pilot/results_v2/`: `FULL_RESULTS_v2.md`, 41 figures, `summary_v2.json`.
