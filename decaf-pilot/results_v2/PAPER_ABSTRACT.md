# DECAF — Title & Abstract (paper-ready draft)

## Title

**When Does Context Compression Pay Off on CPU? A Decode/Prefill Characterization of
Retrieval-Augmented Generation with Break-Even Thresholds**

*Alternative:* *Compression Is Not Automatically Cheap: Measuring When Retrieved-Context Compression
Accelerates CPU RAG.*

## Abstract

Context compression is the standard remedy for the latency cost of retrieval-augmented generation
(RAG), but almost all evidence that it helps comes from GPU serving, where prefill dominates. Most
deployments without a GPU run on CPU, where the cost profile is different and, we show here, changes
the conclusion. We present a controlled characterization of **when retrieved-context compression
actually pays off on CPU** for small, quantized LLMs. Using a staged, fully-instrumented harness
(4 model×quantization configs: Qwen2.5 3B/7B in Q4_K_M/Q8_0; three benchmarks — two multi-hop,
one single-hop; 200 questions each; 15,236 runs), we separate retrieval, compression, prefill and
decode. We find that on CPU
**prefill — not decode — dominates** end-to-end latency (prefill ≈ 5–11 ms/token while per-token decode
is nearly flat in context length), so compression's benefit is in time-to-first-token, and the
GPU-era decode-centric premise does not hold for concise answers. Compression is a **net latency win
only above a break-even context length `L*` (228–312 tokens)** — a threshold that is non-trivial
*precisely because the compressor is expensive* (a cross-encoder costs ~0.9 s and roughly halves the
net speedup). With paired statistics, compression **significantly helps** the 3B model (ΔF1 +0.077,
Holm p = 0.009) yet **significantly hurts** the 7B model (ΔF1 −0.092, p = 0.019) on multi-hop
HotpotQA — while on single-hop TriviaQA it **helps even the 7B**, so the size interaction is
**dataset-dependent**. **Grounding degrades monotonically** with compression across datasets. Hybrid
retrieval improves both quality and
grounding and can matter more than the compression policy, quantization is second-order, and
llama.cpp prefix/KV reuse cuts TTFT by ~99% because prefill dominates. We release the harness,
telemetry and an accuracy–latency–grounding benchmark, and give practical break-even guidance for
CPU RAG deployment.

**Keywords:** retrieval-augmented generation; context compression; CPU inference; latency
characterization; break-even analysis; quantized LLMs; grounding evaluation.

## Highlights

- On CPU, **prefill dominates** RAG latency; compression pays off in **TTFT**, not per-token decode.
- A **break-even context length `L*` ≈ 228–312 tokens** separates net-loss from net-win compression —
  and exists only because the compressor itself is costly.
- **Model size decides the sign** of compression's effect (3B: significantly helps; 7B: significantly
  hurts on multi-hop) — but the interaction is **dataset-dependent** (on single-hop TriviaQA it helps
  both), while **quantization is second-order**.
- Compression **improves or preserves accuracy but monotonically degrades grounding** — accuracy alone
  hides the cost.
- **Hybrid retrieval** and **KV-cache reuse** are larger, safer wins than a cleverer compressor.
