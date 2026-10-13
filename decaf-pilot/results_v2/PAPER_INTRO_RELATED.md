# DECAF — Introduction & Related Work (paper-ready draft)

*When does retrieved-context compression pay off on CPU? A decode/prefill characterization with
break-even thresholds.*

---

## 1. Introduction

Retrieval-augmented generation (RAG) improves factual grounding by conditioning a language model on
retrieved evidence, but it lengthens the prompt — and long prompts cost latency, memory, and energy.
**Context compression** is the standard remedy: keep only the evidence that matters, so the decoder
processes fewer tokens. Almost all of the evidence that compression helps comes from **GPU serving**,
where **prefill dominates** and shortening the prompt directly cuts the dominant term.

Most deployments without a GPU, however, run on **CPU**. There the cost profile is different in a way
that has not been characterized under controlled, reproducible conditions: per-token decode is slow,
memory-bandwidth-bound, and its cost depends on the KV-cache size (hence on context length), while
fill is also expensive. Whether compressing the context *actually* pays off — once the compressor's
own cost is counted — is an empirical question, not a given. A compressor that is only marginally
cheaper than the decoding it saves can be a **net loss**; and compression is known to damage
**citation grounding** far more than it damages answer correctness.

This paper is a controlled characterization of that question for small, quantized LLMs running
**entirely on CPU**. Rather than proposing yet another compressor — that space is now crowded — we
measure **when** context compression translates into a net latency win, and **what it costs** in
grounding. Concretely we contribute:

1. **A measurement framework** that separates retrieval, compression, prefill and decode, and
   formalizes compression efficiency, overhead, **net speedup `S_net`**, and a **break-even context
   length `L*`**.
2. **A decode/prefill characterization** across two model sizes and two quantization levels, showing
   that on CPU **prefill — not decode — dominates** end-to-end latency for realistic contexts, and
   that **dictates where compression helps**.
3. **A break-even analysis**: an explicit `L*` per (model, quantization) below which compression is a
   net latency loss and above which it is a net win — and the finding that `L*` is non-trivial
   *only because the compressor is expensive*.
4. **A statistically powered evaluation** (n = 200 questions/dataset) that reveals a **significant
   model-size interaction**: compression *helps* the 3B model and *hurts* the 7B, and that retrieval
   quality can matter more than the compression policy.
5. **A reproducible efficiency benchmark** with quality, latency, memory, grounding and Pareto
   frontiers, plus a secondary systems analysis of KV-cache reuse and prefetching.

**Primary research question.** When does reducing retrieved context actually lower CPU RAG latency,
and what does that saving cost in answer quality and grounding?

## 2. Related Work

Four streams are relevant.

**KV-cache reuse.** A large body of work reuses attention state across queries to avoid recomputing
prefill (e.g., prefix/KV caching, cache-blending, cache-aware schedulers). These methods accelerate
the **prefill** term and are evaluated on GPU. Our secondary analysis measures the same effect on CPU
and finds it *large* precisely because prefill dominates there — a different conclusion than the
GPU-era expectation that reuse moves only TTFT.

**Retrieval prefetching / pipelining.** Overlapping retrieval with generation hides retrieval latency
behind compute. The benefit is bounded by how much compute there is to hide behind; on CPU we quantify
this and find the retrieval/compression cost can exceed the decode it would overlap.

**Context / prompt compression.** Training-free token- and sentence-level compressors (e.g.,
LLMLingua-2, and adaptive/selective methods such as SARA, Perception Compressor and BRIEF-Pro)
reduce prompt length. Most report tokens, first-token time, or aggregate latency on GPU, and few
isolate **decode** cost or the **compressor's own overhead**. We show that on CPU the overhead is the
decisive term: a cross-encoder scorer (~0.9 s) roughly halves the net speedup, and a cheap BM25
scorer is competitive on the quality–latency frontier.

**CPU-only / on-device RAG efficiency.** Lightweight CPU RAG and quantized-CPU benchmarking exist
(e.g., HS-RAG and quantized CPU LLM benchmarks), but the **interaction** of context length,
compression, quantization and model size — and its break-even structure — is under-characterized. That
interaction is our subject.

### Positioning

We do **not** claim a new training-free compressor. Our contribution is the **CPU-specific,
decode/prefill characterization**: an explicit, hardware-dependent **break-even `L*`**, a measurement
framework that counts the compressor, and a powered evaluation showing that compression's benefit is
**model-size-dependent and grounding-limited**. Relative to the GPU-era compression literature, our
headline is a *reversal*: on CPU the lever that pays is prefill/TTFT, and the binding constraint is the
compressor's cost plus lost grounding — not decode.
