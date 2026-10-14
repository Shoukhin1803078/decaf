# DECAF — Discussion & Conclusion (paper-ready draft)

*When does retrieved-context compression pay off on CPU? A decode/prefill characterization with
break-even thresholds.*

---

## 4. Discussion

### 4.1 The lever on CPU is prefill, not decode

The GPU-era intuition behind context compression is that short prompts mean cheap decoding. On CPU we
find the opposite cost structure: **prefill grows ~5–11 ms/token** while **TPOT is nearly flat in
context length** (`+0.8–1.3 ms per 1k tokens`). Consequently the **decode share of end-to-end latency
falls** with context, and prefill dominates for any realistic multi-hop context (crossover at
`L ≈ 584–916` tokens for concise answers). Compression therefore *does* pay on CPU — but the saving is
in **TTFT/prefill**, not per-token decode, and it is capped by the compressor's own cost. This also
explains why **prefix/KV-cache reuse is a large win here** (~99% TTFT reduction, §8): it removes
prefill, the dominant term. The practical implication is that on CPU, *anything that shortens or
reuses the prompt* is the high-value lever, and decode-oriented optimizations have less headroom than
GPU intuition suggests.

### 4.2 Why the break-even `L*` exists — and why it is non-trivial

Compression is a net latency loss until the prefill it saves exceeds the compressor's cost, so there is
an explicit threshold `L*` (we measure **228–312 tokens**, depending on model). The key observation is
that `L*` is **only non-trivial because the compressor is expensive**: with a cheap BM25 scorer
(~1.9 ms) `L*` collapses to roughly the retained length — "compression is always a net win" — whereas
the cross-encoder (~865 ms) moves it into the range of realistic retrieved contexts. This makes
`O_c`/`S_net` **first-class results**, not bookkeeping: a compressor that is not much cheaper than the
decoding it saves can erase its own benefit. The proposal's design principle — *the compressor must be
cheaper than the decoding it saves* — is thus confirmed and quantified on CPU.

### 4.3 A significant model-size interaction

With 200 queries/dataset the effect of compression on answer quality is **statistically significant and
model-dependent**: on the **3B** model, removing distractor context **helps** (3B/Q4 ΔF1 +0.077,
Holm p = 0.009; fixed-20% +0.095, p = 0.001), while on the **7B** model it **hurts** (7B/Q4 −0.092,
p = 0.019). The mechanism is intuitive: a small model is degraded by long distractor context, whereas a
larger model can exploit it. This interaction was **invisible at n = 30** and is exactly the kind of
conclusion that requires adequate statistical power. Crucially, it is **dataset-dependent**: on
single-hop TriviaQA compression is **neutral for the 7B** (ΔF1 ≈ 0, n = 200) rather than harmful,
because single-hop provided contexts lack the long distractor structure that the larger model can
exploit. And on **long-context LongBench** compression is **harmful for both sizes** (3B DECAF λ6
−0.146, p<0.001): with the answer evidence spread across a long context, aggressive selection removes it
(gold-in-evidence 0.64 → 0.38). The finding is therefore about the *interaction of model size with
task/context structure*, not model size alone. It reframes the
deployment question from "should I compress?" to "**for which model, task and context length** should
I compress?".

### 4.4 Grounding is the binding constraint, not correctness

Compression preserves (or even improves) aggregate accuracy while **monotonically degrading grounding**
(gold-in-evidence 0.745 → 0.515 as evidence is removed; 0.68 → 0.40 on 2Wiki). Accuracy-only reporting
would hide this. Grounding-aware selection — rather than accuracy- or token-optimized selection — is
therefore a necessary design target, and our results quantify how much grounding each operating point
costs.

### 4.5 The CPU-aware gate does not beat plain relevance ranking

The "quality gain per unit CPU cost" gate matches but does not beat a plain relevance top-k at matched
budget (H3 not supported), and the coverage/reordering terms move accuracy only marginally. A likely
explanation is that the *scorer* dominates `T_compress` and the greedy selection itself is ~free, so
the marginal value of a smarter gate is small relative to the scoring cost. This is a caution for
adaptive-compression designs on CPU: complexity in the *policy* adds little when the *scoring* is the
expensive part.

### 4.6 Retrieval quality is a first-order lever

Switching from BM25 to **hybrid** retrieval (BM25 + dense RRF) improves both quality and grounding, and
**changes the compression conclusion**: the 7B's compressed F1 rises from 0.443 to 0.589. The apparent
"compression hurts the 7B" result is thus partly a *retrieval* limitation. Practitioners should spend
effort on retrieval before on ever-more-clever compression.

### 4.7 Quantization is second-order for these decisions

`L*` is set by model **size** (~306 vs ~228 tokens) and barely moves with quantization (±6 tokens);
quantization has no consistent effect on quality here, and Q8 is not slower in prefill than Q4 on this
CPU. For deployment, choosing the model size matters for the compression decision; Q4 vs Q8 mostly
trades memory for a small, inconsistent quality delta.

### 4.8 Practical guidance

A conservative decision rule from these measurements:

1. **Compress only above the break-even.** For the studied sizes, compression is a net *latency* loss
   below `L* ≈ 230–310` context tokens. The `L*` table (§2 of Results) is the deployment artifact.
2. **Budget the compressor.** Use a scorer whose cost is small relative to the per-token prefill cost;
   otherwise `O_c` eats the win (S_net can fall below 1 at high retention).
3. **Match the policy to model size.** Moderate compression helps small models and hurts larger ones;
   the optimum retention ratio is model-dependent and non-monotone.
4. **Verify grounding, not just accuracy.** Report grounding alongside EM/F1; monitor the gold-in-evidence
   curve.
5. **Improve retrieval first.** Hybrid retrieval is a larger, safer win than a fancier compressor.

---

## 5. Conclusion

We set out to ask, in a controlled CPU-only setting, **when context compression actually accelerates
RAG** — and found a clear, reproducible answer that reverses the GPU-era premise. On CPU, **prefill,
not decode, dominates** end-to-end latency for realistic contexts; compression's benefit is in TTFT;
and it is a **net win only above a hardware-, model- and compression-dependent break-even `L*`**, whose
existence is a direct consequence of the compressor's own cost. A powered evaluation (n = 200/dataset)
shows compression **significantly helps** the 3B model and **significantly hurts** the 7B — a
model-size interaction — while **grounding degrades monotonically** in both. Hybrid retrieval can
matter more than the compression policy, quantization is second-order, and KV-prefix reuse is a large
win precisely because prefill dominates.

The contribution is not another training-free compressor but a **decode-centric characterization of
when compression pays off on CPU**, with a reusable measurement framework (that counts the compressor),
explicit break-even thresholds, and an accuracy–latency–grounding benchmark.

**Future work.** (i) Genuinely long-generation regimes (EOS-disabled) to test where the decode share
flips; (ii) broader datasets (single-hop, long-context multi-document) and a second hardware platform;
(iii) energy and achieved-bandwidth telemetry where the OS exposes it; (iv) grounding-aware compression
policies that treat citation support as a first-class objective.

*Draft figure/table provenance: `results_v2/summary_v2.json`; 41 figures in `results_v2/`.*
