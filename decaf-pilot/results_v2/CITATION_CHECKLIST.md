# DECAF — Citation Checklist & Bibliography Skeleton

The paper drafts deliberately keep external claims **generic** so no references were invented. Below,
each claim that needs a citation is mapped to a **candidate source by name**. Fill in verified
bibliographic details from the actual papers (a `.bib` skeleton is at the bottom).

> ⚠️ **Do not trust any metadata below as final** — confirm authors, venue, year and pages against the
> real publications before submission.

## Claims → candidate sources

| # | Claim (where it appears) | Candidate source(s) |
|---:|---|---|
| 1 | Multi-hop QA benchmark used as a main RAG benchmark | **HotpotQA** (Yang et al.) |
| 2 | Second multi-hop benchmark | **2WikiMultihopQA** (Ho et al.) |
| 3 | Training-free token-level prompt compression baseline | **LLMLingua-2** (Pan et al.); **LLMLingua** (Jiang et al.) |
| 4 | Adaptive / selective context compression (positioning) | **SARA**, **Perception Compressor**, **BRIEF-Pro**, **ECoRAG**, **ACC-RAG** |
| 5 | KV-cache reuse / prefix caching accelerates prefill | **CacheBlend**, **RAGCache**, **TurboRAG**, **CacheTune** |
| 6 | Retrieval prefetching / pipelining | **TeleRAG**, **HedraRAG**, **RAGO**, predictive-prefetching work |
| 7 | On-device / CPU-only RAG & quantized CPU benchmarking | **HS-RAG**, quantized-CPU-LLM benchmark studies |
| 8 | Compression can preserve correctness but damage citation grounding | evidence-grounding study (2–4% correctness vs 40–50% grounding drop) |
| 9 | Compression reduces prompt/prefill cost on GPU | GPU compression literature (REFRAG, CORE-RAG) |
| 10 | Cross-encoder reranker used as the evidence scorer | **BAAI/bge-reranker-base** (BGE / Xiao et al.) |
| 11 | Dense retriever used in the hybrid mode | **BAAI/bge-small-en-v1.5** (BGE) |
| 12 | Hybrid retrieval via reciprocal-rank fusion | **Reciprocal Rank Fusion** (Cormack et al.) |
| 13 | Reader models | **Qwen2.5-Instruct** technical report |
| 14 | CPU inference runtime | **llama.cpp** / **Ollama** (software citation) |
| 15 | Paired accuracy test | **McNemar's test** (McNemar) |
| 16 | Multiple-comparison correction | **Holm–Bonferroni** (Holm) |
| 17 | Bootstrap confidence intervals | **Efron & Tibshirani** |
| 18 | "Lost in the middle" motivation for reordering | **Liu et al., Lost in the Middle** |
| 19 | BM25 baseline | **Robertson & Zaragoza**, and the `rank_bm25` library |
| 20 | Accuracy metric (SQuAD-style EM/F1) | **Rajpurkar et al., SQuAD** |

## BibTeX skeleton (fill verified fields)

```bibtex
% --- Datasets ---
@inproceedings{yang2018hotpotqa,
  title   = {HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author  = {TODO and TODO},
  booktitle = {TODO},
  year    = {TODO}
}
@inproceedings{ho2020constructing,
  title   = {"Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps"},
  author  = {TODO},
  booktitle = {TODO},
  year    = {TODO}
}

% --- Compression ---
@inproceedings{jiang2023llmlingua,
  title = {LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models},
  author = {TODO}, booktitle = {TODO}, year = {TODO}
}
@article{pan2024llmlingua2,
  title = {LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression},
  author = {TODO}, journal = {TODO}, year = {TODO}
}
% SARA / Perception Compressor / BRIEF-Pro / ECoRAG / ACC-RAG: add as cited in the proposal's Table 1

% --- KV reuse / prefetch (proposal Table 1) ---
% CacheBlend, TurboRAG, RAGCache, CacheTune, FusionRAG, SpecCache, CoinRAG, PCR
% TeleRAG, HedraRAG, RAGO, Predictive Prefetching

% --- CPU efficiency ---
% HS-RAG; quantized-CPU-LLM benchmark

% --- Models & tools ---
@article{qwen2024qwen25,
  title = {Qwen2.5 Technical Report}, author = {TODO}, journal = {TODO}, year = {TODO}
}
@inproceedings{xiao2024bge,
  title = {C-Pack: Packed Resources for General Chinese Embeddings}, author = {TODO}, year = {TODO}
}
@misc{llamacpp, title = {llama.cpp}, author = {Gerganov, Georgi and contributors}, year = {TODO}}
@misc{ollama, title = {Ollama}, author = {TODO}, year = {TODO}}

% --- Methods ---
@article{mcnemar1947note, title = {Note on the sampling error of the difference between correlated proportions or percentages}, author = {McNemar, Quinn}, journal = {Psychometrika}, year = {1947}}
@article{holm1979simple, title = {A Simple Sequentially Rejective Multiple Test Procedure}, author = {Holm, Sture}, journal = {Scandinavian Journal of Statistics}, year = {1979}}
@book{efron1994introduction, title = {An Introduction to the Bootstrap}, author = {Efron, Bradley and Tibshirani, Robert J.}, year = {1994}}
@inproceedings{cormack2009reciprocal, title = {Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods}, author = {Cormack, Gordon V. and Clarke, Charles L. A. and Buettcher, Stefan}, booktitle = {SIGIR}, year = {2009}}
@inproceedings{liu2023lost, title = {Lost in the Middle: How Language Models Use Long Contexts}, author = {Liu, Nelson F. and others}, year = {2023}}
@article{robertson2009probabilistic, title = {The Probabilistic Relevance Framework: BM25 and Beyond}, author = {Robertson, Stephen and Zaragoza, Hugo}, year = {2009}}
@inproceedings{rajpurkar2016squad, title = {SQuAD: 100,000+ Questions for Machine Comprehension of Text}, author = {Rajpurkar, Pranav and others}, year = {2016}}
```

## Suggested citation-management flow

1. Create `references.bib` from the skeleton above and fill each `TODO` from the real source.
2. Re-check every related-work sentence in `PAPER_INTRO_RELATED.md` against its entry.
3. For the proposal's Table 1 works (SARA, Perception Compressor, CacheBlend, …), reuse the exact
   citations listed in `DECAF_CPU_RAG_Proposal_v2.md`.
4. Run a reference manager (`biber`/Zotero) to produce the final bibliography.
