# DECAF — Full Results v2

CPU-only, Ollama/llama.cpp · models: 3B / Q4_K_M, 3B / Q8_0, 7B / Q4_K_M, 7B / Q8_0 · datasets: hotpot, twowiki · 56 scaling, 60 crossover, 11760 compression runs · cross-encoder scorer `BAAI/bge-reranker-base`.

> v2 supersedes the single-model pilot: quantization (E5) + model-size (E6) pairs, a cross-encoder scorer, long-context scaling to 8K, a generation-length crossover (H1), 2Wiki generalization (E9), and paired statistics (bootstrap CI, McNemar, Holm).

## 0. Headline findings

- **3B / Q4_K_M**: prefill grows 5.51 ms/token (r²=0.990) while TPOT barely moves (+0.85 ms per 1k tokens). Decode = prefill at **L ≈ 867 tokens** (32-token outputs) → **prefill dominates above it (H1 refuted on this setup)**.
- **3B / Q8_0**: prefill grows 5.28 ms/token (r²=0.994) while TPOT barely moves (+0.81 ms per 1k tokens). Decode = prefill at **L ≈ 916 tokens** (32-token outputs) → **prefill dominates above it (H1 refuted on this setup)**.
- **7B / Q4_K_M**: prefill grows 10.92 ms/token (r²=0.998) while TPOT barely moves (+1.30 ms per 1k tokens). Decode = prefill at **L ≈ 584 tokens** (32-token outputs) → **prefill dominates above it (H1 refuted on this setup)**.
- **7B / Q8_0**: prefill grows 10.57 ms/token (r²=0.996) while TPOT barely moves (+1.24 ms per 1k tokens). Decode = prefill at **L ≈ 867 tokens** (32-token outputs) → **prefill dominates above it (H1 refuted on this setup)**.
- **Compressor cost T_compress ≈ 864 ms** (cross-encoder `bge-reranker-base`, CPU): ~3 orders of magnitude above the pilot's BM25 (~1.4 ms) and no longer negligible — this is the cost the proposal warns about.
- **3B / Q4_K_M**: retained L_c≈150 tok, break-even **L*≈306 tokens** (compression is a net latency loss below, a net win above).
- **3B / Q8_0**: retained L_c≈150 tok, break-even **L*≈312 tokens** (compression is a net latency loss below, a net win above).
- **7B / Q4_K_M**: retained L_c≈150 tok, break-even **L*≈228 tokens** (compression is a net latency loss below, a net win above).
- **7B / Q8_0**: retained L_c≈150 tok, break-even **L*≈231 tokens** (compression is a net latency loss below, a net win above).
- **Overhead matters**: 3B / Q4_K_M·hotpot DECAF(decaf_lam6): gross speedup 3.71× falls to **S_net 2.00×** once the 905 ms cross-encoder cost is counted (S_net = T_full / (T_compress + T_e2e)).
- **hotpot**: best F1 = 0.584 (7B / Q4_K_M · Fixed 60%). DECAF(decaf_lam6) F1 → 7B / Q8_0:0.445, 7B / Q4_K_M:0.443, 3B / Q4_K_M:0.262, 3B / Q8_0:0.222.
- **twowiki**: best F1 = 0.354 (7B / Q8_0 · Fixed 60%). DECAF(decaf_lam6) F1 → 7B / Q8_0:0.240, 7B / Q4_K_M:0.228, 3B / Q4_K_M:0.187, 3B / Q8_0:0.126.
- **vs LLMLingua-2 (external training-free baseline)**: its own T_compress (≈1685 ms) exceeds the cross-encoder's, and its S_net (≈1.13×) trails DECAF's (≈2.25×); DECAF also keeps more gold evidence at aggressive ratios.
- **H5 (secondary systems analysis)**: llama.cpp prefix/KV-cache reuse cuts TTFT by ~**100%** on an identical prompt and ~92–96% on a shared-context/partial overlap — far beyond the 'reuse moves only TTFT' expectation, and a large end-to-end win here precisely because prefill dominates.

## 1. Decoding scaling (E1/E8, per model)

| model       |   TPOT slope (ms/1k/L) |   prefill slope (ms/tok) |   prefill r² |   decode/prefill cross L |   N_out |
|:------------|-----------------------:|-------------------------:|-------------:|-------------------------:|--------:|
| 3B / Q4_K_M |                  0.85  |                   5.5054 |        0.99  |                      867 |      32 |
| 3B / Q8_0   |                  0.811 |                   5.2841 |        0.994 |                      916 |      32 |
| 7B / Q4_K_M |                  1.299 |                  10.9156 |        0.998 |                      584 |      32 |
| 7B / Q8_0   |                  1.244 |                  10.5736 |        0.996 |                      867 |      32 |

Figures: `fig_v2_scaling_tpot.png`, `fig_v2_scaling_prefill.png`, `fig_v2_scaling_ttft.png`, `fig_v2_scaling_e2e.png`, `fig_v2_scaling_decode_share.png`, `fig_v2_scaling_prefill_per_tok.png`, `fig_v2_scaling_decode_ms.png`, `fig_v2_scaling_peak_rss.png`, `fig_v2_scaling_panel.png`, `fig_v2_crossover_decode_share.png`, `fig_v2_crossover_length.png`.

## 2. Break-even L* per model (E4)

| model       |   L_c (tok) |   T_compress (ms) |   prefill+decode slope (ms/tok) |   L* (tok) |
|:------------|------------:|------------------:|--------------------------------:|-----------:|
| 3B / Q4_K_M |         150 |             864.3 |                           5.533 |        306 |
| 3B / Q8_0   |         150 |             864.3 |                           5.31  |        312 |
| 7B / Q4_K_M |         150 |             864.6 |                          10.957 |        228 |
| 7B / Q8_0   |         150 |             865   |                          10.613 |        231 |

Figures: `fig_v2_breakeven_curves.png`, `fig_v2_Lstar_bar.png`, `fig_v2_overhead_vs_savings.png`.

## 3. Compression: quality / latency / overhead / grounding

### hotpot

| model       | method                  |    F1 |    EM |   ev.rec |   gold-in-ev |   retained |   T_c(ms) |   E2E(ms) |   speedup× |   S_net× |
|:------------|:------------------------|------:|------:|---------:|-------------:|-----------:|----------:|----------:|-----------:|---------:|
| 3B / Q4_K_M | DECAF λ=6 (BM25 scorer) | 0.268 | 0.2   |    0.532 |        0.6   |      0.19  |     1.858 |  1300.76  |      3.015 |    3.011 |
| 3B / Q4_K_M | DECAF λ=15              | 0.3   | 0.233 |    0.356 |        0.467 |      0.055 |   988.838 |  1049.95  |      3.735 |    1.924 |
| 3B / Q4_K_M | DECAF λ=2               | 0.264 | 0.167 |    0.614 |        0.667 |      0.208 |   988.735 |  1127.84  |      3.477 |    1.853 |
| 3B / Q4_K_M | DECAF λ=6               | 0.262 | 0.13  |    0.459 |        0.515 |      0.12  |   905.267 |  1058.1   |      3.707 |    1.998 |
| 3B / Q4_K_M | DECAF no-cov (λ=6)      | 0.253 | 0.13  |    0.505 |        0.575 |      0.166 |   905.409 |  1144.49  |      3.427 |    1.913 |
| 3B / Q4_K_M | DECAF no-reorder (λ=6)  | 0.267 | 0.2   |    0.572 |        0.633 |      0.128 |   988.719 |  1169.8   |      3.353 |    1.817 |
| 3B / Q4_K_M | Fixed 20%               | 0.28  | 0.155 |    0.586 |        0.655 |      0.238 |   904.872 |  1097.11  |      3.575 |    1.959 |
| 3B / Q4_K_M | Fixed 40%               | 0.298 | 0.2   |    0.646 |        0.7   |      0.451 |   988.815 |  1007.68  |      3.892 |    1.964 |
| 3B / Q4_K_M | Fixed 50%               | 0.258 | 0.13  |    0.68  |        0.72  |      0.55  |   905.014 |  1126.69  |      3.481 |    1.93  |
| 3B / Q4_K_M | Fixed 60%               | 0.25  | 0.133 |    0.707 |        0.8   |      0.664 |   988.658 |  1189.04  |      3.298 |    1.801 |
| 3B / Q4_K_M | Fixed 80%               | 0.217 | 0.1   |    0.726 |        0.8   |      0.847 |   988.615 |  3175.02  |      1.235 |    0.942 |
| 3B / Q4_K_M | Full context            | 0.185 | 0.05  |    0.722 |        0.745 |      1     |   904.806 |  3921.94  |      1     |    1     |
| 3B / Q4_K_M | LLMLingua-2 (20%)       | 0.201 | 0.167 |  nan     |        0.433 |      0.206 |  1069.54  |  1747.74  |      2.244 |    1.392 |
| 3B / Q4_K_M | LLMLingua-2 (50%)       | 0.281 | 0.167 |  nan     |        0.7   |      0.499 |  1117.15  |  2263.61  |      1.733 |    1.16  |
| 3B / Q4_K_M | Relevance top-m (iso)   | 0.261 | 0.125 |    0.47  |        0.53  |      0.128 |   904.912 |  1126.57  |      3.481 |    1.931 |
| 3B / Q8_0   | DECAF λ=6 (BM25 scorer) | 0.279 | 0.2   |    0.532 |        0.6   |      0.19  |     1.887 |  2002.19  |      2.169 |    2.167 |
| 3B / Q8_0   | DECAF λ=15              | 0.249 | 0.2   |    0.356 |        0.467 |      0.055 |   988.929 |  1879.81  |      2.31  |    1.514 |
| 3B / Q8_0   | DECAF λ=2               | 0.293 | 0.2   |    0.614 |        0.667 |      0.208 |   989.241 |  1631.43  |      2.662 |    1.657 |
| 3B / Q8_0   | DECAF λ=6               | 0.222 | 0.095 |    0.459 |        0.515 |      0.12  |   905.302 |  1805.56  |      2.405 |    1.602 |
| 3B / Q8_0   | DECAF no-cov (λ=6)      | 0.233 | 0.11  |    0.505 |        0.575 |      0.166 |   905.445 |  1831.05  |      2.372 |    1.587 |
| 3B / Q8_0   | DECAF no-reorder (λ=6)  | 0.273 | 0.2   |    0.572 |        0.633 |      0.128 |   988.806 |  1920.26  |      2.262 |    1.493 |
| 3B / Q8_0   | Fixed 20%               | 0.249 | 0.13  |    0.586 |        0.655 |      0.238 |   905.37  |  1771.6   |      2.451 |    1.622 |
| 3B / Q8_0   | Fixed 40%               | 0.239 | 0.133 |    0.646 |        0.7   |      0.451 |   988.69  |  1798.13  |      2.415 |    1.558 |
| 3B / Q8_0   | Fixed 50%               | 0.232 | 0.09  |    0.68  |        0.72  |      0.55  |   905.275 |  1755.49  |      2.474 |    1.632 |
| 3B / Q8_0   | Fixed 60%               | 0.272 | 0.167 |    0.707 |        0.8   |      0.664 |   988.824 |  1754.68  |      2.475 |    1.583 |
| 3B / Q8_0   | Fixed 80%               | 0.211 | 0.1   |    0.726 |        0.8   |      0.847 |   988.603 |  3658.8   |      1.187 |    0.934 |
| 3B / Q8_0   | Full context            | 0.196 | 0.06  |    0.722 |        0.745 |      1     |   904.811 |  4342.7   |      1     |    1     |
| 3B / Q8_0   | LLMLingua-2 (20%)       | 0.097 | 0.067 |  nan     |        0.433 |      0.206 |  1598.61  |  2347.09  |      1.85  |    1.101 |
| 3B / Q8_0   | LLMLingua-2 (50%)       | 0.176 | 0.067 |  nan     |        0.7   |      0.499 |  1519.83  |  2948.08  |      1.473 |    0.972 |
| 3B / Q8_0   | Relevance top-m (iso)   | 0.224 | 0.09  |    0.47  |        0.53  |      0.128 |   905.106 |  1885.34  |      2.303 |    1.556 |
| 7B / Q4_K_M | DECAF λ=6 (BM25 scorer) | 0.52  | 0.433 |    0.532 |        0.6   |      0.19  |     1.884 |  2063.88  |      3.87  |    3.866 |
| 7B / Q4_K_M | DECAF λ=15              | 0.335 | 0.3   |    0.356 |        0.467 |      0.055 |   988.941 |  1542.44  |      5.178 |    3.155 |
| 7B / Q4_K_M | DECAF λ=2               | 0.513 | 0.433 |    0.614 |        0.667 |      0.208 |   988.856 |  1509.54  |      5.291 |    3.197 |
| 7B / Q4_K_M | DECAF λ=6               | 0.443 | 0.3   |    0.459 |        0.515 |      0.12  |   905.503 |   874.469 |      9.133 |    4.487 |
| 7B / Q4_K_M | DECAF no-cov (λ=6)      | 0.45  | 0.315 |    0.505 |        0.575 |      0.166 |   905.245 |  1433.18  |      5.573 |    3.415 |
| 7B / Q4_K_M | DECAF no-reorder (λ=6)  | 0.501 | 0.4   |    0.572 |        0.633 |      0.128 |   989.466 |  1576.64  |      5.066 |    3.112 |
| 7B / Q4_K_M | Fixed 20%               | 0.48  | 0.335 |    0.586 |        0.655 |      0.238 |   905.065 |  1136.81  |      7.025 |    3.911 |
| 7B / Q4_K_M | Fixed 40%               | 0.501 | 0.4   |    0.646 |        0.7   |      0.451 |   988.723 |   865.414 |      9.229 |    4.307 |
| 7B / Q4_K_M | Fixed 50%               | 0.516 | 0.38  |    0.68  |        0.72  |      0.55  |   905.182 |   939.809 |      8.498 |    4.329 |
| 7B / Q4_K_M | Fixed 60%               | 0.584 | 0.5   |    0.707 |        0.8   |      0.664 |   988.801 |   837.078 |      9.541 |    4.374 |
| 7B / Q4_K_M | Fixed 80%               | 0.565 | 0.5   |    0.726 |        0.8   |      0.847 |   988.523 |  6457.44  |      1.237 |    1.073 |
| 7B / Q4_K_M | Full context            | 0.535 | 0.405 |    0.722 |        0.745 |      1     |   904.803 |  7986.67  |      1     |    1     |
| 7B / Q4_K_M | LLMLingua-2 (20%)       | 0.245 | 0.167 |  nan     |        0.433 |      0.206 |  2082.96  |  2399.37  |      3.329 |    1.782 |
| 7B / Q4_K_M | LLMLingua-2 (50%)       | 0.573 | 0.433 |  nan     |        0.7   |      0.499 |  1770.59  |  4569.17  |      1.748 |    1.26  |
| 7B / Q4_K_M | Relevance top-m (iso)   | 0.443 | 0.315 |    0.47  |        0.53  |      0.128 |   905.21  |  1135.47  |      7.034 |    3.914 |
| 7B / Q8_0   | DECAF λ=6 (BM25 scorer) | 0.539 | 0.467 |    0.532 |        0.6   |      0.19  |     2.131 |  2325.19  |      3.508 |    3.505 |
| 7B / Q8_0   | DECAF λ=15              | 0.358 | 0.267 |    0.356 |        0.467 |      0.055 |   989.341 |  1886.66  |      4.323 |    2.836 |
| 7B / Q8_0   | DECAF λ=2               | 0.547 | 0.433 |    0.614 |        0.667 |      0.208 |   989.42  |  1678.37  |      4.86  |    3.057 |
| 7B / Q8_0   | DECAF λ=6               | 0.445 | 0.3   |    0.459 |        0.515 |      0.12  |   905.166 |  1436.24  |      5.679 |    3.484 |
| 7B / Q8_0   | DECAF no-cov (λ=6)      | 0.452 | 0.315 |    0.505 |        0.575 |      0.166 |   905.265 |  1750.32  |      4.66  |    3.071 |
| 7B / Q8_0   | DECAF no-reorder (λ=6)  | 0.522 | 0.433 |    0.572 |        0.633 |      0.128 |   989.178 |  1846.85  |      4.416 |    2.876 |
| 7B / Q8_0   | Fixed 20%               | 0.485 | 0.345 |    0.586 |        0.655 |      0.238 |   904.989 |  1709.95  |      4.77  |    3.119 |
| 7B / Q8_0   | Fixed 40%               | 0.534 | 0.433 |    0.646 |        0.7   |      0.451 |   988.838 |  1099.31  |      7.42  |    3.906 |
| 7B / Q8_0   | Fixed 50%               | 0.523 | 0.385 |    0.68  |        0.72  |      0.55  |   905.409 |  1217.82  |      6.698 |    3.842 |
| 7B / Q8_0   | Fixed 60%               | 0.565 | 0.467 |    0.707 |        0.8   |      0.664 |   989.384 |  1064.07  |      7.665 |    3.972 |
| 7B / Q8_0   | Fixed 80%               | 0.578 | 0.5   |    0.726 |        0.8   |      0.847 |   988.47  |  7140.95  |      1.142 |    1.003 |
| 7B / Q8_0   | Full context            | 0.509 | 0.39  |    0.722 |        0.745 |      1     |   904.788 |  8156.46  |      1     |    1     |
| 7B / Q8_0   | LLMLingua-2 (20%)       | 0.232 | 0.167 |  nan     |        0.433 |      0.206 |  4249.33  |  2541.59  |      3.209 |    1.201 |
| 7B / Q8_0   | LLMLingua-2 (50%)       | 0.542 | 0.433 |  nan     |        0.7   |      0.499 |  4382.64  |  4742.77  |      1.72  |    0.894 |
| 7B / Q8_0   | Relevance top-m (iso)   | 0.446 | 0.31  |    0.47  |        0.53  |      0.128 |   905.036 |  1606.49  |      5.077 |    3.248 |

### twowiki

| model       | method                  |    F1 |    EM |   ev.rec |   gold-in-ev |   retained |   T_c(ms) |   E2E(ms) |   speedup× |   S_net× |
|:------------|:------------------------|------:|------:|---------:|-------------:|-----------:|----------:|----------:|-----------:|---------:|
| 3B / Q4_K_M | DECAF λ=6 (BM25 scorer) | 0.281 | 0.233 |    0.417 |        0.467 |      0.208 |     1.692 |  1316.34  |      2.589 |    2.585 |
| 3B / Q4_K_M | DECAF λ=15              | 0.251 | 0.167 |    0.325 |        0.4   |      0.084 |   897.186 |   984.222 |      3.462 |    1.811 |
| 3B / Q4_K_M | DECAF λ=2               | 0.309 | 0.233 |    0.533 |        0.7   |      0.156 |   897.227 |  1079.24  |      3.157 |    1.724 |
| 3B / Q4_K_M | DECAF λ=6               | 0.187 | 0.085 |    0.349 |        0.4   |      0.115 |   833.165 |   960.687 |      3.547 |    1.899 |
| 3B / Q4_K_M | DECAF no-cov (λ=6)      | 0.198 | 0.1   |    0.399 |        0.47  |      0.161 |   832.989 |  1062.83  |      3.206 |    1.797 |
| 3B / Q4_K_M | DECAF no-reorder (λ=6)  | 0.299 | 0.233 |    0.458 |        0.533 |      0.102 |   897.132 |  1110.53  |      3.068 |    1.697 |
| 3B / Q4_K_M | Fixed 20%               | 0.223 | 0.12  |    0.474 |        0.555 |      0.267 |   832.883 |  1001.13  |      3.404 |    1.858 |
| 3B / Q4_K_M | Fixed 40%               | 0.243 | 0.167 |    0.6   |        0.767 |      0.469 |   897.246 |  1114.69  |      3.057 |    1.694 |
| 3B / Q4_K_M | Fixed 50%               | 0.202 | 0.1   |    0.574 |        0.65  |      0.562 |   832.88  |  1106.97  |      3.078 |    1.757 |
| 3B / Q4_K_M | Fixed 60%               | 0.143 | 0.067 |    0.617 |        0.767 |      0.663 |   897.366 |  1116.23  |      3.053 |    1.692 |
| 3B / Q4_K_M | Fixed 80%               | 0.192 | 0.133 |    0.617 |        0.767 |      0.846 |   897.051 |  3590     |      0.949 |    0.759 |
| 3B / Q4_K_M | Full context            | 0.187 | 0.07  |    0.612 |        0.68  |      1     |   832.705 |  3407.38  |      1     |    1     |
| 3B / Q4_K_M | LLMLingua-2 (20%)       | 0.216 | 0.2   |  nan     |        0.167 |      0.202 |  1281.02  |  1349.39  |      2.525 |    1.295 |
| 3B / Q4_K_M | LLMLingua-2 (50%)       | 0.17  | 0.1   |  nan     |        0.6   |      0.495 |  1156.13  |  2388.64  |      1.426 |    0.961 |
| 3B / Q4_K_M | Relevance top-m (iso)   | 0.187 | 0.08  |    0.356 |        0.4   |      0.127 |   832.832 |  1076.3   |      3.166 |    1.785 |
| 3B / Q8_0   | DECAF λ=6 (BM25 scorer) | 0.161 | 0.067 |    0.417 |        0.467 |      0.208 |     1.899 |  2061.06  |      1.91  |    1.908 |
| 3B / Q8_0   | DECAF λ=15              | 0.152 | 0.033 |    0.325 |        0.4   |      0.084 |   897.234 |  1891.03  |      2.082 |    1.412 |
| 3B / Q8_0   | DECAF λ=2               | 0.138 | 0     |    0.533 |        0.7   |      0.156 |   897.215 |  1923.94  |      2.046 |    1.395 |
| 3B / Q8_0   | DECAF λ=6               | 0.126 | 0.015 |    0.349 |        0.4   |      0.115 |   832.946 |  1857.07  |      2.12  |    1.464 |
| 3B / Q8_0   | DECAF no-cov (λ=6)      | 0.147 | 0.03  |    0.399 |        0.47  |      0.161 |   832.908 |  1878.82  |      2.095 |    1.452 |
| 3B / Q8_0   | DECAF no-reorder (λ=6)  | 0.136 | 0     |    0.458 |        0.533 |      0.102 |   897.21  |  1933.91  |      2.036 |    1.391 |
| 3B / Q8_0   | Fixed 20%               | 0.175 | 0.05  |    0.474 |        0.555 |      0.267 |   832.768 |  1929.25  |      2.041 |    1.425 |
| 3B / Q8_0   | Fixed 40%               | 0.212 | 0.133 |    0.6   |        0.767 |      0.469 |   897.145 |  1713.17  |      2.298 |    1.508 |
| 3B / Q8_0   | Fixed 50%               | 0.188 | 0.075 |    0.574 |        0.65  |      0.562 |   832.865 |  1928.73  |      2.041 |    1.426 |
| 3B / Q8_0   | Fixed 60%               | 0.182 | 0.067 |    0.617 |        0.767 |      0.663 |   897.279 |  1846.5   |      2.132 |    1.435 |
| 3B / Q8_0   | Fixed 80%               | 0.169 | 0.067 |    0.617 |        0.767 |      0.846 |   897.037 |  3842.55  |      1.025 |    0.831 |
| 3B / Q8_0   | Full context            | 0.158 | 0.035 |    0.612 |        0.68  |      1     |   832.808 |  3936.9   |      1     |    1     |
| 3B / Q8_0   | LLMLingua-2 (20%)       | 0.15  | 0.067 |  nan     |        0.167 |      0.202 |  1155.88  |  2267.95  |      1.736 |    1.15  |
| 3B / Q8_0   | LLMLingua-2 (50%)       | 0.172 | 0.067 |  nan     |        0.6   |      0.495 |  1255.84  |  2924.04  |      1.346 |    0.942 |
| 3B / Q8_0   | Relevance top-m (iso)   | 0.13  | 0.015 |    0.356 |        0.4   |      0.127 |   832.895 |  1914.14  |      2.057 |    1.433 |
| 7B / Q4_K_M | DECAF λ=6 (BM25 scorer) | 0.231 | 0.167 |    0.417 |        0.467 |      0.208 |     1.851 |  2996.47  |      2.492 |    2.491 |
| 7B / Q4_K_M | DECAF λ=15              | 0.193 | 0.033 |    0.325 |        0.4   |      0.084 |   897.266 |  1564.49  |      4.773 |    3.034 |
| 7B / Q4_K_M | DECAF λ=2               | 0.279 | 0.133 |    0.533 |        0.7   |      0.156 |   897.306 |  2111.67  |      3.536 |    2.482 |
| 7B / Q4_K_M | DECAF λ=6               | 0.228 | 0.105 |    0.349 |        0.4   |      0.115 |   833.098 |  1476.4   |      5.058 |    3.234 |
| 7B / Q4_K_M | DECAF no-cov (λ=6)      | 0.241 | 0.11  |    0.399 |        0.47  |      0.161 |   832.842 |  1691.97  |      4.414 |    2.958 |
| 7B / Q4_K_M | DECAF no-reorder (λ=6)  | 0.336 | 0.2   |    0.458 |        0.533 |      0.102 |   897.394 |  2046.78  |      3.649 |    2.536 |
| 7B / Q4_K_M | Fixed 20%               | 0.26  | 0.115 |    0.474 |        0.555 |      0.267 |   832.801 |  1960.17  |      3.81  |    2.674 |
| 7B / Q4_K_M | Fixed 40%               | 0.313 | 0.2   |    0.6   |        0.767 |      0.469 |   897.229 |  1524.62  |      4.898 |    3.084 |
| 7B / Q4_K_M | Fixed 50%               | 0.263 | 0.125 |    0.574 |        0.65  |      0.562 |   833.32  |  1954.95  |      3.82  |    2.678 |
| 7B / Q4_K_M | Fixed 60%               | 0.333 | 0.2   |    0.617 |        0.767 |      0.663 |   897.38  |  1762.27  |      4.238 |    2.808 |
| 7B / Q4_K_M | Fixed 80%               | 0.352 | 0.233 |    0.617 |        0.767 |      0.846 |   897.011 |  6628.3   |      1.127 |    0.992 |
| 7B / Q4_K_M | Full context            | 0.251 | 0.12  |    0.612 |        0.68  |      1     |   832.75  |  7467.77  |      1     |    1     |
| 7B / Q4_K_M | LLMLingua-2 (20%)       | 0.283 | 0.2   |  nan     |        0.167 |      0.202 |  2414.51  |  3096.74  |      2.411 |    1.355 |
| 7B / Q4_K_M | LLMLingua-2 (50%)       | 0.294 | 0.167 |  nan     |        0.6   |      0.495 |  2483.88  |  4858.98  |      1.537 |    1.017 |
| 7B / Q4_K_M | Relevance top-m (iso)   | 0.24  | 0.105 |    0.356 |        0.4   |      0.127 |   832.798 |  1676.74  |      4.454 |    2.976 |
| 7B / Q8_0   | DECAF λ=6 (BM25 scorer) | 0.237 | 0.167 |    0.417 |        0.467 |      0.208 |     2.339 |  4280.22  |      1.883 |    1.882 |
| 7B / Q8_0   | DECAF λ=15              | 0.217 | 0.133 |    0.325 |        0.4   |      0.084 |   897.659 |  2053.28  |      3.925 |    2.731 |
| 7B / Q8_0   | DECAF λ=2               | 0.288 | 0.167 |    0.533 |        0.7   |      0.156 |   897.534 |  2226.48  |      3.62  |    2.58  |
| 7B / Q8_0   | DECAF λ=6               | 0.24  | 0.12  |    0.349 |        0.4   |      0.115 |   833.094 |  2389.26  |      3.373 |    2.501 |
| 7B / Q8_0   | DECAF no-cov (λ=6)      | 0.278 | 0.15  |    0.399 |        0.47  |      0.161 |   833.072 |  2426.77  |      3.321 |    2.472 |
| 7B / Q8_0   | DECAF no-reorder (λ=6)  | 0.322 | 0.2   |    0.458 |        0.533 |      0.102 |   897.968 |  2296.41  |      3.509 |    2.523 |
| 7B / Q8_0   | Fixed 20%               | 0.291 | 0.14  |    0.474 |        0.555 |      0.267 |   832.878 |  2577.48  |      3.127 |    2.363 |
| 7B / Q8_0   | Fixed 40%               | 0.331 | 0.2   |    0.6   |        0.767 |      0.469 |   897.323 |  2378.52  |      3.388 |    2.46  |
| 7B / Q8_0   | Fixed 50%               | 0.285 | 0.145 |    0.574 |        0.65  |      0.562 |   833.087 |  2860.76  |      2.817 |    2.182 |
| 7B / Q8_0   | Fixed 60%               | 0.354 | 0.233 |    0.617 |        0.767 |      0.663 |   897.442 |  1965.23  |      4.101 |    2.815 |
| 7B / Q8_0   | Fixed 80%               | 0.29  | 0.167 |    0.617 |        0.767 |      0.846 |   897.086 |  7207.89  |      1.118 |    0.994 |
| 7B / Q8_0   | Full context            | 0.259 | 0.12  |    0.612 |        0.68  |      1     |   832.7   |  8059.23  |      1     |    1     |
| 7B / Q8_0   | LLMLingua-2 (20%)       | 0.309 | 0.233 |  nan     |        0.167 |      0.202 |  4282.89  |  3188.97  |      2.527 |    1.079 |
| 7B / Q8_0   | LLMLingua-2 (50%)       | 0.3   | 0.167 |  nan     |        0.6   |      0.495 |  4472.94  |  6159.25  |      1.308 |    0.758 |
| 7B / Q8_0   | Relevance top-m (iso)   | 0.251 | 0.12  |    0.356 |        0.4   |      0.127 |   834.098 |  2495.77  |      3.229 |    2.42  |

Figures: `fig_v2_pareto_f1_e2e.png`, `fig_v2_pareto_f1_ttft.png`, `fig_v2_pareto_f1_retained.png`, `fig_v2_pareto_all.png`, `fig_v2_pareto_frontier.png`, `fig_v2_ttft_vs_tpot.png`, `fig_v2_grounding_vs_ratio.png`, `fig_v2_quality_vs_grounding.png`, `fig_v2_f1_by_method.png`, `fig_v2_speedup_bar.png`, `fig_v2_latency_breakdown.png`, `fig_v2_ev_recall_prec.png`, `fig_v2_ce_cost.png`, `fig_v2_qe_bar.png`, `fig_v2_method_model_heatmap.png`, `fig_v2_e2e_ci.png`, `fig_v2_dataset_compare.png`.

## 4. λ sweep and component ablation (Sec. 17)

| dataset   | model       |   λ |    F1 |   retained |   gold-in-ev |   E2E(ms) |
|:----------|:------------|----:|------:|-----------:|-------------:|----------:|
| hotpot    | 3B / Q4_K_M |   2 | 0.264 |      0.208 |        0.667 |      1128 |
| hotpot    | 3B / Q4_K_M |   6 | 0.262 |      0.12  |        0.515 |      1058 |
| hotpot    | 3B / Q4_K_M |  15 | 0.3   |      0.055 |        0.467 |      1050 |
| hotpot    | 3B / Q8_0   |   2 | 0.293 |      0.208 |        0.667 |      1631 |
| hotpot    | 3B / Q8_0   |   6 | 0.222 |      0.12  |        0.515 |      1806 |
| hotpot    | 3B / Q8_0   |  15 | 0.249 |      0.055 |        0.467 |      1880 |
| hotpot    | 7B / Q4_K_M |   2 | 0.513 |      0.208 |        0.667 |      1510 |
| hotpot    | 7B / Q4_K_M |   6 | 0.443 |      0.12  |        0.515 |       874 |
| hotpot    | 7B / Q4_K_M |  15 | 0.335 |      0.055 |        0.467 |      1542 |
| hotpot    | 7B / Q8_0   |   2 | 0.547 |      0.208 |        0.667 |      1678 |
| hotpot    | 7B / Q8_0   |   6 | 0.445 |      0.12  |        0.515 |      1436 |
| hotpot    | 7B / Q8_0   |  15 | 0.358 |      0.055 |        0.467 |      1887 |
| twowiki   | 3B / Q4_K_M |   2 | 0.309 |      0.156 |        0.7   |      1079 |
| twowiki   | 3B / Q4_K_M |   6 | 0.187 |      0.115 |        0.4   |       961 |
| twowiki   | 3B / Q4_K_M |  15 | 0.251 |      0.084 |        0.4   |       984 |
| twowiki   | 3B / Q8_0   |   2 | 0.138 |      0.156 |        0.7   |      1924 |
| twowiki   | 3B / Q8_0   |   6 | 0.126 |      0.115 |        0.4   |      1857 |
| twowiki   | 3B / Q8_0   |  15 | 0.152 |      0.084 |        0.4   |      1891 |
| twowiki   | 7B / Q4_K_M |   2 | 0.279 |      0.156 |        0.7   |      2112 |
| twowiki   | 7B / Q4_K_M |   6 | 0.228 |      0.115 |        0.4   |      1476 |
| twowiki   | 7B / Q4_K_M |  15 | 0.193 |      0.084 |        0.4   |      1564 |
| twowiki   | 7B / Q8_0   |   2 | 0.288 |      0.156 |        0.7   |      2226 |
| twowiki   | 7B / Q8_0   |   6 | 0.24  |      0.115 |        0.4   |      2389 |
| twowiki   | 7B / Q8_0   |  15 | 0.217 |      0.084 |        0.4   |      2053 |

Ablation (coverage / reorder / CPU-relevance-only vs full DECAF λ6):

| dataset   | model       | component              |    F1 |   ev.rec |   gold-in-ev |   retained |
|:----------|:------------|:-----------------------|------:|---------:|-------------:|-----------:|
| hotpot    | 3B / Q4_K_M | DECAF λ=6              | 0.262 |    0.459 |        0.515 |      0.12  |
| hotpot    | 3B / Q4_K_M | DECAF no-cov (λ=6)     | 0.253 |    0.505 |        0.575 |      0.166 |
| hotpot    | 3B / Q4_K_M | DECAF no-reorder (λ=6) | 0.267 |    0.572 |        0.633 |      0.128 |
| hotpot    | 3B / Q4_K_M | Relevance top-m (iso)  | 0.261 |    0.47  |        0.53  |      0.128 |
| hotpot    | 3B / Q8_0   | DECAF λ=6              | 0.222 |    0.459 |        0.515 |      0.12  |
| hotpot    | 3B / Q8_0   | DECAF no-cov (λ=6)     | 0.233 |    0.505 |        0.575 |      0.166 |
| hotpot    | 3B / Q8_0   | DECAF no-reorder (λ=6) | 0.273 |    0.572 |        0.633 |      0.128 |
| hotpot    | 3B / Q8_0   | Relevance top-m (iso)  | 0.224 |    0.47  |        0.53  |      0.128 |
| hotpot    | 7B / Q4_K_M | DECAF λ=6              | 0.443 |    0.459 |        0.515 |      0.12  |
| hotpot    | 7B / Q4_K_M | DECAF no-cov (λ=6)     | 0.45  |    0.505 |        0.575 |      0.166 |
| hotpot    | 7B / Q4_K_M | DECAF no-reorder (λ=6) | 0.501 |    0.572 |        0.633 |      0.128 |
| hotpot    | 7B / Q4_K_M | Relevance top-m (iso)  | 0.443 |    0.47  |        0.53  |      0.128 |
| hotpot    | 7B / Q8_0   | DECAF λ=6              | 0.445 |    0.459 |        0.515 |      0.12  |
| hotpot    | 7B / Q8_0   | DECAF no-cov (λ=6)     | 0.452 |    0.505 |        0.575 |      0.166 |
| hotpot    | 7B / Q8_0   | DECAF no-reorder (λ=6) | 0.522 |    0.572 |        0.633 |      0.128 |
| hotpot    | 7B / Q8_0   | Relevance top-m (iso)  | 0.446 |    0.47  |        0.53  |      0.128 |
| twowiki   | 3B / Q4_K_M | DECAF λ=6              | 0.187 |    0.349 |        0.4   |      0.115 |
| twowiki   | 3B / Q4_K_M | DECAF no-cov (λ=6)     | 0.198 |    0.399 |        0.47  |      0.161 |
| twowiki   | 3B / Q4_K_M | DECAF no-reorder (λ=6) | 0.299 |    0.458 |        0.533 |      0.102 |
| twowiki   | 3B / Q4_K_M | Relevance top-m (iso)  | 0.187 |    0.356 |        0.4   |      0.127 |
| twowiki   | 3B / Q8_0   | DECAF λ=6              | 0.126 |    0.349 |        0.4   |      0.115 |
| twowiki   | 3B / Q8_0   | DECAF no-cov (λ=6)     | 0.147 |    0.399 |        0.47  |      0.161 |
| twowiki   | 3B / Q8_0   | DECAF no-reorder (λ=6) | 0.136 |    0.458 |        0.533 |      0.102 |
| twowiki   | 3B / Q8_0   | Relevance top-m (iso)  | 0.13  |    0.356 |        0.4   |      0.127 |
| twowiki   | 7B / Q4_K_M | DECAF λ=6              | 0.228 |    0.349 |        0.4   |      0.115 |
| twowiki   | 7B / Q4_K_M | DECAF no-cov (λ=6)     | 0.241 |    0.399 |        0.47  |      0.161 |
| twowiki   | 7B / Q4_K_M | DECAF no-reorder (λ=6) | 0.336 |    0.458 |        0.533 |      0.102 |
| twowiki   | 7B / Q4_K_M | Relevance top-m (iso)  | 0.24  |    0.356 |        0.4   |      0.127 |
| twowiki   | 7B / Q8_0   | DECAF λ=6              | 0.24  |    0.349 |        0.4   |      0.115 |
| twowiki   | 7B / Q8_0   | DECAF no-cov (λ=6)     | 0.278 |    0.399 |        0.47  |      0.161 |
| twowiki   | 7B / Q8_0   | DECAF no-reorder (λ=6) | 0.322 |    0.458 |        0.533 |      0.102 |
| twowiki   | 7B / Q8_0   | Relevance top-m (iso)  | 0.251 |    0.356 |        0.4   |      0.127 |

**E2 — compression-ratio grid (hotpot):**

| model       | method       |    F1 |   gold-in-ev |   retained |   S_net |
|:------------|:-------------|------:|-------------:|-----------:|--------:|
| 3B / Q4_K_M | Fixed 20%    | 0.28  |        0.655 |      0.238 |    1.96 |
| 3B / Q4_K_M | Fixed 40%    | 0.298 |        0.7   |      0.451 |    1.96 |
| 3B / Q4_K_M | Fixed 50%    | 0.258 |        0.72  |      0.55  |    1.93 |
| 3B / Q4_K_M | Fixed 60%    | 0.25  |        0.8   |      0.664 |    1.8  |
| 3B / Q4_K_M | Fixed 80%    | 0.217 |        0.8   |      0.847 |    0.94 |
| 3B / Q4_K_M | Full context | 0.185 |        0.745 |      1     |    1    |
| 3B / Q8_0   | Fixed 20%    | 0.249 |        0.655 |      0.238 |    1.62 |
| 3B / Q8_0   | Fixed 40%    | 0.239 |        0.7   |      0.451 |    1.56 |
| 3B / Q8_0   | Fixed 50%    | 0.232 |        0.72  |      0.55  |    1.63 |
| 3B / Q8_0   | Fixed 60%    | 0.272 |        0.8   |      0.664 |    1.58 |
| 3B / Q8_0   | Fixed 80%    | 0.211 |        0.8   |      0.847 |    0.93 |
| 3B / Q8_0   | Full context | 0.196 |        0.745 |      1     |    1    |
| 7B / Q4_K_M | Fixed 20%    | 0.48  |        0.655 |      0.238 |    3.91 |
| 7B / Q4_K_M | Fixed 40%    | 0.501 |        0.7   |      0.451 |    4.31 |
| 7B / Q4_K_M | Fixed 50%    | 0.516 |        0.72  |      0.55  |    4.33 |
| 7B / Q4_K_M | Fixed 60%    | 0.584 |        0.8   |      0.664 |    4.37 |
| 7B / Q4_K_M | Fixed 80%    | 0.565 |        0.8   |      0.847 |    1.07 |
| 7B / Q4_K_M | Full context | 0.535 |        0.745 |      1     |    1    |
| 7B / Q8_0   | Fixed 20%    | 0.485 |        0.655 |      0.238 |    3.12 |
| 7B / Q8_0   | Fixed 40%    | 0.534 |        0.7   |      0.451 |    3.91 |
| 7B / Q8_0   | Fixed 50%    | 0.523 |        0.72  |      0.55  |    3.84 |
| 7B / Q8_0   | Fixed 60%    | 0.565 |        0.8   |      0.664 |    3.97 |
| 7B / Q8_0   | Fixed 80%    | 0.578 |        0.8   |      0.847 |    1    |
| 7B / Q8_0   | Full context | 0.509 |        0.745 |      1     |    1    |

**Sec.17 — scorer ablation (hotpot):**

| model       | scorer            |    F1 |   gold-in-ev |   retained |   T_c (ms) |
|:------------|:------------------|------:|-------------:|-----------:|-----------:|
| 3B / Q4_K_M | cross-encoder     | 0.262 |        0.515 |       0.12 |     905.27 |
| 3B / Q4_K_M | BM25 (scorer off) | 0.268 |        0.6   |       0.19 |       1.86 |
| 3B / Q8_0   | cross-encoder     | 0.222 |        0.515 |       0.12 |     905.3  |
| 3B / Q8_0   | BM25 (scorer off) | 0.279 |        0.6   |       0.19 |       1.89 |
| 7B / Q4_K_M | cross-encoder     | 0.443 |        0.515 |       0.12 |     905.5  |
| 7B / Q4_K_M | BM25 (scorer off) | 0.52  |        0.6   |       0.19 |       1.88 |
| 7B / Q8_0   | cross-encoder     | 0.445 |        0.515 |       0.12 |     905.17 |
| 7B / Q8_0   | BM25 (scorer off) | 0.539 |        0.6   |       0.19 |       2.13 |

Figures: `fig_v2_lambda_sweep.png`, `fig_v2_ablation_bars.png`, `fig_v2_ratio_grid.png`, `fig_v2_scorer_ablation.png`.

## 5. Paired statistics (bootstrap CI, McNemar, Holm)

| run                 | method                  |   ΔF1 vs full | 95% CI           |   McNemar p |   Holm p |
|:--------------------|:------------------------|--------------:|:-----------------|------------:|---------:|
| 3B / Q4_K_M|hotpot  | Fixed 80%               |         0.018 | [-0.031, 0.096]  |       1     |    1     |
| 3B / Q4_K_M|hotpot  | Fixed 60%               |         0.05  | [-0.012, 0.145]  |       0.5   |    1     |
| 3B / Q4_K_M|hotpot  | Fixed 50%               |         0.073 | [0.039, 0.110]   |       0     |    0.005 |
| 3B / Q4_K_M|hotpot  | Fixed 40%               |         0.099 | [0.009, 0.205]   |       0.125 |    1     |
| 3B / Q4_K_M|hotpot  | Fixed 20%               |         0.095 | [0.054, 0.137]   |       0     |    0.001 |
| 3B / Q4_K_M|hotpot  | LLMLingua-2 (50%)       |         0.082 | [-0.002, 0.180]  |       0.25  |    1     |
| 3B / Q4_K_M|hotpot  | LLMLingua-2 (20%)       |         0.002 | [-0.139, 0.140]  |       0.375 |    1     |
| 3B / Q4_K_M|hotpot  | Relevance top-m (iso)   |         0.076 | [0.036, 0.119]   |       0.001 |    0.015 |
| 3B / Q4_K_M|hotpot  | DECAF no-cov (λ=6)      |         0.068 | [0.028, 0.108]   |       0     |    0.005 |
| 3B / Q4_K_M|hotpot  | DECAF no-reorder (λ=6)  |         0.068 | [-0.042, 0.189]  |       0.125 |    1     |
| 3B / Q4_K_M|hotpot  | DECAF λ=6 (BM25 scorer) |         0.069 | [-0.026, 0.176]  |       0.125 |    1     |
| 3B / Q4_K_M|hotpot  | DECAF λ=2               |         0.065 | [-0.018, 0.163]  |       0.25  |    1     |
| 3B / Q4_K_M|hotpot  | DECAF λ=6               |         0.077 | [0.036, 0.119]   |       0.001 |    0.009 |
| 3B / Q4_K_M|hotpot  | DECAF λ=15              |         0.101 | [-0.009, 0.225]  |       0.062 |    0.562 |
| 3B / Q4_K_M|twowiki | Fixed 80%               |        -0.021 | [-0.096, 0.053]  |       1     |    1     |
| 3B / Q4_K_M|twowiki | Fixed 60%               |        -0.07  | [-0.169, 0.027]  |       0.625 |    1     |
| 3B / Q4_K_M|twowiki | Fixed 50%               |         0.015 | [-0.017, 0.049]  |       0.18  |    1     |
| 3B / Q4_K_M|twowiki | Fixed 40%               |         0.03  | [-0.023, 0.105]  |       1     |    1     |
| 3B / Q4_K_M|twowiki | Fixed 20%               |         0.036 | [-0.002, 0.075]  |       0.031 |    0.432 |
| 3B / Q4_K_M|twowiki | LLMLingua-2 (50%)       |        -0.043 | [-0.135, 0.048]  |       1     |    1     |
| 3B / Q4_K_M|twowiki | LLMLingua-2 (20%)       |         0.003 | [-0.106, 0.121]  |       0.625 |    1     |
| 3B / Q4_K_M|twowiki | Relevance top-m (iso)   |         0     | [-0.039, 0.039]  |       0.804 |    1     |
| 3B / Q4_K_M|twowiki | DECAF no-cov (λ=6)      |         0.012 | [-0.028, 0.051]  |       0.21  |    1     |
| 3B / Q4_K_M|twowiki | DECAF no-reorder (λ=6)  |         0.086 | [-0.043, 0.218]  |       0.375 |    1     |
| 3B / Q4_K_M|twowiki | DECAF λ=6 (BM25 scorer) |         0.068 | [-0.083, 0.219]  |       0.453 |    1     |
| 3B / Q4_K_M|twowiki | DECAF λ=2               |         0.096 | [-0.033, 0.223]  |       0.375 |    1     |
| 3B / Q4_K_M|twowiki | DECAF λ=6               |        -0     | [-0.039, 0.039]  |       0.607 |    1     |
| 3B / Q4_K_M|twowiki | DECAF λ=15              |         0.038 | [-0.109, 0.194]  |       1     |    1     |
| 3B / Q8_0|hotpot    | Fixed 80%               |         0.031 | [-0.014, 0.107]  |       1     |    1     |
| 3B / Q8_0|hotpot    | Fixed 60%               |         0.091 | [0.004, 0.204]   |       0.25  |    1     |
| 3B / Q8_0|hotpot    | Fixed 50%               |         0.037 | [0.012, 0.064]   |       0.146 |    1     |
| 3B / Q8_0|hotpot    | Fixed 40%               |         0.058 | [0.002, 0.133]   |       0.5   |    1     |
| 3B / Q8_0|hotpot    | Fixed 20%               |         0.053 | [0.019, 0.089]   |       0.001 |    0.018 |
| 3B / Q8_0|hotpot    | LLMLingua-2 (50%)       |        -0.004 | [-0.038, 0.028]  |       1     |    1     |
| 3B / Q8_0|hotpot    | LLMLingua-2 (20%)       |        -0.083 | [-0.180, 0.009]  |       1     |    1     |
| 3B / Q8_0|hotpot    | Relevance top-m (iso)   |         0.029 | [-0.005, 0.065]  |       0.146 |    1     |
| 3B / Q8_0|hotpot    | DECAF no-cov (λ=6)      |         0.037 | [0.003, 0.075]   |       0.013 |    0.168 |
| 3B / Q8_0|hotpot    | DECAF no-reorder (λ=6)  |         0.093 | [-0.023, 0.215]  |       0.125 |    1     |
| 3B / Q8_0|hotpot    | DECAF λ=6 (BM25 scorer) |         0.099 | [-0.002, 0.223]  |       0.125 |    1     |
| 3B / Q8_0|hotpot    | DECAF λ=2               |         0.112 | [0.029, 0.216]   |       0.125 |    1     |
| 3B / Q8_0|hotpot    | DECAF λ=6               |         0.027 | [-0.007, 0.062]  |       0.092 |    1     |
| 3B / Q8_0|hotpot    | DECAF λ=15              |         0.069 | [-0.062, 0.205]  |       0.219 |    1     |
| 3B / Q8_0|twowiki   | Fixed 80%               |        -0.031 | [-0.106, 0.024]  |       1     |    1     |
| 3B / Q8_0|twowiki   | Fixed 60%               |        -0.018 | [-0.108, 0.063]  |       1     |    1     |
| 3B / Q8_0|twowiki   | Fixed 50%               |         0.03  | [-0.001, 0.063]  |       0.039 |    0.54  |
| 3B / Q8_0|twowiki   | Fixed 40%               |         0.012 | [-0.097, 0.123]  |       1     |    1     |
| 3B / Q8_0|twowiki   | Fixed 20%               |         0.017 | [-0.017, 0.052]  |       0.581 |    1     |
| 3B / Q8_0|twowiki   | LLMLingua-2 (50%)       |        -0.028 | [-0.111, 0.037]  |       1     |    1     |
| 3B / Q8_0|twowiki   | LLMLingua-2 (20%)       |        -0.051 | [-0.119, 0.009]  |       1     |    1     |
| 3B / Q8_0|twowiki   | Relevance top-m (iso)   |        -0.028 | [-0.057, 0.001]  |       0.289 |    1     |
| 3B / Q8_0|twowiki   | DECAF no-cov (λ=6)      |        -0.011 | [-0.044, 0.019]  |       1     |    1     |
| 3B / Q8_0|twowiki   | DECAF no-reorder (λ=6)  |        -0.065 | [-0.171, 0.027]  |       0.25  |    1     |
| 3B / Q8_0|twowiki   | DECAF λ=6 (BM25 scorer) |        -0.039 | [-0.171, 0.085]  |       1     |    1     |
| 3B / Q8_0|twowiki   | DECAF λ=2               |        -0.062 | [-0.176, 0.040]  |       0.25  |    1     |
| 3B / Q8_0|twowiki   | DECAF λ=6               |        -0.032 | [-0.061, -0.002] |       0.289 |    1     |
| 3B / Q8_0|twowiki   | DECAF λ=15              |        -0.048 | [-0.158, 0.065]  |       0.625 |    1     |
| 7B / Q4_K_M|hotpot  | Fixed 80%               |        -0.031 | [-0.083, 0.001]  |       1     |    1     |
| 7B / Q4_K_M|hotpot  | Fixed 60%               |        -0.011 | [-0.113, 0.105]  |       1     |    1     |
| 7B / Q4_K_M|hotpot  | Fixed 50%               |        -0.019 | [-0.056, 0.022]  |       0.359 |    1     |
| 7B / Q4_K_M|hotpot  | Fixed 40%               |        -0.094 | [-0.212, 0.014]  |       0.125 |    1     |
| 7B / Q4_K_M|hotpot  | Fixed 20%               |        -0.055 | [-0.098, -0.011] |       0.007 |    0.066 |
| 7B / Q4_K_M|hotpot  | LLMLingua-2 (50%)       |        -0.023 | [-0.148, 0.087]  |       0.453 |    1     |
| 7B / Q4_K_M|hotpot  | LLMLingua-2 (20%)       |        -0.35  | [-0.505, -0.208] |       0.001 |    0.014 |
| 7B / Q4_K_M|hotpot  | Relevance top-m (iso)   |        -0.092 | [-0.152, -0.037] |       0.005 |    0.056 |
| 7B / Q4_K_M|hotpot  | DECAF no-cov (λ=6)      |        -0.085 | [-0.141, -0.032] |       0.002 |    0.025 |
| 7B / Q4_K_M|hotpot  | DECAF no-reorder (λ=6)  |        -0.094 | [-0.245, 0.040]  |       0.219 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF λ=6 (BM25 scorer) |        -0.076 | [-0.225, 0.066]  |       0.375 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF λ=2               |        -0.082 | [-0.231, 0.047]  |       0.375 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF λ=6               |        -0.092 | [-0.151, -0.037] |       0.001 |    0.019 |
| 7B / Q4_K_M|hotpot  | DECAF λ=15              |        -0.261 | [-0.451, -0.073] |       0.039 |    0.352 |
| 7B / Q4_K_M|twowiki | Fixed 80%               |         0.03  | [-0.048, 0.113]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | Fixed 60%               |         0.011 | [-0.099, 0.120]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | Fixed 50%               |         0.012 | [-0.023, 0.044]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | Fixed 40%               |        -0.008 | [-0.104, 0.081]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | Fixed 20%               |         0.009 | [-0.031, 0.050]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | LLMLingua-2 (50%)       |        -0.027 | [-0.108, 0.043]  |       0.5   |    1     |
| 7B / Q4_K_M|twowiki | LLMLingua-2 (20%)       |        -0.039 | [-0.179, 0.089]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | Relevance top-m (iso)   |        -0.011 | [-0.055, 0.032]  |       0.69  |    1     |
| 7B / Q4_K_M|twowiki | DECAF no-cov (λ=6)      |        -0.01  | [-0.057, 0.036]  |       0.839 |    1     |
| 7B / Q4_K_M|twowiki | DECAF no-reorder (λ=6)  |         0.015 | [-0.144, 0.173]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | DECAF λ=6 (BM25 scorer) |        -0.091 | [-0.227, 0.033]  |       0.688 |    1     |
| 7B / Q4_K_M|twowiki | DECAF λ=2               |        -0.043 | [-0.186, 0.082]  |       0.375 |    1     |
| 7B / Q4_K_M|twowiki | DECAF λ=6               |        -0.023 | [-0.067, 0.021]  |       0.678 |    1     |
| 7B / Q4_K_M|twowiki | DECAF λ=15              |        -0.129 | [-0.279, 0.000]  |       0.031 |    0.438 |
| 7B / Q8_0|hotpot    | Fixed 80%               |        -0.033 | [-0.089, 0.007]  |       1     |    1     |
| 7B / Q8_0|hotpot    | Fixed 60%               |        -0.046 | [-0.135, 0.021]  |       0.5   |    1     |
| 7B / Q8_0|hotpot    | Fixed 50%               |         0.014 | [-0.019, 0.048]  |       1     |    1     |
| 7B / Q8_0|hotpot    | Fixed 40%               |        -0.077 | [-0.180, 0.004]  |       0.25  |    1     |
| 7B / Q8_0|hotpot    | Fixed 20%               |        -0.024 | [-0.067, 0.021]  |       0.122 |    1     |
| 7B / Q8_0|hotpot    | LLMLingua-2 (50%)       |        -0.07  | [-0.150, 0.003]  |       0.375 |    1     |
| 7B / Q8_0|hotpot    | LLMLingua-2 (20%)       |        -0.379 | [-0.537, -0.235] |       0.001 |    0.014 |
| 7B / Q8_0|hotpot    | Relevance top-m (iso)   |        -0.063 | [-0.119, -0.015] |       0.009 |    0.099 |
| 7B / Q8_0|hotpot    | DECAF no-cov (λ=6)      |        -0.057 | [-0.113, -0.004] |       0.014 |    0.135 |
| 7B / Q8_0|hotpot    | DECAF no-reorder (λ=6)  |        -0.089 | [-0.234, 0.046]  |       0.453 |    1     |
| 7B / Q8_0|hotpot    | DECAF λ=6 (BM25 scorer) |        -0.072 | [-0.187, 0.035]  |       0.625 |    1     |
| 7B / Q8_0|hotpot    | DECAF λ=2               |        -0.064 | [-0.186, 0.060]  |       0.375 |    1     |
| 7B / Q8_0|hotpot    | DECAF λ=6               |        -0.064 | [-0.118, -0.014] |       0.004 |    0.051 |
| 7B / Q8_0|hotpot    | DECAF λ=15              |        -0.253 | [-0.410, -0.105] |       0.008 |    0.094 |
| 7B / Q8_0|twowiki   | Fixed 80%               |        -0.002 | [-0.080, 0.054]  |       1     |    1     |
| 7B / Q8_0|twowiki   | Fixed 60%               |         0.062 | [-0.046, 0.186]  |       1     |    1     |
| 7B / Q8_0|twowiki   | Fixed 50%               |         0.025 | [-0.013, 0.063]  |       0.383 |    1     |
| 7B / Q8_0|twowiki   | Fixed 40%               |         0.039 | [-0.092, 0.170]  |       1     |    1     |
| 7B / Q8_0|twowiki   | Fixed 20%               |         0.031 | [-0.013, 0.075]  |       0.557 |    1     |
| 7B / Q8_0|twowiki   | LLMLingua-2 (50%)       |         0.007 | [-0.053, 0.058]  |       1     |    1     |
| 7B / Q8_0|twowiki   | LLMLingua-2 (20%)       |         0.017 | [-0.163, 0.190]  |       1     |    1     |
| 7B / Q8_0|twowiki   | Relevance top-m (iso)   |        -0.008 | [-0.060, 0.043]  |       1     |    1     |
| 7B / Q8_0|twowiki   | DECAF no-cov (λ=6)      |         0.019 | [-0.031, 0.064]  |       0.345 |    1     |
| 7B / Q8_0|twowiki   | DECAF no-reorder (λ=6)  |         0.03  | [-0.126, 0.191]  |       1     |    1     |
| 7B / Q8_0|twowiki   | DECAF λ=6 (BM25 scorer) |        -0.055 | [-0.178, 0.054]  |       1     |    1     |
| 7B / Q8_0|twowiki   | DECAF λ=2               |        -0.004 | [-0.147, 0.136]  |       1     |    1     |
| 7B / Q8_0|twowiki   | DECAF λ=6               |        -0.02  | [-0.071, 0.028]  |       1     |    1     |
| 7B / Q8_0|twowiki   | DECAF λ=15              |        -0.076 | [-0.251, 0.093]  |       0.727 |    1     |

Figures: `fig_v2_stats_forest.png`. Full numbers in `summary_v2.json` under `stats`.

## 6. Secondary systems analysis (H5: prefix/KV reuse + prefetch)

| run                |   cold TTFT (ms) |   warm TTFT (ms) |   partial TTFT (ms) |   warm ΔTTFT % |
|:-------------------|-----------------:|-----------------:|--------------------:|---------------:|
| 3B / Q4_K_M|L=2048 |             8399 |               54 |                 169 |           99.4 |
| 3B / Q4_K_M|L=8192 |            39693 |               87 |                 289 |           99.8 |
| 3B / Q8_0|L=2048   |             7996 |               79 |                 166 |           99   |
| 3B / Q8_0|L=8192   |            38154 |              118 |                 285 |           99.7 |
| 7B / Q4_K_M|L=2048 |            18668 |              116 |                 333 |           99.4 |
| 7B / Q4_K_M|L=8192 |            81780 |              147 |                 486 |           99.8 |
| 7B / Q8_0|L=2048   |            19122 |              175 |                 310 |           99.1 |
| 7B / Q8_0|L=8192   |            82527 |              205 |                 465 |           99.8 |

Figures: `fig_v2_h5_prefix.png`, `fig_v2_h5_reuse.png`, `fig_v2_h5_prefetch.png`.

## 7. Retrieval modes (Sec.13: BM25 vs dense vs hybrid)

| run                        |    F1 |   gold-in-ev |   retained |   E2E(ms) |
|:---------------------------|------:|-------------:|-----------:|----------:|
| 3B / Q4_K_M|hotpot|bm25    | 0.262 |        0.515 |      0.12  |      1058 |
| 3B / Q4_K_M|hotpot|dense   | 0.3   |        0.7   |      0.139 |      1060 |
| 3B / Q4_K_M|hotpot|hybrid  | 0.31  |        0.733 |      0.143 |       879 |
| 3B / Q4_K_M|twowiki|bm25   | 0.187 |        0.4   |      0.115 |       961 |
| 3B / Q4_K_M|twowiki|dense  | 0.275 |        0.567 |      0.115 |      1090 |
| 3B / Q4_K_M|twowiki|hybrid | 0.304 |        0.567 |      0.108 |       882 |
| 7B / Q4_K_M|hotpot|bm25    | 0.443 |        0.515 |      0.12  |       874 |
| 7B / Q4_K_M|hotpot|dense   | 0.587 |        0.7   |      0.139 |       855 |
| 7B / Q4_K_M|hotpot|hybrid  | 0.589 |        0.733 |      0.143 |       706 |
| 7B / Q4_K_M|twowiki|bm25   | 0.228 |        0.4   |      0.115 |      1476 |
| 7B / Q4_K_M|twowiki|dense  | 0.263 |        0.567 |      0.115 |      1614 |
| 7B / Q4_K_M|twowiki|hybrid | 0.315 |        0.567 |      0.108 |      1744 |

Figures: `fig_v2_retrieval.png`, `fig_v2_retrieval_pareto.png`.

## 8. Figure index

- `fig_v2_scaling_tpot.png`
- `fig_v2_scaling_prefill.png`
- `fig_v2_scaling_ttft.png`
- `fig_v2_scaling_e2e.png`
- `fig_v2_scaling_prefill_per_tok.png`
- `fig_v2_scaling_decode_ms.png`
- `fig_v2_scaling_decode_share.png`
- `fig_v2_scaling_peak_rss.png`
- `fig_v2_scaling_panel.png`
- `fig_v2_crossover_decode_share.png`
- `fig_v2_crossover_length.png`
- `fig_v2_h5_prefix.png`
- `fig_v2_h5_reuse.png`
- `fig_v2_retrieval.png`
- `fig_v2_retrieval_pareto.png`
- `fig_v2_breakeven_curves.png`
- `fig_v2_Lstar_bar.png`
- `fig_v2_pareto_f1_e2e.png`
- `fig_v2_pareto_f1_ttft.png`
- `fig_v2_pareto_f1_retained.png`
- `fig_v2_pareto_all.png`
- `fig_v2_grounding_vs_ratio.png`
- `fig_v2_quality_vs_grounding.png`
- `fig_v2_f1_by_method.png`
- `fig_v2_speedup_bar.png`
- `fig_v2_latency_breakdown.png`
- `fig_v2_ev_recall_prec.png`
- `fig_v2_ce_cost.png`
- `fig_v2_qe_bar.png`
- `fig_v2_ttft_vs_tpot.png`
- `fig_v2_overhead_vs_savings.png`
- `fig_v2_pareto_frontier.png`
- `fig_v2_h5_prefetch.png`
- `fig_v2_method_model_heatmap.png`
- `fig_v2_dataset_compare.png`
- `fig_v2_e2e_ci.png`
- `fig_v2_lambda_sweep.png`
- `fig_v2_ablation_bars.png`
- `fig_v2_ratio_grid.png`
- `fig_v2_scorer_ablation.png`
- `fig_v2_stats_forest.png`
