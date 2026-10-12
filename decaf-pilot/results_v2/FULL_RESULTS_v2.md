# DECAF — Full Results v2

CPU-only, Ollama/llama.cpp · models: 3B / Q4_K_M, 3B / Q8_0, 7B / Q4_K_M · datasets: hotpot, twowiki · 36 scaling, 27 crossover, 1620 compression runs · cross-encoder scorer `BAAI/bge-reranker-base`.

> v2 supersedes the single-model pilot: quantization (E5) + model-size (E6) pairs, a cross-encoder scorer, long-context scaling to 8K, a generation-length crossover (H1), 2Wiki generalization (E9), and paired statistics (bootstrap CI, McNemar, Holm).

## 0. Headline findings

- **3B / Q4_K_M**: prefill grows 4.49 ms/token (r²=0.997) while TPOT barely moves (+0.90 ms per 1k tokens). Decode = prefill at **L ≈ 535 tokens** (32-token outputs) → **prefill dominates above it (H1 refuted on this setup)**.
- **3B / Q8_0**: prefill grows 4.56 ms/token (r²=0.997) while TPOT barely moves (+0.96 ms per 1k tokens). Decode = prefill at **L ≈ 690 tokens** (32-token outputs) → **prefill dominates above it (H1 refuted on this setup)**.
- **7B / Q4_K_M**: prefill grows 10.13 ms/token (r²=0.999) while TPOT barely moves (+1.58 ms per 1k tokens). Decode = prefill at **L ≈ 448 tokens** (32-token outputs) → **prefill dominates above it (H1 refuted on this setup)**.
- **Compressor cost T_compress ≈ 962 ms** (cross-encoder `bge-reranker-base`, CPU): ~3 orders of magnitude above the pilot's BM25 (~1.4 ms) and no longer negligible — this is the cost the proposal warns about.
- **3B / Q4_K_M**: retained L_c≈154 tok, break-even **L*≈367 tokens** (compression is a net latency loss below, a net win above).
- **3B / Q8_0**: retained L_c≈154 tok, break-even **L*≈363 tokens** (compression is a net latency loss below, a net win above).
- **7B / Q4_K_M**: retained L_c≈154 tok, break-even **L*≈248 tokens** (compression is a net latency loss below, a net win above).
- **Overhead matters**: 3B / Q4_K_M·hotpot DECAF(decaf_lam6): gross speedup 3.66× falls to **S_net 1.89×** once the 989 ms cross-encoder cost is counted (S_net = T_full / (T_compress + T_e2e)).
- **hotpot**: best F1 = 0.595 (7B / Q4_K_M · Full context). DECAF(decaf_lam6) F1 → 7B / Q4_K_M:0.522, 3B / Q8_0:0.330, 3B / Q4_K_M:0.291.
- **twowiki**: best F1 = 0.336 (7B / Q4_K_M · DECAF no-reorder (λ=6)). DECAF(decaf_lam6) F1 → 3B / Q4_K_M:0.281, 7B / Q4_K_M:0.274, 3B / Q8_0:0.151.

## 1. Decoding scaling (E1/E8, per model)

| model       |   TPOT slope (ms/1k/L) |   prefill slope (ms/tok) |   prefill r² |   decode/prefill cross L |   N_out |
|:------------|-----------------------:|-------------------------:|-------------:|-------------------------:|--------:|
| 3B / Q4_K_M |                  0.901 |                   4.4853 |        0.997 |                      535 |      32 |
| 3B / Q8_0   |                  0.961 |                   4.5557 |        0.997 |                      690 |      32 |
| 7B / Q4_K_M |                  1.58  |                  10.1319 |        0.999 |                      448 |      32 |

Figures: `fig_v2_scaling_tpot.png`, `fig_v2_scaling_prefill.png`, `fig_v2_scaling_ttft.png`, `fig_v2_scaling_e2e.png`, `fig_v2_scaling_decode_share.png`, `fig_v2_scaling_prefill_per_tok.png`, `fig_v2_scaling_decode_ms.png`, `fig_v2_scaling_peak_rss.png`, `fig_v2_scaling_panel.png`, `fig_v2_crossover_decode_share.png`, `fig_v2_crossover_length.png`.

## 2. Break-even L* per model (E4)

| model       |   L_c (tok) |   T_compress (ms) |   prefill+decode slope (ms/tok) |   L* (tok) |
|:------------|------------:|------------------:|--------------------------------:|-----------:|
| 3B / Q4_K_M |         154 |             962.4 |                           4.514 |        367 |
| 3B / Q8_0   |         154 |             962.5 |                           4.586 |        363 |
| 7B / Q4_K_M |         154 |             962.8 |                          10.182 |        248 |

Figures: `fig_v2_breakeven_curves.png`, `fig_v2_Lstar_bar.png`, `fig_v2_overhead_vs_savings.png`.

## 3. Compression: quality / latency / overhead / grounding

### hotpot

| model       | method                 |    F1 |    EM |   ev.rec |   gold-in-ev |   retained |   T_c(ms) |   E2E(ms) |   speedup× |   S_net× |
|:------------|:-----------------------|------:|------:|---------:|-------------:|-----------:|----------:|----------:|-----------:|---------:|
| 3B / Q4_K_M | DECAF λ=15             | 0.3   | 0.233 |    0.356 |        0.467 |      0.055 |   988.838 |  1049.95  |      3.673 |    1.892 |
| 3B / Q4_K_M | DECAF λ=2              | 0.264 | 0.167 |    0.614 |        0.667 |      0.208 |   988.735 |  1127.84  |      3.419 |    1.822 |
| 3B / Q4_K_M | DECAF λ=6              | 0.291 | 0.2   |    0.572 |        0.633 |      0.128 |   989.115 |  1054.27  |      3.658 |    1.887 |
| 3B / Q4_K_M | DECAF no-cov (λ=6)     | 0.285 | 0.2   |    0.606 |        0.633 |      0.172 |   988.812 |  1031.18  |      3.74  |    1.909 |
| 3B / Q4_K_M | DECAF no-reorder (λ=6) | 0.267 | 0.2   |    0.572 |        0.633 |      0.128 |   988.719 |  1169.8   |      3.297 |    1.787 |
| 3B / Q4_K_M | Fixed 20%              | 0.317 | 0.233 |    0.618 |        0.667 |      0.227 |   988.547 |   986.35  |      3.91  |    1.953 |
| 3B / Q4_K_M | Fixed 50%              | 0.33  | 0.233 |    0.657 |        0.733 |      0.553 |   988.829 |  1137.18  |      3.391 |    1.814 |
| 3B / Q4_K_M | Full context           | 0.199 | 0.067 |    0.751 |        0.833 |      1     |   988.5   |  3856.55  |      1     |    1     |
| 3B / Q4_K_M | Relevance top-m (iso)  | 0.295 | 0.2   |    0.583 |        0.633 |      0.139 |   988.579 |  1074.65  |      3.589 |    1.869 |
| 3B / Q8_0   | DECAF λ=15             | 0.249 | 0.2   |    0.356 |        0.467 |      0.055 |   988.929 |  1879.81  |      2.311 |    1.514 |
| 3B / Q8_0   | DECAF λ=2              | 0.293 | 0.2   |    0.614 |        0.667 |      0.208 |   989.241 |  1631.43  |      2.662 |    1.657 |
| 3B / Q8_0   | DECAF λ=6              | 0.33  | 0.233 |    0.572 |        0.633 |      0.128 |   988.919 |  1582.16  |      2.745 |    1.689 |
| 3B / Q8_0   | DECAF no-cov (λ=6)     | 0.322 | 0.233 |    0.606 |        0.633 |      0.172 |   989.036 |  1529.06  |      2.841 |    1.725 |
| 3B / Q8_0   | DECAF no-reorder (λ=6) | 0.273 | 0.2   |    0.572 |        0.633 |      0.128 |   988.806 |  1920.26  |      2.262 |    1.493 |
| 3B / Q8_0   | Fixed 20%              | 0.282 | 0.2   |    0.618 |        0.667 |      0.227 |   988.87  |  1815.29  |      2.393 |    1.549 |
| 3B / Q8_0   | Fixed 50%              | 0.266 | 0.167 |    0.657 |        0.733 |      0.553 |   988.652 |  1814.33  |      2.394 |    1.55  |
| 3B / Q8_0   | Full context           | 0.181 | 0.067 |    0.751 |        0.833 |      1     |   988.533 |  4343.39  |      1     |    1     |
| 3B / Q8_0   | Relevance top-m (iso)  | 0.332 | 0.233 |    0.583 |        0.633 |      0.139 |   988.61  |  1430.36  |      3.037 |    1.796 |
| 7B / Q4_K_M | DECAF λ=15             | 0.335 | 0.3   |    0.356 |        0.467 |      0.055 |   988.941 |  1542.44  |      5.327 |    3.246 |
| 7B / Q4_K_M | DECAF λ=2              | 0.513 | 0.433 |    0.614 |        0.667 |      0.208 |   988.856 |  1509.54  |      5.443 |    3.289 |
| 7B / Q4_K_M | DECAF λ=6              | 0.522 | 0.433 |    0.572 |        0.633 |      0.128 |   989.163 |   837.398 |      9.812 |    4.499 |
| 7B / Q4_K_M | DECAF no-cov (λ=6)     | 0.503 | 0.433 |    0.606 |        0.633 |      0.172 |   989.126 |  1611.35  |      5.099 |    3.16  |
| 7B / Q4_K_M | DECAF no-reorder (λ=6) | 0.501 | 0.4   |    0.572 |        0.633 |      0.128 |   989.466 |  1576.64  |      5.212 |    3.202 |
| 7B / Q4_K_M | Fixed 20%              | 0.512 | 0.433 |    0.618 |        0.667 |      0.227 |   988.895 |   938.882 |      8.752 |    4.262 |
| 7B / Q4_K_M | Fixed 50%              | 0.578 | 0.5   |    0.657 |        0.733 |      0.553 |   988.748 |   844.605 |      9.729 |    4.482 |
| 7B / Q4_K_M | Full context           | 0.595 | 0.533 |    0.751 |        0.833 |      1     |   988.552 |  8216.93  |      1     |    1     |
| 7B / Q4_K_M | Relevance top-m (iso)  | 0.539 | 0.467 |    0.583 |        0.633 |      0.139 |   988.74  |   773.73  |     10.62  |    4.662 |

### twowiki

| model       | method                 |    F1 |    EM |   ev.rec |   gold-in-ev |   retained |   T_c(ms) |   E2E(ms) |   speedup× |   S_net× |
|:------------|:-----------------------|------:|------:|---------:|-------------:|-----------:|----------:|----------:|-----------:|---------:|
| 3B / Q4_K_M | DECAF λ=15             | 0.251 | 0.167 |    0.325 |        0.4   |      0.084 |   897.186 |   984.222 |      4.088 |    2.139 |
| 3B / Q4_K_M | DECAF λ=2              | 0.309 | 0.233 |    0.533 |        0.7   |      0.156 |   897.227 |  1079.24  |      3.728 |    2.036 |
| 3B / Q4_K_M | DECAF λ=6              | 0.281 | 0.167 |    0.458 |        0.533 |      0.102 |   897.337 |   957.603 |      4.202 |    2.169 |
| 3B / Q4_K_M | DECAF no-cov (λ=6)     | 0.295 | 0.2   |    0.508 |        0.667 |      0.144 |   897.138 |  1094.48  |      3.677 |    2.02  |
| 3B / Q4_K_M | DECAF no-reorder (λ=6) | 0.299 | 0.233 |    0.458 |        0.533 |      0.102 |   897.132 |  1110.53  |      3.623 |    2.004 |
| 3B / Q4_K_M | Fixed 20%              | 0.333 | 0.233 |    0.542 |        0.7   |      0.275 |   897.225 |  1065.85  |      3.775 |    2.05  |
| 3B / Q4_K_M | Fixed 50%              | 0.192 | 0.1   |    0.608 |        0.767 |      0.558 |   897.053 |  1162.74  |      3.461 |    1.954 |
| 3B / Q4_K_M | Full context           | 0.213 | 0.133 |    0.617 |        0.767 |      1     |   896.965 |  4023.93  |      1     |    1     |
| 3B / Q4_K_M | Relevance top-m (iso)  | 0.314 | 0.2   |    0.45  |        0.533 |      0.11  |   897.199 |   991.246 |      4.059 |    2.131 |
| 3B / Q8_0   | DECAF λ=15             | 0.152 | 0.033 |    0.325 |        0.4   |      0.084 |   897.234 |  1891.03  |      2.233 |    1.515 |
| 3B / Q8_0   | DECAF λ=2              | 0.138 | 0     |    0.533 |        0.7   |      0.156 |   897.215 |  1923.94  |      2.195 |    1.497 |
| 3B / Q8_0   | DECAF λ=6              | 0.151 | 0.033 |    0.458 |        0.533 |      0.102 |   897.229 |  1879.79  |      2.247 |    1.521 |
| 3B / Q8_0   | DECAF no-cov (λ=6)     | 0.13  | 0     |    0.508 |        0.667 |      0.144 |   897.279 |  1956.25  |      2.159 |    1.48  |
| 3B / Q8_0   | DECAF no-reorder (λ=6) | 0.136 | 0     |    0.458 |        0.533 |      0.102 |   897.21  |  1933.91  |      2.184 |    1.492 |
| 3B / Q8_0   | Fixed 20%              | 0.211 | 0.1   |    0.542 |        0.7   |      0.275 |   897.194 |  1936.19  |      2.181 |    1.49  |
| 3B / Q8_0   | Fixed 50%              | 0.233 | 0.133 |    0.608 |        0.767 |      0.558 |   897.43  |  1947.75  |      2.168 |    1.484 |
| 3B / Q8_0   | Full context           | 0.2   | 0.1   |    0.617 |        0.767 |      1     |   896.994 |  4223.15  |      1     |    1     |
| 3B / Q8_0   | Relevance top-m (iso)  | 0.153 | 0.033 |    0.45  |        0.533 |      0.11  |   897.129 |  1907.38  |      2.214 |    1.506 |
| 7B / Q4_K_M | DECAF λ=15             | 0.193 | 0.033 |    0.325 |        0.4   |      0.084 |   897.266 |  1564.49  |      5.317 |    3.379 |
| 7B / Q4_K_M | DECAF λ=2              | 0.279 | 0.133 |    0.533 |        0.7   |      0.156 |   897.306 |  2111.67  |      3.94  |    2.765 |
| 7B / Q4_K_M | DECAF λ=6              | 0.274 | 0.133 |    0.458 |        0.533 |      0.102 |   897.572 |  1501.32  |      5.541 |    3.468 |
| 7B / Q4_K_M | DECAF no-cov (λ=6)     | 0.271 | 0.133 |    0.508 |        0.667 |      0.144 |   897.35  |  2152.66  |      3.865 |    2.728 |
| 7B / Q4_K_M | DECAF no-reorder (λ=6) | 0.336 | 0.2   |    0.458 |        0.533 |      0.102 |   897.394 |  2046.78  |      4.064 |    2.826 |
| 7B / Q4_K_M | Fixed 20%              | 0.246 | 0.1   |    0.542 |        0.7   |      0.275 |   897.255 |  1599.57  |      5.201 |    3.332 |
| 7B / Q4_K_M | Fixed 50%              | 0.297 | 0.2   |    0.608 |        0.767 |      0.558 |   897.261 |  1877.82  |      4.43  |    2.998 |
| 7B / Q4_K_M | Full context           | 0.322 | 0.233 |    0.617 |        0.767 |      1     |   897.069 |  8318.96  |      1     |    1     |
| 7B / Q4_K_M | Relevance top-m (iso)  | 0.274 | 0.133 |    0.45  |        0.533 |      0.11  |   897.18  |  1590.96  |      5.229 |    3.343 |

Figures: `fig_v2_pareto_f1_e2e.png`, `fig_v2_pareto_f1_ttft.png`, `fig_v2_pareto_f1_retained.png`, `fig_v2_pareto_all.png`, `fig_v2_pareto_frontier.png`, `fig_v2_ttft_vs_tpot.png`, `fig_v2_grounding_vs_ratio.png`, `fig_v2_quality_vs_grounding.png`, `fig_v2_f1_by_method.png`, `fig_v2_speedup_bar.png`, `fig_v2_latency_breakdown.png`, `fig_v2_ev_recall_prec.png`, `fig_v2_ce_cost.png`, `fig_v2_qe_bar.png`, `fig_v2_method_model_heatmap.png`, `fig_v2_e2e_ci.png`, `fig_v2_dataset_compare.png`.

## 4. λ sweep and component ablation (Sec. 17)

| dataset   | model       |   λ |    F1 |   retained |   gold-in-ev |   E2E(ms) |
|:----------|:------------|----:|------:|-----------:|-------------:|----------:|
| hotpot    | 3B / Q4_K_M |   2 | 0.264 |      0.208 |        0.667 |      1128 |
| hotpot    | 3B / Q4_K_M |   6 | 0.291 |      0.128 |        0.633 |      1054 |
| hotpot    | 3B / Q4_K_M |  15 | 0.3   |      0.055 |        0.467 |      1050 |
| hotpot    | 3B / Q8_0   |   2 | 0.293 |      0.208 |        0.667 |      1631 |
| hotpot    | 3B / Q8_0   |   6 | 0.33  |      0.128 |        0.633 |      1582 |
| hotpot    | 3B / Q8_0   |  15 | 0.249 |      0.055 |        0.467 |      1880 |
| hotpot    | 7B / Q4_K_M |   2 | 0.513 |      0.208 |        0.667 |      1510 |
| hotpot    | 7B / Q4_K_M |   6 | 0.522 |      0.128 |        0.633 |       837 |
| hotpot    | 7B / Q4_K_M |  15 | 0.335 |      0.055 |        0.467 |      1542 |
| twowiki   | 3B / Q4_K_M |   2 | 0.309 |      0.156 |        0.7   |      1079 |
| twowiki   | 3B / Q4_K_M |   6 | 0.281 |      0.102 |        0.533 |       958 |
| twowiki   | 3B / Q4_K_M |  15 | 0.251 |      0.084 |        0.4   |       984 |
| twowiki   | 3B / Q8_0   |   2 | 0.138 |      0.156 |        0.7   |      1924 |
| twowiki   | 3B / Q8_0   |   6 | 0.151 |      0.102 |        0.533 |      1880 |
| twowiki   | 3B / Q8_0   |  15 | 0.152 |      0.084 |        0.4   |      1891 |
| twowiki   | 7B / Q4_K_M |   2 | 0.279 |      0.156 |        0.7   |      2112 |
| twowiki   | 7B / Q4_K_M |   6 | 0.274 |      0.102 |        0.533 |      1501 |
| twowiki   | 7B / Q4_K_M |  15 | 0.193 |      0.084 |        0.4   |      1564 |

Ablation (coverage / reorder / CPU-relevance-only vs full DECAF λ6):

| dataset   | model       | component              |    F1 |   ev.rec |   gold-in-ev |   retained |
|:----------|:------------|:-----------------------|------:|---------:|-------------:|-----------:|
| hotpot    | 3B / Q4_K_M | DECAF λ=6              | 0.291 |    0.572 |        0.633 |      0.128 |
| hotpot    | 3B / Q4_K_M | DECAF no-cov (λ=6)     | 0.285 |    0.606 |        0.633 |      0.172 |
| hotpot    | 3B / Q4_K_M | DECAF no-reorder (λ=6) | 0.267 |    0.572 |        0.633 |      0.128 |
| hotpot    | 3B / Q4_K_M | Relevance top-m (iso)  | 0.295 |    0.583 |        0.633 |      0.139 |
| hotpot    | 3B / Q8_0   | DECAF λ=6              | 0.33  |    0.572 |        0.633 |      0.128 |
| hotpot    | 3B / Q8_0   | DECAF no-cov (λ=6)     | 0.322 |    0.606 |        0.633 |      0.172 |
| hotpot    | 3B / Q8_0   | DECAF no-reorder (λ=6) | 0.273 |    0.572 |        0.633 |      0.128 |
| hotpot    | 3B / Q8_0   | Relevance top-m (iso)  | 0.332 |    0.583 |        0.633 |      0.139 |
| hotpot    | 7B / Q4_K_M | DECAF λ=6              | 0.522 |    0.572 |        0.633 |      0.128 |
| hotpot    | 7B / Q4_K_M | DECAF no-cov (λ=6)     | 0.503 |    0.606 |        0.633 |      0.172 |
| hotpot    | 7B / Q4_K_M | DECAF no-reorder (λ=6) | 0.501 |    0.572 |        0.633 |      0.128 |
| hotpot    | 7B / Q4_K_M | Relevance top-m (iso)  | 0.539 |    0.583 |        0.633 |      0.139 |
| twowiki   | 3B / Q4_K_M | DECAF λ=6              | 0.281 |    0.458 |        0.533 |      0.102 |
| twowiki   | 3B / Q4_K_M | DECAF no-cov (λ=6)     | 0.295 |    0.508 |        0.667 |      0.144 |
| twowiki   | 3B / Q4_K_M | DECAF no-reorder (λ=6) | 0.299 |    0.458 |        0.533 |      0.102 |
| twowiki   | 3B / Q4_K_M | Relevance top-m (iso)  | 0.314 |    0.45  |        0.533 |      0.11  |
| twowiki   | 3B / Q8_0   | DECAF λ=6              | 0.151 |    0.458 |        0.533 |      0.102 |
| twowiki   | 3B / Q8_0   | DECAF no-cov (λ=6)     | 0.13  |    0.508 |        0.667 |      0.144 |
| twowiki   | 3B / Q8_0   | DECAF no-reorder (λ=6) | 0.136 |    0.458 |        0.533 |      0.102 |
| twowiki   | 3B / Q8_0   | Relevance top-m (iso)  | 0.153 |    0.45  |        0.533 |      0.11  |
| twowiki   | 7B / Q4_K_M | DECAF λ=6              | 0.274 |    0.458 |        0.533 |      0.102 |
| twowiki   | 7B / Q4_K_M | DECAF no-cov (λ=6)     | 0.271 |    0.508 |        0.667 |      0.144 |
| twowiki   | 7B / Q4_K_M | DECAF no-reorder (λ=6) | 0.336 |    0.458 |        0.533 |      0.102 |
| twowiki   | 7B / Q4_K_M | Relevance top-m (iso)  | 0.274 |    0.45  |        0.533 |      0.11  |

Figures: `fig_v2_lambda_sweep.png`, `fig_v2_ablation_bars.png`.

## 5. Paired statistics (bootstrap CI, McNemar, Holm)

| run                 | method                 |   ΔF1 vs full | 95% CI           |   McNemar p |   Holm p |
|:--------------------|:-----------------------|--------------:|:-----------------|------------:|---------:|
| 3B / Q4_K_M|hotpot  | Fixed 50%              |         0.131 | [0.035, 0.240]   |       0.062 |    0.5   |
| 3B / Q4_K_M|hotpot  | Fixed 20%              |         0.117 | [0.022, 0.224]   |       0.062 |    0.5   |
| 3B / Q4_K_M|hotpot  | Relevance top-m (iso)  |         0.095 | [0.002, 0.204]   |       0.125 |    0.625 |
| 3B / Q4_K_M|hotpot  | DECAF no-cov (λ=6)     |         0.086 | [-0.008, 0.190]  |       0.125 |    0.625 |
| 3B / Q4_K_M|hotpot  | DECAF no-reorder (λ=6) |         0.068 | [-0.042, 0.189]  |       0.125 |    0.625 |
| 3B / Q4_K_M|hotpot  | DECAF λ=2              |         0.065 | [-0.018, 0.163]  |       0.25  |    0.625 |
| 3B / Q4_K_M|hotpot  | DECAF λ=6              |         0.092 | [-0.001, 0.201]  |       0.125 |    0.625 |
| 3B / Q4_K_M|hotpot  | DECAF λ=15             |         0.101 | [-0.009, 0.225]  |       0.062 |    0.5   |
| 3B / Q4_K_M|twowiki | Fixed 50%              |        -0.021 | [-0.109, 0.073]  |       1     |    1     |
| 3B / Q4_K_M|twowiki | Fixed 20%              |         0.12  | [0.002, 0.248]   |       0.375 |    1     |
| 3B / Q4_K_M|twowiki | Relevance top-m (iso)  |         0.101 | [-0.018, 0.223]  |       0.625 |    1     |
| 3B / Q4_K_M|twowiki | DECAF no-cov (λ=6)     |         0.082 | [-0.045, 0.210]  |       0.625 |    1     |
| 3B / Q4_K_M|twowiki | DECAF no-reorder (λ=6) |         0.086 | [-0.043, 0.218]  |       0.375 |    1     |
| 3B / Q4_K_M|twowiki | DECAF λ=2              |         0.096 | [-0.033, 0.223]  |       0.375 |    1     |
| 3B / Q4_K_M|twowiki | DECAF λ=6              |         0.068 | [-0.048, 0.180]  |       1     |    1     |
| 3B / Q4_K_M|twowiki | DECAF λ=15             |         0.038 | [-0.109, 0.194]  |       1     |    1     |
| 3B / Q8_0|hotpot    | Fixed 50%              |         0.085 | [0.006, 0.185]   |       0.25  |    0.625 |
| 3B / Q8_0|hotpot    | Fixed 20%              |         0.101 | [0.029, 0.191]   |       0.125 |    0.625 |
| 3B / Q8_0|hotpot    | Relevance top-m (iso)  |         0.151 | [0.049, 0.265]   |       0.062 |    0.5   |
| 3B / Q8_0|hotpot    | DECAF no-cov (λ=6)     |         0.142 | [0.047, 0.251]   |       0.062 |    0.5   |
| 3B / Q8_0|hotpot    | DECAF no-reorder (λ=6) |         0.093 | [-0.023, 0.215]  |       0.125 |    0.625 |
| 3B / Q8_0|hotpot    | DECAF λ=2              |         0.112 | [0.029, 0.216]   |       0.125 |    0.625 |
| 3B / Q8_0|hotpot    | DECAF λ=6              |         0.149 | [0.047, 0.263]   |       0.062 |    0.5   |
| 3B / Q8_0|hotpot    | DECAF λ=15             |         0.069 | [-0.062, 0.205]  |       0.219 |    0.625 |
| 3B / Q8_0|twowiki   | Fixed 50%              |         0.032 | [-0.032, 0.104]  |       1     |    1     |
| 3B / Q8_0|twowiki   | Fixed 20%              |         0.011 | [-0.134, 0.149]  |       1     |    1     |
| 3B / Q8_0|twowiki   | Relevance top-m (iso)  |        -0.047 | [-0.154, 0.055]  |       0.625 |    1     |
| 3B / Q8_0|twowiki   | DECAF no-cov (λ=6)     |        -0.07  | [-0.182, 0.029]  |       0.25  |    1     |
| 3B / Q8_0|twowiki   | DECAF no-reorder (λ=6) |        -0.065 | [-0.171, 0.027]  |       0.25  |    1     |
| 3B / Q8_0|twowiki   | DECAF λ=2              |        -0.062 | [-0.176, 0.040]  |       0.25  |    1     |
| 3B / Q8_0|twowiki   | DECAF λ=6              |        -0.05  | [-0.159, 0.053]  |       0.625 |    1     |
| 3B / Q8_0|twowiki   | DECAF λ=15             |        -0.048 | [-0.158, 0.065]  |       0.625 |    1     |
| 7B / Q4_K_M|hotpot  | Fixed 50%              |        -0.018 | [-0.140, 0.103]  |       1     |    1     |
| 7B / Q4_K_M|hotpot  | Fixed 20%              |        -0.084 | [-0.208, 0.025]  |       0.25  |    1     |
| 7B / Q4_K_M|hotpot  | Relevance top-m (iso)  |        -0.057 | [-0.207, 0.079]  |       0.688 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF no-cov (λ=6)     |        -0.093 | [-0.244, 0.042]  |       0.375 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF no-reorder (λ=6) |        -0.094 | [-0.245, 0.040]  |       0.219 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF λ=2              |        -0.082 | [-0.231, 0.047]  |       0.375 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF λ=6              |        -0.073 | [-0.224, 0.066]  |       0.453 |    1     |
| 7B / Q4_K_M|hotpot  | DECAF λ=15             |        -0.261 | [-0.451, -0.073] |       0.039 |    0.312 |
| 7B / Q4_K_M|twowiki | Fixed 50%              |        -0.025 | [-0.119, 0.060]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | Fixed 20%              |        -0.076 | [-0.195, 0.021]  |       0.125 |    0.875 |
| 7B / Q4_K_M|twowiki | Relevance top-m (iso)  |        -0.048 | [-0.185, 0.074]  |       0.375 |    1     |
| 7B / Q4_K_M|twowiki | DECAF no-cov (λ=6)     |        -0.051 | [-0.197, 0.073]  |       0.375 |    1     |
| 7B / Q4_K_M|twowiki | DECAF no-reorder (λ=6) |         0.015 | [-0.144, 0.173]  |       1     |    1     |
| 7B / Q4_K_M|twowiki | DECAF λ=2              |        -0.043 | [-0.186, 0.082]  |       0.375 |    1     |
| 7B / Q4_K_M|twowiki | DECAF λ=6              |        -0.048 | [-0.183, 0.074]  |       0.375 |    1     |
| 7B / Q4_K_M|twowiki | DECAF λ=15             |        -0.129 | [-0.279, 0.000]  |       0.031 |    0.25  |

Figures: `fig_v2_stats_forest.png`. Full numbers in `summary_v2.json` under `stats`.

## 6. Figure index

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
- `fig_v2_method_model_heatmap.png`
- `fig_v2_dataset_compare.png`
- `fig_v2_e2e_ci.png`
- `fig_v2_lambda_sweep.png`
- `fig_v2_ablation_bars.png`
- `fig_v2_stats_forest.png`
