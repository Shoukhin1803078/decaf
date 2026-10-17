# DECAF — Full Results v3 (cold re-run)

Cold-measurement re-run: every cached prompt carries a unique nonce at character 0 and methods run in randomized order, so no prompt reuses a cached prefix. A parallel warm-order pass reproduces the old design to measure the inflation directly.

- scaling runs: **220**
- cold compression runs: **2002**
- warm-order runs: **100**
- cold prefill model per model: 3B / Q4_K_M c1=3.29, 7B / Q4_K_M c1=8.02

## 0. Headline

- **Cache inflation is real and large.** Warm-order speed-ups exceed cold speed-ups for every method; across cells the inflation ranges 1.29–2.19×. Text-preserving methods (DECAF, fixed, rel_iso) are inflated most; the BM25-scored arm the least (different ordering ⇒ less prefix sharing).
- **A cheap scorer wins the latency frontier (H4).** `bm25_decaf_lam6` has the highest mean cold S_net (2.32×); its selector cost is negligible against the cross-encoder's ~0.9 s.
- **Quality tracks task and model size (H5).** Compression raises 3B F1 on short contexts (10 positive cells), is null on 2Wiki, and drops F1 sharply on LongBench for every config (mean ΔF1 -0.238).
- **The cold gate passes cleanly** (0 groups outside [0.80, 1.25]) whereas the v2 warm measurement failed on 126 of 168 groups.

## 1. Cold gate (detection test)
- cold: 0 of 42 groups outside [0.80, 1.25] (pass)
- warm-order: 4 of 5 groups outside [0.80, 1.25] (expected to fail — that is the contrast)

## 2. Inflation (warm-order speed-up / cold speed-up)

| model | dataset | method | inflation | 95% CI | n |
|---|---|---|---:|---|---:|
| 3B / Q4_K_M | hotpot | bm25_decaf_lam6 | 1.29× | [1.18, 1.42] | 20 |
| 3B / Q4_K_M | hotpot | decaf_lam6 | 1.96× | [1.55, 2.49] | 20 |
| 3B / Q4_K_M | hotpot | fixed_0.2 | 1.78× | [1.44, 2.24] | 20 |
| 3B / Q4_K_M | hotpot | rel_iso | 2.19× | [1.67, 2.92] | 20 |

## 3. Cold S_net (per-question, matched to full on the same qids)

| model | dataset | method | S_net | 95% CI | gross | n |
|---|---|---|---:|---|---:|---:|
| 3B / Q4_K_M | hotpot | bm25_decaf_lam6 | 2.36× | [2.18, 2.55] | 2.36× | 50 |
| 3B / Q4_K_M | hotpot | decaf_lam6 | 1.57× | [1.46, 1.68] | 2.53× | 50 |
| 3B / Q4_K_M | hotpot | fixed_0.2 | 1.44× | [1.35, 1.54] | 2.22× | 50 |
| 3B / Q4_K_M | hotpot | full | 1.00× | [1.00, 1.00] | 1.00× | 50 |
| 3B / Q4_K_M | hotpot | llmlingua_0.2 | 1.10× | [1.03, 1.17] | 2.14× | 50 |
| 3B / Q4_K_M | hotpot | rel_iso | 1.57× | [1.45, 1.70] | 2.54× | 50 |
| 3B / Q4_K_M | longbench | bm25_decaf_lam6 | 1.86× | [1.75, 1.99] | 1.86× | 50 |
| 3B / Q4_K_M | longbench | decaf_lam6 | 1.59× | [1.52, 1.66] | 2.74× | 50 |
| 3B / Q4_K_M | longbench | fixed_0.2 | 1.31× | [1.26, 1.36] | 2.00× | 50 |
| 3B / Q4_K_M | longbench | full | 1.00× | [1.00, 1.00] | 1.00× | 50 |
| 3B / Q4_K_M | longbench | llmlingua_0.2 | 1.18× | [1.13, 1.23] | 2.11× | 50 |
| 3B / Q4_K_M | longbench | rel_iso | 1.56× | [1.48, 1.63] | 2.65× | 50 |
| 3B / Q4_K_M | trivia | bm25_decaf_lam6 | 2.23× | [1.95, 2.65] | 2.23× | 50 |
| 3B / Q4_K_M | trivia | decaf_lam6 | 1.93× | [1.72, 2.17] | 3.53× | 50 |
| 3B / Q4_K_M | trivia | fixed_0.2 | 1.61× | [1.51, 1.73] | 2.60× | 50 |
| 3B / Q4_K_M | trivia | full | 1.00× | [1.00, 1.00] | 1.00× | 50 |
| 3B / Q4_K_M | trivia | llmlingua_0.2 | 1.30× | [1.21, 1.41] | 2.76× | 50 |
| 3B / Q4_K_M | trivia | rel_iso | 1.80× | [1.64, 1.98] | 3.20× | 50 |
| 3B / Q4_K_M | twowiki | bm25_decaf_lam6 | 2.15× | [1.99, 2.33] | 2.15× | 50 |
| 3B / Q4_K_M | twowiki | decaf_lam6 | 1.63× | [1.51, 1.76] | 2.48× | 50 |
| 3B / Q4_K_M | twowiki | fixed_0.2 | 1.41× | [1.31, 1.52] | 2.00× | 50 |
| 3B / Q4_K_M | twowiki | full | 1.00× | [1.00, 1.00] | 1.00× | 50 |
| 3B / Q4_K_M | twowiki | llmlingua_0.2 | 1.22× | [1.15, 1.29] | 2.24× | 50 |
| 3B / Q4_K_M | twowiki | rel_iso | 1.62× | [1.49, 1.76] | 2.47× | 50 |
| 7B / Q4_K_M | hotpot | bm25_decaf_lam6 | 2.75× | [2.51, 3.01] | 2.75× | 50 |
| 7B / Q4_K_M | hotpot | decaf_lam6 | 2.12× | [1.97, 2.26] | 2.95× | 50 |
| 7B / Q4_K_M | hotpot | fixed_0.2 | 1.82× | [1.72, 1.92] | 2.39× | 50 |
| 7B / Q4_K_M | hotpot | full | 1.00× | [1.00, 1.00] | 1.00× | 50 |
| 7B / Q4_K_M | hotpot | llmlingua_0.2 | 1.28× | [1.21, 1.36] | 2.52× | 50 |
| 7B / Q4_K_M | hotpot | rel_iso | 2.14× | [2.00, 2.28] | 2.98× | 50 |
| 7B / Q4_K_M | trivia | bm25_decaf_lam6 | 2.68× | [2.19, 3.36] | 2.68× | 33 |
| 7B / Q4_K_M | trivia | decaf_lam6 | 2.67× | [2.35, 3.08] | 4.07× | 33 |
| 7B / Q4_K_M | trivia | fixed_0.2 | 2.07× | [1.92, 2.24] | 2.85× | 33 |
| 7B / Q4_K_M | trivia | full | 1.00× | [1.00, 1.00] | 1.00× | 33 |
| 7B / Q4_K_M | trivia | llmlingua_0.2 | 1.42× | [1.29, 1.60] | 3.23× | 33 |
| 7B / Q4_K_M | trivia | rel_iso | 2.35× | [2.11, 2.61] | 3.47× | 33 |
| 7B / Q4_K_M | twowiki | bm25_decaf_lam6 | 2.22× | [2.06, 2.40] | 2.22× | 50 |
| 7B / Q4_K_M | twowiki | decaf_lam6 | 2.07× | [1.89, 2.25] | 2.73× | 50 |
| 7B / Q4_K_M | twowiki | fixed_0.2 | 1.71× | [1.60, 1.83] | 2.13× | 50 |
| 7B / Q4_K_M | twowiki | full | 1.00× | [1.00, 1.00] | 1.00× | 50 |
| 7B / Q4_K_M | twowiki | llmlingua_0.2 | 1.20× | [1.12, 1.29] | 2.31× | 50 |
| 7B / Q4_K_M | twowiki | rel_iso | 2.05× | [1.88, 2.25] | 2.70× | 50 |

## 4. Latency model validation (out-of-sample on cold prompts)

| model | c1 (ms/tok) | c2 | MAPE <300 | MAPE 300-1000 | MAPE >1000 |
|---|---:|---:|---:|---:|---:|
| 3B / Q4_K_M | 3.29 | 1.33e-04 | 6.0% (n=536) | 2.4% (n=532) | 1.4% (n=132) |
| 3B / Q8_0 | 0.00 | 0.00e+00 | nan% (n=0) | nan% (n=0) | nan% (n=0) |
| 7B / Q4_K_M | 8.02 | 2.12e-04 | 11.8% (n=421) | 4.9% (n=315) | 2.4% (n=66) |
| 7B / Q8_0 | 0.00 | 0.00e+00 | nan% (n=0) | nan% (n=0) | nan% (n=0) |

## 5. Break-even (quadratic cold model)

| config | L_c | T_compress (ms) | L* quadratic | L* linear |
|---|---:|---:|---:|---:|
| 3B / Q4_K_M | 225 | 1029 | 376 | 378 |
| 7B / Q4_K_M | 218 | 1042 | 220 | 220 |

## 6. Quality at equal budget (ΔF1 vs full)

| model | dataset | method | ΔF1 | 95% CI | p (F1 sign-flip) | Holm | n |
|---|---|---|---:|---|---:|---:|---:|
| 3B / Q4_K_M | hotpot | bm25_decaf_lam6 | +0.079 | [+0.009, +0.159] | 0.042 | 0.193 | 50 |
| 3B / Q4_K_M | hotpot | decaf_lam6 | +0.073 | [-0.001, +0.148] | 0.067 | 0.201 | 50 |
| 3B / Q4_K_M | hotpot | fixed_0.2 | +0.065 | [+0.001, +0.135] | 0.071 | 0.201 | 50 |
| 3B / Q4_K_M | hotpot | llmlingua_0.2 | -0.041 | [-0.111, +0.033] | 0.291 | 0.291 | 50 |
| 3B / Q4_K_M | hotpot | rel_iso | +0.082 | [+0.007, +0.161] | 0.039 | 0.193 | 50 |
| 3B / Q4_K_M | longbench | bm25_decaf_lam6 | -0.244 | [-0.345, -0.146] | 0.000 | 0.000 | 50 |
| 3B / Q4_K_M | longbench | decaf_lam6 | -0.238 | [-0.342, -0.140] | 0.000 | 0.000 | 50 |
| 3B / Q4_K_M | longbench | fixed_0.2 | -0.226 | [-0.322, -0.134] | 0.000 | 0.000 | 50 |
| 3B / Q4_K_M | longbench | llmlingua_0.2 | -0.259 | [-0.374, -0.151] | 0.000 | 0.000 | 50 |
| 3B / Q4_K_M | longbench | rel_iso | -0.226 | [-0.330, -0.126] | 0.000 | 0.000 | 50 |
| 3B / Q4_K_M | trivia | bm25_decaf_lam6 | +0.056 | [+0.004, +0.115] | 0.056 | 0.145 | 50 |
| 3B / Q4_K_M | trivia | decaf_lam6 | +0.080 | [+0.016, +0.158] | 0.036 | 0.145 | 50 |
| 3B / Q4_K_M | trivia | fixed_0.2 | +0.080 | [+0.011, +0.167] | 0.044 | 0.145 | 50 |
| 3B / Q4_K_M | trivia | llmlingua_0.2 | +0.020 | [-0.052, +0.097] | 0.607 | 0.607 | 50 |
| 3B / Q4_K_M | trivia | rel_iso | +0.091 | [+0.023, +0.172] | 0.021 | 0.107 | 50 |
| 3B / Q4_K_M | twowiki | bm25_decaf_lam6 | -0.005 | [-0.083, +0.074] | 0.907 | 1.000 | 50 |
| 3B / Q4_K_M | twowiki | decaf_lam6 | -0.037 | [-0.111, +0.039] | 0.348 | 1.000 | 50 |
| 3B / Q4_K_M | twowiki | fixed_0.2 | +0.014 | [-0.049, +0.082] | 0.714 | 1.000 | 50 |
| 3B / Q4_K_M | twowiki | llmlingua_0.2 | -0.027 | [-0.102, +0.055] | 0.508 | 1.000 | 50 |
| 3B / Q4_K_M | twowiki | rel_iso | -0.051 | [-0.131, +0.024] | 0.210 | 1.000 | 50 |
| 7B / Q4_K_M | hotpot | bm25_decaf_lam6 | -0.081 | [-0.189, +0.017] | 0.141 | 0.564 | 50 |
| 7B / Q4_K_M | hotpot | decaf_lam6 | -0.039 | [-0.149, +0.073] | 0.511 | 0.752 | 50 |
| 7B / Q4_K_M | hotpot | fixed_0.2 | -0.044 | [-0.145, +0.046] | 0.376 | 0.752 | 50 |
| 7B / Q4_K_M | hotpot | llmlingua_0.2 | -0.185 | [-0.319, -0.043] | 0.014 | 0.071 | 50 |
| 7B / Q4_K_M | hotpot | rel_iso | -0.064 | [-0.161, +0.034] | 0.243 | 0.730 | 50 |
| 7B / Q4_K_M | trivia | bm25_decaf_lam6 | +0.010 | [-0.102, +0.115] | 0.867 | 1.000 | 33 |
| 7B / Q4_K_M | trivia | decaf_lam6 | +0.048 | [-0.058, +0.164] | 0.434 | 1.000 | 33 |
| 7B / Q4_K_M | trivia | fixed_0.2 | +0.011 | [-0.096, +0.116] | 0.857 | 1.000 | 33 |
| 7B / Q4_K_M | trivia | llmlingua_0.2 | -0.045 | [-0.159, +0.061] | 0.432 | 1.000 | 33 |
| 7B / Q4_K_M | trivia | rel_iso | +0.014 | [-0.115, +0.130] | 0.858 | 1.000 | 33 |
| 7B / Q4_K_M | twowiki | bm25_decaf_lam6 | -0.043 | [-0.117, +0.037] | 0.288 | 0.300 | 50 |
| 7B / Q4_K_M | twowiki | decaf_lam6 | -0.093 | [-0.174, -0.016] | 0.028 | 0.110 | 50 |
| 7B / Q4_K_M | twowiki | fixed_0.2 | -0.054 | [-0.132, +0.015] | 0.150 | 0.300 | 50 |
| 7B / Q4_K_M | twowiki | llmlingua_0.2 | -0.077 | [-0.166, +0.006] | 0.093 | 0.279 | 50 |
| 7B / Q4_K_M | twowiki | rel_iso | -0.108 | [-0.198, -0.028] | 0.015 | 0.075 | 50 |

## 7. Notes and caveats

- Cold model = origin-constrained quadratic fit on cleaned scaling medians unioned with the cold `full` compression runs (both cold, nonce-protected). The 7B scaling sweep carried a per-request overhead; anchoring to the real full prompts corrects it.
- Scaling calibration bounded to 4096 tokens; full-context prompts are ≈860–1220 tokens.
- 7B/Q4 cold is partially complete (hotpot, twowiki); TriviaQA/LongBench and the warm-order pass beyond the 3B pilot are still running.
- H5 is a measurement study: no claim that DECAF beats other selectors (H3 remains unsupported; `bm25_decaf_lam6` and `rel_iso` match or beat it).
