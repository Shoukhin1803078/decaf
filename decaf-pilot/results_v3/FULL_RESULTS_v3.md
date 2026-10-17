# DECAF — Full Results v3 (cold re-run)

scaling=220 · cold compression=128 · warm-order=100

## 1. Cold gate (detection test)
- cold: 0 of 10 groups outside [0.80, 1.25]
- warm-order: 3 of 5 groups outside [0.80, 1.25] (expected to fail — that is the contrast)

## 2. Inflation (warm-order speed-up / cold speed-up)

| cell | method | inflation (geo-mean) | 95% CI | n |
|---|---|---:|---|---:|
| 3B / Q4_K_M|hotpot|k5|bm25_decaf_lam6 | | 1.287 | [1.178, 1.426] | 20 |
| 3B / Q4_K_M|hotpot|k5|decaf_lam6 | | 1.956 | [1.565, 2.496] | 20 |
| 3B / Q4_K_M|hotpot|k5|fixed_0.2 | | 1.776 | [1.450, 2.279] | 20 |
| 3B / Q4_K_M|hotpot|k5|rel_iso | | 2.189 | [1.685, 2.929] | 20 |

## 3. Cold S_net (per-question, matched)

| cell | method | S_net (geo-mean) | 95% CI | gross | n |
|---|---|---:|---|---:|---:|
| 3B / Q4_K_M|hotpot|k5|bm25_decaf_lam6 | | 2.423 | [2.097, 2.812] | 2.426 | 20 |
| 3B / Q4_K_M|hotpot|k5|decaf_lam15 | | 1.752 | [1.579, 1.943] | 2.495 | 2 |
| 3B / Q4_K_M|hotpot|k5|decaf_lam2 | | 1.738 | [1.609, 1.878] | 2.472 | 2 |
| 3B / Q4_K_M|hotpot|k5|decaf_lam6 | | 1.518 | [1.312, 1.741] | 2.395 | 20 |
| 3B / Q4_K_M|hotpot|k5|decaf_nocov_lam6 | | 1.739 | [1.612, 1.877] | 2.475 | 2 |
| 3B / Q4_K_M|hotpot|k5|decaf_noreorder_lam6 | | 1.860 | [1.660, 2.084] | 2.719 | 2 |
| 3B / Q4_K_M|hotpot|k5|fixed_0.2 | | 1.391 | [1.235, 1.533] | 2.092 | 20 |
| 3B / Q4_K_M|hotpot|k5|full | | 1.000 | [1.000, 1.000] | 1.000 | 20 |
| 3B / Q4_K_M|hotpot|k5|llmlingua_0.2 | | 1.080 | [0.961, 1.201] | 2.022 | 20 |
| 3B / Q4_K_M|hotpot|k5|rel_iso | | 1.481 | [1.278, 1.718] | 2.294 | 20 |

## 4. Latency model validation

### 3B / Q4_K_M
- cold prefill coef: c1=3.1583 c2=1.422e-04 (r²=0.9999)
- MAPE <300 tokens: 4.0% (n=70)
- MAPE 300-1000 tokens: 2.9% (n=49)
- MAPE >1000 tokens: 3.4% (n=9)

### 3B / Q8_0
- cold prefill coef: c1=0.0153 c2=-6.565e-07 (r²=0.6884)
- MAPE <300 tokens: nan% (n=0)
- MAPE 300-1000 tokens: nan% (n=0)
- MAPE >1000 tokens: nan% (n=0)

### 7B / Q4_K_M
- cold prefill coef: c1=16.5668 c2=3.471e-04 (r²=0.9082)
- MAPE <300 tokens: nan% (n=0)
- MAPE 300-1000 tokens: nan% (n=0)
- MAPE >1000 tokens: nan% (n=0)

### 7B / Q8_0
- cold prefill coef: c1=13.3208 c2=2.797e-04 (r²=0.9434)
- MAPE <300 tokens: nan% (n=0)
- MAPE 300-1000 tokens: nan% (n=0)
- MAPE >1000 tokens: nan% (n=0)

## 5. Break-even (quadratic cold model)

| config | L_c | T_compress (ms) | L* quadratic | L* linear |
|---|---:|---:|---:|---:|
| 3B / Q4_K_M | 228 | 1002 | 378 | 380 |

## 6. Quality at equal budget (ΔF1 vs full)

| cell | method | ΔF1 | 95% CI | p (F1 sign-flip) | Holm | n |
|---|---|---:|---|---:|---:|---:|
| 3B / Q4_K_M|hotpot|k5|bm25_decaf_lam6 | | +0.097 | [-0.011, +0.217] | 0.151 | 0.605 | 20 |
| 3B / Q4_K_M|hotpot|k5|decaf_lam6 | | +0.044 | [-0.031, +0.138] | 0.379 | 0.758 | 20 |
| 3B / Q4_K_M|hotpot|k5|fixed_0.2 | | +0.003 | [-0.019, +0.028] | 0.820 | 0.820 | 20 |
| 3B / Q4_K_M|hotpot|k5|llmlingua_0.2 | | -0.110 | [-0.218, -0.037] | 0.004 | 0.020 | 20 |
| 3B / Q4_K_M|hotpot|k5|rel_iso | | +0.055 | [-0.015, +0.143] | 0.240 | 0.719 | 20 |
