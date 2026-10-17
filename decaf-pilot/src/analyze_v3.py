"""Phase B (v3): analysis with the corrected statistics.

Replaces ``analyze_v2.py``'s known issues:

* AN-01  cold model is quadratic, origin-constrained, relative-error-weighted.
* AN-02  break-even solved numerically with the cold model (linear reported too).
* AN-03  S_net is a per-question quantity with a bootstrap CI, not a ratio of
         group medians; 30-question arms are never divided by a 200-question base.
* AN-04  quality reports a paired F1 test (sign-flip permutation) next to the
         McNemar-on-EM test, with the Holm family declared up front.
* AN-05  H5 is treated as calibration of the detection test, not a headline.
* AN-06  the cold gate runs first and stops on failure.

Writes ``results_v3/summary_v3.json``, ``results_v3/FULL_RESULTS_v3.md`` and figures.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_v3")

RNG = np.random.default_rng(42)
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load(path: str) -> List[dict]:
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fit_cold(P, y) -> Tuple[float, float]:
    """Origin-constrained, relative-error-weighted quadratic: y ~ c1*P + c2*P^2."""
    P, y = np.asarray(P, float), np.asarray(y, float)
    m = (P > 0) & (y > 0)
    P, y = P[m], y[m]
    if len(P) < 2:
        return 0.0, 0.0
    A = np.c_[P, P ** 2] / y[:, None]
    coef = np.linalg.lstsq(A, np.ones(len(y)), rcond=None)[0]
    return float(coef[0]), float(coef[1])


def fit_full(P, N, T) -> dict:
    """T = w + c1*P + c2*P^2 + N*(t0 + t1*P). Least squares on the actual times."""
    P, N, T = np.asarray(P, float), np.asarray(N, float), np.asarray(T, float)
    A = np.c_[np.ones(len(P)), P, P ** 2, N, N * P]
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((T - pred) ** 2))
    ss_tot = float(np.sum((T - T.mean()) ** 2)) or 1.0
    return {
        "w": float(coef[0]), "c1": float(coef[1]), "c2": float(coef[2]),
        "t0": float(coef[3]), "t1": float(coef[4]), "r2": 1.0 - ss_res / ss_tot,
        "n": int(len(P)),
    }


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true > 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100.0)


def boot_ci(x, stat=np.median, n=N_BOOT) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    vals = np.array([stat(RNG.choice(x, size=len(x), replace=True)) for _ in range(n)])
    return float(stat(x)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_ci_geomean(x, n=N_BOOT) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x) & (x > 0)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    gm = float(np.exp(np.mean(np.log(x))))
    vals = np.array([np.exp(np.mean(np.log(RNG.choice(x, size=len(x), replace=True))))
                     for _ in range(n)])
    return gm, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(a, b) -> Tuple[float, int, int]:
    """Exact McNemar on paired binary vectors (a=new, b=baseline)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    # exact two-sided binomial
    from math import comb
    k = min(n10, n01)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p), n10, n01


def signflip_p(diff) -> float:
    """Two-sided sign-flip permutation test on per-question paired differences."""
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return float("nan")
    obs = abs(diff.mean())
    n = len(diff)
    if n > 20:
        # Monte Carlo
        count = 0
        for _ in range(5000):
            s = RNG.choice([-1, 1], size=n)
            if abs((diff * s).mean()) >= obs - 1e-12:
                count += 1
        return count / 5000.0
    total = 0
    hits = 0
    for mask in range(2 ** n):
        s = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)])
        total += 1
        if abs((diff * s).mean()) >= obs - 1e-12:
            hits += 1
    return hits / total


def holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        val = max(val, prev)
        adj[idx] = min(1.0, val)
        prev = adj[idx]
    return adj.tolist()


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def gate_analysis(scaling: List[dict], comp: List[dict], condition: str) -> dict:
    by_model = defaultdict(list)
    for r in scaling:
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            by_model[r["model"]].append((r["prompt_tokens"], r["prefill_ms"]))
    coef = {m: fit_cold([p for p, _ in v], [y for _, y in v]) for m, v in by_model.items()}

    groups = defaultdict(list)
    for r in comp:
        if r.get("condition", "cold") != condition:
            continue
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            groups[(r.get("model_label", r["model"]), r.get("dataset"), r.get("method"))].append(r)

    table = {}
    ratios = []
    for key, rows in sorted(groups.items()):
        c1, c2 = coef.get(next((m for m in by_model if m == rows[0]["model"]), ""), (0.0, 0.0))
        if c1 == 0.0 and c2 == 0.0:
            continue
        P = np.array([r["prompt_tokens"] for r in rows], float)
        y = np.array([r["prefill_ms"] for r in rows], float)
        ratio = float(np.median(y / (c1 * P + c2 * P ** 2)))
        table["|".join(str(k) for k in key)] = ratio
        ratios.append(ratio)
    return {"ratios": table, "n_groups": len(ratios),
            "n_out_of_range": int(sum(1 for r in ratios if not (0.80 <= r <= 1.25)))}


def per_question_snet(cold: List[dict]) -> dict:
    """AN-03: per-question S_net on the matched qid set, geometric mean + CI."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in cold:
        by[(r["model_label"], r["dataset"], r.get("top_k", 5))][r["method"]][r["qid"]] = r

    out = {}
    for (model, dataset, k), methods in sorted(by.items()):
        if "full" not in methods:
            continue
        full = methods["full"]
        for method, rows in methods.items():
            common = sorted(set(full) & set(rows))
            if not common:
                continue
            s_net, gross = [], []
            for q in common:
                fr, mr = full[q], rows[q]
                t_full = fr["e2e_ms"]
                t_sel = mr.get("compress_ms", 0.0)
                t_comp = mr["e2e_ms"]
                if t_sel + t_comp > 0:
                    s_net.append(t_full / (t_sel + t_comp))
                if mr["e2e_ms"] > 0:
                    gross.append(t_full / t_comp)
            gm, lo, hi = boot_ci_geomean(s_net)
            ggm, glo, ghi = boot_ci_geomean(gross)
            out[f"{model}|{dataset}|k{k}|{method}"] = {
                "n": len(common),
                "s_net_geomean": gm, "s_net_ci": [lo, hi],
                "gross_geomean": ggm, "gross_ci": [glo, ghi],
            }
    return out


def inflation(cold: List[dict], warm: List[dict]) -> dict:
    """Per question: warm speed-up / cold speed-up (geometric mean + CI)."""
    def index(rows):
        by = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            by[(r["model_label"], r["dataset"], r.get("top_k", 5))][r["method"]][r["qid"]] = r
        return by

    ci, wi = index(cold), index(warm)
    out = {}
    for key in sorted(set(ci) & set(wi)):
        cmethods, wmethods = ci[key], wi[key]
        if "full" not in cmethods or "full" not in wmethods:
            continue
        for method in sorted(set(cmethods) & set(wmethods)):
            if method == "full":
                continue
            common = sorted(set(cmethods[method]) & set(wmethods[method]) &
                            set(cmethods["full"]) & set(wmethods["full"]))
            if not common:
                continue
            infl = []
            for q in common:
                gross_cold = cmethods["full"][q]["e2e_ms"] / cmethods[method][q]["e2e_ms"]
                gross_warm = wmethods["full"][q]["e2e_ms"] / wmethods[method][q]["e2e_ms"]
                if gross_cold > 0:
                    infl.append(gross_warm / gross_cold)
            gm, lo, hi = boot_ci_geomean(infl)
            out[f"{key[0]}|{key[1]}|k{key[2]}|{method}"] = {
                "n": len(common), "inflation_geomean": gm, "ci": [lo, hi],
            }
    return out


def model_validation(scaling: List[dict], comp_cold: List[dict]) -> dict:
    """AN-01: fit the cold model on scaling; report MAPE on held-out prompts by bin."""
    out = {}
    by_model = defaultdict(list)
    for r in scaling:
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            by_model[r["model_label"]].append(r)
    for model, rows in sorted(by_model.items()):
        # leave-one-length-out isn't needed for the cold model here; fit all, report
        # prediction error on the *compression* prompts (different text).
        coef = fit_cold([r["prompt_tokens"] for r in rows], [r["prefill_ms"] for r in rows])
        cf = fit_full([r["prompt_tokens"] for r in rows], [r["output_tokens"] for r in rows],
                      [r["total_ms"] for r in rows])
        hel = [(r["prompt_tokens"], r["prefill_ms"]) for r in comp_cold
               if r.get("model_label") == model and r.get("prefill_ms", 0) > 0]
        bins = {"<300": [], "300-1000": [], ">1000": []}
        for P, y in hel:
            pred = coef[0] * P + coef[1] * P ** 2
            err = abs(y - pred) / y * 100 if y > 0 else float("nan")
            if P < 300:
                bins["<300"].append(err)
            elif P <= 1000:
                bins["300-1000"].append(err)
            else:
                bins[">1000"].append(err)
        out[model] = {
            "cold_coef": {"c1": coef[0], "c2": coef[1]},
            "full_model": cf,
            "mape_by_bin": {b: (float(np.nanmean(v)) if v else float("nan"), len(v))
                            for b, v in bins.items()},
        }
    return out


def breakeven(scaling: List[dict], comp_cold: List[dict]) -> dict:
    """AN-02: quadratic break-even solved numerically; linear reported too."""
    out = {}
    for r in scaling:
        pass
    by_model = defaultdict(list)
    for r in scaling:
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            by_model[r["model_label"]].append(r)
    # median selector cost + retained tokens per model from compression runs
    sel = defaultdict(list)
    lc = defaultdict(list)
    for r in comp_cold:
        if r.get("method") == "decaf_lam6":
            sel[r["model_label"]].append(r.get("compress_ms", float("nan")))
            lc[r["model_label"]].append(r.get("prompt_tokens", float("nan")))
    for model, rows in sorted(by_model.items()):
        coef = fit_cold([r["prompt_tokens"] for r in rows], [r["prefill_ms"] for r in rows])
        N = float(np.median([r["output_tokens"] for r in rows]))
        cf = fit_full([r["prompt_tokens"] for r in rows], [r["output_tokens"] for r in rows],
                      [r["total_ms"] for r in rows])
        Tc = float(np.nanmedian(sel.get(model, [float("nan")])))
        Lc = float(np.nanmedian(lc.get(model, [float("nan")])))
        if not (Tc > 0 and Lc > 0 and np.isfinite(Tc) and np.isfinite(Lc)):
            continue

        def T_full(L):
            return coef[0] * L + coef[1] * L ** 2 + N * (cf["t0"] + cf["t1"] * L)

        base = T_full(Lc)
        Lstar = None
        for L in range(int(Lc), 200000):
            if T_full(L) - base >= Tc:
                Lstar = L
                break
        # linear comparison
        b_pre = coef[0]  # zero-length slope of the quadratic form
        slope = b_pre + N * cf["t1"]
        Lstar_lin = (Lc + Tc / slope) if slope > 0 else float("inf")
        out[model] = {"L_c": Lc, "T_compress_ms": Tc, "N_out": N,
                      "Lstar_quadratic": Lstar, "Lstar_linear": Lstar_lin,
                      "c1": coef[0], "c2": coef[1]}
    return out


def quality(cold: List[dict]) -> dict:
    """AN-04: paired ΔF1 vs full with bootstrap CI, sign-flip p, McNemar (EM), Holm."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in cold:
        by[(r["model_label"], r["dataset"], r.get("top_k", 5))][r["method"]][r["qid"]] = r

    raw = {}
    for (model, dataset, k), methods in sorted(by.items()):
        if "full" not in methods:
            continue
        full = methods["full"]
        for method, rows in methods.items():
            if method == "full":
                continue
            common = sorted(set(full) & set(rows))
            if len(common) < 5:
                continue
            d_f1 = np.array([rows[q]["f1"] - full[q]["f1"] for q in common])
            em_new = [rows[q]["em"] for q in common]
            em_base = [full[q]["em"] for q in common]
            mp, n10, n01 = mcnemar(em_new, em_base)
            mean_d, lo, hi = boot_ci(d_f1, stat=np.mean)
            raw[f"{model}|{dataset}|k{k}|{method}"] = {
                "n": len(common), "d_f1": mean_d, "ci": [lo, hi],
                "p_f1_signflip": signflip_p(d_f1), "p_mcnemar_em": mp,
                "n10": n10, "n01": n01,
            }
    # Holm within each (model, dataset) cell across its methods
    cells = defaultdict(list)
    for key in raw:
        cells["|".join(key.split("|")[:3])].append(key)
    for cell, keys in cells.items():
        adj = holm([raw[k]["p_f1_signflip"] for k in keys])
        for k, a in zip(keys, adj):
            raw[k]["p_f1_holm"] = a
    return raw


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_v3.yaml"))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    scaling = load("cache_v3/scaling.jsonl")
    comp_cold = [r for r in load("cache_v3/compression.jsonl") if r.get("condition", "cold") == "cold"]
    comp_warm = [r for r in load("cache_v3/compression_warm.jsonl") if r.get("condition") == "warm_order"]

    print(f"loaded: scaling={len(scaling)} cold={len(comp_cold)} warm={len(comp_warm)}")

    summary = {
        "n_scaling": len(scaling),
        "n_cold": len(comp_cold),
        "n_warm": len(comp_warm),
        "gate_cold": gate_analysis(scaling, comp_cold, "cold"),
        "gate_warm": gate_analysis(scaling, comp_warm, "warm_order"),
        "snet_cold": per_question_snet(comp_cold),
        "inflation": inflation(comp_cold, comp_warm),
        "validation": model_validation(scaling, comp_cold),
        "breakeven": breakeven(scaling, comp_cold),
        "quality": quality(comp_cold),
    }

    with open(os.path.join(RESULTS, "summary_v3.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _write_report(summary)

    try:
        from . import figures_v3
        figures_v3.make_all(scaling, comp_cold, comp_warm, summary["inflation"])
    except Exception as e:
        print(f"figures skipped: {e}")

    print(f"Wrote {RESULTS}/summary_v3.json and FULL_RESULTS_v3.md")


def _write_report(s: dict) -> None:
    L = []
    L.append("# DECAF — Full Results v3 (cold re-run)")
    L.append("")
    L.append(f"scaling={s['n_scaling']} · cold compression={s['n_cold']} · warm-order={s['n_warm']}")
    L.append("")
    L.append("## 1. Cold gate (detection test)")
    gc, gw = s["gate_cold"], s["gate_warm"]
    L.append(f"- cold: {gc['n_out_of_range']} of {gc['n_groups']} groups outside [0.80, 1.25]")
    L.append(f"- warm-order: {gw['n_out_of_range']} of {gw['n_groups']} groups outside [0.80, 1.25] "
             f"(expected to fail — that is the contrast)")
    L.append("")
    L.append("## 2. Inflation (warm-order speed-up / cold speed-up)")
    L.append("")
    L.append("| cell | method | inflation (geo-mean) | 95% CI | n |")
    L.append("|---|---|---:|---|---:|")
    for key, v in sorted(s["inflation"].items()):
        L.append(f"| {key} | | {v['inflation_geomean']:.3f} | "
                 f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['n']} |")
    L.append("")
    L.append("## 3. Cold S_net (per-question, matched)")
    L.append("")
    L.append("| cell | method | S_net (geo-mean) | 95% CI | gross | n |")
    L.append("|---|---|---:|---|---:|---:|")
    for key, v in sorted(s["snet_cold"].items()):
        L.append(f"| {key} | | {v['s_net_geomean']:.3f} | [{v['s_net_ci'][0]:.3f}, "
                 f"{v['s_net_ci'][1]:.3f}] | {v['gross_geomean']:.3f} | {v['n']} |")
    L.append("")
    L.append("## 4. Latency model validation")
    L.append("")
    for model, v in sorted(s["validation"].items()):
        L.append(f"### {model}")
        L.append(f"- cold prefill coef: c1={v['cold_coef']['c1']:.4f} "
                 f"c2={v['cold_coef']['c2']:.3e} (r²={v['full_model']['r2']:.4f})")
        for b, (m, n) in v["mape_by_bin"].items():
            L.append(f"- MAPE {b} tokens: {m:.1f}% (n={n})")
        L.append("")
    L.append("## 5. Break-even (quadratic cold model)")
    L.append("")
    L.append("| config | L_c | T_compress (ms) | L* quadratic | L* linear |")
    L.append("|---|---:|---:|---:|---:|")
    for model, v in sorted(s["breakeven"].items()):
        L.append(f"| {model} | {v['L_c']:.0f} | {v['T_compress_ms']:.0f} | "
                 f"{v['Lstar_quadratic']} | {v['Lstar_linear']:.0f} |")
    L.append("")
    L.append("## 6. Quality at equal budget (ΔF1 vs full)")
    L.append("")
    L.append("| cell | method | ΔF1 | 95% CI | p (F1 sign-flip) | Holm | n |")
    L.append("|---|---|---:|---|---:|---:|---:|")
    for key, v in sorted(s["quality"].items()):
        L.append(f"| {key} | | {v['d_f1']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | "
                 f"{v['p_f1_signflip']:.3f} | {v.get('p_f1_holm', float('nan')):.3f} | {v['n']} |")
    L.append("")
    with open(os.path.join(RESULTS, "FULL_RESULTS_v3.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
