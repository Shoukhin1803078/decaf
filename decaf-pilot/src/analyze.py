"""Phase B: aggregation, the derived quantities of proposal Table 2
(CE, O_c, S_net, QE, L*), the statistics of Sec. 21, every figure, and the
results write-up.

Latency accounting
------------------
`e2e_total_ms` = retrieval + compression + the model's own end-to-end time, so
the compressor's cost is *charged* to the method that incurs it. Net speedup is
computed against the full-context pipeline measured the same way, which is the
proposal's `S_net` rather than a prefill-only comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import figures as F
from . import telemetry
from .run_experiments import ROOT, JsonlCache, load_config

METHOD_ORDER = [
    "full", "fixed_0.8", "fixed_0.6", "fixed_0.5", "fixed_0.4", "fixed_0.2",
    "rel_iso", "decaf_lam2", "decaf_lam4", "decaf_lam6", "decaf_lam10", "decaf_lam15",
    "decaf_nocov_lam6", "decaf_nocost_lam6", "decaf_noreorder_lam6",
    "decaf_absms_lam0.001", "decaf_absms_lam0.002", "decaf_absms_lam0.005",
    "decaf_ce_lam6", "decaf_ce_nocost_lam6", "fixed_ce_0.2", "closed_book",
]


# --- statistics ------------------------------------------------------------

def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 4000,
                     seed: int = 0) -> Tuple[float, float, float, float]:
    """Paired bootstrap on (a - b). Returns mean delta, lo, hi, two-sided p."""
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided p: how often the bootstrap crosses zero
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar on paired binary correctness."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m].astype(bool), b[m].astype(bool)
    if a.size == 0:
        return float("nan")
    tbl = [[int(np.sum(a & b)), int(np.sum(a & ~b))],
           [int(np.sum(~a & b)), int(np.sum(~a & ~b))]]
    if tbl[0][1] + tbl[1][0] == 0:
        return 1.0
    return float(mcnemar(tbl, exact=True).pvalue)


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni across the comparison family."""
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, bool] = {k: False for k in pvals}
    for i, (k, p) in enumerate(items):
        if p <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


# --- aggregation -----------------------------------------------------------

def fit_scaling(sc: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Median curves and linear fits of prefill(L) and TPOT(L)."""
    s = (sc.groupby("L").agg(
        prompt_tokens=("prompt_tokens", "median"), prefill_ms=("prefill_ms", "median"),
        decode_ms=("decode_ms", "median"), tpot_ms=("tpot_ms", "median"),
        ttft_ms=("ttft_ms", "median"), e2e_ms=("e2e_ms", "median"),
        total_ms=("total_ms", "median"), output_tokens=("output_tokens", "median"),
        achieved_bw_gbs=("achieved_bw_gbs", "median"), n=("e2e_ms", "size"),
    ).reset_index().sort_values("prompt_tokens"))
    s["decode_share"] = s["decode_ms"] / s["total_ms"].clip(lower=1e-9)
    s["prefill_ms_per_tok"] = s["prefill_ms"] / s["prompt_tokens"].clip(lower=1)
    L = s["prompt_tokens"].values
    fits = {}
    for key, col in [("tpot", "tpot_ms"), ("prefill", "prefill_ms")]:
        if len(L) >= 2:
            b, a = np.polyfit(L, s[col].values, 1)
            r2 = float(np.corrcoef(L, s[col].values)[0, 1] ** 2) if len(L) > 2 else 1.0
            fits[key] = {"intercept_ms": float(a), "slope_ms_per_tok": float(b), "r2": r2}
    return s, fits


def break_even(sc: pd.DataFrame, comp: pd.DataFrame, cfg: dict) -> Dict:
    """E4 — L* per (quantization, generation length) from the fitted curves."""
    primary = cfg["compression"]["decaf_primary"]
    per_config = []
    for (model, quant), d in sc.groupby(["model", "quant"]):
        base_n = cfg["scaling"]["n_out_base"]
        dn = d[d["n_out"] == base_n] if "n_out" in d else d
        if len(dn) < 2:
            continue
        L = dn.groupby("L")["prompt_tokens"].median().values
        pre = dn.groupby("L")["prefill_ms"].median().values
        tp = dn.groupby("L")["tpot_ms"].median().values
        pre_b, pre_a = np.polyfit(L, pre, 1)
        tpot_b, tpot_a = np.polyfit(L, tp, 1)
        cm = comp[(comp["model"] == model) & (comp["method"] == primary)]
        if cm.empty:
            cm = comp[(comp["model"] == model) & (comp["method"].str.startswith("decaf"))]
        if cm.empty:
            continue
        Lc = float(cm["prompt_tokens"].median())
        Tc = float(cm["compress_ms"].median())
        n_out = float(cm["output_tokens"].median() or base_n)
        slope = pre_b + n_out * tpot_b
        Lstar = Lc + Tc / slope if slope > 0 else float("inf")
        per_config.append({
            "model": model, "quant": quant, "n_out": n_out,
            "pre_a": float(pre_a), "pre_b": float(pre_b),
            "tpot_a": float(tpot_a), "tpot_b": float(tpot_b),
            "L_c": Lc, "T_compress_ms": Tc, "slope_ms_per_tok": float(slope),
            "L_star_tokens": float(Lstar), "L_max_observed": float(L.max()),
            "marginal_prefill_ms_per_tok": float(pre_b),
            "marginal_decode_ms_per_tok": float(n_out * tpot_b),
        })
    return {"per_config": per_config, "primary_method": primary}


def aggregate_methods(comp: pd.DataFrame, model: str) -> pd.DataFrame:
    d = comp[comp["model"] == model]
    if d.empty:
        return pd.DataFrame()
    extra = {}
    for c in ("contains_gold", "answer_recall", "pred_tokens", "e2e_norm_ms"):
        if c in d.columns:
            extra[c] = (c, "median" if c in ("pred_tokens", "e2e_norm_ms") else "mean")
    by = (d.groupby("method").agg(
        f1=("f1", "mean"), em=("em", "mean"), **extra,
        evidence_recall=("evidence_recall", "mean"),
        evidence_precision=("evidence_precision", "mean"),
        answer_support=("answer_support", "mean"),
        gold_in_evidence=("gold_in_evidence", "mean"),
        retained_ratio=("retained_ratio", "mean"),
        n_selected=("n_selected", "mean"),
        compress_ms=("compress_ms", "median"), score_ms=("score_ms", "median"),
        t_retrieve_ms=("t_retrieve_ms", "median"),
        prompt_tokens=("prompt_tokens", "median"),
        prefill_ms=("prefill_ms", "median"), decode_ms=("decode_ms", "median"),
        tpot_ms=("tpot_ms", "median"), ttft_ms=("ttft_ms", "median"),
        e2e_ms=("e2e_ms", "median"), e2e_p95=("e2e_ms", lambda s: float(np.percentile(s, 95))),
        n=("f1", "size"),
    ).reset_index())
    by["e2e_total_ms"] = by["e2e_ms"] + by["compress_ms"] + by["t_retrieve_ms"]
    base = by[by["method"] == "full"]
    if not base.empty:
        b = base.iloc[0]
        by["net_speedup"] = b["e2e_total_ms"] / by["e2e_total_ms"].clip(lower=1e-9)
        by["overhead_frac"] = by["compress_ms"] / max(1e-9, b["e2e_total_ms"])
        dL = (b["prompt_tokens"] - by["prompt_tokens"]).clip(lower=1e-9)
        by["CE_ms_per_token"] = (b["decode_ms"] - by["decode_ms"]) / dL
        by["CE_total_ms_per_token"] = (b["e2e_ms"] - by["e2e_ms"]) / dL
    by["QE_f1_per_sec"] = by["f1"] / (by["e2e_total_ms"] / 1000.0).clip(lower=1e-9)
    by["QE_grounding_per_sec"] = by["gold_in_evidence"] / (by["e2e_total_ms"] / 1000.0).clip(lower=1e-9)
    by["o"] = by["method"].apply(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    return by.sort_values("o").drop(columns="o")


def method_stats(comp: pd.DataFrame, model: str, seed: int = 0) -> List[Dict]:
    """Paired comparisons of every method against full context (Sec. 21)."""
    d = comp[comp["model"] == model]
    if d.empty or "full" not in set(d["method"]):
        return []
    piv_f1 = d.pivot_table(index="qid", columns="method", values="f1")
    piv_em = d.pivot_table(index="qid", columns="method", values="em")
    piv_e2e = d.pivot_table(index="qid", columns="method", values="e2e_ms")
    piv_g = d.pivot_table(index="qid", columns="method", values="gold_in_evidence")
    rows, pv_f1, pv_em = [], {}, {}
    for m in piv_f1.columns:
        if m == "full":
            continue
        pair = piv_f1[[m, "full"]].dropna()
        d_f1, lo, hi, p = paired_bootstrap(pair[m].values, pair["full"].values, seed=seed)
        pe = piv_em[[m, "full"]].dropna()
        p_em = mcnemar_p(pe[m].values, pe["full"].values)
        ge = piv_g[[m, "full"]].dropna()
        d_g, g_lo, g_hi, p_g = paired_bootstrap(ge[m].values, ge["full"].values, seed=seed)
        te = piv_e2e[[m, "full"]].dropna()
        d_t, t_lo, t_hi, _ = paired_bootstrap(te[m].values, te["full"].values, seed=seed)
        f1_abs = float(piv_f1[m].mean())
        rows.append({
            "method": m, "n_paired": int(len(pair)),
            "f1": f1_abs, "f1_lo": f1_abs + lo - d_f1, "f1_hi": f1_abs + hi - d_f1,
            "f1_delta": d_f1, "f1_delta_lo": lo, "f1_delta_hi": hi, "f1_p_boot": p,
            "em_p_mcnemar": p_em,
            "grounding_delta": d_g, "grounding_delta_lo": g_lo, "grounding_delta_hi": g_hi,
            "grounding_p_boot": p_g,
            "e2e_delta_ms": d_t, "e2e_delta_lo": t_lo, "e2e_delta_hi": t_hi,
        })
        pv_f1[m], pv_em[m] = p, p_em
    sig_f1, sig_em = holm(pv_f1), holm(pv_em)
    for r in rows:
        r["holm_sig"] = bool(sig_f1.get(r["method"]))
        r["holm_sig_em"] = bool(sig_em.get(r["method"]))
    return rows


# --- write-up --------------------------------------------------------------

def _md_table(df: pd.DataFrame, cols: List[str], digits: int = 3) -> str:
    d = df[[c for c in cols if c in df.columns]].copy()
    if "method" in d:
        d["method"] = d["method"].apply(F.lab)
    return d.round(digits).to_markdown(index=False)


def write_report(out: str, summary: Dict, by: pd.DataFrame, s: pd.DataFrame,
                 stats_rows: List[Dict], figs: List[str], comp: pd.DataFrame) -> None:
    be = summary.get("break_even", {})
    tf, pf = summary.get("tpot_fit", {}), summary.get("prefill_fit", {})
    L = []
    L.append("# DECAF — Full Experimental Results\n")
    L.append(f"CPU-only · {summary.get('hardware','')} · "
             f"{summary.get('n_scaling',0)} scaling runs, {summary.get('n_compression',0)} "
             f"compression runs, {summary.get('n_kv',0)} KV-study runs\n")
    L.append("> Absolute latencies are machine-specific. The **shapes, crossovers and "
             "break-even behaviour** are the transferable results.\n")

    L.append("\n## 1. Decode scaling (RQ1 / E1, E8)\n")
    L.append(f"![](fig01_scaling_core.png)\n")
    L.append(f"- `TPOT(L) ≈ {tf.get('intercept_ms',float('nan')):.2f} + "
             f"{tf.get('slope_ms_per_tok',0)*1000:.3f}` ms per 1k context tokens "
             f"(r²={tf.get('r2',float('nan')):.2f})")
    L.append(f"- `prefill(L) ≈ {pf.get('slope_ms_per_tok',float('nan')):.2f}` ms/token "
             f"(r²={pf.get('r2',float('nan')):.2f})")
    if np.isfinite(summary.get("decode_prefill_crossover_L", float("nan"))):
        L.append(f"- decode/prefill crossover at **L ≈ "
                 f"{summary['decode_prefill_crossover_L']:.0f} tokens** "
                 f"(N_out = {summary.get('median_output_tokens',0):.0f})")
    L.append("\n" + _md_table(s, ["prompt_tokens", "prefill_ms", "decode_ms", "tpot_ms",
                                  "ttft_ms", "e2e_ms", "decode_share",
                                  "prefill_ms_per_tok", "achieved_bw_gbs", "n"], 2) + "\n")
    for f in ["fig02_genlen_crossover.png", "fig03_tpot_by_genlen.png",
              "fig04_latency_decomposition.png", "fig05_roofline.png",
              "fig06_quant_scaling.png"]:
        if f in figs:
            L.append(f"![]({f})\n")
    if summary.get("stream"):
        st = summary["stream"]
        L.append(f"- measured STREAM ceiling: **{st['stream_peak_gbs']:.1f} GB/s** "
                 f"(copy {st['stream_copy_gbs']:.1f}, triad {st['stream_triad_gbs']:.1f}); "
                 f"achieved decode bandwidth is in the table above (E8).\n")

    L.append("\n## 2. Break-even context length L* (RQ2 / E3, E4)\n")
    if "fig15_break_even.png" in figs:
        L.append("![](fig15_break_even.png)\n")
    for c in be.get("per_config", []):
        L.append(f"- **{c['quant']}** (N_out={c['n_out']:.0f}): retained `L_c`≈{c['L_c']:.0f} tok, "
                 f"`T_compress`={c['T_compress_ms']:.2f} ms, marginal cost "
                 f"{c['slope_ms_per_tok']:.2f} ms/token → **`L*` ≈ {c['L_star_tokens']:.0f} tokens** "
                 f"(prefill {c['marginal_prefill_ms_per_tok']:.2f} + decode "
                 f"{c['marginal_decode_ms_per_tok']:.2f} ms/token)")
    L.append("")

    L.append("\n## 3. Compression: quality, grounding, latency (RQ3 / E2, E7)\n")
    for f in ["fig07_ratio_quality.png", "fig08_pareto.png", "fig09_method_bars.png",
              "fig10_latency_stack.png", "fig11_overhead_speedup.png",
              "fig17_grounding_vs_accuracy.png"]:
        if f in figs:
            L.append(f"![]({f})\n")
    L.append(_md_table(by, ["method", "f1", "em", "contains_gold", "answer_recall",
                            "evidence_recall", "evidence_precision",
                            "gold_in_evidence", "retained_ratio", "compress_ms",
                            "prefill_ms", "decode_ms", "e2e_ms", "e2e_total_ms",
                            "e2e_norm_ms", "net_speedup", "overhead_frac",
                            "QE_f1_per_sec", "n"]) + "\n")
    L.append("`contains_gold` and `answer_recall` are verbosity-robust companions to EM/F1; "
             "`e2e_norm_ms` fixes the generation length at 32 tokens so a method cannot look "
             "faster merely by emitting a shorter answer.\n")

    L.append("\n## 4. Component analysis and λ sweep (Sec. 17)\n")
    for f in ["fig13_lambda_sweep.png", "fig14_ablation.png", "fig12_scorer_tradeoff.png"]:
        if f in figs:
            L.append(f"![]({f})\n")

    L.append("\n## 5. Statistics (Sec. 21)\n")
    if stats_rows:
        if "fig21_significance.png" in figs:
            L.append("![](fig21_significance.png)\n")
        sdf = pd.DataFrame(stats_rows)
        L.append(_md_table(sdf, ["method", "n_paired", "f1_delta", "f1_delta_lo",
                                 "f1_delta_hi", "f1_p_boot", "holm_sig",
                                 "em_p_mcnemar", "grounding_delta", "e2e_delta_ms"], 4) + "\n")
        L.append("All comparisons are paired per query against full context; "
                 "F1/grounding use a 4000-sample paired bootstrap, EM uses exact McNemar, "
                 "and `holm_sig` applies Holm–Bonferroni across the method family.\n")

    L.append("\n## 6. Quantization interaction (RQ4 / E5)\n")
    if "fig16_quant_compression.png" in figs:
        L.append("![](fig16_quant_compression.png)\n")
    if comp is not None and not comp.empty and comp["quant"].nunique() > 1:
        q = (comp.groupby(["quant", "method"]).agg(f1=("f1", "mean"),
             gold_in_evidence=("gold_in_evidence", "mean"), e2e_ms=("e2e_ms", "median"),
             tpot_ms=("tpot_ms", "median")).reset_index())
        L.append(_md_table(q, ["quant", "method", "f1", "gold_in_evidence",
                               "e2e_ms", "tpot_ms"]) + "\n")
    else:
        L.append("_Single quantization measured; run `--secondary` for the Q8 comparison._\n")

    L.append("\n## 7. Secondary analysis: KV reuse and prefetch (H5)\n")
    if "fig18_kv_reuse.png" in figs:
        L.append("![](fig18_kv_reuse.png)\n")
    if summary.get("kv_study"):
        k = summary["kv_study"]
        L.append(f"- prompt-prefix reuse changes TTFT by **{k['ttft_pct']:.1f}%**, "
                 f"prefill by {k['prefill_pct']:.1f}%, decode by {k['decode_pct']:.1f}%, "
                 f"E2E by **{k['e2e_pct']:.1f}%**")
        L.append(f"- retrieval is {k['retrieve_pct_of_e2e']:.2f}% of E2E, so prefetch can hide "
                 f"at most that much\n")
    else:
        L.append("_Not yet run._\n")

    L.append("\n## 8. Measurement hygiene\n")
    for f in ["fig19_contention.png", "fig20_variance.png"]:
        if f in figs:
            L.append(f"![]({f})\n")
    L.append(f"- contention guard: {summary.get('contention_note','')}")
    L.append(f"- num_ctx held constant within each family "
             f"(scaling {summary.get('num_ctx_scaling')}, compression "
             f"{summary.get('num_ctx_compression')}) so KV allocation does not confound L.")
    L.append("- every measured run carries a unique prefix nonce, so no prefill is reused "
             "except in the KV study where reuse is the object of measurement.\n")

    L.append("\n## 9. Extended figures\n")
    for f in ["fig23_crossover_heatmap.png", "fig24_real_context_scaling.png",
              "fig25_metric_heatmap.png", "fig26_metric_correlation.png",
              "fig27_qtype_breakdown.png", "fig28_win_loss.png",
              "fig29_compressor_cost.png", "fig30_answer_length.png",
              "fig31_norm_vs_raw_latency.png", "fig32_tradeoff.png",
              "fig33_grounding_components.png", "fig34_lambda_frontier.png",
              "fig35_study_cost.png", "fig22_dashboard.png"]:
        if f in figs:
            L.append(f"![]({f})\n")

    L.append("\n## 10. Figure index\n")
    for f in figs:
        L.append(f"- `{f}`")
    with open(os.path.join(out, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-stream", action="store_true", help="skip the STREAM bandwidth probe")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = os.path.join(ROOT, cfg["paths"]["results"])
    os.makedirs(out, exist_ok=True)
    model = args.model or cfg["models"]["primary"]

    sc = pd.DataFrame(JsonlCache(cfg["paths"]["scaling_cache"],
                                 ["mode", "model", "L", "n_out", "rep"]).all())
    comp = pd.DataFrame(JsonlCache(cfg["paths"]["compress_cache"],
                                   ["mode", "model", "qid", "method"]).all())
    kv = pd.DataFrame(JsonlCache(cfg["paths"]["kv_cache"],
                                 ["mode", "model", "qid", "variant", "rep"]).all())
    if sc.empty:
        raise SystemExit("no scaling runs yet — run `python -m src.run_experiments --mode scaling`")

    summary: Dict = {
        "n_scaling": len(sc), "n_compression": len(comp), "n_kv": len(kv),
        "model_primary": model,
        "num_ctx_scaling": cfg["scaling"]["num_ctx"],
        "num_ctx_compression": cfg["compression"]["num_ctx"],
        "hardware": summary_hw(),
    }

    base_n = cfg["scaling"]["n_out_base"]
    sc_m = sc[sc["model"] == model]
    sc_base = sc_m[sc_m["n_out"] == base_n] if "n_out" in sc_m else sc_m
    s, fits = fit_scaling(sc_base if not sc_base.empty else sc_m)
    summary["tpot_fit"], summary["prefill_fit"] = fits.get("tpot", {}), fits.get("prefill", {})
    summary["scaling_table"] = s.round(3).to_dict(orient="records")

    N_out = float(sc_base["output_tokens"].median()) if not sc_base.empty else float(base_n)
    summary["median_output_tokens"] = N_out
    bt = fits.get("tpot", {}).get("slope_ms_per_tok", 0.0)
    at = fits.get("tpot", {}).get("intercept_ms", 0.0)
    bp = fits.get("prefill", {}).get("slope_ms_per_tok", 0.0)
    apre = fits.get("prefill", {}).get("intercept_ms", 0.0)
    denom = N_out * bt - bp
    summary["decode_prefill_crossover_L"] = float((apre - N_out * at) / denom) if denom else float("nan")

    if not args.no_stream:
        summary["stream"] = telemetry.stream_bandwidth_gbs()

    by = aggregate_methods(comp, model) if not comp.empty else pd.DataFrame()
    stats_rows = method_stats(comp, model, seed=cfg["seed"]) if not comp.empty else []
    summary["methods"] = by.to_dict(orient="records") if not by.empty else []
    summary["stats"] = stats_rows
    summary["break_even"] = break_even(sc, comp, cfg) if not comp.empty else {}

    if not kv.empty and kv["variant"].nunique() > 1:
        g = kv.groupby("variant").agg(ttft=("ttft_ms", "median"), prefill=("prefill_ms", "median"),
                                      decode=("decode_ms", "median"), e2e=("e2e_ms", "median"),
                                      retrieve=("t_retrieve_ms", "median"))
        c, w = g.loc["cold"], g.loc["warm"]
        summary["kv_study"] = {
            "ttft_pct": 100 * (c["ttft"] - w["ttft"]) / max(1e-9, c["ttft"]),
            "prefill_pct": 100 * (c["prefill"] - w["prefill"]) / max(1e-9, c["prefill"]),
            "decode_pct": 100 * (c["decode"] - w["decode"]) / max(1e-9, c["decode"]),
            "e2e_pct": 100 * (c["e2e"] - w["e2e"]) / max(1e-9, c["e2e"]),
            "retrieve_pct_of_e2e": 100 * g["retrieve"].median() / max(1e-9, c["e2e"]),
        }
    if "sys_load1" in sc:
        summary["contention_note"] = (
            f"median load1 during runs = {float(sc['sys_load1'].median()):.2f}, "
            f"median free RAM = {float(sc['sys_mem_avail_gb'].median()):.1f} GB")

    # ---------------- figures ----------------
    figs: List[str] = []

    def add(x: Optional[str]) -> None:
        if x:
            figs.append(x)

    add(F.fig_scaling_core(s, fits, out, summary.get("decode_prefill_crossover_L")))
    add(F.fig_genlen_crossover(sc_m, out))
    add(F.fig_tpot_by_genlen(sc_m, out))
    add(F.fig_latency_decomposition_scaling(s, out))
    add(F.fig_roofline(sc, summary.get("stream", {}), out))
    add(F.fig_quant_scaling(sc, out))
    if not by.empty:
        add(F.fig_ratio_quality(by, out))
        add(F.fig_pareto(by, out))
        add(F.fig_method_bars(by, METHOD_ORDER, out))
        add(F.fig_latency_stack_methods(by, METHOD_ORDER, out))
        add(F.fig_overhead_speedup(by, out))
        add(F.fig_scorer_tradeoff(by, out))
        add(F.fig_lambda_sweep(by, out))
        add(F.fig_ablation(by, stats_rows, out))
        add(F.fig_break_even(summary.get("break_even", {}), out))
        add(F.fig_quant_compression(comp, out))
        add(F.fig_grounding_vs_accuracy(by, out))
        add(F.fig_significance(stats_rows, out))
        add(F.fig_variance(comp[comp["model"] == model], METHOD_ORDER, out))
    # extended set
    add(F.fig_crossover_heatmap(sc_m, out))
    cm = comp[comp["model"] == model] if not comp.empty else comp
    add(F.fig_real_context_scaling(cm, out))
    if not by.empty:
        add(F.fig_metric_heatmap(by, METHOD_ORDER, out))
        add(F.fig_metric_correlation(cm, out))
        add(F.fig_qtype_breakdown(cm, METHOD_ORDER, out))
        add(F.fig_win_loss(cm, METHOD_ORDER, out))
        add(F.fig_compressor_cost_breakdown(by, METHOD_ORDER, out))
        add(F.fig_answer_length(by, METHOD_ORDER, out))
        add(F.fig_norm_vs_raw_latency(by, METHOD_ORDER, out))
        add(F.fig_tradeoff_practitioner(by, out))
        add(F.fig_grounding_components(by, METHOD_ORDER, out))
        add(F.fig_lambda_frontier(by, out))
    add(F.fig_kv_reuse(kv, out))
    add(F.fig_contention({"scaling": sc, "compression": comp}, out))
    add(F.fig_study_cost({"scaling": sc, "compression": comp, "kv": kv}, out))
    if not by.empty:
        add(F.fig_dashboard(summary, by, s, out))

    summary["figures"] = figs
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    if not by.empty:
        by.to_csv(os.path.join(out, "methods.csv"), index=False)
        s.to_csv(os.path.join(out, "scaling.csv"), index=False)
        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(os.path.join(out, "stats.csv"), index=False)
    write_report(out, summary, by, s, stats_rows, figs, comp)
    print(f"Wrote {len(figs)} figures + RESULTS.md to {out}")


def summary_hw() -> str:
    try:
        import platform
        import psutil
        name = ""
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        return (f"{name}, {psutil.cpu_count(logical=False)} cores / "
                f"{psutil.cpu_count()} threads, "
                f"{psutil.virtual_memory().total/1e9:.0f} GB RAM, {platform.system()}")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
