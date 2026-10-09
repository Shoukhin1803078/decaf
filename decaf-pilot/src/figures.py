"""Figure generation for the DECAF study.

Every function takes already-aggregated frames and returns the filename it wrote
(or None when the required data is absent), so the analysis runs after each
experiment phase instead of only at the end.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 10, "legend.frameon": False, "savefig.bbox": "tight",
})

C = {"q4": "#4c72b0", "q8": "#c44e52", "decode": "#c0392b", "prefill": "#4c72b0",
     "compress": "#dd8452", "retrieve": "#937860", "grounding": "#55a868",
     "quality": "#8172b3", "accent": "#da8bc3", "grey": "#8c8c8c"}

LABEL = {
    "full": "Full context", "closed_book": "Closed book (no ctx)",
    "fixed_0.8": "Fixed 80%", "fixed_0.6": "Fixed 60%", "fixed_0.5": "Fixed 50%",
    "fixed_0.4": "Fixed 40%", "fixed_0.2": "Fixed 20%",
    "fixed_ce_0.2": "Fixed 20% (CE)", "rel_iso": "Relevance top-m (iso)",
    "decaf_lam2": "DECAF λ=2", "decaf_lam4": "DECAF λ=4", "decaf_lam6": "DECAF λ=6",
    "decaf_lam10": "DECAF λ=10", "decaf_lam15": "DECAF λ=15",
    "decaf_nocov_lam6": "DECAF −coverage", "decaf_nocost_lam6": "DECAF −CPU-cost",
    "decaf_noreorder_lam6": "DECAF −reorder",
    "decaf_absms_lam0.001": "DECAF abs-ms λ=.001",
    "decaf_absms_lam0.002": "DECAF abs-ms λ=.002",
    "decaf_absms_lam0.005": "DECAF abs-ms λ=.005",
    "decaf_ce_lam6": "DECAF λ=6 (CE)", "decaf_ce_nocost_lam6": "DECAF −cost (CE)",
}


def lab(m: str) -> str:
    return LABEL.get(m, m)


def _save(fig, out: str, name: str) -> str:
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    return name


def _pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices on the lower-x / higher-y frontier (minimise latency, maximise quality)."""
    idx = np.argsort(x)
    best, keep = -np.inf, []
    for i in idx:
        if y[i] > best:
            keep.append(i)
            best = y[i]
    return np.array(keep, dtype=int)


# ======================= scaling family (E1 / E8) ==========================

def fig_scaling_core(s: pd.DataFrame, fits: Dict, out: str,
                     crossover: Optional[float] = None) -> str:
    fig, ax = plt.subplots(2, 2, figsize=(9.4, 6.2))
    L = s["prompt_tokens"].values
    xs = np.linspace(L.min(), L.max(), 100)

    ax[0, 0].plot(L, s["tpot_ms"], "o-", color=C["decode"])
    if fits.get("tpot"):
        f = fits["tpot"]
        ax[0, 0].plot(xs, f["intercept_ms"] + f["slope_ms_per_tok"] * xs, "k--", lw=1,
                      label=f"{f['intercept_ms']:.1f}+{f['slope_ms_per_tok']*1000:.2f}/1k tok (r²={f['r2']:.2f})")
        ax[0, 0].legend(fontsize=7)
    ax[0, 0].set(xlabel="context length L (tokens)", ylabel="TPOT (ms/token)",
                 title="Decode cost per token vs L")

    ax[0, 1].plot(L, s["prefill_ms"], "o-", color=C["prefill"])
    if fits.get("prefill"):
        f = fits["prefill"]
        ax[0, 1].plot(xs, f["intercept_ms"] + f["slope_ms_per_tok"] * xs, "k--", lw=1,
                      label=f"{f['slope_ms_per_tok']:.2f} ms/tok (r²={f['r2']:.2f})")
        ax[0, 1].legend(fontsize=7)
    ax[0, 1].set(xlabel="context length L (tokens)", ylabel="prefill (ms)",
                 title="Prefill cost vs L")

    ax[1, 0].plot(L, s["decode_share"], "o-", color=C["quality"])
    ax[1, 0].axhline(0.5, color="k", ls=":", lw=1)
    if crossover and np.isfinite(crossover) and L.min() <= crossover <= L.max() * 3:
        ax[1, 0].axvline(crossover, color="r", ls="--", lw=1,
                         label=f"crossover L≈{crossover:.0f}")
        ax[1, 0].legend(fontsize=7)
    ax[1, 0].set(xlabel="context length L (tokens)", ylabel="decode share of (prefill+decode)",
                 title="Who dominates? decode share vs L  (H1)")

    ax[1, 1].plot(L, s["ttft_ms"], "o-", color=C["grounding"], label="TTFT")
    ax[1, 1].plot(L, s["e2e_ms"], "s--", color=C["grey"], lw=1, label="E2E")
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set(xlabel="context length L (tokens)", ylabel="ms", title="TTFT and E2E vs L")
    fig.suptitle("E1/E8 — decode-scaling characterization", y=1.0, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig01_scaling_core.png")


def fig_genlen_crossover(sc: pd.DataFrame, out: str) -> Optional[str]:
    """Decode share vs L for each generation length — the proper H1 test."""
    if "n_out" not in sc or sc["n_out"].nunique() < 2:
        return None
    g = (sc.groupby(["n_out", "L"])
           .agg(prompt_tokens=("prompt_tokens", "median"),
                prefill_ms=("prefill_ms", "median"),
                decode_ms=("decode_ms", "median"),
                tpot_ms=("tpot_ms", "median")).reset_index())
    g["decode_share"] = g["decode_ms"] / (g["decode_ms"] + g["prefill_ms"]).clip(lower=1e-9)
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, g["n_out"].nunique()))

    cross = []
    for col, (n_out, d) in zip(cmap, g.groupby("n_out")):
        d = d.sort_values("prompt_tokens")
        ax[0].plot(d["prompt_tokens"], d["decode_share"], "o-", color=col,
                   label=f"N_out={n_out:.0f}")
        ax[1].plot(d["prompt_tokens"], d["decode_ms"] + d["prefill_ms"], "o-", color=col,
                   label=f"N_out={n_out:.0f}")
        # crossover where decode share crosses 0.5
        x, y = d["prompt_tokens"].values, d["decode_share"].values
        for i in range(len(x) - 1):
            if (y[i] - 0.5) * (y[i + 1] - 0.5) < 0:
                t = (0.5 - y[i]) / (y[i + 1] - y[i])
                cross.append({"n_out": n_out, "L_cross": x[i] + t * (x[i + 1] - x[i])})
                break
    ax[0].axhline(0.5, color="k", ls=":", lw=1)
    ax[0].set(xlabel="context length L (tokens)", ylabel="decode share",
              title="Decode share vs L, per generation length", xscale="log")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="context length L (tokens)", ylabel="prefill+decode (ms)",
              title="Total compute time vs L", xscale="log", yscale="log")
    ax[1].legend(fontsize=7)

    if cross:
        cd = pd.DataFrame(cross).sort_values("n_out")
        ax[2].plot(cd["n_out"], cd["L_cross"], "o-", color=C["decode"])
        ax[2].set(xlabel="generation length N_out (tokens)",
                  ylabel="crossover context length L (tokens)",
                  title="Where decode starts to dominate")
        for _, r in cd.iterrows():
            ax[2].annotate(f"{r['L_cross']:.0f}", (r["n_out"], r["L_cross"]),
                           textcoords="offset points", xytext=(4, 4), fontsize=7)
    else:
        ax[2].text(0.5, 0.5, "no 0.5 crossing in the\nmeasured range\n(prefill dominates throughout)",
                   ha="center", va="center", transform=ax[2].transAxes, fontsize=9)
        ax[2].set(title="Where decode starts to dominate")
    fig.suptitle("H1 — decode vs prefill dominance depends on generation length, not just L",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig02_genlen_crossover.png")


def fig_tpot_by_genlen(sc: pd.DataFrame, out: str) -> Optional[str]:
    if "n_out" not in sc or sc["n_out"].nunique() < 2:
        return None
    g = sc.groupby(["n_out", "L"]).agg(prompt_tokens=("prompt_tokens", "median"),
                                       tpot_ms=("tpot_ms", "median")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, g["n_out"].nunique()))
    for col, (n_out, d) in zip(cmap, g.groupby("n_out")):
        d = d.sort_values("prompt_tokens")
        ax[0].plot(d["prompt_tokens"], d["tpot_ms"], "o-", color=col, label=f"N_out={n_out:.0f}")
    ax[0].set(xlabel="context length L (tokens)", ylabel="TPOT (ms/token)",
              title="TPOT(L) is stable across generation length")
    ax[0].legend(fontsize=7)
    b = g.groupby("n_out")["tpot_ms"].median().reset_index()
    ax[1].bar(b["n_out"].astype(str), b["tpot_ms"], color=C["decode"], width=0.6)
    ax[1].set(xlabel="generation length N_out", ylabel="median TPOT (ms/token)",
              title="Median TPOT by generation length")
    fig.tight_layout()
    return _save(fig, out, "fig03_tpot_by_genlen.png")


def fig_latency_decomposition_scaling(s: pd.DataFrame, out: str) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    x = np.arange(len(s))
    ax[0].bar(x, s["prefill_ms"], color=C["prefill"], label="prefill")
    ax[0].bar(x, s["decode_ms"], bottom=s["prefill_ms"], color=C["decode"], label="decode")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{v:.0f}" for v in s["prompt_tokens"]], rotation=45)
    ax[0].set(xlabel="context length L (tokens)", ylabel="ms",
              title="Absolute latency decomposition")
    ax[0].legend(fontsize=7)
    tot = (s["prefill_ms"] + s["decode_ms"]).clip(lower=1e-9)
    ax[1].bar(x, s["prefill_ms"] / tot, color=C["prefill"], label="prefill")
    ax[1].bar(x, s["decode_ms"] / tot, bottom=s["prefill_ms"] / tot, color=C["decode"], label="decode")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{v:.0f}" for v in s["prompt_tokens"]], rotation=45)
    ax[1].set(xlabel="context length L (tokens)", ylabel="fraction of compute time",
              title="Relative decomposition (H1)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig04_latency_decomposition.png")


def fig_roofline(sc: pd.DataFrame, stream: Dict, out: str) -> Optional[str]:
    if "achieved_bw_gbs" not in sc or sc["achieved_bw_gbs"].isna().all():
        return None
    g = (sc.groupby(["quant", "L"])
           .agg(prompt_tokens=("prompt_tokens", "median"),
                bw=("achieved_bw_gbs", "median"),
                kv_gb=("kv_bytes", "median"),
                tpot=("tpot_ms", "median")).reset_index())
    g["kv_gb"] = g["kv_gb"] / 1e9
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7))
    for q, d in g.groupby("quant"):
        d = d.sort_values("prompt_tokens")
        col = C["q8"] if "8" in str(q) else C["q4"]
        ax[0].plot(d["prompt_tokens"], d["bw"], "o-", color=col, label=f"{q} achieved")
        ax[2].plot(d["prompt_tokens"], d["kv_gb"] * 1000, "o-", color=col, label=f"{q} KV")
    peak = stream.get("stream_peak_gbs")
    if peak:
        ax[0].axhline(peak, color="k", ls="--", lw=1, label=f"STREAM peak ≈{peak:.1f} GB/s")
    ax[0].set(xlabel="context length L (tokens)", ylabel="implied bandwidth (GB/s)",
              title="E8 — achieved memory bandwidth vs roofline ceiling")
    ax[0].legend(fontsize=7)

    for q, d in g.groupby("quant"):
        d = d.sort_values("prompt_tokens")
        col = C["q8"] if "8" in str(q) else C["q4"]
        ax[1].plot(d["kv_gb"] * 1000, d["tpot"], "o-", color=col, label=str(q))
    ax[1].set(xlabel="KV-cache size (MB)", ylabel="TPOT (ms/token)",
              title="TPOT vs KV-cache size")
    ax[1].legend(fontsize=7)
    ax[2].set(xlabel="context length L (tokens)", ylabel="KV cache (MB)",
              title="KV-cache growth with L")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig05_roofline.png")


def fig_quant_scaling(sc: pd.DataFrame, out: str) -> Optional[str]:
    if sc["quant"].nunique() < 2:
        return None
    base = sc[sc["n_out"] == sc["n_out"].mode().iloc[0]] if "n_out" in sc else sc
    g = (base.groupby(["quant", "L"])
             .agg(prompt_tokens=("prompt_tokens", "median"),
                  tpot_ms=("tpot_ms", "median"), prefill_ms=("prefill_ms", "median"),
                  e2e_ms=("e2e_ms", "median")).reset_index())
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7))
    for q, d in g.groupby("quant"):
        d = d.sort_values("prompt_tokens")
        col = C["q8"] if "8" in str(q) else C["q4"]
        ax[0].plot(d["prompt_tokens"], d["tpot_ms"], "o-", color=col, label=str(q))
        ax[1].plot(d["prompt_tokens"], d["prefill_ms"], "o-", color=col, label=str(q))
        ax[2].plot(d["prompt_tokens"], d["e2e_ms"], "o-", color=col, label=str(q))
    for a, t, yl in zip(ax, ["TPOT vs L by quantization", "Prefill vs L by quantization",
                             "E2E vs L by quantization"],
                        ["TPOT (ms/token)", "prefill (ms)", "E2E (ms)"]):
        a.set(xlabel="context length L (tokens)", ylabel=yl, title=t)
        a.legend(fontsize=7)
    fig.suptitle("E5 — quantization interaction with the decode-scaling curve", y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig06_quant_scaling.png")


# ==================== compression family (E2 / E3 / E7) ====================

def fig_ratio_quality(by: pd.DataFrame, out: str) -> str:
    """E2 — what retaining less evidence costs, in accuracy and in grounding."""
    d = by.sort_values("retained_ratio")
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7))
    fx = d[d["method"].str.startswith("fixed_") & ~d["method"].str.contains("ce")]
    fx = fx.sort_values("retained_ratio")

    ax[0].plot(d["retained_ratio"], d["f1"], "o", color=C["quality"], alpha=0.55, label="all methods")
    if len(fx):
        ax[0].plot(fx["retained_ratio"], fx["f1"], "s-", color=C["accent"], label="fixed-ratio sweep")
    ax[0].set(xlabel="retained context (fraction of tokens)", ylabel="F1",
              title="Accuracy vs retained evidence")
    ax[0].legend(fontsize=7)

    for col, k, nm in [(C["grounding"], "gold_in_evidence", "gold answer in context"),
                       (C["prefill"], "evidence_recall", "gold-evidence recall"),
                       (C["compress"], "evidence_precision", "gold-evidence precision")]:
        if k in d:
            ax[1].plot(d["retained_ratio"], d[k], "o", color=col, alpha=0.7, label=nm)
    ax[1].set(xlabel="retained context (fraction of tokens)", ylabel="score",
              title="E7 — grounding vs retained evidence")
    ax[1].legend(fontsize=7)

    base = by[by["method"] == "full"]
    if len(base):
        b = base.iloc[0]
        rel = d.copy()
        rel["f1_rel"] = rel["f1"] / max(1e-9, b["f1"])
        rel["gnd_rel"] = rel["gold_in_evidence"] / max(1e-9, b["gold_in_evidence"])
        rel = rel.sort_values("retained_ratio")
        ax[2].plot(rel["retained_ratio"], rel["f1_rel"], "o-", color=C["quality"], label="F1 (rel. to full)")
        ax[2].plot(rel["retained_ratio"], rel["gnd_rel"], "s-", color=C["grounding"],
                   label="grounding (rel. to full)")
        ax[2].axhline(1.0, color="k", ls=":", lw=1)
        ax[2].set(xlabel="retained context (fraction of tokens)", ylabel="relative to full context",
                  title="Grounding degrades faster than accuracy")
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig07_ratio_quality.png")


def _pareto_panel(ax, by, ycol, ylabel, title, color):
    x = by["e2e_total_ms"].values
    y = by[ycol].values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, names = x[ok], y[ok], by["method"].values[ok]
    ax.scatter(x, y, s=34, color=color, alpha=0.8, zorder=3)
    fr = _pareto_front(x, y)
    if len(fr) > 1:
        ax.plot(x[fr], y[fr], "-", color="k", lw=1.1, alpha=0.65, zorder=2, label="Pareto frontier")
        ax.legend(fontsize=7)
    for xi, yi, nm in zip(x, y, names):
        ax.annotate(lab(nm), (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=6.2)
    ax.set(xlabel="end-to-end latency incl. overhead (ms)", ylabel=ylabel, title=title)


def fig_pareto(by: pd.DataFrame, out: str) -> str:
    """RQ3 — the quality/grounding vs latency frontiers (the primary story)."""
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
    _pareto_panel(ax[0], by, "f1", "F1", "Accuracy vs latency", C["quality"])
    _pareto_panel(ax[1], by, "gold_in_evidence", "gold answer in retained context",
                  "Grounding vs latency", C["grounding"])
    ycol = "contains_gold" if "contains_gold" in by.columns else "evidence_recall"
    ylab = ("gold answer present in prediction" if ycol == "contains_gold"
            else "gold-evidence recall")
    _pareto_panel(ax[2], by, ycol, ylab,
                  "Verbosity-robust accuracy vs latency", C["prefill"])
    fig.suptitle("RQ3 — accuracy–latency and grounding–latency Pareto frontiers", y=1.03, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig08_pareto.png")


def fig_method_bars(by: pd.DataFrame, order: List[str], out: str) -> str:
    d = by.set_index("method").reindex([m for m in order if m in set(by["method"])]).reset_index()
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 6.6))
    specs = [("f1", "F1", C["quality"]), ("em", "EM", C["accent"]),
             ("gold_in_evidence", "gold answer in context", C["grounding"]),
             ("e2e_total_ms", "E2E latency incl. overhead (ms)", C["decode"])]
    for a, (k, t, col) in zip(ax.ravel(), specs):
        a.barh(range(len(d)), d[k], color=col, height=0.7)
        a.set_yticks(range(len(d)))
        a.set_yticklabels([lab(m) for m in d["method"]], fontsize=7)
        a.invert_yaxis()
        a.set(xlabel=t, title=t)
        for i, v in enumerate(d[k]):
            if np.isfinite(v):
                a.annotate(f"{v:.3g}", (v, i), textcoords="offset points",
                           xytext=(3, 0), va="center", fontsize=6.2)
    fig.suptitle("Per-method summary", y=1.01, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig09_method_bars.png")


def fig_latency_stack_methods(by: pd.DataFrame, order: List[str], out: str) -> str:
    d = by.set_index("method").reindex([m for m in order if m in set(by["method"])]).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.2))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for k, col, nm in [("t_retrieve_ms", C["retrieve"], "retrieve"),
                       ("compress_ms", C["compress"], "compress (T_compress)"),
                       ("prefill_ms", C["prefill"], "prefill"),
                       ("decode_ms", C["decode"], "decode")]:
        v = d[k].fillna(0).values
        ax[0].barh(y, v, left=left, color=col, label=nm, height=0.72)
        left += v
    ax[0].set_yticks(y); ax[0].set_yticklabels([lab(m) for m in d["method"]], fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="ms", title="E3 — where the time actually goes")
    ax[0].legend(fontsize=7, loc="lower right")

    # the same, normalised — shows T_compress is invisible at this scale
    tot = left.clip(min=1e-9)
    left = np.zeros(len(d))
    for k, col, nm in [("t_retrieve_ms", C["retrieve"], "retrieve"),
                       ("compress_ms", C["compress"], "compress"),
                       ("prefill_ms", C["prefill"], "prefill"),
                       ("decode_ms", C["decode"], "decode")]:
        v = d[k].fillna(0).values / tot
        ax[1].barh(y, v, left=left, color=col, label=nm, height=0.72)
        left += v
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].set(xlabel="fraction of end-to-end time", title="Normalised stage breakdown")
    fig.tight_layout()
    return _save(fig, out, "fig10_latency_stack.png")


def fig_overhead_speedup(by: pd.DataFrame, out: str) -> str:
    """E3 — compression overhead O_c against the net speedup S_net it buys."""
    d = by[by["method"] != "closed_book"].copy()
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax[0].scatter(d["retained_ratio"], d["net_speedup"], s=34, color=C["prefill"])
    for _, r in d.iterrows():
        ax[0].annotate(lab(r["method"]), (r["retained_ratio"], r["net_speedup"]),
                       textcoords="offset points", xytext=(4, 3), fontsize=6.2)
    ax[0].axhline(1.0, color="r", ls="--", lw=1, label="break-even (S_net=1)")
    ax[0].set(xlabel="retained context (fraction)", ylabel="net speedup S_net (×)",
              title="E3 — net speedup vs retained context")
    ax[0].legend(fontsize=7)

    ax[1].barh(range(len(d)), d["overhead_frac"] * 100, color=C["compress"], height=0.7)
    ax[1].set_yticks(range(len(d)))
    ax[1].set_yticklabels([lab(m) for m in d["method"]], fontsize=6.5)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="compressor overhead O_c (% of full-context E2E)",
              title="Compressor cost as a share of the baseline")

    ax[2].scatter(d["compress_ms"], d["net_speedup"], s=34, color=C["decode"])
    for _, r in d.iterrows():
        ax[2].annotate(lab(r["method"]), (r["compress_ms"], r["net_speedup"]),
                       textcoords="offset points", xytext=(4, 3), fontsize=6.2)
    ax[2].axhline(1.0, color="r", ls="--", lw=1)
    ax[2].set(xlabel="T_compress (ms, log)", ylabel="net speedup S_net (×)", xscale="log",
              title="Does a costlier compressor pay for itself?")
    fig.tight_layout()
    return _save(fig, out, "fig11_overhead_speedup.png")


def fig_scorer_tradeoff(by: pd.DataFrame, out: str) -> Optional[str]:
    """The proposal's central design tension: scorer quality vs T_compress."""
    d = by.copy()
    d["scorer"] = np.where(d["method"].str.contains("_ce"), "cross-encoder", "BM25")
    pairs = [("decaf_lam6", "decaf_ce_lam6"), ("decaf_nocost_lam6", "decaf_ce_nocost_lam6"),
             ("fixed_0.2", "fixed_ce_0.2")]
    have = [(a, b) for a, b in pairs
            if a in set(d["method"]) and b in set(d["method"])]
    if not have:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8))
    w = 0.35
    x = np.arange(len(have))
    for off, sc, col in [(-w / 2, 0, C["prefill"]), (w / 2, 1, C["decode"])]:
        vals_t = [d[d["method"] == p[sc]]["compress_ms"].iloc[0] for p in have]
        vals_f = [d[d["method"] == p[sc]]["f1"].iloc[0] for p in have]
        vals_g = [d[d["method"] == p[sc]]["gold_in_evidence"].iloc[0] for p in have]
        nm = "BM25" if sc == 0 else "cross-encoder"
        ax[0].bar(x + off, vals_t, w, color=col, label=nm)
        ax[1].bar(x + off, vals_f, w, color=col, label=nm)
        ax[2].bar(x + off, vals_g, w, color=col, label=nm)
    names = [lab(a).replace(" (CE)", "") for a, _ in have]
    for a, t, yl in zip(ax, ["T_compress by scorer (log)", "F1 by scorer", "Grounding by scorer"],
                        ["T_compress (ms)", "F1", "gold answer in context"]):
        a.set_xticks(x); a.set_xticklabels(names, fontsize=7, rotation=15)
        a.set(ylabel=yl, title=t)
        a.legend(fontsize=7)
    ax[0].set_yscale("log")
    fig.suptitle("Scorer ablation — does a 110M cross-encoder earn its CPU time?",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig12_scorer_tradeoff.png")


def fig_lambda_sweep(by: pd.DataFrame, out: str) -> Optional[str]:
    rel = by[by["method"].str.match(r"decaf_lam[\d.]+$")].copy()
    abs_ms = by[by["method"].str.contains("absms")].copy()
    if rel.empty and abs_ms.empty:
        return None
    for d in (rel, abs_ms):
        if not d.empty:
            d["lam"] = d["method"].str.extract(r"lam([\d.]+)").astype(float)
            d.sort_values("lam", inplace=True)
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for d, nm, col in [(rel, "λ (normalised cost)", C["quality"]),
                       (abs_ms, "λ (ms-absolute cost)", C["decode"])]:
        if d.empty:
            continue
        ax[0].plot(d["lam"], d["retained_ratio"], "o-", color=col, label=nm)
        ax[1].plot(d["lam"], d["f1"], "o-", color=col, label=nm)
        ax[2].plot(d["lam"], d["e2e_total_ms"], "o-", color=col, label=nm)
        if "gold_in_evidence" in d:
            ax[1].plot(d["lam"], d["gold_in_evidence"], "s--", color=col, alpha=0.6,
                       label=f"{nm} grounding")
    for a, t, yl, xs in zip(ax, ["λ controls how much is retained", "λ vs accuracy and grounding",
                                 "λ vs end-to-end latency"],
                            ["retained fraction", "score", "E2E incl. overhead (ms)"],
                            ["log", "log", "log"]):
        a.set(xlabel="λ (budget knob)", ylabel=yl, title=t, xscale=xs)
        a.legend(fontsize=6.5)
    fig.suptitle("Sec.17 — λ sweep for both cost modes", y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig13_lambda_sweep.png")


def fig_ablation(by: pd.DataFrame, stats_rows: List[Dict], out: str) -> Optional[str]:
    """Sec.17 component analysis, with the iso-budget baseline as the sharp test."""
    members = ["rel_iso", "decaf_lam6", "decaf_nocov_lam6", "decaf_nocost_lam6",
               "decaf_noreorder_lam6", "fixed_0.2"]
    d = by[by["method"].isin(members)].copy()
    if d.empty:
        return None
    d["o"] = d["method"].apply(lambda m: members.index(m))
    d = d.sort_values("o")
    fig, ax = plt.subplots(1, 4, figsize=(14.2, 3.8))
    specs = [("f1", "F1", C["quality"]), ("gold_in_evidence", "gold in context", C["grounding"]),
             ("evidence_recall", "evidence recall", C["prefill"]),
             ("retained_ratio", "retained fraction", C["compress"])]
    ci = {r["method"]: r for r in stats_rows} if stats_rows else {}
    x = np.arange(len(d))
    for a, (k, t, col) in zip(ax, specs):
        err = None
        if k == "f1" and ci:
            lo = [max(0, d[d['method']==m]['f1'].iloc[0] - (ci.get(m, {}).get("f1_lo") or np.nan))
                  for m in d["method"]]
            hi = [max(0, (ci.get(m, {}).get("f1_hi") or np.nan) - d[d['method']==m]['f1'].iloc[0])
                  for m in d["method"]]
            if not all(np.isnan(lo)):
                err = [np.nan_to_num(lo), np.nan_to_num(hi)]
        a.bar(x, d[k], color=col, width=0.65, yerr=err, capsize=3)
        a.set_xticks(x); a.set_xticklabels([lab(m) for m in d["method"]], rotation=35,
                                           ha="right", fontsize=6.5)
        a.set(ylabel=t, title=t)
    fig.suptitle("Sec.17 component analysis — does CPU-awareness beat plain relevance ranking?",
                 y=1.06, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig14_ablation.png")


# ================== break-even, quantization, secondary ====================

def fig_break_even(be: Dict, out: str) -> Optional[str]:
    """E4 — total time vs L for full and compressed context; L* is the crossing."""
    if not be or not be.get("per_config"):
        return None
    cfgs = be["per_config"]
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
    Lmax = max(c["L_max_observed"] for c in cfgs) * 1.15
    xs = np.linspace(0, Lmax, 200)
    for c in cfgs:
        col = C["q8"] if "8" in str(c["quant"]) else C["q4"]
        n = c["n_out"]
        full = c["pre_a"] + c["pre_b"] * xs + n * (c["tpot_a"] + c["tpot_b"] * xs)
        Lc = c["L_c"]
        comp = (c["T_compress_ms"] + c["pre_a"] + c["pre_b"] * Lc
                + n * (c["tpot_a"] + c["tpot_b"] * Lc)) * np.ones_like(xs)
        ax[0].plot(xs, full, "-", color=col, label=f"{c['quant']} full context")
        ax[0].plot(xs, comp, "--", color=col, alpha=0.8,
                   label=f"{c['quant']} compressed (L_c={Lc:.0f})")
        Ls = c["L_star_tokens"]
        if np.isfinite(Ls) and 0 < Ls < Lmax:
            ax[0].axvline(Ls, color=col, ls=":", lw=1.2)
            ax[0].annotate(f"L*={Ls:.0f}", (Ls, ax[0].get_ylim()[1] * 0.55),
                           fontsize=7, color=col, rotation=90,
                           textcoords="offset points", xytext=(3, 0))
    ax[0].set(xlabel="retrieved context length L (tokens)", ylabel="total compute time (ms)",
              title="E4 — break-even: full vs compressed")
    ax[0].legend(fontsize=6.8)

    names = [f"{c['quant']}\nN_out={c['n_out']:.0f}" for c in cfgs]
    vals = [c["L_star_tokens"] for c in cfgs]
    cols = [C["q8"] if "8" in str(c["quant"]) else C["q4"] for c in cfgs]
    ax[1].bar(range(len(cfgs)), vals, color=cols, width=0.55)
    ax[1].set_xticks(range(len(cfgs))); ax[1].set_xticklabels(names, fontsize=7)
    ax[1].set(ylabel="break-even context length L* (tokens)",
              title="L* per model × quantization")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax[1].annotate(f"{v:.0f}", (i, v), ha="center",
                           textcoords="offset points", xytext=(0, 3), fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig15_break_even.png")


def fig_quant_compression(comp: pd.DataFrame, out: str) -> Optional[str]:
    if comp["quant"].nunique() < 2:
        return None
    g = (comp.groupby(["quant", "method"])
             .agg(f1=("f1", "mean"), gold=("gold_in_evidence", "mean"),
                  e2e=("e2e_ms", "median"), tpot=("tpot_ms", "median"),
                  retained=("retained_ratio", "mean")).reset_index())
    shared = sorted(set.intersection(*[set(d["method"]) for _, d in g.groupby("quant")]))
    g = g[g["method"].isin(shared)]
    quants = sorted(g["quant"].unique())
    fig, ax = plt.subplots(1, 4, figsize=(14.6, 3.9))
    x = np.arange(len(shared)); w = 0.38
    for i, q in enumerate(quants):
        d = g[g["quant"] == q].set_index("method").reindex(shared).reset_index()
        col = C["q8"] if "8" in str(q) else C["q4"]
        off = (i - (len(quants) - 1) / 2) * w
        ax[0].bar(x + off, d["f1"], w, color=col, label=str(q))
        ax[1].bar(x + off, d["gold"], w, color=col, label=str(q))
        ax[2].bar(x + off, d["e2e"], w, color=col, label=str(q))
        ax[3].bar(x + off, d["tpot"], w, color=col, label=str(q))
    for a, t, yl in zip(ax, ["F1 by quantization", "Grounding by quantization",
                             "E2E latency by quantization", "TPOT by quantization"],
                        ["F1", "gold in context", "E2E (ms)", "TPOT (ms/token)"]):
        a.set_xticks(x); a.set_xticklabels([lab(m) for m in shared], rotation=35,
                                           ha="right", fontsize=6.3)
        a.set(ylabel=yl, title=t); a.legend(fontsize=7)
    fig.suptitle("E5 — does quantization change the compression–quality–latency relationship?",
                 y=1.06, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig16_quant_compression.png")


def fig_grounding_vs_accuracy(by: pd.DataFrame, out: str) -> str:
    """The proposal's motivating claim: compression costs grounding faster than accuracy."""
    d = by[by["method"] != "closed_book"].copy()
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0))
    sc = ax[0].scatter(d["f1"], d["gold_in_evidence"], c=d["retained_ratio"],
                       cmap="viridis", s=50, zorder=3)
    for _, r in d.iterrows():
        ax[0].annotate(lab(r["method"]), (r["f1"], r["gold_in_evidence"]),
                       textcoords="offset points", xytext=(4, 3), fontsize=6.2)
    plt.colorbar(sc, ax=ax[0], label="retained fraction")
    ax[0].set(xlabel="F1 (accuracy)", ylabel="gold answer in retained context (grounding)",
              title="Accuracy and grounding are not the same axis")

    base = by[by["method"] == "full"]
    if len(base):
        b = base.iloc[0]
        d2 = d.sort_values("retained_ratio")
        ax[1].plot(d2["retained_ratio"], 100 * (d2["f1"] - b["f1"]) / max(1e-9, b["f1"]),
                   "o-", color=C["quality"], label="Δ F1 (%)")
        ax[1].plot(d2["retained_ratio"],
                   100 * (d2["gold_in_evidence"] - b["gold_in_evidence"]) / max(1e-9, b["gold_in_evidence"]),
                   "s-", color=C["grounding"], label="Δ grounding (%)")
        ax[1].axhline(0, color="k", ls=":", lw=1)
        ax[1].set(xlabel="retained context (fraction)", ylabel="% change vs full context",
                  title="Relative damage: accuracy vs grounding")
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig17_grounding_vs_accuracy.png")


def fig_kv_reuse(kv: pd.DataFrame, out: str) -> Optional[str]:
    """H5 — prefix/KV-cache reuse and how much retrieval a prefetch could hide."""
    if kv is None or kv.empty or kv["variant"].nunique() < 2:
        return None
    g = (kv.groupby("variant")
           .agg(ttft=("ttft_ms", "median"), prefill=("prefill_ms", "median"),
                decode=("decode_ms", "median"), e2e=("e2e_ms", "median"),
                retrieve=("t_retrieve_ms", "median"), n=("e2e_ms", "size")).reset_index())
    order = ["cold", "warm"]
    g["o"] = g["variant"].apply(lambda v: order.index(v) if v in order else 9)
    g = g.sort_values("o")
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.8))
    x = np.arange(len(g)); w = 0.2
    for i, (k, col, nm) in enumerate([("prefill", C["prefill"], "prefill"),
                                      ("decode", C["decode"], "decode"),
                                      ("ttft", C["grounding"], "TTFT"),
                                      ("e2e", C["grey"], "E2E")]):
        ax[0].bar(x + (i - 1.5) * w, g[k], w, color=col, label=nm)
    ax[0].set_xticks(x); ax[0].set_xticklabels(g["variant"])
    ax[0].set(ylabel="ms", title="H5 — effect of prompt-prefix KV reuse")
    ax[0].legend(fontsize=7)

    cold = g[g["variant"] == "cold"].iloc[0]
    warm = g[g["variant"] == "warm"].iloc[0]
    deltas = {"TTFT": 100 * (cold["ttft"] - warm["ttft"]) / max(1e-9, cold["ttft"]),
              "prefill": 100 * (cold["prefill"] - warm["prefill"]) / max(1e-9, cold["prefill"]),
              "decode": 100 * (cold["decode"] - warm["decode"]) / max(1e-9, cold["decode"]),
              "E2E": 100 * (cold["e2e"] - warm["e2e"]) / max(1e-9, cold["e2e"])}
    cols = [C["grounding"], C["prefill"], C["decode"], C["grey"]]
    ax[1].bar(list(deltas), list(deltas.values()), color=cols, width=0.6)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(ylabel="% reduction from reuse", title="Reuse moves TTFT, not decode")
    for i, (k, v) in enumerate(deltas.items()):
        ax[1].annotate(f"{v:.1f}%", (i, v), ha="center", textcoords="offset points",
                       xytext=(0, 3 if v >= 0 else -10), fontsize=7)

    # prefetch: retrieval is hideable only up to the decode time it can overlap
    r = float(g["retrieve"].median()); dec = float(cold["decode"]); e2e = float(cold["e2e"])
    ax[2].bar(["retrieval\n(hideable)", "decode\n(overlap window)", "E2E"],
              [r, dec, e2e], color=[C["retrieve"], C["decode"], C["grey"]], width=0.6)
    ax[2].set(ylabel="ms", yscale="log",
              title=f"Prefetch ceiling: retrieval is {100*r/max(1e-9,e2e):.2f}% of E2E")
    for i, v in enumerate([r, dec, e2e]):
        ax[2].annotate(f"{v:.1f} ms", (i, v), ha="center", textcoords="offset points",
                       xytext=(0, 3), fontsize=7)
    fig.suptitle("Secondary systems analysis (H5) — KV reuse and retrieval prefetch",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig18_kv_reuse.png")


def fig_contention(frames: Dict[str, pd.DataFrame], out: str) -> Optional[str]:
    """Measurement hygiene: machine state recorded alongside every run."""
    d = pd.concat([f.assign(_fam=k) for k, f in frames.items()
                   if f is not None and not f.empty and "sys_load1" in f],
                  ignore_index=True) if frames else None
    if d is None or d.empty:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.6))
    ax[0].plot(np.arange(len(d)), d["sys_load1"], lw=0.8, color=C["decode"])
    ax[0].set(xlabel="run index (in cache order)", ylabel="1-min load average",
              title="CPU load during measurement")
    ax[1].plot(np.arange(len(d)), d["sys_mem_avail_gb"], lw=0.8, color=C["prefill"])
    ax[1].set(xlabel="run index", ylabel="available RAM (GB)", title="Free memory during measurement")
    if "swap_used_gb" in d or "sys_swap_used_gb" in d:
        k = "sys_swap_used_gb" if "sys_swap_used_gb" in d else "swap_used_gb"
        ax[2].plot(np.arange(len(d)), d[k], lw=0.8, color=C["compress"])
        ax[2].set(xlabel="run index", ylabel="swap used (GB)", title="Swap (must stay flat)")
    fig.suptitle("Measurement hygiene — contention telemetry per run", y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig19_contention.png")


def fig_variance(comp: pd.DataFrame, order: List[str], out: str) -> Optional[str]:
    if comp is None or comp.empty:
        return None
    ms = [m for m in order if m in set(comp["method"])]
    data = [comp[comp["method"] == m]["e2e_ms"].dropna().values for m in ms]
    data = [d for d in data if len(d)]
    if not data:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.2))
    bp = ax[0].boxplot(data, vert=False, patch_artist=True, widths=0.6,
                       flierprops={"markersize": 2, "alpha": 0.4})
    for p in bp["boxes"]:
        p.set_facecolor(C["prefill"]); p.set_alpha(0.65)
    ax[0].set_yticklabels([lab(m) for m in ms][:len(data)], fontsize=6.8)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="per-query E2E latency (ms)", title="Latency distribution per method")

    f1s = [comp[comp["method"] == m]["f1"].dropna().values for m in ms]
    f1s = [d for d in f1s if len(d)]
    bp2 = ax[1].boxplot(f1s, vert=False, patch_artist=True, widths=0.6,
                        flierprops={"markersize": 2, "alpha": 0.4})
    for p in bp2["boxes"]:
        p.set_facecolor(C["quality"]); p.set_alpha(0.65)
    ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set(xlabel="per-query F1", title="Per-query F1 distribution")
    fig.tight_layout()
    return _save(fig, out, "fig20_variance.png")


def fig_significance(stats_rows: List[Dict], out: str) -> Optional[str]:
    """Paired bootstrap F1 deltas vs full context, with Holm-corrected significance."""
    rows = [r for r in stats_rows if np.isfinite(r.get("f1_delta", np.nan))]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["f1_delta"])
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.32 * len(rows))))
    y = np.arange(len(rows))
    d = np.array([r["f1_delta"] for r in rows])
    lo = np.array([r.get("f1_delta_lo", np.nan) for r in rows])
    hi = np.array([r.get("f1_delta_hi", np.nan) for r in rows])
    sig = [bool(r.get("holm_sig")) for r in rows]
    cols = [C["grounding"] if s else C["grey"] for s in sig]
    ax.barh(y, d, color=cols, height=0.66,
            xerr=[np.nan_to_num(d - lo), np.nan_to_num(hi - d)], capsize=2.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([lab(r["method"]) for r in rows], fontsize=7)
    ax.set(xlabel="Δ F1 vs full context (paired bootstrap, 95% CI)",
           title="Which differences survive Holm correction?\n(green = significant)")
    fig.tight_layout()
    return _save(fig, out, "fig21_significance.png")


def fig_dashboard(summary: Dict, by: pd.DataFrame, s: pd.DataFrame, out: str) -> str:
    fig = plt.figure(figsize=(12.6, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

    a = fig.add_subplot(gs[0, 0])
    a.plot(s["prompt_tokens"], s["prefill_ms"], "o-", color=C["prefill"], label="prefill")
    a.plot(s["prompt_tokens"], s["decode_ms"], "o-", color=C["decode"], label="decode")
    a.set(xlabel="L (tokens)", ylabel="ms", title="Prefill vs decode", yscale="log")
    a.legend(fontsize=7)

    a = fig.add_subplot(gs[0, 1])
    a.plot(s["prompt_tokens"], s["decode_share"], "o-", color=C["quality"])
    a.axhline(0.5, color="k", ls=":", lw=1)
    a.set(xlabel="L (tokens)", ylabel="decode share", title="H1: decode share vs L")

    a = fig.add_subplot(gs[0, 2])
    d = by[by["method"] != "closed_book"]
    a.scatter(d["e2e_total_ms"], d["f1"], s=26, color=C["quality"])
    fr = _pareto_front(d["e2e_total_ms"].values, d["f1"].values)
    if len(fr) > 1:
        a.plot(d["e2e_total_ms"].values[fr], d["f1"].values[fr], "k-", lw=1, alpha=0.6)
    a.set(xlabel="E2E (ms)", ylabel="F1", title="Accuracy–latency frontier")

    a = fig.add_subplot(gs[1, 0])
    a.scatter(d["retained_ratio"], d["gold_in_evidence"], s=26, color=C["grounding"])
    a.set(xlabel="retained fraction", ylabel="gold in context", title="E7: grounding vs retention")

    a = fig.add_subplot(gs[1, 1])
    a.scatter(d["retained_ratio"], d["net_speedup"], s=26, color=C["prefill"])
    a.axhline(1.0, color="r", ls="--", lw=1)
    a.set(xlabel="retained fraction", ylabel="S_net (×)", title="E3: net speedup")

    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    be = summary.get("break_even", {})
    tf = summary.get("tpot_fit", {}); pf = summary.get("prefill_fit", {})
    lines = [
        "DECAF — headline numbers",
        "",
        f"runs: {summary.get('n_scaling',0)} scaling, {summary.get('n_compression',0)} compression",
        f"TPOT(L) = {tf.get('intercept_ms',float('nan')):.1f} + {tf.get('slope_ms_per_tok',0)*1000:.2f}/1k tok"
        f"  (r²={tf.get('r2',float('nan')):.2f})",
        f"prefill(L) = {pf.get('slope_ms_per_tok',float('nan')):.2f} ms/tok"
        f"  (r²={pf.get('r2',float('nan')):.2f})",
        f"decode/prefill crossover: L ≈ {summary.get('decode_prefill_crossover_L',float('nan')):.0f} tok",
    ]
    if be.get("per_config"):
        for c in be["per_config"]:
            lines.append(f"L* ({c['quant']}, N_out={c['n_out']:.0f}) ≈ {c['L_star_tokens']:.0f} tok")
    best = by.loc[by["f1"].idxmax()] if not by.empty else None
    if best is not None:
        lines += ["", f"best F1: {lab(best['method'])} = {best['f1']:.3f}",
                  f"   at {best['e2e_total_ms']:.0f} ms, {best['net_speedup']:.2f}× vs full"]
    a.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.4, family="monospace",
           transform=a.transAxes)
    fig.suptitle("DECAF — summary dashboard", y=0.98, fontsize=12)
    return _save(fig, out, "fig22_dashboard.png")


# ===================== extended figure set ================================

def fig_crossover_heatmap(sc: pd.DataFrame, out: str) -> Optional[str]:
    """Decode share over the full (L, N_out) grid — H1 as a surface, not a line."""
    if "n_out" not in sc or sc["n_out"].nunique() < 2:
        return None
    g = (sc.groupby(["n_out", "L"])
           .agg(prefill=("prefill_ms", "median"), decode=("decode_ms", "median"),
                tpot=("tpot_ms", "median")).reset_index())
    g["share"] = g["decode"] / (g["decode"] + g["prefill"]).clip(lower=1e-9)
    piv = g.pivot(index="n_out", columns="L", values="share").sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.0))
    im = ax[0].imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, origin="lower")
    ax[0].set_xticks(range(len(piv.columns))); ax[0].set_xticklabels(piv.columns, rotation=45)
    ax[0].set_yticks(range(len(piv.index))); ax[0].set_yticklabels([f"{v:.0f}" for v in piv.index])
    ax[0].set(xlabel="context length L (tokens)", ylabel="generation length N_out",
              title="Decode share of compute time")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                           color="white" if abs(v - 0.5) > 0.3 else "black")
    plt.colorbar(im, ax=ax[0], label="decode share (red = decode-dominated)")
    piv2 = g.pivot(index="n_out", columns="L", values="tpot").sort_index()
    im2 = ax[1].imshow(piv2.values, aspect="auto", cmap="viridis", origin="lower")
    ax[1].set_xticks(range(len(piv2.columns))); ax[1].set_xticklabels(piv2.columns, rotation=45)
    ax[1].set_yticks(range(len(piv2.index))); ax[1].set_yticklabels([f"{v:.0f}" for v in piv2.index])
    ax[1].set(xlabel="context length L (tokens)", ylabel="generation length N_out",
              title="TPOT (ms/token)")
    plt.colorbar(im2, ax=ax[1], label="TPOT (ms/token)")
    fig.suptitle("H1 surface — dominance depends jointly on L and N_out", y=1.03, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig23_crossover_heatmap.png")


def fig_real_context_scaling(comp: pd.DataFrame, out: str) -> Optional[str]:
    """The scaling relation re-checked on *real* retrieved contexts, not filler."""
    if comp is None or comp.empty:
        return None
    d = comp.dropna(subset=["prompt_tokens", "prefill_ms", "tpot_ms"])
    if d.empty:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax[0].scatter(d["prompt_tokens"], d["prefill_ms"], s=7, alpha=0.3, color=C["prefill"])
    if len(d) > 2:
        b, a = np.polyfit(d["prompt_tokens"], d["prefill_ms"], 1)
        xs = np.linspace(d["prompt_tokens"].min(), d["prompt_tokens"].max(), 50)
        r2 = float(np.corrcoef(d["prompt_tokens"], d["prefill_ms"])[0, 1] ** 2)
        ax[0].plot(xs, a + b * xs, "k--", lw=1.2, label=f"{b:.2f} ms/tok (r²={r2:.2f})")
        ax[0].legend(fontsize=7)
    ax[0].set(xlabel="prompt tokens", ylabel="prefill (ms)",
              title="Prefill vs prompt length (real contexts)")

    ax[1].scatter(d["prompt_tokens"], d["tpot_ms"], s=7, alpha=0.3, color=C["decode"])
    if len(d) > 2:
        b, a = np.polyfit(d["prompt_tokens"], d["tpot_ms"], 1)
        xs = np.linspace(d["prompt_tokens"].min(), d["prompt_tokens"].max(), 50)
        r2 = float(np.corrcoef(d["prompt_tokens"], d["tpot_ms"])[0, 1] ** 2)
        ax[1].plot(xs, a + b * xs, "k--", lw=1.2,
                   label=f"{b*1000:.2f} ms/1k tok (r²={r2:.2f})")
        ax[1].legend(fontsize=7)
    ax[1].set(xlabel="prompt tokens", ylabel="TPOT (ms/token)",
              title="TPOT vs prompt length (real contexts)")

    d2 = d.copy()
    d2["share"] = d2["decode_ms"] / (d2["decode_ms"] + d2["prefill_ms"]).clip(lower=1e-9)
    ax[2].scatter(d2["prompt_tokens"], d2["share"], s=7, alpha=0.3, color=C["quality"])
    ax[2].axhline(0.5, color="k", ls=":", lw=1)
    ax[2].set(xlabel="prompt tokens", ylabel="decode share",
              title="Decode share on real contexts")
    fig.suptitle("Independent check of E1 on retrieved contexts (filler-free)", y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig24_real_context_scaling.png")


def fig_metric_heatmap(by: pd.DataFrame, order: List[str], out: str) -> str:
    """All methods x all metrics, min-max normalised per column."""
    cols = [c for c in ["f1", "em", "contains_gold", "answer_recall", "evidence_recall",
                        "evidence_precision", "answer_support", "gold_in_evidence",
                        "retained_ratio", "net_speedup", "QE_f1_per_sec"] if c in by.columns]
    d = by.set_index("method").reindex([m for m in order if m in set(by["method"])])
    M = d[cols].astype(float).values
    norm = np.zeros_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(cols), 0.33 * len(d) + 1.6))
    im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([lab(m) for m in d.index], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if norm[i, j] < 0.55 else "black")
    plt.colorbar(im, ax=ax, label="min–max normalised per column")
    ax.set_title("All methods × all metrics (cells show raw values)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig25_metric_heatmap.png")


def fig_metric_correlation(comp: pd.DataFrame, out: str) -> Optional[str]:
    cols = [c for c in ["f1", "em", "contains_gold", "answer_recall", "evidence_recall",
                        "evidence_precision", "gold_in_evidence", "answer_support",
                        "retained_ratio", "prompt_tokens", "prefill_ms", "decode_ms",
                        "tpot_ms", "e2e_ms", "compress_ms"] if c in comp.columns]
    if len(cols) < 3:
        return None
    M = comp[cols].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2.2, 0.55 * len(cols) + 1.8))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if abs(M.values[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Which measurements actually co-vary?\n(per-query, pooled over methods)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig26_metric_correlation.png")


def fig_qtype_breakdown(comp: pd.DataFrame, order: List[str], out: str) -> Optional[str]:
    if "qtype" not in comp.columns or comp["qtype"].nunique() < 2:
        return None
    ms = [m for m in order if m in set(comp["method"])]
    g = (comp[comp["method"].isin(ms)].groupby(["qtype", "method"])
         .agg(f1=("f1", "mean"), gold=("gold_in_evidence", "mean"),
              n=("f1", "size")).reset_index())
    types = sorted(g["qtype"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.2))
    x = np.arange(len(ms)); w = 0.8 / len(types)
    cmap = plt.cm.Set2(np.linspace(0, 0.6, len(types)))
    for i, (t, col) in enumerate(zip(types, cmap)):
        d = g[g["qtype"] == t].set_index("method").reindex(ms).reset_index()
        off = (i - (len(types) - 1) / 2) * w
        ax[0].bar(x + off, d["f1"], w, color=col, label=f"{t} (n={int(d['n'].median() or 0)})")
        ax[1].bar(x + off, d["gold"], w, color=col, label=t)
    for a, t, yl in zip(ax, ["F1 by question type", "Grounding by question type"],
                        ["F1", "gold answer in context"]):
        a.set_xticks(x); a.set_xticklabels([lab(m) for m in ms], rotation=40, ha="right", fontsize=6.3)
        a.set(ylabel=yl, title=t); a.legend(fontsize=7)
    fig.suptitle("Does compression hurt multi-hop (bridge) more than comparison questions?",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig27_qtype_breakdown.png")


def fig_win_loss(comp: pd.DataFrame, order: List[str], out: str) -> Optional[str]:
    """Per-query wins/ties/losses against full context."""
    if "full" not in set(comp["method"]):
        return None
    piv = comp.pivot_table(index="qid", columns="method", values="f1")
    if "full" not in piv.columns:
        return None
    ms = [m for m in order if m in piv.columns and m != "full"]
    rows = []
    for m in ms:
        p = piv[[m, "full"]].dropna()
        d = p[m].values - p["full"].values
        rows.append({"method": m, "win": int((d > 1e-9).sum()),
                     "tie": int((np.abs(d) <= 1e-9).sum()), "loss": int((d < -1e-9).sum())})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(r) + 1.6))
    y = np.arange(len(r))
    ax.barh(y, r["win"], color=C["grounding"], label="win vs full")
    ax.barh(y, r["tie"], left=r["win"], color=C["grey"], label="tie")
    ax.barh(y, r["loss"], left=r["win"] + r["tie"], color=C["decode"], label="loss")
    ax.set_yticks(y); ax.set_yticklabels([lab(m) for m in r["method"]], fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="number of queries", title="Per-query F1 outcome vs full context")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, "fig28_win_loss.png")


def fig_compressor_cost_breakdown(by: pd.DataFrame, order: List[str], out: str) -> str:
    """T_compress split into scoring vs selection, against the prefill it saves."""
    d = by.set_index("method").reindex([m for m in order if m in set(by["method"])]).reset_index()
    d = d[d["compress_ms"].fillna(0) > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.0))
    y = np.arange(len(d))
    sc_ms = d["score_ms"].fillna(0).values
    se_ms = (d["compress_ms"].fillna(0) - d["score_ms"].fillna(0)).clip(lower=0).values
    ax[0].barh(y, sc_ms, color=C["compress"], label="scoring")
    ax[0].barh(y, se_ms, left=sc_ms, color=C["retrieve"], label="selection")
    ax[0].set_yticks(y); ax[0].set_yticklabels([lab(m) for m in d["method"]], fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="T_compress (ms, log)", xscale="log",
              title="What the compressor spends its time on")
    ax[0].legend(fontsize=7)

    base = by[by["method"] == "full"]
    if len(base):
        saved = base.iloc[0]["prefill_ms"] - d["prefill_ms"].values
        ratio = d["compress_ms"].values / np.clip(saved, 1e-9, None)
        cols = [C["decode"] if r > 1 else C["grounding"] for r in ratio]
        ax[1].barh(y, ratio, color=cols, height=0.7)
        ax[1].axvline(1.0, color="k", ls="--", lw=1.2)
        ax[1].set_yticks(y); ax[1].set_yticklabels([])
        ax[1].invert_yaxis()
        ax[1].set(xlabel="T_compress / prefill time saved  (log; >1 = compressor costs more than it saves)",
                  xscale="log", title="Does the compressor pay for itself?")
    fig.suptitle("The central design constraint: the compressor must be cheaper than what it saves",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig29_compressor_cost.png")


def fig_answer_length(by: pd.DataFrame, order: List[str], out: str) -> Optional[str]:
    if "pred_tokens" not in by.columns:
        return None
    d = by.set_index("method").reindex([m for m in order if m in set(by["method"])]).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.0))
    y = np.arange(len(d))
    ax[0].barh(y, d["pred_tokens"], color=C["accent"], height=0.7)
    ax[0].set_yticks(y); ax[0].set_yticklabels([lab(m) for m in d["method"]], fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="median answer length (words)", title="Answer verbosity per method")
    ax[1].scatter(d["pred_tokens"], d["f1"], s=36, color=C["quality"], label="F1")
    if "contains_gold" in d:
        ax[1].scatter(d["pred_tokens"], d["contains_gold"], s=36, marker="s",
                      color=C["grounding"], label="contains gold")
    ax[1].set(xlabel="median answer length (words)", ylabel="score",
              title="Verbosity confounds EM/F1 but not contains-gold")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig30_answer_length.png")


def fig_norm_vs_raw_latency(by: pd.DataFrame, order: List[str], out: str) -> Optional[str]:
    if "e2e_norm_ms" not in by.columns:
        return None
    d = by.set_index("method").reindex([m for m in order if m in set(by["method"])]).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.0))
    x = np.arange(len(d)); w = 0.38
    ax[0].bar(x - w / 2, d["e2e_ms"], w, color=C["grey"], label="measured E2E")
    ax[0].bar(x + w / 2, d["e2e_norm_ms"], w, color=C["prefill"],
              label="E2E at fixed N_out=32")
    ax[0].set_xticks(x); ax[0].set_xticklabels([lab(m) for m in d["method"]], rotation=40,
                                               ha="right", fontsize=6.3)
    ax[0].set(ylabel="ms", title="Raw vs generation-length-controlled latency")
    ax[0].legend(fontsize=7)
    ax[1].scatter(d["e2e_ms"], d["e2e_norm_ms"], s=34, color=C["decode"])
    lim = [0, max(d["e2e_ms"].max(), d["e2e_norm_ms"].max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1)
    for _, r in d.iterrows():
        ax[1].annotate(lab(r["method"]), (r["e2e_ms"], r["e2e_norm_ms"]),
                       textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax[1].set(xlabel="measured E2E (ms)", ylabel="E2E at fixed N_out (ms)",
              title="Deviation = answer-length effect")
    fig.tight_layout()
    return _save(fig, out, "fig31_norm_vs_raw_latency.png")


def fig_tradeoff_practitioner(by: pd.DataFrame, out: str) -> Optional[str]:
    """The decision plot: % latency saved against % quality/grounding given up."""
    base = by[by["method"] == "full"]
    if base.empty:
        return None
    b = base.iloc[0]
    d = by[~by["method"].isin(["full", "closed_book"])].copy()
    d["lat_saved_pct"] = 100 * (b["e2e_total_ms"] - d["e2e_total_ms"]) / max(1e-9, b["e2e_total_ms"])
    d["f1_lost_pct"] = 100 * (b["f1"] - d["f1"]) / max(1e-9, b["f1"])
    d["gnd_lost_pct"] = 100 * (b["gold_in_evidence"] - d["gold_in_evidence"]) / max(1e-9, b["gold_in_evidence"])
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for a, k, t in [(ax[0], "f1_lost_pct", "accuracy"), (ax[1], "gnd_lost_pct", "grounding")]:
        a.scatter(d["lat_saved_pct"], d[k], s=40,
                  color=C["quality"] if k.startswith("f1") else C["grounding"], zorder=3)
        for _, r in d.iterrows():
            a.annotate(lab(r["method"]), (r["lat_saved_pct"], r[k]),
                       textcoords="offset points", xytext=(4, 3), fontsize=6.2)
        a.axhline(0, color="k", ls=":", lw=1)
        a.set(xlabel="% end-to-end latency saved vs full context",
              ylabel=f"% {t} given up",
              title=f"Latency saved vs {t} lost\n(below the dotted line = free lunch)")
    fig.suptitle("The practitioner's view: what does compression actually cost you?",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig32_tradeoff.png")


def fig_grounding_components(by: pd.DataFrame, order: List[str], out: str) -> str:
    d = by.set_index("method").reindex([m for m in order if m in set(by["method"])]).reset_index()
    keys = [("evidence_recall", "evidence recall", C["prefill"]),
            ("evidence_precision", "evidence precision", C["compress"]),
            ("gold_in_evidence", "gold answer in context", C["grounding"]),
            ("answer_support", "answer supported by context", C["quality"])]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 4.2), sharey=True)
    y = np.arange(len(d))
    for a, (k, t, col) in zip(ax, keys):
        if k not in d:
            continue
        a.barh(y, d[k], color=col, height=0.72)
        a.set_yticks(y)
        a.set_yticklabels([lab(m) for m in d["method"]] if a is ax[0] else [], fontsize=6.8)
        a.invert_yaxis()
        a.set(xlabel=t, title=t, xlim=(0, 1))
    fig.suptitle("E7 — the four grounding measures separately", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig33_grounding_components.png")


def fig_lambda_frontier(by: pd.DataFrame, out: str) -> Optional[str]:
    """Does sweeping lambda trace a better frontier than sweeping a fixed ratio?"""
    fam = {
        "DECAF λ (normalised)": by[by["method"].str.match(r"decaf_lam[\d.]+$")],
        "DECAF λ (absolute ms)": by[by["method"].str.contains("absms")],
        "fixed ratio": by[by["method"].str.match(r"fixed_0\.\d+$")],
    }
    fam = {k: v.sort_values("e2e_total_ms") for k, v in fam.items() if len(v) >= 2}
    if not fam:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
    cols = [C["quality"], C["decode"], C["accent"]]
    for (k, d), col in zip(fam.items(), cols):
        ax[0].plot(d["e2e_total_ms"], d["f1"], "o-", color=col, label=k)
        ax[1].plot(d["retained_ratio"], d["gold_in_evidence"], "o-", color=col, label=k)
    base = by[by["method"] == "full"]
    if len(base):
        ax[0].scatter(base["e2e_total_ms"], base["f1"], marker="*", s=160,
                      color="k", label="full context", zorder=4)
    ax[0].set(xlabel="E2E latency incl. overhead (ms)", ylabel="F1",
              title="Which knob traces the better frontier?")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="retained fraction", ylabel="gold answer in context",
              title="Grounding retained per unit of context kept")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig34_lambda_frontier.png")


def fig_study_cost(frames: Dict[str, pd.DataFrame], out: str) -> Optional[str]:
    """Transparency: how much machine time the study itself consumed."""
    rows = []
    for name, f in (frames or {}).items():
        if f is not None and not f.empty and "e2e_ms" in f:
            rows.append({"family": name, "runs": len(f),
                         "minutes": float(f["e2e_ms"].sum()) / 60000.0})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
    ax[0].bar(d["family"], d["runs"], color=C["prefill"], width=0.55)
    ax[0].set(ylabel="measured runs", title="Runs per experiment family")
    for i, v in enumerate(d["runs"]):
        ax[0].annotate(str(v), (i, v), ha="center", textcoords="offset points",
                       xytext=(0, 3), fontsize=8)
    ax[1].bar(d["family"], d["minutes"], color=C["decode"], width=0.55)
    ax[1].set(ylabel="model time (minutes)", title=f"Total model time: {d['minutes'].sum():.0f} min")
    for i, v in enumerate(d["minutes"]):
        ax[1].annotate(f"{v:.0f}", (i, v), ha="center", textcoords="offset points",
                       xytext=(0, 3), fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig35_study_cost.png")
