"""Figures for the v3 report (F1-F7 in DECAF_Q1_MASTER_PLAN.md §6.2).

Called from ``analyze_v3``; reads ``cache_v3/`` and writes ``results_v3/fig_v3_*.png``.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_v3")
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def _fit_cold(P, y):
    P, y = np.asarray(P, float), np.asarray(y, float)
    m = (P > 0) & (y > 0)
    P, y = P[m], y[m]
    if len(P) < 2:
        return 0.0, 0.0
    A = np.c_[P, P ** 2] / y[:, None]
    return tuple(np.linalg.lstsq(A, np.ones(len(y)), rcond=None)[0])


def f1_prefill_vs_length(scaling, cold, warm):
    fig, ax = plt.subplots(figsize=(6, 4))
    by = defaultdict(list)
    for r in scaling:
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            by[r["model_label"]].append((r["prompt_tokens"], r["prefill_ms"]))
    for i, (model, pts) in enumerate(sorted(by.items())):
        pts = sorted(pts)
        xs = np.array([p for p, _ in pts])
        ys = np.array([y for _, y in pts])
        ax.scatter(xs, ys, s=12, color=PALETTE[i % len(PALETTE)], label=f"{model} scaling")
        c1, c2 = _fit_cold(xs, ys)
        gx = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(gx, c1 * gx + c2 * gx ** 2, color=PALETTE[i % len(PALETTE)], lw=1)

    def med(rows, cond):
        g = defaultdict(list)
        for r in rows:
            if r.get("condition", "cold") == cond and r.get("prompt_tokens", 0) > 0:
                g[r["model_label"]].append((r["prompt_tokens"], r["prefill_ms"]))
        return {m: (np.median([p for p, _ in v]), np.median([y for _, y in v]))
                for m, v in g.items() if v}

    for rows, cond, marker in [(cold, "cold", "o"), (warm, "warm_order", "x")]:
        for model, (P, y) in sorted(med(rows, cond).items()):
            ax.scatter([P], [y], marker=marker, s=45, facecolors="none",
                       edgecolors="k", label=f"{model} {cond}")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("prefill (ms)")
    ax.set_title("F1 — prefill vs prompt length (cold model + cold/warm points)")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(os.path.join(RESULTS, "fig_v3_f1_prefill_length.png"))
    plt.close(fig)


def f2_ratio_distribution(scaling, cold, warm):
    by = defaultdict(list)
    for r in scaling:
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            by[r["model"]].append((r["prompt_tokens"], r["prefill_ms"]))
    coef = {m: _fit_cold([p for p, _ in v], [y for _, y in v]) for m, v in by.items()}

    def ratios(rows, cond):
        g = defaultdict(list)
        for r in rows:
            if r.get("condition", "cold") != cond:
                continue
            m = r.get("model")
            if m not in coef or r.get("prefill_ms", 0) <= 0:
                continue
            c1, c2 = coef[m]
            pred = c1 * r["prompt_tokens"] + c2 * r["prompt_tokens"] ** 2
            if pred > 0:
                g[r["method"]].append(r["prefill_ms"] / pred)
        return g

    rc, rw = ratios(cold, "cold"), ratios(warm, "warm_order")
    methods = sorted(set(rc) | set(rw))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    wdt = 0.38
    ax.bar(x - wdt / 2, [np.median(rc.get(m, [np.nan])) for m in methods], wdt,
           label="cold", color="#1f77b4")
    ax.bar(x + wdt / 2, [np.median(rw.get(m, [np.nan])) for m in methods], wdt,
           label="warm-order", color="#d62728")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhspan(0.8, 1.25, color="green", alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("median logged / cold-model prefill")
    ax.set_title("F2 — detection test: logged/cold prefill ratio by method")
    ax.legend()
    fig.savefig(os.path.join(RESULTS, "fig_v3_f2_ratio_distribution.png"))
    plt.close(fig)


def f3_inflation(inflation):
    if not inflation:
        return
    keys = sorted(inflation)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(keys))
    vals = [inflation[k]["inflation_geomean"] for k in keys]
    lo = [inflation[k]["inflation_geomean"] - inflation[k]["ci"][0] for k in keys]
    hi = [inflation[k]["ci"][1] - inflation[k]["inflation_geomean"] for k in keys]
    ax.barh(y, vals, xerr=[lo, hi], color="#1f77b4", alpha=0.8)
    ax.axvline(1.0, color="k", lw=0.8)
    ax.axvline(1.2, color="r", ls="--", lw=0.8, label="1.2× threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=6)
    ax.set_xlabel("inflation (warm / cold speed-up)")
    ax.set_title("F3 — inflation factor with 95% CI")
    ax.legend()
    fig.savefig(os.path.join(RESULTS, "fig_v3_f3_inflation.png"))
    plt.close(fig)


def f4_model_validation(scaling, cold):
    by = defaultdict(list)
    for r in scaling:
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            by[r["model_label"]].append((r["prompt_tokens"], r["prefill_ms"]))
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (model, pts) in enumerate(sorted(by.items())):
        c1, c2 = _fit_cold([p for p, _ in pts], [y for _, y in pts])
        hel = [(r["prompt_tokens"], r["prefill_ms"]) for r in cold
               if r.get("model_label") == model and r.get("prefill_ms", 0) > 0]
        if not hel:
            continue
        xs = np.array([p for p, _ in hel])
        yt = np.array([y for _, y in hel])
        yp = c1 * xs + c2 * xs ** 2
        ax.scatter(yt, yp, s=10, color=PALETTE[i % len(PALETTE)], label=model)
    lims = [1, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured prefill (ms)")
    ax.set_ylabel("predicted prefill (ms)")
    ax.set_title("F4 — cold-model prediction vs measured (held-out prompts)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(RESULTS, "fig_v3_f4_model_validation.png"))
    plt.close(fig)


def f5_breakeven(scaling, cold):
    by = defaultdict(list)
    for r in scaling:
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            by[r["model_label"]].append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (model, rows) in enumerate(sorted(by.items())):
        c1, c2 = _fit_cold([r["prompt_tokens"] for r in rows], [r["prefill_ms"] for r in rows])
        N = float(np.median([r["output_tokens"] for r in rows]))
        t0 = float(np.median([r["total_ms"] / max(1, r["output_tokens"]) for r in rows]))
        sel = [r.get("compress_ms", np.nan) for r in cold
               if r.get("model_label") == model and r.get("method") == "decaf_lam6"]
        Tc = float(np.nanmedian(sel)) if sel else 1000.0
        L = np.arange(50, 4000)
        saved = (c1 * L + c2 * L ** 2) + N * t0
        # net speedup: full-time / (selector + compressed-time at retained length)
        Lc = 160
        comp = (c1 * Lc + c2 * Lc ** 2) + N * t0 + Tc
        ax.plot(L, saved / comp, color=PALETTE[i % len(PALETTE)], label=model)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("full-context prompt tokens")
    ax.set_ylabel("net speed-up (S_net)")
    ax.set_title("F5 — break-even: S_net vs prompt length")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(RESULTS, "fig_v3_f5_breakeven.png"))
    plt.close(fig)


def f6_pareto(cold):
    if not cold:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    g = defaultdict(list)
    for r in cold:
        g[(r["model_label"], r["method"])].append((r["e2e_ms"], r["f1"]))
    for i, ((model, method), pts) in enumerate(sorted(g.items())):
        ax.scatter([np.median([p for p, _ in pts])], [np.mean([f for _, f in pts])],
                   s=25, color=PALETTE[i % len(PALETTE)], alpha=0.7)
    ax.set_xlabel("median E2E latency (ms)")
    ax.set_ylabel("mean F1")
    ax.set_title("F6 — latency vs F1 by (model, method)")
    fig.savefig(os.path.join(RESULTS, "fig_v3_f6_pareto.png"))
    plt.close(fig)


def f7_decode_share(scaling):
    by = defaultdict(list)
    for r in scaling:
        if r.get("total_ms", 0) > 0:
            by[r["model_label"]].append((r["prompt_tokens"], r["decode_ms"] / r["total_ms"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (model, pts) in enumerate(sorted(by.items())):
        pts = sorted(pts)
        ax.plot([p for p, _ in pts], [s for _, s in pts], "o-", ms=3,
                color=PALETTE[i % len(PALETTE)], label=model)
    ax.set_xscale("log")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("decode share of total")
    ax.set_title("F7 — decode share falls with context (prefill dominates)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(RESULTS, "fig_v3_f7_decode_share.png"))
    plt.close(fig)


def make_all(scaling: List[dict], cold: List[dict], warm: List[dict], inflation: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    try:
        f1_prefill_vs_length(scaling, cold, warm)
        f2_ratio_distribution(scaling, cold, warm)
        f3_inflation(inflation)
        f4_model_validation(scaling, cold)
        f5_breakeven(scaling, cold)
        f6_pareto(cold)
        f7_decode_share(scaling)
        print("figures written")
    except Exception as e:  # keep analysis usable even if a figure fails
        print(f"figure error: {e}")
