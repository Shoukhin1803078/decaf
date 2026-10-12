"""H5 secondary analysis: llama.cpp prefix/KV-cache reuse (CPU).

Measures TTFT/prefill for three prompt regimes at the same total context
length, to test the Amdahl-based expectation (H5) that KV reuse moves mainly
TTFT and hides little when decode dominates:

  cold     — a unique leading nonce defeats the prefix cache (no reuse)
  warm     — the *identical* prompt repeated (full prefix/KV hit)
  partial  — a long shared context prefix with a changed trailing question
             (partial reuse — the realistic multi-turn / same-document case)

Writes `cache_v2/h5_prefix.jsonl`; figures are produced by `src.analyze_v2`.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import yaml
from tqdm import tqdm

from . import contexts
from .llm import OllamaClient, build_prompt
from .run_experiments import JsonlCache, _timing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _client(cfg: dict, model: str) -> OllamaClient:
    c = copy.deepcopy(cfg)
    c["ollama"]["model"] = model
    return OllamaClient(c)


def run(cfg: dict) -> int:
    cache_path = os.path.join(ROOT, cfg["paths"]["h5_cache"])
    cache = JsonlCache(cache_path, ["model", "kind", "L", "rep"])
    q = cfg["scaling"]["question"]
    q2 = "Answer the same question again, step by step."
    n_new = 0
    for m in cfg["models"]:
        cl = _client(cfg, m["name"])
        cl.generate(build_prompt(q, ["Warmup."], ["reference"]))
        for L in cfg["h5"]["prefix_lengths"]:
            for rep in range(cfg["h5"]["replicates"]):
                filler = " ".join(contexts.filler_sentences(L, seed=cfg["seed"] + rep))
                # cold: unique nonce invalidates any cached prefix
                nonce = f"Cold run {L}-{rep}-{L * 7919 + rep}."
                p_cold = build_prompt(q, [nonce + " " + filler], ["reference"])
                r_cold = cl.generate(p_cold)
                # warm: identical prompt -> full prefix/KV reuse
                r_warm = cl.generate(p_cold)
                # partial: same context, different trailing question
                p_ctx = build_prompt(q, [filler], ["reference"])
                r_p1 = cl.generate(p_ctx)
                p_ctx2 = build_prompt(q + " " + q2, [filler], ["reference"])
                r_p2 = cl.generate(p_ctx2)

                base = {"model": m["name"], "model_label": m["label"],
                        "family": m["family"], "quant": m["quant"], "L": L, "rep": rep}
                for kind, resp in [("cold", r_cold), ("warm", r_warm),
                                   ("partial_cold", r_p1), ("partial_warm", r_p2)]:
                    cache.put({**base, "kind": kind, **_timing(resp)})
                    n_new += 1
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_v2.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("ollama", {})["model"] = cfg["models"][0]["name"]
    if not OllamaClient(cfg).ping():
        raise SystemExit("Ollama not reachable")
    n = run(cfg)

    rows = JsonlCache(os.path.join(ROOT, cfg["paths"]["h5_cache"]),
                      ["model", "kind", "L", "rep"]).all()
    print(f"H5: {n} new runs. Summary (median prefill ms / TTFT ms):")
    for m in cfg["models"]:
        for L in cfg["h5"]["prefix_lengths"]:
            sub = [r for r in rows if r["model"] == m["name"] and r["L"] == L]
            by = {}
            for k in ("cold", "warm", "partial_cold", "partial_warm"):
                v = [r["ttft_ms"] for r in sub if r["kind"] == k]
                if v:
                    by[k] = sorted(v)[len(v) // 2]
            print(f"  {m['label']:12s} L={L:5d} cold={by.get('cold', float('nan')):8.0f}  "
                  f"warm={by.get('warm', float('nan')):8.0f}  "
                  f"partial={by.get('partial_warm', float('nan')):8.0f} ms TTFT")


if __name__ == "__main__":
    main()
