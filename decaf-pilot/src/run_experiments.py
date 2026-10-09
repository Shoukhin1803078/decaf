"""Phase A: run the experiment families.

  scaling      E1/E8  TPOT(L) and prefill(L) over context length x generation length
  compression  E2/E3/E4/E7 + Sec.17  evidence-selection grid with quality,
               grounding, per-stage timing and contention telemetry
  kv           H5     prefix/KV-cache reuse and retrieval-prefetch micro-study

Every run is cached by a key that includes the model, so adding a quantization
or a model never invalidates existing measurements and the whole study is
resumable after an interruption.

Two measurement hazards found in the pilot are guarded here:
  * prompt truncation when the context exceeds `num_ctx` -> hard error, because
    it silently flattens the scaling curve;
  * the cross-run prompt-prefix KV cache making a longer prompt look free -> a
    unique per-run nonce is prepended for every measured run except the ones in
    the KV study, where reuse is the thing being measured.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from . import contexts, telemetry
from . import scorer as scorer_mod
from .data import load_examples
from .llm import OllamaClient, build_prompt
from .metrics import score_answer
from .retriever import rank_paragraphs
from .select import CostModel, build_context, parse_method, scorers_needed, select

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NORM = re.compile(r"[^a-z0-9 ]")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class JsonlCache:
    def __init__(self, path: str, key_fields: List[str]):
        self.path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        self.key_fields = key_fields
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.entries: Dict[str, dict] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        self.entries[self._k(r)] = r

    def _k(self, r: dict) -> str:
        return "|".join(str(r.get(f)) for f in self.key_fields)

    def has(self, **kw) -> bool:
        return self._k(kw) in self.entries

    def put(self, rec: dict) -> None:
        self.entries[self._k(rec)] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def all(self) -> List[dict]:
        return list(self.entries.values())


def _timing(resp: dict) -> dict:
    prefill_ms = (resp.get("prompt_eval_duration") or 0) / 1e6
    decode_ms = (resp.get("eval_duration") or 0) / 1e6
    n_out = float(resp.get("eval_count") or 0)
    return {
        "prompt_tokens": float(resp.get("prompt_eval_count") or 0),
        "output_tokens": n_out,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": prefill_ms + decode_ms,
        "tpot_ms": (decode_ms / n_out) if n_out else 0.0,
        "prefill_ms_per_tok": prefill_ms / max(1.0, float(resp.get("prompt_eval_count") or 1)),
        "ttft_ms": resp["ttft_ms"],
        "e2e_ms": resp["e2e_ms"],
        "load_ms": (resp.get("load_duration") or 0) / 1e6,
    }


def _check_truncation(rec: dict, num_ctx: int, target: Optional[int] = None) -> None:
    """Pilot bug #1: Ollama silently truncates a prompt longer than num_ctx."""
    pt = rec["prompt_tokens"]
    if pt >= num_ctx - 8:
        raise SystemExit(
            f"PROMPT TRUNCATED: prompt_eval_count={pt:.0f} at num_ctx={num_ctx}. "
            f"Raise num_ctx; the scaling curve would be invalid."
        )
    if target and pt < 0.5 * target:
        print(f"  warn: target ~{target} tokens but got {pt:.0f}")


# --- cost model from the measured curve ------------------------------------

def fit_cost_model(cfg: dict, model: str, n_out: Optional[float] = None) -> CostModel:
    """Marginal ms per context token, fitted from this model's scaling runs.

    Falls back to a documented default when the scaling sweep has not been run
    yet, so the compression grid is never blocked on it.
    """
    n_out = float(n_out or cfg["ollama"]["num_predict"])
    pinned = (cfg.get("compression", {}) or {}).get("cost_ms_per_token")
    if pinned:
        # An explicit pin is used when the scaling sweep for this model has not
        # been measured on a quiet machine yet: a fit taken under CPU contention
        # inflates ms/token by an order of magnitude, which would make the
        # absolute-ms selector reject nearly every sentence.
        return CostModel(ms_per_token=float(pinned), n_out=n_out, source="pinned(config)")
    cache = JsonlCache(cfg["paths"]["scaling_cache"], ["mode", "model", "L", "n_out", "rep"])
    rows = [r for r in cache.all() if r.get("model") == model]
    if len(rows) < 4:
        return CostModel(ms_per_token=4.1, n_out=n_out,
                         source="default(no scaling runs yet)")
    contended = [r.get("sys_load1", 0) or 0 for r in rows]
    if contended and float(np.median(contended)) > 4.0:
        print(f"  WARNING: scaling runs for {model} were measured at median load "
              f"{np.median(contended):.1f}; the fitted cost model is unreliable.")
    L = np.array([r["prompt_tokens"] for r in rows], dtype=float)
    pre = np.array([r["prefill_ms"] for r in rows], dtype=float)
    tpot = np.array([r["tpot_ms"] for r in rows], dtype=float)
    b_pre = float(np.polyfit(L, pre, 1)[0])
    b_tpot = float(np.polyfit(L, tpot, 1)[0])
    return CostModel.from_fit(b_pre, b_tpot, n_out, source=f"fit(n={len(rows)})")


# --- E1 / E8: scaling ------------------------------------------------------

def run_scaling(cfg: dict, client: OllamaClient, model: str) -> int:
    cache = JsonlCache(cfg["paths"]["scaling_cache"], ["mode", "model", "L", "n_out", "rep"])
    q = cfg["scaling"]["question"]
    num_ctx = cfg["scaling"]["num_ctx"]
    base_n = cfg["scaling"]["n_out_base"]
    client.num_ctx = num_ctx

    jobs: List[Tuple[int, int, int]] = []
    for L in cfg["scaling"]["lengths"]:
        for n_out in cfg["scaling"]["n_out"]:
            for rep in range(cfg["scaling"]["replicates"]):
                jobs.append((L, n_out, rep))
    for L in cfg["scaling"].get("long_lengths", []):
        for rep in range(cfg["scaling"].get("replicates_long", 2)):
            jobs.append((L, base_n, rep))

    todo = [j for j in jobs if not cache.has(mode="scaling", model=model, L=j[0], n_out=j[1], rep=j[2])]
    if not todo:
        return 0
    # warmup so the first measured point is not a cold-start outlier
    client.num_predict = base_n
    client.generate(build_prompt(q, ["Warmup."], ["reference"]))

    n_new = 0
    for L, n_out, rep in tqdm(todo, desc=f"scaling[{model}]"):
        filler = contexts.filler_sentences(L, seed=cfg["seed"] + rep)
        nonce = f"Run identifier {L}-{n_out}-{rep}-{L * 7919 + n_out * 131 + rep}."
        ctx = nonce + " " + " ".join(filler)
        client.num_predict = n_out
        tele = telemetry.sample()
        resp = client.generate(build_prompt(q, [ctx], ["reference"]))
        t = _timing(resp)
        _check_truncation(t, num_ctx, target=L)
        bw = telemetry.achieved_bandwidth_gbs(model, t["tpot_ms"], t["prompt_tokens"])
        rec = {
            "mode": "scaling", "model": model,
            "quant": cfg["models"]["quant_label"].get(model, "?"),
            "L": L, "n_out": n_out, "rep": rep, "num_ctx": num_ctx,
            **t,
            "kv_bytes": telemetry.kv_bytes(model, t["prompt_tokens"]),
            "achieved_bw_gbs": bw,
            **tele,
            "raw_response": resp["response"][:200],
        }
        cache.put(rec)
        n_new += 1
    return n_new


# --- E2 / E3 / E4 / E7 / Sec.17: compression --------------------------------

def _gold_sentences(ex) -> set:
    gold = set()
    by_title = {p.title: p.sentences for p in ex.paragraphs}
    for title, ids in ex.support_sentences.items():
        sents = by_title.get(title, [])
        for sid in ids:
            if 0 <= sid < len(sents):
                gold.add(sents[sid].strip())
    return gold


def run_compression(cfg: dict, client: OllamaClient, model: str,
                    methods: List[str], n_questions: int) -> int:
    cache = JsonlCache(cfg["paths"]["compress_cache"], ["mode", "model", "qid", "method"])
    num_ctx = cfg["compression"]["num_ctx"]
    client.num_ctx = num_ctx
    client.num_predict = cfg["ollama"]["num_predict"]

    cfg_data = dict(cfg)
    examples = load_examples(cfg)[:n_questions]
    cost_model = fit_cost_model(cfg, model)
    print(f"  cost model: {cost_model.ms_per_token:.3f} ms/context-token "
          f"(N_out={cost_model.n_out:.0f}, {cost_model.source})")

    for sc in scorers_needed(methods):
        ms = scorer_mod.warmup(sc, cfg["scorer"]["ce_model"])
        if ms:
            print(f"  warmed up scorer '{sc}' in {ms:.0f} ms (load cost not charged to runs)")

    primary = cfg["compression"]["decaf_primary"]
    n_new = 0
    for ex in tqdm(examples, desc=f"compression[{model}]"):
        pending = [m for m in methods
                   if not cache.has(mode="compression", model=model, qid=ex.qid, method=m)]
        if not pending:
            continue

        t_ret0 = time.perf_counter()
        ranked = rank_paragraphs(ex, cfg["retriever"])[: cfg["top_k"]]
        sentences: List[str] = []
        titles: List[str] = []
        for pi, _s in ranked:
            for s in contexts.split_sentences(ex.paragraphs[pi].text):
                sentences.append(s)
                titles.append(ex.paragraphs[pi].title)
        t_retrieve_ms = (time.perf_counter() - t_ret0) * 1000.0
        if not sentences:
            continue

        # one scorer pass per (query, scorer), shared by every method using it
        shared: Dict[str, Tuple[np.ndarray, float]] = {}
        for sc in scorers_needed(pending + [primary]):
            shared[sc] = scorer_mod.score(sentences, ex.question, sc, cfg["scorer"]["ce_model"])

        # DECAF first: its retained count defines the iso-budget baseline
        p_spec = parse_method(primary)
        decaf_idx, _ = select(sentences, ex.question, primary, cost_model=cost_model,
                              precomputed=shared.get(p_spec.scorer))
        n_iso = len(decaf_idx)
        gold = _gold_sentences(ex)

        for method in pending:
            spec = parse_method(method)
            idx, stats = select(sentences, ex.question, method, cost_model=cost_model,
                                n_iso=n_iso, precomputed=shared.get(spec.scorer),
                                ce_model=cfg["scorer"]["ce_model"])
            ctx = build_context(sentences, idx, titles)
            prompt = build_prompt(ex.question, [ctx] if ctx else [], ["evidence"])
            tele = telemetry.sample()
            resp = client.generate(prompt)
            t = _timing(resp)
            _check_truncation(t, num_ctx)
            scored = score_answer(resp["response"], ex.answer)

            retained = {sentences[i].strip() for i in idx}
            inter = len(retained & gold)
            ctx_norm = _NORM.sub(" ", ctx.lower())
            pred_norm = _NORM.sub(" ", scored["pred"].lower()).strip()
            gold_norm = _NORM.sub(" ", ex.answer.lower()).strip()

            rec = {
                "mode": "compression", "model": model,
                "quant": cfg["models"]["quant_label"].get(model, "?"),
                "qid": ex.qid, "method": method, "num_ctx": num_ctx,
                "question": ex.question, "gold": ex.answer, "pred": scored["pred"],
                "qtype": ex.qtype, "level": ex.level,
                "em": scored["em"], "f1": scored["f1"],
                "contains_gold": scored["contains_gold"],
                "answer_recall": scored["answer_recall"],
                "pred_tokens": scored["pred_tokens"],
                "n_gold": len(gold),
                "evidence_recall": (inter / len(gold)) if gold else float("nan"),
                "evidence_precision": (inter / len(retained)) if retained else float("nan"),
                "answer_support": float(bool(pred_norm) and pred_norm in ctx_norm),
                "gold_in_evidence": float(bool(gold_norm) and gold_norm in ctx_norm),
                "t_retrieve_ms": t_retrieve_ms,
                **stats, **t, **tele,
                "e2e_with_overhead_ms": t["e2e_ms"] + stats["compress_ms"] + t_retrieve_ms,
                # latency at a *fixed* generation length: methods that happen to
                # emit fewer tokens must not look faster for that reason alone
                "e2e_norm_ms": t["prefill_ms"] + 32.0 * t["tpot_ms"],
                "raw_response": resp["response"][:300],
            }
            cache.put(rec)
            n_new += 1
    return n_new


# --- H5: KV reuse / prefetch ------------------------------------------------

def run_kv_study(cfg: dict, client: OllamaClient, model: str) -> int:
    """Measures (i) prefix/KV-cache reuse on repeated prompts and (ii) how much
    retrieval time an async prefetch could hide behind decoding."""
    cache = JsonlCache(cfg["paths"]["kv_cache"], ["mode", "model", "qid", "variant", "rep"])
    num_ctx = cfg["kv_study"]["num_ctx"]
    client.num_ctx = num_ctx
    client.num_predict = cfg["ollama"]["num_predict"]
    examples = load_examples(cfg)[: cfg["kv_study"]["n_questions"]]
    reps = cfg["kv_study"]["repeats"]
    n_new = 0

    for ex in tqdm(examples, desc=f"kv-study[{model}]"):
        t0 = time.perf_counter()
        ranked = rank_paragraphs(ex, cfg["retriever"])[: cfg["top_k"]]
        t_retrieve_ms = (time.perf_counter() - t0) * 1000.0
        ctx = " ".join(ex.paragraphs[pi].text for pi, _ in ranked)
        base_prompt = build_prompt(ex.question, [ctx], ["evidence"])

        for rep in range(reps):
            # cold: unique nonce defeats the prefix cache; warm: identical prompt reused
            for variant in ("cold", "warm"):
                if cache.has(mode="kv", model=model, qid=ex.qid, variant=variant, rep=rep):
                    continue
                prompt = (f"Run {ex.qid}-{rep}-{time.time_ns()}. " + base_prompt
                          if variant == "cold" else base_prompt)
                tele = telemetry.sample()
                resp = client.generate(prompt)
                t = _timing(resp)
                _check_truncation(t, num_ctx)
                cache.put({
                    "mode": "kv", "model": model, "qid": ex.qid, "variant": variant,
                    "rep": rep, "t_retrieve_ms": t_retrieve_ms, **t, **tele,
                })
                n_new += 1
    return n_new


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--mode", default="all",
                    choices=["all", "scaling", "compression", "kv"])
    ap.add_argument("--model", default=None, help="override the model tag")
    ap.add_argument("--secondary", action="store_true",
                    help="use the Q8 model and its reduced method subset (E5)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--methods", default=None, help="comma-separated method override")
    ap.add_argument("--allow-contention", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    model = args.model or (cfg["models"]["secondary"] if args.secondary
                           else cfg["models"]["primary"])
    cfg["ollama"]["model"] = model
    methods = (args.methods.split(",") if args.methods
               else cfg["compression"]["methods_secondary"] if args.secondary
               else cfg["compression"]["methods"])
    n_q = args.limit or (cfg["data"]["n_questions_secondary"] if args.secondary
                         else cfg["data"]["n_questions"])
    if args.lengths:
        cfg["scaling"]["lengths"] = [int(x) for x in args.lengths.split(",")]
        cfg["scaling"]["long_lengths"] = []

    if not args.skip_preflight:
        rep = telemetry.preflight(cfg["preflight"]["max_load1"],
                                  cfg["preflight"]["min_mem_gb"],
                                  strict=not args.allow_contention,
                                  max_swap_used_pct=cfg["preflight"].get(
                                      "max_swap_used_pct", 50.0))
        print(f"preflight: load1={rep['sys_load1']:.2f} "
              f"mem_avail={rep['sys_mem_avail_gb']:.1f}GB clean={rep['clean']}")

    client = OllamaClient(cfg)
    if not client.ping():
        sys.exit(f"Ollama not reachable at {cfg['ollama']['host']}.")
    tags = client.list_models()
    if tags and model not in tags:
        sys.exit(f"model '{model}' not pulled. Available: {', '.join(sorted(tags))}")
    print(f"Ollama OK | model={model} | mode={args.mode}")

    n = 0
    t0 = time.time()
    if args.mode in ("all", "scaling"):
        n += run_scaling(cfg, client, model)
    if args.mode in ("all", "compression"):
        n += run_compression(cfg, client, model, methods, n_q)
    if args.mode in ("all", "kv"):
        n += run_kv_study(cfg, client, model)
    print(f"Done. {n} new runs cached in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
