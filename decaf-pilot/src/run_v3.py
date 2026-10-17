"""Phase A (v3): cold-measurement experiment driver.

This is the fixed harness for the v3 re-run. It differs from ``run_v2.py`` in
exactly the ways the audit identified:

* FX-01  cache key includes ``condition``; writes to ``cache_v3/``.
* FX-02  cold condition: nonce at character 0, randomized method order, same
         questions across methods. warm-order condition: old design (no nonce,
         ``full`` first) so the inflation can be measured, not estimated.
* FX-03  the nonce is the first characters of the whole prompt everywhere.
* FX-05  the cross-encoder is timed live on every run (no shared cached time)
         and warm-up calls are discarded.
* FX-06  ``full`` stores zero selector cost.
* FX-07  timer guard: any row with ``prefill_ms > e2e_ms`` is re-run.
* FX-08  every method runs on the same question set.
* FX-10  ``top_k`` sweep for a real long-prompt regime (own cache file).
* FX-12  unused config entries ignored.

Families: scaling / scaling_compressed / crossover / compression.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import threading
import time
import zlib
from typing import Dict, List, Optional

import yaml
from tqdm import tqdm

from . import contexts
from .data_v2 import load_examples_v2
from .llm import OllamaClient, build_prompt, with_nonce
from .metrics import score_answer
from .retriever import rank_paragraphs
from .run_experiments import JsonlCache, _timing
from .scorer_v2 import CrossEncoderScorer
from .select import relevance_scores
from .select_v2 import build_context, select_v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NORM = re.compile(r"[^a-z0-9 ]")

_REJECTED: List[dict] = []


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _client_for(cfg: dict, model_name: str) -> OllamaClient:
    c = copy.deepcopy(cfg)
    c["ollama"]["model"] = model_name
    return OllamaClient(c)


class MemSampler:
    """Peak RSS (MB) of any process whose name contains 'ollama'.

    FX-11: disabled by default for timing runs; enable in a dedicated memory
    pass via ``ollama.sample_memory: true``.
    """

    def __init__(self, interval: float = 0.15):
        self.interval = interval
        self.peak = 0.0
        self._stop = False
        self._t = None

    @staticmethod
    def _rss() -> float:
        import psutil

        tot = 0
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                nm = (p.info["name"] or "").lower()
                if "ollama" in nm:
                    tot += p.info["memory_info"].rss
            except Exception:
                pass
        return tot

    def _loop(self):
        while not self._stop:
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = 0.0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._t:
            self._t.join(timeout=1.0)
        return False

    @property
    def peak_mb(self) -> float:
        return self.peak / 1e6


def _generate(client: OllamaClient, prompt: str, sample_memory: bool = False) -> dict:
    if sample_memory:
        with MemSampler() as mem:
            resp = client.generate(prompt)
        resp["peak_rss_mb"] = mem.peak_mb
    else:
        resp = client.generate(prompt)
        resp["peak_rss_mb"] = float("nan")
    return resp


def _valid(t: dict) -> bool:
    """FX-07 timer guard."""
    return 0 < t.get("prefill_ms", 0) <= t.get("e2e_ms", float("inf")) and t.get("output_tokens", 0) > 0


def _generate_guarded(client: OllamaClient, prompt: str, tag: str,
                      attempts: int = 3, sample_memory: bool = False) -> tuple:
    """Generate with the timer guard; re-run invalid rows up to ``attempts`` times."""
    resp, t = None, None
    for attempt in range(attempts):
        resp = _generate(client, prompt, sample_memory=sample_memory)
        t = _timing(resp)
        if _valid(t):
            return resp, t, attempt
    _REJECTED.append({"tag": tag, "prefill_ms": t["prefill_ms"], "e2e_ms": t["e2e_ms"]})
    return resp, t, attempts


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def run_scaling(cfg: dict, client: OllamaClient, model: dict) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["scaling_cache"]),
                       ["model", "mode", "L", "rep"])
    q = cfg["scaling"]["question"]
    _generate(client, build_prompt(q, ["Warmup."], ["reference"]))
    jobs = [(L, r) for L in cfg["scaling"]["lengths"] for r in range(cfg["scaling"]["replicates"])]
    n_new = 0
    for L, r in tqdm(jobs, desc=f"scaling[{model['label']}]"):
        if cache.has(model=model["name"], mode="scaling", L=L, rep=r):
            continue
        filler = contexts.filler_sentences(L, seed=cfg["seed"] + r)
        # FX-03: nonce is the FIRST characters of the whole prompt.
        prompt = with_nonce(build_prompt(q, [" ".join(filler)], ["reference"]),
                            f"scal-{model['label']}-{L}-{r}")
        resp, t, _att = _generate_guarded(client, prompt, f"scal-{model['label']}-{L}-{r}")
        cache.put({"model": model["name"], "model_label": model["label"],
                   "family": model["family"], "quant": model["quant"],
                   "mode": "scaling", "L": L, "rep": r, "question": q,
                   "condition": "cold",
                   "peak_rss_mb": resp["peak_rss_mb"], **t,
                   "raw_response": resp["response"]})
        n_new += 1
    return n_new


def run_scaling_compressed(cfg: dict, client: OllamaClient, model: dict) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["scaling_compressed_cache"]),
                       ["model", "L", "ratio", "rep"])
    q = cfg["scaling"]["question"]
    ratio = cfg["scaling_compressed"]["ratio"]
    _generate(client, build_prompt(q, ["Warmup."], ["reference"]))
    jobs = [(L, r) for L in cfg["scaling_compressed"]["lengths"]
            for r in range(cfg["scaling_compressed"]["replicates"])]
    n_new = 0
    for L, r in tqdm(jobs, desc=f"scalingC[{model['label']}]"):
        if cache.has(model=model["name"], L=L, ratio=ratio, rep=r):
            continue
        filler = contexts.filler_sentences(L, seed=cfg["seed"] + r)
        m = max(1, int(round(len(filler) * ratio)))
        prompt = with_nonce(build_prompt(q, [" ".join(filler[:m])], ["reference"]),
                            f"scalC-{model['label']}-{L}-{r}")
        resp, t, _att = _generate_guarded(client, prompt, f"scalC-{model['label']}-{L}-{r}")
        cache.put({"model": model["name"], "model_label": model["label"],
                   "family": model["family"], "quant": model["quant"],
                   "mode": "scaling_compressed", "L": L, "ratio": ratio, "rep": r,
                   "kept_sentences": m, "total_sentences": len(filler), "question": q,
                   "condition": "cold",
                   "peak_rss_mb": resp["peak_rss_mb"], **t,
                   "raw_response": resp["response"]})
        n_new += 1
    return n_new


def run_crossover(cfg: dict, client: OllamaClient, model: dict) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["crossover_cache"]),
                       ["model", "L", "num_predict", "rep"])
    q = cfg["crossover"]["question"]
    jobs = [(L, np_, r)
            for L in cfg["crossover"]["lengths"]
            for np_ in cfg["crossover"]["num_predict"]
            for r in range(cfg["crossover"]["replicates"])]
    n_new = 0
    for L, np_, r in tqdm(jobs, desc=f"crossover[{model['label']}]"):
        if cache.has(model=model["name"], L=L, num_predict=np_, rep=r):
            continue
        c = copy.deepcopy(cfg)
        c["ollama"]["model"] = model["name"]
        c["ollama"]["num_predict"] = np_
        cl = OllamaClient(c)
        filler = contexts.filler_sentences(L, seed=cfg["seed"] + r)
        prompt = with_nonce(build_prompt(q, [" ".join(filler)], ["reference"]),
                            f"cross-{model['label']}-{L}-{np_}-{r}")
        resp, t, _att = _generate_guarded(cl, prompt, f"cross-{model['label']}-{L}-{np_}-{r}")
        cache.put({"model": model["name"], "model_label": model["label"],
                   "family": model["family"], "quant": model["quant"],
                   "L": L, "num_predict": np_, "rep": r, "question": q,
                   "condition": "cold",
                   "peak_rss_mb": resp["peak_rss_mb"], **t,
                   "raw_response": resp["response"]})
        n_new += 1
    return n_new


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #
def _gold_sentences(ex) -> set:
    gold = set()
    by_title = {p.title: p.sentences for p in ex.paragraphs}
    for title, ids in ex.support_sentences.items():
        sents = by_title.get(title, [])
        for sid in ids:
            if 0 <= sid < len(sents):
                gold.add(sents[sid].strip())
    return gold


def _method_order(methods: List[str], qid: str, condition: str, seed: int) -> List[str]:
    """FX-02: random order for cold, fixed config order (full first) for warm."""
    order = list(methods)
    if condition == "cold":
        # zlib.crc32, not hash(): Python salts str hashes per process.
        rng = random.Random(seed * 1_000_003 + zlib.crc32(qid.encode()))
        rng.shuffle(order)
    return order


def run_compression(cfg: dict, client: OllamaClient, model: dict,
                    dataset_tag: str, dataset_spec: dict, scorer: CrossEncoderScorer,
                    compressor=None, condition: str = "cold",
                    top_k: Optional[int] = None, cache_path: Optional[str] = None) -> int:
    methods = (cfg["compression"]["methods"] if condition == "cold"
               else cfg["compression_warm_methods"])
    lam = cfg["compression"]["lambda"]
    primary = cfg["compression"].get("decaf_primary", "decaf_lam6")
    tk = top_k if top_k is not None else cfg["top_k"]
    sample_memory = bool(cfg["ollama"].get("sample_memory", False))

    path = cache_path or (cfg["paths"]["compress_cache"] if condition == "cold"
                          else cfg["paths"]["compress_warm_cache"])
    cache = JsonlCache(os.path.join(ROOT, path),
                       ["model", "dataset", "qid", "method", "condition", "top_k"])
    examples = load_examples_v2(dataset_spec, cfg["seed"], cfg["data"]["cache_dir"])[
        : dataset_spec["n_questions"]]
    n_new = 0
    for ex in tqdm(examples, desc=f"compress[{model['label']}|{dataset_tag}|{condition}|k{tk}]"):
        ranked = rank_paragraphs(ex, cfg["retriever"])[: tk]
        sentences: List[str] = []
        titles: List[str] = []
        for pi, _score in ranked:
            for s in contexts.split_sentences(ex.paragraphs[pi].text):
                sentences.append(s)
                titles.append(ex.paragraphs[pi].title)
        if not sentences:
            continue

        order = _method_order(methods, ex.qid, condition, cfg["seed"])
        # Skip the question entirely if every (method) row already exists.
        if all(cache.has(model=model["name"], dataset=dataset_tag, qid=ex.qid,
                         method=m, condition=condition, top_k=tk) for m in order):
            continue

        # FX-05: time the cross-encoder live (scores are deterministic; the time
        # is the measured T_compress and must not be shared across models).
        rel, ce_ms = scorer.score(ex.question, sentences)
        rel = [float(x) for x in rel]
        bm_ms = 0.0
        rel_bm = None
        if any(m.startswith("bm25_") for m in order):
            _t0 = time.perf_counter()
            rel_bm = relevance_scores(sentences, ex.question)
            bm_ms = (time.perf_counter() - _t0) * 1000.0
        decaf_idx, _ = select_v2(sentences, ex.question, rel, primary, lam=lam)
        n_iso = len(decaf_idx)
        gold = _gold_sentences(ex)

        full_ctx = None
        for pos, method in enumerate(order):
            if cache.has(model=model["name"], dataset=dataset_tag, qid=ex.qid,
                         method=method, condition=condition, top_k=tk):
                continue
            if method.startswith("llmlingua"):
                rate = float(method.split("_", 1)[1])
                if full_ctx is None:
                    full_ctx = build_context(sentences, list(range(len(sentences))), titles)
                ctx, ll_ms, orig_tok, comp_tok = compressor.compress(full_ctx, rate)
                stats = {
                    "n_total": len(sentences), "n_selected": -1,
                    "tokens_total": float(orig_tok or 0), "tokens_selected": float(comp_tok or 0),
                    "retained_ratio": (comp_tok / orig_tok) if orig_tok else float("nan"),
                    "select_ms": ll_ms,
                }
                ce_eff = 0.0
                evidence_recall = float("nan")
                evidence_precision = float("nan")
            else:
                if method.startswith("bm25_"):
                    rel_used, ce_eff, base_method = rel_bm, bm_ms, method[5:]
                else:
                    rel_used, ce_eff, base_method = rel, ce_ms, method
                idx, stats = select_v2(sentences, ex.question, rel_used, base_method,
                                       lam=lam, n_iso=n_iso)
                ctx = build_context(sentences, idx, titles)
                retained = {sentences[i].strip() for i in idx}
                inter = len(retained & gold)
                evidence_recall = inter / len(gold) if gold else float("nan")
                evidence_precision = inter / len(retained) if retained else float("nan")

            # FX-06: `full` carries no selector cost.
            if method == "full":
                ce_eff = 0.0
                stats["select_ms"] = 0.0

            prompt = build_prompt(ex.question, [ctx], ["evidence"])
            if condition == "cold":
                prompt = with_nonce(prompt, f"{model['label']}-{dataset_tag}-{ex.qid}-{method}")
            tag = f"{model['label']}-{dataset_tag}-{ex.qid}-{method}-{condition}"
            resp, t, reattempts = _generate_guarded(client, prompt, tag, sample_memory=sample_memory)
            scored = score_answer(resp["response"], ex.answer)

            ctx_norm = _NORM.sub(" ", ctx.lower())
            pred_norm = _NORM.sub(" ", scored["pred"].lower()).strip()
            gold_norm = _NORM.sub(" ", ex.answer.lower()).strip()
            answer_support = float(bool(pred_norm) and pred_norm in ctx_norm)
            gold_in_evidence = float(bool(gold_norm) and gold_norm in ctx_norm)

            cache.put({
                "model": model["name"], "model_label": model["label"],
                "family": model["family"], "quant": model["quant"],
                "dataset": dataset_tag, "mode": "compression",
                "qid": ex.qid, "method": method, "condition": condition,
                "order_pos": pos, "seed": cfg["seed"], "top_k": tk,
                "question": ex.question, "gold": ex.answer, "pred": scored["pred"],
                "em": scored["em"], "f1": scored["f1"],
                "evidence_recall": evidence_recall, "evidence_precision": evidence_precision,
                "answer_support": answer_support, "gold_in_evidence": gold_in_evidence,
                "ce_ms": ce_eff, **stats,
                "compress_ms": ce_eff + stats["select_ms"],
                "reattempts": reattempts,
                "peak_rss_mb": resp["peak_rss_mb"], **t,
                "raw_response": resp["response"],
            })
            n_new += 1
    return n_new


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_v3.yaml"))
    ap.add_argument("--mode", default="all",
                    choices=["all", "scaling", "scaling_compressed", "crossover", "compression"])
    ap.add_argument("--models", default=None, help="comma-separated model names or labels")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset tags")
    ap.add_argument("--limit", type=int, default=None, help="override n_questions")
    ap.add_argument("--lengths", default=None, help="comma-separated scaling lengths")
    ap.add_argument("--methods", default=None, help="comma-separated compression methods override")
    ap.add_argument("--condition", default="cold", choices=["cold", "warm_order"])
    ap.add_argument("--top-k", type=int, default=None, help="override top_k")
    ap.add_argument("--cache", default=None, help="override compression cache path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.limit:
        for k, v in cfg["data"].items():
            if isinstance(v, dict):
                v["n_questions"] = args.limit
    if args.lengths:
        cfg["scaling"]["lengths"] = [int(x) for x in args.lengths.split(",")]
    if args.methods:
        cfg["compression"]["methods"] = [m for m in args.methods.split(",")]

    models = cfg["models"]
    if args.models:
        want = set(args.models.split(","))
        models = [m for m in models if m["name"] in want or m["label"] in want]
    tags = [t for t, v in cfg["data"].items() if isinstance(v, dict)]
    if args.datasets:
        tags = [t for t in tags if t in args.datasets.split(",")]

    cfg.setdefault("ollama", {})["model"] = models[0]["name"]
    if not OllamaClient(cfg).ping():
        sys.exit(f"Ollama not reachable at {cfg['ollama']['host']}.")

    scorer = None
    compressor = None
    if args.mode in ("all", "compression"):
        scorer = CrossEncoderScorer(cfg["scorer"]["cross_encoder"], cfg["scorer"]["max_length"])
        warmup = int(cfg["scorer"].get("warmup_calls", 0))
        if warmup:
            print(f"Scorer warm-up: {warmup} throw-away calls")
            for _ in range(warmup):
                scorer.score("warm up the cross-encoder", ["a candidate sentence."])
        if args.condition == "cold" and any(
                m.startswith("llmlingua") for m in cfg["compression"]["methods"]):
            from .llmlingua_v2 import LLMLinguaCompressor
            compressor = LLMLinguaCompressor()

    n = 0
    for model in models:
        print(f"=== model {model['label']} ({model['name']}) ===")
        client = _client_for(cfg, model["name"])
        if args.mode in ("all", "scaling"):
            n += run_scaling(cfg, client, model)
        if args.mode in ("all", "scaling_compressed"):
            n += run_scaling_compressed(cfg, client, model)
        if args.mode in ("all", "crossover"):
            n += run_crossover(cfg, client, model)
        if args.mode in ("all", "compression"):
            for tag in tags:
                n += run_compression(cfg, client, model, tag, cfg["data"][tag], scorer,
                                     compressor, condition=args.condition,
                                     top_k=args.top_k, cache_path=args.cache)
    print(f"Done. {n} new runs cached. Rejected rows: {len(_REJECTED)}")
    if _REJECTED:
        rej_path = os.path.join(ROOT, "cache_v3", "rejected.jsonl")
        with open(rej_path, "a") as f:
            for r in _REJECTED:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
