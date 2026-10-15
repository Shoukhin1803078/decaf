"""Phase A (v2): full experiment driver.

Families
--------
* scaling     — TPOT(L)/prefill(L) decode-scaling curves per model (E1/E8).
* crossover   — decode/prefill crossover vs generation length (H1).
* compression — evidence selection over HotpotQA + 2Wiki (E2/E3/E4/E5/E6/E7/E9
                + the Sec. 17 ablation), per model (quantization + size).

All runs are cached to JSONL keyed so re-runs resume; nothing from the pilot
(`cache/`, `results/`) is touched.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from typing import Dict, List

import yaml
from tqdm import tqdm

from . import contexts
from .data_v2 import load_all, load_examples_v2
from .llm import OllamaClient, build_prompt
from .metrics import score_answer
from .retriever import rank_paragraphs
from .run_experiments import JsonlCache, _timing
from .scorer_v2 import CrossEncoderScorer
from .select import relevance_scores
from .select_v2 import build_context, select_v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NORM = re.compile(r"[^a-z0-9 ]")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _client_for(cfg: dict, model_name: str) -> OllamaClient:
    c = copy.deepcopy(cfg)
    c["ollama"]["model"] = model_name
    return OllamaClient(c)


class MemSampler:
    """Peak RSS (MB) of any process whose name contains 'ollama'."""

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


def _generate(client: OllamaClient, prompt: str) -> dict:
    with MemSampler() as mem:
        resp = client.generate(prompt)
    resp["peak_rss_mb"] = mem.peak_mb
    return resp


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
        nonce = f"Run identifier {L}-{r}-{L * 1000 + r}."
        ctx = nonce + " " + " ".join(filler)
        prompt = build_prompt(q, [ctx], ["reference"])
        resp = _generate(client, prompt)
        cache.put({"model": model["name"], "model_label": model["label"],
                   "family": model["family"], "quant": model["quant"],
                   "mode": "scaling", "L": L, "rep": r, "question": q,
                   "peak_rss_mb": resp["peak_rss_mb"], **_timing(resp),
                   "raw_response": resp["response"]})
        n_new += 1
    return n_new


# --------------------------------------------------------------------------- #
# compressed-context scaling (E1: full vs compressed TPOT(L))
# --------------------------------------------------------------------------- #
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
        nonce = f"Run identifier C{L}-{r}-{L * 1000 + r}."
        ctx = nonce + " " + " ".join(filler[:m])
        resp = _generate(client, build_prompt(q, [ctx], ["reference"]))
        cache.put({"model": model["name"], "model_label": model["label"],
                   "family": model["family"], "quant": model["quant"],
                   "mode": "scaling_compressed", "L": L, "ratio": ratio, "rep": r,
                   "kept_sentences": m, "total_sentences": len(filler), "question": q,
                   "peak_rss_mb": resp["peak_rss_mb"], **_timing(resp),
                   "raw_response": resp["response"]})
        n_new += 1
    return n_new


# --------------------------------------------------------------------------- #
# crossover (generation length)
# --------------------------------------------------------------------------- #
def run_crossover(cfg: dict, client: OllamaClient, model: dict) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["crossover_cache"]),
                       ["model", "L", "num_predict", "rep"])
    q = cfg["crossover"]["question"]
    base_client = client
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
        nonce = f"Run identifier {L}-{np_}-{L * 1000 + np_}."
        ctx = nonce + " " + " ".join(filler)
        prompt = build_prompt(q, [ctx], ["reference"])
        resp = _generate(cl, prompt)
        cache.put({"model": model["name"], "model_label": model["label"],
                   "family": model["family"], "quant": model["quant"],
                   "L": L, "num_predict": np_, "rep": r, "question": q,
                   "peak_rss_mb": resp["peak_rss_mb"], **_timing(resp),
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


def _ce_scores(scorer: CrossEncoderScorer, cfg: dict, dataset: str, ex, sentences: List[str]):
    """Cross-encoder scores, cached per (dataset, qid). Returns (rel, ce_ms)."""
    ce_dir = os.path.join(ROOT, cfg["paths"]["ce_cache"])
    os.makedirs(ce_dir, exist_ok=True)
    path = os.path.join(ce_dir, f"{dataset}_{ex.qid}.json")
    if os.path.exists(path):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("sentences") == sentences:
            return rec["rel"], rec["ce_ms"]
    rel, ce_ms = scorer.score(ex.question, sentences)
    with open(path, "w") as f:
        json.dump({"sentences": sentences, "rel": [float(x) for x in rel], "ce_ms": ce_ms}, f)
    return [float(x) for x in rel], ce_ms


def run_compression(cfg: dict, client: OllamaClient, model: dict,
                    dataset_tag: str, dataset_spec: dict, scorer: CrossEncoderScorer,
                    compressor=None) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["compress_cache"]),
                       ["model", "dataset", "qid", "method"])
    examples = load_examples_v2(dataset_spec, cfg["seed"], cfg["data"]["cache_dir"])[
        : dataset_spec["n_questions"]]
    methods = cfg["compression"]["methods"]
    lam = cfg["compression"]["lambda"]
    primary = cfg["compression"].get("decaf_primary", "decaf_lam6")
    n_new = 0
    for ex in tqdm(examples, desc=f"compress[{model['label']}|{dataset_tag}]"):
        ranked = rank_paragraphs(ex, cfg["retriever"])[: cfg["top_k"]]
        sentences: List[str] = []
        titles: List[str] = []
        for pi, _score in ranked:
            for s in contexts.split_sentences(ex.paragraphs[pi].text):
                sentences.append(s)
                titles.append(ex.paragraphs[pi].title)
        if not sentences:
            continue
        rel, ce_ms = _ce_scores(scorer, cfg, dataset_tag, ex, sentences)
        # Sec.17 scorer ablation: BM25 relevance as the "scorer off" condition.
        bm_ms = 0.0
        rel_bm = None
        if any(m.startswith("bm25_") for m in methods):
            _t0 = time.perf_counter()
            rel_bm = relevance_scores(sentences, ex.question)
            bm_ms = (time.perf_counter() - _t0) * 1000.0
        # DECAF first: its retained count defines the iso-budget baseline.
        decaf_idx, _ = select_v2(sentences, ex.question, rel, primary, lam=lam)
        n_iso = len(decaf_idx)
        gold = _gold_sentences(ex)

        full_ctx = None
        for method in methods:
            if cache.has(model=model["name"], dataset=dataset_tag, qid=ex.qid, method=method):
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
                idx, stats = select_v2(sentences, ex.question, rel_used, base_method, lam=lam, n_iso=n_iso)
                ctx = build_context(sentences, idx, titles)
                retained = {sentences[i].strip() for i in idx}
                inter = len(retained & gold)
                evidence_recall = inter / len(gold) if gold else float("nan")
                evidence_precision = inter / len(retained) if retained else float("nan")
            prompt = build_prompt(ex.question, [ctx], ["evidence"])
            resp = _generate(client, prompt)
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
                "qid": ex.qid, "method": method,
                "question": ex.question, "gold": ex.answer, "pred": scored["pred"],
                "em": scored["em"], "f1": scored["f1"],
                "evidence_recall": evidence_recall, "evidence_precision": evidence_precision,
                "answer_support": answer_support, "gold_in_evidence": gold_in_evidence,
                "ce_ms": ce_eff, **stats,
                "compress_ms": ce_eff + stats["select_ms"],
                "peak_rss_mb": resp["peak_rss_mb"], **_timing(resp),
                "raw_response": resp["response"],
            })
            n_new += 1
    return n_new


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_v2.yaml"))
    ap.add_argument("--mode", default="all",
                    choices=["all", "scaling", "scaling_compressed", "crossover", "compression"])
    ap.add_argument("--models", default=None, help="comma-separated model names to run")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset tags")
    ap.add_argument("--limit", type=int, default=None, help="override n_questions")
    ap.add_argument("--lengths", default=None, help="comma-separated scaling lengths")
    ap.add_argument("--methods", default=None, help="comma-separated compression methods override")
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
        if any(m.startswith("llmlingua") for m in cfg["compression"]["methods"]):
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
                n += run_compression(cfg, client, model, tag, cfg["data"][tag], scorer, compressor)
    print(f"Done. {n} new runs cached.")


if __name__ == "__main__":
    main()
