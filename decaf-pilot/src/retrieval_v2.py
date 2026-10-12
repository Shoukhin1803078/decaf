"""Sec.13 retrieval-mode study: BM25 vs dense vs hybrid top-k.

Dense = `all-MiniLM`/`bge-small` sentence embeddings; hybrid = reciprocal-rank
fusion (RRF) of BM25 + dense. BM25 is the existing main-cache baseline; this
runner produces the *new* retrievers (dense, hybrid) into their own cache so the
main compression cache is untouched.

Reuses the Phase-A helpers (generation, cross-encoder scoring, grounding).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

from . import contexts
from .data_v2 import load_examples_v2
from .llm import build_prompt
from .metrics import score_answer
from .retriever import rank_bm25, rank_dense
from .run_experiments import JsonlCache, _timing
from .run_v2 import _NORM, _ce_scores, _client_for, _generate, _gold_sentences, load_config
from .scorer_v2 import CrossEncoderScorer
from .select_v2 import build_context, select_v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rrf(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    """Reciprocal rank fusion of several ranked lists."""
    s: Dict[int, float] = defaultdict(float)
    for r in rankings:
        for pos, (idx, _sc) in enumerate(r):
            s[idx] += 1.0 / (k + pos + 1)
    return sorted(s.items(), key=lambda x: -x[1])


def rank(example, retriever: str, dense_model: str) -> List[Tuple[int, float]]:
    if retriever == "dense":
        return rank_dense(example, dense_model)
    if retriever == "hybrid":
        return _rrf([rank_bm25(example), rank_dense(example, dense_model)])
    return rank_bm25(example)


def run(cfg: dict) -> int:
    rc = cfg["retrieval"]
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["retrieval_cache"]),
                       ["model", "dataset", "retriever", "qid", "method"])
    scorer = CrossEncoderScorer(cfg["scorer"]["cross_encoder"], cfg["scorer"]["max_length"])
    by_name = {m["name"]: m for m in cfg["models"]}
    models = [by_name[n] for n in rc["models"] if n in by_name]
    tags = [t for t, v in cfg["data"].items() if isinstance(v, dict)]
    lam = cfg["compression"]["lambda"]
    primary = cfg["compression"].get("decaf_primary", "decaf_lam6")
    n_new = 0
    for model in models:
        client = _client_for(cfg, model["name"])
        for tag in tags:
            spec = dict(cfg["data"][tag])
            spec["n_questions"] = rc["n_questions"]
            examples = load_examples_v2(spec, cfg["seed"], cfg["data"]["cache_dir"])[: rc["n_questions"]]
            for retriever in rc["retrievers"]:
                for ex in tqdm(examples, desc=f"retr[{model['label']}|{tag}|{retriever}]"):
                    ranked = rank(ex, retriever, rc["dense_model"])[: cfg["top_k"]]
                    sentences: List[str] = []
                    titles: List[str] = []
                    for pi, _score in ranked:
                        for s in contexts.split_sentences(ex.paragraphs[pi].text):
                            sentences.append(s)
                            titles.append(ex.paragraphs[pi].title)
                    if not sentences:
                        continue
                    rel, ce_ms = _ce_scores(scorer, cfg, tag, ex, sentences)
                    decaf_idx, _ = select_v2(sentences, ex.question, rel, primary, lam=lam)
                    n_iso = len(decaf_idx)
                    gold = _gold_sentences(ex)
                    for method in rc["methods"]:
                        if cache.has(model=model["name"], dataset=tag, retriever=retriever,
                                     qid=ex.qid, method=method):
                            continue
                        idx, stats = select_v2(sentences, ex.question, rel, method, lam=lam, n_iso=n_iso)
                        ctx = build_context(sentences, idx, titles)
                        resp = _generate(client, build_prompt(ex.question, [ctx], ["evidence"]))
                        scored = score_answer(resp["response"], ex.answer)
                        retained = {sentences[i].strip() for i in idx}
                        inter = len(retained & gold)
                        ctx_norm = _NORM.sub(" ", ctx.lower())
                        pred_norm = _NORM.sub(" ", scored["pred"].lower()).strip()
                        gold_norm = _NORM.sub(" ", ex.answer.lower()).strip()
                        cache.put({
                            "model": model["name"], "model_label": model["label"],
                            "family": model["family"], "quant": model["quant"],
                            "dataset": tag, "retriever": retriever, "qid": ex.qid, "method": method,
                            "question": ex.question, "gold": ex.answer, "pred": scored["pred"],
                            "em": scored["em"], "f1": scored["f1"],
                            "evidence_recall": inter / len(gold) if gold else float("nan"),
                            "evidence_precision": inter / len(retained) if retained else float("nan"),
                            "gold_in_evidence": float(bool(gold_norm) and gold_norm in ctx_norm),
                            "answer_support": float(bool(pred_norm) and pred_norm in ctx_norm),
                            "ce_ms": ce_ms, **stats, "compress_ms": ce_ms + stats["select_ms"],
                            "peak_rss_mb": resp["peak_rss_mb"], **_timing(resp),
                            "raw_response": resp["response"],
                        })
                        n_new += 1
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_v2.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = run(cfg)
    print(f"retrieval study: {n} new runs cached.")


if __name__ == "__main__":
    main()
