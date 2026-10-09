"""Pluggable relevance scorers for evidence selection.

The proposal's Sec. 12.1 specifies an off-the-shelf cross-encoder
(`BAAI/bge-reranker-base`, ~110M) run on CPU; the pilot used BM25. Both are
provided here behind one interface so the scorer can be *ablated*, because the
scorer is exactly where the proposal's central tension lives:

    a better scorer should retain better evidence, but it also inflates
    `T_compress`, and `T_compress` must stay below the decode time it saves.

`score()` returns scores in [0,1] plus the wall-clock cost of producing them, so
the cost of each scorer is a first-class measurement rather than an afterthought.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .contexts import tokens

_CE_CACHE: Dict[str, object] = {}


def _normalise(s: np.ndarray) -> np.ndarray:
    """Map raw scores to [0,1]; min-max so the lambda knob is comparable."""
    if s.size == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return np.ones_like(s) * 0.5
    return (s - lo) / (hi - lo)


def _bm25(sentences: List[str], question: str) -> np.ndarray:
    bm = BM25Okapi([tokens(s) for s in sentences])
    raw = np.asarray(bm.get_scores(tokens(question)), dtype=float)
    mx = raw.max()
    return np.clip(raw / mx, 0.0, 1.0) if mx > 0 else np.zeros_like(raw)


def load_cross_encoder(model_name: str = "BAAI/bge-reranker-base"):
    """Lazy-load (and cache) the cross-encoder. CPU, single thread-pool."""
    if model_name not in _CE_CACHE:
        from sentence_transformers import CrossEncoder
        _CE_CACHE[model_name] = CrossEncoder(model_name, device="cpu", max_length=512)
    return _CE_CACHE[model_name]


def _cross_encoder(sentences: List[str], question: str, model_name: str,
                   batch_size: int = 16) -> np.ndarray:
    model = load_cross_encoder(model_name)
    pairs = [(question, s) for s in sentences]
    raw = np.asarray(model.predict(pairs, batch_size=batch_size,
                                   show_progress_bar=False), dtype=float)
    return _normalise(raw)


def score(sentences: List[str], question: str, scorer: str = "bm25",
          ce_model: str = "BAAI/bge-reranker-base") -> Tuple[np.ndarray, float]:
    """Return (scores in [0,1], scoring wall-clock in ms)."""
    if not sentences:
        return np.zeros(0), 0.0
    t0 = time.perf_counter()
    if scorer == "bm25":
        s = _bm25(sentences, question)
    elif scorer in ("ce", "cross_encoder"):
        s = _cross_encoder(sentences, question, ce_model)
    else:
        raise ValueError(f"unknown scorer: {scorer}")
    return s, (time.perf_counter() - t0) * 1000.0


def warmup(scorer: str, ce_model: str = "BAAI/bge-reranker-base") -> float:
    """Pay the model-load cost before measurement so it is not charged to a run."""
    if scorer in ("ce", "cross_encoder"):
        t0 = time.perf_counter()
        _cross_encoder(["warmup sentence one.", "warmup sentence two."], "warmup?", ce_model)
        return (time.perf_counter() - t0) * 1000.0
    return 0.0
