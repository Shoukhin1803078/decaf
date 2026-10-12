"""Cross-encoder relevance scorer (proposal Sec. 12.1).

`BAAI/bge-reranker-base` (~110M) run on CPU. This is DECAF's fine-grained
evidence scorer; its forward-pass time *is* the measured compressor overhead
T_compress, so it is timed explicitly and returned alongside the scores.
"""
from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np

_CACHE: dict = {}


class CrossEncoderScorer:
    def __init__(self, model_name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        if model_name not in _CACHE:
            _CACHE[model_name] = CrossEncoder(model_name, max_length=max_length)
        self.model = _CACHE[model_name]
        self.model_name = model_name

    def score(self, question: str, sentences: Sequence[str]) -> Tuple[np.ndarray, float]:
        """Return (relevance in [0,1], forward_ms). Relevance is min-max scaled."""
        if not sentences:
            return np.zeros(0, dtype=float), 0.0
        pairs = [(question, s) for s in sentences]
        t0 = time.perf_counter()
        raw = np.asarray(self.model.predict(pairs), dtype=float).ravel()
        ms = (time.perf_counter() - t0) * 1000.0
        # bge-reranker emits logits; squash to (0,1) then min-max normalise.
        sig = 1.0 / (1.0 + np.exp(-raw))
        lo, hi = float(sig.min()), float(sig.max())
        if hi > lo:
            sig = (sig - lo) / (hi - lo)
        return np.clip(sig, 0.0, 1.0), ms
