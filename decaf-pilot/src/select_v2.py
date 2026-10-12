"""Evidence selection v2 — cross-encoder relevance + DECAF CPU-aware gate.

Methods
-------
full                  : keep all retrieved sentences.
fixed_{r}             : top round(r*n) by relevance (fixed ratio).
rel_iso               : top-m by relevance, m matched to DECAF's retained count
                        (iso-budget relevance baseline; sharp ablation of Sec. 17).
decaf_lam{K}          : single-pass greedy, accept while
                        gain >= lambda * cost, gain = rel * (0.5 + 0.5*cov).
decaf_nocov_lam{K}    : DECAF with the coverage term off (ablation).
decaf_noreorder_lam{K}: DECAF selection but retained in document order
                        (relevance-reordering ablation).

Relevance arrives pre-computed (cross-encoder) so the selector itself is
decoder-free and its own cost is only the greedy pass plus the scorer.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contexts import content_terms, tokens


def _tok(s: str) -> int:
    return max(1, len(tokens(s)))


def select_v2(
    sentences: Sequence[str],
    question: str,
    rel: np.ndarray,
    method: str,
    lam: float = 1.5,
    n_iso: Optional[int] = None,
) -> Tuple[List[int], Dict]:
    """Return (retained_indices, stats). Greedy pass is timed as `select_ms`."""
    t0 = time.perf_counter()
    n = len(sentences)
    rel = np.asarray(rel, dtype=float)
    order = [int(i) for i in np.argsort(-rel)]
    tok = [_tok(s) for s in sentences]
    total_tok = max(1, sum(tok))
    qterms = content_terms(question)
    selected: List[int] = []

    if method == "full":
        selected = order
    elif method.startswith("fixed_"):
        r = float(method.split("_", 1)[1])
        m = max(1, int(math.ceil(r * n)))
        selected = order[:m]
    elif method == "rel_iso":
        m = n_iso if n_iso is not None else max(1, n // 2)
        selected = order[:m]
    elif method.startswith("decaf"):
        use_cov = "nocov" not in method
        reorder = "noreorder" not in method
        lam_eff = lam
        if "lam" in method:
            lam_eff = float(method.split("lam")[1])
        covered: set = set()
        for i in order:
            new = (content_terms(sentences[i]) & qterms) - covered
            cov = len(new) / max(1, len(qterms))
            gain = rel[i] * (0.5 + 0.5 * cov) if use_cov else rel[i]
            cost = tok[i] / total_tok
            if gain - lam_eff * cost >= 0.0:
                selected.append(i)
                covered |= new
        if not selected:
            selected = [order[0]]
        if not reorder:
            selected = sorted(selected)
    else:
        raise ValueError(method)

    selected = list(dict.fromkeys(selected))
    sel_tok = sum(tok[i] for i in selected)
    stats = {
        "n_total": n,
        "n_selected": len(selected),
        "tokens_total": total_tok,
        "tokens_selected": sel_tok,
        "retained_ratio": sel_tok / total_tok,
        "select_ms": (time.perf_counter() - t0) * 1000.0,
    }
    return selected, stats


def build_context(
    sentences: Sequence[str], selected: Sequence[int], titles: Optional[Sequence[str]] = None
) -> str:
    parts = []
    for i in selected:
        prefix = f"[{titles[i]}] " if titles else ""
        parts.append(prefix + sentences[i])
    return "\n".join(parts)
