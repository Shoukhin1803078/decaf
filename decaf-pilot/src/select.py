"""Evidence scoring and selection — fully parametric, so every Sec. 17 ablation
is a flag rather than a separate code path.

Method spec grammar (parsed by `parse_method`)
----------------------------------------------
    full                     keep every retrieved sentence
    closed_book              no context at all (retrieval-harm reference point)
    fixed_<r>                top ceil(r*n) sentences by relevance
    rel_iso                  top-m by relevance, m matched to DECAF's retained
                             count (the iso-budget baseline: the sharp test of
                             whether DECAF's *rule* beats plain ranking)
    decaf_lam<L>             full DECAF
    ...optional modifiers, any order, before `lam`:
       _ce                   cross-encoder scorer instead of BM25
       _nocov                coverage term off
       _nocost               CPU-cost term off (uniform cost)
       _absms                CPU cost in *absolute ms* from the measured curve
       _noreorder            keep document order instead of relevance order

The selection rule (proposal Sec. 12.2) is single-pass and decoder-free:

    accept s_i  while  MarginalQualityGain(s_i) >= lambda * MarginalCPUCost(s_i)

Cost modes
----------
`relative`   cost_i = tokens_i / tokens_total           (pilot behaviour)
`absolute_ms` cost_i = tokens_i * ms_per_token          (from the fitted
             prefill/TPOT curve of this model+quantization, so lambda is in
             quality-per-millisecond and is *hardware-meaningful*: the same
             lambda selects differently on a different model or quantization,
             which is what makes the mechanism "CPU-aware" and is testable in E5)
`uniform`    cost_i = 1 / n                             (ablation: length-blind)

Note that under a *linear* cost model `relative` is proportional to token count,
so it coincides with token-proportionality up to a constant absorbed into
lambda. `absolute_ms` is therefore the mode in which CPU-awareness can actually
be distinguished, and `uniform` is the mode that removes it entirely.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .contexts import content_terms, tokens
from . import scorer as scorer_mod


@dataclass
class MethodSpec:
    kind: str                       # full | closed_book | fixed | rel_iso | decaf
    lam: float = 6.0
    ratio: float = 0.5
    scorer: str = "bm25"            # bm25 | ce
    use_coverage: bool = True
    cost_mode: str = "relative"     # relative | absolute_ms | uniform
    reorder: bool = True
    raw: str = ""


def parse_method(name: str) -> MethodSpec:
    """Parse a method-spec string into a MethodSpec."""
    if name == "full":
        return MethodSpec("full", raw=name)
    if name in ("closed_book", "no_context"):
        return MethodSpec("closed_book", raw=name)
    if name.startswith("fixed_"):
        rest = name[len("fixed_"):]
        sc = "bm25"
        if rest.startswith("ce_"):
            sc, rest = "ce", rest[3:]
        return MethodSpec("fixed", ratio=float(rest), scorer=sc, raw=name)
    if name.startswith("rel_iso"):
        sc = "ce" if "_ce" in name else "bm25"
        return MethodSpec("rel_iso", scorer=sc, raw=name)
    if name.startswith("decaf"):
        body, _, lam_s = name.partition("lam")
        lam = float(lam_s) if lam_s else 6.0
        mods = body.split("_")
        return MethodSpec(
            "decaf",
            lam=lam,
            scorer="ce" if "ce" in mods else "bm25",
            use_coverage="nocov" not in mods,
            cost_mode=("uniform" if "nocost" in mods
                       else "absolute_ms" if "absms" in mods else "relative"),
            reorder="noreorder" not in mods,
            raw=name,
        )
    raise ValueError(f"unknown method: {name}")


# --- cost model ------------------------------------------------------------

@dataclass
class CostModel:
    """Marginal CPU cost per context token, from the measured scaling curve.

    `ms_per_token` = d(prefill)/dL + N_out * d(TPOT)/dL, i.e. the true marginal
    end-to-end cost of admitting one more context token on this machine for this
    model+quantization and generation length.
    """
    ms_per_token: float = 4.1
    n_out: float = 32.0
    source: str = "default"

    @classmethod
    def from_fit(cls, prefill_slope: float, tpot_slope: float, n_out: float,
                 source: str = "fit") -> "CostModel":
        return cls(ms_per_token=float(prefill_slope + n_out * tpot_slope),
                   n_out=float(n_out), source=source)


def _tok(s: str) -> int:
    return max(1, len(tokens(s)))


def select(
    sentences: List[str],
    question: str,
    method: str,
    cost_model: Optional[CostModel] = None,
    n_iso: Optional[int] = None,
    precomputed: Optional[Tuple[np.ndarray, float]] = None,
    ce_model: str = "BAAI/bge-reranker-base",
) -> Tuple[List[int], Dict]:
    """Return (retained indices in output order, stats including T_compress).

    `precomputed` lets several methods share one scorer pass; the shared scoring
    cost is then charged to each of them (it would be paid once per query in a
    real deployment, so charging it is the honest accounting).
    """
    spec = parse_method(method)
    cost_model = cost_model or CostModel()
    n = len(sentences)
    t0 = time.perf_counter()

    if spec.kind == "closed_book":
        return [], {
            "method": method, "n_total": n, "n_selected": 0, "tokens_total": 0,
            "tokens_selected": 0, "retained_ratio": 0.0, "retained_frac_sents": 0.0,
            "compress_ms": 0.0, "score_ms": 0.0, "select_ms": 0.0,
            "scorer": spec.scorer, "cost_mode": spec.cost_mode, "lam": spec.lam,
        }

    # --- scoring (the expensive stage) ---
    if precomputed is not None:
        rel, score_ms = precomputed
    else:
        rel, score_ms = scorer_mod.score(sentences, question, spec.scorer, ce_model)

    t_sel = time.perf_counter()
    order = [int(i) for i in np.argsort(-rel)]
    tok = [_tok(s) for s in sentences]
    total_tok = max(1, sum(tok))

    if spec.kind == "full":
        selected = list(range(n))
    elif spec.kind == "fixed":
        selected = order[: max(1, int(math.ceil(spec.ratio * n)))]
    elif spec.kind == "rel_iso":
        m = n_iso if n_iso is not None else max(1, n // 2)
        selected = order[: max(1, min(m, n))]
    elif spec.kind == "decaf":
        qterms = content_terms(question)
        covered: set = set()
        selected = []
        for i in order:
            new = (content_terms(sentences[i]) & qterms) - covered
            cov = len(new) / max(1, len(qterms))
            gain = rel[i] * (0.5 + 0.5 * cov) if spec.use_coverage else float(rel[i])
            if spec.cost_mode == "uniform":
                cost = 1.0 / max(1, n)
            elif spec.cost_mode == "absolute_ms":
                cost = tok[i] * cost_model.ms_per_token
            else:
                cost = tok[i] / total_tok
            if gain - spec.lam * cost >= 0.0:
                selected.append(i)
                covered |= new
        if not selected:
            selected = [order[0]]
    else:
        raise ValueError(spec.kind)

    selected = list(dict.fromkeys(selected))
    if spec.kind == "full":
        selected = order if spec.reorder else selected
    elif not spec.reorder:
        selected = sorted(selected)          # document order

    select_ms = (time.perf_counter() - t_sel) * 1000.0
    sel_tok = sum(tok[i] for i in selected)
    return selected, {
        "method": method,
        "n_total": n,
        "n_selected": len(selected),
        "tokens_total": total_tok,
        "tokens_selected": sel_tok,
        "retained_ratio": sel_tok / total_tok,
        "retained_frac_sents": len(selected) / max(1, n),
        "score_ms": score_ms,
        "select_ms": select_ms,
        "compress_ms": score_ms + select_ms,
        "scorer": spec.scorer,
        "cost_mode": spec.cost_mode,
        "lam": spec.lam,
        "use_coverage": spec.use_coverage,
        "reorder": spec.reorder,
        "cost_ms_per_token": cost_model.ms_per_token,
        "_total_ms_unused": (time.perf_counter() - t0) * 1000.0,
    }


def build_context(sentences: List[str], selected: List[int],
                  titles: Optional[List[str]] = None) -> str:
    parts = []
    for i in selected:
        prefix = f"[{titles[i]}] " if titles else ""
        parts.append(prefix + sentences[i])
    return "\n".join(parts)


def scorers_needed(methods: List[str]) -> List[str]:
    """Distinct scorers used by a method list (so each is warmed up once)."""
    out = []
    for m in methods:
        try:
            s = parse_method(m).scorer
        except ValueError:
            continue
        if s not in out:
            out.append(s)
    return out
