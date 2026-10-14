"""Dataset loading v2: HotpotQA distractor + 2WikiMultihopQA, unified.

Both ship candidate paragraphs with gold supporting facts, so the same
selection pipeline applies. v2 handles the differing on-disk schemas
(dict-of-lists vs list-of-pairs) and caches the sampled subset as JSON.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

from . import contexts
from .data import Example, Paragraph


def _parse_context(context) -> List[Tuple[str, List[str]]]:
    if isinstance(context, dict):
        return [(t, list(s)) for t, s in zip(context["title"], context["sentences"])]
    out: List[Tuple[str, List[str]]] = []
    for c in context:
        if isinstance(c, dict):
            out.append((c["title"], list(c["sentences"])))
        else:  # [title, [sentences]]
            out.append((c[0], list(c[1])))
    return out


def _parse_sf(sf) -> List[Tuple[str, int]]:
    if isinstance(sf, dict):
        return list(zip(sf["title"], sf["sent_id"]))
    return [(x[0], x[1]) for x in sf]


def _build_examples(ds, idx: List[int]) -> List[Example]:
    examples: List[Example] = []
    for i in idx:
        row = ds[i]
        paras = [Paragraph(t, s) for t, s in _parse_context(row["context"])]
        sf = _parse_sf(row["supporting_facts"])
        gold_titles = sorted({t for t, _ in sf})
        support: Dict[str, List[int]] = {}
        for t, s in sf:
            support.setdefault(t, []).append(int(s))
        qid = row.get("id", row.get("_id", i))
        examples.append(
            Example(
                qid=str(qid),
                question=row["question"],
                answer=row["answer"],
                paragraphs=paras,
                gold_titles=gold_titles,
                qtype=row.get("type", "unknown"),
                level=row.get("level", "unknown"),
                support_sentences=support,
            )
        )
    return examples


def _iter_docs(sr):
    """Yield per-document dicts from either a list-of-dicts or a columnar dict-of-lists."""
    if isinstance(sr, dict):
        keys = list(sr.keys())
        n = len(sr[keys[0]]) if keys else 0
        for j in range(n):
            yield {k: sr[k][j] for k in keys}
    else:
        for d in (sr or []):
            yield d if isinstance(d, dict) else {}


def _build_trivia(ds, idx: List[int], max_sent: int = 20) -> List[Example]:
    """TriviaQA (rc.wikipedia): candidate docs are Wikipedia articles (entity_pages),
    truncated to `max_sent` sentences; gold = sentences containing the answer aliases."""
    examples: List[Example] = []
    for i in idx:
        row = ds[i]
        docs = list(_iter_docs(row.get("entity_pages")))
        if not docs:
            docs = list(_iter_docs(row.get("search_results")))
        paras: List[Paragraph] = []
        for d in docs:
            text = d.get("wiki_context") or d.get("search_context") or ""
            if not text.strip():
                continue
            sents = contexts.split_sentences(text)[:max_sent]
            if sents:
                paras.append(Paragraph(d.get("title") or "", sents))
        if not paras:
            continue
        ans = row.get("answer") or {}
        aliases = [ans.get("value", "")] + list(ans.get("aliases") or [])
        aliases = [str(a).strip().lower() for a in aliases if a and str(a).strip()]
        support: Dict[str, List[int]] = {}
        gold_titles: List[str] = []
        for p in paras:
            hits = [j for j, s in enumerate(p.sentences) if any(a in s.lower() for a in aliases)]
            if hits:
                support[p.title] = hits
                gold_titles.append(p.title)
        if not gold_titles:
            continue
        examples.append(
            Example(
                qid=str(row.get("question_id", i)),
                question=row["question"],
                answer=str(ans.get("value", "")),
                paragraphs=paras,
                gold_titles=sorted(set(gold_titles)),
                qtype="single-hop",
                level="triviaqa",
                support_sentences=support,
            )
        )
    return examples


def _build_longbench(ds, idx: List[int], chunk_sent: int = 6, max_sent: int = 600) -> List[Example]:
    """LongBench (multi-document QA): the provided context is one long concatenated string.
    We chunk it into ~`chunk_sent`-sentence segments (the paraphrase of retrieved paragraphs)
    and take gold evidence from answer-bearing sentences."""
    examples: List[Example] = []
    for i in idx:
        row = ds[i]
        sents = contexts.split_sentences(row.get("context") or "")[:max_sent]
        paras: List[Paragraph] = []
        for j in range(0, len(sents), chunk_sent):
            seg = sents[j:j + chunk_sent]
            if seg:
                paras.append(Paragraph(f"seg{len(paras)}", seg))
        if not paras:
            continue
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        aliases = [str(a).strip().lower() for a in answers if str(a).strip()]
        support: Dict[str, List[int]] = {}
        gold_titles: List[str] = []
        for p in paras:
            hits = [j for j, s in enumerate(p.sentences) if any(a in s.lower() for a in aliases)]
            if hits:
                support[p.title] = hits
                gold_titles.append(p.title)
        examples.append(
            Example(
                qid=str(row.get("_id", i)),
                question=str(row.get("question") or row.get("input") or ""),
                answer=str(answers[0]) if answers else "",
                paragraphs=paras,
                gold_titles=sorted(set(gold_titles)),
                qtype="multidoc-qa",
                level="longbench",
                support_sentences=support,
            )
        )
    return examples


def load_examples_v2(spec: dict, seed: int, cache_dir: str) -> List[Example]:
    """Load or materialise a sampled subset for a dataset spec."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{spec['tag']}_{spec['split']}_{spec['n_questions']}.json")
    if os.path.exists(path):
        with open(path) as f:
            return [Example.from_record(r) for r in json.load(f)]

    from datasets import load_dataset

    ds = load_dataset(spec["hf_name"], spec.get("hf_config"), split=spec["split"])
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if spec.get("loader") == "trivia":
        # oversample: some questions are dropped (no gold doc / too few documents)
        pool = idx[: min(len(idx), spec["n_questions"] * 5)]
        examples = _build_trivia(ds, pool)[: spec["n_questions"]]
    elif spec.get("loader") == "longbench":
        examples = _build_longbench(ds, idx[: spec["n_questions"]])
    else:
        examples = _build_examples(ds, idx[: spec["n_questions"]])
    with open(path, "w") as f:
        json.dump([e.to_record() for e in examples], f)
    return examples


def load_all(cfg: dict) -> Dict[str, List[Example]]:
    out: Dict[str, List[Example]] = {}
    for tag, spec in cfg["data"].items():
        if not isinstance(spec, dict):
            continue
        out[tag] = load_examples_v2(spec, cfg["seed"], cfg["data"]["cache_dir"])
    return out
