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
    idx = idx[: spec["n_questions"]]
    examples = _build_examples(ds, idx)
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
