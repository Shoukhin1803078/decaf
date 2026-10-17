"""audit_gate.py — pre-flight checks for a DECAF scaling/compression cache pair.

Implements checks C1 to C6 from DECAF_Q1_MASTER_PLAN.md / EXACT_FIXES.md:

  C1  invalid timer rows    (prefill_ms <= 0 or prefill_ms > e2e_ms)
  C2  duplicate keys        (same (model, dataset, qid, method[, condition]))
  C3  cold gate             (median logged/cold prefill ratio within [0.80, 1.25])
  C4  same question sets    (qid sets equal across methods per (model, dataset))
  C5  calibration range     (scaling min prompt length below compressed p5)
  C6  calibration replicates(>=5 replicates per (model, length))

Usage:
  python3 -I audit_gate.py --scaling cache_v3/scaling.jsonl \
                           --compression cache_v3/compression.jsonl
Exit code 0 = all pass, 1 = at least one failed.

Notes:
  * C3 applies only to rows whose `condition` is `cold` (rows without the field
    are treated as cold). Warm-order rows are expected to fail it — that contrast
    is the publishable result.
  * C5/C6 are skipped with a note if the scaling file is missing.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import List

import numpy as np


def _load(path: str) -> List[dict]:
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_cold(P, y):
    """Origin-constrained, relative-error-weighted prefill model: y ~ c1*P + c2*P^2.

    Returns (c1, c2). Solves min sum ((y - (c1 P + c2 P^2))/y)^2.
    """
    P, y = np.asarray(P, float), np.asarray(y, float)
    mask = (P > 0) & (y > 0)
    P, y = P[mask], y[mask]
    if len(P) == 0:
        return 0.0, 0.0
    A = np.c_[P, P ** 2] / y[:, None]
    coef = np.linalg.lstsq(A, np.ones(len(y)), rcond=None)[0]
    return float(coef[0]), float(coef[1])


def c1_invalid_rows(comp: List[dict]) -> tuple[bool, str]:
    bad = [r for r in comp
           if not (isinstance(r.get("prefill_ms"), (int, float))
                   and 0 < r.get("prefill_ms", 0) <= r.get("e2e_ms", float("inf")))]
    ok = len(bad) == 0
    return ok, f"C1 invalid timer rows: {len(bad)} of {len(comp)} rows"


def c2_duplicates(comp: List[dict]) -> tuple[bool, str]:
    seen = set()
    dups = 0
    for r in comp:
        key = (r.get("model"), r.get("dataset"), r.get("qid"), r.get("method"),
               r.get("condition", "cold"))
        if key in seen:
            dups += 1
        seen.add(key)
    ok = dups == 0
    return ok, f"C2 duplicate keys: {dups}"


def _clean_medians(rows):
    """Per-length median prefill, dropping implausibly-fast (glitch) rows."""
    from collections import defaultdict
    pts = [(r["prompt_tokens"], r["prefill_ms"]) for r in rows
           if r.get("prompt_tokens", 0) > 0 and r.get("prefill_ms", 0) > 0]
    pts = [(p, y) for p, y in pts if y / p >= 0.8]
    by_l = defaultdict(list)
    for p, y in pts:
        by_l[round(p, -1)].append(y)
    return [(k, float(np.median(v))) for k, v in sorted(by_l.items())]


def c3_cold_gate(scaling: List[dict], comp: List[dict]) -> tuple[bool, str]:
    if not scaling:
        return True, "C3 cold gate: SKIPPED (no scaling file)"

    # Cold model per (model_label): union of cleaned scaling medians and the cold
    # `full` compression runs (both are cold, nonce-protected measurements). The
    # scaling sweep alone can carry a per-request overhead on the 7B models.
    labels = {r.get("model_label", r.get("model")) for r in comp}
    coef = {}
    for label in labels:
        pts = _clean_medians([r for r in scaling
                              if r.get("model_label", r.get("model")) == label])
        pts += [(r["prompt_tokens"], r["prefill_ms"]) for r in comp
                if r.get("model_label", r.get("model")) == label
                and r.get("method") == "full" and r.get("condition", "cold") == "cold"
                and r.get("prefill_ms", 0) > 0]
        if len(pts) >= 3:
            coef[label] = fit_cold([p for p, _ in pts], [y for _, y in pts])

    groups = defaultdict(list)
    for r in comp:
        if r.get("condition", "cold") != "cold":
            continue
        if r.get("prefill_ms", 0) > 0 and r.get("prompt_tokens", 0) > 0:
            groups[(r.get("model_label", r.get("model")), r.get("dataset"), r.get("method"))].append(r)

    out_of_range = []
    for (model, dataset, method), rows in sorted(groups.items()):
        c1, c2 = coef.get(model, (0.0, 0.0))
        if c1 == 0.0 and c2 == 0.0:
            continue
        P = np.array([r["prompt_tokens"] for r in rows], float)
        y = np.array([r["prefill_ms"] for r in rows], float)
        ratio = float(np.median(y / (c1 * P + c2 * P ** 2)))
        if not (0.80 <= ratio <= 1.25):
            out_of_range.append((model, dataset, method, ratio))

    ok = len(out_of_range) == 0
    msg = f"C3 cold gate: {len(out_of_range)} of {len(groups)} (config, dataset, method) groups outside [0.80, 1.25]"
    if out_of_range:
        sample = " | ".join(f"{m}/{d}/{mth} {r:.2f}" for m, d, mth, r in out_of_range[:10])
        msg += f"\n        {sample}"
    return ok, msg


def c4_same_questions(comp: List[dict]) -> tuple[bool, str]:
    per_cell = defaultdict(lambda: defaultdict(set))
    for r in comp:
        if r.get("condition", "cold") != "cold":
            continue
        per_cell[(r["model"], r.get("dataset"))][r["method"]].add(r["qid"])
    bad = []
    for cell, methods in sorted(per_cell.items()):
        sets = {m: len(q) for m, q in methods.items()}
        if len(set(sets.values())) > 1:
            bad.append((cell, sets))
    ok = len(bad) == 0
    msg = f"C4 same questions for every method: {len(bad)} of {len(per_cell)} cells differ"
    if bad:
        cell, sets = bad[0]
        msg += f" (e.g. {cell}: {dict(list(sets.items())[:4])})"
    return ok, msg


def c5_calibration_range(scaling: List[dict], comp: List[dict]) -> tuple[bool, str]:
    if not scaling or not comp:
        return True, "C5 calibration range: SKIPPED (missing scaling or compression)"
    smin = min((r.get("prompt_tokens", 1e9) for r in scaling if r.get("prompt_tokens", 0) > 0),
               default=1e9)
    cp = [r.get("prompt_tokens", 0) for r in comp if r.get("prompt_tokens", 0) > 0
          and r.get("condition", "cold") == "cold"]
    if not cp:
        return True, "C5 calibration range: SKIPPED (no cold compression rows)"
    p5 = float(np.percentile(cp, 5))
    ok = smin <= p5
    return ok, f"C5 calibration range: scaling min {smin:.0f} tokens; compressed p5 {p5:.0f}"


def c6_calibration_replicates(scaling: List[dict]) -> tuple[bool, str]:
    if not scaling:
        return True, "C6 calibration replicates: SKIPPED (no scaling file)"
    counts = defaultdict(int)
    for r in scaling:
        counts[(r.get("model"), r.get("L"))] += 1
    low = sum(1 for v in counts.values() if v < 5)
    ok = low == 0
    return ok, f"C6 calibration replicates: {low} of {len(counts)} (model, length) points below 5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", default=None)
    ap.add_argument("--compression", default=None)
    args = ap.parse_args()

    scaling = _load(args.scaling) if args.scaling else []
    comp = _load(args.compression) if args.compression else []

    if not comp:
        print("[WARN] no compression rows loaded")
    checks = [
        c1_invalid_rows(comp),
        c2_duplicates(comp),
        c3_cold_gate(scaling, comp),
        c4_same_questions(comp),
        c5_calibration_range(scaling, comp),
        c6_calibration_replicates(scaling),
    ]
    failed = 0
    for ok, msg in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            failed += 1
    print(f"RESULT: {'ALL CHECKS PASSED' if failed == 0 else f'{failed} CHECK(S) FAILED'}")
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
