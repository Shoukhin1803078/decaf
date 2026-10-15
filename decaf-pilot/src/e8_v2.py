"""E8 telemetry: CPU utilization and achieved weight-streaming bandwidth.

For each (model, context length) we run one generation while sampling CPU utilization,
then derive an **achieved decode bandwidth** proxy: the model's weight bytes streamed per
second during decode (`model_bytes × tokens/s`). This is the practical roofline check for
the "memory-bandwidth-bound decode" hypothesis on this host.

Writes `cache_v2/e8.jsonl`; the figure is produced by `src.analyze_v2`.
"""
from __future__ import annotations

import argparse
import os
import threading

import requests
from tqdm import tqdm

from . import contexts
from .llm import build_prompt
from .run_experiments import JsonlCache, _timing
from .run_v2 import ROOT, _client_for, load_config


def _model_sizes(cfg: dict) -> dict:
    r = requests.get(cfg["ollama"]["host"].rstrip("/") + "/api/tags", timeout=5).json()
    return {m["name"]: float(m.get("size") or 0) for m in r.get("models", [])}


class CpuSampler:
    """Mean CPU utilization (%) sampled in a background thread."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.samples = []
        self._stop = False
        self._t = None

    def _loop(self):
        import psutil

        psutil.cpu_percent(interval=None)  # prime
        while not self._stop:
            self.samples.append(psutil.cpu_percent(interval=self.interval))

    def __enter__(self):
        self.samples = []
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
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float("nan")


def run(cfg: dict) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["e8_cache"]), ["model", "L", "rep"])
    sizes = _model_sizes(cfg)
    q = cfg["scaling"]["question"]
    n_new = 0
    for m in cfg["models"]:
        client = _client_for(cfg, m["name"])
        client.generate(build_prompt(q, ["Warmup."], ["reference"]))
        mb = sizes.get(m["name"], 0.0)
        for L in cfg["e8"]["lengths"]:
            for r in range(cfg["e8"]["replicates"]):
                if cache.has(model=m["name"], L=L, rep=r):
                    continue
                filler = " ".join(contexts.filler_sentences(L, seed=cfg["seed"] + r))
                nonce = f"Run E8 {L}-{r}-{L * 7 + r}."
                prompt = build_prompt(q, [nonce + " " + filler], ["reference"])
                with CpuSampler() as cs:
                    resp = client.generate(prompt)
                t = _timing(resp)
                dec_s = t["decode_ms"] / 1000.0
                toks = t["output_tokens"]
                pre_s = t["prefill_ms"] / 1000.0
                cache.put({
                    "model": m["name"], "model_label": m["label"], "family": m["family"],
                    "quant": m["quant"], "L": L, "rep": r, "model_size_gb": mb / 1e9,
                    "cpu_pct": cs.mean,
                    "decode_tok_s": (toks / dec_s) if dec_s > 0 else float("nan"),
                    "decode_bw_gb_s": (mb * toks / dec_s / 1e9) if dec_s > 0 else float("nan"),
                    "prefill_tok_s": (t["prompt_tokens"] / pre_s) if pre_s > 0 else float("nan"),
                    **t})
                n_new += 1
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_v2.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("ollama", {})["model"] = cfg["models"][0]["name"]
    n = run(cfg)
    print(f"E8 telemetry: {n} new runs cached.")


if __name__ == "__main__":
    main()
