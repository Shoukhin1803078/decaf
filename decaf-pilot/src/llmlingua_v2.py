"""LLMLingua-2 prompt-compression baseline (external, training-free).

Compresses the retrieved-context *text* to a target token-retention `rate` and
times the compression; that time IS the baseline's T_compress. Runs on CPU.
"""
from __future__ import annotations

import time
from typing import Tuple

_CACHE: dict = {}


class LLMLinguaCompressor:
    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                 device: str = "cpu"):
        from llmlingua import PromptCompressor

        if model_name not in _CACHE:
            _CACHE[model_name] = PromptCompressor(model_name=model_name, use_llmlingua2=True,
                                                  device_map=device)
        self.model = _CACHE[model_name]

    def compress(self, text: str, rate: float) -> Tuple[str, float, int, int]:
        """Return (compressed_text, compress_ms, origin_tokens, compressed_tokens)."""
        t0 = time.perf_counter()
        out = self.model.compress_prompt(text, rate=rate, force_tokens=["\n", "?"],
                                         drop_consecutive=True)
        ms = (time.perf_counter() - t0) * 1000.0
        return (out.get("compressed_prompt", text), ms,
                int(out.get("origin_tokens", 0) or 0), int(out.get("compressed_tokens", 0) or 0))
