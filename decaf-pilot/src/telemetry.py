"""System telemetry, contention preflight, and the memory-bandwidth roofline (E8).

Three jobs:
  * `sample()`   — a per-run snapshot of CPU load / available RAM, so contention
                   is recorded alongside every latency measurement rather than
                   silently inflating it.
  * `preflight()`— refuse (or warn) to measure on a busy machine.
  * `stream_bandwidth_gbs()` — an empirical memory-bandwidth ceiling for the
                   roofline analysis, plus `achieved_bandwidth_gbs()` which turns
                   a measured TPOT into the bandwidth it implies.

The roofline relation under test (proposal Sec. 5) is
    TPOT(L) ~= (W_model + KV(L)) / BW_mem
so a decode step touches the weights once plus the KV cache once; dividing those
bytes by the measured TPOT gives the *achieved* bandwidth, which can be compared
against the measured STREAM-like ceiling.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import psutil

# KV-cache geometry per model (bytes per token = 2 * layers * kv_heads * head_dim * dtype)
MODEL_GEOMETRY: Dict[str, Dict[str, float]] = {
    # Qwen2.5-3B: 36 layers, 16 q-heads, 2 kv-heads (GQA), head_dim 128
    "qwen2.5:3b-instruct":      {"layers": 36, "kv_heads": 2, "head_dim": 128, "weight_gb": 1.93},
    "qwen2.5:3b-instruct-q8_0": {"layers": 36, "kv_heads": 2, "head_dim": 128, "weight_gb": 3.26},
    "llama3.2:3b":              {"layers": 28, "kv_heads": 8, "head_dim": 128, "weight_gb": 2.02},
}
_KV_DTYPE_BYTES = 2.0  # llama.cpp default f16 KV cache


def kv_bytes_per_token(model: str) -> float:
    g = MODEL_GEOMETRY.get(model)
    if not g:
        return float("nan")
    return 2.0 * g["layers"] * g["kv_heads"] * g["head_dim"] * _KV_DTYPE_BYTES


def kv_bytes(model: str, n_tokens: float) -> float:
    return kv_bytes_per_token(model) * float(n_tokens)


def weight_bytes(model: str) -> float:
    g = MODEL_GEOMETRY.get(model)
    return float(g["weight_gb"]) * 1e9 if g else float("nan")


def achieved_bandwidth_gbs(model: str, tpot_ms: float, context_tokens: float) -> float:
    """Bytes touched per decode step / TPOT -> implied GB/s."""
    if not tpot_ms or tpot_ms <= 0:
        return float("nan")
    b = weight_bytes(model) + kv_bytes(model, context_tokens)
    return (b / 1e9) / (tpot_ms / 1000.0)


# --- contention ------------------------------------------------------------

def sample() -> Dict[str, float]:
    """Snapshot of machine state; attached to every measured run."""
    vm = psutil.virtual_memory()
    la1, la5, la15 = os.getloadavg()
    return {
        "sys_cpu_pct": psutil.cpu_percent(interval=None),
        "sys_load1": la1,
        "sys_load5": la5,
        "sys_mem_avail_gb": vm.available / 1e9,
        "sys_mem_used_pct": vm.percent,
        "sys_swap_used_gb": psutil.swap_memory().used / 1e9,
    }


BUSY_PROCS = ("ng", "node", "java", "chrome", "code", "mysqld", "firefox")


def busy_processes(min_rss_gb: float = 0.3) -> List[Dict]:
    """Large non-benchmark processes that would contend for the 4 physical cores."""
    out = []
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            if p.info["pid"] == me or not p.info["memory_info"]:
                continue
            rss = p.info["memory_info"].rss / 1e9
            if rss >= min_rss_gb and any(b in (p.info["name"] or "").lower() for b in BUSY_PROCS):
                out.append({"pid": p.info["pid"], "name": p.info["name"], "rss_gb": round(rss, 2)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(out, key=lambda d: -d["rss_gb"])


def preflight(max_load1: float = 1.5, min_mem_gb: float = 3.0, strict: bool = True,
              max_swap_used_pct: float = 50.0) -> Dict:
    """Check the machine is quiet enough to measure latency on."""
    psutil.cpu_percent(interval=None)
    time.sleep(1.0)
    st = sample()
    procs = busy_processes()
    # Load average and free memory are the hard gates: they are what actually
    # distorts a latency measurement. The process list is recorded and warned
    # about but does not block -- an editor or a browser sitting idle costs far
    # less than its resident size suggests, and requiring literally none of them
    # is a condition a working machine may never satisfy.
    problems = []
    if st["sys_load1"] > max_load1:
        problems.append(f"1-min load average {st['sys_load1']:.2f} > {max_load1}")
    if st["sys_mem_avail_gb"] < min_mem_gb:
        problems.append(f"only {st['sys_mem_avail_gb']:.1f} GB RAM available (< {min_mem_gb} GB)")
    # Swap was the failure this guard originally missed: MemAvailable looked
    # healthy (7.9 GB) while swap was already saturated, so the machine began
    # thrashing as soon as llama-server grew, inflating prefill ~15x.
    sw = psutil.swap_memory()
    sw_pct = sw.percent if sw.total else 0.0
    if sw_pct > max_swap_used_pct:
        problems.append(f"swap {sw_pct:.0f}% used ({sw.used/1e9:.1f} GB) — the machine is "
                        f"already under memory pressure and will thrash")
    if procs:
        top = ", ".join(f"{p['name'][:28]}({p['rss_gb']}GB)" for p in procs[:5])
        print(f"note: large processes present (not blocking): {top}")
    report = {**st, "busy_processes": procs, "problems": problems, "clean": not problems}
    if problems:
        msg = "Machine is not quiet enough for latency measurement:\n  - " + "\n  - ".join(problems)
        if strict:
            raise SystemExit(msg + "\n\nClose these and retry, or pass --allow-contention "
                                   "to measure anyway (noise will be logged per run).")
        print("WARNING: " + msg)
    return report


# --- roofline ceiling ------------------------------------------------------

def stream_bandwidth_gbs(size_mb: int = 256, reps: int = 5) -> Dict[str, float]:
    """STREAM-like triad/copy bandwidth, as an empirical ceiling for the roofline.

    Arrays are sized well beyond the 8 MB L3 so this measures DRAM, not cache.
    """
    n = int(size_mb * 1e6 / 8)
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64) * 2.0
    c = np.empty(n, dtype=np.float64)
    out: Dict[str, List[float]] = {"copy": [], "triad": []}
    for _ in range(reps):
        t = time.perf_counter()
        np.copyto(c, a)
        dt = time.perf_counter() - t
        out["copy"].append((2 * a.nbytes / 1e9) / dt)          # 1 read + 1 write
        t = time.perf_counter()
        np.multiply(b, 3.0, out=c); np.add(a, c, out=c)
        dt = time.perf_counter() - t
        out["triad"].append((4 * a.nbytes / 1e9) / dt)         # ~3 reads + 1 write
    return {
        "stream_copy_gbs": float(np.median(out["copy"])),
        "stream_triad_gbs": float(np.median(out["triad"])),
        "stream_peak_gbs": float(max(np.median(out["copy"]), np.median(out["triad"]))),
        "array_mb": size_mb,
    }
