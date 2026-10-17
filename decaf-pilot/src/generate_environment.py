"""Record hardware and software environment to ENVIRONMENT.md.

Run: .venv/bin/python -m src.generate_environment
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=10).strip()
    except Exception as e:
        return f"<error: {e}>"


def _ollama_models() -> list:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        return data.get("models", [])
    except Exception:
        return []


def _pip_freeze() -> list:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=15)
        return [line for line in out.strip().splitlines() if line]
    except Exception:
        return []


def generate() -> str:
    lines = []
    lines.append("# DECAF v3 — Environment Record")
    lines.append("")
    lines.append("Generated automatically before the v3 re-run.")
    lines.append("")

    lines.append("## Hardware")
    lines.append(f"- OS: {platform.system()} {platform.release()} {platform.version()}")
    lines.append(f"- Machine: {platform.machine()}")
    lines.append(f"- Processor: {_sh('sysctl -n machdep.cpu.brand_string')}")
    lines.append(f"- Physical cores: {_sh('sysctl -n hw.physicalcpu')}")
    lines.append(f"- Logical cores: {_sh('sysctl -n hw.logicalcpu')}")
    lines.append(f"- Memory (total): {_sh('sysctl -n hw.memsize')} bytes")
    lines.append("")

    lines.append("## Software")
    lines.append(f"- Python: {sys.version}")
    lines.append(f"- Ollama: {_sh('ollama version') if _sh('ollama version') else 'not found'}")
    lines.append(f"- Ollama host: http://localhost:11434")
    lines.append("")

    lines.append("## Loaded models")
    for m in _ollama_models():
        details = m.get("details", {})
        lines.append(f"- {m['name']} ({details.get('parameter_size', '?')}, "
                      f"{details.get('quantization_level', '?')}) — "
                      f"{m['size'] / 1e9:.1f} GB, context {details.get('context_length', '?')}")
    lines.append("")

    lines.append("## Python packages (relevant)")
    frozen = _pip_freeze()
    relevant = ["datasets", "rank-bm25", "requests", "pandas", "numpy", "scipy",
                "scikit-learn", "matplotlib", "pyyaml", "tqdm", "tenacity", "tabulate",
                "sentence-transformers", "llmlingua", "psutil"]
    for pkg in relevant:
        found = [p for p in frozen if p.lower().startswith(pkg.lower().replace("-", "_"))]
        lines.append(f"- {pkg}: {found[0] if found else 'NOT INSTALLED'}")
    lines.append("")

    lines.append("## Ollama runtime options (pinned for v3)")
    lines.append("- num_thread: 8")
    lines.append("- num_batch: 512")
    lines.append("- num_ctx: 20480")
    lines.append("- temperature: 0.0")
    lines.append("- seed: 0")
    lines.append("")

    text = "\n".join(lines)
    path = os.path.join(ROOT, "ENVIRONMENT.md")
    with open(path, "w") as f:
        f.write(text)
    print(f"Wrote {path}")
    return text


if __name__ == "__main__":
    generate()
