#!/usr/bin/env bash
# End-to-end DECAF v2: full experiments (all models x datasets), then analysis.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "== Phase A (v2): experiments =="
$PY -m src.run_v2 --config config_v2.yaml "$@"

echo "== Phase B (v2): analysis =="
$PY -m src.analyze_v2 --config config_v2.yaml

echo "Done. See results_v2/FULL_RESULTS_v2.md"
