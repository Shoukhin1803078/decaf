#!/usr/bin/env bash
# DECAF v3 — end-to-end cold re-run.
#
# Usage:  bash scripts/run_v3.sh [pilot|main|all]
#   pilot : scaling (3B/Q4) + 20-question pilot, then the audit gate
#   main  : full scaling + main cold grid + warm-order pass
#   all   : pilot then main
#
# Writes only to cache_v3/ and results_v3/. v2 artifacts are never touched.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CFG=config_v3.yaml
MODE="${1:-all}"

run_scaling() {
  echo "=== scaling calibration ==="
  $PY -m src.run_v3 --config $CFG --mode scaling
}

run_pilot() {
  echo "=== 20-question pilot (3B/Q4, hotpot, both conditions) ==="
  $PY -m src.run_v3 --config $CFG --mode compression \
      --models "qwen2.5:3b-instruct" --datasets hotpot --limit 20 --condition cold
  $PY -m src.run_v3 --config $CFG --mode compression \
      --models "qwen2.5:3b-instruct" --datasets hotpot --limit 20 --condition warm_order
}

run_main() {
  echo "=== main cold grid (2 configs x 4 datasets x 100 q x cold methods) ==="
  $PY -m src.run_v3 --config $CFG --mode compression \
      --models "qwen2.5:3b-instruct,qwen2.5:7b-instruct-q4_K_M" --condition cold
  echo "=== warm-order pass (inflation measurement) ==="
  $PY -m src.run_v3 --config $CFG --mode compression \
      --models "qwen2.5:3b-instruct,qwen2.5:7b-instruct-q4_K_M" --condition warm_order
}

gate() {
  echo "=== audit gate ==="
  $PY -I audit_gate.py --scaling cache_v3/scaling.jsonl \
     --compression cache_v3/compression.jsonl || true
}

case "$MODE" in
  pilot) run_scaling; run_pilot; gate ;;
  main)  run_scaling; run_main; gate ;;
  all)   run_scaling; run_pilot; gate; run_main; gate ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac

echo "=== analysis ==="
$PY -m src.analyze_v3 --config $CFG
echo "Done. See results_v3/FULL_RESULTS_v3.md"
