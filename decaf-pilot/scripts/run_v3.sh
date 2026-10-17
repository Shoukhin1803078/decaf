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

run_extension() {
  echo "=== Q8 extension (3B/Q8 + 7B/Q8, n=50, cold) ==="
  $PY -m src.run_v3 --config $CFG --mode compression \
      --models "qwen2.5:3b-instruct-q8_0,qwen2.5:7b-instruct-q8_0" --limit 50 --condition cold
}

run_ablation() {
  echo "=== ablation arm (decaf variants cold, hotpot+2wiki+trivia+longbench) ==="
  $PY -m src.run_v3 --config $CFG --mode compression \
      --models "qwen2.5:3b-instruct,qwen2.5:7b-instruct-q4_K_M" --limit 50 --condition cold \
      --methods "decaf_nocov_lam6,decaf_noreorder_lam6,decaf_lam2,decaf_lam15" \
      --cache cache_v3/compression_ablation.jsonl
}

run_topk() {
  echo "=== top_k sweep (hotpot + longbench, k in {5,10,20}) ==="
  for K in 10 20; do
    $PY -m src.run_v3 --config $CFG --mode compression \
        --models "qwen2.5:3b-instruct,qwen2.5:7b-instruct-q4_K_M" --datasets "hotpot,longbench" \
        --limit 50 --condition cold --top-k $K --cache cache_v3/compression_topk.jsonl
  done
}

gate() {
  echo "=== audit gate ==="
  $PY -I audit_gate.py --scaling cache_v3/scaling.jsonl \
     --compression cache_v3/compression.jsonl || true
}

case "$MODE" in
  pilot)     run_scaling; run_pilot; gate ;;
  main)      run_scaling; run_main; gate ;;
  extension) run_extension ;;
  ablation)  run_ablation ;;
  topk)      run_topk ;;
  all)       run_scaling; run_pilot; gate; run_main; gate; run_extension; run_ablation ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac

echo "=== analysis ==="
$PY -m src.analyze_v3 --config $CFG
echo "Done. See results_v3/FULL_RESULTS_v3.md"
