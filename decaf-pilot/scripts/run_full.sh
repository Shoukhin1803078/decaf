#!/usr/bin/env bash
# DECAF full experiment suite, ordered so an early stop loses only the least
# important work. Every phase is resumable: re-running skips cached runs.
#
#   bash scripts/run_full.sh              # all phases
#   bash scripts/run_full.sh A B          # selected phases
#   ALLOW_CONTENTION=1 bash scripts/...   # measure on a busy machine (logged)
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/tasnia/decaf/.venv/bin/python}"
EXTRA=""
[[ "${ALLOW_CONTENTION:-0}" == "1" ]] && EXTRA="--allow-contention"
PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && PHASES=(A B C D)

log() { printf '\n\033[1m== %s ==\033[0m  (%s)\n' "$1" "$(date +%H:%M:%S)"; }
t0=$(date +%s)

run_phase() {
  case "$1" in
    A) log "Phase A — E1/E8 decode scaling (context x generation length)"
       $PY -m src.run_experiments --mode scaling $EXTRA ;;
    B) log "Phase B — E2/E3/E4/E7 + Sec.17 compression grid (Q4_K_M)"
       $PY -m src.run_experiments --mode compression $EXTRA ;;
    C) log "Phase C — E5 quantization: Q8_0 scaling + compression subset"
       $PY -m src.run_experiments --mode scaling     --secondary $EXTRA
       $PY -m src.run_experiments --mode compression --secondary $EXTRA ;;
    D) log "Phase D — H5 KV-reuse / prefetch micro-study"
       $PY -m src.run_experiments --mode kv $EXTRA ;;
    *) echo "unknown phase: $1" >&2; return 1 ;;
  esac
}

for p in "${PHASES[@]}"; do
  if ! run_phase "$p"; then
    echo "!! phase $p failed or was interrupted — analysing what exists so far" >&2
    break
  fi
  log "Analysis after phase $p"
  $PY -m src.analyze --no-stream || echo "!! analysis failed after phase $p" >&2
done

log "Final analysis (with STREAM bandwidth probe)"
$PY -m src.analyze

printf '\nTotal wall clock: %d min\n' $(( ($(date +%s) - t0) / 60 ))
echo "Results: results/RESULTS.md  +  results/fig*.png"
