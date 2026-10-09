#!/usr/bin/env bash
# Waits for the machine to go quiet, then runs the full DECAF suite.
#
# A latency study cannot be measured under CPU contention: on this 4-core
# machine a load average of ~13 inflated prefill from ~4 ms/token to ~85
# ms/token, which both invalidates the numbers and makes the full grid
# infeasible in the available time.
#
# The gate is set BELOW the in-code preflight threshold so that opening the
# gate guarantees preflight will pass (an earlier version gated at 2.5 against
# a preflight of 2.0 and launched straight into a refusal).
#
# If the machine never goes quiet within FALLBACK_AFTER_MIN, the suite runs
# anyway with contention logged per run, so a limited time budget still yields
# data -- clearly marked as contention-affected rather than silently wrong.
set -uo pipefail
cd "$(dirname "$0")/.."

MAX_LOAD="${MAX_LOAD:-1.8}"            # must stay < preflight.max_load1 (2.0)
MIN_MEM_GB="${MIN_MEM_GB:-3.0}"
FALLBACK_AFTER_MIN="${FALLBACK_AFTER_MIN:-45}"
LOG="${LOG:-results/run_full.log}"
mkdir -p results

echo "[$(date +%H:%M:%S)] waiting for load < $MAX_LOAD and free RAM > ${MIN_MEM_GB}GB"
echo "[$(date +%H:%M:%S)] fallback: run with contention logged after ${FALLBACK_AFTER_MIN}m"
deadline=$(( $(date +%s) + FALLBACK_AFTER_MIN * 60 ))
MODE="clean"
while :; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  mem=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  if awk "BEGIN{exit !($load < $MAX_LOAD && $mem > $MIN_MEM_GB)}"; then
    echo "[$(date +%H:%M:%S)] QUIET: load=$load mem=${mem}GB -> launching (clean)"
    break
  fi
  if [[ $(date +%s) -ge $deadline ]]; then
    echo "[$(date +%H:%M:%S)] FALLBACK: still load=$load mem=${mem}GB after ${FALLBACK_AFTER_MIN}m."
    echo "[$(date +%H:%M:%S)] Launching with --allow-contention. ABSOLUTE LATENCIES WILL BE INFLATED;"
    echo "[$(date +%H:%M:%S)] per-run load is recorded in every record and plotted in fig19."
    MODE="contended"; break
  fi
  sleep 20
done

[[ "$MODE" == "contended" ]] && export ALLOW_CONTENTION=1
exec bash scripts/run_full.sh A B C D 2>&1 | tee -a "$LOG"
