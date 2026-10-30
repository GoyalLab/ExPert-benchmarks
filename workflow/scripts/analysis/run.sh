#!/usr/bin/env bash
# Run one or more analysis scripts, teeing all output to a watchable log.
#   ./analysis/run.sh axis1_dataset.py          # one
#   ./analysis/run.sh all                        # full pipeline
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-analysis/bin/python
LOG=analysis/out/analysis.log
mkdir -p analysis/out

scripts=("$@")
if [ "${1:-}" = "all" ]; then
  scripts=(load.py axis1_dataset.py axis2_perturbation.py axis3_efficiency.py synthesis.py)
fi

{
  echo "=================================================================="
  echo "RUN $(date '+%Y-%m-%d %H:%M:%S')  scripts: ${scripts[*]}"
  echo "=================================================================="
} | tee -a "$LOG"

for s in "${scripts[@]}"; do
  arg=""
  [ "$s" = "load.py" ] && arg="--selftest"
  echo "" | tee -a "$LOG"
  echo ">>> $s $arg" | tee -a "$LOG"
  "$PY" "analysis/$s" $arg 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! $s FAILED (rc=$rc)" | tee -a "$LOG"
    exit "$rc"
  fi
done
echo "" | tee -a "$LOG"
echo "ALL DONE $(date '+%H:%M:%S')" | tee -a "$LOG"
