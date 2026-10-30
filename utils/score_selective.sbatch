#!/bin/bash
#SBATCH --job-name score-selective
#SBATCH --nodes 1 --cpus-per-task 2 --mem 7G --time 01:30:00
#SBATCH --array 0-19
#SBATCH --output logs/score_selective_%A_%a.out
#SBATCH --error  logs/score_selective_%A_%a.err
# Per-cell rank / top-1 / confidence / fingerprint-abstention table for a FINISHED collect, for the
# matched-cell and coverage-accuracy analyses. ONE ARRAY TASK PER METHOD (20 >= the 19 method dirs;
# a task with no method exits at once) -> <metrics>/selective/<method>.parquet (resumable
# per method; nothing else is rewritten). mem: measured 5.2GB peak (smoke) -> 7G.
#   SIZE=ctx7 sbatch utils/score_selective.sbatch
#   SIZE=ctx7 SMOKE=1 sbatch ...   (two methods, 2 files each, into selective_smoke/)
set -euo pipefail
# Submit from the repo root (account/partition via SBATCH_ACCOUNT / SBATCH_PARTITION).
# PYTHON: an interpreter with the collect env's packages (workflow/envs/collect.yaml),
# e.g. the env Snakemake built under .conda-envs/.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${PYTHON:-python}"
SIZE="${SIZE:?set SIZE}"
SOURCE="$("$PYTHON" "$REPO/utils/config_value.py" results_root)"
SPLITS="$("$PYTHON" "$REPO/utils/config_value.py" splits_root)"
OUT="$REPO/metrics/$SIZE/selective"
EXTRA=()
if [ -n "${SMOKE:-}" ]; then OUT="${OUT}_smoke"; EXTRA=(--methods ExPert fingerprint_perdataset --max-files 2); fi
cd "$REPO"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"$PYTHON" workflow/scripts/score_selective.py \
    --source "$SOURCE" --splits-dir "$SPLITS" --dataset-size "$SIZE" \
    --method-depth 1 --splits id_test ood_test --out-dir "$OUT" \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks "${SLURM_ARRAY_TASK_COUNT:-1}" "${EXTRA[@]}"
