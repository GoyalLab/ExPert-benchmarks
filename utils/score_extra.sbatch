#!/bin/bash
#SBATCH --job-name score-extra
#SBATCH --nodes 1 --cpus-per-task 2 --mem 12G --time 04:00:00
#SBATCH --output logs/score_extra_%j.out
#SBATCH --error  logs/score_extra_%j.err
# OvR AUROC (+ weighted) and per-perturbation weighted MRR for a FINISHED collect,
# appended to <metrics>/extra_metrics.csv (resumable; nothing else is rewritten).
# mem: collect streams one file at a time, measured peak 7.8GiB -> 12G.
#   SIZE=ctx7 sbatch utils/score_extra.sbatch
#   SIZE=ctx7 SMOKE=1 sbatch ...   (smoke: writes extra_metrics.smoke.csv, 1 method)
set -euo pipefail
# Submit from the repo root (account/partition via SBATCH_ACCOUNT / SBATCH_PARTITION).
# PYTHON: an interpreter with the collect env's packages (workflow/envs/collect.yaml),
# e.g. the env Snakemake built under .conda-envs/.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${PYTHON:-python}"
SIZE="${SIZE:?set SIZE}"
SOURCE="$("$PYTHON" "$REPO/utils/config_value.py" results_root)"
SPLITS="$("$PYTHON" "$REPO/utils/config_value.py" splits_root)"
OUT="${OUT:-$REPO/metrics/$SIZE/extra_metrics.csv}"
[ -n "${SMOKE:-}" ] && OUT="${OUT%.csv}.smoke.csv"
cd "$REPO"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"$PYTHON" workflow/scripts/score_extra.py \
    --source "$SOURCE" --splits-dir "$SPLITS" --dataset-size "$SIZE" \
    --method-depth 1 --splits id_test ood_test --out "$OUT" ${SMOKE:+--methods ExPert baseline_random --max-files 3}
