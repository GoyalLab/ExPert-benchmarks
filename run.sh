#!/bin/bash
#SBATCH --job-name expbench-run
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 24G
#SBATCH --time 10-00:00:00
#SBATCH --output logs/run_%j.out
#
# Account / partition / mail are site settings: pass them on the command line or via
# sbatch's input environment variables, e.g.
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<long-walltime partition>
#   sbatch run.sh metrics/small
# (the controller must outlive the whole DAG, so use a multi-day partition).
#
# Launch the benchmark pipeline. This is a long-lived *controller* job: it runs
# Snakemake locally (local-cores: 1) and submits each rule to SLURM through
# profiles/slurm — GPU partitions for scgpt/gears/scouter training, CPU for the
# rest — then polls them to completion. Per-rule resources come from
# config/resources.yaml (partition classes mapped by config/site.yaml); concurrency is capped by the profile
# (jobs: 30). Child-job logs land under logs/slurm/<rule>/.
#
# The controller must outlive the whole DAG, so submit it to a multi-day
# partition. If it ever hits the wall before `collect` finishes, just re-submit —
# Snakemake resumes from the outputs already on disk (only missing/incomplete
# targets rerun).
#
# Envs and the scgpt checkpoint must already be in place — this script does NOT
# build envs (run `sbatch setup.sh` first).
#
# Usage from the repo root:
#   sbatch run.sh                      # default target: metrics/small
#   sbatch run.sh metrics/small        # explicit target
#   sbatch run.sh <path/to/logits.csv> # any single Snakemake target
#   sbatch run.sh metrics/ctx7 metrics/zs_f4   # several targets, one controller
#
# KEEP_GOING (default 0 = FAIL FAST). Snakemake's --keep-going carries on with every
# independent job after a failure, which on this DAG means a systematic breakage
# (e.g. a method's env or a bad path) burns queue time on all ~230 sibling jobs
# before anyone looks. Failing fast costs nothing here: the controller stops
# SUBMITTING but lets running jobs finish, and a re-submit resumes from whatever is
# already on disk. Set KEEP_GOING=1 only when you want a long unattended run to
# harvest everything it can despite one known-bad leaf.

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"

# Site overlay (paths, partition map): config/site.yaml, git-ignored -- see
# config/site.example.yaml. Layered on automatically when present.
# Targets: all args. Empty => Snakemake's `rule all` (metrics/<size> for every
# size in the active config) — so with SUBSET the config's sizes/methods drive it.
TARGETS=("${@}")
# SUBSET: optional extra --configfile layered over config.yaml, e.g.
#   SUBSET=config/model_subset.yaml sbatch run.sh   # full run of a method subset
CONFIGFILES=(config/config.yaml)
[ -f config/site.yaml ] && CONFIGFILES+=(config/site.yaml)
[ -n "${SUBSET:-}" ] && CONFIGFILES+=("${SUBSET}")
# Concurrency: overrides the profile's jobs:30 so the fingerprint eval fan-out
# (thousands of small CPU chunks) parallelizes hard. 400 is well under the 5000
# per-user SLURM cap on a typical cluster. Raise if you have the nodes; lower if
# you also submit many GPU rules.
JOBS="${JOBS:-400}"

# Launcher env: snakemake 7.x + mamba (environment.yaml). LAUNCHER_ENV may be an env
# name or a prefix path. `module load mamba` only where an environment-modules
# system provides it.
source ~/.bashrc
if command -v module >/dev/null 2>&1; then module load mamba 2>/dev/null || true; fi
mamba activate "${LAUNCHER_ENV:-${REPO}/.conda-envs/expert-bench}"

set -eo pipefail
export PIP_USER=0
export PYTHONNOUSERSITE=1

echo "=== plan for [${TARGETS[*]:-rule all}] (config: ${CONFIGFILES[*]}) — dry-run ==="
snakemake "${TARGETS[@]}" --profile profiles/slurm --configfile "${CONFIGFILES[@]}" -n 2>&1 \
    | sed -n '/^Job stats:/,/^$/p' || true

echo "=== launching [${TARGETS[*]:-rule all}] at $(date) with --jobs ${JOBS} ==="
KEEP_GOING_FLAG=()
if [ "${KEEP_GOING:-0}" = "1" ]; then
  KEEP_GOING_FLAG=(--keep-going)
  echo "=== KEEP_GOING=1: continuing past failures (independent jobs keep running) ==="
else
  echo "=== fail-fast (KEEP_GOING=0, default): a failure stops further submission;"
  echo "===   running jobs finish, then fix and re-submit to resume ==="
fi

snakemake "${TARGETS[@]}" \
    --profile profiles/slurm \
    --configfile "${CONFIGFILES[@]}" \
    --jobs "${JOBS}" \
    "${KEEP_GOING_FLAG[@]}" \
    --rerun-incomplete

echo "=== pipeline controller finished at $(date) ==="
