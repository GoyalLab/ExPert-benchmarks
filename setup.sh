#!/bin/bash
#SBATCH --job-name expbench-build-envs
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 4:00:00
#SBATCH --output logs/build_envs_%j.out
#
# Account/partition: pass on the command line or via SBATCH_ACCOUNT / SBATCH_PARTITION.
#
# Pre-build every per-rule conda env (the workflow/envs/*.yaml referenced by the
# rules' `conda:` directives) into .conda-envs/. Runs as a SLURM job because the
# login nodes OOM at 6GB on conda/mamba solves + the torch install. Submit from
# the repo root:
#
#   mamba activate ExPert        # needs snakemake 7.x + mamba on PATH
#   sbatch setup.sh
#
# PIP_USER=0 + PYTHONNOUSERSITE=1 keep the torch pip install INSIDE the env (a
# global ~/.config/pip/pip.conf `user=true` otherwise leaks it to ~/.local).
# CONDA_PKGS_DIRS points at a writable in-repo cache (the cluster's default
# first pkgs dir is read-only).


# SLURM_SUBMIT_DIR is the dir you ran `sbatch` from (the repo root); fall back to
# cwd for an interactive `bash setup.sh` inside an allocation.
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"

source ~/.bashrc
if command -v module >/dev/null 2>&1; then module load mamba 2>/dev/null || true; fi
mamba activate "${LAUNCHER_ENV:-${REPO}/.conda-envs/expert-bench}"

set -eo pipefail
export CONDA_PKGS_DIRS="${REPO}/.conda-envs/pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"
export PIP_USER=0
export PYTHONNOUSERSITE=1

CONFIGFILES=(config/config.yaml); [ -f config/site.yaml ] && CONFIGFILES+=(config/site.yaml)
snakemake -s workflow/Snakefile --configfile "${CONFIGFILES[@]}" \
    --use-conda --conda-frontend mamba --conda-prefix .conda-envs \
    --conda-create-envs-only --cores 1

echo "All conda envs built under ${REPO}/.conda-envs/."