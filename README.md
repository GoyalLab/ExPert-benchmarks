# ExPert-benchmarks

A Snakemake pipeline that benchmarks **ExPert** against single-cell perturbation-prediction
methods on the reverse task: given one perturbed cell's expression profile, predict which
perturbation it received. Every method is adapted to emit the same artifact, a per-cell ×
per-perturbation score matrix (`logits.h5`), and one collector scores them all on identical
cells. The benchmark data are built by [ExPert-data](https://github.com/GoyalLab/ExPert-data);
the ExPert model itself lives in [ExPert](https://github.com/GoyalLab/ExPert) and is only
scored here.

The full methods description (data, folds, adaptations, metrics, cost accounting) is in
[METHODS.md](METHODS.md).

## Methods

| method | type | training | compute |
|---|---|---|---|
| **ExPert** | contrastive perturbation model (scored, trained elsewhere) | one model per fold on the full perturbation pool | CPU (eval) |
| **fingerprint (ensemble / merged)** | RNA fingerprinting (Seurat + susie) | per dataset | CPU |
| **cellot** | neural optimal transport in an autoencoder latent | per dataset | CPU |
| **GEARS** | graph neural network | per dataset | GPU |
| **scouter** | perturbation-response network with gene embeddings | per dataset | GPU |
| **GenePert** | ridge regression on gene embeddings | per dataset | CPU |
| **scGPT** | pretrained transformer, fine-tuned, classifier readout | one model per fold | GPU |
| **XGBoost / linreg** | supervised cell → perturbation classifiers | one model per fold | CPU |
| **baselines** | random, prior, neg_target, centroid, centroid_notarget | — | CPU |

Methods that are single-dataset by design are trained per dataset and evaluated
dataset-blind: every test cell is scored against every dataset's model.

## Requirements

- A SLURM cluster (the pipeline submits one job per rule; a full `ctx7` pass is ~24k jobs).
- `mamba`/`conda`. Per-method environments are built by Snakemake from `workflow/envs/`.
- The benchmark store from ExPert-data: perturbed + control h5ad files and the
  `splits/<size>/` directories (`metadata.json` + `fold_*/{id_train,id_val,id_test,ood_test}.txt.gz`).
- For ExPert scoring: trained fold models and an environment with
  [ExPert](https://github.com/GoyalLab/ExPert) installed.
- Method resources: the scGPT `scGPT_human` checkpoint, a gene-embedding `.npz` for GenePert,
  and a GenePT `{gene: vector}` pickle for scouter.

## Setup

```bash
# 1. Launcher environment (snakemake 7 + mamba), as a prefix env inside the repo.
mamba env create -f environment.yaml -p .conda-envs/expert-bench
mamba activate .conda-envs/expert-bench

# 2. Site configuration: data paths, output roots and the SLURM partition map.
cp config/site.example.yaml config/site.yaml     # git-ignored; edit every /path/to/...

# 3. SLURM defaults for the helper scripts (sbatch reads these variables).
export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>

# 4. Build every per-method conda env (a compute-node job; solves can exceed 6 GB).
sbatch setup.sh

# 5. Stage the scGPT checkpoint into config scgpt_pretrained_model.
SCGPT_CKPT_SRC=/path/to/scGPT_human bash workflow/methods/scgpt/.setup.sh
```

`config/resources.yaml` requests resources by **partition class** (`cpu`, `cpu_himem`,
`gpu`, …), never by a real partition name. `config/site.yaml` maps each class to your
cluster's partition, account and optional node constraint. A class you leave unmapped is
used verbatim as the partition name.

## Running

```bash
# Dry run: show the job plan for one size.
snakemake -n metrics/small --profile profiles/slurm --configfile config/config.yaml config/site.yaml

# Full run, as a long-lived controller job (use a multi-day partition):
SBATCH_PARTITION=<long-partition> sbatch run.sh metrics/small metrics/ctx7
```

`run.sh` layers `config/site.yaml` automatically, prints the dry-run plan, then runs
Snakemake with the SLURM profile. Re-submitting resumes from finished outputs. Useful
variables: `JOBS` (concurrent jobs, default 400), `KEEP_GOING=1` (continue past failures),
`SUBSET=config/model_subset.yaml` (an extra config overlay).

Put explicit targets **before** `--configfile`, which takes several values and would
otherwise swallow them.

## Outputs

```
<results_root>/<method>/.../<size>/<fold>/<split>/logits.h5   # per-method scores
metrics/<size>/
  weighted_metrics.csv           # efficiency-weighted metrics per method x split x fold
  sweep_metrics*.csv             # metrics at efficiency thresholds none/1/2/3
  {none,e:1,e:2,e:3}/            # per-threshold tables, per-class metrics, plots
  plots/  analysis/out/          # figures, per-axis analyses, REPORT.md, COST.md
```

Add-on scorers read a finished collect without re-running it (submit from the repo root):

- `utils/score_extra.sbatch`: macro one-vs-rest AUROC, class-balanced weighted F1 and MRR.
- `utils/score_selective.sbatch`: a per-cell rank/confidence table for matched-cell and coverage analyses.
- `utils/score_pbulk_groups.sbatch`: pseudobulk (group-level) ranks.

`utils/slurm_costs.py` books compute cost from `sacct`. `utils/plot_scaling.py` compares
two benchmark sizes.

## Layout

```
config/
  config.yaml          # benchmark grid, methods, hyper-parameters, default paths
  site.example.yaml    # template for config/site.yaml (paths + partition map)
  resources.yaml       # per-rule time/mem/threads/partition class/gpu
  cost_overrides.yaml  # measured replacements for sacct-booked costs
profiles/slurm/        # Snakemake 7 SLURM profile
workflow/
  Snakefile
  rules/               # one .smk per method + common helpers + collect
  envs/                # per-method conda specs
  methods/             # per-method adapters and vendored third-party code
  scripts/             # collect_results.py, add-on scorers, analysis/
utils/                 # cost accounting, scaling plots, scGPT target padding, wrappers
```

## Third-party code

`workflow/methods/` vendors code from other projects under their own licenses:

| directory | project | license |
|---|---|---|
| `cellot/` | [cellot](https://github.com/bunnech/cellot) (Bunne, Stark et al.) | BSD 3-Clause, `cellot/LICENSE` |
| `gears/GEARS/` | [GEARS](https://github.com/snap-stanford/GEARS) (Roohani et al.) | MIT, `gears/GEARS/LICENSE` |
| `scgpt/scGPT/` | [scGPT](https://github.com/bowang-lab/scGPT) (Cui et al.) | MIT, `scgpt/scGPT/LICENSE` |

RNA fingerprinting and scouter are installed from their published packages; the
directories here hold only the benchmark adapters.

## License

MIT (see [LICENSE](LICENSE)); vendored third-party code keeps its own license.
