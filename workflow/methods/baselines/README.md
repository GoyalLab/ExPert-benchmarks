# Reverse-direction baselines

Generates cheap baselines that `collect_results.py` picks up alongside the
real methods (`baseline_random`, `baseline_prior`, `baseline_neg_target`,
`baseline_centroid`).

| name        | what it predicts                                                |
|-------------|-----------------------------------------------------------------|
| random      | uniform logits over training perturbations                      |
| prior       | log P(pert) from `id_train` class frequencies (broadcast)       |
| neg_target  | `ctrl_mean[dataset(c), g] − X[c, g]` for perts whose name is in `var_names` (others `-inf`) |
| centroid    | cosine sim between cell delta vs. control and per-pert delta centroid |

`neg_target` and `centroid` use per-dataset control means from
`data/source/ctrl.h5ad` for centring; both expression matrices are
`normalize_total(1e4) + log1p`-normalised on the fly.

## Run

One (size, fold) at a time:

```
python benchmarks/context/baselines/run.py --size small --fold fold_0
```

Per-size slurm arrays:

```
sbatch benchmarks/context/baselines/run_small.sh   # small × 3 folds
sbatch benchmarks/context/baselines/run_full.sh    # full  × 3 folds
```

Each baseline writes `logits.h5` (`/logits`, `/columns`, `/idx`) + `labels.csv`
to `benchmarks/context/reverse/baseline_<name>/output/<size>/fold_N/<split>/`.
If `output/` is a symlink (e.g. to scratch storage), writes follow it transparently.

## Discovery

With `--source benchmarks/context --method-depth 2`, the collector treats
`reverse/baseline_<name>` as method `baseline_<name>` and writes its metrics
into the same all_metrics.csv as the real methods.
