# Collect — score every method's logits.h5 for a size into one metrics dir.
# Wraps the vendored collect_results.py (same args as the legacy
# src/collect_results_<size>.sh). Because its inputs are the explicit per-fold
# logits.h5 targets, a missing/failed leaf blocks this rule loudly instead of
# silently producing partial metrics.

rule collect:
    input:
        logits = lambda wc: logits_for_size(wc.size),
        classes = lambda wc: ([collect_classes(wc.size)] if collect_classes(wc.size) else []),
    output:
        metrics = directory(metrics_dir("{size}")),
    params:
        # size-aware: padding writes to its own results root (padding.results_root),
        # so `collect` must scan that tree, not the main one.
        source     = lambda wc: results_root_for(wc.size),
        # size-aware: the padding mode's split space is pipeline-generated and lives
        # outside the upstream store (which stays read-only).
        splits_dir = lambda wc: splits_root_for(wc.size),
        # perturbed source adata, resolved from <size>/metadata.json:source_path
        # (the rebuilt DB keeps no source.h5ad inside the split dir).
        source_adata = lambda wc: source_h5ad(wc.size),
        # a param (not a constant) so flipping it in config re-triggers collect
        recall_mode = config.get("recall_mode", "balanced"),
        splits      = lambda wc: " ".join(collect_splits(wc.size)),
        # Non-empty ONLY for the padding mode -> a FIXED class space instead of the
        # cross-method union (see collect_classes).
        classes     = lambda wc: collect_classes(wc.size),
        classes_arg = lambda wc: (f"--classes {collect_classes(wc.size)}"
                                  if collect_classes(wc.size) else ""),
        report_arg  = lambda wc: collect_report_args(wc.size),
    threads: lambda wc: res_threads("collect", wc.size)
    resources:
        time      = lambda wc: res_time("collect", wc.size),
        mem       = lambda wc: res_mem("collect", wc.size),
        partition = lambda wc: res_partition("collect", wc.size),
        slurm     = lambda wc: res_slurm("collect", wc.size),
    conda:
        "../envs/collect.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python workflow/scripts/collect_results.py \
            --source {params.source} \
            --splits-dir {params.splits_dir} \
            --dataset-size {wildcards.size} \
            --method-depth 1 \
            --output-dir {output.metrics} \
            --splits {params.splits} {params.classes_arg} {params.report_arg} \
            --recall-mode {params.recall_mode} \
            --efficiency-sweep none 1 2 3

        # Per-cell covariate table the analysis reads (dataset/perturbation/
        # efficiency) — not part of collect_results.py, so emit it here.
        python workflow/scripts/dump_source_obs.py \
            --source {params.source_adata} \
            --out {output.metrics}/data/source.obs.csv

        # Rich 3-axis analysis suite (dataset / perturbation / efficiency) on the
        # freshly-collected metrics -> <metrics>/analysis/out/ as png + svg + csv,
        # plus REPORT.md. Env points the vendored loader at this metrics dir.
        # (No braces in this comment: Snakemake format-expands the whole shell body.)
        # Per-script guard so one axis hiccup doesn't wipe the metrics/other plots.
        REPO="$PWD"
        export EXPERT_METRICS_DIR="$REPO/{output.metrics}"
        export EXPERT_ANALYSIS_OUT="$REPO/{output.metrics}/analysis/out"
        export EXPERT_RECALL_MODE="{params.recall_mode}"
        export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/mpl-$USER"; mkdir -p "$MPLCONFIGDIR"
        cd "$REPO/workflow/scripts/analysis"
        for s in load.py axis1_dataset.py axis2_perturbation.py axis3_efficiency.py cost.py synthesis.py; do
            arg=""; [ "$s" = load.py ] && arg="--selftest"
            echo ">>> analysis/$s $arg"
            python "$s" $arg || echo "!!! analysis/$s failed (non-fatal — metrics still valid)"
        done
        """
