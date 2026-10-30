# Baselines — one CPU job per (size, fold) computes ALL variants for the
# configured splits in a single run.py pass (internally chunked). Each variant
# lands in its own baseline_<v>/output/… tree, which collect_results scores as a
# separate method. Uses the shared `expert` env (no torch/GPU).

BASELINE_VARIANTS = config["baseline_variants"]


rule baselines:
    input:
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        idtest = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_test.txt.gz"),
        ood    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "ood_test.txt.gz"),
        meta   = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl   = config["ctrl_h5ad"],
    output:
        expand(
            RESULTS + "/baseline_{variant}/output/{{size}}/{{fold}}/{split}/logits.h5",
            variant=BASELINE_VARIANTS,
            split=config["method_splits"]["baselines"],
        ),
    params:
        splits   = SPLITS_ROOT,
        out_root = RESULTS,
        variants = " ".join(BASELINE_VARIANTS),
    threads: res_threads("baselines")
    resources:
        time      = res_time("baselines"),
        mem       = res_mem("baselines"),
        partition = res_partition("baselines"),
        slurm     = res_slurm("baselines"),
    conda:
        "../envs/baselines.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python workflow/methods/baselines/run.py \
            --splits-dir {params.splits} \
            --ctrl-h5ad {input.ctrl} \
            --size {wildcards.size} \
            --fold {wildcards.fold} \
            --out-root {params.out_root} \
            --baselines {params.variants} \
            --label-col perturbation \
            --context-col dataset
        """
