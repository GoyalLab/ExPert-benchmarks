# Scouter — neural perturbation model, PER-DATASET by design (single-dataset:
# predicts per-pert mean expression from controls). Converted to per-dataset
# training + dataset-blind ensemble eval, like fingerprint/genepert/cellot/gears
# (scouter's predict μ + distance IS the genepert pattern, so the ensemble reuses
# genepert's combiner). Per-dataset also RELIEVES the global-merge OOM (the parked
# 892GB peak): each job loads only D's own cells, not all 12M.
#
#   scouter_train_pds(size, fold, dataset)  GPU — train D's model on D's cells +
#                                           OWN panel, save D's per-pert μ
#                                           fingerprints.npz.
#   scouter_ens_eval(size, fold, split)     CPU — score every cell vs every
#                                           dataset's fingerprints (own-control
#                                           Pearson delta + softmax + covering-avg
#                                           + z-score) → logits.h5.
#
# Replaces the global shared-genes scouter in place (same logits path — one version).

SCOUTER_CFG  = "workflow/methods/scouter/configs/tasks/perturb-seq/{size}.yaml"
SCOUTER_FOLD = f"{RESULTS}/scouter/perturb-seq/{{size}}/{{fold}}"
SCOUTER_ROOT = f"{RESULTS}/scouter/perturb-seq"


rule scouter_train_pds:
    """Train dataset D's Scouter model on its OWN cells + panel; save D's per-pert
    predicted fingerprints for the ensemble. GPU."""
    input:
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        val    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_val.txt.gz"),
        config = SCOUTER_CFG,
    output:
        fp = SCOUTER_FOLD + "/per_dataset/{dataset}/fingerprints.npz",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
    params:
        gene_panel = gene_panel(),
        splits = lambda wc: splits_dir(wc.size, wc.fold),
        outdir = SCOUTER_FOLD + "/per_dataset/{dataset}",
    threads: lambda wc: res_threads("scouter_train_pds", wc.size)
    resources:
        time      = lambda wc: res_time("scouter_train_pds", wc.size),
        mem       = lambda wc: res_mem("scouter_train_pds", wc.size),
        partition = lambda wc: res_partition("scouter_train_pds", wc.size),
        slurm     = lambda wc: res_slurm("scouter_train_pds", wc.size),
    conda:
        "../envs/scouter.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python workflow/methods/scouter/scripts/train.py \
            --splits_dir {params.splits} \
            --config {input.config} \
            --outdir {params.outdir} \
            --dataset {wildcards.dataset} \
            --gene-panel {params.gene_panel} \
            --ctrl-h5ad {config[ctrl_h5ad]} \
            --genept-path {config[scouter_gene_embeddings]}
        """


def scouter_ens_fps(wc):
    """Every training dataset's fingerprints.npz for this fold (ensemble fan-in)."""
    return [f"{SCOUTER_ROOT}/{wc.size}/{wc.fold}/per_dataset/{d}/fingerprints.npz"
            for d in training_datasets(wc.size, wc.fold)]


rule scouter_ens_eval:
    """Dataset-blind ensemble: score one split's cells against every dataset's
    scouter fingerprints (genepert combiner) → logits.h5. CPU."""
    input:
        fps    = scouter_ens_fps,
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
        idx    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), f"{wc.split}.txt.gz"),
    output:
        logits = SCOUTER_FOLD + "/{split}/logits.h5",
    wildcard_constraints:
        split = r"id_test|ood_test",
    params:
        srcdir = "workflow/methods/scouter/scripts", froot = SCOUTER_FOLD,
        splits = lambda wc: splits_dir(wc.size, wc.fold),
    threads: lambda wc: res_threads("scouter_ens_eval", wc.size)
    resources:
        time      = lambda wc: res_time("scouter_ens_eval", wc.size),
        mem       = lambda wc: res_mem("scouter_ens_eval", wc.size),
        partition = lambda wc: res_partition("scouter_ens_eval", wc.size),
        slurm     = lambda wc: res_slurm("scouter_ens_eval", wc.size),
    conda:
        "../envs/scouter.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/scouter_ensemble.py \
            --fold_root {params.froot} --source {input.source} --ctrl {input.ctrl} \
            --splits_dir {params.splits} --split {wildcards.split} --out {output.logits}
        """
