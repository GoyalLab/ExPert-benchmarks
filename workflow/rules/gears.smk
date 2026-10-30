# GEARS — GNN perturbation model, PER-DATASET by design (single-dataset method:
# one GNN over a dataset's own gene graph). Converted to per-dataset training +
# dataset-blind ensemble eval, like fingerprint/genepert/cellot:
#
#   gears_prep_pds(size, dataset)        CPU — build D's cell_graphs on D's OWN
#                                        panel (fold-independent, cached per D).
#   gears_train_pds(size, fold, dataset) GPU — train D's GNN on D's train cells,
#                                        save D's per-pert fingerprints.npz.
#   gears_ens_eval(size, fold, split)    CPU — score every cell vs every dataset's
#                                        fingerprints (own-control Pearson delta +
#                                        softmax + covering-avg + z-score, the
#                                        genepert combiner) → logits.h5.
#
# Replaces the global shared-genes GEARS in place (same logits path — one version).
# Each dataset gets its OWN PertData dir (D's cell_graphs), so datasets never race.

GEARS_SRC      = os.path.join(REPO, "workflow", "methods", "gears")
GEARS_PDS_PD   = f"{RESULTS}/gears/perturb-seq/{{size}}/_pertdata_pds/{{dataset}}"   # per-dataset PertData
GEARS_PDS_CG   = GEARS_PDS_PD + "/{size}/data_pyg/cell_graphs.pkl"
GEARS_FOLD     = f"{RESULTS}/gears/perturb-seq/{{size}}/{{fold}}"
GEARS_ROOT     = f"{RESULTS}/gears/perturb-seq"


rule gears_prep_pds:
    """Build dataset D's per-cell PyG graph cache on D's OWN panel (CPU, per size+D,
    fold-independent)."""
    input:
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
    output:
        cellgraphs = GEARS_PDS_CG,
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
    params:
        gene_panel = gene_panel(),
        srcdir = GEARS_SRC, pertdata = GEARS_PDS_PD,
    threads: lambda wc: res_threads("gears_prep", wc.size)
    resources:
        time = lambda wc: res_time("gears_prep", wc.size),
        mem = lambda wc: res_mem("gears_prep", wc.size),
        partition = lambda wc: res_partition("gears_prep", wc.size),
        slurm = lambda wc: res_slurm("gears_prep", wc.size),
    conda: "../envs/gears.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 GEARS_PREP_WORKERS={threads}
        python {params.srcdir}/run_GEARS.py --stage prep \
            --size {wildcards.size} --dataset {wildcards.dataset} \
            --gene-panel {params.gene_panel} \
            --source {input.source} --ctrl {input.ctrl} \
            --pertdata-dir {params.pertdata}
        """


rule gears_train_pds:
    """Train dataset D's GNN on its own train cells; save D's per-pert fingerprints
    (predicted mean over D's panel) for the ensemble. GPU."""
    input:
        cellgraphs = GEARS_PDS_CG,
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        idval  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_val.txt.gz"),
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
    output:
        fp = GEARS_FOLD + "/per_dataset/{dataset}/fingerprints.npz",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
    params:
        gene_panel = gene_panel(),
        srcdir = GEARS_SRC, results_root = RESULTS,
        fold_idx = lambda wc: int(wc.fold.split("_")[1]),
        splits = lambda wc: splits_dir(wc.size, wc.fold), pertdata = GEARS_PDS_PD,
    threads: lambda wc: res_threads_ds("gears", wc.size, wc.dataset)
    resources:
        time = lambda wc: res_time_ds("gears", wc.size, wc.dataset),
        mem = lambda wc: res_mem_ds("gears", wc.size, wc.dataset),
        partition = lambda wc: res_partition_ds("gears", wc.size, wc.dataset),
        slurm = lambda wc: res_slurm_ds("gears", wc.size, wc.dataset),
    conda: "../envs/gears.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/run_GEARS.py --stage train \
            --fold {params.fold_idx} --size {wildcards.size} --dataset {wildcards.dataset} \
            --gene-panel {params.gene_panel} \
            --source {input.source} --ctrl {input.ctrl} \
            --splits-dir {params.splits} --results-root {params.results_root} \
            --pertdata-dir {params.pertdata}
        """


def gears_ens_fps(wc):
    """Every training dataset's fingerprints.npz for this fold (the ensemble fan-in)."""
    return [f"{GEARS_ROOT}/{wc.size}/{wc.fold}/per_dataset/{d}/fingerprints.npz"
            for d in training_datasets(wc.size, wc.fold)]


rule gears_ens_eval:
    """Dataset-blind ensemble: score one split's cells against every dataset's GEARS
    fingerprints (genepert combiner) → logits.h5. CPU."""
    input:
        fps    = gears_ens_fps,
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
        idx    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), f"{wc.split}.txt.gz"),
    output:
        logits = GEARS_FOLD + "/{split}/logits.h5",
    wildcard_constraints:
        split = r"id_test|ood_test",
    params:
        srcdir = GEARS_SRC, froot = GEARS_FOLD,
        splits = lambda wc: splits_dir(wc.size, wc.fold),
    threads: lambda wc: res_threads("gears_ens_eval", wc.size)
    resources:
        time = lambda wc: res_time("gears_ens_eval", wc.size),
        mem = lambda wc: res_mem("gears_ens_eval", wc.size),
        partition = lambda wc: res_partition("gears_ens_eval", wc.size),
        slurm = lambda wc: res_slurm("gears_ens_eval", wc.size),
    conda: "../envs/gears.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/gears_ensemble.py \
            --fold_root {params.froot} --source {input.source} --ctrl {input.ctrl} \
            --splits_dir {params.splits} --split {wildcards.split} --out {output.logits}
        """
