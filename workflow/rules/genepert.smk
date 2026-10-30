# GenePert — ridge regression over GenePT-style gene embeddings. One CPU rule
# per (size, fold): a single run fits on id_train's per-perturbation means and
# scores BOTH splits (id_test + ood_test), since they share that (expensive)
# training-mean build. No GPU, no pretrained checkpoint — only a gene-embedding
# .npz (config: genepert_gene_embeddings). The adapter streams test cells in
# chunks and writes logits.h5 incrementally (a dense CSV is hundreds of GB at
# `full`'s ~9k perts x millions of cells). collect reads logits.h5 directly.
# Resources are size-aware (res_*(name, size)): `full` overrides live in
# config/resources.yaml jobs_by_size.

GENEPERT_SRC  = os.path.join(REPO, "workflow", "methods", "genepert")
GENEPERT_FOLD = f"{RESULTS}/genepert/perturb-seq/{{size}}/{{fold}}"


rule genepert:
    """Fit on the fold's id_train means, score both splits → logits.h5."""
    input:
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        idtest = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_test.txt.gz"),
        ood    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "ood_test.txt.gz"),
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
        emb    = config["genepert_gene_embeddings"],
    output:
        idtest = GENEPERT_FOLD + "/id_test/logits.h5",
        ood    = GENEPERT_FOLD + "/ood_test/logits.h5",
    params:
        gene_panel = gene_panel(),
        srcdir       = GENEPERT_SRC,
        results_root = RESULTS,
        fold_idx     = lambda wc: int(wc.fold.split("_")[1]),
        splits       = lambda wc: splits_dir(wc.size, wc.fold),
    threads: lambda wc: res_threads("genepert", wc.size)
    resources:
        time      = lambda wc: res_time("genepert", wc.size),
        mem       = lambda wc: res_mem("genepert", wc.size),
        partition = lambda wc: res_partition("genepert", wc.size),
        slurm     = lambda wc: res_slurm("genepert", wc.size),
    conda:
        "../envs/genepert.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/scripts/run_genepert.py \
            --size {wildcards.size} --fold {params.fold_idx} \
            --source {input.source} --ctrl {input.ctrl} \
            --embeddings {input.emb} \
            --splits-dir {params.splits} \
            --results-root {params.results_root} \
            --gene-panel {params.gene_panel}
        """
