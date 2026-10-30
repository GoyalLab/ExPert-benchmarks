# fingerprint — R/Seurat perturbation classifier, expressed as fine-grained
# chunk targets. The eval step is chunked per (fold, split, chunk) so an OOM'd
# eval chunk (the fold_3/5 failure that motivated this repo) reruns by itself
# instead of taking down the whole fold; concat_eval then blocks loudly if any
# chunk is missing.
#
# Stage DAG:  fp_prep ─→ fp_deg_chunk[d] ─→ fp_concat_deg ─→ fp_fit_chunk[i]
#               └→ fp_concat_dict ─→ fp_eval_chunk[split,j] ─→ fp_concat_eval
#
# DEG (FindMarkers over perts) is chunked as a SLURM array: `prep --skip_deg`
# saves train_obj.rds only, fp_deg_chunk computes one pert-slice's DEGs, and
# fp_concat_deg unions them into features.rds. At `full` (~9k perts) an
# in-process DEG loop in prep would be tens of hours in a single job; small uses
# deg:1 (one chunk ≈ the old in-process DEG). Chunk counts are config-fixed per
# size (config["fingerprint_chunks"]), so the fan-out is plain expand().
#
# Resources are size-aware (res_*(name, size)): only the per-fold singletons
# fp_prep + fp_concat_eval go to himem at `full` (jobs_by_size in resources.yaml);
# the hundreds of deg/fit/eval chunks stay on the `cpu` class, sized under a node.
#
# fingerprint.R writes under <output_dir>/<size>/<fold>/shared_genes/ (data_bn ==
# size); --output_dir is RESULTS/fingerprint. Each rule cd's into the vendored
# method dir because fingerprint.R source()s utils.R relative to cwd, and exports
# R_LIBS_USER so the post-deploy GitHub/Bioc packages are on the library path.
# Paths handed to the R script are absolute, so the cd is safe.

FP_SRC   = os.path.join(REPO, "workflow", "methods", "fingerprint")
FP_OUT   = f"{RESULTS}/fingerprint"                              # --output_dir
FP_MODEL = f"{RESULTS}/fingerprint/{{size}}/{{fold}}/shared_genes"  # output pattern
FP_SPLITS = config["method_splits"]["fingerprint"]              # [id_test, ood_test]


rule fp_prep:
    """Fit ONE SCT model on a ≤nnz-cap subsample (--skip_deg) → sct_model.rds; the
    deg/fit chunks realize their own perts and apply it (no 940k cap). Singleton
    heavy job → himem at `full`."""
    input:
        train = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        meta  = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl  = config["ctrl_h5ad"],
    output:
        sct_model = FP_MODEL + "/sct_model.rds",
    params:
        srcdir   = FP_SRC,
        fold_dir = lambda wc: splits_dir(wc.size, wc.fold),
        outdir   = FP_OUT,
        max_cpp  = lambda wc: config["fingerprint_max_cells_per_pert"][wc.size],
    threads: lambda wc: res_threads("fp_prep", wc.size)
    resources:
        time      = lambda wc: res_time("fp_prep", wc.size),
        mem       = lambda wc: res_mem("fp_prep", wc.size),
        partition = lambda wc: res_partition("fp_prep", wc.size),
        slurm     = lambda wc: res_slurm("fp_prep", wc.size),
    conda:
        "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode prep --skip_deg \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --shared_genes --split_key dataset \
            --max_cells_per_pert {params.max_cpp} --cores {threads}
        """


rule fp_deg_chunk:
    """Compute DEGs (FindMarkers) for one pert-slice (chunk d of n_deg): realize
    the slice's perts (uncapped) + controls, apply the cached SCT model, FindMarkers."""
    input:
        sct_model = FP_MODEL + "/sct_model.rds",
        meta      = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl      = config["ctrl_h5ad"],
    output:
        chunk = FP_MODEL + "/deg_chunks/chunk_{d}.rds",
    wildcard_constraints:
        d = r"\d{4}",
    params:
        srcdir   = FP_SRC,
        fold_dir = lambda wc: splits_dir(wc.size, wc.fold),
        outdir   = FP_OUT,
        n_chunks = lambda wc: fp_chunks(wc.size, "deg"),
        idx      = lambda wc: int(wc.d),
    threads: lambda wc: res_threads("fp_deg_chunk", wc.size)
    resources:
        time      = lambda wc: res_time("fp_deg_chunk", wc.size),
        mem       = lambda wc: res_mem("fp_deg_chunk", wc.size),
        partition = lambda wc: res_partition("fp_deg_chunk", wc.size),
        slurm     = lambda wc: res_slurm("fp_deg_chunk", wc.size),
    conda:
        "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode deg_chunk \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --shared_genes --split_key dataset \
            --chunk_idx {params.idx} --n_chunks {params.n_chunks} --cores {threads}
        """


rule fp_concat_deg:
    """Union the DEG chunks into features.rds (consumed by fp_fit_chunk)."""
    input:
        chunks = lambda wc: [
            f"{fp_model_dir(wc.size, wc.fold)}/deg_chunks/chunk_{d:04d}.rds"
            for d in range(fp_chunks(wc.size, "deg"))
        ],
        ctrl = config["ctrl_h5ad"],
    output:
        features = FP_MODEL + "/features.rds",
    params:
        srcdir   = FP_SRC,
        fold_dir = lambda wc: splits_dir(wc.size, wc.fold),
        outdir   = FP_OUT,
    threads: lambda wc: res_threads("fp_concat_deg", wc.size)
    resources:
        time      = lambda wc: res_time("fp_concat_deg", wc.size),
        mem       = lambda wc: res_mem("fp_concat_deg", wc.size),
        partition = lambda wc: res_partition("fp_concat_deg", wc.size),
        slurm     = lambda wc: res_slurm("fp_concat_deg", wc.size),
    conda:
        "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode concat_deg \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --shared_genes --split_key dataset \
            --cores {threads}
        """


rule fp_fit_chunk:
    """Fit one slice of the perturbation dictionary (chunk i of n_fit): realize the
    slice's perts (uncapped) + controls, apply the cached SCT model, LearnDictionary."""
    input:
        sct_model = FP_MODEL + "/sct_model.rds",
        features  = FP_MODEL + "/features.rds",
        meta      = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl      = config["ctrl_h5ad"],
    output:
        chunk = FP_MODEL + "/dict_chunks/chunk_{i}.rds",
    wildcard_constraints:
        i = r"\d{4}",
    params:
        srcdir   = FP_SRC,
        fold_dir = lambda wc: splits_dir(wc.size, wc.fold),
        outdir   = FP_OUT,
        n_chunks = lambda wc: fp_chunks(wc.size, "fit"),
        idx      = lambda wc: int(wc.i),
    threads: lambda wc: res_threads("fp_fit_chunk", wc.size)
    resources:
        time      = lambda wc: res_time("fp_fit_chunk", wc.size),
        mem       = lambda wc: res_mem("fp_fit_chunk", wc.size),
        partition = lambda wc: res_partition("fp_fit_chunk", wc.size),
        slurm     = lambda wc: res_slurm("fp_fit_chunk", wc.size),
    conda:
        "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode fit_chunk \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --shared_genes --split_key dataset \
            --chunk_idx {params.idx} --n_chunks {params.n_chunks} --cores {threads}
        """


rule fp_concat_dict:
    """cbind the fitted dict_chunks into dict.rds (needed before eval arrays)."""
    input:
        chunks = lambda wc: [
            f"{fp_model_dir(wc.size, wc.fold)}/dict_chunks/chunk_{i:04d}.rds"
            for i in range(fp_chunks(wc.size, "fit"))
        ],
        ctrl = config["ctrl_h5ad"],
    output:
        dict = FP_MODEL + "/dict.rds",
    params:
        srcdir   = FP_SRC,
        fold_dir = lambda wc: splits_dir(wc.size, wc.fold),
        outdir   = FP_OUT,
    threads: lambda wc: res_threads("fp_concat_dict", wc.size)
    resources:
        time      = lambda wc: res_time("fp_concat_dict", wc.size),
        mem       = lambda wc: res_mem("fp_concat_dict", wc.size),
        partition = lambda wc: res_partition("fp_concat_dict", wc.size),
        slurm     = lambda wc: res_slurm("fp_concat_dict", wc.size),
    conda:
        "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode concat_dict \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --shared_genes --split_key dataset \
            --cores {threads}
        """


rule fp_eval_chunk:
    """Score one slice of one eval split's cells (chunk j of n_eval). THE
    chunk-level OOM-rerun target."""
    input:
        dict = FP_MODEL + "/dict.rds",
        test = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), f"{wc.split}.txt.gz"),
        meta = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl = config["ctrl_h5ad"],
    output:
        chunk = FP_MODEL + "/eval_chunks/{split}/chunk_{j}.rds",
    wildcard_constraints:
        split = r"id_test|ood_test",
        j     = r"\d{4}",
    params:
        srcdir   = FP_SRC,
        fold_dir = lambda wc: splits_dir(wc.size, wc.fold),
        outdir   = FP_OUT,
        n_chunks = lambda wc: fp_chunks(wc.size, "eval"),
        idx      = lambda wc: int(wc.j),
    threads: lambda wc: res_threads("fp_eval_chunk", wc.size)
    resources:
        time      = lambda wc: res_time("fp_eval_chunk", wc.size),
        mem       = lambda wc: res_mem("fp_eval_chunk", wc.size),
        partition = lambda wc: res_partition("fp_eval_chunk", wc.size),
        slurm     = lambda wc: res_slurm("fp_eval_chunk", wc.size),
    conda:
        "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode eval_chunk \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --shared_genes --split_key dataset \
            --eval_split {wildcards.split} --chunk_idx {params.idx} \
            --n_chunks {params.n_chunks} --cores {threads}
        """


rule fp_concat_eval:
    """Concatenate every split's eval chunks into final logits.h5 + report.csv.
    One job per fold (fingerprint.R loops the splits internally); --eval_splits
    restricts it to the scored splits so id_val is never run in-process."""
    input:
        dict = FP_MODEL + "/dict.rds",
        eval_chunks = lambda wc: [
            f"{fp_model_dir(wc.size, wc.fold)}/eval_chunks/{s}/chunk_{j:04d}.rds"
            for s in FP_SPLITS
            for j in range(fp_chunks(wc.size, "eval"))
        ],
        meta = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl = config["ctrl_h5ad"],
    output:
        logits = [FP_MODEL + f"/{s}/logits.h5" for s in FP_SPLITS],
        report = [FP_MODEL + f"/{s}/report.csv" for s in FP_SPLITS],
    params:
        srcdir      = FP_SRC,
        fold_dir    = lambda wc: splits_dir(wc.size, wc.fold),
        outdir      = FP_OUT,
        eval_splits = ",".join(FP_SPLITS),
    threads: lambda wc: res_threads("fp_concat_eval", wc.size)
    resources:
        time      = lambda wc: res_time("fp_concat_eval", wc.size),
        mem       = lambda wc: res_mem("fp_concat_eval", wc.size),
        partition = lambda wc: res_partition("fp_concat_eval", wc.size),
        slurm     = lambda wc: res_slurm("fp_concat_eval", wc.size),
    conda:
        "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode concat_eval \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --shared_genes --split_key dataset \
            --eval_splits {params.eval_splits} --cores {threads}
        """


# =============================================================================
# Per-dataset fingerprint: train ONE model per dataset on its own gene panel.
# Mirrors the shared-path chain (prep→deg→concat_deg→fit→concat_dict) but adds a
# {dataset} wildcard + --train_dataset (which restricts train_idx to that dataset
# and swaps --shared_genes for the dataset's own feature_mask panel). deg/fit
# chunk counts are sized PER DATASET (fp_pds_chunks). The ensemble inference (a
# separate stage) scores every test cell against every dataset's dict.rds.
# =============================================================================
FP_PDS = f"{RESULTS}/fingerprint/{{size}}/{{fold}}/per_dataset_panel/{{dataset}}"


rule fp_pds_prep:
    """SCT model on one dataset's train subsample (--train_dataset)."""
    input:
        train = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        meta  = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl  = config["ctrl_h5ad"],
    output:
        sct_model = FP_PDS + "/sct_model.rds",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
    params:
        gene_panel = gene_panel(),
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
        max_cpp = lambda wc: config["fingerprint_max_cells_per_pert"][wc.size],
    threads: lambda wc: res_threads("fp_pds_prep", wc.size)
    resources:
        time = lambda wc: res_time("fp_pds_prep", wc.size), mem = lambda wc: res_mem("fp_pds_prep", wc.size),
        partition = lambda wc: res_partition("fp_pds_prep", wc.size), slurm = lambda wc: res_slurm("fp_pds_prep", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode prep --skip_deg \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --split_key dataset \
            --train_dataset {wildcards.dataset} --gene_panel {params.gene_panel} \
            --max_cells_per_pert {params.max_cpp} --cores {threads}
        """


rule fp_pds_deg_chunk:
    input:
        sct_model = FP_PDS + "/sct_model.rds",
        meta = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl = config["ctrl_h5ad"],
    output:
        chunk = FP_PDS + "/deg_chunks/chunk_{d}.rds",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+", d = r"\d{4}",
    params:
        gene_panel = gene_panel(),
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
        n_chunks = lambda wc: fp_pds_chunks(wc.size, wc.fold, wc.dataset, "deg"), idx = lambda wc: int(wc.d),
    threads: lambda wc: res_threads("fp_pds_deg_chunk", wc.size)
    resources:
        time = lambda wc: res_time("fp_pds_deg_chunk", wc.size), mem = lambda wc: res_mem("fp_pds_deg_chunk", wc.size),
        partition = lambda wc: res_partition("fp_pds_deg_chunk", wc.size), slurm = lambda wc: res_slurm("fp_pds_deg_chunk", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode deg_chunk \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --split_key dataset --train_dataset {wildcards.dataset} --gene_panel {params.gene_panel} \
            --chunk_idx {params.idx} --n_chunks {params.n_chunks} --cores {threads}
        """


rule fp_pds_concat_deg:
    input:
        chunks = lambda wc: [
            f"{fp_pds_model_dir(wc.size, wc.fold, wc.dataset)}/deg_chunks/chunk_{d:04d}.rds"
            for d in range(fp_pds_chunks(wc.size, wc.fold, wc.dataset, "deg"))
        ],
        ctrl = config["ctrl_h5ad"],
    output:
        features = FP_PDS + "/features.rds",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
    params:
        gene_panel = gene_panel(),
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
    threads: lambda wc: res_threads("fp_pds_concat_deg", wc.size)
    resources:
        time = lambda wc: res_time("fp_pds_concat_deg", wc.size), mem = lambda wc: res_mem("fp_pds_concat_deg", wc.size),
        partition = lambda wc: res_partition("fp_pds_concat_deg", wc.size), slurm = lambda wc: res_slurm("fp_pds_concat_deg", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode concat_deg \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --split_key dataset --train_dataset {wildcards.dataset} --gene_panel {params.gene_panel} \
            --cores {threads}
        """


rule fp_pds_fit_chunk:
    input:
        sct_model = FP_PDS + "/sct_model.rds",
        features  = FP_PDS + "/features.rds",
        meta = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl = config["ctrl_h5ad"],
    output:
        chunk = FP_PDS + "/dict_chunks/chunk_{i}.rds",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+", i = r"\d{4}",
    params:
        gene_panel = gene_panel(),
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
        n_chunks = lambda wc: fp_pds_chunks(wc.size, wc.fold, wc.dataset, "fit"), idx = lambda wc: int(wc.i),
    threads: lambda wc: res_threads("fp_pds_fit_chunk", wc.size)
    resources:
        time = lambda wc: res_time("fp_pds_fit_chunk", wc.size), mem = lambda wc: res_mem("fp_pds_fit_chunk", wc.size),
        partition = lambda wc: res_partition("fp_pds_fit_chunk", wc.size), slurm = lambda wc: res_slurm("fp_pds_fit_chunk", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode fit_chunk \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --split_key dataset --train_dataset {wildcards.dataset} --gene_panel {params.gene_panel} \
            --chunk_idx {params.idx} --n_chunks {params.n_chunks} --cores {threads}
        """


rule fp_pds_concat_dict:
    input:
        chunks = lambda wc: [
            f"{fp_pds_model_dir(wc.size, wc.fold, wc.dataset)}/dict_chunks/chunk_{i:04d}.rds"
            for i in range(fp_pds_chunks(wc.size, wc.fold, wc.dataset, "fit"))
        ],
        ctrl = config["ctrl_h5ad"],
    output:
        dict = FP_PDS + "/dict.rds",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
    params:
        gene_panel = gene_panel(),
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
    threads: lambda wc: res_threads("fp_pds_concat_dict", wc.size)
    resources:
        time = lambda wc: res_time("fp_pds_concat_dict", wc.size), mem = lambda wc: res_mem("fp_pds_concat_dict", wc.size),
        partition = lambda wc: res_partition("fp_pds_concat_dict", wc.size), slurm = lambda wc: res_slurm("fp_pds_concat_dict", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript fingerprint.R --mode concat_dict \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} \
            --output_dir {params.outdir} --split_key dataset --train_dataset {wildcards.dataset} --gene_panel {params.gene_panel} \
            --cores {threads}
        """


# =============================================================================
# Per-dataset fingerprint ENSEMBLE inference. Each eval chunk (by CELLS) scores
# its cells against EVERY dataset's dict.rds and combines by softmax-averaging
# over covering models -> per-cell logits over the global pert space. Concat
# stacks the chunks into logits.h5 (write_logits_h5 schema). Fans in on ALL the
# per-dataset dicts for the fold.
# =============================================================================
FP_ENS = f"{RESULTS}/fingerprint/{{size}}/{{fold}}/ensemble/{{split}}"


rule fp_ens_eval_ds_chunk:
    """FAN-OUT: score one cell-chunk against ONE dataset's dict (own-control,
    capped controls) → ds_probs/<dataset>/chunk_<j>.rds. Parallelises the serial
    17-dataset susie loop (the wall-clock pole) into one job per dataset, and
    bounds memory to a single dataset's realize+SCT+query."""
    input:
        dict = lambda wc: f"{fp_pds_model_dir(wc.size, wc.fold, wc.dataset)}/dict.rds",
        test = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), f"{wc.split}.txt.gz"),
        meta = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl = config["ctrl_h5ad"],
    output:
        chunk = FP_ENS + "/ds_probs/{dataset}/chunk_{j}.rds",
    wildcard_constraints:
        split = r"id_test|ood_test", dataset = r"[A-Za-z0-9_.+-]+", j = r"\d{4}",
    params:
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
        n_chunks = lambda wc: fp_ens_chunks(wc.size), idx = lambda wc: int(wc.j),
    threads: lambda wc: res_threads_ds("fp_ens_eval_ds_chunk", wc.size, wc.dataset)
    resources:
        time = lambda wc: res_time_ds("fp_ens_eval_ds_chunk", wc.size, wc.dataset), mem = lambda wc: res_mem_ds("fp_ens_eval_ds_chunk", wc.size, wc.dataset),
        partition = lambda wc: res_partition_ds("fp_ens_eval_ds_chunk", wc.size, wc.dataset), slurm = lambda wc: res_slurm_ds("fp_ens_eval_ds_chunk", wc.size, wc.dataset),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript ensemble_eval.R \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} --output_dir {params.outdir} \
            --split {wildcards.split} --chunk_idx {params.idx} --n_chunks {params.n_chunks} \
            --dataset {wildcards.dataset} --cores {threads}
        """


def fp_ens_ds_probs(wc):
    """All (dataset x cell-chunk) fan-out prob files for this fold+split — the
    concat fans in on every dataset's every chunk."""
    n = fp_ens_chunks(wc.size)
    return [
        f"{RESULTS}/fingerprint/{wc.size}/{wc.fold}/ensemble/{wc.split}/ds_probs/{d}/chunk_{j:04d}.rds"
        for d in training_datasets(wc.size, wc.fold)
        for j in range(n)
    ]


rule fp_ens_concat:
    input:
        probs = fp_ens_ds_probs,
    output:
        logits = f"{RESULTS}/fingerprint_perdataset/{{size}}/{{fold}}/{{split}}/logits.h5",
    wildcard_constraints:
        split = r"id_test|ood_test",
    params:
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
        n_chunks = lambda wc: fp_ens_chunks(wc.size),
    threads: lambda wc: res_threads("fp_ens_concat", wc.size)
    resources:
        time = lambda wc: res_time("fp_ens_concat", wc.size), mem = lambda wc: res_mem("fp_ens_concat", wc.size),
        partition = lambda wc: res_partition("fp_ens_concat", wc.size), slurm = lambda wc: res_slurm("fp_ens_concat", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript ensemble_concat_ds.R \
            --output_dir {params.outdir} --fold_dir {params.fold_dir} \
            --split {wildcards.split} --n_chunks {params.n_chunks} \
            --out_h5 {output.logits}
        """


# ---- MERGED (combined-weights) fingerprint — the affordable full-scale variant ----
# Merge all per-dataset dicts into ONE (union features x union perts, shared perts
# averaged) and score cells against that single dict — ~15x fewer jobs than the
# per-dataset fan-out. Accuracy is much weaker (small top1 ~0.04 vs the ensemble
# ~0.24 — it averages fingerprints across mismatched per-dataset SCT spaces), but it
# actually SCALES: the millions of full cells are cell-chunked into small single-node
# jobs (one merged dict/chunk vs the ensemble's 17), then rbound. Tracked as a
# DISTINCT method dir so collect scores it next to fingerprint_perdataset.
FP_MERGE = f"{RESULTS}/fingerprint/{{size}}/{{fold}}/merged/{{split}}"


rule fp_merge_eval_chunk:
    """Score one cell-chunk against the merged dict -> chunk_<j>.h5 (its perturbed
    cells only). n_chunks from config (small=1 single pass; full=500 -> ~5.8k
    cells/chunk, fits the 192GB `short` pool)."""
    input:
        dicts = lambda wc: [f"{fp_pds_model_dir(wc.size, wc.fold, d)}/dict.rds"
                            for d in training_datasets(wc.size, wc.fold)],
        meta = lambda wc: os.path.join(SPLITS_ROOT, wc.size, "metadata.json"),
        ctrl = config["ctrl_h5ad"],
    output:
        chunk = FP_MERGE + "/chunks/chunk_{j}.h5",
    wildcard_constraints:
        split = r"id_test|ood_test", j = r"\d{4}",
    params:
        srcdir = FP_SRC, fold_dir = lambda wc: splits_dir(wc.size, wc.fold), outdir = FP_OUT,
        n_chunks = lambda wc: fp_merge_chunks(wc.size), idx = lambda wc: int(wc.j),
    threads: lambda wc: res_threads("fp_merge_eval_chunk", wc.size)
    resources:
        time = lambda wc: res_time("fp_merge_eval_chunk", wc.size), mem = lambda wc: res_mem("fp_merge_eval_chunk", wc.size),
        partition = lambda wc: res_partition("fp_merge_eval_chunk", wc.size), slurm = lambda wc: res_slurm("fp_merge_eval_chunk", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript merge_eval.R \
            --fold_dir {params.fold_dir} --ctrl_p {input.ctrl} --output_dir {params.outdir} \
            --split {wildcards.split} --chunk_idx {params.idx} --n_chunks {params.n_chunks} \
            --out_h5 {output.chunk} --cores {threads}
        """


def fp_merge_chunk_files(wc):
    n = fp_merge_chunks(wc.size)
    return [f"{RESULTS}/fingerprint/{wc.size}/{wc.fold}/merged/{wc.split}/chunks/chunk_{j:04d}.h5"
            for j in range(n)]


rule fp_merge_concat:
    """rbind the merged cell-chunks (all share one dict -> identical pert cols) ->
    fingerprint_merged/<...>/logits.h5."""
    input:
        chunks = fp_merge_chunk_files,
    output:
        logits = f"{RESULTS}/fingerprint_merged/{{size}}/{{fold}}/{{split}}/logits.h5",
    wildcard_constraints:
        split = r"id_test|ood_test",
    params:
        srcdir = FP_SRC, n_chunks = lambda wc: fp_merge_chunks(wc.size),
        chunk_dir = lambda wc: f"{RESULTS}/fingerprint/{wc.size}/{wc.fold}/merged/{wc.split}/chunks",
    threads: lambda wc: res_threads("fp_merge_concat", wc.size)
    resources:
        time = lambda wc: res_time("fp_merge_concat", wc.size), mem = lambda wc: res_mem("fp_merge_concat", wc.size),
        partition = lambda wc: res_partition("fp_merge_concat", wc.size), slurm = lambda wc: res_slurm("fp_merge_concat", wc.size),
    conda: "../envs/fingerprint.yaml"
    shell:
        r"""
        export R_LIBS_USER="${{CONDA_PREFIX}}/lib/R/library"
        cd {params.srcdir}
        Rscript merge_concat.R \
            --chunk_dir {params.chunk_dir} --n_chunks {params.n_chunks} --out_h5 {output.logits}
        """
