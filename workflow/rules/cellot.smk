# cellot — neural optimal-transport perturbation models, PER-DATASET by design.
#
# cellot is single-dataset (one scgen AE + per-perturbation OT maps in that AE's
# latent). We train one AE + per-pert OT maps PER dataset D (D's own controls +
# D's own gene panel via --config.data.dataset=D), then evaluate DATASET-BLIND:
# score every test cell against every dataset's models (the cell doesn't reveal
# which of the 21 models to use), softmax-avg over covering models, per-pert
# z-score. Same method name/path as before — per-dataset replaces global (one
# version), matching the fingerprint/genepert per-dataset ensemble.
#
# Stage DAG (per dataset D in the fold's id_train):
#   cellot_ae_pds[D] ──→ cellot_pert_chunk[D,k]  ─┐  (K OT maps/job, chunked)
#                                                 ├─→ cellot_ens_eval[split]
#   (all datasets' chunks) ───────────────────────┘   → <fold>/<split>/logits.h5
#
# OT-map training is CHUNKED (K perts/job) because per-dataset is ~97k OT maps
# per fold at full. Chunk count per dataset = ceil(dataset_perts / K) via
# cellot_pds_chunks (reuses fingerprint's per-dataset pert-count JSON).
#
# cellot is CPU-only. The in-tree `cellot` package is put on PYTHONPATH
# (equivalent to `python setup.py develop`); rules cd into the vendored method dir
# so ./configs/... and ./scripts/... resolve. Outputs land under
# RESULTS/cellot/perturb-seq/<size>/<fold>/, matching method_logits.

CO_SRC   = os.path.join(REPO, "workflow", "methods", "cellot")
CO_ROOT  = f"{RESULTS}/cellot/perturb-seq"
CO_MODEL = f"{RESULTS}/cellot/perturb-seq/{{size}}/{{fold}}"     # fold root
CO_PDS   = CO_MODEL + "/per_dataset/{dataset}"                   # per-dataset subtree


rule cellot_ae_pds:
    """Per-dataset scgen AE (D's controls + D's panel), cellot's working latent."""
    input:
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        config = "workflow/methods/cellot/configs/tasks/perturb-seq/{size}.yaml",
    output:
        model  = CO_PDS + "/ae/cache/model.pt",
        config = CO_PDS + "/ae/config.yaml",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
    params:
        gene_panel = gene_panel(),
        srcdir  = CO_SRC,
        splits  = lambda wc: splits_dir(wc.size, wc.fold),
        outdir  = CO_PDS + "/ae",
        taskcfg = lambda wc: f"configs/tasks/perturb-seq/{wc.size}.yaml",
    threads: lambda wc: res_threads("cellot_ae", wc.size)
    resources:
        time      = lambda wc: res_time("cellot_ae", wc.size),
        mem       = lambda wc: res_mem("cellot_ae", wc.size),
        partition = lambda wc: res_partition("cellot_ae", wc.size),
        slurm     = lambda wc: res_slurm("cellot_ae", wc.size),
    conda:
        "../envs/cellot.yaml"
    shell:
        r"""
        # Data paths come from the split metadata + config (site overlay), never the
        # task yaml. Resolved here rather than as params so they don't trip the
        # params rerun-trigger on outputs that already exist.
        SRC=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['source_path'])" "$(dirname {params.splits})/metadata.json")
        CTRL=$(realpath "{config[ctrl_h5ad]}")
        cd {params.srcdir}
        export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
        export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} PYTHONUNBUFFERED=1
        python scripts/train.py \
            --outdir {params.outdir} \
            --config {params.taskcfg} \
            --config configs/models/scgen.yaml \
            --config.datasplit.path={params.splits} \
            --config.data.path="$SRC" \
            --config.data.path_ctrl="$CTRL" \
            --config.data.controls_only=true \
            --config.data.dataset={wildcards.dataset} \
            --config.data.gene_panel={params.gene_panel} \
            --config.compute_scgen_shift=false \
            --config.training.n_iters=30000 \
            --config.training.eval_freq=500 \
            --config.training.cache_freq=2000 \
            --config.training.logs_freq=100 \
            --config.training.early_stopping_patience=10
        """


rule cellot_pert_chunk:
    """Train chunk k of dataset D's per-pert OT maps (K perts/job) in D's AE latent.
    train_pert_chunk.py derives D's train perts, slices [k*K:(k+1)*K], and fits
    each with --config.data.target={pert} --config.data.dataset={D}."""
    input:
        ae_model = CO_PDS + "/ae/cache/model.pt",
        config   = "workflow/methods/cellot/configs/tasks/perturb-seq/{size}.yaml",
    output:
        marker = CO_PDS + "/pert_chunks/chunk_{k}.done",
    wildcard_constraints:
        dataset = r"[A-Za-z0-9_.+-]+",
        k       = r"\d{4}",
    params:
        gene_panel = gene_panel(),
        srcdir      = CO_SRC,
        splits      = lambda wc: splits_dir(wc.size, wc.fold),
        models_root = CO_PDS + "/models",
        ae_dir      = CO_PDS + "/ae",
        taskcfg     = lambda wc: f"configs/tasks/perturb-seq/{wc.size}.yaml",
        idx         = lambda wc: int(wc.k),
        ppc         = config.get("cellot_perts_per_chunk", 100),
    threads: lambda wc: res_threads("cellot_pert_chunk", wc.size)
    resources:
        time      = lambda wc: res_time("cellot_pert_chunk", wc.size),
        mem       = lambda wc: res_mem("cellot_pert_chunk", wc.size),
        partition = lambda wc: res_partition("cellot_pert_chunk", wc.size),
        slurm     = lambda wc: res_slurm("cellot_pert_chunk", wc.size),
    conda:
        "../envs/cellot.yaml"
    shell:
        r"""
        # Data paths come from the split metadata + config (site overlay), never the
        # task yaml. Resolved here rather than as params so they don't trip the
        # params rerun-trigger on outputs that already exist.
        SRC=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['source_path'])" "$(dirname {params.splits})/metadata.json")
        CTRL=$(realpath "{config[ctrl_h5ad]}")
        cd {params.srcdir}
        export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
        export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} PYTHONUNBUFFERED=1
        python scripts/train_pert_chunk.py \
            --splits_dir {params.splits} --dataset {wildcards.dataset} \
            --gene-panel {params.gene_panel} \
            --chunk_idx {params.idx} --perts_per_chunk {params.ppc} \
            --models_root {params.models_root} --ae_dir {params.ae_dir} \
            --taskcfg {params.taskcfg} --marker {output.marker} \
            --data_path "$SRC" --data_path_ctrl "$CTRL"
        """


def cellot_ens_markers(wc):
    """Every per-dataset chunk marker for this fold — one dataset's AE + all its
    OT-map chunks feed the dataset-blind ensemble. Fans out over the fold's
    training datasets and each dataset's ceil(perts/K) chunks."""
    out = []
    for D in training_datasets(wc.size, wc.fold):
        n_chunks = cellot_pds_chunks(wc.size, wc.fold, D)
        out += [f"{CO_ROOT}/{wc.size}/{wc.fold}/per_dataset/{D}/pert_chunks/chunk_{k:04d}.done"
                for k in range(n_chunks)]
    return out


rule cellot_ens_eval:
    """Dataset-blind ensemble: score one split's cells against every dataset's
    per-pert OT maps (own-control latent delta), softmax-avg over covering models,
    per-pert z-score → logits.h5 (same path as the old global cellot)."""
    input:
        markers = cellot_ens_markers,
        idx     = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), f"{wc.split}.txt.gz"),
        ctrl    = config["ctrl_h5ad"],
    output:
        logits = CO_MODEL + "/{split}/logits.h5",
    wildcard_constraints:
        split = r"id_test|ood_test",
    params:
        gene_panel = gene_panel(),
        srcdir = CO_SRC,
        splits = lambda wc: splits_dir(wc.size, wc.fold),
        froot  = CO_MODEL,
    threads: lambda wc: res_threads("cellot_eval", wc.size)
    resources:
        time      = lambda wc: res_time("cellot_eval", wc.size),
        mem       = lambda wc: res_mem("cellot_eval", wc.size),
        partition = lambda wc: res_partition("cellot_eval", wc.size),
        slurm     = lambda wc: res_slurm("cellot_eval", wc.size),
    conda:
        "../envs/cellot.yaml"
    shell:
        r"""
        cd {params.srcdir}
        export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
        export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} NUMEXPR_MAX_THREADS={threads} PYTHONUNBUFFERED=1
        python scripts/predict_ensemble.py \
            --fold_root {params.froot} \
            --splits_dir {params.splits} \
            --split {wildcards.split} \
            --ctrl {input.ctrl} \
            --out {output.logits} \
            --gene-panel {params.gene_panel}
        """
