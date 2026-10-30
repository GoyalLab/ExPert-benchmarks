# Padding mode — the unseen-perturbation benchmark (docs/padding_scope.md).
#
# Scores perturbations that have NO training cells. ExPert can do this through its
# padded class graph; the only baselines that can are forward models conditioned on
# a TRANSFERABLE representation of the target gene (a gene embedding, or a GO/graph
# node). Everything here is keyed on size == PADDING_SIZE (`zs_f4`), which is NOT in
# config['sizes'] — so `rule all` and every other size are untouched.
#
# The defining asymmetry: TRAINING reads the ordinary `full` split space (real
# cells, the 8,964 seen perturbations) while EVAL reads the zero-shot store (10,245
# perturbations, disjoint from every training pool). Hence every rule below takes
# two sources, and none of them reuses `splits_dir(wc.size, ...)` for training.
#
#   zs_splits                        CPU/himem — build the eval split space + the
#                                    canonical class space; ASSERT disjointness.
#   expert_atoms                     CPU — export ExPert's learned class-embedding
#                                    table as a drop-in feature set (sharp test).
#   expert_zs                        CPU — score the trained ExPert model.
#   genepert_zs / genepert_atoms_zs  CPU — ridge on gene embeddings; predicts an
#                                    unseen pert straight from its embedding.
#   scouter_zs_train_pds/_eval       GPU/CPU — frozen embedding lookup, so unseen
#                                    perts need a table row, NOT a retrain.
#   gears_zs_prep/_train/_eval       CPU/GPU/CPU — MUST retrain: num_perts sizes
#                                    nn.Embedding, and the pert graph grows
#                                    9,853 -> 18,595 nodes.
#   baselines_zs                     CPU — the two untrained floors.
#   emb_nn_zs                        CPU — nearest-seen-perturbation transfer.

# Scheduling: one controller can drive `metrics/ctx7 metrics/zs_f4` together. ctx7 is
# ~24k jobs and the profile caps concurrency at 60, so without a nudge the ~30 padding
# jobs would queue behind it for days. A higher `priority:` makes Snakemake submit
# them first whenever both are runnable — the padding arm finishes early and ctx7
# simply uses the slots afterwards.
PAD_PRIORITY = int((config.get("padding") or {}).get("priority", 50))

# Padding writes to its OWN results root (config padding.results_root): that is what
# keeps these rules from being ambiguous with the main grid's {size}-wildcard rules
# (gears_prep_pds, scouter_train_pds, genepert, baselines, expert_logits all match
# size=zs_f4 otherwise), and keeps ~8GB logits out of the collected tree.
PAD_RESULTS = PADDING_RESULTS

PAD_SIZE   = PADDING_SIZE
PAD_FOLD   = padding_fold()
PAD_DIR    = f"{PADDING_SPLITS_ROOT}/{PAD_SIZE}"
PAD_FOLDDIR = f"{PAD_DIR}/{PAD_FOLD}"
PAD_SCRIPTS = "workflow/scripts"


def pad_graph_indices():
    """Index files whose rows GEARS may build cell graphs for. Identical in prep and
    train because they share the cache. Bounding this is not an optimisation: on the
    full store an unbounded per-dataset graph is tens of GB (~600-950GB over 20
    datasets), against ~325GB free on the results allocation."""
    return [os.path.join(padding_train_dir(), f"{padding_train_index('gears')}.txt.gz"),
            os.path.join(padding_train_dir(), "id_val_cap25.txt.gz")]


def pad_train_idx(method):
    """The training index file this method consumes (scope §2: methods that only
    need per-pert MEANS stream the uncapped 15.7M; cell-level methods get a cap)."""
    return os.path.join(padding_train_dir(), f"{padding_train_index(method)}.txt.gz")


# ---------------------------------------------------------------------------
rule zs_splits:
    """Materialise the padding eval split space + classes.txt, and hard-assert that
    the zero-shot label set is disjoint from the training pool."""
    input:
        eval_source  = PADDING.get("eval_source", ""),
        train_source = lambda wc: source_h5ad(padding_train_size()),
        train_idx    = lambda wc: pad_train_idx("default"),
        # ExPert refuses to set up if ANY observed label lacks a class embedding, so
        # the eval set must be filtered to embeddable classes up front.
        emb          = config["genepert_gene_embeddings"],
    output:
        meta    = f"{PAD_DIR}/metadata.json",
        classes = f"{PAD_DIR}/classes.txt",
        summary = f"{PAD_DIR}/summary.json",
        trperts = f"{PAD_DIR}/train_perts.txt",
        zs_id   = f"{PAD_FOLDDIR}/zs_id.txt.gz",
        zs_ood  = f"{PAD_FOLDDIR}/zs_ood.txt.gz",
    params:
        out_root    = PADDING_SPLITS_ROOT,
        size        = PAD_SIZE,
        fold        = PAD_FOLD,
        splits_root = SPLITS_ROOT,
        train_size  = padding_train_size(),
        train_index = padding_train_index("default"),
        cap         = int(PADDING.get("eval_cap", 20)),
        seed        = int(PADDING.get("eval_cap_seed", 42)),
        class_space = PADDING.get("class_space", "zs_only"),
    priority: PAD_PRIORITY
    threads: res_threads("zs_splits")
    resources:
        time      = res_time("zs_splits"),
        mem       = res_mem("zs_splits"),
        partition = res_partition("zs_splits"),
        slurm     = res_slurm("zs_splits"),
    conda:
        "../envs/baselines.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {PAD_SCRIPTS}/make_zs_splits.py \
            --eval-source {input.eval_source} --train-source {input.train_source} \
            --splits-root {params.splits_root} --train-size {params.train_size} \
            --train-index {params.train_index} \
            --out-root {params.out_root} --size {params.size} --fold {params.fold} \
            --eval-cap {params.cap} --seed {params.seed} \
            --class-space {params.class_space} \
            --require-embedding {input.emb}
        """


# ---------------------------------------------------------------------------
rule expert_atoms:
    """Export the trained model's class-embedding table in both consumer formats
    (npz for GenePert, pickle for scouter) — the §12 sharp test's feature set."""
    input:
        model_pt = PADDING.get("expert_model_pt", ""),
    output:
        npz = padding_atoms_npz(),
        pkl = padding_atoms_pkl(),
    params:
        tensor = (PADDING.get("expert_atoms") or {}).get("tensor", "cls_emb.embedding.weight"),
        python = config["expert_python"],
    priority: PAD_PRIORITY
    threads: res_threads("expert_atoms")
    resources:
        time      = res_time("expert_atoms"),
        mem       = res_mem("expert_atoms"),
        partition = res_partition("expert_atoms"),
        slurm     = res_slurm("expert_atoms"),
    shell:
        r"""
        # No `conda:` — needs torch; reuse the ExPert env like expert_logits does.
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        {params.python} {PAD_SCRIPTS}/export_expert_atoms.py \
            --model-pt {input.model_pt} --tensor {params.tensor} \
            --out-npz {output.npz} --out-pkl {output.pkl} --verify-rows 40
        """


# ---------------------------------------------------------------------------
rule expert_zs:
    """Score the trained ExPert model over the zero-shot cells against its PADDED
    class vocabulary. No leak filter: the zs perturbations are disjoint from the
    model's training pool, so --leak-split is deliberately omitted."""
    input:
        zs_id   = f"{PAD_FOLDDIR}/zs_id.txt.gz",
        zs_ood  = f"{PAD_FOLDDIR}/zs_ood.txt.gz",
        classes = f"{PAD_DIR}/classes.txt",
        query   = lambda wc: source_h5ad(PAD_SIZE),
        # NOTE: with --no-model-adata this input is unused by the runner; kept only so
        # the DAG still expresses the dependency. run_expert_logits.py
        # maps query cells into the model adata BY OBS_NAME
        # (`mpos.get_indexer(q_names[ii])`) and calls `model.predict(indices=...)`
        # against it. On the main grid the benchmark cells ARE rows of the training
        # store, so that identity holds; here the query lives in a different file, so
        # passing the training store matched 0/102,211 cells and both splits silently
        # skipped (job exits 0, MissingOutputException). Head A only needs z_shared
        # from the encoder (expression + dataset batch), so the labels being outside
        # the model's 8,964-class registry does not matter for the forward pass.
        model_adata = lambda wc: source_h5ad(PAD_SIZE),
        ctrl    = config["ctrl_h5ad"],
        weights = PADDING.get("expert_model_pt", ""),
        # The padded anchor table -- reuse the SAME export the sharp test consumes, so
        # checkpoint-layout knowledge lives in exactly one script.
        atoms   = padding_atoms_npz(),
    output:
        zs_id  = f"{PAD_RESULTS}/ExPert/perturb-seq/{PAD_SIZE}/{PAD_FOLD}/zs_id/logits.h5",
        zs_ood = f"{PAD_RESULTS}/ExPert/perturb-seq/{PAD_SIZE}/{PAD_FOLD}/zs_ood/logits.h5",
    params:
        model_dir = lambda wc, input: os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(input.weights)))),
        fold_dir  = PAD_FOLDDIR,
        out_in    = f"{PAD_RESULTS}/ExPert/perturb-seq/{PAD_SIZE}/{PAD_FOLD}",
        # The full-vocabulary export (19,895 classes = the `union` diagnostic of §6)
        # is ~16GB per arm. keep it under the padding root, which is meant to sit
        # on storage with room to spare (expert_fullpanel_root may not).
        out_fp    = f"{PAD_RESULTS}/_fullpanel/ExPert/perturb-seq/{PAD_SIZE}/{PAD_FOLD}",
        head      = config.get("expert_head", "A"),
        repo      = config["expert_repo"],
        python    = config["expert_python"],
        src       = os.path.join(REPO, "workflow", "methods", "expert"),
    priority: PAD_PRIORITY
    threads: res_threads("expert_zs")
    resources:
        time      = res_time("expert_zs"),
        mem       = res_mem("expert_zs"),
        partition = res_partition("expert_zs"),
        slurm     = res_slurm("expert_zs"),
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS={threads}
        export CUDA_VISIBLE_DEVICES=""
        {params.python} {params.src}/run_expert_logits.py \
            --model-dir {params.model_dir} \
            --query-adata {input.query} --model-adata {input.model_adata} \
            --ctrl {input.ctrl} --fold-dir {params.fold_dir} \
            --splits zs_id zs_ood \
            --class-list {input.classes} --padded-anchors {input.atoms} \
            --no-model-adata \
            --allow-leak \
            --out-inpanel {params.out_in} --out-fullpanel {params.out_fp} \
            --expert-repo {params.repo} --head {params.head}
        """


# ---------------------------------------------------------------------------
GENEPERT_ZS_METHODS = ["genepert"] + [m for m in padding_atoms_methods() if m.startswith("genepert")]


rule genepert_zs:
    """Ridge on gene embeddings: fit on the TRAINING space's per-pert means (streamed
    from the uncapped id_train), predict every zs class from its embedding, score both
    arms. `{gpm}` is `genepert` (text embeddings) or `genepert_atoms` (ExPert atoms)."""
    input:
        train     = lambda wc: pad_train_idx(wc.gpm),
        zs_id     = f"{PAD_FOLDDIR}/zs_id.txt.gz",
        zs_ood    = f"{PAD_FOLDDIR}/zs_ood.txt.gz",
        classes   = f"{PAD_DIR}/classes.txt",
        source    = lambda wc: source_h5ad(padding_train_size()),
        eval_src  = lambda wc: source_h5ad(PAD_SIZE),
        ctrl      = config["ctrl_h5ad"],
        emb       = lambda wc: padding_embeddings_for(wc.gpm),
    output:
        zs_id  = PAD_RESULTS + "/{gpm}/perturb-seq/" + f"{PAD_SIZE}/{PAD_FOLD}" + "/zs_id/logits.h5",
        zs_ood = PAD_RESULTS + "/{gpm}/perturb-seq/" + f"{PAD_SIZE}/{PAD_FOLD}" + "/zs_ood/logits.h5",
    wildcard_constraints:
        gpm = r"genepert|genepert_atoms",
    params:
        gene_panel   = gene_panel(),
        srcdir       = os.path.join(REPO, "workflow", "methods", "genepert"),
        results_root = PAD_RESULTS,
        fold_idx     = int(PAD_FOLD.split("_")[1]),
        train_splits = padding_train_dir(),
        eval_splits  = PAD_FOLDDIR,
        size         = PAD_SIZE,
        train_index  = lambda wc: padding_train_index(wc.gpm),
    priority: PAD_PRIORITY
    threads: lambda wc: res_threads("genepert_zs")
    resources:
        time      = res_time("genepert_zs"),
        mem       = res_mem("genepert_zs"),
        partition = res_partition("genepert_zs"),
        slurm     = res_slurm("genepert_zs"),
    conda:
        "../envs/genepert.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/scripts/run_genepert.py \
            --size {params.size} --fold {params.fold_idx} \
            --method-dir {wildcards.gpm} \
            --source {input.source} --ctrl {input.ctrl} \
            --embeddings {input.emb} \
            --splits-dir {params.train_splits} --train-index {params.train_index} \
            --eval-source {input.eval_src} --eval-splits-dir {params.eval_splits} \
            --splits zs_id zs_ood \
            --predict-perts {input.classes} \
            --results-root {params.results_root} \
            --gene-panel {params.gene_panel}
        """


# ---------------------------------------------------------------------------
# (no SCOUTER root constant: the scouter arms are parametrised by {spm}, so their
# paths are built per-wildcard below. GEARS keeps a constant because it has no
# atoms arm -- its perturbation representation is a GO graph node, not an
# embedding, so there is nothing for ExPert's atom table to substitute for.)


SCOUTER_ZS_METHODS = ["scouter"] + [m for m in padding_atoms_methods() if m.startswith("scouter")]


rule scouter_zs_train_pds:
    """Train dataset D's scouter model on the TRAINING space's cells, then emit D's
    predicted fingerprints for EVERY class in classes.txt. The perturbation
    representation is a frozen lookup (`nn.Parameter(..., requires_grad=False)`), so
    unseen classes need a table row, not a retrain."""
    input:
        train   = lambda wc: pad_train_idx("scouter"),
        val     = lambda wc: os.path.join(padding_train_dir(), "id_val_cap25.txt.gz"),
        classes = f"{PAD_DIR}/classes.txt",
        source  = lambda wc: source_h5ad(padding_train_size()),
        config  = "workflow/methods/scouter/configs/tasks/perturb-seq/full.yaml",
        emb     = lambda wc: ([padding_atoms_pkl()] if wc.spm.endswith("_atoms") else []),
    output:
        fps = PAD_RESULTS + "/{spm}/perturb-seq/" + f"{PAD_SIZE}/{PAD_FOLD}" + "/per_dataset/{dataset}/fingerprints.npz",
    wildcard_constraints:
        spm = r"scouter|scouter_atoms",
    params:
        srcdir     = "workflow/methods/scouter/scripts",
        # train.py writes fingerprints.npz DIRECTLY into --outdir (it does not append
        # per_dataset/<dataset> itself), so the segment must be here -- exactly as the
        # main-grid rule does it. Without it the job runs to completion and then fails
        # on MissingOutputException.
        outdir     = lambda wc: (f"{PAD_RESULTS}/{wc.spm}/perturb-seq/{PAD_SIZE}/{PAD_FOLD}"
                                 f"/per_dataset/{wc.dataset}"),
        splits     = padding_train_dir(),
        gene_panel = gene_panel(),
        # scouter's own table is a GenePT PICKLE (set in its task config); the atoms arm
        # swaps in ExPert's exported pickle. Both are {gene: vector} dicts.
        emb_arg    = lambda wc: (f"--embedding-path {padding_atoms_pkl()}"
                                 if wc.spm.endswith("_atoms") else ""),
    priority: PAD_PRIORITY
    threads: lambda wc: res_threads("scouter_zs_train_pds")
    resources:
        time      = res_time("scouter_zs_train_pds"),
        mem       = res_mem("scouter_zs_train_pds"),
        partition = res_partition("scouter_zs_train_pds"),
        slurm     = res_slurm("scouter_zs_train_pds"),
    conda:
        "../envs/scouter.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/train.py \
            --splits_dir {params.splits} \
            --config {input.config} \
            --outdir {params.outdir} \
            --dataset {wildcards.dataset} \
            --gene-panel {params.gene_panel} \
            --ctrl-h5ad {config[ctrl_h5ad]} \
            --genept-path {config[scouter_gene_embeddings]} \
            {params.emb_arg} \
            --predict-perts {input.classes}
        """


def scouter_zs_fps(wc):
    return [f"{PAD_RESULTS}/{wc.spm}/perturb-seq/{PAD_SIZE}/{PAD_FOLD}/per_dataset/{d}/fingerprints.npz"
            for d in padding_datasets()]


rule scouter_zs_eval:
    """Dataset-blind ensemble over the zero-shot cells. Same script as the main grid;
    only --source / --splits_dir point at the padding space."""
    input:
        fps     = scouter_zs_fps,
        source  = lambda wc: source_h5ad(PAD_SIZE),
        ctrl    = config["ctrl_h5ad"],
        idx     = PAD_FOLDDIR + "/{split}.txt.gz",
        classes = f"{PAD_DIR}/classes.txt",
    output:
        logits = PAD_RESULTS + "/{spm}/perturb-seq/" + f"{PAD_SIZE}/{PAD_FOLD}" + "/{split}/logits.h5",
    wildcard_constraints:
        split = r"zs_id|zs_ood",
        spm   = r"scouter|scouter_atoms",
    params:
        srcdir = "workflow/methods/scouter/scripts",
        froot  = lambda wc: f"{PAD_RESULTS}/{wc.spm}/perturb-seq/{PAD_SIZE}/{PAD_FOLD}",
        splits = PAD_FOLDDIR,
    priority: PAD_PRIORITY
    threads: res_threads("scouter_zs_eval")
    resources:
        time      = res_time("scouter_zs_eval"),
        mem       = res_mem("scouter_zs_eval"),
        partition = res_partition("scouter_zs_eval"),
        slurm     = res_slurm("scouter_zs_eval"),
    conda:
        "../envs/scouter.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/scouter_ensemble.py \
            --fold_root {params.froot} --source {input.source} --ctrl {input.ctrl} \
            --splits_dir {params.splits} --split {wildcards.split} \
            --classes {input.classes} --out {output.logits}
        """


# ---------------------------------------------------------------------------
GEARS_ZS_ROOT = f"{PAD_RESULTS}/gears/perturb-seq"
GEARS_ZS_PD   = GEARS_ZS_ROOT + f"/{PAD_SIZE}/_pertdata_pds/" + "{dataset}"


rule gears_zs_geneset:
    """The custom perturbation-graph gene set. GEARS' default graph
    (essential_all_data_pert_genes.pkl, 9,976 genes) covers only 961/10,245 zs
    perturbations; (train_pool | zs) & gene2go is 18,595 genes and covers 9,709.
    Passing it via gene_set_path is what makes GEARS able to express an unseen pert
    at all — and is also why GEARS must be RETRAINED (num_perts sizes nn.Embedding)."""
    input:
        classes = f"{PAD_DIR}/classes.txt",
        meta    = f"{PAD_DIR}/metadata.json",
        trperts = f"{PAD_DIR}/train_perts.txt",
        # INPUT, not params: GEARS normally downloads gene2go_all.pkl into each
        # PertData dir at runtime, so the vendored tree does not ship it. Declaring
        # it makes a missing copy a DAG-build error a dry run catches, instead of a
        # FileNotFoundError two minutes into the job.
        gene2go = config.get("gears_gene2go",
                             os.path.join(REPO, "workflow", "methods", "gears",
                                          "GEARS", "saved_data", "gene2go_all.pkl")),
    output:
        geneset = f"{PAD_DIR}/gears_pert_geneset.pkl",
    params:
        srcdir  = os.path.join(REPO, "workflow", "methods", "gears"),
    priority: PAD_PRIORITY
    threads: res_threads("gears_zs_geneset")
    resources:
        time      = res_time("gears_zs_geneset"),
        mem       = res_mem("gears_zs_geneset"),
        partition = res_partition("gears_zs_geneset"),
        slurm     = res_slurm("gears_zs_geneset"),
    conda:
        "../envs/gears.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {PAD_SCRIPTS}/make_gears_geneset.py \
            --classes {input.classes} --meta {input.meta} \
            --train-perts {input.trperts} \
            --gene2go {input.gene2go} --out {output.geneset}
        """


rule gears_zs_prep_pds:
    """Build dataset D's PyG graph cache with the WIDENED pert graph."""
    input:
        source  = lambda wc: source_h5ad(padding_train_size()),
        ctrl    = config["ctrl_h5ad"],
        geneset = f"{PAD_DIR}/gears_pert_geneset.pkl",
        gidx    = lambda wc: pad_graph_indices(),
    output:
        cg = GEARS_ZS_PD + f"/{PAD_SIZE}/data_pyg/cell_graphs.pkl",
    params:
        srcdir     = os.path.join(REPO, "workflow", "methods", "gears"),
        pertdata   = GEARS_ZS_PD,
        gene_panel = gene_panel(),
        size       = PAD_SIZE,
        train_idx  = lambda wc: pad_train_idx("gears"),
    priority: PAD_PRIORITY
    threads: res_threads("gears_zs_prep_pds")
    resources:
        time      = res_time("gears_zs_prep_pds"),
        mem       = res_mem("gears_zs_prep_pds"),
        partition = res_partition("gears_zs_prep_pds"),
        slurm     = res_slurm("gears_zs_prep_pds"),
    conda:
        "../envs/gears.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/run_GEARS.py --stage prep \
            --size {params.size} --dataset {wildcards.dataset} \
            --gene-panel {params.gene_panel} \
            --source {input.source} --ctrl {input.ctrl} \
            --pert-gene-set {input.geneset} --zs-perts \
            --graph-indices {input.gidx} \
            --pertdata-dir {params.pertdata}
        """


rule gears_zs_train_pds:
    """Train D's GNN on the widened graph and emit fingerprints for every class."""
    input:
        cg      = GEARS_ZS_PD + f"/{PAD_SIZE}/data_pyg/cell_graphs.pkl",
        train   = lambda wc: pad_train_idx("gears"),
        idval   = lambda wc: os.path.join(padding_train_dir(), "id_val_cap25.txt.gz"),
        classes = f"{PAD_DIR}/classes.txt",
        geneset = f"{PAD_DIR}/gears_pert_geneset.pkl",
        source  = lambda wc: source_h5ad(padding_train_size()),
        ctrl    = config["ctrl_h5ad"],
    output:
        fps = GEARS_ZS_ROOT + f"/{PAD_SIZE}/{PAD_FOLD}" + "/per_dataset/{dataset}/fingerprints.npz",
    params:
        srcdir       = os.path.join(REPO, "workflow", "methods", "gears"),
        results_root = PAD_RESULTS,
        pertdata     = GEARS_ZS_PD,
        gene_panel   = gene_panel(),
        size         = PAD_SIZE,
        fold_idx     = int(PAD_FOLD.split("_")[1]),
        splits       = padding_train_dir(),
        train_index  = padding_train_index("gears"),
        gidx         = " ".join(pad_graph_indices()),
    priority: PAD_PRIORITY
    threads: res_threads("gears_zs_train_pds")
    resources:
        time      = res_time("gears_zs_train_pds"),
        mem       = res_mem("gears_zs_train_pds"),
        partition = res_partition("gears_zs_train_pds"),
        slurm     = res_slurm("gears_zs_train_pds"),
    conda:
        "../envs/gears.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/run_GEARS.py --stage train \
            --fold {params.fold_idx} --size {params.size} --dataset {wildcards.dataset} \
            --gene-panel {params.gene_panel} \
            --source {input.source} --ctrl {input.ctrl} \
            --splits-dir {params.splits} --train-index {params.train_index} \
            --results-root {params.results_root} \
            --pert-gene-set {input.geneset} --zs-perts \
            --predict-perts {input.classes} \
            --graph-indices {params.gidx} \
            --pertdata-dir {params.pertdata}
        """


def gears_zs_fps(wc):
    return [f"{GEARS_ZS_ROOT}/{PAD_SIZE}/{PAD_FOLD}/per_dataset/{d}/fingerprints.npz"
            for d in padding_datasets()]


rule gears_zs_eval:
    input:
        fps     = gears_zs_fps,
        source  = lambda wc: source_h5ad(PAD_SIZE),
        ctrl    = config["ctrl_h5ad"],
        idx     = PAD_FOLDDIR + "/{split}.txt.gz",
        classes = f"{PAD_DIR}/classes.txt",
    output:
        logits = GEARS_ZS_ROOT + f"/{PAD_SIZE}/{PAD_FOLD}" + "/{split}/logits.h5",
    wildcard_constraints:
        split = r"zs_id|zs_ood",
    params:
        srcdir = os.path.join(REPO, "workflow", "methods", "gears"),
        froot  = GEARS_ZS_ROOT + f"/{PAD_SIZE}/{PAD_FOLD}",
        splits = PAD_FOLDDIR,
    priority: PAD_PRIORITY
    threads: res_threads("gears_zs_eval")
    resources:
        time      = res_time("gears_zs_eval"),
        mem       = res_mem("gears_zs_eval"),
        partition = res_partition("gears_zs_eval"),
        slurm     = res_slurm("gears_zs_eval"),
    conda:
        "../envs/gears.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/gears_ensemble.py \
            --fold_root {params.froot} --source {input.source} --ctrl {input.ctrl} \
            --splits_dir {params.splits} --split {wildcards.split} \
            --classes {input.classes} --out {output.logits}
        """


# ---------------------------------------------------------------------------
rule baselines_zs:
    """The floors that need NO training on the held-out perturbation: `random` and
    `neg_target` (the targeted transcript should go down — needs only the eval cells
    and the control means). Both read the zero-shot store alone."""
    input:
        zs_id   = f"{PAD_FOLDDIR}/zs_id.txt.gz",
        zs_ood  = f"{PAD_FOLDDIR}/zs_ood.txt.gz",
        classes = f"{PAD_DIR}/classes.txt",
        meta    = f"{PAD_DIR}/metadata.json",
        ctrl    = config["ctrl_h5ad"],
    output:
        expand(PAD_RESULTS + "/baseline_{variant}/output/" + f"{PAD_SIZE}/{PAD_FOLD}" + "/{split}/logits.h5",
               variant=padding_baseline_variants(), split=padding_splits("baselines")),
    params:
        splits   = PADDING_SPLITS_ROOT,
        out_root = PAD_RESULTS,
        variants = " ".join(padding_baseline_variants()),
        size     = PAD_SIZE,
        fold     = PAD_FOLD,
        evsplits = " ".join(padding_splits("baselines")),
    priority: PAD_PRIORITY
    threads: res_threads("baselines_zs")
    resources:
        time      = res_time("baselines_zs"),
        mem       = res_mem("baselines_zs"),
        partition = res_partition("baselines_zs"),
        slurm     = res_slurm("baselines_zs"),
    conda:
        "../envs/baselines.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python workflow/methods/baselines/run.py \
            --splits-dir {params.splits} \
            --ctrl-h5ad {input.ctrl} \
            --size {params.size} --fold {params.fold} \
            --out-root {params.out_root} \
            --baselines {params.variants} \
            --eval-splits {params.evsplits} \
            --classes {input.classes} \
            --label-col perturbation --context-col dataset
        """


rule emb_nn_zs:
    """Embedding-nearest-seen-perturbation transfer — the control the padding claim
    rests on. For an unseen pert p, score cells with the delta-centroid of the SEEN
    perturbations nearest to p in ExPert's OWN embedding space, so prior knowledge is
    held fixed and only the model varies. Centroids stream from the uncapped
    id_train, so this floor sees exactly the cells ExPert trained on."""
    input:
        train   = lambda wc: pad_train_idx("emb_nn"),
        zs_id   = f"{PAD_FOLDDIR}/zs_id.txt.gz",
        zs_ood  = f"{PAD_FOLDDIR}/zs_ood.txt.gz",
        classes = f"{PAD_DIR}/classes.txt",
        source  = lambda wc: source_h5ad(padding_train_size()),
        evsrc   = lambda wc: source_h5ad(PAD_SIZE),
        ctrl    = config["ctrl_h5ad"],
        emb     = config["genepert_gene_embeddings"],
    output:
        expand(PAD_RESULTS + "/emb_nn_{variant}/output/" + f"{PAD_SIZE}/{PAD_FOLD}" + "/{split}/logits.h5",
               variant=padding_emb_nn_variants(), split=padding_splits("emb_nn")),
    params:
        out_root = PAD_RESULTS,
        size     = PAD_SIZE,
        fold     = PAD_FOLD,
        variants = " ".join(padding_emb_nn_variants()),
        evsplits = " ".join(padding_splits("emb_nn")),
        train_splits = padding_train_dir(),
        eval_splits  = PAD_FOLDDIR,
        k        = int((PADDING.get("emb_nn") or {}).get("k", 10)),
        tau      = float((PADDING.get("emb_nn") or {}).get("tau", 0.05)),
    priority: PAD_PRIORITY
    threads: res_threads("emb_nn_zs")
    resources:
        time      = res_time("emb_nn_zs"),
        mem       = res_mem("emb_nn_zs"),
        partition = res_partition("emb_nn_zs"),
        slurm     = res_slurm("emb_nn_zs"),
    conda:
        "../envs/baselines.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {PAD_SCRIPTS}/run_emb_nn.py \
            --source {input.source} --train-splits-dir {params.train_splits} \
            --train-index $(basename {input.train} .txt.gz) \
            --eval-source {input.evsrc} --eval-splits-dir {params.eval_splits} \
            --eval-splits {params.evsplits} \
            --ctrl {input.ctrl} --embeddings {input.emb} \
            --classes {input.classes} \
            --variants {params.variants} --k {params.k} --tau {params.tau} \
            --out-root {params.out_root} --size {params.size} --fold {params.fold}
        """
