# ExPert: score an ALREADY-TRAINED model per fold over the benchmark's query cells
# and export logits.h5. No training here -- the models live in
# config['expert_models_root'] and were trained on the `full` splits.
#
# Two artifacts per (fold, split), from ONE prediction pass:
#   in-panel   -> <results>/ExPert/perturb-seq/<size>/<fold>/<split>/logits.h5
#                 restricted to the benchmark's own classes; this is the row that
#                 goes in the main table, apples-to-apples with every other method.
#   full-panel -> <expert_fullpanel_root>/ExPert/perturb-seq/... , scored against
#                 the model's ENTIRE trained vocabulary (~9k classes). Deliberately
#                 in a SEPARATE root: collect_results.py takes the global union of
#                 class names across methods and pads the rest with -inf, so a 9k
#                 file in the shared root would silently re-score every other
#                 method over 9k classes and deflate their macro-F1.
#
# Folds whose OOD datasets no model held out are SKIPPED (see expert_fold_map).

EXPERT_SRC = os.path.join(REPO, "workflow", "methods", "expert")
EXPERT_FOLD = f"{RESULTS}/ExPert/perturb-seq/{{size}}/{{fold}}"


def _expert_model_split_dir(size, fold):
    """The MODEL's own fold dir (full splits) -- its id_train is the leak list."""
    mf = expert_fold_map(size).get(fold)
    if mf is None:
        return []
    return [os.path.join(SPLITS_ROOT, config.get("expert_model_split_size", "full"),
                         mf, "id_train.txt.gz")]


rule expert_logits:
    """Query the benchmark's cells through fold `fold`'s trained ExPert model."""
    input:
        idtest = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_test.txt.gz"),
        ood    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "ood_test.txt.gz"),
        query  = lambda wc: source_h5ad(wc.size),
        # only its obs NAMES are read (--no-model-adata), to map the leak split
        model_adata = lambda wc: source_h5ad(config.get("expert_model_split_size", "full")),
        ctrl   = config["ctrl_h5ad"],
        leak   = lambda wc: _expert_model_split_dir(wc.size, wc.fold),
        weights = lambda wc: expert_require_weights(wc.size, wc.fold),
    output:
        idtest = EXPERT_FOLD + "/id_test/logits.h5",
        ood    = EXPERT_FOLD + "/ood_test/logits.h5",
        fp_idtest = expert_fullpanel_root() + "/ExPert/perturb-seq/{size}/{fold}/id_test/logits.h5",
        fp_ood    = expert_fullpanel_root() + "/ExPert/perturb-seq/{size}/{fold}/ood_test/logits.h5",
    params:
        model_dir = lambda wc: os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(expert_require_weights(wc.size, wc.fold))))),
        fold_dir  = lambda wc: splits_dir(wc.size, wc.fold),
        out_in    = lambda wc: f"{RESULTS}/ExPert/perturb-seq/{wc.size}/{wc.fold}",
        out_fp    = lambda wc: f"{expert_fullpanel_root()}/ExPert/perturb-seq/{wc.size}/{wc.fold}",
        head      = config.get("expert_head", "A"),
        repo      = config["expert_repo"],
        python    = config["expert_python"],
        src       = EXPERT_SRC,
    threads: lambda wc: res_threads("expert_logits", wc.size)
    resources:
        time      = lambda wc: res_time("expert_logits", wc.size),
        mem       = lambda wc: res_mem("expert_logits", wc.size),
        partition = lambda wc: res_partition("expert_logits", wc.size),
        slurm     = lambda wc: res_slurm("expert_logits", wc.size),
    shell:
        r"""
        # No `conda:` -- ExPert needs its own env (config expert_python), which is
        # not one of this repo's snakemake-managed envs.
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS={threads}
        export CUDA_VISIBLE_DEVICES=""
        {params.python} {params.src}/run_expert_logits.py \
            --model-dir {params.model_dir} \
            --query-adata {input.query} --model-adata {input.model_adata} \
            --ctrl {input.ctrl} --fold-dir {params.fold_dir} \
            --leak-split {input.leak} \
            --out-inpanel {params.out_in} --out-fullpanel {params.out_fp} \
            --expert-repo {params.repo} --head {params.head} \
            --no-model-adata
        """
