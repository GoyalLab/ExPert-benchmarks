# scGPT — GPU train, then per-perturbation-chunk eval, then concat. Eval is the
# scaling bottleneck (a model forward per candidate perturbation, O(n_perts) — at
# 9k perts that's days serially), so it's split into chunks that run in parallel:
#
#   scgpt_train(size, fold)              GPU — fine-tune + cache best_model.pt
#   scgpt_eval_chunk(size, fold, split, i)  GPU — load the cached model, predict
#                                        pert-chunk i of <split> → partial logits.h5
#   scgpt_concat(size, fold, split)      CPU — hstack the chunks' pert columns →
#                                        the final <split>/logits.h5 collect reads
#
# Chunk count is config-fixed per size (config["scgpt_chunks"]) → plain expand().

SCGPT_SRC = os.path.join(REPO, "workflow", "methods", "scgpt")
SCGPT_CFG = "workflow/methods/scgpt/configs/tasks/perturb-seq/{size}.yaml"
SCGPT_FOLD = f"{RESULTS}/scgpt/perturb-seq/{{size}}/{{fold}}"


def scgpt_splits_dir(size, fold):
    """Like splits_dir(), but honours config["scgpt_splits_root"].

    scGPT cannot express a perturbation whose target gene is missing from the panel,
    so it trains and evaluates on a PADDED copy of the store, reached through a shadow
    split set whose metadata.json source_path names it (run_scgpt.sh reads --data-path
    from there). Only scGPT, and only for sizes listed in that config block; anything
    else falls through to the canonical splits unchanged.
    """
    root = (config.get("scgpt_splits_root") or {}).get(size)
    return os.path.join(root, size, fold) if root else splits_dir(size, fold)



rule scgpt_train:
    """Fine-tune scGPT for the fold and cache the model (no eval)."""
    input:
        train  = lambda wc: os.path.join(scgpt_splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        idtest = lambda wc: os.path.join(scgpt_splits_dir(wc.size, wc.fold), "id_test.txt.gz"),
        ood    = lambda wc: os.path.join(scgpt_splits_dir(wc.size, wc.fold), "ood_test.txt.gz"),
        config = SCGPT_CFG,
    output:
        model = SCGPT_FOLD + "/model/best_model.pt",
    params:
        splits    = lambda wc: scgpt_splits_dir(wc.size, wc.fold),
        out       = SCGPT_FOLD,
        model_dir = SCGPT_FOLD + "/model",
    threads: lambda wc: res_threads("scgpt_train", wc.size)
    resources:
        time      = lambda wc: res_time("scgpt_train", wc.size),
        mem       = lambda wc: res_mem("scgpt_train", wc.size),
        partition = lambda wc: res_partition("scgpt_train", wc.size),
        slurm     = lambda wc: res_slurm("scgpt_train", wc.size),
    conda:
        "../envs/scgpt.yaml"
    shell:
        r"""
        export SCGPT_CTRL_H5AD="{config[ctrl_h5ad]}" SCGPT_PRETRAINED="{config[scgpt_pretrained_model]}"
        bash workflow/methods/scgpt/scripts/run_scgpt.sh \
            {params.splits} {input.config} {params.out} {params.model_dir} train
        """


rule scgpt_eval_chunk:
    """Predict one pert-chunk of one split against the cached model → partial logits."""
    input:
        model = SCGPT_FOLD + "/model/best_model.pt",
        idx   = lambda wc: os.path.join(scgpt_splits_dir(wc.size, wc.fold), f"{wc.split}.txt.gz"),
        config = SCGPT_CFG,
    output:
        logits = SCGPT_FOLD + "/_eval_chunks/{split}/chunk_{i}/{split}/logits.h5",
    wildcard_constraints:
        split = r"id_test|ood_test",
        i     = r"\d+",
    params:
        splits    = lambda wc: scgpt_splits_dir(wc.size, wc.fold),
        out       = SCGPT_FOLD + "/_eval_chunks/{split}/chunk_{i}",
        model_dir = SCGPT_FOLD + "/model",
        n_chunks  = lambda wc: scgpt_chunks(wc.size),
    threads: lambda wc: res_threads("scgpt_eval_chunk", wc.size)
    resources:
        time      = lambda wc: res_time("scgpt_eval_chunk", wc.size),
        mem       = lambda wc: res_mem("scgpt_eval_chunk", wc.size),
        partition = lambda wc: res_partition("scgpt_eval_chunk", wc.size),
        slurm     = lambda wc: res_slurm("scgpt_eval_chunk", wc.size),
    conda:
        "../envs/scgpt.yaml"
    shell:
        r"""
        export SCGPT_CTRL_H5AD="{config[ctrl_h5ad]}" SCGPT_PRETRAINED="{config[scgpt_pretrained_model]}"
        bash workflow/methods/scgpt/scripts/run_scgpt.sh \
            {params.splits} {input.config} {params.out} {params.model_dir} \
            eval {wildcards.i} {params.n_chunks} {wildcards.split}
        """


rule scgpt_concat:
    """hstack the per-pert-chunk partial logits into the final <split>/logits.h5."""
    input:
        chunks = lambda wc: [
            f"{RESULTS}/scgpt/perturb-seq/{wc.size}/{wc.fold}/_eval_chunks/{wc.split}/chunk_{i}/{wc.split}/logits.h5"
            for i in range(scgpt_chunks(wc.size))
        ],
    output:
        logits = SCGPT_FOLD + "/{split}/logits.h5",
    wildcard_constraints:
        split = r"id_test|ood_test",
    params:
        srcdir = SCGPT_SRC,
    threads: lambda wc: res_threads("scgpt_concat", wc.size)
    resources:
        time      = lambda wc: res_time("scgpt_concat", wc.size),
        mem       = lambda wc: res_mem("scgpt_concat", wc.size),
        partition = lambda wc: res_partition("scgpt_concat", wc.size),
        slurm     = lambda wc: res_slurm("scgpt_concat", wc.size),
    conda:
        "../envs/scgpt.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        python {params.srcdir}/scripts/scgpt_concat_logits.py \
            --out {output.logits} --chunks {input.chunks}
        """
