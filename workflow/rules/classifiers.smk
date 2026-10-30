# Supervised discriminative baselines trained directly on cell -> perturbation:
#   xgboost  : XGBClassifier(multi:softprob, hist), early-stopped on id_val
#   linreg   : SGDClassifier(log_loss, OvR) -- logistic regression, parallel
# One job per (size, fold) per method; each writes both eval splits' logits.h5.
# Port of ExPert/expert/benchmarks/xgboost.py -- see run_classifier.py.

CLF_SRC = os.path.join(REPO, "workflow", "methods", "classifiers")


def _clf_args(method):
    """CLI for run_classifier.py from config['classifiers']. Returned as a PARAM so a
    config change (normalisation, hyperparameters) triggers a rerun under
    rerun-triggers: params."""
    c = config.get("classifiers", {}) or {}
    n = c.get("normalisation", {}) or {}
    a = [f"--target_sum {n.get('target_sum', 10000)}"]
    if n.get("log1p", True):
        a.append("--log1p")
    if n.get("ctrl_delta", True):
        a.append(f"--ctrl_delta {config['ctrl_h5ad']}")
    m = c.get(method, {}) or {}
    if method.startswith("xgboost"):
        for k in ("n_estimators", "max_depth", "lr", "early_stopping_rounds",
                  "colsample_bytree", "subsample", "max_bin"):
            if k in m:
                a.append(f"--{k} {m[k]}")
    a.append(f"--max_train_cells {m.get('max_train_cells', 0)}")
    return " ".join(a)


rule xgboost:
    """XGBoost on the fold's id_train, early-stopped on id_val -> both splits' logits.h5."""
    input:
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        val    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_val.txt.gz"),
        idtest = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_test.txt.gz"),
        ood    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "ood_test.txt.gz"),
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
    output:
        idtest = f"{RESULTS}/xgboost/perturb-seq/{{size}}/{{fold}}/id_test/logits.h5",
        ood    = f"{RESULTS}/xgboost/perturb-seq/{{size}}/{{fold}}/ood_test/logits.h5",
    params:
        args   = lambda wc: _clf_args("xgboost"),
        splits = lambda wc: splits_dir(wc.size, wc.fold),
        out    = lambda wc: f"{RESULTS}/xgboost/perturb-seq/{wc.size}/{wc.fold}",
        src    = CLF_SRC,
    threads: lambda wc: res_threads("xgboost", wc.size)
    resources:
        time      = lambda wc: res_time("xgboost", wc.size),
        mem       = lambda wc: res_mem("xgboost", wc.size),
        partition = lambda wc: res_partition("xgboost", wc.size),
        slurm     = lambda wc: res_slurm("xgboost", wc.size),
    conda:
        "../envs/classifiers.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS={threads}
        python {params.src}/run_classifier.py --model xgboost \
            --split_dir {params.splits} --out_dir {params.out} \
            --n_jobs {threads} {params.args}
        """


rule xgboost_capped:
    """Same as `xgboost` but with classifiers.xgboost_capped.max_train_cells.
    Separate output dir so collect scores it as its own method and the capped
    vs uncapped arms are directly comparable on identical eval cells."""
    input:
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        val    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_val.txt.gz"),
        idtest = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_test.txt.gz"),
        ood    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "ood_test.txt.gz"),
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
    output:
        idtest = f"{RESULTS}/xgboost_capped/perturb-seq/{{size}}/{{fold}}/id_test/logits.h5",
        ood    = f"{RESULTS}/xgboost_capped/perturb-seq/{{size}}/{{fold}}/ood_test/logits.h5",
    params:
        args   = lambda wc: _clf_args("xgboost_capped"),
        splits = lambda wc: splits_dir(wc.size, wc.fold),
        out    = lambda wc: f"{RESULTS}/xgboost_capped/perturb-seq/{wc.size}/{wc.fold}",
        src    = CLF_SRC,
    threads: lambda wc: res_threads("xgboost_capped", wc.size)
    resources:
        time      = lambda wc: res_time("xgboost_capped", wc.size),
        mem       = lambda wc: res_mem("xgboost_capped", wc.size),
        partition = lambda wc: res_partition("xgboost_capped", wc.size),
        slurm     = lambda wc: res_slurm("xgboost_capped", wc.size),
    conda:
        "../envs/classifiers.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS={threads}
        python {params.src}/run_classifier.py --model xgboost \
            --split_dir {params.splits} --out_dir {params.out} \
            --n_jobs {threads} {params.args}
        """


rule linreg:
    """Linear (SGD log-loss) classifier on the fold's id_train -> both splits' logits.h5."""
    input:
        train  = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_train.txt.gz"),
        val    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_val.txt.gz"),
        idtest = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "id_test.txt.gz"),
        ood    = lambda wc: os.path.join(splits_dir(wc.size, wc.fold), "ood_test.txt.gz"),
        source = lambda wc: source_h5ad(wc.size),
        ctrl   = config["ctrl_h5ad"],
    output:
        idtest = f"{RESULTS}/linreg/perturb-seq/{{size}}/{{fold}}/id_test/logits.h5",
        ood    = f"{RESULTS}/linreg/perturb-seq/{{size}}/{{fold}}/ood_test/logits.h5",
    params:
        args   = lambda wc: _clf_args("linreg"),
        splits = lambda wc: splits_dir(wc.size, wc.fold),
        out    = lambda wc: f"{RESULTS}/linreg/perturb-seq/{wc.size}/{wc.fold}",
        src    = CLF_SRC,
    threads: lambda wc: res_threads("linreg", wc.size)
    resources:
        time      = lambda wc: res_time("linreg", wc.size),
        mem       = lambda wc: res_mem("linreg", wc.size),
        partition = lambda wc: res_partition("linreg", wc.size),
        slurm     = lambda wc: res_slurm("linreg", wc.size),
    conda:
        "../envs/classifiers.yaml"
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS={threads}
        python {params.src}/run_classifier.py --model linreg \
            --split_dir {params.splits} --out_dir {params.out} \
            --n_jobs {threads} {params.args}
        """
