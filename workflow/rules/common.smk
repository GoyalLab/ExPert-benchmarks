import os
import glob
import json
import math
import yaml

# workflow.basedir is .../ExPert-benchmarks/workflow ; REPO is the repo root.
REPO = os.path.dirname(workflow.basedir)
RESULTS = config["results_root"]
SPLITS_ROOT = config["splits_root"]

# ---- Padding mode (unseen-perturbation benchmark; docs/padding_scope.md) ----
# A SEPARATE benchmark "size" whose eval cells come from the zero-shot store
# (perturbations disjoint from every model's training pool) while TRAINING still
# reads the normal `full` split space. It is deliberately NOT in config['sizes'],
# so `rule all` and every other size are untouched: nothing below changes
# behaviour unless size == PADDING_SIZE.
PADDING = config.get("padding") or {}
PADDING_SIZE = str(PADDING.get("size", "__no_padding__"))
# Pipeline-OWNED split space (never the upstream store dir, which stays read-only).
PADDING_SPLITS_ROOT = PADDING.get("splits_root") or (RESULTS.rstrip("/") + "_zs_splits")


def is_padding(size):
    return bool(PADDING) and str(size) == PADDING_SIZE


def splits_root_for(size):
    """Where <size>/<fold>/*.txt.gz live. Padding sizes are pipeline-generated."""
    return PADDING_SPLITS_ROOT if is_padding(size) else SPLITS_ROOT


PADDING_RESULTS = PADDING.get("results_root") or (RESULTS.rstrip("/") + "_zs")


def results_root_for(size):
    """Where a size's logits live. Padding has its own root so its paths cannot
    collide with the main grid's {size}-wildcard rules (and its 8GB logits stay out
    of the collected tree)."""
    return PADDING_RESULTS if is_padding(size) else RESULTS


def padding_fold():
    return str(PADDING.get("fold", "fold_4"))


def padding_train_size():
    """The split space the padding mode TRAINS on (real cells, real perts)."""
    return str(PADDING.get("train_size", "full"))


def padding_train_dir():
    return os.path.join(SPLITS_ROOT, padding_train_size(), padding_fold())


def padding_train_index(method):
    """Per-method training index basename — the point of scope §2: methods that
    only need per-pert means stream the UNCAPPED id_train; cell-level methods that
    cannot ingest 15.7M cells get a capped index. Recorded, not hidden."""
    ti = PADDING.get("train_index") or {}
    return str(ti.get(method, ti.get("default", "id_train")))


def padding_classes():
    """Canonical class space for the padding mode. Fixed on disk, never unioned
    across methods (collect would otherwise reshape every method's macro-F1)."""
    return os.path.join(PADDING_SPLITS_ROOT, PADDING_SIZE, "classes.txt")


def padding_methods():
    """Methods in the padding target. The ExPert-atom ablation arms (scope §12) are
    OPT-IN: they answer a follow-up question ("is the win in the representation?"),
    so they should not gate the headline comparison. Flip
    `padding.include_atoms_arms: true` once the main arms have landed."""
    ms = list(PADDING.get("methods") or [])
    if PADDING.get("include_atoms_arms"):
        ms += [m for m in padding_atoms_methods() if m not in ms]
    return ms


def padding_splits(method):
    ms = PADDING.get("method_splits") or {}
    return list(ms.get(method, PADDING.get("splits") or ["zs_id", "zs_ood"]))


def padding_datasets():
    """The per-dataset models the padding mode trains (= the training fold's
    datasets). Explicit in config so DAG construction does no I/O over the 140GB
    training store; zs_splits cross-checks it against the data."""
    return sorted(PADDING.get("train_datasets") or [])


def padding_baseline_variants():
    return list(PADDING.get("baseline_variants") or ["random", "neg_target"])


def padding_emb_nn_variants():
    return list((PADDING.get("emb_nn") or {}).get("variants") or ["nn1"])


def padding_atoms_pkl():
    return os.path.join(PADDING_SPLITS_ROOT, PADDING_SIZE, "expert_atoms.pkl")


def padding_embeddings_for(method):
    """Which embedding table a padding method consumes: the TEXT table by default,
    or ExPert's exported atom table for the `*_atoms` sharp-test arms (scope §12)."""
    if str(method).endswith("_atoms"):
        return padding_atoms_npz()
    return config["genepert_gene_embeddings"]


def padding_atoms_methods():
    return list((PADDING.get("expert_atoms") or {}).get("methods") or [])


def padding_atoms_npz():
    """ExPert-atom embedding table (scope §12 sharp test), emitted by
    export_expert_atoms.py in the same {embeddings,genes} layout as the text npz."""
    return os.path.join(PADDING_SPLITS_ROOT, PADDING_SIZE, "expert_atoms.npz")

# ---- Per-rule SLURM resources (config/resources.yaml) ----------------------
# The legacy profile maps resources time/mem/partition/threads → sbatch flags;
# account + gres have no mapping key, so we pass them through the profile's
# `resources: slurm="k=v k=v"` escape hatch (see slurm-submit.py).
with open(os.path.join(REPO, "config", "resources.yaml")) as _fh:
    RES = yaml.safe_load(_fh)


def _site_partition(j):
    """Resolve resources.yaml's LOGICAL partition class (cpu, cpu_himem, gpu, ...) to
    this cluster's real partition + account, from the site config:

        slurm:
          partitions:
            cpu:        {partition: <name>, account: <account>}
            gpu_bigmem: {partition: <name>, account: <account>, constraint: <feature>}

    (config/site.example.yaml documents every class.) A class with no site entry is
    passed through as the partition name, with no account -- so a cluster whose
    partitions happen to be called cpu/gpu needs no site config at all. A job-level
    `account:` still wins over the site's."""
    j = dict(j)
    site = (config.get("slurm") or {}).get("partitions") or {}
    cls = j.get("partition")
    m = site.get(cls) or {}
    j["partition"] = m.get("partition", cls)
    for k in ("account", "constraint"):           # job-level values win
        if k not in j and m.get(k):
            j[k] = m[k]
    return j


def _job(name, size=None):
    """Per-rule resources, with optional per-size overrides.

    Base values live in resources.yaml `jobs.<name>`; `jobs_by_size.<size>.<name>`
    overrides individual keys for a heavier size (e.g. `full` → himem/longer time)
    without inflating the cheap sizes. Rules pass `wc.size`; size=None = base."""
    return _site_partition(_job_raw(name, size))


def _job_raw(name, size=None):
    """_job before the site lookup (partition still the logical class)."""
    base = RES["jobs"][name]
    if size is not None:
        override = RES.get("jobs_by_size", {}).get(size, {}).get(name)
        if override:
            base = {**base, **override}
    return base


def res_time(name, size=None):      return _job(name, size)["time"]
def res_mem(name, size=None):       return _job(name, size)["mem"]
def res_threads(name, size=None):   return int(_job(name, size).get("threads", 1))
def res_partition(name, size=None): return _job(name, size)["partition"]


def _job_ds(name, size, dataset):
    """Like _job, plus an optional `by_dataset.<dataset>` layer for rules that
    fan out one job per dataset and whose cost varies wildly by dataset (e.g.
    fp_ens_eval_ds_chunk: the genome-wide screens need ~3x the mem of the small
    ones). Falls back to the size-merged values when the dataset isn't listed."""
    j = dict(_job_raw(name, size))
    by_ds = j.pop("by_dataset", None) or {}
    if dataset in by_ds:
        j.update(by_ds[dataset])          # may name its own partition class
    return _site_partition(j)


def res_mem_ds(name, size, dataset):     return _job_ds(name, size, dataset)["mem"]
def res_threads_ds(name, size, dataset): return int(_job_ds(name, size, dataset).get("threads", 1))
def res_time_ds(name, size, dataset):    return _job_ds(name, size, dataset)["time"]
def res_partition_ds(name, size, dataset): return _job_ds(name, size, dataset)["partition"]


def _slurm_str(j):
    parts = [f"account={j['account']}"] if j.get("account") else []
    if j.get("gpu"):
        parts.append(f"gres=gpu:{int(j['gpu'])}")
    if j.get("constraint"):
        parts.append(f"constraint={j['constraint']}")
    return " ".join(parts)


def res_slurm(name, size=None):
    """`account=… [gres=gpu:N] [constraint=…]` string for the profile's slurm
    passthrough (each k=v becomes an sbatch --k v; see slurm_utils.py)."""
    return _slurm_str(_job(name, size))


# dataset-aware partition/account too, so a rule can send light datasets to a
# different partition class (e.g. `cpu_short`) than the heavy ones (`cpu` big nodes).
def res_slurm_ds(name, size, dataset): return _slurm_str(_job_ds(name, size, dataset))


# ---- Split-data discovery --------------------------------------------------
def folds_for_size(size):
    # Padding: the fold is declared in config, NOT globbed -- its split dir is an
    # OUTPUT of zs_splits, so globbing would yield [] on the first DAG build.
    if is_padding(size):
        return [padding_fold()]
    base = os.path.join(SPLITS_ROOT, size)
    present = sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(base, "fold_*"))
        if os.path.isdir(d)
    )
    # optional `folds_by_size: {full: [fold_0]}` limiter (e.g. run only 1 full fold
    # via a SUBSET overlay) — intersect with what's on disk; unset = all folds.
    want = (config.get("folds_by_size") or {}).get(size)
    return [f for f in present if f in set(want)] if want else present


def splits_dir(size, fold):
    return os.path.join(splits_root_for(size), size, fold)


def gene_panel():
    """Gene space for per-dataset models: 'all' (union panel, default), 'dataset'
    (that dataset's own active panel), or 'shared' (AND across datasets). See the
    `gene_panel` block in config.yaml."""
    gp = str(config.get("gene_panel", "all")).lower()
    if gp not in ("all", "dataset", "shared"):
        raise ValueError(f"gene_panel must be all|dataset|shared, got {gp!r}")
    return gp


def meta_json(size):
    """<splits_root>/<size>/metadata.json — the single source of truth for a split
    set (source_path, folds, index_base)."""
    return os.path.join(splits_root_for(size), size, "metadata.json")


_SOURCE_CACHE = {}


def source_h5ad(size):
    """Absolute path to the perturbed source adata for `size`.

    Read from metadata.json's `source_path`. The rebuilt splits DB no longer keeps
    a source.h5ad inside the split dir (it lives in a content-addressed store), so
    NOTHING may assume <splits_root>/<size>/source.h5ad. Cached per size: this is
    called from many rules' input functions during DAG construction."""
    # Padding: metadata.json is an OUTPUT of zs_splits, so it cannot be read at
    # DAG-build time. Take the eval store straight from config instead.
    if is_padding(size):
        src = PADDING.get("eval_source")
        if not src:
            raise ValueError("config['padding'] needs 'eval_source' (the zero-shot store h5ad)")
        return src
    if size not in _SOURCE_CACHE:
        with open(meta_json(size)) as fh:
            meta = json.load(fh)
        src = meta.get("source_path")
        if not src:
            raise ValueError(f"metadata.json for size={size} has no 'source_path'")
        _SOURCE_CACHE[size] = src
    return _SOURCE_CACHE[size]


# ---- Target paths ----------------------------------------------------------
def fp_chunks(size, kind):
    """fingerprint chunk count for `kind` in {'fit','eval'} at this size."""
    return int(config["fingerprint_chunks"][size][kind])


def scgpt_chunks(size):
    """scGPT per-perturbation eval-chunk count at this size (parallel eval jobs)."""
    return int(config["scgpt_chunks"][size])


def fp_ens_chunks(size):
    """Per-dataset ensemble eval CELL-chunk count. Separate from fp_chunks(eval)
    (the legacy shared path) — the fan-out already parallelises over datasets, so
    it needs far fewer cell-chunks. Falls back to fp_chunks(eval) if unset."""
    cfg = config.get("fingerprint_perdataset_eval_chunks")
    return int(cfg[size]) if cfg else fp_chunks(size, "eval")


def fp_merge_chunks(size):
    """Merged-fingerprint eval CELL-chunk count. One merged dict (not 17), so far
    fewer jobs than the ensemble — but full's millions of cells still need chunking
    to fit a node. small=1 (single pass), full=500."""
    cfg = config.get("fingerprint_merged_eval_chunks")
    return int(cfg[size]) if cfg else 1


def fp_model_dir(size, fold):
    """fingerprint writes everything under <output_dir>/<size>/<fold>/shared_genes
    (data_bn == basename(dirname(fold_dir)) == size; --shared_genes suffix)."""
    return f"{RESULTS}/fingerprint/{size}/{fold}/shared_genes"


# ---- Per-dataset fingerprint (train one model per dataset on its own panel) ---
# Pert counts per (size, fold, dataset) are precomputed (scripts/precompute_
# perdataset_perts.py) so chunk counts can be sized PER DATASET — at full they
# range ~880..8400 perts, so one fixed count would over- or under-chunk badly.
_FP_PDS_PERTS_P = os.path.join(REPO, "workflow", "methods", "fingerprint",
                               "perdataset_pert_counts.json")
FP_PDS_PERTS = (json.load(open(_FP_PDS_PERTS_P))
                if os.path.exists(_FP_PDS_PERTS_P) else {})


def training_datasets(size, fold):
    """Datasets present in this fold's id_train = the per-dataset models to train
    (the pert-count JSON's keys; held-out contexts are already excluded there)."""
    return sorted(FP_PDS_PERTS.get(size, {}).get(fold, {}).keys())


def fp_pds_perts(size, fold, dataset):
    return int(FP_PDS_PERTS.get(size, {}).get(fold, {}).get(dataset, 0))


def fp_pds_chunks(size, fold, dataset, kind):
    """Per-dataset deg/fit chunk count = ceil(perts / target). Sized per dataset so
    an ~8k-pert screen gets ~90 chunks and an ~80-pert dataset gets 1 (no empty
    no-op jobs). Target from config['fingerprint_perdataset_perts_per_chunk'][kind]."""
    n = fp_pds_perts(size, fold, dataset)
    target = int(config.get("fingerprint_perdataset_perts_per_chunk", {}).get(kind, 90))
    return max(1, math.ceil(n / max(1, target)))


def fp_pds_model_dir(size, fold, dataset):
    return f"{RESULTS}/fingerprint/{size}/{fold}/per_dataset_panel/{dataset}"


def cellot_pds_chunks(size, fold, dataset):
    """Per-dataset cellot OT-map chunk count = ceil(dataset_perts / K). cellot is
    per-PERTURBATION, so per-dataset would be ~97k OT maps/fold at full; K perts
    per job (config cellot_perts_per_chunk) keeps the job count sane. Reuses the
    same per-dataset pert-count JSON as fingerprint (identical training perts)."""
    n = fp_pds_perts(size, fold, dataset)
    k = int(config.get("cellot_perts_per_chunk", 100))
    return max(1, math.ceil(n / max(1, k)))


# ---- ExPert: trained FULL-split models scored over the benchmark's cells -----
# The models in config['expert_models_root'] were trained on the `full` splits.
# FOLD NUMBERS DO NOT CORRESPOND between the split spaces -- e.g. small fold_1
# holds out D3+Jurkat while full fold_1 holds out D4+RPE1 -- so pairing by number
# would score a model on datasets it trained on. Pair by HELD-OUT SET instead:
# a model is usable for a benchmark fold only if its training excluded every one
# of that fold's OOD datasets. Exact match preferred, else the smallest superset.
_EXPERT_MAP_CACHE = {}


def _fold_holdouts(split_size):
    with open(os.path.join(SPLITS_ROOT, split_size, "metadata.json")) as fh:
        meta = json.load(fh)
    return {f"fold_{f['fold']}": frozenset(f["held_out_datasets"]) for f in meta["folds"]}


def expert_fold_map(size):
    """benchmark fold -> model fold. Folds with no safe model are ABSENT."""
    # Padding: the zs perturbations are disjoint from EVERY model's training pool,
    # so there is no leakage to guard against -- map the padding fold straight to
    # the same-numbered model fold.
    if is_padding(size):
        return {padding_fold(): padding_fold()}
    if size in _EXPERT_MAP_CACHE:
        return _EXPERT_MAP_CACHE[size]
    model_size = config.get("expert_model_split_size", "full")
    try:
        bench, model = _fold_holdouts(size), _fold_holdouts(model_size)
    except (OSError, KeyError, ValueError):
        _EXPERT_MAP_CACHE[size] = {}
        return {}
    out = {}
    for f, held in bench.items():
        cand = [(len(h - held), mf) for mf, h in model.items() if held <= h]
        if cand:
            out[f] = min(cand)[1]          # exact (0 extra) first, else smallest superset
    _EXPERT_MAP_CACHE[size] = out
    return out


def expert_model_dir(size, fold):
    mf = expert_fold_map(size).get(fold)
    if mf is None:
        return None
    return os.path.join(config["expert_models_root"],
                        config.get("expert_model_pattern", "ExPert_{model_fold}/run_0")
                        .format(model_fold=mf.replace("fold_", "fold")))


def expert_weights(size, fold):
    """model.pt for this benchmark fold, or None if unmapped OR not on disk yet.
    The fold models land progressively as training finishes, so a fold without
    weights is simply not requested rather than breaking DAG construction."""
    d = expert_model_dir(size, fold)
    if d is None:
        return None
    p = os.path.join(d, "config", "ExPert_v0", "model", "model.pt")
    return p if os.path.exists(p) else None


def expert_require_weights(size, fold):
    """Like expert_weights but RAISES. Used by the rule itself so explicitly
    requesting an unusable fold fails at DAG build with a reason, instead of
    silently running with an empty model dir. (method_logits filters these folds
    out of the `all` target, so a normal run never reaches this.)"""
    mf = expert_fold_map(size).get(fold)
    if mf is None:
        raise ValueError(
            f"ExPert: no model is safe for {size}/{fold} -- no `"
            f"{config.get('expert_model_split_size','full')}` fold held out all of its OOD "
            f"datasets, so every candidate model trained on at least one of them.")
    p = expert_weights(size, fold)
    if p is None:
        raise ValueError(
            f"ExPert: {size}/{fold} maps to model fold {mf}, but its weights are not on "
            f"disk yet ({expert_model_dir(size, fold)}/config/ExPert_v0/model/model.pt).")
    return p


def expert_fullpanel_root():
    return config.get("expert_fullpanel_root", RESULTS + "_fullpanel")


def method_logits(method, size, fold, split):
    """logits.h5 path(s) a method emits for one (fold, split). Most methods emit
    one under <method>/perturb-seq/…; `baselines` is one rule that emits four
    baseline_<variant>/output/… files; fingerprint uses its own R layout."""
    ROOT = results_root_for(size)
    if method == "baselines":
        # Padding uses its OWN variant list: only the floors that need no training
        # on the held-out perturbation (random, neg_target). `centroid`/`prior`
        # require observed cells for the class, so they cannot express one zs label.
        variants = padding_baseline_variants() if is_padding(size) else config["baseline_variants"]
        return [
            f"{ROOT}/baseline_{v}/output/{size}/{fold}/{split}/logits.h5"
            for v in variants
        ]
    if method == "emb_nn":
        # The control the padding claim rests on -- one dir per variant so collect
        # scores each independently (same shape as `baselines`).
        return [
            f"{ROOT}/emb_nn_{v}/output/{size}/{fold}/{split}/logits.h5"
            for v in padding_emb_nn_variants()
        ]
    if method == "fingerprint":
        return [f"{fp_model_dir(size, fold)}/{split}/logits.h5"]
    if method == "fingerprint_perdataset":
        # per-dataset ensemble output — a DISTINCT top-level dir so the collector's
        # first-segment method discovery doesn't collide with shared `fingerprint`.
        return [f"{ROOT}/fingerprint_perdataset/{size}/{fold}/{split}/logits.h5"]
    if method == "fingerprint_merged":
        # combined-weights variant (one classify pass) — own top-level dir too.
        return [f"{ROOT}/fingerprint_merged/{size}/{fold}/{split}/logits.h5"]
    if method == "ExPert":
        # only folds with a model whose training excluded this fold's OOD datasets
        if expert_weights(size, fold) is None:
            return []
        return [f"{ROOT}/ExPert/perturb-seq/{size}/{fold}/{split}/logits.h5"]
    return [f"{ROOT}/{method}/perturb-seq/{size}/{fold}/{split}/logits.h5"]


def logits_for_size(size):
    out = []
    # Padding has its OWN method + split lists so that adding zs splits here can
    # never make small/ctx7 ask for zs logits (method_splits is shared).
    if is_padding(size):
        for fold in folds_for_size(size):
            for method in padding_methods():
                for split in padding_splits(method):
                    out.extend(method_logits(method, size, fold, split))
        return out
    for fold in folds_for_size(size):
        for method in config["methods"]:
            for split in config["method_splits"][method]:
                out.extend(method_logits(method, size, fold, split))
    return out


def collect_splits(size):
    """Splits `collect` scores for this size. Padding has its own arms; every other
    size keeps the historical id_test/ood_test pair verbatim."""
    if is_padding(size):
        seen, out = set(), []
        for m in padding_methods():
            for sp in padding_splits(m):
                if sp not in seen:
                    seen.add(sp); out.append(sp)
        return out or ["zs_id", "zs_ood"]
    return list(config.get("collect_splits") or ["id_test", "ood_test"])


def collect_report_args(size):
    """--group-by / --over-random for `collect`, from config['padding']['report'].

    Only the padding size sets these today: its ~9.9k-class space makes per-cell
    numbers unreadable in absolute terms (chance recall@1 = 1.0e-4) and noisy at ~2
    cells per (dataset, perturbation). Other sizes keep the historical output
    verbatim, so nothing about small/ctx7 changes."""
    if not is_padding(size):
        return ""
    rep = (PADDING.get("report") or {})
    args = []
    gb = rep.get("group_by") or []
    if gb:
        args.append("--group-by " + " ".join(str(x) for x in gb))
    if rep.get("over_random"):
        args.append("--over-random")
    return " ".join(args)


def collect_classes(size):
    """A FIXED class space for padding (classes.txt), else '' = let collect take the
    cross-method union as it always has. Fixing it matters here: methods cover
    9.4%-100% of a 10,245-class space, and a unioned space would silently reshape
    every method's macro-F1 (the same hazard that keeps the full-panel ExPert export
    in its own root)."""
    return padding_classes() if is_padding(size) else ""


def metrics_dir(size):
    # Sibling of results/ (not under it) so collect_results.py doesn't scan its
    # own output as a "method".
    return f"metrics/{size}"


wildcard_constraints:
    # ctx<k> = context-breadth tiers (ctx7 = 1,157 perts); zs_f<n> = the padding
    # mode's eval-only size (unseen perturbations, docs/padding_scope.md).
    # production_v010 = the no-holdout production split (single-dataset fingerprint dicts only).
    size   = r"small|medium|full|ctx\d+|zs_f\d+|production_v\d+",
    fold   = r"fold_\d+",
    method = r"scouter|scgpt|cellot|fingerprint|gears|baselines|xgboost|emb_nn",
    # zs_id / zs_ood are the padding arms: unseen perturbation, and unseen
    # perturbation x unseen context.
    split  = r"id_test|ood_test|id_val|zs_id|zs_ood",
