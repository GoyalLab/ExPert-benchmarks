#!/usr/bin/python3

import anndata
import json
import numpy as np
import pandas as pd
from absl import logging
from scipy import sparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from cellot.models import load_autoencoder_model
from cellot.utils import load_config
from cellot.data.utils import cast_dataset_to_loader
from cellot.utils.helpers import nest_dict


class AnnDataDataset(Dataset):
    def __init__(
        self, adata, obs=None, categories=None, include_index=False, dim_red=None
    ):
        self.adata = adata
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs = obs
        self.categories = categories
        self.include_index = include_index

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        value = self.adata.X[idx]

        if self.obs is not None:
            meta = self.categories.index(self.adata.obs[self.obs].iloc[idx])
            value = value, int(meta)

        if self.include_index:
            return self.adata.obs_names[idx], value

        return value


def read_list(arg):

    if isinstance(arg, str):
        arg = Path(arg)
        assert arg.exists()
        lst = arg.read_text().split()
    else:
        lst = arg

    return list(lst)


def _shared_active_gene_mask(adata, held_out=None):
    """Intersection (AND) mask over per-dataset active-gene masks in varm.

    Expects adata.varm['feature_mask_per_dataset'] (n_genes, n_datasets) and
    adata.uns['dataset_to_mask_idx'] (dataset_name -> column index). If
    held_out is given (str or list of dataset names), each matching dataset's
    column is excluded from the AND so its own zero-padding pattern doesn't
    shrink the training gene set.
    """
    if "feature_mask_per_dataset" not in adata.varm:
        raise KeyError(
            "adata.varm['feature_mask_per_dataset'] is required for "
            "shared_active_genes_only; not found."
        )
    feat_mask = np.asarray(adata.varm["feature_mask_per_dataset"]).astype(bool)
    dataset_to_idx = adata.uns.get("dataset_to_mask_idx", {})

    held_out_list = _as_held_out_list(held_out)
    excluded_idx = {dataset_to_idx[d] for d in held_out_list if d in dataset_to_idx}
    cols = [c for c in range(feat_mask.shape[1]) if c not in excluded_idx]
    mask = feat_mask[:, cols].all(axis=1)
    logging.info(
        f"shared_active_genes_only: keeping {mask.sum()} / {len(mask)} genes "
        f"({len(cols)} datasets in AND; held_out={held_out_list})"
    )
    return mask


def _panel_mode(config):
    """Gene space for a per-dataset model, from config.data.gene_panel.
    'all' (DEFAULT) = no gene restriction: train on the full var axis, which keeps
    every dataset's model in one shared coordinate system (the panels are hard
    zero-padded, so the extra columns are structural zeros, not noise).
    'dataset' = restrict to D's own active panel."""
    return str(config.data.get("gene_panel", "all")).lower()


def _dataset_gene_mask(adata, dataset):
    """Active-gene panel for ONE dataset (varm feature_mask column, via the
    uns dataset->column map). Used by the per-dataset path so each dataset's model
    trains on ITS OWN panel (recovers signal the shared intersection loses),
    instead of the AND across all datasets."""
    if "feature_mask_per_dataset" not in adata.varm:
        raise KeyError("adata.varm['feature_mask_per_dataset'] required for per-dataset mode.")
    feat_mask = np.asarray(adata.varm["feature_mask_per_dataset"]).astype(bool)
    d2i = adata.uns.get("dataset_to_mask_idx", {})
    if dataset not in d2i:
        raise ValueError(f"dataset {dataset!r} not in uns['dataset_to_mask_idx']; "
                         f"have: {list(d2i)[:5]}...")
    mask = feat_mask[:, d2i[dataset]]
    logging.info(f"per-dataset panel '{dataset}': {mask.sum()} / {len(mask)} genes")
    return mask


def _as_held_out_list(value):
    """Normalize singular/list/None held_out spec to a list of dataset names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _ensure_normalized(adata, target_sum=1e4):
    """If X has an integer dtype, library-size normalize + log1p in place.

    Raw counts (e.g. int16 from cellranger) make cellot's ICNN explode; scRNA
    data needs standard normalization before feeding an OT model.
    """
    if not np.issubdtype(adata.X.dtype, np.integer):
        return
    import scanpy as sc
    import scipy.sparse as sp
    logging.info(
        f"Normalizing integer X ({adata.X.dtype}): "
        f"cast→float32 + library-size-normalize(target={target_sum:g}) + log1p"
    )
    # Cast counts to float32 before normalize_total. We avoid `.astype(float32)`
    # on a sparse int16 matrix because scipy routes that through
    # sum_duplicates → sort_indices → csr_sort_indices, whose C impl refuses
    # int16 data and raises "Output dtype not compatible with inputs."
    # Re-wrapping the raw arrays sidesteps that code path entirely.
    if sp.issparse(adata.X):
        X = adata.X
        adata.X = type(X)(
            (X.data.astype(np.float32), X.indices, X.indptr), shape=X.shape,
        )
    else:
        adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def _held_out_context_from_metadata(splits_dir):
    """Auto-detect held-out contexts for a fold from sibling metadata.json.

    Returns (context_col, held_out_list) or (None, None) if unavailable.
    Accepts the current schema (`held_out_datasets`: list of dataset names) and the
    older ones (`held_out_contexts`: list / `held_out_context`: str); all normalize
    to a list.

    NOTE the rebuilt splits DB dropped `context_col` from metadata.json, and its
    per-fold `held_out` is an AXIS spec ({'donor': 'D2', ...}) rather than a dataset
    list — the resolved names are in `held_out_datasets`. The grouping column is
    `dataset` in every source adata, so default to that.
    """
    meta_path = Path(splits_dir).parent / "metadata.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text())
    name = Path(splits_dir).name
    if not name.startswith("fold_"):
        return None, None
    try:
        fold_idx = int(name.split("_")[1])
    except (ValueError, IndexError):
        return None, None
    if fold_idx >= len(meta.get("folds", [])):
        return None, None
    fold = meta["folds"][fold_idx]
    raw = fold.get(
        "held_out_datasets",
        fold.get("held_out_contexts", fold.get("held_out_context")),
    )
    return meta.get("context_col", "dataset"), _as_held_out_list(raw)


def _lazy_load_perturbed(config, train_idx, val_idx):
    """Backed-mode load: only materialize cells we actually need.

    Reads obs from disk to find rows that are (a) in train+val splits AND
    (b) match config.data.target, then loads just those rows into memory.
    Avoids materializing the full source h5ad for large perturbation atlases
    (300GB+). Refuses target='all' since that would defeat the purpose.

    Indices are sorted before the backed read so HDF5 access is sequential
    (much faster than random fancy indexing).
    """
    target = config.data.target
    if target == "all":
        raise ValueError(
            "lazy_load_target=true requires config.data.target to be a "
            "specific perturbation (or list); got 'all'."
        )

    backed = anndata.read(config.data.path, backed="r")
    n = backed.n_obs

    split = np.full(n, "ignore", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "test"

    obs_cond = backed.obs[config.data.condition].values
    targets = target if isinstance(target, list) else [target]
    target_mask = np.isin(obs_cond, targets)

    keep_mask = (split != "ignore") & target_mask
    ds = config.data.get("dataset")                     # per-dataset: this dataset only
    if ds is not None:
        dcol = config.data.get("context_col", "dataset")
        keep_mask &= (backed.obs[dcol].values == ds)
    keep_idx = np.where(keep_mask)[0]  # np.where already returns sorted indices
    if len(keep_idx) == 0:
        raise RuntimeError(
            f"No cells found for target={target} within train+val splits."
        )
    logging.info(
        f"Lazy load: materializing {len(keep_idx)} / {n} cells (target={target})"
    )

    perturbed = backed[keep_idx].to_memory()
    perturbed.obs["split"] = split[keep_idx]
    return perturbed


def _read_controls_only_for_ae(config):
    """Load only controls (for AE pre-training). Excludes held-out-context
    controls to prevent ood leakage into the encoder. Random 80/20 train/test
    split for in-training eval."""
    splits_dir = Path(config.datasplit.path)

    controls = anndata.read(config.data.path_ctrl)
    # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's own
    # control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control), so
    # comparing the label against one value misses 18/22 datasets.
    controls.obs[config.data.condition] = "control"
    _ensure_normalized(controls)
    controls.X = controls.X.astype(np.float32)

    # Same held-out filter the cellot per-pert path applies to controls.
    ctx_col, held_out = _held_out_context_from_metadata(splits_dir)
    # Config override accepts singular `held_out_context` (legacy) or list
    # `held_out_contexts`; both normalize via _as_held_out_list.
    cfg_override = config.data.get("held_out_contexts",
                                   config.data.get("held_out_context"))
    if cfg_override is not None:
        held_out = _as_held_out_list(cfg_override)
        ctx_col = config.data.get("context_col", ctx_col)
    if held_out and ctx_col is not None and ctx_col in controls.obs:
        before = len(controls)
        controls = controls[~controls.obs[ctx_col].isin(held_out)].copy()
        logging.info(
            f"Excluded {before - len(controls)} controls in {held_out} "
            f"from AE training to avoid ood leakage."
        )
    else:
        logging.warn(
            f"No held-out context filter applied to controls "
            f"(ctx_col={ctx_col}, held_out={held_out})."
        )

    # per-dataset mode: AE trains on THIS dataset's controls + its own panel.
    _ds = config.data.get("dataset")
    if _ds is not None:
        _dcol = config.data.get("context_col", ctx_col or "dataset")
        if _dcol in controls.obs:
            controls = controls[controls.obs[_dcol].astype(str) == _ds].copy()
        if _panel_mode(config) == "dataset":
            source = anndata.read(config.data.path, backed="r")
            kept = source.var_names[_dataset_gene_mask(source, _ds)].tolist()
            keep_genes = controls.var_names.intersection(kept)
            controls = controls[:, keep_genes].copy()
        logging.info(f"AE per-dataset '{_ds}': {len(controls)} controls, "
                     f"{controls.n_vars} genes (gene_panel={_panel_mode(config)})")
    # Optional shared-active-genes filter: varm lives on the perturbed source
    # h5ad; read it backed and intersect the resulting gene set onto controls.
    elif config.data.get("shared_active_genes_only", False):
        source = anndata.read(config.data.path, backed="r")
        gene_mask = _shared_active_gene_mask(source, held_out=held_out)
        kept = source.var_names[gene_mask].tolist()
        keep_genes = controls.var_names.intersection(kept)
        controls = controls[:, keep_genes].copy()
        logging.info(
            f"Restricted controls to {len(keep_genes)} shared-active genes."
        )

    # Random 80/20 train/test split for AE in-training eval.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(controls))
    n_test = max(1, int(0.1 * len(controls)))
    split = np.full(len(controls), "train", dtype=object)
    split[perm[:n_test]] = "test"
    controls.obs["split"] = pd.Categorical(split)

    # AE doesn't read `transport`, but stamp condition so any downstream code
    # that *does* (e.g. compute_scgen_shift) sees them as the source class.
    controls.obs[config.data.condition] = config.data.source
    return controls


def _read_with_precomputed_splits(config):
    """Load perturbed + control h5ads and stamp splits from index files.

    Uses positional indices from `{datasplit.path}/id_train.txt.gz` → split="train"
    and `id_val.txt.gz` → split="test" (cellot's in-training eval name). Cells not
    in either set are dropped. Controls from `config.data.path_ctrl` are all marked
    split="train"; pair with `config.training.share_controls=true` so the eval
    iterator reuses them.

    Set `config.data.lazy_load_target=true` to open the perturbed h5ad in
    backed mode and materialize only the cells matching the current target.
    Set `config.data.controls_only=true` (for AE pre-training) to skip the
    perturbed h5ad entirely and use the (held-out-filtered) controls only.
    """
    if config.data.get("controls_only", False):
        return _read_controls_only_for_ae(config)

    splits_dir = Path(config.datasplit.path)
    train_idx = np.loadtxt(splits_dir / "id_train.txt.gz", dtype=int)
    val_idx = np.loadtxt(splits_dir / "id_val.txt.gz", dtype=int)

    if config.data.get("lazy_load_target", False):
        perturbed = _lazy_load_perturbed(config, train_idx, val_idx)
    else:
        perturbed = anndata.read(config.data.path)
        split = pd.Series("ignore", index=perturbed.obs.index, dtype=object)
        split.iloc[train_idx] = "train"
        split.iloc[val_idx] = "test"
        perturbed.obs["split"] = split
        perturbed = perturbed[perturbed.obs["split"] != "ignore"].copy()

    _ensure_normalized(perturbed)
    # Cast to float32 up-front: AnnDataDataset requires it downstream, and it
    # avoids a scipy csr_row_index dtype-compat error when subsetting int16
    # sparse X (e.g. raw counts stored as int16).
    perturbed.X = perturbed.X.astype(np.float32)

    # Optional shared-active-genes filter: drop genes that are zero-padded in
    # any training-contributing dataset. Applied to perturbed first; the
    # `concat(join="inner")` below propagates it to controls via var_names.
    _ds = config.data.get("dataset")
    if _ds is not None:
        # per-dataset mode. gene_panel='dataset' restricts to D's OWN panel; the
        # default 'all' keeps the full var axis so every dataset's model shares one
        # coordinate system (never the cross-dataset AND, which is ~1 gene here).
        if _panel_mode(config) == "dataset":
            perturbed = perturbed[:, _dataset_gene_mask(perturbed, _ds)].copy()
    elif config.data.get("shared_active_genes_only", False):
        _ctx_col, _held_out = _held_out_context_from_metadata(splits_dir)
        _cfg_override = config.data.get("held_out_contexts",
                                        config.data.get("held_out_context"))
        if _cfg_override is not None:
            _held_out = _as_held_out_list(_cfg_override)
        gene_mask = _shared_active_gene_mask(perturbed, held_out=_held_out)
        perturbed = perturbed[:, gene_mask].copy()

    parts = [perturbed]
    if "path_ctrl" in config.data:
        controls = anndata.read(config.data.path_ctrl)
        # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's own
        # control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control), so
        # comparing the label against one value misses 18/22 datasets.
        controls.obs[config.data.condition] = "control"
        _ensure_normalized(controls)
        controls.X = controls.X.astype(np.float32)

        # Exclude controls from the held-out contexts so they don't leak into
        # train/val. Source: sibling metadata.json, else explicit config override.
        ctx_col, held_out = _held_out_context_from_metadata(splits_dir)
        cfg_override = config.data.get("held_out_contexts",
                                       config.data.get("held_out_context"))
        if cfg_override is not None:
            held_out = _as_held_out_list(cfg_override)
            ctx_col = config.data.get("context_col", ctx_col)
        if held_out and ctx_col is not None and ctx_col in controls.obs:
            controls = controls[~controls.obs[ctx_col].isin(held_out)].copy()
        else:
            logging.warn(
                "No held-out context filter applied to controls "
                f"(ctx_col={ctx_col}, held_out={held_out})."
            )
        # per-dataset mode: this dataset's controls only (its own control baseline).
        if _ds is not None:
            _dcol = config.data.get("context_col", "dataset")
            if _dcol in controls.obs:
                controls = controls[controls.obs[_dcol].astype(str) == _ds].copy()

        controls.obs[config.data.condition] = config.data.source
        controls.obs["split"] = "train"
        parts.append(controls)

    data = anndata.concat(parts, join="inner", index_unique="-")
    data.obs["split"] = data.obs["split"].astype("category")

    transport_mapper = {}
    for value in ["source", "target"]:
        key = config.data[value]
        if isinstance(key, list):
            for item in key:
                transport_mapper[item] = value
        else:
            transport_mapper[key] = value
    data.obs["transport"] = data.obs[config.data.condition].apply(transport_mapper.get)
    if config.data["target"] == "all":
        data.obs["transport"].fillna("target", inplace=True)

    data = data[data.obs["transport"].notna()].copy()
    return data


def read_single_anndata(config, path=None):
    if "datasplit" in config and config.datasplit.get("name") == "precomputed":
        return _read_with_precomputed_splits(config)

    if path is None:
        path = config.data.path

    data = anndata.read(path)

    if "features" in config.data:
        features = read_list(config.data.features)
        data = data[:, features].copy()

    # select subgroup of individuals
    if "individuals" in config.data:
        data = data[
            data.obs[config.data.individuals[0]].isin(config.data.individuals[1])
        ]

    # label conditions as source/target distributions
    # config.data.{source,target} can be a list now
    transport_mapper = dict()
    for value in ["source", "target"]:
        key = config.data[value]
        if isinstance(key, list):
            for item in key:
                transport_mapper[item] = value
        else:
            transport_mapper[key] = value

    data.obs["transport"] = data.obs[config.data.condition].apply(transport_mapper.get)

    if config.data["target"] == "all":
        data.obs["transport"].fillna("target", inplace=True)

    mask = data.obs["transport"].notna()
    assert "subset" not in config.data
    if "subset" in config.datasplit:
        for key, value in config.datasplit.subset.items():
            if not isinstance(value, list):
                value = [value]
            mask = mask & data.obs[key].isin(value)

    # write train/test/valid into split column
    data = data[mask].copy()
    if "datasplit" in config:
        data.obs["split"] = split_cell_data(data, **config.datasplit)

    return data


def load_cell_data(
    config,
    data=None,
    split_on=None,
    return_as="loader",
    include_model_kwargs=False,
    pair_batch_on=None,
    **kwargs
):

    if isinstance(return_as, str):
        return_as = [return_as]

    assert set(return_as).issubset({"anndata", "dataset", "loader"})
    config.data.condition = config.data.get("condition", "drug")
    condition = config.data.condition

    if data is None:
        if config.data.type == "cell":
            data = read_single_anndata(config, **kwargs)
        else:
            raise ValueError

    if config.data.get("select") is not None:
        keep = pd.Series(False, index=data.obs_names)
        for key, value in config.data.select.items():
            if not isinstance(value, list):
                value = [value]
            keep.loc[data.obs[key].isin(value)] = True
            assert keep.sum() > 0

        data = data[keep].copy()

    if "dimension_reduction" in config.data:
        genes = data.var_names.to_list()
        name = config.data.dimension_reduction.name
        if name == "pca":
            dims = config.data.dimension_reduction.get(
                "dims", data.obsm["X_pca"].shape[1]
            )

            data = anndata.AnnData(
                data.obsm["X_pca"][:, :dims], obs=data.obs.copy(), uns=data.uns.copy()
            )
            data.uns["genes"] = genes

    if "ae_emb" in config.data:
        # load path to autoencoder
        assert config.get("model.name", "cellot") == "cellot"
        path_ae = Path(config.data.ae_emb.path)
        model_kwargs = {"input_dim": data.n_vars}
        config_ae = load_config(path_ae / "config.yaml")
        ae_model, _ = load_autoencoder_model(
            config_ae, restore=path_ae / "cache/model.pt", **model_kwargs
        )

        inputs = torch.Tensor(
            data.X if not sparse.issparse(data.X) else data.X.todense()
        )

        genes = data.var_names.to_list()
        data = anndata.AnnData(
            ae_model.eval().encode(inputs).detach().numpy(),
            obs=data.obs.copy(),
            uns=data.uns.copy(),
        )
        data.uns["genes"] = genes

    # cast to dense and check for nans.
    # .toarray() (not .todense()) — torch's default_collate rejects np.matrix.
    if sparse.issparse(data.X):
        data.X = data.X.toarray()
    assert not np.isnan(data.X).any()

    dataset_args = dict()
    model_kwargs = {}

    model_kwargs["input_dim"] = data.n_vars

    if config.get("model.name") == "cae":
        condition_labels = sorted(data.obs[condition].cat.categories)
        model_kwargs["conditions"] = condition_labels
        dataset_args["obs"] = condition
        dataset_args["categories"] = condition_labels

    if "training" in config:
        pair_batch_on = config.training.get("pair_batch_on", pair_batch_on)

    if split_on is None:
        if config.model.name == "cellot":
            # datasets & dataloaders accessed as loader.train.source
            split_on = ["split", "transport"]
            if pair_batch_on is not None:
                split_on.append(pair_batch_on)

        elif (config.model.name == "scgen" or config.model.name == "cae"
              or config.model.name == "popalign"):
            split_on = ["split"]

        else:
            raise ValueError

    if isinstance(split_on, str):
        split_on = [split_on]

    for key in split_on:
        assert key in data.obs.columns

    if len(split_on) > 0:
        splits = {
            (key if isinstance(key, str) else ".".join(key)): data[index]
            for key, index in data.obs[split_on].groupby(split_on).groups.items()
        }

        dataset = nest_dict(
            {
                key: AnnDataDataset(val.copy(), **dataset_args)
                for key, val in splits.items()
            },
            as_dot_dict=True,
        )

    else:
        dataset = AnnDataDataset(data.copy(), **dataset_args)

    if "loader" in return_as:
        kwargs = dict(config.dataloader)
        kwargs.setdefault("drop_last", True)
        loader = cast_dataset_to_loader(dataset, **kwargs)

    returns = list()
    for key in return_as:
        if key == "anndata":
            returns.append(data)

        elif key == "dataset":
            returns.append(dataset)

        elif key == "loader":
            returns.append(loader)

    if include_model_kwargs:
        returns.append(model_kwargs)

    if len(returns) == 1:
        return returns[0]

    return tuple(returns)


def split_cell_data_train_test(
    data, groupby=None, random_state=0, holdout=None, subset=None, **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)
    groups = {None: data.obs.index}
    if groupby is not None:
        groups = data.obs.groupby(groupby).groups

    for key, index in groups.items():
        trainobs, testobs = train_test_split(index, random_state=random_state, **kwargs)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    return split


def split_cell_data_train_test_eval(
    data,
    test_size=0.15,
    eval_size=0.15,
    groupby=None,
    random_state=0,
    holdout=None,
    **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    groups = {None: data.obs.loc[split != "ood"].index}
    if groupby is not None:
        groups = data.obs.loc[split != "ood"].groupby(groupby).groups

    for key, index in groups.items():
        training, evalobs = train_test_split(
            index, random_state=random_state, test_size=eval_size
        )

        trainobs, testobs = train_test_split(
            training, random_state=random_state, test_size=test_size
        )

        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
        split.loc[evalobs] = "eval"

    return split


def split_cell_data_toggle_ood(data, holdout, key, mode, random_state=0, **kwargs):

    """Hold out ood sample, coordinated with iid split

    ood sample defined with key, value pair

    for ood mode: hold out all cells from a sample
    for iid mode: include half of cells in split
    """

    split = split_cell_data_train_test(data, random_state=random_state, **kwargs)

    if not isinstance(holdout, list):
        value = [holdout]

    ood = data.obs_names[data.obs[key].isin(value)]
    trainobs, testobs = train_test_split(ood, random_state=random_state, test_size=0.5)

    if mode == "ood":
        split.loc[trainobs] = "ignore"
        split.loc[testobs] = "ood"

    elif mode == "iid":
        split.loc[trainobs] = "train"
        split.loc[testobs] = "ood"

    else:
        raise ValueError

    return split


def split_cell_data(data, name="train_test", **kwargs):
    if name == "train_test":
        split = split_cell_data_train_test(data, **kwargs)
    elif name == "toggle_ood":
        split = split_cell_data_toggle_ood(data, **kwargs)
    elif name == "train_test_eval":
        split = split_cell_data_train_test_eval(data, **kwargs)
    else:
        raise ValueError

    return split.astype("category")
