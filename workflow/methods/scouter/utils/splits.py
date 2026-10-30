"""Adapters that map our `data/splits/<size>/fold_N/...` layout into the
in-memory shapes Scouter expects.

Keeps scouter independent of cellot's package — the held-out + metadata logic
here is a deliberate small copy of cellot's `cellot/data/cell.py` helpers
(`_held_out_context_from_metadata`, `_as_held_out_list`) so the two methods
have no cross-imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import anndata
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# ---------------------------------------------------------------------------
# metadata + held-out helpers (copy of cellot's logic — see plan)
# ---------------------------------------------------------------------------

def _as_held_out_list(value) -> list[str]:
    """Normalize None | str | list-like → list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def read_metadata(splits_dir: Path) -> dict:
    """Load <splits_dir>/../metadata.json."""
    meta_path = Path(splits_dir).parent / "metadata.json"
    with open(meta_path) as f:
        return json.load(f)


def fold_index_from_dir(splits_dir: Path) -> int:
    name = Path(splits_dir).name
    if not name.startswith("fold_"):
        raise ValueError(f"splits_dir name must start with 'fold_': {name}")
    return int(name.split("_")[1])


def held_out_from_metadata(splits_dir: Path) -> tuple[str | None, list[str]]:
    """Returns (context_col, held_out_contexts_list).

    Accepts the current schema (`held_out_datasets`: list of dataset names) and the
    older ones (`held_out_contexts`: list / `held_out_context`: str). Returns
    (None, []) if metadata missing.

    NOTE the rebuilt splits DB dropped `context_col` from metadata.json, and its
    per-fold `held_out` is an AXIS spec ({'donor': 'D2', 'cell_line': 'HepG2'}), not
    a dataset list — the resolved dataset names live in `held_out_datasets`. The
    grouping column is `dataset` in every source adata, so default to that.
    """
    meta_path = Path(splits_dir).parent / "metadata.json"
    if not meta_path.exists():
        return None, []
    meta = json.loads(meta_path.read_text())
    context_col = meta.get("context_col", "dataset")
    fold_idx = fold_index_from_dir(splits_dir)
    if fold_idx >= len(meta.get("folds", [])):
        return context_col, []
    fold = meta["folds"][fold_idx]
    raw = fold.get(
        "held_out_datasets",
        fold.get("held_out_contexts", fold.get("held_out_context")),
    )
    return context_col, _as_held_out_list(raw)


# ---------------------------------------------------------------------------
# h5ad loading + normalization
# ---------------------------------------------------------------------------

def _ensure_normalized(adata: anndata.AnnData, target_sum: float = 1e4) -> None:
    """Library-size + log1p when X is integer. Mirrors cellot's
    `_ensure_normalized`: int16 sparse → cast to float32 first to dodge a
    scipy `csr_sort_indices` dtype-compat error.
    """
    if not np.issubdtype(adata.X.dtype, np.integer):
        return
    print(
        f"[splits] Normalizing integer X ({adata.X.dtype}): "
        f"cast→float32 + library-size-normalize(target={target_sum:g}) + log1p",
        flush=True,
    )
    if sparse.issparse(adata.X):
        X = adata.X
        adata.X = type(X)(
            (X.data.astype(np.float32), X.indices, X.indptr), shape=X.shape,
        )
    else:
        adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def shared_gene_cols(source_path: Path) -> np.ndarray | None:
    """Column indices of genes measured (non-zero-able) in EVERY dataset, read
    from varm['feature_mask_per_dataset'] (n_genes × n_datasets, 0/1). These are
    the features free of structural zero-padding: restricting to them (a) fixes
    the near-random logits that zero-padded union genes produce and (b) drops the
    matrix to ~1/4 the columns (~68% of nnz → ~1/3 the RAM). Returns None if the
    mask is absent (caller then keeps all genes). Mirrors fingerprint's
    `shared_genes_from_mask`."""
    with h5py.File(source_path, "r") as f:
        if "varm/feature_mask_per_dataset" not in f:
            return None
        mask = f["varm/feature_mask_per_dataset"][:]        # (n_genes, n_datasets)
    return np.where(mask.sum(1) == mask.shape[1])[0].astype(np.int64)


def dataset_gene_cols(source_path: Path, dataset: str) -> np.ndarray | None:
    """Column indices of dataset D's OWN gene panel (feature_mask_per_dataset[:, D],
    via uns['dataset_to_mask_idx']) — the per-dataset analogue of shared_gene_cols.
    Returns None if the mask is absent."""
    with h5py.File(source_path, "r") as f:
        if "varm/feature_mask_per_dataset" not in f:
            return None
        mask = f["varm/feature_mask_per_dataset"][:]                 # (n_genes, n_datasets)
        d2i = f["uns/dataset_to_mask_idx"]
        if dataset not in d2i:
            raise ValueError(f"dataset {dataset!r} not in uns/dataset_to_mask_idx")
        col = int(d2i[dataset][()])
    return np.where(mask[:, col] == 1)[0].astype(np.int64)


def dataset_indices_at(source_path: Path, index_path: Path, dataset: str,
                       dataset_col: str = "dataset") -> np.ndarray:
    """Subset of `index_path` rows whose obs[dataset_col] == dataset (0-based source
    positions). Load only D's own cells — this is what relieves the global-merge OOM
    (the whole point of per-dataset scouter)."""
    idx = np.sort(np.unique(np.loadtxt(index_path, dtype=np.int64)))
    with h5py.File(source_path, "r") as f:
        node = f[f"obs/{dataset_col}"]
        if isinstance(node, h5py.Group):                            # categorical
            cats = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                    for c in node["categories"][:]]
            labels = np.asarray(cats)[node["codes"][idx]]
        else:
            raw = node[idx]
            labels = np.array([v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                               for v in raw])
    return idx[labels == dataset]


def load_cells_by_index(
    source_path: Path,
    index_path: Path,
    normalize: bool = True,
    gene_cols: np.ndarray | None = None,
    row_chunk: int = 2_000_000,
    indices: np.ndarray | None = None,
) -> anndata.AnnData:
    """Backed-mode read of source.h5ad, materialize only the rows referenced
    by `index_path` (one int per line, gzip-OK). Sorts indices so the HDF5
    read is sequential.

    If `gene_cols` is given, the matrix is subset to those columns DURING the
    read, in row chunks, so we never hold the full-width (union-gene) matrix and
    its subset at once — peak RAM is the chunk (bounded) plus the growing subset,
    not 2× the union matrix. Essential at full scale where the union matrix is
    hundreds of GB.
    """
    if indices is not None:                       # pre-restricted (e.g. one dataset)
        test_idx = np.sort(np.unique(np.asarray(indices, dtype=np.int64)))
    else:
        test_idx = np.sort(np.unique(np.loadtxt(index_path, dtype=np.int64)))
    if test_idx.size == 0:
        raise RuntimeError(f"empty index set (path={index_path})")
    backed = anndata.read(source_path, backed="r")
    try:
        if gene_cols is None:
            adata = backed[test_idx].to_memory()
        elif test_idx.size <= row_chunk:
            adata = backed[test_idx].to_memory()[:, gene_cols].copy()
        else:
            parts = []
            for s in range(0, test_idx.size, row_chunk):
                chunk = backed[test_idx[s:s + row_chunk]].to_memory()
                parts.append(chunk[:, gene_cols].copy())     # subset; full chunk freed next loop
                del chunk
            adata = anndata.concat(parts, axis=0) if len(parts) > 1 else parts[0]
            del parts
    finally:
        backed.file.close()
    if normalize:
        _ensure_normalized(adata)
    return adata


def load_ctrl(ctrl_path: Path, normalize: bool = True) -> anndata.AnnData:
    """Full read of ctrl.h5ad (smaller than perturbed source)."""
    ctrl = anndata.read(ctrl_path)
    # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's
    # own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control),
    # so comparing the label against a single value misses 18/22 datasets.
    ctrl.obs["perturbation"] = "control"
    if normalize:
        _ensure_normalized(ctrl)
    return ctrl


def filter_by_context(
    adata: anndata.AnnData,
    ctx_col: str | None,
    held_out: Iterable[str],
    keep_held_out: bool,
) -> anndata.AnnData:
    """If keep_held_out=False (id split), drop rows whose ctx_col is in
    held_out. If True (ood split), keep only those rows. Pass-through when
    held_out is empty or ctx_col missing.
    """
    held_out = list(held_out)
    if not held_out or ctx_col is None or ctx_col not in adata.obs:
        return adata
    mask = adata.obs[ctx_col].isin(held_out)
    if not keep_held_out:
        mask = ~mask
    return adata[mask].copy()


# ---------------------------------------------------------------------------
# Perturbation label space (mirrors cellot's list_perts.py contract)
# ---------------------------------------------------------------------------

def list_train_perts(
    source_path: Path,
    train_idx_path: Path,
    cls_col: str = "perturbation",
    ctrl_label: str = "ctrl",
) -> list[str]:
    """Read obs[cls_col] at the rows in train_idx_path, return sorted unique
    non-control class names. Cheap because we only materialize obs, not X.
    """
    idx = np.loadtxt(train_idx_path, dtype=np.int64)
    idx = np.sort(np.unique(idx))
    backed = anndata.read(source_path, backed="r")
    try:
        labels = backed.obs[cls_col].iloc[idx].astype(str)
    finally:
        backed.file.close()
    uniq = sorted(set(labels.tolist()) - {ctrl_label})
    return uniq
