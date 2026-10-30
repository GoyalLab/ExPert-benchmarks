"""Build ScouterData from our `data/splits` layout.

Lifts the working pieces of the prelim `run.py` (embedding + condition wiring)
into reusable helpers so train.py and predict.py can share them. Splits-side
helpers live in `splits.py`; nothing here depends on cellot.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .splits import (
    _ensure_normalized,
    load_cells_by_index,
    load_ctrl,
)


CTRL_LABEL = "ctrl"


# ---------------------------------------------------------------------------
# adata massaging
# ---------------------------------------------------------------------------

def _ensure_log1p_base(adata: anndata.AnnData) -> None:
    """Scanpy keeps a `log1p.base` field around; some downstream tools die
    when it's missing."""
    if adata.uns.get("log1p") is None:
        adata.uns["log1p"] = {"base": None}
    elif isinstance(adata.uns["log1p"], dict) and "base" not in adata.uns["log1p"]:
        adata.uns["log1p"]["base"] = None


def _get_gene_names(adata: anndata.AnnData) -> list[str]:
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _normalize_for_scouter(adata: anndata.AnnData) -> None:
    """Library-size normalize to 1e4 + log1p, sparse-CSR float32. Idempotent
    when X is already log-normalized (we still cast to CSR float32).
    """
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    adata.X = adata.X.tocsr().astype(np.float32, copy=False)
    # Heuristic: row sums > 100 imply raw or partly-normalized counts; rescale.
    row_sums = np.asarray(adata.X.sum(axis=1)).ravel()
    if row_sums.size and float(np.median(row_sums)) > 100.0:
        scale = np.divide(
            1e4, row_sums,
            out=np.zeros_like(row_sums, dtype=np.float32),
            where=row_sums > 0,
        )
        adata.X = sparse.diags(scale).dot(adata.X).tocsr()
        adata.X.data = np.log1p(adata.X.data)


# ---------------------------------------------------------------------------
# Merging controls + perturbed slice + writing the condition column
# ---------------------------------------------------------------------------

def _intersect_genes(*adatas: anndata.AnnData) -> list[str]:
    shared = adatas[0].var_names
    for a in adatas[1:]:
        shared = shared.intersection(a.var_names)
    if len(shared) == 0:
        raise ValueError("No genes overlap across the provided adatas.")
    return list(shared)


def _label_conditions(
    adata: anndata.AnnData,
    cls_col: str,
    ctrl_label_in: str | None,
) -> None:
    """Write a clean `condition` column to obs: 'ctrl' for controls, otherwise
    the cls_col value (e.g. perturbation target gene name).
    """
    if cls_col not in adata.obs.columns:
        # Controls path: stamp 'ctrl' for every cell.
        adata.obs["condition"] = CTRL_LABEL
        return
    raw = adata.obs[cls_col].astype(str).copy()
    if ctrl_label_in is not None:
        raw[raw == ctrl_label_in] = CTRL_LABEL
    adata.obs["condition"] = raw


def merge_scouter_adata(
    perturbed_parts: dict[str, anndata.AnnData],
    ctrl: anndata.AnnData,
    cls_col: str,
    ctrl_label_in: str | None = None,
) -> anndata.AnnData:
    """Concat controls + every entry in perturbed_parts on the intersection of
    var_names. Tags rows with `obs["_split"]` from the perturbed_parts key
    (controls get '_split' = 'ctrl'). Writes obs['condition'] + obs['cell_type'].
    Returns the merged AnnData; assumes inputs are already normalized.
    """
    # Stamp condition + cell_type on each part
    for name, adata in perturbed_parts.items():
        _label_conditions(adata, cls_col, ctrl_label_in)
        adata.obs["_split"] = name
        if "cell_type" not in adata.obs.columns:
            adata.obs["cell_type"] = "cell"
    _label_conditions(ctrl, cls_col=cls_col, ctrl_label_in=ctrl_label_in)
    # _label_conditions sets ctrl based on cls_col; force here.
    ctrl.obs["condition"] = CTRL_LABEL
    ctrl.obs["_split"] = "ctrl"
    if "cell_type" not in ctrl.obs.columns:
        ctrl.obs["cell_type"] = "cell"

    # Intersect gene set. When callers already loaded every part on the SAME gene
    # set (e.g. all subset to the shared-gene columns), `[:, shared].copy()` would
    # be a redundant full-size copy of each part — at full scale that's an extra
    # ~200GB per part and the difference between fitting an a100 node and OOMing.
    # So only re-subset a part whose var isn't already exactly `shared` (in order).
    shared = _intersect_genes(ctrl, *perturbed_parts.values())

    def _aligned(a: anndata.AnnData) -> bool:
        return list(a.var_names) == list(shared)

    ctrl_sub = ctrl if _aligned(ctrl) else ctrl[:, shared].copy()
    parts_sub = [a if _aligned(a) else a[:, shared].copy()
                 for a in perturbed_parts.values()]

    gene_names = _get_gene_names(ctrl_sub)
    ctrl_sub.var["gene_name"] = gene_names
    for adata in parts_sub:
        adata.var["gene_name"] = gene_names

    merged = anndata.concat(
        [ctrl_sub] + parts_sub,
        join="inner",
        merge="same",
        index_unique=None,
    )
    del ctrl_sub, parts_sub
    merged.obs_names_make_unique()
    merged.obs["condition"] = pd.Categorical(merged.obs["condition"].astype(str))
    merged.var["gene_name"] = gene_names
    if merged.var.index.name == "gene_name":
        merged.var.index.name = "var_idx"
    _ensure_log1p_base(merged)
    return merged


def keep_perturbations_with_var_match(
    adata: anndata.AnnData,
    cls_col: str = "condition",
) -> tuple[anndata.AnnData, list[str]]:
    """No-op pass-through, kept for API compatibility.

    Earlier we required every perturbation label to be a gene in var_names,
    but Scouter conditions are encoded via GenePT embeddings — they don't
    need to be present in the expression matrix. The real gate-keeper is
    `attach_embedding_indices`, which drops conds whose gene tokens are
    missing from the GenePT pickle. Letting perts whose target gene isn't
    in var pass through here means Scouter still predicts the var-set
    expression conditioned on the perturbation's embedding; we just don't
    have a dedicated output column for the perturbed gene's response.
    """
    return adata, []


# ---------------------------------------------------------------------------
# GenePT embedding
# ---------------------------------------------------------------------------

def build_genept_embd(
    embedding_path: Path,
    genes_needed: Iterable[str],
) -> pd.DataFrame:
    """Load GenePT pickle, build a DataFrame with rows indexed by gene names
    (plus a zero row for 'ctrl'). Genes in `genes_needed` that aren't in the
    pickle get the mean embedding vector.
    """
    with open(embedding_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Embedding pickle not a non-empty dict: {embedding_path}")

    payload_arrays: dict[str, np.ndarray] = {}
    all_vecs: list[np.ndarray] = []
    for gene, vec in payload.items():
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        payload_arrays[str(gene)] = arr
        all_vecs.append(arr)
    if not all_vecs:
        raise ValueError(f"No usable embeddings in {embedding_path}")
    mean_vec = np.mean(np.vstack(all_vecs), axis=0).astype(np.float32)

    genes_needed = {str(g) for g in genes_needed} - {CTRL_LABEL}
    missing: list[str] = []
    vectors: dict[str, np.ndarray] = {}
    for gene in sorted(genes_needed):
        vec = payload_arrays.get(gene)
        if vec is None:
            missing.append(gene)
            vec = mean_vec
        vectors[gene] = vec
    if missing:
        head = ", ".join(missing[:10])
        print(
            f"[scouter] {len(missing)} genes missing from GenePT — using mean vec. "
            f"first 10: {head}",
            flush=True,
        )
    vectors[CTRL_LABEL] = np.zeros(mean_vec.shape[0], dtype=np.float32)
    embd = pd.DataFrame.from_dict(vectors, orient="index")
    embd.index = embd.index.astype(str)
    return embd


def attach_embedding_indices(
    adata: anndata.AnnData,
    embd: pd.DataFrame,
    key: str = "embd_index",
) -> tuple[anndata.AnnData, int]:
    """Maps each `condition` to a padded list of embedding row indices in obs[key].
    Drops conditions whose split-by-'+' tokens have no embedding row.
    Returns (adata, ctrl_idx).
    """
    embd_names = embd.index.astype(str).tolist()
    embd_index = {gene: idx for idx, gene in enumerate(embd_names)}
    ctrl_idx = embd_index[CTRL_LABEL]

    cond_to_indices: dict[str, list[int]] = {}
    unmatched: set[str] = set()
    for cond in adata.obs["condition"].astype(str).unique():
        genes = cond.split("+")
        if any(g not in embd_index for g in genes):
            unmatched.add(cond)
            continue
        cond_to_indices[cond] = [embd_index[g] for g in genes]
    if unmatched:
        print(
            f"[scouter] dropping {len(unmatched)} conditions without embeddings",
            flush=True,
        )
        adata = adata[~adata.obs["condition"].astype(str).isin(unmatched)].copy()

    if not cond_to_indices:
        raise RuntimeError("No conditions survived embedding lookup.")

    max_len = max(len(v) for v in cond_to_indices.values())
    padded = {c: v + [ctrl_idx] * (max_len - len(v)) for c, v in cond_to_indices.items()}
    adata.obs[key] = adata.obs["condition"].astype(str).map(padded)
    return adata, ctrl_idx
