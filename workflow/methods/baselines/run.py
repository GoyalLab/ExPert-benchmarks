"""Baseline logits for reverse-direction (cell -> perturbation) benchmarks.

Writes per-(fold, split) logits.h5 + labels.csv under

    <out-root>/baseline_<name>/output/<size>/fold_N/<split>/

logits.h5 has datasets /logits (n_cells, n_classes float32), /columns (class
names), and /idx (source adata positions, one per row). Rows are written in
sorted-/idx order for efficient backed reads; collect_results.py honours /idx
when present so obs alignment still works.

Discovered by ``collect_results.py --source benchmarks/context --method-depth 2``
as methods ``baseline_random``, ``baseline_prior``, ``baseline_neg_target``,
``baseline_centroid``.

Baselines
---------
- random      : uniform logits (sanity floor).
- prior       : log P(pert) from id_train class frequencies, broadcast to every cell.
- neg_target  : logit[cell c, pert g] = ctrl_mean[dataset(c), g] - X[c, g]
                (gene name g must be in var_names). Perts with no gene match get -inf
                (--neg-target-fill neginf, the original runs) or, by default ('decoy'), are scored
                on a DECOY input gene (seeded, distinct per pert, never any pert's target):
                -inf hands the matched ~24% a ~275-way problem instead of a 1,157-way one,
                and no constant fill fixes that (the true knockdown is > 0, so a constant
                below it never competes); a decoy gene competes per cell AND after pooling.
- centroid    : per-pert mean delta vs. control in log-normalised space; logit is
                cosine sim between (cell delta) and (pert centroid).
- centroid_notarget : `centroid` with each perturbation's OWN target gene removed from
                its comparison (both from its centroid and from the cell). Under
                CRISPRi the target's mRNA collapses in its own cells, so an unmasked
                centroid carries the neg_target signal inside it; the gap between the
                two variants is how much of `centroid` is that one gene.

For all baselines, the class space is the sorted unique perturbations in id_train
of the same fold. labels.csv mirrors adata.obs[cls_col] for the split indices.

Layout reminder
---------------
Two directions of method discovery interact here:
  - direction = first path segment under --source (we write under reverse/)
  - method    = next segment (we write under baseline_<name>/)
  - 'output' and the per-size segment are intermediate dirs the collector walks.
"""

import argparse
import gzip
import json
import logging
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

log = logging.getLogger("baselines")

BASELINES = ("random", "prior", "neg_target", "centroid", "centroid_notarget")
CENTROID_VARIANTS = ("centroid", "centroid_notarget")
SPLITS = ("id_train", "id_val", "id_test", "ood_test")
EVAL_SPLITS = ("id_val", "id_test", "ood_test")  # splits we write logits for
NORM_TARGET = 1e4  # scanpy normalize_total default
# Experiment toggle (set in main): when True AND shared-genes is on, library-size
# normalise over ALL genes and THEN subset to shared — isolating the gene-set
# effect from the normalisation change (default is subset-then-normalise).
_NORM_OVER_ALL = False


def load_indices(path: Path) -> np.ndarray:
    with gzip.open(path, "rt") as f:
        return np.fromiter((int(x) for x in f if x.strip()), dtype=np.int64)


def normalize_log1p(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Per-cell normalise to NORM_TARGET counts, then log1p. Returns dense float32."""
    if sp.issparse(x):
        x = x.toarray()
    x = x.astype(np.float32, copy=False)
    totals = x.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    x = x * (NORM_TARGET / totals)
    np.log1p(x, out=x)
    return x


def shared_gene_cols_from_source(source_h5ad) -> np.ndarray | None:
    """Column indices of genes measured in EVERY dataset (varm
    feature_mask_per_dataset); None if the mask is absent. Same definition as
    scouter/fingerprint's shared genes."""
    with h5py.File(source_h5ad, "r") as f:
        if "varm/feature_mask_per_dataset" not in f:
            return None
        mask = f["varm/feature_mask_per_dataset"][:]
    return np.where(mask.sum(1) == mask.shape[1])[0].astype(np.int64)


def _read_norm(X_backed, block: np.ndarray, gene_cols: np.ndarray | None) -> np.ndarray:
    """Backed read of `block` rows, optional column subset to shared genes, then
    per-cell library-size + log1p. Normalisation is over the SELECTED genes
    (subset-then-normalise), matching how scouter/genepert treat their shared-gene
    inputs — so the all-vs-shared comparison mirrors the real pipeline regimes."""
    x = X_backed[block]
    if gene_cols is None:
        return normalize_log1p(x)
    if _NORM_OVER_ALL:
        return normalize_log1p(x)[:, gene_cols]     # normalise over ALL genes, then subset
    return normalize_log1p(x[:, gene_cols])          # subset, then normalise over shared


def iter_rows(adata: ad.AnnData, idx: np.ndarray, chunk: int, gene_cols=None):
    """Stream log1p-normalised rows of a backed AnnData in input order.

    Yields (positions, X_chunk) — `positions` are indices into `idx`, so the
    caller writes into the output array at exactly those rows.
    """
    # Sort indices for efficient backed reads, but emit positions in `idx`
    # so callers can place rows back in original order.
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    for start in range(0, len(sorted_idx), chunk):
        block = sorted_idx[start : start + chunk]
        positions = order[start : start + chunk]
        yield positions, _read_norm(adata.X, block, gene_cols)


def per_dataset_ctrl_mean(
    ctrl: ad.AnnData, datasets: list[str], chunk: int, gene_cols=None,
) -> dict[str, np.ndarray]:
    """Mean log1p-normalised expression of control cells, per dataset."""
    n_genes = len(gene_cols) if gene_cols is not None else ctrl.shape[1]
    out: dict[str, np.ndarray] = {}
    ds_col = ctrl.obs["dataset"].astype(str).to_numpy()
    for d in datasets:
        idx = np.where(ds_col == d)[0]
        if len(idx) == 0:
            log.warning("[ctrl] no control cells for dataset=%s", d)
            out[d] = np.zeros(n_genes, dtype=np.float32)
            continue
        acc = np.zeros(n_genes, dtype=np.float64)
        idx_sorted = np.sort(idx)
        for s in range(0, len(idx_sorted), chunk):
            block = idx_sorted[s : s + chunk]
            acc += _read_norm(ctrl.X, block, gene_cols).sum(axis=0, dtype=np.float64)
        out[d] = (acc / len(idx)).astype(np.float32)
    return out


def open_h5_writer(
    out_dir: Path, classes: list[str], n_rows: int, source_idx: np.ndarray, labels: np.ndarray,
) -> tuple[h5py.File, h5py.Dataset]:
    """Create logits.h5 with /columns + /idx pre-filled and an empty /logits ready
    for streaming chunk writes. Also writes labels.csv in the same row order.

    HDF5 chunk size targets ~4 MB per chunk so per-row writes don't trigger
    full-chunk rewrites and reads stay cheap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(classes)
    chunk_rows = max(64, min(n_rows, 1_000_000 // max(1, n_classes)))
    f = h5py.File(out_dir / "logits.h5", "w")
    dt = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("columns", data=np.asarray(classes, dtype=object), dtype=dt)
    logits = f.create_dataset(
        "logits", shape=(n_rows, n_classes), dtype=np.float32,
        chunks=(chunk_rows, n_classes), compression="lzf",
    )
    f.create_dataset("idx", data=source_idx.astype(np.int64))
    pd.DataFrame({"label": labels}).to_csv(out_dir / "labels.csv", index=False)
    return f, logits


def neg_target_chunk(
    X_chunk: np.ndarray,
    ds_chunk: np.ndarray,
    gene_cols: np.ndarray,        # (n_classes,) int — var-index per class, -1 if unmatched
    ctrl_mean: dict[str, np.ndarray],
) -> np.ndarray:
    """logit[c, p] = ctrl_mean[ds(c), g(p)] - X[c, g(p)]; -inf where p has no gene match."""
    n_cells = X_chunk.shape[0]
    n_classes = len(gene_cols)
    out = np.full((n_cells, n_classes), -np.inf, dtype=np.float32)
    matched = gene_cols >= 0
    if not matched.any():
        return out
    cols = gene_cols[matched]                          # (k,)
    # X expression at matched gene columns, per cell (n_cells, k)
    X_at_cols = X_chunk[:, cols]
    # ctrl means stacked rowwise per cell, at the same matched columns
    ds_unique = np.unique(ds_chunk)
    cm_per_cell = np.zeros_like(X_at_cols)
    for d in ds_unique:
        rows = np.where(ds_chunk == d)[0]
        if d not in ctrl_mean:
            log.warning("[neg_target] no ctrl mean for dataset=%s; %d cells stay -inf", d, len(rows))
            continue
        cm_per_cell[rows] = ctrl_mean[d][cols]
    out[:, matched] = cm_per_cell - X_at_cols
    return out


def control_delta(X_chunk: np.ndarray, ds_chunk: np.ndarray,
                  ctrl_mean: dict[str, np.ndarray]) -> np.ndarray:
    """(cell - ctrl_mean[dataset(cell)]) as a NEW array. centroid_chunk used to do
    this in place on the caller's chunk, which only worked because neg_target (which
    needs the raw values) happened to be scored first -- and with two centroid
    variants the second would have subtracted the control mean twice."""
    delta = X_chunk.astype(np.float32, copy=True)
    for d in np.unique(ds_chunk):
        if d in ctrl_mean:
            delta[ds_chunk == d] -= ctrl_mean[d]
    return delta


def mask_query_target(X_chunk: np.ndarray, labels: np.ndarray, ds: np.ndarray,
                      vn_to_pos: dict, ctrl_mean: dict[str, np.ndarray]) -> np.ndarray:
    """ABLATION: each cell's own target gene -> its dataset's control mean, i.e. delta 0 for that
    gene (the query-side mask of the fingerprinting paper; nothing learned is changed)."""
    cols = np.array([vn_to_pos.get(l, -1) for l in labels], dtype=np.int64)
    rows = np.flatnonzero(cols >= 0)
    n = 0
    for r in rows:
        cm = ctrl_mean.get(ds[r])
        if cm is not None:
            X_chunk[r, cols[r]] = cm[cols[r]]
            n += 1
    log.info("[mask-target] target set to control mean in %d of %d cells", n, len(labels))
    return X_chunk


def centroid_chunk(
    delta: np.ndarray,                                 # (n_cells, n_genes) control-subtracted
    centroids: np.ndarray,                             # (n_classes, n_genes), float32
    centroid_norms: np.ndarray,                        # (n_classes,)
    mask_cols: np.ndarray | None = None,               # (n_classes,) own-target col, -1 none
) -> np.ndarray:
    """Cosine similarity between (cell delta vs. ctrl) and per-pert centroid.

    With `mask_cols`, class p's cosine is computed with gene mask_cols[p] removed from
    BOTH vectors. Exact, and without re-fitting: dropping one coordinate leaves every
    other coordinate of the centroid (a mean) unchanged, so only the dot product and
    the two norms need the per-class correction.
    """
    eps = 1e-8
    d = delta.astype(np.float64)
    dot = d @ centroids.T.astype(np.float64)
    dn2 = (d ** 2).sum(axis=1, keepdims=True)
    cn2 = centroid_norms.astype(np.float64) ** 2
    if mask_cols is None or not (mask_cols >= 0).any():
        return (dot / (np.sqrt(dn2) * np.sqrt(cn2)[None, :] + eps)).astype(np.float32)
    cls = np.flatnonzero(mask_cols >= 0)
    g = mask_cols[cls]
    mu_g = centroids[cls, g].astype(np.float64)
    dot[:, cls] -= d[:, g] * mu_g[None, :]
    dn2c = np.repeat(dn2, centroids.shape[0], axis=1)
    dn2c[:, cls] -= d[:, g] ** 2
    cn2 = cn2.copy()
    cn2[cls] -= mu_g ** 2
    return (dot / (np.sqrt(np.clip(dn2c, 0, None)) * np.sqrt(np.clip(cn2, 0, None))[None, :]
                   + eps)).astype(np.float32)


def compute_centroids(
    adata: ad.AnnData,
    train_idx: np.ndarray,
    train_labels: np.ndarray,
    train_ds: np.ndarray,
    classes: list[str],
    ctrl_mean: dict[str, np.ndarray],
    chunk: int,
    gene_cols=None,
) -> np.ndarray:
    """Per-pert mean of (log1p-normalised cell - ctrl_mean[dataset(cell)]).

    Streams training cells; accumulates per-class sums via a single sparse
    (n_classes × chunk_n) indicator matmul per chunk, which is dramatically
    faster than looping over `unique(perts)` when n_classes is in the thousands.
    """
    n_classes = len(classes)
    n_genes = len(gene_cols) if gene_cols is not None else adata.shape[1]
    # Pre-map every training row to its class index (-1 for any class not in
    # `classes`, which we then skip). Categorical.codes vectorises the lookup
    # so this stays fast on 10M+ training rows.
    full_lbl_idx = pd.Categorical(train_labels, categories=classes).codes.astype(np.int64)
    sums = np.zeros((n_classes, n_genes), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    n_chunks = (len(train_idx) + chunk - 1) // chunk

    for k, (positions, X) in enumerate(iter_rows(adata, train_idx, chunk, gene_cols)):
        block_lbl_idx = full_lbl_idx[positions]
        block_ds = train_ds[positions]
        for d in np.unique(block_ds):
            if d in ctrl_mean:
                X[block_ds == d] -= ctrl_mean[d]

        valid = block_lbl_idx >= 0
        if valid.any():
            rows = block_lbl_idx[valid]                          # (k,) class indices
            cols = np.flatnonzero(valid)                         # (k,) cell indices in chunk
            data = np.ones(rows.size, dtype=np.float32)
            indicator = csr_matrix(
                (data, (rows, cols)), shape=(n_classes, X.shape[0])
            )
            sums += indicator @ X                                # (n_classes, n_genes)
            counts += np.bincount(rows, minlength=n_classes)
        log.info("[centroid] training chunk %d/%d (%d cells)", k + 1, n_chunks, X.shape[0])

    counts_safe = np.where(counts == 0, 1, counts)
    return (sums / counts_safe[:, None]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--ctrl-h5ad", default="data/source/ctrl.h5ad")
    # No whitelist: sizes are pipeline config (ctx7, zs_f4, ...), not a fixed set.
    p.add_argument("--size", required=True)
    p.add_argument("--fold", required=True)
    p.add_argument(
        "--out-root", default="benchmarks/context/reverse",
        help="Root for baseline_<name>/output/<size>/fold_N/<split>/ writes. "
             "If output/ is a symlink to scratch (the project convention), "
             "writes follow it transparently."
    )
    p.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--eval-splits", nargs="+", default=None,
                   help=f"splits to write logits for (default: {list(EVAL_SPLITS)}). The "
                        "padding mode passes its own arms (zs_id zs_ood), which is also "
                        "why id_train/id_val are only loaded when a baseline needs them.")
    p.add_argument("--classes", type=Path, default=None,
                   help="newline file pinning the label space. Required whenever the "
                        "evaluated classes are NOT in id_train (the padding benchmark: "
                        "the class space is unseen perturbations by definition, so "
                        "deriving it from training labels would yield the wrong space).")
    p.add_argument("--label-col", default="perturbation")
    p.add_argument("--context-col", default="dataset")
    p.add_argument("--shared-genes", action="store_true",
                   help="Restrict expression to genes measured in every dataset "
                        "(varm feature_mask_per_dataset). Writes under "
                        "baseline_<name>_shared/ so it doesn't collide with the "
                        "all-genes run.")
    p.add_argument("--norm-over-all", action="store_true",
                   help="Control experiment (with --shared-genes): normalise over "
                        "ALL genes then subset to shared, isolating the gene-set "
                        "effect from normalisation. Writes under _shared_normfull/.")
    p.add_argument("--neg-target-fill", choices=("neginf", "decoy"), default="decoy",
                   help="neg_target classes whose target is not an input gene: a decoy input gene that is "
                        "no class's target, so every class competes (default since 2026-10-19; Fig. 2), or "
                        "-inf (the original runs, backed up at benchmarks/backup_neg_target_neginf)")
    p.add_argument("--neg-target-seed", type=int, default=0)
    p.add_argument("--mask-query-target", action="store_true",
                   help="ABLATION: for every eval cell, set its OWN target gene to its dataset's control "
                        "mean (delta 0) before scoring. Centroids and training cells are untouched.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    global _NORM_OVER_ALL
    _NORM_OVER_ALL = args.norm_over_all

    splits_dir = Path(args.splits_dir).resolve()
    size_dir = splits_dir / args.size
    fold_dir = size_dir / args.fold
    meta = json.loads((size_dir / "metadata.json").read_text())
    source_h5ad = Path(meta["source_path"])
    label_col = args.label_col
    ds_col = args.context_col

    log.info("[setup] size=%s fold=%s source=%s", args.size, args.fold, source_h5ad)
    adata = ad.read_h5ad(source_h5ad, backed="r")
    obs = adata.obs
    var_names = pd.Index(adata.var_names.astype(str))

    # NOTE: named shared_cols (not gene_cols) — `gene_cols` is later reused below
    # for the neg_target class→gene index map; the two must not collide.
    shared_cols = shared_gene_cols_from_source(source_h5ad) if args.shared_genes else None
    if shared_cols is not None:
        var_names = var_names[shared_cols]
        log.info("[setup] shared-gene subset: %d of %d genes measured in every dataset",
                 len(shared_cols), adata.shape[1])
    elif args.shared_genes:
        log.warning("[setup] --shared-genes set but no feature_mask_per_dataset; using all genes")

    eval_splits = tuple(args.eval_splits) if args.eval_splits else EVAL_SPLITS
    # `prior` and `centroid` are the only baselines that read training cells; the
    # untrained floors (`random`, `neg_target`) do not, which is exactly why they are
    # the ones usable on an unseen-perturbation class space.
    need_train = bool({"prior", *CENTROID_VARIANTS} & set(args.baselines)) or args.classes is None
    load_splits = list(dict.fromkeys(list(eval_splits) + (["id_train"] if need_train else [])))
    indices = {s: load_indices(fold_dir / f"{s}.txt.gz") for s in load_splits}

    train_labels = (obs.iloc[indices["id_train"]][label_col].astype(str).to_numpy()
                    if need_train else np.empty(0, dtype=object))
    if args.classes:
        classes = [l.strip() for l in args.classes.read_text().splitlines() if l.strip()]
        if len(set(classes)) != len(classes):
            raise SystemExit(f"--classes {args.classes} has duplicate entries")
        log.info("[setup] label space PINNED to %d classes from %s", len(classes), args.classes)
        if need_train:
            seen = set(np.unique(train_labels).tolist())
            overlap = len(seen & set(classes))
            log.info("[setup] %d/%d pinned classes also appear in id_train", overlap, len(classes))
    else:
        classes = sorted(np.unique(train_labels).tolist())
        log.info("[setup] %d training classes (perturbations)", len(classes))

    eval_data: dict[str, dict] = {}
    for s in eval_splits:
        idx = indices[s]
        eval_data[s] = {
            "idx": idx,
            "labels": obs.iloc[idx][label_col].astype(str).to_numpy(),
            "datasets": obs.iloc[idx][ds_col].astype(str).to_numpy(),
        }

    # which baselines need what
    need_X = bool({"neg_target", *CENTROID_VARIANTS} & set(args.baselines))
    need_ctrl = need_X
    need_centroids = bool(set(CENTROID_VARIANTS) & set(args.baselines))

    ctrl_mean: dict[str, np.ndarray] = {}
    if need_ctrl:
        ctrl_path = Path(args.ctrl_h5ad).resolve()
        log.info("[ctrl] loading %s", ctrl_path)
        ctrl = ad.read_h5ad(ctrl_path, backed="r")
        # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's own
        # control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control), so
        # comparing the label against one value misses 18/22 datasets.
        ctrl.obs["perturbation"] = "control"
        # union of datasets needed: training + every eval split's datasets
        ds_needed = set(obs.iloc[indices["id_train"]][ds_col].astype(str).unique()) \
            if "id_train" in indices else set()
        for s in eval_splits:
            ds_needed.update(eval_data[s]["datasets"])
        ds_needed = sorted(ds_needed)
        ctrl_mean = per_dataset_ctrl_mean(ctrl, ds_needed, args.chunk, shared_cols)
        log.info("[ctrl] computed means for %d datasets", len(ctrl_mean))

    # Resolve gene-index-per-class once; -1 means no match. Cheap even when
    # neg_target isn't requested, used only when it is.
    vn_to_pos = {v: i for i, v in enumerate(var_names)}
    gene_cols = np.array(
        [vn_to_pos.get(p, -1) for p in classes], dtype=np.int64
    )
    neg_cols = gene_cols                              # neg_target's own copy (decoys must not reach centroid_notarget)
    if "neg_target" in args.baselines:
        n_matched = int((gene_cols >= 0).sum())
        log.info(
            "[neg_target] coverage: %d/%d (%.1f%%) perturbations match var_names",
            n_matched, len(classes), 100.0 * n_matched / max(1, len(classes)),
        )
        if args.neg_target_fill == "decoy":
            unmatched = np.flatnonzero(gene_cols < 0)
            pool = np.setdiff1d(np.arange(len(var_names)), gene_cols[gene_cols >= 0])
            if len(pool) < len(unmatched):
                raise SystemExit(f"[neg_target] decoy pool {len(pool)} < {len(unmatched)} unmatched classes")
            rng = np.random.default_rng(args.neg_target_seed)
            neg_cols = gene_cols.copy()
            neg_cols[unmatched] = rng.choice(pool, size=len(unmatched), replace=False)
            log.info("[neg_target] decoy fill: %d unmatched classes scored on distinct decoy genes "
                     "(pool %d non-target input genes, seed %d)", len(unmatched), len(pool),
                     args.neg_target_seed)

    centroids: np.ndarray | None = None
    centroid_norms: np.ndarray | None = None
    if need_centroids:
        train_ds = obs.iloc[indices["id_train"]][ds_col].astype(str).to_numpy()
        log.info("[centroid] computing per-pert centroids from %d training cells",
                 len(indices["id_train"]))
        centroids = compute_centroids(
            adata, indices["id_train"], train_labels, train_ds,
            classes, ctrl_mean, args.chunk, shared_cols,
        )
        centroid_norms = np.linalg.norm(centroids, axis=1).astype(np.float32)

    out_root = Path(args.out_root).resolve()

    n_classes = len(classes)
    # Static prior row — same for every cell of every split, so precompute once.
    prior_row: np.ndarray | None = None
    if "prior" in args.baselines:
        counts = pd.Series(train_labels).value_counts().reindex(classes, fill_value=0)
        freq = (counts + 1) / (counts.sum() + len(classes))  # additive smoothing
        prior_row = np.log(freq.to_numpy(dtype=np.float32))

    for split in eval_splits:
        info = eval_data[split]
        idx = info["idx"]
        n_cells = len(idx)
        # Sorted source order: backed reads stay efficient, and /idx tells the
        # collector where each output row came from.
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        sorted_labels = info["labels"][order]
        sorted_ds = info["datasets"][order]
        log.info("[%s] %d cells", split, n_cells)

        # One HDF5 writer per baseline (logits.h5 with /logits + /columns + /idx).
        writers: dict[str, tuple[h5py.File, h5py.Dataset]] = {}
        try:
            suffix = ("_shared_normfull" if _NORM_OVER_ALL else "_shared") if shared_cols is not None else ""
            for name in args.baselines:
                out_dir = out_root / f"baseline_{name}{suffix}" / "output" / args.size / args.fold / split
                writers[name] = open_h5_writer(out_dir, classes, n_cells, sorted_idx, sorted_labels)
                log.info("[%s/%s] -> %s", name, split, out_dir)

            n_chunks = (n_cells + args.chunk - 1) // args.chunk
            for k, start in enumerate(range(0, n_cells, args.chunk)):
                end = min(start + args.chunk, n_cells)
                chunk_n = end - start
                slc = slice(start, end)
                block_idx = sorted_idx[slc]
                block_ds = sorted_ds[slc]

                X_chunk: np.ndarray | None = None
                if need_X:
                    X_chunk = _read_norm(adata.X, block_idx, shared_cols)
                    if args.mask_query_target:
                        X_chunk = mask_query_target(X_chunk, sorted_labels[slc], block_ds,
                                                    vn_to_pos, ctrl_mean)
                delta = (control_delta(X_chunk, block_ds, ctrl_mean)
                         if need_centroids and X_chunk is not None else None)

                for name, (_, ldset) in writers.items():
                    if name == "random":
                        ldset[slc] = np.zeros((chunk_n, n_classes), dtype=np.float32)
                    elif name == "prior":
                        assert prior_row is not None
                        ldset[slc] = np.broadcast_to(prior_row[None, :], (chunk_n, n_classes))
                    elif name == "neg_target":
                        assert X_chunk is not None
                        ldset[slc] = neg_target_chunk(X_chunk, block_ds, neg_cols, ctrl_mean)
                    elif name in CENTROID_VARIANTS:
                        assert delta is not None and centroids is not None and centroid_norms is not None
                        ldset[slc] = centroid_chunk(
                            delta, centroids, centroid_norms,
                            mask_cols=gene_cols if name == "centroid_notarget" else None,
                        )

                log.info("[%s] chunk %d/%d (%d cells)", split, k + 1, n_chunks, chunk_n)
        finally:
            for f, _ in writers.values():
                f.close()

    log.info("[done] %s / %s", args.size, args.fold)


if __name__ == "__main__":
    main()
