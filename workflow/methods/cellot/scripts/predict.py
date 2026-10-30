"""Per-cell scoring of held-out target cells against every per-perturbation
cellot model's predicted distribution.

For each test cell y we score it against each trained pert model P by pushing
control cells through g_P (the per-P transport map) and measuring kNN proximity
of y to that cloud. Output shape mirrors a classifier's logits:
    logits[i, j] = score of test cell i under pert j's predicted distribution.

Splits:
    id  — test indices id_test.txt.gz, controls from train-pool contexts
    ood  — test indices ood_test.txt.gz, controls from the held-out context

Outputs (per split) under <models_root>/<split>/:
    logits.h5         lzf-compressed HDF5 with three datasets:
                        /logits     (n_test, n_perts) float32
                        /obs_names  (n_test,)         str — adata.obs.index
                                                      values (cell barcodes)
                                                      for each scored cell;
                                                      lets downstream join
                                                      back to any obs column
                                                      without baking one into
                                                      this file
                        /perts      (n_perts,)        column order
    metrics.json                          top-k acc, MRR, macro-F1
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from tqdm import tqdm

from cellot.data.cell import (
    _ensure_normalized,
    _held_out_context_from_metadata,
    _shared_active_gene_mask,
)
from cellot.models import load_autoencoder_model, load_cellot_model
from cellot.utils import load_config


def load_controls(ctrl_path, splits_dir, split, ctx_col, held_out_context):
    ctrl = anndata.read(ctrl_path)
    # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's
    # own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control),
    # so comparing the label against a single value misses 18/22 datasets.
    ctrl.obs["perturbation"] = "control"
    _ensure_normalized(ctrl)
    ctrl.X = ctrl.X.astype(np.float32)

    # held_out_context is a list of dataset names (new metadata schema). For
    # split=id we want controls NOT in any held-out context; for split=ood we
    # want only controls in some held-out context.
    if ctx_col is not None and ctx_col in ctrl.obs and held_out_context:
        in_held = ctrl.obs[ctx_col].isin(held_out_context)
        mask = ~in_held if split == "id" else in_held
        ctrl = ctrl[mask].copy()
    return ctrl


def load_targets(source_path, splits_dir, split, condition_col):
    idx_file = splits_dir / ("id_test.txt.gz" if split == "id" else "ood_test.txt.gz")
    test_idx = np.loadtxt(idx_file, dtype=np.int64)
    # Sort so the backed HDF5 read is sequential (much faster than random fancy
    # indexing). Row order doesn't matter — test_idx stays aligned with target_X.
    test_idx = np.sort(test_idx)

    # Backed open: avoids materializing the full perturbed h5ad (300GB+ for
    # 7k-pert atlases). Only the requested rows hit memory.
    backed = anndata.read(source_path, backed="r")
    adata = backed[test_idx].to_memory()
    _ensure_normalized(adata)
    adata.X = adata.X.astype(np.float32)

    # Return cell barcodes rather than per-cell class labels — downstream can
    # look them up in adata.obs to get the class label / dataset / efficiency
    # / any other column it cares about, no need to bake one column into the
    # file. obs.index is already cell barcodes after the backed slice.
    return adata, adata.obs.index.to_numpy()


def pushforward(g, X, batch_size=512):
    """Push a float32 matrix through cellot's g (ICNN transport map)."""
    g.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        chunk = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32)
        chunk.requires_grad_(True)
        y = g.transport(chunk).detach().cpu().numpy()
        outs.append(y)
    return np.concatenate(outs, axis=0)


def apply_training_gene_filter(controls, targets, cfg, source_file, held_out):
    """Replicate the shared_active_genes filter that training applied, so ctrl
    + target end up in the same gene axis the per-pert model saw. No-op when
    the flag is off in cfg."""
    if not cfg.data.get("shared_active_genes_only", False):
        return controls, targets
    # varm lives on the perturbed source h5ad; read that file to compute mask.
    source = anndata.read(source_file, backed="r")
    mask = _shared_active_gene_mask(source, held_out=held_out)
    kept = source.var_names[mask].tolist()
    controls = controls[:, controls.var_names.intersection(kept)].copy()
    targets = targets[:, targets.var_names.intersection(kept)].copy()
    print(f"Shared-active-genes filter (from training): "
          f"{len(kept)} kept → ctrl_genes={controls.n_vars} tgt_genes={targets.n_vars}")
    return controls, targets


def maybe_encode_with_ae(X, cfg, batch_size=4096):
    """If cfg.data.ae_emb.path is set, encode X into the AE's latent space.
    Encoding runs in batches so a giant gene-space target matrix (millions of
    cells x ~14k genes) doesn't peak memory by allocating a single torch tensor.
    Returns (X_encoded, was_encoded)."""
    if "ae_emb" not in cfg.data:
        return X, False
    path_ae = Path(cfg.data.ae_emb.path)
    ae_cfg = load_config(path_ae / "config.yaml")
    ae_model, _ = load_autoencoder_model(
        ae_cfg, restore=path_ae / "cache/model.pt", input_dim=X.shape[1]
    )
    ae_model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            chunk = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32)
            out.append(ae_model.encode(chunk).numpy())
    return np.concatenate(out).astype(np.float32), True


def determine_kept_genes(controls, source_path, cfg, held_out):
    """Compute the final gene name list given the training-time filter +
    intersection with controls. Used to bound memory for streaming targets."""
    source = anndata.read(source_path, backed="r")
    source_genes = source.var_names
    if cfg.data.get("shared_active_genes_only", False):
        mask = _shared_active_gene_mask(source, held_out=held_out)
        source_genes = source_genes[mask]
    return controls.var_names.intersection(source_genes).tolist()


def stream_load_encode_targets(source_path, splits_dir, split, condition_col,
                                kept_genes, ae_path, chunk_size=20000):
    """Stream test cells from backed h5ad in chunks; per chunk: normalize,
    restrict to kept_genes, AE-encode → keep latent only.

    Bounds peak memory at ~chunk_size x len(kept_genes) (gene-space chunk) +
    n_test x latent_dim (final encoded matrix). Avoids ever materializing
    n_test x n_genes which can be 50–100GB for full atlases.
    """
    idx_file = splits_dir / ("id_test.txt.gz" if split == "id" else "ood_test.txt.gz")
    test_idx = np.sort(np.loadtxt(idx_file, dtype=np.int64))

    backed = anndata.read(source_path, backed="r")
    gene_index = backed.var_names.get_indexer(kept_genes)
    if (gene_index < 0).any():
        raise RuntimeError(
            "Some kept genes are missing from source.var_names — training "
            "filter doesn't match eval-time data."
        )

    ae_cfg = load_config(Path(ae_path) / "config.yaml")
    ae_model, _ = load_autoencoder_model(
        ae_cfg, restore=Path(ae_path) / "cache/model.pt",
        input_dim=len(kept_genes),
    )
    ae_model.eval()

    n = len(test_idx)
    chunks_X = []
    chunks_ids = []
    for start in tqdm(range(0, n, chunk_size), desc="streaming targets",
                      unit="chunk"):
        sub_idx = test_idx[start : start + chunk_size]
        sub = backed[sub_idx].to_memory()
        _ensure_normalized(sub)
        sub.X = sub.X.astype(np.float32)
        x = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
        x = x[:, gene_index]
        with torch.no_grad():
            latent = ae_model.encode(torch.as_tensor(x)).numpy()
        chunks_X.append(latent.astype(np.float32))
        chunks_ids.append(sub.obs.index.to_numpy())  # cell barcodes
        del x, sub  # release the gene-space chunk before the next iter

    # Cell barcodes (adata.obs.index for the kept rows) — downstream uses these
    # to look up any obs column needed without us baking labels into the file.
    return np.concatenate(chunks_X), np.concatenate(chunks_ids)


# Module-level globals consumed by the worker pool. Populated in main() before
# Pool() is created so fork-based children inherit them (copy-on-write read-only
# means no per-worker memory overhead).
_W_CTRL_X = None
_W_TARGET_X = None
_W_SCORE_MASK = None
_W_PCA = None
_W_K = None


def _worker_init():
    """Pin BLAS to 1 thread per worker — otherwise N workers x M threads each
    will oversubscribe the node and slow everyone down. Fork inherits the
    parent's already-initialized OpenMP/MKL thread pools, so env vars and
    torch.set_num_threads alone aren't enough; threadpoolctl re-limits the
    live pools at runtime.
    """
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _score_one_pert(args):
    col, mdir = args
    cfg = load_config(mdir / "config.yaml")
    (_, g_net), _ = load_cellot_model(
        cfg, restore=mdir / "cache" / "model.pt", input_dim=_W_CTRL_X.shape[1]
    )
    cloud = pushforward(g_net, _W_CTRL_X)[:, _W_SCORE_MASK]
    if _W_PCA is not None:
        cloud = _W_PCA.transform(cloud).astype(np.float32)
    return col, score_against_cloud(_W_TARGET_X, cloud, _W_K)


def _compute_centroid_one_pert(args):
    """Worker: load model, push ctrl through g, return mean of predicted cloud.
    Returning ~50 floats per pert (vs 5MB for kNN scores) drops IPC ~100,000x.
    """
    col, mdir = args
    cfg = load_config(mdir / "config.yaml")
    (_, g_net), _ = load_cellot_model(
        cfg, restore=mdir / "cache" / "model.pt", input_dim=_W_CTRL_X.shape[1]
    )
    cloud = pushforward(g_net, _W_CTRL_X)[:, _W_SCORE_MASK]
    if _W_PCA is not None:
        cloud = _W_PCA.transform(cloud).astype(np.float32)
    return col, cloud.mean(axis=0).astype(np.float32)


def score_via_centroids(target_X, centroids, chunk_size=200_000):
    """Bulk score: -||target - centroid|| for every (cell, pert) pair.
    One torch matmul per chunk; total cost ≈ n_test x n_perts x dim flops,
    vs n_test x n_cloud x dim x n_perts for kNN. ~5000x faster at 7k perts.
    """
    target_t = torch.as_tensor(target_X, dtype=torch.float32)
    cent_t = torch.as_tensor(centroids, dtype=torch.float32)
    cent_sq = (cent_t ** 2).sum(dim=1)  # (n_perts,)

    logits = np.empty((len(target_X), len(centroids)), dtype=np.float32)
    for i in tqdm(range(0, len(target_X), chunk_size),
                  desc="centroid scoring", unit="chunk"):
        chunk = target_t[i : i + chunk_size]
        chunk_sq = (chunk ** 2).sum(dim=1, keepdim=True)
        d2 = chunk_sq - 2.0 * chunk @ cent_t.T + cent_sq[None, :]
        d2 = torch.clamp(d2, min=0.0)
        logits[i : i + chunk_size] = (-torch.sqrt(d2)).numpy()
    return logits


def score_against_cloud(targets, cloud, k, chunk_size=200_000):
    """Score = -mean(kNN distance). Higher = closer = better match.

    Torch-matmul-based pairwise distance, processed in target chunks. ~10x
    faster than sklearn NearestNeighbors for many queries x small cloud in
    low-D, since BLAS gemm beats per-query tree traversal at this scale.
    """
    k_eff = min(k, len(cloud))
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    cloud_t = torch.as_tensor(cloud, dtype=torch.float32)
    cloud_sq = (cloud_t ** 2).sum(dim=1)  # (n_cloud,)

    out = np.empty(len(targets), dtype=np.float32)
    for i in range(0, len(targets), chunk_size):
        chunk = targets_t[i : i + chunk_size]
        chunk_sq = (chunk ** 2).sum(dim=1, keepdim=True)
        # ||t - c||² = ||t||² - 2·t·cᵀ + ||c||²
        d2 = chunk_sq - 2 * chunk @ cloud_t.T + cloud_sq[None, :]
        d2 = torch.clamp(d2, min=0.0)
        topk = torch.topk(d2, k_eff, dim=1, largest=False).values
        out[i : i + chunk_size] = (-torch.sqrt(topk).mean(dim=1)).numpy()
    return out


def compute_metrics(logits, labels, perts):
    pert_to_idx = {p: i for i, p in enumerate(perts)}
    # Keep only rows whose label is in the trained pert set.
    keep = np.array([l in pert_to_idx for l in labels])
    if keep.sum() == 0:
        return {"n_scored": 0}
    logits = logits[keep]
    true_cols = np.array([pert_to_idx[l] for l in labels[keep]])

    order = np.argsort(-logits, axis=1)  # high→low
    ranks = (order == true_cols[:, None]).argmax(axis=1) + 1
    preds = order[:, 0]

    return {
        "n_scored": int(keep.sum()),
        "n_perts": len(perts),
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "macro_f1": float(
            f1_score(true_cols, preds, average="macro", zero_division=0)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True,
                    help="Dir containing per-pert subdirs (each with config.yaml + cache/model.pt)")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", choices=["id", "ood"], required=True)
    ap.add_argument("--source_file", default=None,
                    help="Override perturbed h5ad (default: metadata.json source_path)")
    ap.add_argument("--ctrl_file", default=None,
                    help="Override control h5ad (default: read from any model's config.yaml)")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--source_subsample", type=int, default=5000,
                    help="Cap controls pushed through g per pert (0 = no cap)")
    ap.add_argument("--pca_dims", type=int, default=50,
                    help="Run kNN in PCA space of this many dims (0 disables)")
    ap.add_argument("--active_genes_only", action="store_true",
                    help="Restrict scoring to genes non-zero in both controls "
                         "and targets — strips zero-padded dims introduced by "
                         "per-dataset HVG filtering. Pushforwards still happen "
                         "in full gene space (models trained there).")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="Process-pool size for per-pert scoring. Each worker "
                         "is pinned to 1 BLAS thread; set workers ≈ cpus-per-task.")
    ap.add_argument("--score_method", choices=["centroid", "knn"],
                    default="centroid",
                    help="centroid: -||target - mean(g(ctrl))||, ~5000x faster. "
                         "knn: -mean kNN distance, more local but expensive.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cellot_root = Path(args.models_root)
    if not (cellot_root / "models").exists():
        raise FileNotFoundError(f'Root directory is missing a models directory.')
    models_root = cellot_root / "models"
    splits_dir = Path(args.splits_dir)
    
    outdir = Path(args.outdir) if args.outdir else cellot_root / f"{args.split}_test"
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover trained per-pert models
    print(f"Looking for models in {models_root}...")
    model_dirs = sorted(
        p.parent.parent for p in models_root.glob("*/cache/model.pt")
    )
    if not model_dirs:
        raise RuntimeError(f"No trained models found under {models_root}")
    print(f"Found {len(model_dirs)} trained per-pert models")

    # Pull shared config bits from the first model's config.yaml.
    first_config = load_config(model_dirs[0] / "config.yaml")
    source_file = args.source_file or first_config.data.path
    ctrl_file = args.ctrl_file or first_config.data.path_ctrl
    condition_col = first_config.data.condition

    ctx_col, held_out = _held_out_context_from_metadata(splits_dir)
    print(f"split={args.split}  ctx_col={ctx_col}  held_out={held_out}")

    # Controls are small (~5GB ctrl.h5ad) — fine to load eagerly.
    controls = load_controls(ctrl_file, splits_dir, args.split, ctx_col, held_out)

    # Compute the final gene name list once (matches what training saw).
    kept_genes = determine_kept_genes(controls, source_file, first_config, held_out)
    controls = controls[:, kept_genes].copy()
    print(f"ctrl n={len(controls)}  kept_genes={len(kept_genes)}")

    # Optional subsampling of the control pool.
    if args.source_subsample > 0 and len(controls) > args.source_subsample:
        idx = rng.choice(len(controls), args.source_subsample, replace=False)
        controls = controls[idx].copy()
        print(f"Subsampled controls to n={len(controls)}")

    def to_dense(X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)

    ctrl_X = to_dense(controls.X)
    ctrl_X, encoded = maybe_encode_with_ae(ctrl_X, first_config)

    if encoded:
        # Stream targets: never holds n_test x n_genes in memory.
        target_X, obs_names = stream_load_encode_targets(
            source_file, splits_dir, args.split, condition_col,
            kept_genes=kept_genes,
            ae_path=first_config.data.ae_emb.path,
        )
        print(f"AE-encoded ctrl + target to latent dim={ctrl_X.shape[1]}  "
              f"targets n={len(target_X)}")
    else:
        # No AE: fall back to the eager load + densify (small datasets only).
        targets, obs_names = load_targets(source_file, splits_dir, args.split,
                                           condition_col)
        targets = targets[:, kept_genes].copy()
        target_X = to_dense(targets.X)
        print(f"targets n={len(target_X)} (gene space, no AE)")

    score_mask = np.ones(ctrl_X.shape[1], dtype=bool)
    if args.active_genes_only and not encoded:
        ctrl_active = (ctrl_X != 0).any(axis=0)
        tgt_active = (target_X != 0).any(axis=0)
        score_mask = ctrl_active & tgt_active
        print(f"Active-gene filter: {score_mask.sum()} / {len(score_mask)} genes "
              f"(ctrl-only active={ctrl_active.sum()}, tgt-only active={tgt_active.sum()})")
    elif args.active_genes_only and encoded:
        print("Ignoring --active_genes_only: data is AE-encoded (no gene axis).")

    target_X_scored = target_X[:, score_mask]

    # Optional PCA: only in gene space. AE latent is already dense & low-dim.
    pca = None
    if args.pca_dims and args.pca_dims > 0 and not encoded:
        n_comp = min(args.pca_dims, target_X_scored.shape[1],
                     target_X_scored.shape[0] + ctrl_X.shape[0])
        pca = PCA(n_components=n_comp, random_state=args.seed)
        stack = np.vstack([target_X_scored, ctrl_X[:, score_mask]])
        pca.fit(stack)
        target_X_knn = pca.transform(target_X_scored).astype(np.float32)
        var_explained = float(pca.explained_variance_ratio_.sum())
        print(f"PCA: {n_comp} dims, explained_variance_ratio_sum={var_explained:.3f}")
    else:
        target_X_knn = target_X_scored

    perts = [d.name for d in model_dirs]
    logits = np.full((len(target_X), len(perts)), np.nan, dtype=np.float32)

    # Hand the worker pool the read-only data via module globals + fork's
    # copy-on-write — avoids pickling/copying target_X (potentially GB) per task.
    global _W_CTRL_X, _W_TARGET_X, _W_SCORE_MASK, _W_PCA, _W_K
    _W_CTRL_X, _W_TARGET_X = ctrl_X, target_X_knn
    _W_SCORE_MASK, _W_PCA, _W_K = score_mask, pca, args.k

    work = list(enumerate(model_dirs))
    # Print one bar line every ~30s OR every 50 completions, whichever first.
    bar = tqdm(
        total=len(work), desc=f"scoring ({args.score_method})", unit="pert",
        mininterval=30.0, miniters=50,
    )
    if args.score_method == "centroid":
        # Per pert: only return the cloud centroid (~50 floats), then a single
        # bulk matmul produces the full logits matrix. Trades local kNN
        # information for a ~5000x speedup at scale.
        worker_fn = _compute_centroid_one_pert
        centroids = np.zeros((len(perts), ctrl_X.shape[1]), dtype=np.float32)
        if args.workers > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(args.workers, initializer=_worker_init) as pool:
                for col, centroid in pool.imap_unordered(worker_fn, work):
                    centroids[col] = centroid
                    bar.update(1)
        else:
            for col, mdir in work:
                _, centroid = worker_fn((col, mdir))
                centroids[col] = centroid
                bar.update(1)
        bar.close()
        logits = score_via_centroids(target_X_knn, centroids)
    else:
        if args.workers > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(args.workers, initializer=_worker_init) as pool:
                for col, score_col in pool.imap_unordered(_score_one_pert, work):
                    logits[:, col] = score_col
                    bar.update(1)
        else:
            for col, mdir in work:
                _, score_col = _score_one_pert((col, mdir))
                logits[:, col] = score_col
                bar.update(1)
        bar.close()

    # HDF5 with lzf compression: 1.3M x 7k float32 = ~36GB raw, lzf gets it to
    # ~12-18GB at fast write speed (~1-2 min). All three arrays in one file.
    import h5py
    str_dt = h5py.string_dtype()
    with h5py.File(outdir / "logits.h5", "w") as f:
        f.create_dataset("logits", data=logits, dtype="float32",
                         compression="lzf", chunks=True)
        # /obs_names = adata.obs.index (cell barcodes), one per logit row.
        # Lets downstream join back to any obs column (perturbation label,
        # dataset, efficiency, ...) without baking a single one into this file.
        f.create_dataset("obs_names",
                         data=np.asarray(obs_names, dtype=object),
                         dtype=str_dt, compression="lzf")
        f.create_dataset("perts", data=np.asarray(perts, dtype=object),
                         dtype=str_dt)

    print(f"Wrote {outdir}/{{logits.h5,metrics.json}}")


if __name__ == "__main__":
    main()
