#!/usr/bin/env python3
"""Scouter dataset-blind ensemble eval.

Each gears_train[D] job saved D's per-pert predicted fingerprints (fingerprints.npz:
P = per-pert predicted mean over D's OWN panel, perts, panel_cols, ctrl_mean). This
scores EVERY test cell against EVERY dataset's fingerprints — the real-life setting
where a cell doesn't reveal which of the 21 models to use — with the SAME combiner
as genepert (GEARS' per-pert prediction is genepert's ridge prediction, just from a
GNN): Pearson(cell_over_D_panel - cell's OWN-dataset ctrl mean, P_D centered) ->
per-cell logits over D's perts, softmax per cell, average over covering models per
global pert, log, then per-pert z-score. Own-control centering (not the model's
control) is batch normalization, not model selection.

Output (unchanged path/schema): <root>/scouter/perturb-seq/<size>/fold_<n>/<split>/logits.h5
"""
import argparse
import glob
import gzip
import logging
import os

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
from scipy import sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Must match run_GEARS.py GEARS_TARGET_SUM: the fingerprints P were computed on
# cells normalized to this target + log1p, so score cells the SAME way.
GEARS_TARGET_SUM = 1e4


def read_indices(splits_dir, name):
    with gzip.open(os.path.join(splits_dir, f"{name}.txt.gz"), mode="rt") as fh:
        return np.fromiter((int(l.rstrip()) for l in fh if l.strip()), dtype=np.int64)


def pearson_matrix(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return A @ B.T


def softmax_rows(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)


def load_models(fold_root):
    """Load every per-dataset fingerprints.npz -> genepert-style model dicts.
    P is centered here (P - ctrl_mean) so scoring compares control-relative deltas."""
    models = {}
    for npz in sorted(glob.glob(os.path.join(fold_root, "per_dataset", "*", "fingerprints.npz"))):
        d = np.load(npz, allow_pickle=True)
        D = str(d["dataset"])
        perts = [str(p) for p in d["perts"]]
        if not perts:
            continue
        ctrl_mean = d["ctrl_mean"].astype(np.float32)
        models[D] = dict(panel_cols=d["panel_cols"].astype(np.int64),
                         ctrl_mean=ctrl_mean, perts=perts,
                         P=(d["P"].astype(np.float32) - ctrl_mean))    # control-relative
        logging.info("  [%s] %d perts, %d-gene panel", D, len(perts), len(ctrl_mean))
    return models


def _chunk_rows(n_cells, n_perts, target_bytes=4 << 20):
    """HDF5 chunk height targeting ~4 MB, NOT the streaming chunk size.

    Chunks are allocated whole, so a chunk taller than the data wastes the
    remainder: at 200 perts a 100k-row chunk is 80 MB and 110,218 rows allocate
    two full chunks (1.8x the real bytes -- measurably, small/fold_1 genepert is
    169 MB for 88 MB of logits). At the padding mode's 10,245 columns the same
    setting would be a 4.1 GB chunk and a 12.3 GB file for 8.2 GB of data, plus a
    4 GB read/modify/write per touch. Bound by BYTES instead, as the baselines
    writer already does.
    """
    row = max(1, n_perts) * 4
    return max(1, min(int(n_cells) or 1, max(1, target_bytes // row)))


def ensemble_score_to_h5(source, split_idx, models, global_perts, ctrl_by_ds,
                         global_ctrl, out_path, chunk_size):
    split_idx = np.sort(np.asarray(split_idx, dtype=np.int64))
    n_cells, n_perts = len(split_idx), len(global_perts)
    col_of = {p: i for i, p in enumerate(global_perts)}
    for m in models.values():
        m["gcols"] = np.array([col_of[p] for p in m["perts"]], dtype=np.int64)

    def score_chunk(ci):
        block = source[ci].to_memory()
        # Re-normalize test cells into GEARS' log space (P is log-normalized), on
        # the FULL gene axis before restricting to each dataset's panel — matches
        # prepare_dataset_set (normalize_total(target_sum) on all genes, then log1p).
        sc.pp.normalize_total(block, target_sum=GEARS_TARGET_SUM)
        sc.pp.log1p(block)
        Xf = block.X.tocsr() if sparse.issparse(block.X) else np.asarray(block.X)
        cds = block.obs["dataset"].astype(str).to_numpy()
        prob_sum = np.zeros((len(ci), n_perts), dtype=np.float64)
        cover = np.zeros(n_perts, dtype=np.float64)
        for m in models.values():
            Xd = Xf[:, m["panel_cols"]]
            Xd = (Xd.toarray() if sparse.issparse(Xd) else np.asarray(Xd)).astype(np.float32)
            ctrl_mat = np.empty_like(Xd)
            for ds in np.unique(cds):
                ctrl_mat[cds == ds] = ctrl_by_ds.get(ds, global_ctrl)[m["panel_cols"]]
            scores = pearson_matrix(Xd - ctrl_mat, m["P"])
            prob_sum[:, m["gcols"]] += softmax_rows(scores)
            cover[m["gcols"]] += 1
        covered = cover > 0
        prob_sum[:, covered] /= cover[covered]
        out = np.full((len(ci), n_perts), -18.0, dtype=np.float32)
        out[:, covered] = np.log(prob_sum[:, covered] + 1e-8)
        return out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as fh:
        logits = fh.create_dataset("logits", shape=(n_cells, n_perts), dtype="float32",
                                   chunks=(_chunk_rows(n_cells, n_perts), n_perts), compression="lzf")
        fh.create_dataset("columns", data=np.array(global_perts, dtype=h5py.string_dtype()))
        fh.create_dataset("idx", data=split_idx)
        csum = np.zeros(n_perts, dtype=np.float64)
        csq = np.zeros(n_perts, dtype=np.float64)
        tag = os.path.basename(os.path.dirname(out_path))
        for start in range(0, n_cells, chunk_size):
            out = score_chunk(split_idx[start:start + chunk_size])
            logits[start:start + out.shape[0]] = out
            csum += out.sum(axis=0); csq += (out.astype(np.float64) ** 2).sum(axis=0)
            logging.info("  %s: scored %d/%d cells", tag, min(start + out.shape[0], n_cells), n_cells)
        mean = csum / max(n_cells, 1)
        std = np.sqrt(np.maximum(csq / max(n_cells, 1) - mean ** 2, 0.0))
        active = std > 1e-6
        logging.info("  per-pert z-score: %d/%d active", int(active.sum()), n_perts)
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            blk = logits[start:end]
            blk[:, active] = (blk[:, active] - mean[active]) / std[active]
            logits[start:end] = blk
    logging.info("Wrote %s (%d cells x %d perts)", out_path, n_cells, n_perts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_root", required=True, help=".../gears/perturb-seq/<size>/fold_<n>")
    ap.add_argument("--source", required=True)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", required=True, help="id_test | ood_test | zs_id | zs_ood")
    ap.add_argument("--classes", default=None,
                    help="newline file pinning the output column space. Without it the "
                         "columns are the union of what the per-dataset models emit "
                         "(unchanged behaviour). The padding mode pins it so every "
                         "method and fold emit identical columns regardless of which "
                         "datasets trained.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=100_000)
    args = ap.parse_args()

    models = load_models(args.fold_root)
    if not models:
        raise RuntimeError("no per-dataset fingerprints.npz under " + args.fold_root)
    if args.classes:
        with open(args.classes) as fh:
            global_perts = sorted({l.strip() for l in fh if l.strip()})
        # Restrict every model to the pinned space; a model that emitted classes
        # outside it would otherwise KeyError in col_of below.
        keep_n, drop_n = 0, 0
        for d, m in list(models.items()):
            sel = [i for i, p in enumerate(m["perts"]) if p in set(global_perts)]
            drop_n += len(m["perts"]) - len(sel)
            if not sel:
                logging.warning("dataset %s emits no class in the pinned space; dropping", d)
                models.pop(d)
                continue
            m["perts"] = [m["perts"][i] for i in sel]
            m["P"] = m["P"][sel]
            keep_n += len(sel)
        if not models:
            raise RuntimeError("no dataset model emits any class in --classes")
        logging.info("Pinned perturbation space: %d classes (kept %d model-class rows, "
                     "dropped %d outside the space) across %d datasets",
                     len(global_perts), keep_n, drop_n, len(models))
    else:
        global_perts = sorted({p for m in models.values() for p in m["perts"]})
        logging.info("Global perturbation space: %d perts across %d datasets",
                     len(global_perts), len(models))

    ctrl_all = sc.read_h5ad(args.ctrl)
    # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's
    # own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control),
    # so comparing the label against a single value misses 18/22 datasets.
    ctrl_all.obs["perturbation"] = "control"
    # Log-normalize (same target_sum) so per-dataset control means match the space
    # of P and the re-normalized test cells. normalize_total is per-cell, so a
    # dataset's mean here equals its training ctrl_mean regardless of population.
    sc.pp.normalize_total(ctrl_all, target_sum=GEARS_TARGET_SUM)
    sc.pp.log1p(ctrl_all)
    cds = ctrl_all.obs["dataset"].astype(str).to_numpy()
    Xctrl = ctrl_all.X.tocsr() if sparse.issparse(ctrl_all.X) else np.asarray(ctrl_all.X)
    ctrl_by_ds = {ds: np.asarray(Xctrl[cds == ds].mean(axis=0), dtype=np.float32).ravel()
                  for ds in np.unique(cds)}
    global_ctrl = np.asarray(Xctrl.mean(axis=0), dtype=np.float32).ravel()

    source = sc.read_h5ad(args.source, backed="r")
    split_idx = read_indices(args.splits_dir, args.split)
    ensemble_score_to_h5(source, split_idx, models, global_perts, ctrl_by_ds,
                         global_ctrl, args.out, args.chunk_size)


if __name__ == "__main__":
    main()
