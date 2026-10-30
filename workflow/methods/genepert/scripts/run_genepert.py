#!/usr/bin/env python3
"""GenePert adapter — PER-DATASET training + dataset-blind ensemble evaluation.

GenePert (Chen & Zou 2024) is a single-dataset method by design: one ridge per
screen mapping a perturbation's gene embedding -> that dataset's mean post-pert
expression, over that dataset's own gene panel. Pooling 21 heterogeneous cell
lines into one global fit forces "identical embedding X, conflicting response Y"
and discards the cell-line-specific structure the method relies on. So we run it
as designed:

  TRAIN (per dataset D): restrict to D's train cells + D's OWN active-gene panel
    (varm feature_mask_per_dataset[:, D]); build D's per-perturbation means; fit
    ridge_D (embedding -> mean); predict D's perts -> P_D (predicted mean minus
    D's control mean), the per-pert "response fingerprints" over D's panel.

  EVAL (dataset-blind, both id_test and ood_test): a test cell does NOT reveal
    which of the 21 models to use (real-life). So score EVERY cell against EVERY
    model D — Pearson(cell_over_D_panel - D_ctrl_mean, P_D) -> per-cell logits
    over D's perts, softmax per cell over D's perts — then for each global pert
    average the prob over ONLY the covering models (those trained on it) and log.
    That gives one logit vector over the global perturbation space per cell.

Output (unchanged path/schema — genepert stays one method, now per-dataset):
  <results_root>/genepert/perturb-seq/<size>/fold_<n>/<split>/logits.h5
  (datasets: logits [cells x perts], columns [pert names], idx [0-based rows])
"""
import argparse
import gzip
import logging
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import h5py
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import GenePertExperiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RIDGE_ALPHA = 0.1
RIDGE_KEY = (("alpha", RIDGE_ALPHA),)
CTRL_LABEL = "control"


def read_indices(splits_dir, name):
    with gzip.open(os.path.join(splits_dir, f"{name}.txt.gz"), mode="rt") as fh:
        return np.fromiter((int(line.rstrip()) for line in fh if line.strip()), dtype=np.int64)


def build_pert_means(backed, indices, gene_cols, control_adata=None):
    """One row per perturbation = mean expression over the gene columns `gene_cols`
    (integer indices into backed.var). Streams the perturbed cells sequentially
    (bounded memory). Control means (from `control_adata`, already subset to the
    same genes + in memory) are prepended. Returns AnnData (n_perts x len genes)."""
    mean_rows, pert_list = [], []
    if control_adata is not None and control_adata.n_obs:
        cp = control_adata.obs["perturbation"].to_numpy()
        Xc = control_adata.X
        Xc = Xc.tocsr() if sparse.issparse(Xc) else np.asarray(Xc)
        for pert in np.unique(cp):
            m = cp == pert
            mean_rows.append(np.asarray(Xc[m].mean(axis=0), dtype=np.float32).ravel())
            pert_list.append(pert)

    obs = backed.obs.iloc[indices]
    idx_arr = np.asarray(indices)
    pert_all = obs["perturbation"].to_numpy()
    order = np.argsort(idx_arr, kind="stable")
    idx_sorted, pert_sorted = idx_arr[order], pert_all[order]

    sums, counts = {}, {}
    read_chunk = 500_000
    n = idx_sorted.size
    for s in range(0, n, read_chunk):
        ci, cpk = idx_sorted[s:s + read_chunk], pert_sorted[s:s + read_chunk]
        block = backed[ci].to_memory()
        X = block.X.tocsr() if sparse.issparse(block.X) else np.asarray(block.X)
        X = X[:, gene_cols]
        for pert in np.unique(cpk):
            m = cpk == pert
            rowsum = np.asarray(X[m].sum(axis=0), dtype=np.float64).ravel()
            if pert in sums:
                sums[pert] += rowsum; counts[pert] += int(m.sum())
            else:
                sums[pert] = rowsum; counts[pert] = int(m.sum())
        del block, X
    for pert, tot in sums.items():
        mean_rows.append((tot / max(counts[pert], 1)).astype(np.float32))
        pert_list.append(pert)
    return ad.AnnData(
        X=np.vstack(mean_rows).astype(np.float32),
        obs=pd.DataFrame({"perturbation": pert_list}),
        var=pd.DataFrame(index=[str(i) for i in gene_cols]),
    )


def pearson_matrix(A, B):
    """Row-centered cosine == Pearson between every row of A and every row of B."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return A @ B.T


def softmax_rows(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)


def predict_pert_fingerprints(train_means, predict_perts, embeddings, ctrl_mean, gene_names):
    """Fit ridge on the training means, predict each of `predict_perts` -> centered
    response (predicted mean - control mean). Returns (sorted_perts, P_centered)."""
    dummy = ad.AnnData(
        X=np.zeros((len(predict_perts), len(gene_names)), dtype=np.float32),
        obs=pd.DataFrame({"perturbation": list(predict_perts)}),
        var=pd.DataFrame(index=list(gene_names)),
    )
    exp = GenePertExperiment.GenePertExperiment(embeddings=embeddings)
    exp.mean_expression = ctrl_mean
    results = exp.run_experiment_with_adata(
        adata_train=train_means, adata_test=dummy, ridge_params=[{"alpha": RIDGE_ALPHA}]
    )
    names = list(results["per_gene"].keys())
    predicted = np.array(
        [results["per_gene"][p]["ridge"][RIDGE_KEY][2] for p in names], dtype=np.float32
    )
    order = np.argsort(names)
    sorted_perts = [names[i] for i in order]
    return sorted_perts, predicted[order] - ctrl_mean


def resolve_panel_cols(mask, d2m, dataset, mode="all"):
    """Gene columns a per-dataset model trains/scores on.

    all     -> every gene on the source var axis (DEFAULT). The per-dataset panels
               are hard zero-padded, so this keeps every dataset's model in ONE
               shared coordinate system (directly comparable across datasets).
    dataset -> that dataset's own active panel (varm feature_mask[:, D]).
    shared  -> intersection of the active panels across ALL datasets. On a DB with
               500-gene disjoint panels this collapses to ~1 gene; guarded below.
    """
    import numpy as _np
    n_genes = int(mask.shape[0])
    if mode == "all":
        return _np.arange(n_genes, dtype=_np.int64)
    if mode == "shared":
        cols = _np.where((mask == 1).all(axis=1))[0]
        if cols.size < 10:
            raise ValueError(
                f"gene_panel='shared' yields only {cols.size} gene(s) across "
                f"{mask.shape[1]} datasets - the panels are effectively disjoint. "
                f"Use gene_panel: all."
            )
        return cols.astype(_np.int64)
    return _np.where(mask[:, d2m[dataset]] == 1)[0].astype(_np.int64)


def build_dataset_model(source, train_idx, dataset, panel_cols, ctrl_all, embeddings,
                        predict_perts=None):
    """Train one GenePert model on dataset `dataset`: its own panel, its own cells.

    `predict_perts` (padding mode) makes the model emit fingerprints for classes it
    was NOT trained on: the ridge maps a gene EMBEDDING to a response, so any gene
    with an embedding is predictable. Widening the emitted list here is also what
    makes the dataset-blind combiner cover those classes -- `perts` flows into
    `m["perts"]` -> `gcols` -> `cover`, so an unseen class ends up averaged over all
    20 models instead of falling to -inf for lack of a "covering" model.
    """
    ds_train = source.obs["dataset"].to_numpy()[train_idx]
    idx_D = train_idx[ds_train == dataset]
    gene_names = source.var_names[panel_cols].astype(str).tolist()
    ctrl_D = ctrl_all[(ctrl_all.obs["dataset"].astype(str) == dataset)][:, panel_cols]
    ctrl_mean = (np.asarray(ctrl_D.X.mean(axis=0), dtype=np.float32).ravel()
                 if ctrl_D.n_obs else np.zeros(len(panel_cols), dtype=np.float32))
    train_means = build_pert_means(source, idx_D, panel_cols, control_adata=ctrl_D)
    trained_perts = [p for p in train_means.obs["perturbation"].tolist() if p != CTRL_LABEL]
    if not trained_perts:
        return None
    if predict_perts is None:
        perts = trained_perts
    else:
        # Emit ONLY the evaluated classes: the benchmark's class space is fixed on
        # disk, and mixing the training vocabulary into the columns would make the
        # scored space depend on which datasets happened to train.
        perts = [p for p in predict_perts if p != CTRL_LABEL and p in embeddings]
        if not perts:
            logging.warning("  [%s] none of the %d requested classes have an embedding",
                            dataset, len(predict_perts))
            return None
    sorted_perts, P = predict_pert_fingerprints(train_means, perts, embeddings, ctrl_mean, gene_names)
    logging.info("  [%s] fit on %d perts -> predicting %d, %d-gene panel, %d train cells",
                 dataset, len(trained_perts), len(sorted_perts), len(panel_cols), idx_D.size)
    return dict(panel_cols=panel_cols, ctrl_mean=ctrl_mean, perts=sorted_perts, P=P)


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
                         global_ctrl, out_path, chunk_size, labels=None):
    """Score every test cell against EVERY dataset model, softmax-avg over covering
    models, then remove the per-perturbation baseline (z-score each pert column
    across the scored cells) -> global-pert logits; stream to logits.h5.

    Each eval cell is centered by ITS OWN dataset's control mean (ctrl_by_ds[cell
    dataset], over the model's panel), NOT the model's control — a cell's
    perturbation response is a control-relative delta w.r.t. its own batch, and
    comparing that delta's SHAPE to a model's control-relative predicted deltas is
    the dataset-normalized comparison. Centering by the model's (other dataset's)
    control would inject a cross-dataset baseline shift and unfairly penalize every
    off-dataset model. Using the cell's own control is batch normalization (you ran
    the experiment, you know its controls), not model selection.

    The per-pert z-score is the composition-level fix for the dominant-fingerprint
    bias: some perts (e.g. LSM2) have predicted responses that correlate with
    almost every cell, so their averaged prob is high for everyone and the argmax
    collapses onto them. Since the combiner sees every model's score for every
    cell, each pert column's mean+std across cells IS that per-pert baseline —
    subtracting it (and scaling) leaves the per-cell signal. Uses no labels/dataset
    identity (a per-cell log-softmax by collect_results preserves this ranking)."""
    split_idx = np.sort(np.asarray(split_idx, dtype=np.int64))
    n_cells, n_perts = len(split_idx), len(global_perts)
    col_of = {p: i for i, p in enumerate(global_perts)}
    for m in models.values():
        m["gcols"] = np.array([col_of[p] for p in m["perts"]], dtype=np.int64)

    def score_chunk(ci):
        block = source[ci].to_memory()
        Xf = block.X.tocsr() if sparse.issparse(block.X) else np.asarray(block.X)
        cds = block.obs["dataset"].astype(str).to_numpy()      # each cell's own dataset (batch)
        prob_sum = np.zeros((len(ci), n_perts), dtype=np.float64)
        cover = np.zeros(n_perts, dtype=np.float64)
        for m in models.values():
            Xd = Xf[:, m["panel_cols"]]
            Xd = (Xd.toarray() if sparse.issparse(Xd) else np.asarray(Xd)).astype(np.float32)
            # center each cell by ITS OWN dataset's control mean over this panel
            ctrl_mat = np.empty_like(Xd)
            for ds in np.unique(cds):
                ctrl_mat[cds == ds] = ctrl_by_ds.get(ds, global_ctrl)[m["panel_cols"]]
            scores = pearson_matrix(Xd - ctrl_mat, m["P"])
            prob_sum[:, m["gcols"]] += softmax_rows(scores)
            cover[m["gcols"]] += 1
        covered = cover > 0
        prob_sum[:, covered] /= cover[covered]
        out = np.full((len(ci), n_perts), -18.0, dtype=np.float32)   # log(~1e-8) floor
        out[:, covered] = np.log(prob_sum[:, covered] + 1e-8)
        return out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as fh:
        logits = fh.create_dataset(
            "logits", shape=(n_cells, n_perts), dtype="float32",
            chunks=(_chunk_rows(n_cells, n_perts), n_perts), compression="lzf")
        fh.create_dataset("columns", data=np.array(global_perts, dtype=h5py.string_dtype()))
        fh.create_dataset("idx", data=split_idx)
        # pass 1: score + write raw combined logits, accumulate per-pert sum/sumsq
        csum = np.zeros(n_perts, dtype=np.float64)
        csq = np.zeros(n_perts, dtype=np.float64)
        for start in range(0, n_cells, chunk_size):
            out = score_chunk(split_idx[start:start + chunk_size])
            logits[start:start + out.shape[0]] = out
            csum += out.sum(axis=0); csq += (out.astype(np.float64) ** 2).sum(axis=0)
            logging.info("  %s: scored %d/%d cells", os.path.basename(os.path.dirname(out_path)),
                         min(start + out.shape[0], n_cells), n_cells)
        # per-pert baseline (mean/std across all scored cells); skip constant cols
        mean = csum / max(n_cells, 1)
        std = np.sqrt(np.maximum(csq / max(n_cells, 1) - mean ** 2, 0.0))
        active = std > 1e-6
        logging.info("  per-pert z-score: %d/%d pert columns active", int(active.sum()), n_perts)
        # pass 2: normalize each pert column (x - mean) / std
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            blk = logits[start:end]
            blk[:, active] = (blk[:, active] - mean[active]) / std[active]
            logits[start:end] = blk
    logging.info("Wrote %s (%d cells x %d perts)", out_path, n_cells, n_perts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--gene-panel", dest="gene_panel", default="all",
                    choices=("all", "dataset", "shared"),
                    help="gene space per per-dataset model (see config gene_panel)")
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--splits-dir", required=True,
                    help="fold dir holding the TRAINING index")
    ap.add_argument("--train-index", default="id_train",
                    help="training index basename in --splits-dir. Uncapped by default: "
                         "build_pert_means streams in bounded memory, so there is no "
                         "reason to hand this method less data than the model under test.")
    ap.add_argument("--eval-source", default=None,
                    help="adata the EVAL indices refer to (default: --source). The padding "
                         "mode trains on the full store and evaluates on the zero-shot one.")
    ap.add_argument("--eval-splits-dir", default=None,
                    help="fold dir holding the EVAL indices (default: --splits-dir)")
    ap.add_argument("--splits", nargs="+", default=["id_test", "ood_test"],
                    help="eval split basenames to score")
    ap.add_argument("--predict-perts", default=None,
                    help="newline file of classes to emit fingerprints for, INCLUDING ones "
                         "with no training cells (padding mode). Default: the trained perts.")
    ap.add_argument("--method-dir", default="genepert",
                    help="output dir name under --results-root (e.g. genepert_atoms for the "
                         "ExPert-atom feature ablation)")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    args = ap.parse_args()

    d = np.load(args.embeddings, allow_pickle=True)
    emb_df = pd.DataFrame(d["embeddings"], index=d["genes"])
    gene_embeddings = {gene: np.array(row) for gene, row in emb_df.iterrows()}
    logging.info("Loaded %d gene embeddings", len(gene_embeddings))

    ctrl_all = sc.read_h5ad(args.ctrl)
    # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's
    # own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control),
    # so comparing the label against a single value misses 18/22 datasets.
    ctrl_all.obs["perturbation"] = "control"
    source = sc.read_h5ad(args.source, backed="r")
    mask = np.asarray(source.varm["feature_mask_per_dataset"])
    d2m = {str(k): int(v) for k, v in dict(source.uns["dataset_to_mask_idx"]).items()}

    train_idx = read_indices(args.splits_dir, args.train_index)
    predict_perts = None
    if args.predict_perts:
        with open(args.predict_perts) as fh:
            predict_perts = [l.strip() for l in fh if l.strip()]
        have = sum(1 for p in predict_perts if p in gene_embeddings)
        logging.info("predicting a FIXED class space: %d classes, %d with an embedding (%.1f%%)",
                     len(predict_perts), have, 100.0 * have / max(1, len(predict_perts)))
    train_ds = pd.unique(source.obs["dataset"].to_numpy()[train_idx])
    logging.info("Training %d per-dataset GenePert models: %s", len(train_ds), list(train_ds))

    models = {}
    for D in train_ds:
        D = str(D)
        if D not in d2m:
            logging.warning("dataset %s not in dataset_to_mask_idx; skipping", D); continue
        panel_cols = resolve_panel_cols(mask, d2m, D, args.gene_panel)
        m = build_dataset_model(source, train_idx, D, panel_cols, ctrl_all, gene_embeddings,
                                predict_perts=predict_perts)
        if m is not None:
            models[D] = m
    if not models:
        raise RuntimeError("no per-dataset models trained")
    # With a fixed class space the columns are that space (sorted), so every fold and
    # every method emit identical columns regardless of which datasets trained.
    global_perts = (sorted({p for m in models.values() for p in m["perts"]})
                    if predict_perts is None else sorted(set(predict_perts)))
    logging.info("Global perturbation space: %d perts across %d models", len(global_perts), len(models))

    # Per-dataset control means (all genes) so each eval cell is centered by ITS OWN
    # dataset's control (batch normalization), not the scoring model's control.
    cds_all = ctrl_all.obs["dataset"].astype(str).to_numpy()
    Xctrl = ctrl_all.X.tocsr() if sparse.issparse(ctrl_all.X) else np.asarray(ctrl_all.X)
    ctrl_by_ds = {ds: np.asarray(Xctrl[cds_all == ds].mean(axis=0), dtype=np.float32).ravel()
                  for ds in np.unique(cds_all)}
    global_ctrl = np.asarray(Xctrl.mean(axis=0), dtype=np.float32).ravel()

    # EVAL may come from a different store than training (padding mode). The gene
    # axis must match: every per-dataset panel index is a column position.
    eval_source = source
    if args.eval_source and os.path.abspath(args.eval_source) != os.path.abspath(args.source):
        eval_source = sc.read_h5ad(args.eval_source, backed="r")
        if not np.array_equal(np.asarray(source.var_names), np.asarray(eval_source.var_names)):
            raise SystemExit(
                f"gene axis mismatch: train {args.source} has {source.n_vars} vars, "
                f"eval {args.eval_source} has {eval_source.n_vars} -- the per-dataset "
                "panel columns would point at different genes.")
        logging.info("eval source: %s (%d cells)", args.eval_source, eval_source.n_obs)
    eval_splits_dir = args.eval_splits_dir or args.splits_dir

    for split in args.splits:
        split_idx = read_indices(eval_splits_dir, split)
        out_path = os.path.join(args.results_root, args.method_dir, "perturb-seq", args.size,
                                f"fold_{args.fold}", split, "logits.h5")
        ensemble_score_to_h5(eval_source, split_idx, models, global_perts, ctrl_by_ds,
                             global_ctrl, out_path, args.chunk_size)


if __name__ == "__main__":
    main()
