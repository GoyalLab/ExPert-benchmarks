#!/usr/bin/env python3
"""emb_nn — embedding-nearest-seen-perturbation transfer (docs/padding_scope.md §5).

**The control the padding claim rests on.** Graph padding reduced to one line: an
unseen perturbation's response signature is borrowed from the SEEN perturbations
nearest to it in embedding space. If ExPert's padding does not beat this, padding is
not doing anything a lookup cannot.

    nn1  signature(p) = delta-centroid of the single nearest seen perturbation
    knn  signature(p) = softmax(cos/tau)-weighted mean over the k nearest seen ones

Scoring then matches the existing `centroid` baseline exactly — cosine between a
cell's own-dataset-control-relative delta and the signature — so the only thing that
differs from a trained method is where the signature came from.

Two deliberate choices that make this a fair control rather than a strawman:
  * it uses the SAME embedding table ExPert uses for padding, so prior knowledge is
    held fixed and only the model varies;
  * centroids stream from the UNCAPPED training index, so this floor sees exactly
    the cells the ExPert model was trained on (scope §2).

Reuses the baselines module's normalisation/streaming/writer so the floors and this
control are numerically comparable by construction.
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emb_nn")

# Reuse the baselines helpers verbatim (same normalisation => comparable numbers).
_B = Path(__file__).resolve().parents[1] / "methods" / "baselines" / "run.py"
_spec = importlib.util.spec_from_file_location("_baselines_run", _B)
_bl = importlib.util.module_from_spec(_spec)
sys.modules["_baselines_run"] = _bl
_spec.loader.exec_module(_bl)


def read_classes(path):
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip()]


def obs_cols(h5path, cols):
    out = {}
    with h5py.File(h5path, "r") as f:
        for c in cols:
            o = f["obs"][c]
            if isinstance(o, h5py.Group):
                cats = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                                   for x in o["categories"][:]])
                out[c] = cats[o["codes"][:]]
            else:
                out[c] = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                                     for x in o[:]])
    return out


def var_names(h5path):
    with h5py.File(h5path, "r") as f:
        v = f["var"]["_index"][:]
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def load_embeddings(path):
    d = np.load(path, allow_pickle=True)
    genes = np.asarray(d["genes"], dtype=object).tolist()
    E = np.asarray(d["embeddings"], dtype=np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return {g: E[i] for i, g in enumerate(genes)}, E.shape[1]


def build_signatures(classes, seen, C_seen, emb, variants, k, tau):
    """unseen class -> borrowed signature, per variant. Classes with no embedding,
    or with no embeddable seen neighbour, are left as NaN and scored -inf."""
    dim = len(next(iter(emb.values())))
    seen_ok = [i for i, s in enumerate(seen) if s in emb]
    if not seen_ok:
        raise SystemExit("no seen perturbation has an embedding — check --embeddings")
    Es = np.stack([emb[seen[i]] for i in seen_ok])                  # (n_seen_ok, dim)
    n_genes = C_seen.shape[1]

    sig = {v: np.full((len(classes), n_genes), np.nan, dtype=np.float32) for v in variants}
    covered = np.zeros(len(classes), dtype=bool)
    nn_report = []
    for ci, p in enumerate(classes):
        if p not in emb:
            continue
        s = Es @ emb[p]                                             # cosine, rows L2-normed
        covered[ci] = True
        j = int(np.argmax(s))
        if "nn1" in sig:
            sig["nn1"][ci] = C_seen[seen_ok[j]]
        if "knn" in sig:
            kk = min(k, s.size)
            top = np.argpartition(-s, kk - 1)[:kk]
            w = np.exp((s[top] - s[top].max()) / max(tau, 1e-8))
            w /= w.sum()
            sig["knn"][ci] = (w[:, None] * C_seen[[seen_ok[t] for t in top]]).sum(0)
        nn_report.append({"class": p, "nearest_seen": seen[seen_ok[j]], "cos": float(s[j])})
    log.info("signatures built for %d/%d classes (%d have no embedding)",
             int(covered.sum()), len(classes), len(classes) - int(covered.sum()))
    return sig, covered, nn_report


def score_split(ev_src, ev_idx, ev_labels, ev_ds, sig, covered, ctrl_mean,
                out_dir, chunk, classes):
    """Cosine(cell own-control delta, borrowed signature) -> logits.h5."""
    n_cells = len(ev_idx)
    n_classes = sig.shape[0]
    S = np.nan_to_num(sig, nan=0.0)
    norms = np.linalg.norm(S, axis=1)
    norms[norms == 0] = 1.0
    adata = ad.read_h5ad(ev_src, backed="r")
    f, dset = _bl.open_h5_writer(Path(out_dir), classes, n_cells, ev_idx, ev_labels)
    try:
        n_chunks = (n_cells + chunk - 1) // chunk
        for k, (pos, X) in enumerate(_bl.iter_rows(adata, ev_idx, chunk, None)):
            blk_ds = ev_ds[pos]
            for d in np.unique(blk_ds):
                if d in ctrl_mean:
                    X[blk_ds == d] -= ctrl_mean[d]
            cn = np.linalg.norm(X, axis=1)
            cn[cn == 0] = 1.0
            out = (X @ S.T) / cn[:, None] / norms[None, :]
            out[:, ~covered] = -np.inf
            order = np.argsort(pos, kind="stable")
            dset[np.sort(pos)] = out[order].astype(np.float32)
            log.info("[emb_nn] eval chunk %d/%d (%d cells)", k + 1, n_chunks, X.shape[0])
    finally:
        f.close()
    log.info("wrote %s/logits.h5 (%d x %d)", out_dir, n_cells, n_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="TRAINING store h5ad")
    ap.add_argument("--train-splits-dir", required=True)
    ap.add_argument("--train-index", default="id_train")
    ap.add_argument("--eval-source", required=True, help="zero-shot store h5ad")
    ap.add_argument("--eval-splits-dir", required=True)
    ap.add_argument("--eval-splits", nargs="+", default=["zs_id", "zs_ood"])
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--classes", required=True, help="canonical class space (classes.txt)")
    ap.add_argument("--variants", nargs="+", default=["nn1", "knn"], choices=("nn1", "knn"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--label-col", default="perturbation")
    ap.add_argument("--dataset-col", default="dataset")
    a = ap.parse_args()

    global CLASSES
    CLASSES = read_classes(a.classes)

    # Both stores must share the gene axis — every per-dataset panel assumption
    # in this pipeline depends on one coordinate system.
    vt, ve = var_names(a.source), var_names(a.eval_source)
    if not np.array_equal(vt, ve):
        raise SystemExit(f"gene axis mismatch: train has {len(vt)} vars, eval {len(ve)} "
                         f"(equal={np.array_equal(vt, ve)}) — emb_nn needs one coordinate system")

    tr_idx = _bl.load_indices(Path(a.train_splits_dir) / f"{a.train_index}.txt.gz")
    tcols = obs_cols(a.source, [a.label_col, a.dataset_col])
    tr_labels, tr_ds = tcols[a.label_col][tr_idx], tcols[a.dataset_col][tr_idx]
    seen = sorted(np.unique(tr_labels).tolist())
    log.info("training: %d cells, %d SEEN perturbations, %d datasets",
             len(tr_idx), len(seen), len(np.unique(tr_ds)))

    overlap = sorted(set(CLASSES) & set(seen))
    if overlap:
        raise SystemExit(f"{len(overlap)} eval classes are in the training pool "
                         f"(e.g. {overlap[:8]}) — emb_nn must transfer to UNSEEN classes only")

    ctrl = ad.read_h5ad(a.ctrl, backed="r")
    ctrl_datasets = sorted(np.unique(ctrl.obs[a.dataset_col].astype(str).to_numpy()).tolist())
    ctrl_mean = _bl.per_dataset_ctrl_mean(ctrl, ctrl_datasets, a.chunk, None)

    tr_adata = ad.read_h5ad(a.source, backed="r")
    log.info("streaming per-perturbation delta centroids over %d SEEN classes", len(seen))
    C_seen = _bl.compute_centroids(tr_adata, tr_idx, tr_labels, tr_ds, seen,
                                   ctrl_mean, a.chunk, None)

    emb, dim = load_embeddings(a.embeddings)
    log.info("embeddings: %d genes x %d dims", len(emb), dim)
    sig, covered, nn_report = build_signatures(CLASSES, seen, C_seen, emb,
                                               a.variants, a.k, a.tau)

    ecols = obs_cols(a.eval_source, [a.label_col, a.dataset_col])
    for variant in a.variants:
        for split in a.eval_splits:
            ev_idx = _bl.load_indices(Path(a.eval_splits_dir) / f"{split}.txt.gz")
            out_dir = os.path.join(a.out_root, f"emb_nn_{variant}", "output",
                                   a.size, a.fold, split)
            score_split(a.eval_source, ev_idx, ecols[a.label_col][ev_idx],
                        ecols[a.dataset_col][ev_idx], sig[variant], covered,
                        ctrl_mean, out_dir, a.chunk, CLASSES)

    rep_dir = os.path.join(a.out_root, "emb_nn_nn1", "output", a.size, a.fold)
    os.makedirs(rep_dir, exist_ok=True)
    pd.DataFrame(nn_report).to_csv(os.path.join(rep_dir, "nearest_seen.csv"), index=False)
    with open(os.path.join(rep_dir, "emb_nn_summary.json"), "w") as fh:
        json.dump({"n_classes": len(CLASSES), "n_covered": int(covered.sum()),
                   "n_seen": len(seen), "train_cells_consumed": int(len(tr_idx)),
                   "train_index": a.train_index, "k": a.k, "tau": a.tau,
                   "variants": a.variants, "embeddings": a.embeddings}, fh, indent=1)


if __name__ == "__main__":
    main()
