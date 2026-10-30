#!/usr/bin/env python3
"""Materialise the padding mode's split space (docs/padding_scope.md §3).

The padding benchmark evaluates on the ZERO-SHOT store — perturbations that the
store recipe routed out of the main pool for falling below
`min_cells_per_perturbation`, so no model has ever seen one — while TRAINING still
reads the ordinary `full` split space. This script writes a split space that looks
exactly like an upstream one, so every path helper in common.smk resolves it with
no special casing:

    <padding_splits_root>/<size>/
        metadata.json        source_path -> the zero-shot store, + fold/held-out info
        classes.txt          the CANONICAL class space (never unioned across methods)
        summary.json         the numbers a reader needs to interpret the arms
        <fold>/zs_id.txt.gz  eval rows from the fold's TRAINING datasets  (cap applied)
        <fold>/zs_ood.txt.gz eval rows from the fold's HELD-OUT datasets  (never capped)

Two arms, so "unseen perturbation" and "unseen perturbation x unseen context" are
separable:
    zs_id  — cells whose dataset IS in the fold's training set: only the
             perturbation is new.
    zs_ood — cells from the fold's held-out datasets: perturbation AND context new.

HARD ASSERT: the zs label set must be disjoint from the training pool. It is today
(measured overlap 0); making it fatal means a future store rebuild cannot silently
introduce leakage into the headline claim.
"""
import argparse
import gzip
import json
import logging
import os
import sys

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zs_splits")


def read_str_col(h5, col):
    """obs column -> np.ndarray[str], handling both plain and categorical layouts."""
    o = h5["obs"][col]
    if isinstance(o, h5py.Group):                      # categorical
        cats = np.asarray([c.decode() if isinstance(c, bytes) else str(c)
                           for c in o["categories"][:]])
        return cats[o["codes"][:]]
    v = o[:]
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def write_idx(path, idx):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.writelines(f"{int(i)}\n" for i in np.sort(np.asarray(idx, dtype=np.int64)))
    log.info("wrote %s (%d rows)", path, len(idx))


def held_out_datasets(splits_root, train_size, fold):
    """The fold's held-out datasets, from the TRAINING split space's metadata."""
    with open(os.path.join(splits_root, train_size, "metadata.json")) as fh:
        meta = json.load(fh)
    folds = meta["folds"]
    if isinstance(folds, str):                        # stored as a repr string
        import ast
        folds = ast.literal_eval(folds)
    want = int(str(fold).split("_")[1])
    for f in folds:
        if int(f["fold"]) == want:
            return sorted(f.get("held_out_datasets", [])), f.get("held_out", {})
    raise KeyError(f"{fold} not found in {train_size}/metadata.json")


def training_pool(train_source, splits_root, train_size, fold, train_index):
    """Perturbation vocabulary the models actually train on — for the disjointness
    assert and for the optional `union` class space. One sequential read of the
    training store's label column (the fancy-index path is far slower)."""
    ipath = os.path.join(splits_root, train_size, fold, f"{train_index}.txt.gz")
    with gzip.open(ipath, "rt") as fh:
        tr = np.fromiter((int(l) for l in fh if l.strip()), dtype=np.int64)
    with h5py.File(train_source, "r") as h5:
        lab = read_str_col(h5, "perturbation")
    pool = np.unique(lab[tr])
    log.info("training pool: %d perturbations over %d cells (%s)", len(pool), len(tr), train_index)
    return pool, len(tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-source", required=True, help="the zero-shot store h5ad")
    ap.add_argument("--train-source", required=True, help="the training store h5ad")
    ap.add_argument("--splits-root", required=True, help="upstream splits root (training space)")
    ap.add_argument("--train-size", default="full")
    ap.add_argument("--out-root", required=True, help="padding splits root")
    ap.add_argument("--size", required=True, help="padding size name, e.g. zs_f4")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--train-index", default="id_train")
    ap.add_argument("--eval-cap", type=int, default=2,
                    help="cells per (DATASET, PERTURBATION) on zs_id; 0 = uncapped. Same unit "
                         "as the upstream recipe's zs_eval_cap, so the arm stays stratified "
                         "by construction rather than by a separate balancing step.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-space", choices=("zs_only", "union"), default="zs_only")
    ap.add_argument("--require-embedding", default=None,
                    help="npz {embeddings,genes} of the class-embedding table. Classes absent "
                         "from it are dropped from BOTH classes.txt and the eval indices. "
                         "This is not optional in practice: ExPert's setup_anndata raises "
                         "`[Registry] class_emb file ... is missing N/M observed labels` and "
                         "refuses to load if any OBSERVED label has no embedding, so such "
                         "classes cannot be scored as -inf misses -- their cells must not be "
                         "in the eval set at all.")
    ap.add_argument("--label-col", default="perturbation")
    ap.add_argument("--dataset-col", default="dataset")
    a = ap.parse_args()

    out_dir = os.path.join(a.out_root, a.size)
    fold_dir = os.path.join(out_dir, a.fold)
    os.makedirs(fold_dir, exist_ok=True)

    ho, ho_spec = held_out_datasets(a.splits_root, a.train_size, a.fold)
    log.info("%s held-out datasets: %s", a.fold, ho)

    with h5py.File(a.eval_source, "r") as h5:
        lab = read_str_col(h5, a.label_col)
        ds = read_str_col(h5, a.dataset_col)
        n_cells = int(h5["X"].attrs["shape"][0]) if "X" in h5 else len(lab)
    if len(lab) != n_cells:
        raise RuntimeError(f"obs/{a.label_col} has {len(lab)} rows but X has {n_cells}")

    is_ood = np.isin(ds, ho)
    zs_perts = np.unique(lab)

    # ---- drop classes with no embedding, and their cells ---------------------
    n_dropped_cls = n_dropped_cells = 0
    if a.require_embedding:
        emb = np.load(a.require_embedding, allow_pickle=True)
        have = set(np.asarray(emb["genes"], dtype=object).tolist())
        keep_cls = np.array([p in have for p in zs_perts])
        n_dropped_cls = int((~keep_cls).sum())
        if n_dropped_cls:
            dropped = zs_perts[~keep_cls]
            keep_cell = np.isin(lab, zs_perts[keep_cls])
            n_dropped_cells = int((~keep_cell).sum())
            log.warning("%d/%d classes have no entry in %s -- dropping them AND their "
                        "%d cells (e.g. %s). ExPert cannot be set up otherwise.",
                        n_dropped_cls, zs_perts.size, os.path.basename(a.require_embedding),
                        n_dropped_cells, list(dropped[:6]))
            lab = lab[keep_cell]; ds = ds[keep_cell]; is_ood = is_ood[keep_cell]
            row_map = np.flatnonzero(keep_cell)          # local -> source row
            zs_perts = zs_perts[keep_cls]
        else:
            row_map = np.arange(len(lab))
    else:
        row_map = np.arange(len(lab))

    # ---- the assert the headline claim depends on ---------------------------
    pool, n_train_cells = training_pool(a.train_source, a.splits_root, a.train_size,
                                        a.fold, a.train_index)
    overlap = np.intersect1d(zs_perts, pool)
    if overlap.size:
        raise SystemExit(
            f"LEAKAGE: {overlap.size} of the {zs_perts.size} zero-shot perturbations are "
            f"in the training pool for {a.train_size}/{a.fold} (e.g. {list(overlap[:8])}). "
            "The padding benchmark is invalid until the eval set is made disjoint.")
    log.info("disjointness OK: 0 / %d zs perturbations appear in the training pool", zs_perts.size)

    # ---- zs_ood: never capped (already small) -------------------------------
    ood_local = np.flatnonzero(is_ood)
    ood_rows = row_map[ood_local]
    write_idx(os.path.join(fold_dir, "zs_ood.txt.gz"), ood_rows)

    # ---- zs_id: cap per (DATASET, PERTURBATION) ---------------------------
    # The SAME stratification unit the upstream recipe uses (zs_eval_cap: 10 per
    # dataset x perturbation -- verified on the store: 70,182 pairs, min 1, median 10,
    # max 10). An earlier version here capped per PERTURBATION globally, which
    # collapsed ~30 cells spanning ~3 datasets down to 10 drawn arbitrarily: that
    # turns a per-class score into a per-class x per-DATASET score, and the
    # dataset-blind combiners (genepert/scouter/gears all average over covering
    # models) would then score a class against contexts it never sees for it.
    id_local_all = np.flatnonzero(~is_ood)
    id_rows_all = row_map[id_local_all]
    if a.eval_cap and a.eval_cap > 0:
        rng = np.random.default_rng(a.seed)
        key = np.char.add(np.char.add(ds[id_local_all].astype(str), "\x00"),
                          lab[id_local_all].astype(str))
        order = np.argsort(key, kind="stable")
        k_sorted = key[order]
        rows_sorted, local_sorted = id_rows_all[order], id_local_all[order]
        bounds = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1], True])
        keep, keep_local = [], []
        for s_, e_ in zip(bounds[:-1], bounds[1:]):
            n = e_ - s_
            sel = (np.arange(s_, e_) if n <= a.eval_cap
                   else rng.choice(np.arange(s_, e_), size=a.eval_cap, replace=False))
            keep.append(rows_sorted[sel]); keep_local.append(local_sorted[sel])
        id_rows = np.concatenate(keep); id_local = np.concatenate(keep_local)
    else:
        id_rows, id_local = id_rows_all, id_local_all
    write_idx(os.path.join(fold_dir, "zs_id.txt.gz"), id_rows)

    # ---- training pool, for consumers that need to bound a graph -----------
    # Without this, make_gears_geneset falls back to ALL 67,832 gene2go genes, giving
    # GEARS a 67.8k-node perturbation graph instead of the ~18.6k the design calls
    # for (and 3.6x the nn.Embedding rows / GO-similarity edges).
    with open(os.path.join(out_dir, "train_perts.txt"), "w") as fh:
        fh.write("\n".join(sorted(pool.tolist())) + "\n")
    log.info("train_perts.txt: %d training-pool perturbations", pool.size)

    # ---- canonical class space --------------------------------------------
    classes = sorted(zs_perts.tolist()) if a.class_space == "zs_only" \
        else sorted(set(zs_perts.tolist()) | set(pool.tolist()))
    with open(os.path.join(out_dir, "classes.txt"), "w") as fh:
        fh.write("\n".join(classes) + "\n")
    log.info("classes.txt: %d classes (%s); chance accuracy %.2e",
             len(classes), a.class_space, 1.0 / max(1, len(classes)))

    # ---- metadata.json: same contract as an upstream split space -----------
    meta = {
        "source_path": a.eval_source,
        "context_col": a.dataset_col,
        "cls_col": a.label_col,
        "index_format": "txt.gz",
        "index_base": 0,
        "n_folds": 1,
        "seed": a.seed,
        "padding_mode": True,
        "train_space": {"splits_root": a.splits_root, "size": a.train_size,
                        "fold": a.fold, "index": a.train_index,
                        "source_path": a.train_source,
                        "n_train_cells": int(n_train_cells),
                        "n_train_perts": int(pool.size)},
        "class_space": a.class_space,
        "n_classes": len(classes),
        "folds": [{"fold": int(str(a.fold).split("_")[1]),
                   "held_out": ho_spec, "held_out_datasets": ho,
                   "n_zs_id": int(id_rows.size), "n_zs_ood": int(ood_rows.size)}],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    summary = {
        "eval_source": a.eval_source, "size": a.size, "fold": a.fold,
        "n_eval_cells_total": int(n_cells),
        "n_zs_perturbations": int(zs_perts.size),
        "class_space": a.class_space, "n_classes": len(classes),
        "eval_cap": a.eval_cap,
        "eval_cap_unit": "cells per (dataset, perturbation)",
        "zs_id": {"cells": int(id_rows.size), "cells_uncapped": int(id_rows_all.size),
                  "perts": int(np.unique(lab[id_local]).size),
                  "datasets": int(np.unique(ds[id_local]).size)},
        "zs_ood": {"cells": int(ood_rows.size),
                   "perts": int(np.unique(lab[ood_local]).size) if ood_local.size else 0,
                   "datasets": sorted(np.unique(ds[ood_local]).tolist()) if ood_local.size else []},
        "training_pool_perts": int(pool.size),
        "train_cells_consumed": int(n_train_cells),
        "overlap_with_training_pool": 0,
        "classes_dropped_no_embedding": int(n_dropped_cls),
        "cells_dropped_no_embedding": int(n_dropped_cells),
        "embedding_table": a.require_embedding,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log.info("summary: %s", json.dumps(summary["zs_id"]) + " | " + json.dumps(summary["zs_ood"]))


if __name__ == "__main__":
    sys.exit(main())
