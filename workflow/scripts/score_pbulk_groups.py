#!/usr/bin/env python3
"""Pseudobulk (group-level) ranks for every method, WITHOUT re-running collect or any method.

Streams the logits collect scored (its discover_files / apply_consistency / load_one_file: same class
union, same tie-break), pools the cells of each (fold, split, dataset, perturbation) group into ONE
score vector and ranks it -- collect's score_group_level 'mean_log_prob', i.e. the group-level
assignment of the fingerprinting paper -- and writes one row per group:

  agg        'mean'  = plain mean of the cells' log-probs
             'effw'  = mean weighted by clip(efficiency, 0) (groups with zero total weight dropped)
  thr        minimum cell efficiency kept before pooling (NaN = all cells; else >= 1 / 2 / 3, as the sweep)
  rank       0-based rank of the true class, ties averaged (collect's _true_rank)
  pred       top-1 class (collect's tie-broken argmax)

Metrics (per-perturbation F1 over PRESENT classes, recall@k, MRR, ...) are computed downstream, so the
1/C macro-F1 of collect's own group table never enters. One parquet per method under --out-dir.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_results as cr  # noqa: E402

log = logging.getLogger("score_pbulk_groups")


def score_file(fd, thresholds, min_cells: int) -> pd.DataFrame | None:
    cls = np.asarray(fd.classes, dtype=object)
    ci = {c: i for i, c in enumerate(cls)}
    t = np.fromiter((ci.get(c, -1) for c in fd.labels), count=len(fd.labels), dtype=np.int64)
    ok = t >= 0
    if not ok.any():
        return None
    lp = fd.log_probs[ok]
    lp = np.where(np.isfinite(lp), lp, -1e30).astype(np.float64)    # -inf padding stays the floor
    t = t[ok]
    obs = fd.obs.iloc[np.flatnonzero(ok)].reset_index(drop=True)
    ds = obs[f"{cr.OBS_PREFIX}dataset"].astype(str).to_numpy()
    eff = pd.to_numeric(obs[f"{cr.OBS_PREFIX}efficiency"], errors="coerce").to_numpy(np.float64)
    w = np.nan_to_num(np.clip(eff, 0, None))
    groups = pd.Series(np.arange(len(t))).groupby([ds, t]).indices
    vecs, meta = [], []
    for (d, ti), ii in groups.items():
        for thr in thresholds:
            jj = ii if np.isnan(thr) else ii[np.nan_to_num(eff[ii], nan=-np.inf) >= thr]
            if len(jj) < min_cells:
                continue
            vecs.append(lp[jj].mean(0)); meta.append((d, ti, thr, "mean", len(jj)))
            if w[jj].sum() > 0:
                vecs.append(np.average(lp[jj], axis=0, weights=w[jj])); meta.append((d, ti, thr, "effw", len(jj)))
    if not vecs:
        return None
    V = np.stack(vecs)
    ti = np.array([m[1] for m in meta])
    rank = cr._true_rank(V, ti)
    pred = cr._argtopk_chunked(V, 1)[:, 0]
    return pd.DataFrame({
        "method": fd.meta.method, "fold": fd.meta.fold, "split": fd.meta.split,
        "dataset": [m[0] for m in meta], "label": cls[ti], "thr": [m[2] for m in meta],
        "agg": [m[3] for m in meta], "n_cells": [m[4] for m in meta],
        "rank": rank.astype(np.float32), "pred": cls[pred], "C": len(cls),
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--dataset-size", required=True)
    ap.add_argument("--method-depth", type=int, required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label-col", default="perturbation")
    ap.add_argument("--tie-break-seed", type=int, default=0, help="must match the collect run")
    ap.add_argument("--thresholds", type=float, nargs="*", default=[1.0, 2.0, 3.0],
                    help="efficiency floors for the sweep (all cells is always included)")
    ap.add_argument("--min-cells", type=int, default=1)
    ap.add_argument("--methods", nargs="*", default=None, help="restrict (smoke)")
    ap.add_argument("--max-files", type=int, default=None, help="per method (smoke)")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    cr.TIE_BREAK_SEED = a.tie_break_seed
    thresholds = [np.nan, *a.thresholds]

    metas, cols_by_path = cr.discover_files(a.source, method_depth=a.method_depth,
                                            dataset_size=a.dataset_size, fold=None)
    metas, unions = cr.apply_consistency(metas, cols_by_path)
    keep = {cr._canonical_split(s) for s in a.splits}
    metas = [m for m in metas if m.split in keep and (not a.methods or m.method in a.methods)]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list] = {}
    for m in metas:
        by_method.setdefault(m.method, []).append(m)
    mine = sorted(by_method)[a.task_index::a.n_tasks]
    todo = {k: by_method[k] for k in mine if not (a.out_dir / f"{k}.parquet").exists()}
    log.info("task %d/%d: %s (%d to go)", a.task_index, a.n_tasks, mine, len(todo))
    if not todo:
        return
    obs, size_dir, _ = cr.load_obs(a.splits_dir, a.dataset_size, a.label_col)
    for method, ms in todo.items():
        parts = []
        for i, meta in enumerate(ms[:a.max_files], 1):
            log.info("[%s %d/%d] %s", method, i, len(ms), meta.path)
            fd = cr.load_one_file(meta, obs=obs, size_dir=size_dir, label_col=a.label_col,
                                  obs_cols=("dataset", "efficiency"),
                                  union_sorted=unions[(meta.direction, meta.method)])
            if fd is None:
                continue
            df = score_file(fd, thresholds, a.min_cells)
            if df is not None:
                parts.append(df)
            del fd
        if parts:
            out = pd.concat(parts, ignore_index=True)
            for c in ("method", "fold", "split", "dataset", "label", "pred", "agg"):
                out[c] = out[c].astype("category")
            tmp = a.out_dir / f".{method}.parquet.tmp"
            out.to_parquet(tmp, index=False)
            tmp.rename(a.out_dir / f"{method}.parquet")
            log.info("[%s] wrote %d group rows", method, len(out))


if __name__ == "__main__":
    main()
