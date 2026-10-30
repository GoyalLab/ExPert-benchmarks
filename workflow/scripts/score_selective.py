#!/usr/bin/env python3
"""Per-cell scores for selective-prediction / matched-cell analyses, WITHOUT re-running collect.

Streams the same logits collect_results.py scored -- through its own discover_files /
apply_consistency / load_one_file, so class space, tie-break order and cell alignment are
identical -- and writes one compact row per (method, fold, split, cell):

  row        source-adata row (the cell ID shared by every method)
  dataset, label, efficiency
  rank       0-based rank of the true class, ties averaged (collect's _true_rank)
  pred       top-1 class (collect's tie-broken argmax)
  conf       top-1 minus top-2 log-prob: the method's own confidence, for coverage curves
  abstain    fingerprint* only: the cell got the method's cell-independent default row (no credible
             set in any model -> softmax of a flat row -> the per-pert z-score turns it into the
             same ranking for every such cell). False for every other method.

One parquet per method under --out-dir (resumable: methods already written are skipped).
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

log = logging.getLogger("score_selective")
ROW = "_row"


def default_row_mask(lp: np.ndarray) -> np.ndarray:
    """Rows identical to the file's most common row (a flat, cell-independent default)."""
    key = np.round(np.where(np.isfinite(lp), lp, -1e9), 4)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m = cnt.argmax()
    return (inv == m) if cnt[m] > 1 else np.zeros(len(lp), bool)


def score_file(fd, is_fp: bool) -> pd.DataFrame | None:
    col_idx = {c: i for i, c in enumerate(fd.classes)}
    t = np.fromiter((col_idx.get(c, -1) for c in fd.labels), count=len(fd.labels), dtype=np.int64)
    ok = t >= 0
    if not ok.any():
        return None
    lp, t = fd.log_probs[ok], t[ok]
    obs = fd.obs.iloc[np.flatnonzero(ok)].reset_index(drop=True)
    rank = cr._true_rank(lp, t)
    pred = cr._argtopk_chunked(lp, 1)[:, 0]
    top2 = -np.partition(-np.where(np.isfinite(lp), lp, -1e30), 1, axis=1)[:, :2]
    conf = (top2[:, 0] - top2[:, 1]).astype(np.float32)
    abst = default_row_mask(lp) if is_fp else np.zeros(len(lp), bool)
    cls = np.asarray(fd.classes, dtype=object)
    return pd.DataFrame({
        "method": fd.meta.method, "fold": fd.meta.fold, "split": fd.meta.split,
        "row": obs[f"{cr.OBS_PREFIX}{ROW}"].to_numpy(np.int64),
        "dataset": obs[f"{cr.OBS_PREFIX}dataset"].astype(str).to_numpy(),
        "label": cls[t], "pred": cls[pred],
        "efficiency": pd.to_numeric(obs[f"{cr.OBS_PREFIX}efficiency"], errors="coerce").to_numpy(np.float32),
        "rank": rank.astype(np.float32), "conf": conf, "abstain": abst,
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
    ap.add_argument("--methods", nargs="*", default=None, help="restrict (smoke)")
    ap.add_argument("--max-files", type=int, default=None, help="per method (smoke)")
    ap.add_argument("--task-index", type=int, default=0, help="SLURM array task: takes methods[i::n]")
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    cr.TIE_BREAK_SEED = a.tie_break_seed

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
    log.info("task %d/%d: %s (%d to go; %d methods in total)", a.task_index, a.n_tasks,
             mine, len(todo), len(by_method))
    if not todo:
        return
    obs, size_dir, _ = cr.load_obs(a.splits_dir, a.dataset_size, a.label_col)
    obs[ROW] = np.arange(len(obs), dtype=np.int64)            # cell ID shared across methods
    for method, ms in todo.items():
        parts = []
        for i, meta in enumerate(ms[:a.max_files], 1):
            log.info("[%s %d/%d] %s", method, i, len(ms), meta.path)
            fd = cr.load_one_file(meta, obs=obs, size_dir=size_dir, label_col=a.label_col,
                                  obs_cols=(ROW, "dataset", "efficiency"),
                                  union_sorted=unions[(meta.direction, meta.method)])
            if fd is None:
                continue
            df = score_file(fd, method.startswith("fingerprint"))
            if df is not None:
                parts.append(df)
            del fd
        if parts:
            out = pd.concat(parts, ignore_index=True)
            for c in ("method", "fold", "split", "dataset", "label", "pred"):
                out[c] = out[c].astype("category")
            tmp = a.out_dir / f".{method}.parquet.tmp"
            out.to_parquet(tmp, index=False)
            tmp.rename(a.out_dir / f"{method}.parquet")      # atomic: a partial method is never "done"
            log.info("[%s] wrote %d rows (abstain %.3f)", method, len(out), out.abstain.mean())


if __name__ == "__main__":
    main()
