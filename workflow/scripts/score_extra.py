#!/usr/bin/env python3
"""Add-on metrics for a finished collect, WITHOUT re-running it.

Streams the same logits collect_results.py scored -- through its own discover_files /
apply_consistency / load_one_file / _split_by_dataset, so the class space, tie-break
order and per-dataset views are identical -- and writes ONLY the new columns, one row
per (method, split, direction, fold) view, keyed exactly like weighted_metrics.csv:

  auroc_ovr                     macro one-vs-rest AUROC (sklearn multi_class="ovr",
                                average="macro"): per class, do ITS cells score higher
                                on ITS column than the other cells? Mean over classes
                                present. Same cell set as score_one (all labelled cells).
  weighted_auroc_ovr            the same with efficiency weights (weighted Mann-Whitney:
                                each positive/negative pair counts w_i * w_j). Same cell
                                set as score_weighted_one (finite weight > 0).
  balanced_weighted_macro_f1    per-class efficiency-weighted F1, mean over classes present
                                (collect's weighted_macro_f1 weights classes by support).
  balanced_weighted_mrr         per-class efficiency-weighted MRR, mean over classes
                                present -- the per-perturbation twin of weighted_mrr.
  weighted_mrr_check            weighted_mrr / balanced_weighted_recall@10 recomputed
  balanced_weighted_recall@10_check   here; must equal the cached weighted_metrics.csv
                                values (a key/cell-set sanity check).

Rows are appended per file and the output is RESUMABLE: files whose path is already in
--out are skipped, so a timed-out job continues where it stopped.
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

log = logging.getLogger("score_extra")
COLS = ["method", "split", "direction", "fold", "n", "auroc_ovr", "n_weighted",
        "weighted_auroc_ovr", "balanced_weighted_mrr", "weighted_mrr_check",
        "balanced_weighted_recall@10_check", "balanced_weighted_macro_f1",
        "weighted_macro_f1_check", "path"]


def _weighted_ovr(scores: np.ndarray, true_idx: np.ndarray, w: np.ndarray,
                  present: np.ndarray) -> float:
    """Macro OvR AUROC with sample weights; ties count one half."""
    aucs = []
    w_tot = w.sum()
    for c in np.flatnonzero(present):
        pos = true_idx == c
        w_pos = w[pos].sum()
        w_neg = w_tot - w_pos
        if w_pos <= 0 or w_neg <= 0:
            continue
        _, inv = np.unique(scores[:, c], return_inverse=True)     # ascending tie groups
        pw = np.bincount(inv, weights=np.where(pos, w, 0.0))
        nw = np.bincount(inv, weights=np.where(pos, 0.0, w))
        neg_below = np.cumsum(nw) - nw
        aucs.append(float((pw * (neg_below + 0.5 * nw)).sum() / (w_pos * w_neg)))
    return float(np.mean(aucs)) if aucs else float("nan")


def score_view(fd, weight_col: str) -> dict | None:
    col_idx = {c: i for i, c in enumerate(fd.classes)}
    n_classes = len(fd.classes)
    true_all = np.fromiter((col_idx.get(c, -1) for c in fd.labels),
                           count=len(fd.labels), dtype=np.int64)
    out = cr._meta_row(fd)
    lab = true_all >= 0
    if lab.sum() == 0 or n_classes < 2:
        return None
    lp, ti = fd.log_probs[lab], true_all[lab]
    sup = np.bincount(ti, minlength=n_classes)
    out["n"] = int(lab.sum())
    out["auroc_ovr"] = _weighted_ovr(lp, ti, np.ones(len(ti)), sup > 0)

    resolved = weight_col if weight_col in fd.obs.columns else f"{cr.OBS_PREFIX}{weight_col}"
    if resolved in fd.obs.columns:
        w_all = pd.to_numeric(fd.obs[resolved], errors="coerce").to_numpy()
        valid = lab & np.isfinite(w_all) & (w_all > 0)
        if valid.any():
            lpw, tiw, w = fd.log_probs[valid], true_all[valid], w_all[valid].astype(np.float64)
            wsum = np.bincount(tiw, weights=w, minlength=n_classes)
            present = wsum > 0
            out["n_weighted"] = int(valid.sum())
            out["weighted_auroc_ovr"] = _weighted_ovr(lpw, tiw, w, present)
            rr = 1.0 / (cr._true_rank(lpw, tiw) + 1.0)
            out["weighted_mrr_check"] = float(np.average(rr, weights=w))
            num = np.bincount(tiw, weights=rr * w, minlength=n_classes)
            out["balanced_weighted_mrr"] = float((num[present] / wsum[present]).mean())
            top = cr._argtopk_chunked(lpw, 10)
            hit = (top == tiw[:, None]).any(axis=1).astype(np.float64)
            hw = np.bincount(tiw, weights=hit * w, minlength=n_classes)
            out["balanced_weighted_recall@10_check"] = float((hw[present] / wsum[present]).mean())
            # per-perturbation macro-F1: per-class efficiency-weighted F1, ONE vote per class
            # present (collect's weighted_macro_f1 averages them by weighted support instead)
            _, _, f1, sup = cr._weighted_perclass_prf_from_idx(tiw, top[:, 0], w, n_classes)
            keep = (sup > 0) & np.isfinite(f1)
            out["balanced_weighted_macro_f1"] = float(f1[keep].mean()) if keep.any() else 0.0
            out["weighted_macro_f1_check"] = cr._weighted_macro_f1_from_idx(tiw, top[:, 0], w, n_classes)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--dataset-size", required=True)
    ap.add_argument("--method-depth", type=int, required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="CSV to append to (resumable)")
    ap.add_argument("--label-col", default="perturbation")
    ap.add_argument("--efficiency-col", default="efficiency")
    ap.add_argument("--tie-break-seed", type=int, default=0, help="must match the collect run")
    ap.add_argument("--classes", type=Path, default=None, help="must match the collect run")
    ap.add_argument("--methods", nargs="*", default=None, help="smoke: restrict to these methods")
    ap.add_argument("--max-files", type=int, default=None, help="smoke: stop after N files")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    cr.TIE_BREAK_SEED = args.tie_break_seed
    if args.classes:
        cr.FIXED_CLASSES = [l.strip() for l in args.classes.read_text().splitlines() if l.strip()]

    metas, cols_by_path = cr.discover_files(args.source, method_depth=args.method_depth,
                                            dataset_size=args.dataset_size, fold=None)
    metas, unions = cr.apply_consistency(metas, cols_by_path)
    keep = {cr._canonical_split(s) for s in args.splits}
    metas = [m for m in metas if m.split in keep
             and (not args.methods or m.method in args.methods)]

    done = set()
    if args.out.exists():
        done = set(pd.read_csv(args.out, usecols=["path"])["path"].astype(str))
    todo = [m for m in metas if str(m.path) not in done][:args.max_files]
    log.info("%d files, %d already scored, %d to go", len(metas), len(metas) - len(todo), len(todo))
    if not todo:
        return
    obs, size_dir, _ = cr.load_obs(args.splits_dir, args.dataset_size, args.label_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for i, meta in enumerate(todo, 1):
        log.info("[%d/%d] %s", i, len(todo), meta.path)
        fd_full = cr.load_one_file(meta, obs=obs, size_dir=size_dir, label_col=args.label_col,
                                   obs_cols=None, union_sorted=unions[(meta.direction, meta.method)])
        if fd_full is None:
            continue
        rows = [r for fd in cr._split_by_dataset(fd_full)
                if (r := score_view(fd, args.efficiency_col)) is not None]
        if rows:
            df = pd.DataFrame(rows)
            df["path"] = str(meta.path)
            df = df.reindex(columns=COLS)          # fixed header across appends
            df.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
        del fd_full


if __name__ == "__main__":
    main()
