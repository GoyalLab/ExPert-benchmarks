#!/usr/bin/env python3
"""Dump the source adata's per-cell metadata (dataset, perturbation, efficiency)
to <out>/data/source.obs.csv — the per-cell covariate table the analysis suite
(axis1 dataset efficiency, synthesis) reads. Not part of collect_results.py's
metrics, so the collect rule emits it before running the analysis. h5py-only
(reads just the obs columns, never X), so it's memory-light on the huge source.
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd


def _col(og, name):
    n = og[name]
    if isinstance(n, h5py.Group):                    # categorical
        cats = np.array([c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                         for c in n["categories"][:]])
        return cats[n["codes"][:]]
    return np.array([v.decode() if isinstance(v, (bytes, bytearray)) else v for v in n[:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source.h5ad")
    ap.add_argument("--out", required=True, help="destination CSV")
    ap.add_argument("--cols", nargs="+", default=["dataset", "perturbation", "efficiency"])
    args = ap.parse_args()

    with h5py.File(args.source, "r") as f:
        og = f["obs"]
        cols = {c: _col(og, c) for c in args.cols if c in og}
        ik = og.attrs.get("_index", "_index")
        ik = ik.decode() if isinstance(ik, bytes) else ik
        idx = _col(og, ik) if ik in og else np.arange(len(next(iter(cols.values()))))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(cols, index=idx).to_csv(args.out)
    print(f"wrote {args.out} ({len(idx)} cells, cols={list(cols)})")


if __name__ == "__main__":
    main()
