#!/usr/bin/env python3
"""Precompute per-(size, fold, dataset) training perturbation counts for the
per-dataset fingerprint DAG's chunk sizing.

The per-dataset path trains one model per dataset, chunking that dataset's perts
(deg/fit) exactly like the shared path chunks the global perts. Pert counts vary
wildly (≈80 → ≈9k), so a single fixed chunk count would either under-chunk the
genome-wide screens or spawn thousands of empty no-op jobs on the small datasets.
This emits a JSON the Snakefile reads to size chunks per dataset:

    { "<size>": { "<fold>": { "<dataset>": n_train_perts, ... }, ... }, ... }

Reads only the obs dataset+perturbation columns via h5py (codes), so it's cheap
even at full (18M cells). Re-run whenever the splits change:

    python precompute_perdataset_perts.py --splits-root <dir> --out <json> \
        [--sizes small full] [--control-value control]
"""
import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np


def read_obs_codes(source, col):
    """Return (values_or_codes, categories_or_None) for obs[col] via h5py."""
    with h5py.File(source, "r") as f:
        node = f[f"obs/{col}"]
        if isinstance(node, h5py.Group):                       # categorical
            cats = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                    for c in node["categories"][:]]
            return node["codes"][:], cats
        raw = node[:]
        return np.array([v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                         for v in raw]), None


def load_idx(p):
    with gzip.open(p, "rt") as fh:
        return np.fromiter((int(x) for x in fh if x.strip()), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", default=["small", "full"])
    ap.add_argument("--dataset-col", default="dataset")
    ap.add_argument("--condition-col", default=None,
                    help="default: metadata cls_col")
    ap.add_argument("--control-value", default="control")
    args = ap.parse_args()

    root = Path(args.splits_root)
    result = {}
    for size in args.sizes:
        size_dir = root / size
        meta_p = size_dir / "metadata.json"
        if not meta_p.exists():
            print(f"[skip] {size}: no metadata.json")
            continue
        meta = json.loads(meta_p.read_text())
        source = meta["source_path"]
        cond_col = args.condition_col or meta.get("cls_col", "perturbation")
        ds_codes, ds_cats = read_obs_codes(source, args.dataset_col)
        pr_codes, pr_cats = read_obs_codes(source, cond_col)
        # Work on integer codes (not materialized strings) so 12M+ train cells at
        # full stay light. Map dataset code -> name only for the ~21 datasets.
        ds_name = (lambda c: ds_cats[c]) if ds_cats is not None else (lambda c: str(c))
        ctrl_code = (pr_cats.index(args.control_value)
                     if pr_cats is not None and args.control_value in pr_cats else None)
        result[size] = {}
        for fold_dir in sorted(size_dir.glob("fold_*")):
            ti = load_idx(fold_dir / "id_train.txt.gz")
            dsi, pri = ds_codes[ti], pr_codes[ti]
            keep = (pri != ctrl_code) if ctrl_code is not None else (pri != args.control_value)
            dsi, pri = dsi[keep], pri[keep]
            counts = {}
            for dc in np.unique(dsi):
                counts[ds_name(dc)] = int(len(np.unique(pri[dsi == dc])))
            result[size][fold_dir.name] = dict(sorted(counts.items()))
            print(f"[{size}/{fold_dir.name}] {len(counts)} datasets, "
                  f"perts {min(counts.values())}..{max(counts.values())}")
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
