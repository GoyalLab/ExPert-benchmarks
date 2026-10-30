"""Emit the list of perturbations to train per-pert cellot models on.

Reads obs[condition] at the train indices from the source h5ad and writes the
sorted unique non-control names to <splits_dir>/perts_train.txt.

Source path and condition column are read from the sibling metadata.json unless
explicitly overridden via --source / --condition_col.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def read_obs_column(source, col, idx):
    """Read obs[`col`] at 0-based row `idx` via h5py — just that one column,
    not the whole obs. anndata's backed="r" still materializes ALL of obs into
    memory (only X is lazy); at full (18M rows) that's slow and GBs of RAM.
    Handles both anndata encodings: categorical (a group of codes+categories)
    and a plain string/object dataset."""
    with h5py.File(source, "r") as f:
        node = f[f"obs/{col}"]
        if isinstance(node, h5py.Group):                       # categorical
            cats = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                    for c in node["categories"][:]]
            codes = node["codes"][:]                           # one small int/cell
            return [cats[c] for c in codes[idx] if c >= 0]     # c < 0 == NaN
        raw = node[np.sort(np.unique(idx))]                    # plain string column
        return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits_dir",
        required=True,
        help="Fold directory, e.g. data/store/splits/small/fold_0",
    )
    ap.add_argument(
        "--source",
        default=None,
        help="Override source h5ad path (default: metadata.json source_path).",
    )
    ap.add_argument(
        "--condition_col",
        default=None,
        help="Override condition column (default: metadata.json cls_col).",
    )
    ap.add_argument("--control_value", default="control")
    ap.add_argument(
        "--out",
        default=None,
        help="Output path for perts_train.txt (default: <splits_dir>/perts_train.txt).",
    )
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)

    source = args.source
    cond_col = args.condition_col
    if source is None or cond_col is None:
        meta_path = splits_dir.parent / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found; pass --source and --condition_col."
            )
        meta = json.loads(meta_path.read_text())
        if source is None:
            source = meta["source_path"]
        if cond_col is None:
            # The rebuilt splits DB dropped `cls_col` from metadata.json; the
            # perturbation column is `perturbation` in every source adata.
            cond_col = meta.get("cls_col", "perturbation")

    train_idx = np.loadtxt(splits_dir / "id_train.txt.gz", dtype=int)
    vals = read_obs_column(source, cond_col, train_idx)
    perts = sorted({v for v in vals if v != args.control_value})

    out = Path(args.out) if args.out else splits_dir / "perts_train.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(perts) + "\n")
    print(f"Wrote {len(perts)} perturbations to {out}")


if __name__ == "__main__":
    main()
