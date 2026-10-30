# Concatenate per-perturbation-chunk scGPT logits.h5 files into one logits.h5.
#
# Each chunk holds the same test cells (rows, identical obs_names order) but a
# disjoint slice of candidate perturbations (columns). Chunking eval by pert is a
# pure column split, so concat = hstack the /logits columns + concat /perts.
#
# The output must carry /columns and /idx as well: collect_results.py reads class
# names ONLY from H5_COLUMN_KEYS (columns, classes, ...; NOT perts) and aligns rows
# by /idx, falling back to POSITION without it. This script originally wrote only
# perts/obs_names/logits -- it had never run (no _eval_chunks dir existed anywhere)
# and would have broken every chunked scGPT result at collect.
# Written row-block by row-block so the full (n_cells x n_perts) matrix never
# lives in memory (matters for the `full` split: ~1e6 cells x ~9k perts).
#
#   scgpt_concat_logits.py --out <logits.h5> --chunks <chunk0.h5> <chunk1.h5> ...
import argparse
from pathlib import Path

import h5py
import numpy as np

ROW_BLOCK = 20000


def _as_str(arr):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in arr], dtype=object)


def main():
    ap = argparse.ArgumentParser(description="hstack per-pert-chunk scGPT logits.h5 files.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", nargs="+", required=True)
    args = ap.parse_args()

    files = [h5py.File(p, "r") for p in args.chunks]
    try:
        obs_ref = _as_str(files[0]["obs_names"][:])
        n_cells = len(obs_ref)
        perts = [_as_str(f["perts"][:]) for f in files]
        widths = [f["logits"].shape[1] for f in files]
        for p, f in zip(args.chunks, files):
            obs = _as_str(f["obs_names"][:])
            if len(obs) != n_cells or not np.array_equal(obs, obs_ref):
                raise ValueError(f"obs_names (cells) of {p} differ from the first chunk — "
                                 "chunks must cover the same test cells in the same order.")
            if f["logits"].shape[0] != n_cells:
                raise ValueError(f"{p}: logits rows != obs_names length.")
        # /idx (source row index per cell) must be identical across chunks, like obs_names.
        idx_ref = files[0]["idx"][:] if "idx" in files[0] else None
        for p, f in zip(args.chunks, files):
            if ("idx" in f) != (idx_ref is not None) or (
                    idx_ref is not None and not np.array_equal(f["idx"][:], idx_ref)):
                raise ValueError(f"/idx of {p} differs from the first chunk.")
        full_perts = np.concatenate(perts) if perts else np.array([], dtype=object)
        total_perts = int(sum(widths))

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        str_dt = h5py.string_dtype()
        with h5py.File(args.out, "w") as out:
            out.create_dataset("perts", data=full_perts, dtype=str_dt)
            out.create_dataset("columns", data=full_perts, dtype=str_dt)
            out.create_dataset("obs_names", data=obs_ref, dtype=str_dt)
            if idx_ref is not None:
                out.create_dataset("idx", data=idx_ref)
            dset = out.create_dataset(
                "logits", shape=(n_cells, total_perts), dtype="float32",
                compression="lzf", chunks=True,
            )
            for r0 in range(0, n_cells, ROW_BLOCK):
                r1 = min(r0 + ROW_BLOCK, n_cells)
                col = 0
                for f, w in zip(files, widths):
                    dset[r0:r1, col:col + w] = f["logits"][r0:r1, :]
                    col += w
        print(f"Concatenated {len(files)} chunks -> {args.out} "
              f"({n_cells} cells x {total_perts} perts)")
    finally:
        for f in files:
            f.close()


if __name__ == "__main__":
    main()
