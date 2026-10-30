#!/usr/bin/env python3
"""Rewrite an h5ad's X to the canonical store layout (uncompressed, int32 indices,
64k chunks) by streaming it sequentially; everything else is copied verbatim.

Why: row-wise readers (scGPT batches, centroid passes) decompress a whole chunk per
access. 1M-element gzip chunks read at 49 rows/s vs 1,768-1,901 for AnnData's default
or an uncompressed store -- see H5adWriter in scgpt_pad_targets.py.

  python utils/h5ad_relayout.py SRC.h5ad DST.h5ad
"""
import h5py, numpy as np, sys, time
src, dst = sys.argv[1], sys.argv[2]
t0 = time.time()
with h5py.File(src, 'r') as f, h5py.File(dst, 'w') as g:
    for k, v in f.attrs.items(): g.attrs[k] = v
    for name in f:
        if name != 'X': f.copy(f[name], g, name=name)
    X = f['X']; Y = g.create_group('X')
    for k, v in X.attrs.items(): Y.attrs[k] = v
    n = X['data'].shape[0]; ncol = int(X.attrs['shape'][1]); assert ncol < 2**31
    d = Y.create_dataset('data', shape=(n,), dtype=X['data'].dtype, chunks=(1 << 16,))
    i = Y.create_dataset('indices', shape=(n,), dtype=np.int32, chunks=(1 << 16,))
    step = 1 << 26                                   # 64M elements, aligned to source chunks
    for s in range(0, n, step):
        e = min(s + step, n)
        d[s:e] = X['data'][s:e]; i[s:e] = X['indices'][s:e].astype(np.int32)
        if (s // step) % 10 == 0: print(f'  {e:,}/{n:,}  {time.time()-t0:.0f}s', flush=True)
    Y.create_dataset('indptr', data=X['indptr'][:])
print(f'wrote {dst} in {time.time()-t0:.0f}s', flush=True)
