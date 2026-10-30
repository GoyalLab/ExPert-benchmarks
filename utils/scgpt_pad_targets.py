#!/usr/bin/env python3
"""Write scGPT-only copies of the perturbed + control adata with extra columns for
perturbation target genes that are absent from the expression panel.

WHY
  scGPT encodes a perturbation as a flag on its target gene's token, so
  get_candidate_perturbations() drops any perturbation whose target gene is not in
  adata.var. On the small DB that is 147 of 200 (only 53 survive); on ctx7 it is
  882 of 1,157 (275 survive). The collector then pads scGPT's logits back up to the
  full label space, scoring it wrong on classes it could never express.
  min_train_cells_per_perturbation is NOT the cause -- it drops 0.

THREE WAYS TO FILL A MISSING TARGET COLUMN
  --real-from STORE   PREFERRED. Splice the gene's REAL measurements out of a
      wider-panel store that shares this store's cells (mapped by obs_names, so a
      row subset is fine). ctx7: 437 of the 882 missing targets are measured in the
      16,092-gene store 963044872df9d6a8 -> 712/1,157 perturbations become
      representable instead of 275, with a genuine, noisy, learnable knockdown
      signal (measured own/other ratio median 0.247, 93% dip below 0.8) rather than
      a synthetic one. Real columns are NOT a leak: they are exactly what every
      method would see if the panel were wider, and scoring excludes them anyway
      (prediction.score_n_genes pins the metric to the real panel).
  Poisson (default)   cells where perturbation == g : Poisson(lambda_kd);
      all other cells : Poisson(lambda_bg), lambda_bg = global mean of the real
      panel. Used for targets no store measures (ctx7: the remaining 445).
      *** LABEL-DERIVED, INCLUDING FOR TEST CELLS -- a model can read the answer  ***
      *** out of its input for these classes. Measured worth of that leak on the  ***
      *** small store: 2.8x recall@10. Keep prediction.score_n_genes set to the   ***
      *** real panel width so these columns never enter a metric.                 ***
  --zero-fill         all-zero columns. scGPT only needs the column to exist so the
      gene has a position to flag (`pert_flags[:, pert_gene_idx] = 1`); the value
      need not carry anything. No leak, but no training signal either (measured: it
      does not recover the padded classes, recall@10 0.120 vs 0.123 for masking).

  Real genes are always byte-identical to the canonical store and stay in their
  original order, so the first n_panel columns are what every other method sees and
  score_n_genes=<n_panel> keeps the comparison fair.
"""
import argparse, h5py, numpy as np, scipy.sparse as sp, os, json, time
import pandas as pd

OBS_KEEP = ("perturbation", "dataset", "efficiency", "is_ctrl", "label", "qc_pass", "barcode")


def read_index(f, grp):
    o = f[grp]; k = "_index" if "_index" in o else list(o.keys())[0]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in o[k][:]])

def read_col(f, name, n):
    d = f["obs"][name]
    if isinstance(d, h5py.Group):
        cats = [x.decode() if isinstance(x, bytes) else str(x) for x in d["categories"][:]]
        return np.asarray(cats, dtype=object)[d["codes"][:]].astype(str)
    out = np.empty(n, dtype=object)
    for s in range(0, n, 200_000):
        blk = d[s:s+200_000]
        out[s:s+len(blk)] = [x.decode() if isinstance(x, bytes) else str(x) for x in blk]
    return out.astype(str)


def _cat(vals):
    """String column -> (categories, codes) in the narrowest int that fits."""
    cats, codes = np.unique(np.asarray(vals).astype(str), return_inverse=True)
    for dt in (np.int8, np.int16, np.int32):
        if len(cats) <= np.iinfo(dt).max:
            return cats, codes.astype(dt)
    return cats, codes.astype(np.int64)


class H5adWriter:
    """Append CSR row-blocks straight to disk, AnnData's on-disk layout written by hand.

    The previous implementation kept every block in RAM and sp.vstack'd at the end:
    85.6 GB peak for ctx7 out of a 96 GB allocation, and --real-from adds a dense
    (n_cells x n_real) buffer on top of that. Streaming keeps the whole build in a
    few GB, so the allocation is set by the donor read, not the output.
    """
    STR = h5py.string_dtype()

    # LAYOUT MATTERS MORE THAN SIZE. scGPT reads ROWS (random batches in training, 512-row
    # blocks for centroids), and h5py decompresses a whole chunk per access with only a
    # 1MB chunk cache. The first ctx7 build used 1M-element gzip chunks and read at 49
    # rows/s -- 36x slower than AnnData's own 166k-element gzip chunks (1,768 rows/s) and
    # the uncompressed canonical store (1,901 rows/s): ~12h per eval job just for the
    # centroid pass, and I/O-bound training. Defaults now match the canonical store:
    # uncompressed, int32 indices, 64k-element chunks. Disk is cheap; row reads
    # are not.
    def __init__(self, path, n_cols, dtype, chunk=1 << 16, compression=None):
        self.n_cols, self.n_rows, self.nnz = n_cols, 0, 0
        self.comp = compression
        self.f = h5py.File(path, "w")
        self.f.attrs["encoding-type"] = "anndata"; self.f.attrs["encoding-version"] = "0.1.0"
        X = self.f.create_group("X")
        X.attrs["encoding-type"] = "csr_matrix"; X.attrs["encoding-version"] = "0.1.0"
        mk = lambda name, dt: X.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt,
                                               chunks=(chunk,), compression=compression)
        self.data = mk("data", dtype)
        self.indices = mk("indices", np.int32 if n_cols < 2**31 else np.int64)
        self.indptr = [np.zeros(1, np.int64)]
        self._X = X

    def append(self, blk):
        blk = blk.tocsr(); blk.sort_indices()
        k = blk.data.size
        self.data.resize((self.nnz + k,)); self.data[self.nnz:] = blk.data
        self.indices.resize((self.nnz + k,)); self.indices[self.nnz:] = blk.indices
        self.indptr.append(blk.indptr[1:].astype(np.int64) + self.nnz)
        self.nnz += k; self.n_rows += blk.shape[0]

    def _frame(self, name, index, cols):
        g = self.f.create_group(name)
        g.attrs["_index"] = "_index"
        g.attrs.create("column-order", np.array(list(cols), dtype=object), dtype=self.STR)
        g.attrs["encoding-type"] = "dataframe"; g.attrs["encoding-version"] = "0.2.0"
        d = g.create_dataset("_index", data=np.asarray(index, dtype=object), dtype=self.STR,
                             compression=self.comp)
        d.attrs["encoding-type"] = "string-array"; d.attrs["encoding-version"] = "0.2.0"
        for c, v in cols.items():
            cats, codes = _cat(v)
            cg = g.create_group(c)
            cg.attrs["encoding-type"] = "categorical"; cg.attrs["encoding-version"] = "0.2.0"
            cg.attrs["ordered"] = False
            cd = cg.create_dataset("categories", data=cats.astype(object), dtype=self.STR)
            cd.attrs["encoding-type"] = "string-array"; cd.attrs["encoding-version"] = "0.2.0"
            cc = cg.create_dataset("codes", data=codes, compression=self.comp)
            cc.attrs["encoding-type"] = "array"; cc.attrs["encoding-version"] = "0.2.0"

    def finish(self, obs_names, obs_cols, var_names):
        self._X.attrs["shape"] = np.array([self.n_rows, self.n_cols], np.int64)
        self._X.create_dataset("indptr", data=np.concatenate(self.indptr), dtype=np.int64,
                               compression=self.comp)
        self._frame("obs", obs_names, obs_cols)
        self._frame("var", var_names, {})
        for g in ("layers", "obsm", "obsp", "uns", "varm", "varp"):
            h = self.f.create_group(g)
            h.attrs["encoding-type"] = "dict"; h.attrs["encoding-version"] = "0.1.0"
        self.f.close()
        return self.n_rows, self.n_cols, self.nnz


def splice_real(donor_path, target_obs, missing, chunk, tag):
    """Pull the real measurements of every `missing` gene the donor store has.

    Cells are matched by obs_names, so the donor may hold a superset of the rows
    (ctx7: 2,741,678 of the donor's 19,513,869). Returns (names, dense block) with
    one column per recovered gene, in `missing` order.
    """
    t0 = time.time()
    with h5py.File(donor_path) as f:
        dv = read_index(f, "var")
        dn = f["X"]["indptr"].shape[0] - 1
        xdt = f["X"]["data"].dtype
        pos = pd.Index(read_index(f, "obs")).get_indexer(pd.Index(target_obs))
    if (pos < 0).any():
        raise SystemExit(f"[real-from] {tag}: {int((pos < 0).sum()):,}/{len(pos):,} cells of this "
                         f"store are absent from {donor_path} -- cannot splice")
    have = {g: i for i, g in enumerate(dv)}
    names = [g for g in missing if g in have]
    print(f"[real-from] {tag}: donor {dn:,}x{len(dv):,}  rows mapped in {time.time()-t0:.0f}s  "
          f"recovering {len(names)}/{len(missing)} missing targets as REAL", flush=True)
    if not names:
        return [], None
    dcols = np.array([have[g] for g in names])
    R = np.zeros((len(pos), len(names)), dtype=xdt)
    srt = np.argsort(pos, kind="stable"); rs = pos[srt]
    t0 = time.time(); seen = 0
    with h5py.File(donor_path) as f:
        X = f["X"]; ip = X["indptr"]
        for s in range(0, dn, chunk):
            e = min(s + chunk, dn)
            a, b = np.searchsorted(rs, [s, e])
            if a == b:
                continue
            seg = ip[s:e+1]; lo, hi = int(seg[0]), int(seg[-1])
            blk = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], seg - lo),
                                shape=(e - s, len(dv)))
            R[srt[a:b]] = blk[rs[a:b] - s][:, dcols].toarray()
            del blk
            seen += b - a
            if (s // chunk) % 25 == 0:
                print(f"    donor rows {e:,}/{dn:,}  cells filled {seen:,}/{len(pos):,}  "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"[real-from] {tag}: read donor in {time.time()-t0:.0f}s", flush=True)
    return names, R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="perturbed h5ad")
    p.add_argument("--ctrl", required=True, help="control h5ad")
    p.add_argument("--perts", required=True, help="logits.h5 whose /columns lists the label space")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lambda-kd", type=float, default=0.0, help="Poisson mean inside the target's own cells")
    p.add_argument("--real-from", default=None,
                   help="PREFERRED. Wider-panel perturbed h5ad to splice REAL measurements "
                        "from for every missing target it measures; cells are matched by "
                        "obs_names so a row superset is fine. The rest are synthesised as "
                        "usual. ctx7: 437/882 recovered -> 712/1,157 representable.")
    p.add_argument("--real-from-ctrl", default=None,
                   help="Control h5ad of the same donor store "
                        "(default: database_ctrl.h5ad next to --real-from)")
    p.add_argument("--zero-fill", action="store_true",
                   help="Append the still-missing target genes as ALL-ZERO columns instead of "
                        "synthesising label-dependent values. scGPT needs the column only so "
                        "the gene has a position to flag; no leak, but no signal either.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=50_000)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.real_from and not a.real_from_ctrl:
        a.real_from_ctrl = os.path.join(os.path.dirname(a.real_from), "database_ctrl.h5ad")

    with h5py.File(a.perts) as f:
        perts = [x.decode() if isinstance(x, bytes) else str(x) for x in f["columns"][:]]
    targets = [t for t in perts if t != "control"]

    with h5py.File(a.src) as f:
        genes = read_index(f, "var"); n = f["X"]["indptr"].shape[0] - 1
    missing = [t for t in targets if t not in set(genes)]
    print(f"panel={len(genes)} genes  cells={n:,}  targets={len(targets)}  "
          f"present={len(targets)-len(missing)}  MISSING(to fill)={len(missing)}", flush=True)
    if not missing:
        print("nothing to pad"); return

    # lambda_bg: global mean expression over the real panel (one streaming pass)
    tot = 0.0; cnt = 0
    with h5py.File(a.src) as f:
        X = f["X"]
        for s in range(0, n, a.chunk):
            e = min(s + a.chunk, n); seg = X["indptr"][s:e+1]
            tot += float(X["data"][int(seg[0]):int(seg[-1])].sum()); cnt += (e - s) * len(genes)
    lam_bg = tot / cnt
    if a.zero_fill:
        print("ZERO-FILL for the un-recoverable targets: all-zero columns, no label "
              "information, no train/test mismatch.", flush=True)
    else:
        print(f"lambda_bg (global mean expr) = {lam_bg:.4f} | lambda_kd = {a.lambda_kd}", flush=True)

    man = {"panel": int(len(genes)), "targets": len(targets), "missing": len(missing),
           "lambda_bg": lam_bg, "lambda_kd": a.lambda_kd, "zero_fill": bool(a.zero_fill),
           "real_from": a.real_from, "var_order": "panel ++ missing (missing in label order)"}

    for tag, path, is_ctrl in (("perturbed", a.src, False), ("ctrl", a.ctrl, True)):
        with h5py.File(path) as f:
            g2 = read_index(f, "var"); m = f["X"]["indptr"].shape[0] - 1
            obs_idx = read_index(f, "obs"); xdt = f["X"]["data"].dtype
            lab = (np.full(m, "control") if is_ctrl else read_col(f, "perturbation", m))
            obs_cols = {c: read_col(f, c, m) for c in OBS_KEEP if c in f["obs"]}

        real_names, R = ([], None)
        if a.real_from:
            real_names, R = splice_real(a.real_from_ctrl if is_ctrl else a.real_from,
                                        obs_idx, missing, a.chunk, tag)
        idx_of = {g: j for j, g in enumerate(missing)}
        real_pos = np.array([idx_of[g] for g in real_names], dtype=int)
        syn_pos = np.setdiff1d(np.arange(len(missing)), real_pos)
        syn_names = [missing[j] for j in syn_pos]
        if not is_ctrl and R is not None and len(real_names):
            # Per COLUMN: does gene g actually drop in the cells perturbing g? A block-wide
            # mean would average g's own column together with the other 436 and show nothing.
            code = pd.Index(real_names).get_indexer(lab)          # -1 where not a real target
            oth, own, rat = [], [], []
            for j in range(len(real_names)):
                o = code == j
                if not o.any():
                    continue
                mo = float(R[o, j].mean()); mb = float(R[~o, j].mean())
                own.append(mo); oth.append(mb); rat.append(mo / mb if mb > 0 else np.nan)
            rat = np.asarray(rat, float)
            print(f"[real-from] {tag}: measured knockdown over {len(rat)} spliced targets -- "
                  f"own/other ratio median {np.nanmedian(rat):.3f} mean {np.nanmean(rat):.3f}, "
                  f"{100*np.nanmean(rat < 0.8):.0f}% dip below 0.8 | background mean "
                  f"{np.mean(oth):.3f} vs synthetic fill {lam_bg:.3f}, "
                  f"own mean {np.mean(own):.3f} vs {a.lambda_kd:.3f}", flush=True)
            man["real_ratio_median"] = float(np.nanmedian(rat))
            man["real_ratio_frac_below_0.8"] = float(np.nanmean(rat < 0.8))
            man["real_bg_mean"] = float(np.mean(oth)); man["real_own_mean"] = float(np.mean(own))
            man["real_targets_with_own_cells"] = int(len(rat))
        man[f"{tag}_real"] = len(real_names); man[f"{tag}_synthetic"] = len(syn_names)

        out = os.path.join(a.out_dir, os.path.basename(path).replace(".h5ad", ".padded.h5ad"))
        W = H5adWriter(out, len(g2) + len(missing), xdt)
        t0 = time.time()
        with h5py.File(path) as f:
            X = f["X"]
            for s in range(0, m, a.chunk):
                e = min(s + a.chunk, m); seg = X["indptr"][s:e+1]
                lo, hi = int(seg[0]), int(seg[-1])
                base = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], seg - lo),
                                     shape=(e - s, len(g2)))
                pad = np.zeros((e - s, len(missing)), dtype=xdt)
                if R is not None and len(real_pos):
                    pad[:, real_pos] = R[s:e]                     # real measurements, dips and all
                if len(syn_pos) and not a.zero_fill:
                    pad[:, syn_pos] = rng.poisson(lam_bg, size=(e - s, len(syn_pos))).astype(xdt)
                    if not is_ctrl:                  # knock the gene down in its own cells
                        L = lab[s:e]
                        for j, t in zip(syn_pos, syn_names):
                            o = (L == t)
                            if o.any():
                                pad[o, j] = rng.poisson(a.lambda_kd, size=int(o.sum()))
                W.append(sp.hstack([base, sp.csr_matrix(pad)], format="csr"))
                del base, pad
                if (s // a.chunk) % 20 == 0:
                    print(f"    wrote {e:,}/{m:,} rows  {time.time()-t0:.0f}s", flush=True)
        del R
        nr, nc, nnz = W.finish(obs_idx, obs_cols, np.concatenate([g2, missing]))
        print(f"  wrote {tag}: {out}  shape=({nr}, {nc})  nnz={nnz:,}  "
              f"real={len(real_names)} synthetic={len(syn_names)}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(a.out_dir, "padding_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("manifest:", json.dumps({k: v for k, v in man.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
