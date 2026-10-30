#!/usr/bin/env python
"""Score a trained ExPert model over the benchmark's query cells and export
BENCHMARK-FORMAT logits.h5 -- in-panel and full-panel, from ONE prediction pass.

Pipeline port of ExPert/analysis/readout/pa_benchmark_score.py (the fold_4
run: slurm/pa_bench_score.sbatch with FULLDB=1). Scoring, control refresh, head
selection and the leak check are carried over; what changed:

  * both variants are written from a single predict() pass instead of two runs.
    The model is scored against its FULL trained vocabulary once; the in-panel
    file is that matrix restricted to the query's own classes. For head A
    (cosine to class anchors) the restricted columns are bit-identical to a
    separate --full-panel=0 run. For head H the softmax would have been
    normalised over the restricted set instead, which changes the VALUES but not
    their order, so every rank-based metric (recall@k, accuracy, macro-F1 via
    argmax) is unchanged.
  * the two variants go to SEPARATE roots, passed in explicitly. This is not
    cosmetic: collect_results.py takes the GLOBAL UNION of class names across
    methods and pads everyone else with -inf, so a 9k-class file sitting in the
    shared results root silently re-scores every other method over 9k classes
    and deflates their macro-F1.
  * --leak-split is mandatory unless --allow-leak: these models are trained on
    the FULL splits, whose id_train contains most of the small split's id_test
    cells (measured on fold_4: 25,356 of 30,180, 84%). Those cells are dropped.
    ood_test is unaffected when the folds' held-out sets match (fold_4: 0 dropped).

HEAD. `--head A` = cosine(z_shared, class anchors from model.pt); `--head H` =
the classifier head. The fold models under config expert_models_root
have cls_head_weight = 0.0, i.e. H is untrained and at chance -- A is the live
path for them.
"""
import argparse, gzip, os, sys
import numpy as np


def _read_idx(p):
    with gzip.open(p, "rt") as fh:
        return np.fromiter((int(l) for l in fh if l.strip()), dtype=np.int64)


def _write_logits(path, S, cols, idx):
    """Benchmark schema: /logits (n, C) float32, /idx (n,) source rows, /columns (C,)."""
    import h5py
    if S.shape[0] != len(idx):
        raise ValueError(f"{path}: {S.shape[0]} rows vs {len(idx)} indices")
    if S.shape[1] != len(cols):
        raise ValueError(f"{path}: {S.shape[1]} columns vs {len(cols)} names")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = np.asarray([str(c) for c in cols], dtype=object)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("logits", data=np.asarray(S, np.float32),
                          compression="gzip", compression_opts=4)
        fh.create_dataset("idx", data=np.asarray(idx, np.int64))
        fh.create_dataset("columns", data=names, dtype=h5py.string_dtype())
    print(f"  wrote {path}  {S.shape}", flush=True)


def _stream_scores(Zn, An, cols, keep, idx, path_full, path_in, truth, chunk=16384, pen=None):
    """Score cells against the atoms in row chunks and write both panels as they go.

    Peak memory is one (chunk x C) block instead of the whole (cells x C) matrix -- plus
    the old quick-check's full argsort (a cells x 8,964 int64 array, ~9GB at 125k
    cells), which is what drove the eval to ~40GB. The quick check now uses the rank
    of the true class, which needs no sort. Output schema/compression is unchanged,
    so the logits are bit-identical to the one-shot version.
    """
    import h5py
    n = Zn.shape[0]
    for path, cc in ((path_full, cols), (path_in, cols[keep])):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fh_full = h5py.File(path_full, "w"); fh_in = h5py.File(path_in, "w")
    try:
        dsets = []
        for fh, cc in ((fh_full, cols), (fh_in, cols[keep])):
            dsets.append(fh.create_dataset("logits", shape=(n, len(cc)), dtype=np.float32,
                                           compression="gzip", compression_opts=4))
            fh.create_dataset("idx", data=np.asarray(idx, np.int64))
            fh.create_dataset("columns", data=np.asarray([str(c) for c in cc], dtype=object),
                              dtype=h5py.string_dtype())
        pos_full = {c: j for j, c in enumerate(cols)}
        pos_in = {c: j for j, c in enumerate(cols[keep])}
        t_full = np.array([pos_full.get(g, -1) for g in truth])
        t_in = np.array([pos_in.get(g, -1) for g in truth])
        rk_full = np.full(n, np.iinfo(np.int64).max); rk_in = rk_full.copy()
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            S = (Zn[s:e] @ An.T).astype(np.float32)
            if pen is not None:
                S -= pen[None, :]
            Si = S[:, keep]
            dsets[0][s:e] = S
            dsets[1][s:e] = Si
            for SS, tt, rk in ((S, t_full[s:e], rk_full), (Si, t_in[s:e], rk_in)):
                ok = tt >= 0
                if ok.any():
                    tv = SS[np.flatnonzero(ok), tt[ok]][:, None]
                    rk[s:e][ok] = (SS[ok] > tv).sum(axis=1)
    finally:
        fh_full.close(); fh_in.close()
    print(f"  wrote {path_full}  ({n}, {len(cols)})\n  wrote {path_in}  ({n}, {int(keep.sum())})",
          flush=True)
    return rk_in, rk_full


def _mem(tag):
    """Current / peak RSS, so the log shows which phase owns the job's memory."""
    try:
        with open("/proc/self/status") as fh:
            st = dict(l.split(":", 1) for l in fh if l.startswith(("VmRSS", "VmHWM")))
        g = lambda k: int(st[k].split()[0]) / 2**20
        print(f"  [mem] {tag}: rss {g('VmRSS'):.1f} GB, peak {g('VmHWM'):.1f} GB", flush=True)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="the run_N dir (holds config/ExPert_v0/model/model.pt)")
    ap.add_argument("--query-adata", required=True, help="adata the BENCHMARK split indices refer to")
    ap.add_argument("--model-adata", required=True, help="adata the model was trained with (full DB)")
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--fold-dir", required=True, help="benchmark fold dir (small/fold_N)")
    ap.add_argument("--leak-split", default=None, help="the MODEL's own id_train, in model-adata rows")
    ap.add_argument("--allow-leak", action="store_true")
    ap.add_argument("--predict-chunk", type=int, default=16384,
                    help="target cells per predict() call (head A, --no-model-adata); whole "
                         "datasets are never split, so one call can exceed this")
    ap.add_argument("--splits", nargs="+", default=["id_test", "ood_test"])
    ap.add_argument("--out-inpanel", required=True, help="<root>/ExPert/perturb-seq/<size>/<fold>")
    ap.add_argument("--out-fullpanel", required=True, help="SEPARATE root, never the collected one")
    ap.add_argument("--expert-repo", required=True, help="ExPert checkout (for `import expert`)")
    ap.add_argument("--head", choices=("A", "H"), default="A")
    # ---- padding mode (docs/padding_scope.md). Both default off.
    ap.add_argument("--padded-anchors", dest="padded_anchors", default=None,
                    help="npz {embeddings,genes} of the PADDED class-anchor table, as "
                         "emitted by workflow/scripts/export_expert_atoms.py. Head A is "
                         "cosine(z_shared, anchors); the default load_atoms() path returns "
                         "only the TRAINED classes, so scoring a perturbation with no "
                         "training cells needs the padded rows. Reusing the exported npz "
                         "keeps checkpoint-layout knowledge in exactly one place.")
    ap.add_argument("--no-model-adata", dest="no_model_adata", action="store_true",
                    help="load the model without registering a training adata and pass the "
                         "query to predict(adata=...) instead. Skips the ~157GB unbacked read "
                         "and skips setup_anndata's observed-label validation, which is what "
                         "blocked the padding run. --model-adata is then unused.")
    ap.add_argument("--class-list", dest="class_list", default=None,
                    help="newline file pinning the in-panel class space. Without it the "
                         "in-panel columns are the query's own observed classes.")
    ap.add_argument("--label-key", default="perturbation")
    ap.add_argument("--dataset-key", default="dataset")
    ap.add_argument("--ctrl-label", default="control")
    ap.add_argument("--min-ctrl", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--hub-correction", dest="hub_correction", action="store_true",
                    help="subtract the model's stored per-atom popularity offset (atom_hub_pen row 0, cosine "
                         "units), as predict(hub_correction=True) does. OFF by default: on the ctx7 benchmark it "
                         "LOSES in all 12 folds per cell (OOD recall@10 -12%%, utils/abl_hub)")
    ap.add_argument("--no-hub-correction", dest="hub_correction", action="store_false",
                    help="plain cosine (the default; kept so existing call sites stay valid)")
    ap.add_argument("--dump-z", default=None,
                    help="dir: also save each split's z_shared (float32) + query row idx as <dir>/<split>.npz")
    ap.add_argument("--mask-target", action="store_true",
                    help="ABLATION: zero each cell's own target gene in the encoder's control-centred "
                         "input (delta = 0 for that gene, i.e. 0 in both perturbed and control). Uses "
                         "the TRUE label; write to a separate --out-inpanel root")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if not a.leak_split and not a.allow_leak:
        ap.error("--leak-split is required (or pass --allow-leak): these models train on the FULL "
                 "splits and would otherwise be scored on cells they were trained on")
    sys.path.insert(0, a.expert_repo)
    os.chdir(a.expert_repo)

    import anndata as ad
    import pandas as pd
    from expert.models._expert import ExPert
    from expert.utils.constants import PREDICTION_KEYS

    q = ad.read_h5ad(a.query_adata, backed="r")
    print(f"[query] {a.query_adata} {q.shape}", flush=True)
    _mem("query opened")
    idx = {s: _read_idx(os.path.join(a.fold_dir, f"{s}.txt.gz")) for s in a.splits
           if os.path.exists(os.path.join(a.fold_dir, f"{s}.txt.gz"))}
    for s, v in idx.items():
        print(f"  {s}: {len(v):,}", flush=True)
    q_names = np.asarray(q.obs_names).astype(str)
    q_lab = q.obs[a.label_key].astype(str).values
    q_classes = np.unique(q_lab)
    if a.class_list:
        with open(a.class_list) as fh:
            pinned = [l.strip() for l in fh if l.strip()]
        n_obs_cls = len(q_classes)
        q_classes = np.unique(np.asarray(pinned, dtype=object).astype(str))
        print(f"[query] class space PINNED to {len(q_classes)} classes from {a.class_list} "
              f"({n_obs_cls} are actually observed in these cells)", flush=True)
    else:
        print(f"[query] {len(q_classes)} classes in the benchmark panel", flush=True)

    # --no-model-adata: load the model WITHOUT registering a training adata and pass
    # the query straight to predict(adata=...).
    #
    # predict() already takes an adata, so the model adata is only ever needed to give
    # `indices` something to address. Registering one costs an unbacked ~157GB read of
    # the training store AND runs setup_anndata, which validates the class-embedding
    # table against every OBSERVED label -- the reason the padding run kept dying with
    # `[Registry] class_emb file ... is missing 369/10245 observed labels` even after
    # the scored columns and the split rows had both been filtered. Not registering it
    # sidesteps both.
    madata = None
    mpos = None
    if not a.no_model_adata:
        madata = ad.read_h5ad(a.model_adata)      # unbacked: ~157GB for the full DB
        print(f"[model adata] {a.model_adata} {madata.shape}", flush=True)
        mpos = pd.Index(np.asarray(madata.obs_names).astype(str))

    model = (ExPert.load(a.model_dir, use_latest_checkpoint=False) if a.no_model_adata
             else ExPert.load(a.model_dir, adata=madata, use_latest_checkpoint=False))
    _mem("model loaded")
    ctrl = ad.read_h5ad(a.ctrl)
    ctrl.obs[a.label_key] = a.ctrl_label          # the DB keeps each study's control nomenclature
    rep = model.refresh_ctrl_gene_means(ctrl, control_label=a.ctrl_label, min_ctrl=a.min_ctrl)
    print(f"[ctrl] refreshed {sum(1 for v in rep.values() if isinstance(v, dict) and v.get('updated'))} datasets",
          flush=True)
    del ctrl
    _mem("ctrl means refreshed")
    enc = model.module.z_encoder
    if a.mask_target:
        p2g = getattr(enc, "pert_to_gene_idx", None)
        if p2g is None or int((p2g >= 0).sum()) == 0:
            raise SystemExit("--mask-target: the encoder has no pert->gene mapping (pert_to_gene_idx "
                             "missing or all -1), so nothing would be masked")
        print(f"[mask-target] {int((p2g >= 0).sum()):,}/{p2g.numel():,} labels map to a gene column; "
              f"center_input={getattr(enc, 'center_input', None)} "
              f"center_mode={getattr(enc, 'center_mode', None)}", flush=True)
        enc.eval_mask_target = True

    if a.head == "A":
        from expert.analysis.atom_umap import load_atoms
        mpt = os.path.join(a.model_dir, "config", "ExPert_v0", "model", "model.pt")
        alab, atoms, _ = load_atoms(mpt)
        alab = np.asarray(alab).astype(str)
        atoms = np.asarray(atoms, np.float32)[:len(alab)]
        if a.padded_anchors:
            d = np.load(a.padded_anchors, allow_pickle=True)
            plab = np.asarray(d["genes"], dtype=object).astype(str)
            pat = np.asarray(d["embeddings"], np.float32)
            if pat.shape[1] != atoms.shape[1]:
                raise SystemExit(f"padded anchors are {pat.shape[1]}-dim but the trained "
                                 f"atoms are {atoms.shape[1]}-dim — wrong tensor exported?")
            # The padded table must AGREE with load_atoms on the trained classes;
            # if it does not, the exported vocabulary order is wrong and every
            # padded score would be attributed to the wrong perturbation.
            pos = {c: j for j, c in enumerate(plab)}
            chk = [(i, pos[c]) for i, c in enumerate(alab) if c in pos][:200]
            if chk:
                d0 = max(float(np.abs(atoms[i] - pat[j]).max()) for i, j in chk)
                print(f"[atoms] padded-vs-trained agreement on {len(chk)} classes: "
                      f"max|diff| = {d0:.3g}", flush=True)
                if d0 > 1e-3:
                    raise SystemExit(
                        f"padded anchor rows disagree with load_atoms() on the trained "
                        f"classes (max|diff| = {d0:.3g}). The exported vocabulary order is "
                        "wrong — re-check export_expert_atoms.py's build_vocab().")
            n_new = len(set(plab) - set(alab))
            alab, atoms = plab, pat
            print(f"[atoms] PADDED table: {len(alab)} classes ({n_new} with no training "
                  f"cells); {int(np.isin(alab, q_classes).sum())} of them in the class space",
                  flush=True)
        else:
            print(f"[atoms] {len(alab)} trained classes; {int(np.isin(alab, q_classes).sum())} of them in the query panel",
                  flush=True)
        An = atoms / np.clip(np.linalg.norm(atoms, axis=1, keepdims=True), 1e-12, None)
        # HUBNESS CORRECTION: classify() scores cos - pen_a with pen = atom_hub_pen[0] over the module's
        # label rows; load_atoms returns those rows in the same order. A per-atom constant -- identical
        # for every cell, so it only re-ranks atoms within a cell.
        hub_pen = None
        mod = model.module
        has_hub = hasattr(mod, "has_hub_correction") and mod.has_hub_correction()
        if a.hub_correction:
            if not has_hub:
                raise SystemExit("--hub-correction but the model stores no atom_hub_pen; "
                                 "run add_hub_correction() + save")
            pen = mod.atom_hub_pen[0].detach().float().cpu().numpy()
            if a.padded_anchors:
                raise SystemExit("--padded-anchors + hub correction: row alignment not implemented")
            if pen.shape[0] < len(alab):
                raise SystemExit(f"atom_hub_pen has {pen.shape[0]} rows < {len(alab)} atoms")
            hub_pen = pen[:len(alab)].astype(np.float32)
            print(f"[hub] correction ON: offset median {np.median(hub_pen):.4f} "
                  f"range [{hub_pen.min():.4f}, {hub_pen.max():.4f}] over {len(hub_pen)} atoms", flush=True)
        else:
            print(f"[hub] correction OFF (model {'has' if has_hub else 'has no'} a stored offset)", flush=True)

    leak_idx = _read_idx(a.leak_split) if a.leak_split else np.empty(0, np.int64)
    leak = len(leak_idx) > 0
    # --no-model-adata still has to honour the leak split, which is indexed by MODEL-adata
    # row. Only the model adata's cell NAMES are needed for that -- read the obs index
    # straight from the h5ad (~2GB) instead of the ~157GB unbacked matrix. At `small` this
    # removes 22-27k id_test cells per fold that the fold model trained on.
    mpos_light = None
    if a.no_model_adata and leak and a.model_adata:
        import h5py
        with h5py.File(a.model_adata, "r") as fh:
            o = fh["obs"]
            k = o.attrs.get("_index", "_index")
            names = o[k][:]                       # raw bytes: no 19.5M-element str list
        mpos_light = pd.Index(names)
        del names
        print(f"[leak] model-adata obs names read directly ({len(mpos_light):,}) for the "
              f"leak split ({len(leak_idx):,} training rows)", flush=True)
    n_model = len(mpos_light) if mpos_light is not None else (len(mpos) if mpos is not None else 0)
    leak_mask = np.zeros(max(n_model, int(leak_idx.max()) + 1 if leak else 0), bool)
    leak_mask[leak_idx] = True
    del leak_idx

    def _in_leak(mi):
        """mi: model-adata rows (-1 = not in the model adata, never a leak)."""
        return (mi >= 0) & leak_mask[np.clip(mi, 0, None)]
    _mem("leak split ready")
    rows = []
    for s, ii in idx.items():
        if a.no_model_adata:
            # No registered adata: hand predict() the query rows directly. No obs_name
            # lookup, so no "0/N query cells found in the model adata" failure mode --
            # the query IS the adata being scored.
            ii = np.sort(ii)
            if mpos_light is not None:
                mi = mpos_light.get_indexer(np.char.encode(q_names[ii].astype(str), "utf-8").astype(object))
                lk = _in_leak(mi)
                print(f"[{s}] {int(lk.sum()):,} of them are in the MODEL's id_train -> excluded",
                      flush=True)
                ii = ii[~lk]
                if len(ii) == 0:
                    print(f"[{s}] nothing left to score -- skipping", flush=True)
                    continue
            print(f"[{s}] scoring {len(ii):,} query cells directly (no model adata)", flush=True)
            if a.head == "A":
                # predict() batches internally but returns every output for every cell
                # (z, z_shared, qzm, qzv: 4 x cells x 2048) and builds per-cell work
                # against all atoms, so one call over 125k cells peaked at ~38GB. Call it
                # on groups of cells and keep only the latent.
                #
                # Groups are WHOLE DATASETS: for a dataset absent from training (every OOD
                # dataset) predict() centres z_shared on that dataset's mean over the cells
                # in the call (_auto_align_center). Splitting a dataset across calls changes
                # that mean -- row chunks moved OOD argmax on 1.5% of cells. Packing whole
                # datasets keeps the output bit-identical to one call.
                ds = q.obs[a.dataset_key].astype(str).values[ii]
                groups, cur, n_cur = [], [], 0
                for d in pd.unique(ds):
                    pos = np.flatnonzero(ds == d)
                    if cur and n_cur + len(pos) > a.predict_chunk:
                        groups.append(np.concatenate(cur)); cur, n_cur = [], 0
                    cur.append(pos); n_cur += len(pos)
                if cur:
                    groups.append(np.concatenate(cur))
                Zc = None
                for pos in groups:
                    pos = np.sort(pos)
                    rr = ii[pos]
                    sub = q[rr].to_memory() if hasattr(q, "isbacked") and q.isbacked else q[rr]
                    pc = model.predict(adata=sub, return_latents=True, obs_only=True,
                                       soft_predictions=False, progress_bar=False,
                                       batch_size=a.batch_size, refresh_ctrl_means=False)
                    zk = next((k for k in ("z_shared", "z", "latent_shared") if k in pc), None)
                    zc = np.asarray(pc[zk], np.float32)
                    if Zc is None:
                        Zc = np.empty((len(ii), zc.shape[1]), np.float32)
                    Zc[pos] = zc
                    del pc, sub, zc
                print(f"[{s}] predicted in {len(groups)} dataset-aligned calls "
                      f"(largest {max(len(g) for g in groups):,} cells)", flush=True)
                po = {zk: Zc}
                del Zc
            else:
                sub = q[ii].to_memory() if hasattr(q, "isbacked") and q.isbacked else q[ii]
                po = model.predict(adata=sub, return_latents=(a.head == "A"), obs_only=True,
                                   soft_predictions=(a.head == "H"), progress_bar=False,
                                   batch_size=a.batch_size, refresh_ctrl_means=False)
        else:
            mi = mpos.get_indexer(q_names[ii])
            ok = mi >= 0
            print(f"[{s}] {int(ok.sum()):,}/{len(ii):,} query cells found in the model adata", flush=True)
            if leak:
                lk = _in_leak(mi)
                print(f"[{s}] {int((lk & ok).sum()):,} of them are in the MODEL's id_train -> excluded", flush=True)
                ok &= ~lk
            ii, mi = ii[ok], mi[ok]
            if len(ii) == 0:
                print(f"[{s}] nothing left to score -- skipping", flush=True)
                continue

            po = model.predict(indices=mi, return_latents=(a.head == "A"), obs_only=True,
                               soft_predictions=(a.head == "H"), progress_bar=False,
                               batch_size=a.batch_size, refresh_ctrl_means=False)
        if a.mask_target:
            m, n = getattr(enc, "_eval_mt_counts", (0, 0))
            print(f"[mask-target] {s}: target masked in {m:,} of {n:,} encoded cells", flush=True)
            if m == 0:
                raise SystemExit("--mask-target masked NO cell (labels not reaching the encoder?)")
            enc._eval_mt_counts = (0, 0)
        _mem(f"{s} predicted ({', '.join(f'{k}{tuple(np.shape(v))}' for k, v in po.items())})"
             if isinstance(po, dict) else f"{s} predicted")
        if a.head == "A":
            zk = next((k for k in ("z_shared", "z", "latent_shared") if k in po), None)
            Z = np.asarray(po[zk], np.float32)
            del po
            if a.dump_z:
                os.makedirs(a.dump_z, exist_ok=True)
                np.savez(os.path.join(a.dump_z, f"{s}.npz"), Z=Z, idx=np.asarray(ii, np.int64))
                print(f"  [dump-z] {s}: {Z.shape} -> {a.dump_z}", flush=True)
            Zn = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)
            del Z
            cols_full = np.asarray(alab)
            keep = np.isin(cols_full, q_classes)
            rk_in, rk_full = _stream_scores(
                Zn, An, cols_full, keep, ii,
                os.path.join(a.out_fullpanel, s, "logits.h5"),
                os.path.join(a.out_inpanel, s, "logits.h5"), q_lab[ii], pen=hub_pen)
            _mem(f"{s} logits written")
            for tag, rk, C in (("in-panel", rk_in, int(keep.sum())), ("full-panel", rk_full, len(cols_full))):
                r = {f"recall@{k}": float((rk < k).mean()) for k in (1, 5, 10)}
                print(f"  [{s}/{tag}] n={len(ii):,} C={C}  " +
                      "  ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
                rows.append(dict(split=s, panel=tag, n=len(ii), n_classes=C, **r))
            continue
        else:
            soft = po[PREDICTION_KEYS.SOFT_PREDICTION_KEY]
            cols_full = np.asarray(soft.columns).astype(str)
            S_full = np.asarray(soft.values, np.float32)

        _write_logits(os.path.join(a.out_fullpanel, s, "logits.h5"), S_full, cols_full, ii)
        keep = np.isin(cols_full, q_classes)
        _write_logits(os.path.join(a.out_inpanel, s, "logits.h5"),
                      S_full[:, keep], cols_full[keep], ii)

        # quick sanity only -- collect_results.py is the authoritative scorer
        for tag, S, cols in (("in-panel", S_full[:, keep], cols_full[keep]),
                             ("full-panel", S_full, cols_full)):
            pos = {c: j for j, c in enumerate(cols)}
            t = np.array([pos.get(g, -1) for g in q_lab[ii]])
            order = np.argsort(-S, axis=1)
            r = {f"recall@{k}": float((order[:, :k] == t[:, None]).any(1).mean()) for k in (1, 5, 10)}
            print(f"  [{s}/{tag}] n={len(ii):,} C={len(cols)}  " +
                  "  ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
            rows.append(dict(split=s, panel=tag, n=len(ii), n_classes=len(cols), **r))
    if rows:
        out = os.path.join(a.out_inpanel, "expert_quickcheck.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
