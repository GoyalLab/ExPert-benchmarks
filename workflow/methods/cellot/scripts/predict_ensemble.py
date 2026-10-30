#!/usr/bin/env python3
"""cellot ensemble scorer — PER-DATASET OT maps + dataset-blind evaluation.

cellot is single-dataset by design (one scgen AE + per-perturbation neural OT
maps in that AE's latent). We train one AE + per-pert OT maps PER dataset D on
D's own controls + D's own gene panel, then score EVERY test cell against EVERY
dataset's models (the cell does NOT reveal which of the 21 models to use — the
real-life setting), softmax-avg over the models that cover each perturbation, and
remove the per-pert baseline (z-score each column across cells). This mirrors the
genepert/fingerprint per-dataset ensemble exactly; only the per-model score
differs (latent nearest-delta instead of Pearson-of-fingerprints).

Own-control DELTA scoring (why this and not a raw latent distance):
  For model dataset D and pert p, the OT map g_{D,p} transports D's control
  latent cloud to D's pert-p latent cloud. Its predicted DELTA is
      Δ_{D,p} = mean(g_{D,p}(z_ctrl_D)) - mean(z_ctrl_D)          (in D's AE latent)
  where z_ctrl_D = AE_D.encode(D's own controls). A test cell c from dataset D_c
  is encoded into D's latent, z_c = AE_D.encode(c over D's panel), and its OWN
  control-relative delta is
      δ_c = z_c - mean(AE_D.encode(D_c's own controls))          (own-control center)
  logit_{D,p}(c) = -||δ_c - Δ_{D,p}||.
  Centering the cell by ITS OWN dataset's control (not D's) is batch
  normalization — a cell's perturbation response is a delta w.r.t. its own
  baseline; centering by another dataset's control injects a cross-dataset shift
  and is "too tough" on every off-dataset model. Note g_{D,p} is applied ONLY to
  D's own controls (its training distribution) — never to foreign controls — so
  each Δ_{D,p} is computed ONCE (no per-source-dataset pushforward blowup), and
  the foreign-dataset adaptation is a principled additive re-centering.

Per-pert z-score: some perts' predicted deltas sit near almost every cell, so
their averaged prob is high for everyone and the argmax collapses onto them.
The combiner sees every model's score for every cell, so each pert column's
mean+std across cells IS that per-pert baseline — subtract+scale to leave the
per-cell signal (no labels / dataset identity used; a per-cell log-softmax by
collect_results preserves the ranking).

Output (same path/schema as the old global cellot — one version, per-dataset):
  <fold_root>/<split>/logits.h5  with /logits [cells x perts] float32,
  /obs_names [cells] (cell barcodes), /perts [perts] (column order).
"""
import argparse
import gzip
import json
import logging
import os
from pathlib import Path

import anndata
import h5py
import numpy as np
import torch
from scipy import sparse

from cellot.data.cell import _ensure_normalized
from cellot.models import load_autoencoder_model, load_cellot_model
from cellot.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CTRL_LABEL = "control"


def read_indices(splits_dir, name):
    with gzip.open(os.path.join(splits_dir, f"{name}.txt.gz"), mode="rt") as fh:
        return np.fromiter((int(l.rstrip()) for l in fh if l.strip()), dtype=np.int64)


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def ae_encode(ae_model, X, batch_size=8192):
    """Encode a float32 gene-space matrix into the AE latent, in batches."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            chunk = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32)
            out.append(ae_model.encode(chunk).numpy())
    return (np.concatenate(out).astype(np.float32)
            if out else np.zeros((0, 0), dtype=np.float32))


def pushforward(g, Z, batch_size=512):
    """Push a latent matrix through cellot's g (ICNN transport map)."""
    g.eval()
    outs = []
    for i in range(0, len(Z), batch_size):
        chunk = torch.as_tensor(Z[i:i + batch_size], dtype=torch.float32)
        chunk.requires_grad_(True)
        outs.append(g.transport(chunk).detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def neg_dist(A, B):
    """-||a - b|| for every row a of A (n,d) vs row b of B (m,d) -> (n,m)."""
    A2 = (A * A).sum(1, keepdims=True)
    B2 = (B * B).sum(1)[None, :]
    d2 = A2 - 2.0 * A @ B.T + B2
    return -np.sqrt(np.clip(d2, 0.0, None)).astype(np.float32)


def softmax_rows(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)


def resolve_panel_cols(mask, d2m, dataset, mode="all"):
    """Gene columns a per-dataset model trains/scores on.

    all     -> every gene on the source var axis (DEFAULT). The per-dataset panels
               are hard zero-padded, so this keeps every dataset's model in ONE
               shared coordinate system (directly comparable across datasets).
    dataset -> that dataset's own active panel (varm feature_mask[:, D]).
    shared  -> intersection of the active panels across ALL datasets. On a DB with
               500-gene disjoint panels this collapses to ~1 gene; guarded below.
    """
    import numpy as _np
    n_genes = int(mask.shape[0])
    if mode == "all":
        return _np.arange(n_genes, dtype=_np.int64)
    if mode == "shared":
        cols = _np.where((mask == 1).all(axis=1))[0]
        if cols.size < 10:
            raise ValueError(
                f"gene_panel='shared' yields only {cols.size} gene(s) across "
                f"{mask.shape[1]} datasets - the panels are effectively disjoint. "
                f"Use gene_panel: all."
            )
        return cols.astype(_np.int64)
    return _np.where(mask[:, d2m[dataset]] == 1)[0].astype(_np.int64)


def build_dataset_model(fold_root, D, panel_cols, ctrl_all, ctrl_ds_all):
    """Assemble dataset D's ensemble model: AE_D + per-pert latent deltas +
    own-control latent baseline + per-source-dataset latent baselines.

    Returns None if D has no trained OT maps. Keeps AE_D loaded (small MLP) so
    the scorer can encode test cells into D's latent chunk by chunk."""
    pds = Path(fold_root) / "per_dataset" / D
    ae_dir = pds / "ae"
    models_dir = pds / "models"
    md_paths = sorted(models_dir.glob("*/cache/model.pt"))
    if not md_paths:
        return None

    dim_in = len(panel_cols)
    ae_cfg = load_config(ae_dir / "config.yaml")
    ae_model, _ = load_autoencoder_model(
        ae_cfg, restore=ae_dir / "cache" / "model.pt", input_dim=dim_in)
    ae_model.eval()

    # D's own control latent cloud (the OT maps' source distribution).
    ctrl_D = ctrl_all[ctrl_ds_all == D][:, panel_cols]
    if not ctrl_D.n_obs:
        logging.warning("  [%s] no controls for dataset; skipping", D)
        return None
    z_ctrl_D = ae_encode(ae_model, to_dense(ctrl_D.X))
    b_self = z_ctrl_D.mean(axis=0)

    perts, deltas = [], []
    for mp in md_paths:
        p = mp.parent.parent.name
        cfg = load_config(mp.parent.parent / "config.yaml")
        (_, g_net), _ = load_cellot_model(
            cfg, restore=mp, input_dim=z_ctrl_D.shape[1])
        pushed = pushforward(g_net, z_ctrl_D)
        deltas.append(pushed.mean(axis=0) - b_self)   # Δ_{D,p} (own-control delta)
        perts.append(p)
    order = np.argsort(perts)
    perts = [perts[i] for i in order]
    deltas = np.asarray(deltas, dtype=np.float32)[order]

    # Per-source-dataset baseline: each source dataset's controls encoded into
    # D's latent (mean) — the own-control center a test cell of that source uses.
    baseline_by_source = {}
    for ds in np.unique(ctrl_ds_all):
        c = ctrl_all[ctrl_ds_all == ds][:, panel_cols]
        baseline_by_source[str(ds)] = (ae_encode(ae_model, to_dense(c.X)).mean(axis=0)
                                       if c.n_obs else b_self.copy())

    logging.info("  [%s] %d perts, %d-gene panel, latent dim %d, %d ctrl cells",
                 D, len(perts), dim_in, z_ctrl_D.shape[1], ctrl_D.n_obs)
    return dict(ae_model=ae_model, panel_cols=panel_cols, perts=perts,
                deltas=deltas, b_self=b_self, baseline_by_source=baseline_by_source)


def ensemble_score_to_h5(source, split_idx, obs_names, models, global_perts,
                         out_path, chunk_size):
    """Score every test cell against every dataset model (own-control latent
    delta), softmax-avg over covering models, per-pert z-score, stream to h5."""
    split_idx = np.asarray(split_idx, dtype=np.int64)
    n_cells, n_perts = len(split_idx), len(global_perts)
    col_of = {p: i for i, p in enumerate(global_perts)}
    for m in models.values():
        m["gcols"] = np.array([col_of[p] for p in m["perts"]], dtype=np.int64)

    def score_chunk(ci):
        block = source[ci].to_memory()
        _ensure_normalized(block)
        Xf = block.X.tocsr() if sparse.issparse(block.X) else np.asarray(block.X)
        cds = block.obs["dataset"].astype(str).to_numpy()
        prob_sum = np.zeros((len(ci), n_perts), dtype=np.float64)
        cover = np.zeros(n_perts, dtype=np.float64)
        for m in models.values():
            Xd = to_dense(Xf[:, m["panel_cols"]]).astype(np.float32)
            z = ae_encode(m["ae_model"], Xd)                 # cell -> D's latent
            base = np.empty_like(z)
            for ds in np.unique(cds):
                base[cds == ds] = m["baseline_by_source"].get(ds, m["b_self"])
            delta = z - base                                 # own-control delta
            scores = neg_dist(delta, m["deltas"])            # -||δ - Δ_{D,p}||
            prob_sum[:, m["gcols"]] += softmax_rows(scores)
            cover[m["gcols"]] += 1
        covered = cover > 0
        prob_sum[:, covered] /= cover[covered]
        out = np.full((len(ci), n_perts), -18.0, dtype=np.float32)  # log(~1e-8) floor
        out[:, covered] = np.log(prob_sum[:, covered] + 1e-8)
        return out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    str_dt = h5py.string_dtype()
    with h5py.File(out_path, "w") as fh:
        logits = fh.create_dataset(
            "logits", shape=(n_cells, n_perts), dtype="float32",
            chunks=(min(chunk_size, n_cells) or 1, n_perts), compression="lzf")
        fh.create_dataset("perts", data=np.asarray(global_perts, dtype=object), dtype=str_dt)
        fh.create_dataset("obs_names", data=np.asarray(obs_names, dtype=object),
                          dtype=str_dt, compression="lzf")
        # pass 1: score + write raw combined logits, accumulate per-pert sum/sumsq
        csum = np.zeros(n_perts, dtype=np.float64)
        csq = np.zeros(n_perts, dtype=np.float64)
        tag = os.path.basename(os.path.dirname(out_path))
        for start in range(0, n_cells, chunk_size):
            out = score_chunk(split_idx[start:start + chunk_size])
            logits[start:start + out.shape[0]] = out
            csum += out.sum(axis=0); csq += (out.astype(np.float64) ** 2).sum(axis=0)
            logging.info("  %s: scored %d/%d cells", tag,
                         min(start + out.shape[0], n_cells), n_cells)
        mean = csum / max(n_cells, 1)
        std = np.sqrt(np.maximum(csq / max(n_cells, 1) - mean ** 2, 0.0))
        active = std > 1e-6
        logging.info("  per-pert z-score: %d/%d pert columns active", int(active.sum()), n_perts)
        # pass 2: normalize each pert column (x - mean) / std
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            blk = logits[start:end]
            blk[:, active] = (blk[:, active] - mean[active]) / std[active]
            logits[start:end] = blk
    logging.info("Wrote %s (%d cells x %d perts)", out_path, n_cells, n_perts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_root", required=True,
                    help="…/cellot/perturb-seq/<size>/<fold> (has per_dataset/<D>/…)")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", required=True, help="id_test | ood_test")
    ap.add_argument("--gene-panel", dest="gene_panel", default="all",
                    choices=("all", "dataset", "shared"),
                    help="gene space per per-dataset model (see config gene_panel)")
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=50_000)
    args = ap.parse_args()

    meta = json.loads((Path(args.splits_dir).parent / "metadata.json").read_text())
    source = anndata.read(meta["source_path"], backed="r")
    mask = np.asarray(source.varm["feature_mask_per_dataset"])
    d2m = {str(k): int(v) for k, v in dict(source.uns["dataset_to_mask_idx"]).items()}

    ctrl_all = anndata.read(args.ctrl)
    # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's
    # own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control),
    # so comparing the label against a single value misses 18/22 datasets.
    ctrl_all.obs["perturbation"] = "control"
    _ensure_normalized(ctrl_all)
    ctrl_all.X = ctrl_all.X.astype(np.float32)
    ctrl_ds_all = ctrl_all.obs["dataset"].astype(str).to_numpy()

    # Datasets to ensemble = those with a per_dataset/<D>/models tree on disk.
    pd_root = Path(args.fold_root) / "per_dataset"
    disk_datasets = sorted(p.name for p in pd_root.glob("*") if (p / "models").is_dir())
    logging.info("Ensembling %d per-dataset cellot models: %s", len(disk_datasets), disk_datasets)

    models = {}
    for D in disk_datasets:
        if D not in d2m:
            logging.warning("dataset %s not in dataset_to_mask_idx; skipping", D); continue
        panel_cols = resolve_panel_cols(mask, d2m, D, args.gene_panel)
        m = build_dataset_model(args.fold_root, D, panel_cols, ctrl_all, ctrl_ds_all)
        if m is not None:
            models[D] = m
    if not models:
        raise RuntimeError("no per-dataset cellot models found under " + str(pd_root))
    global_perts = sorted({p for m in models.values() for p in m["perts"]})
    logging.info("Global perturbation space: %d perts across %d models",
                 len(global_perts), len(models))

    split_idx = read_indices(args.splits_dir, args.split)
    obs_names = source.obs_names.to_numpy()[split_idx]
    ensemble_score_to_h5(source, split_idx, obs_names, models, global_perts,
                         args.out, args.chunk_size)


if __name__ == "__main__":
    main()
