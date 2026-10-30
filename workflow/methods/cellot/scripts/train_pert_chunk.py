#!/usr/bin/env python3
"""Train one CHUNK of a dataset's per-perturbation cellot OT maps.

Per-dataset cellot is per-(dataset, perturbation); at full that's ~97k OT maps
per fold, so we chunk: this trains perts[k*K:(k+1)*K] of dataset D (K =
--perts_per_chunk), each a normal cellot OT fit restricted to D (via
--config.data.dataset=D) in D's AE latent space. Writes a chunk marker listing
the perts it trained so the concat/eval steps know the chunk is done.

D's training perturbations are derived here (sorted unique non-control perts of
D's id_train cells, via h5py obs codes) so the chunking matches the config-fixed
count (perdataset_pert_counts.json) with no checkpoint.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _obs_codes(source, col):
    with h5py.File(source, "r") as f:
        node = f[f"obs/{col}"]
        if isinstance(node, h5py.Group):
            cats = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                    for c in node["categories"][:]]
            return node["codes"][:], np.array(cats)
        raw = node[:]
        vals = np.array([v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw])
        return None, vals


def dataset_train_perts(source, splits_dir, dataset, dataset_col, cond_col, ctrl):
    ti = np.fromiter((int(x) for x in gzip.open(Path(splits_dir) / "id_train.txt.gz", "rt")
                      if x.strip()), dtype=np.int64)
    dcodes, dcats = _obs_codes(source, dataset_col)
    pcodes, pcats = _obs_codes(source, cond_col)
    ds = (dcats[dcodes[ti]] if dcodes is not None else dcats[ti])
    pr = (pcats[pcodes[ti]] if pcodes is not None else pcats[ti])
    sel = (ds == dataset) & (pr != ctrl)
    return sorted(np.unique(pr[sel]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--chunk_idx", type=int, required=True)
    ap.add_argument("--perts_per_chunk", type=int, required=True)
    ap.add_argument("--models_root", required=True, help="…/per_dataset/<D>/models")
    ap.add_argument("--ae_dir", required=True, help="…/per_dataset/<D>/ae")
    ap.add_argument("--taskcfg", required=True)
    ap.add_argument("--marker", required=True)
    ap.add_argument("--data_path", default=None,
                    help="perturbed source h5ad; overrides the task config's data.path")
    ap.add_argument("--data_path_ctrl", default=None,
                    help="control h5ad; overrides the task config's data.path_ctrl")
    ap.add_argument("--dataset_col", default="dataset")
    ap.add_argument("--gene-panel", dest="gene_panel", default="all",
                    choices=("all", "dataset"),
                    help="gene space for the per-dataset model (see config gene_panel)")
    ap.add_argument("--cond_col", default=None, help="default: metadata cls_col")
    ap.add_argument("--ctrl", default="control")
    args = ap.parse_args()

    meta = json.loads((Path(args.splits_dir).parent / "metadata.json").read_text())
    source = meta["source_path"]
    cond_col = args.cond_col or meta.get("cls_col", "perturbation")

    perts = dataset_train_perts(source, args.splits_dir, args.dataset,
                                args.dataset_col, cond_col, args.ctrl)
    K = args.perts_per_chunk
    chunk = perts[args.chunk_idx * K:(args.chunk_idx + 1) * K]
    print(f"[{args.dataset}] chunk {args.chunk_idx}: {len(chunk)} of {len(perts)} perts", flush=True)

    trained, failed = [], []
    for i, pert in enumerate(chunk):
        outdir = f"{args.models_root}/{pert}"
        if os.path.exists(f"{outdir}/cache/model.pt"):
            print(f"  [{i+1}/{len(chunk)}] {pert}: cached", flush=True); trained.append(pert); continue
        cmd = [
            sys.executable, "scripts/train.py",
            "--outdir", outdir,
            "--config", args.taskcfg,
            "--config", "configs/models/cellot.yaml",
            "--config.datasplit.path=" + args.splits_dir,
            "--config.data.target=" + pert,
            "--config.data.dataset=" + args.dataset,
            "--config.data.gene_panel=" + args.gene_panel,
            "--config.data.lazy_load_target=true",
            "--config.optim.lr=0.001",
            "--config.training.n_iters=5000",
            "--config.training.eval_freq=50",
            "--config.training.cache_freq=500",
            "--config.training.logs_freq=25",
            "--config.training.early_stopping_patience=10",
            "--config.data.ae_emb.path=" + args.ae_dir,
            *(["--config.data.path=" + args.data_path] if args.data_path else []),
            *(["--config.data.path_ctrl=" + args.data_path_ctrl] if args.data_path_ctrl else []),
        ]
        print(f"  [{i+1}/{len(chunk)}] training {pert}...", flush=True)
        # SKIP-on-failure: a single (dataset, pert) that cellot can't fit — e.g.
        # too few cells for even OT batches (compute_loss_f needs matching source/
        # target batch sizes) — must NOT kill the whole chunk (and block the
        # ensemble eval). The dataset-blind ensemble just covers that pert with one
        # fewer model. Log it, drop the partial outdir, continue.
        rc = subprocess.run(cmd).returncode
        if rc == 0 and os.path.exists(f"{outdir}/cache/model.pt"):
            trained.append(pert)
        else:
            print(f"  [{i+1}/{len(chunk)}] WARNING: {pert} failed (rc={rc}); skipping", flush=True)
            failed.append(pert)
            if not os.path.exists(f"{outdir}/cache/model.pt"):
                shutil.rmtree(outdir, ignore_errors=True)   # drop partial dir

    Path(args.marker).parent.mkdir(parents=True, exist_ok=True)
    Path(args.marker).write_text("\n".join(trained) + ("\n" if trained else ""))
    msg = f"chunk done: {len(trained)} perts -> {args.marker}"
    if failed:
        msg += f" ({len(failed)} skipped: {', '.join(failed)})"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
