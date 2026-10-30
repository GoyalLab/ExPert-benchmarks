"""Score test cells under a trained Scouter model and emit a logits.h5 that
matches cellot's schema (datasets: /logits, /perts, /obs_names).

Strategy:
  1. Rebuild the same merged adata + ScouterData fixture used at training time
     (control cells + train cells — enough to satisfy Scouter's init), with
     the SAME gene set + embedding stored at train time.
  2. Reload the trained network state dict.
  3. For each perturbation in the global label space, call
     `model.pred([p], n_pred=N)` and average to a per-pert mean μ_p.
  4. Load the test cells (id_test or ood_test). For each cell x, compute
     logit[x, p] = -||x - μ_p||_2  (cdist).
  5. Write logits.h5 (n_cells × n_perts), /perts (n_perts,), /obs_names (n_cells,).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import cdist
from scipy import sparse

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from utils.splits import (
    fold_index_from_dir,
    filter_by_context,
    held_out_from_metadata,
    list_train_perts,
    load_cells_by_index,
    load_ctrl,
    read_metadata,
    shared_gene_cols,
)
from utils.scouter_adapter import (
    CTRL_LABEL,
    attach_embedding_indices,
    keep_perturbations_with_var_match,
    merge_scouter_adata,
)

from scouter import Scouter, ScouterData


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True, type=Path,
                    help="${RESULTS_ROOT}/model/ from .train.sh")
    ap.add_argument("--splits_dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=["id", "ood"])
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="${RESULTS_ROOT}/<split>_test/ — logits.h5 lands here")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shared_genes", action="store_true",
                    help="Must match train.py: restrict the feature space to genes "
                         "measured in every dataset. The model's gene space is 3788 "
                         "when trained with --shared_genes, so predict must subset "
                         "identically or the fixture/test gene order won't match.")
    return ap.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.workers))

    cfg = _load_yaml(args.config)
    resolved = _load_yaml(args.model_dir / "config.yaml")
    model_cfg = resolved["model"]
    train_cfg = cfg["training"]
    pred_cfg = cfg.get("prediction", {})
    n_pred = int(pred_cfg.get("n_pred", 300))
    pred_seed = int(pred_cfg.get("seed", 24))

    splits_dir = args.splits_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    meta = read_metadata(splits_dir)
    source_path = Path(data_cfg.get("source_h5ad") or meta["source_path"]).resolve()
    ctrl_path = Path(data_cfg["ctrl_h5ad"]).resolve()
    cls_col = data_cfg.get("condition_col", meta.get("cls_col", "perturbation"))
    ctrl_label_in = data_cfg.get("ctrl_label", "ctrl")
    ctx_col, held_out = held_out_from_metadata(splits_dir)

    print(f"[predict] split={args.split}  ctx_col={ctx_col}  held_out={held_out}", flush=True)

    # ---- Rebuild ScouterData fixture (same shape used at train time) -------
    # Scouter's Scouter() needs a ScouterData to bind to. The model.pred() path
    # depends on the embedding + var, not on test cells — but we still need to
    # hand it a valid ScouterData. We reuse train + ctrl cells (cheap-ish).
    gene_cols = shared_gene_cols(source_path) if args.shared_genes else None
    if gene_cols is not None:
        print(f"[predict] shared-gene subset: {len(gene_cols)} genes", flush=True)

    train_idx = splits_dir / "id_train.txt.gz"
    val_idx = splits_dir / "id_val.txt.gz"
    perturbed_train = load_cells_by_index(source_path, train_idx, gene_cols=gene_cols)
    perturbed_val = load_cells_by_index(source_path, val_idx, gene_cols=gene_cols)
    ctrl = load_ctrl(ctrl_path)
    if gene_cols is not None:
        ctrl = ctrl[:, gene_cols].copy()
    if held_out:
        ctrl = filter_by_context(ctrl, ctx_col, held_out, keep_held_out=False)

    merged = merge_scouter_adata(
        perturbed_parts={"train": perturbed_train, "val": perturbed_val},
        ctrl=ctrl,
        cls_col=cls_col,
        ctrl_label_in=ctrl_label_in,
    )
    del perturbed_train, perturbed_val, ctrl        # now inside `merged`; free the copies
    merged, _ = keep_perturbations_with_var_match(merged)
    embd = pd.read_pickle(args.model_dir / "embd.pkl")
    merged, ctrl_idx = attach_embedding_indices(merged, embd, key="embd_index")
    print(f"[predict]   fixture merged.shape={merged.shape}", flush=True)

    scouter_data = ScouterData(
        adata=merged, embd=embd, key_label="condition", key_var_genename="gene_name",
    )
    scouter_data.setup_ad("embd_index")
    scouter_data.gene_ranks()
    scouter_data.get_dropout_non_zero_genes()
    # Slice per-split adatas AFTER setup so they carry .uns['gene_idx_non_zeros'].
    train_mask = (scouter_data.adata.obs["_split"].astype(str).isin(["train", "ctrl"])).to_numpy()
    val_mask = (scouter_data.adata.obs["_split"].astype(str).isin(["val", "ctrl"])).to_numpy()
    scouter_data.train_adata = scouter_data.adata[train_mask].copy()
    scouter_data.val_adata = scouter_data.adata[val_mask].copy()
    scouter_data.test_adata = scouter_data.val_adata
    scouter_data.train_conds = sorted(set(scouter_data.train_adata.obs["condition"].astype(str)) - {CTRL_LABEL})
    scouter_data.val_conds = sorted(set(scouter_data.val_adata.obs["condition"].astype(str)) - {CTRL_LABEL})
    scouter_data.test_conds = list(scouter_data.val_conds)

    model = Scouter(scouter_data, device="auto")
    arch = model_cfg["model"]
    model.model_init(
        n_encoder=tuple(arch["n_encoder"]),
        n_out_encoder=arch["n_out_encoder"],
        n_decoder=tuple(arch["n_decoder"]),
        use_batch_norm=arch.get("use_batch_norm", True),
        use_layer_norm=arch.get("use_layer_norm", False),
        dropout_rate=arch.get("dropout_rate", 0.0),
    )
    model.network.load_state_dict(torch.load(args.model_dir / "model.pt", map_location="cpu"))
    model.network.eval()

    # ---- Predict per-pert μ over the global label space --------------------
    perts = list_train_perts(source_path, train_idx, cls_col=cls_col, ctrl_label=ctrl_label_in)
    # Drop perts not in the embd table (they'd raise inside scouter.pred()).
    perts = [p for p in perts if p in embd.index]
    print(f"[predict] label space: {len(perts)} perts", flush=True)

    gene_names = merged.var["gene_name"].astype(str).tolist()
    n_genes = len(gene_names)
    mu = np.zeros((len(perts), n_genes), dtype=np.float32)
    t0 = time.time()
    for i, pert in enumerate(perts):
        with torch.no_grad():
            pred = model.pred([pert], n_pred=n_pred, seed=pred_seed + i)[pert]
        mu[i] = np.asarray(pred, dtype=np.float32).mean(axis=0)
        if (i + 1) % 50 == 0 or i == len(perts) - 1:
            print(f"[predict]   pred {i+1}/{len(perts)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- Load test cells, align to model gene order ------------------------
    idx_file = splits_dir / ("id_test.txt.gz" if args.split == "id" else "ood_test.txt.gz")
    print(f"[predict] loading {idx_file.name}...", flush=True)
    test_adata = load_cells_by_index(source_path, idx_file)
    # Reorder/intersect test cells to the model's gene set.
    common = [g for g in gene_names if g in test_adata.var_names]
    if len(common) < n_genes:
        print(f"[predict]   warning: model had {n_genes} genes, test has {len(common)} in common", flush=True)
    test_adata = test_adata[:, common].copy()
    # Re-index mu to the common gene order (mu is in model.var order = gene_names).
    common_idx = np.asarray([gene_names.index(g) for g in common], dtype=np.int64)
    mu_aligned = mu[:, common_idx]

    X = _to_dense(test_adata.X).astype(np.float32, copy=False)
    obs_names = np.asarray(test_adata.obs_names.astype(str).tolist(), dtype=object)
    print(f"[predict]   test X shape={X.shape}", flush=True)

    # ---- Score: logit[i, p] = -||x_i - μ_p|| -------------------------------
    print("[predict] scoring (neg-L2)...", flush=True)
    t0 = time.time()
    # cdist returns (n_cells, n_perts); negate.
    logits = -cdist(X, mu_aligned, metric="euclidean").astype(np.float32)
    print(f"[predict]   scored in {time.time() - t0:.1f}s, shape={logits.shape}", flush=True)

    # ---- Write logits.h5 (cellot schema) -----------------------------------
    out_h5 = out_dir / "logits.h5"
    str_dt = h5py.string_dtype()
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("logits", data=logits, dtype="float32",
                         compression="lzf", chunks=True)
        f.create_dataset("obs_names", data=obs_names, dtype=str_dt, compression="lzf")
        f.create_dataset("perts", data=np.asarray(perts, dtype=object), dtype=str_dt)
    print(f"[predict] wrote {out_h5}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
