"""Train a single Scouter model on a fold's id_train+id_val cells.

Reads the same `data/splits/<size>/fold_N/...` layout cellot uses, plus the
sibling metadata.json for the source.h5ad path and held-out contexts (so we
can drop ood-context controls from the AE-equivalent training pool, matching
the leakage guarantee the other methods provide).

Inputs are bridged via utils/splits.py + utils/scouter_adapter.py — no shared
code with cellot.

Output: ${outdir}/{model.pt, config.yaml, embd.parquet, gene_names.json,
cache/status}.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse

# Make `utils.*` importable when running as `python ./scripts/train.py`.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from utils.splits import (
    fold_index_from_dir,
    filter_by_context,
    held_out_from_metadata,
    load_cells_by_index,
    load_ctrl,
    read_metadata,
    shared_gene_cols,
    dataset_gene_cols,
    dataset_indices_at,
)
from utils.scouter_adapter import (
    CTRL_LABEL,
    attach_embedding_indices,
    build_genept_embd,
    keep_perturbations_with_var_match,
    merge_scouter_adata,
)

from scouter import Scouter, ScouterData


def _patch_scouter_lowmem():
    """Deep low-mem surgery on scouter-learn's data path. Stock BalancedDataset does
    `true_expr = pert_adata.X.toarray()` — a DENSE copy of every perturbed cell
    (12M×~16k×4B ≈ TBs). Even a sparse copy of the perturbed subset is a second
    full-size matrix (~150GB) on top of ScouterData.adata + train_adata. At full
    scale those copies stack to 500GB+ (OOM even at 480GB).

    Here BalancedDataset holds only a REFERENCE to the parent adata's CSR (no
    subset copy at all) plus the perturbed-cell row positions, and densifies ONE
    row per __getitem__. Combined with train.py freeing ScouterData.adata right
    after the train/val slice, scouter keeps ~one in-memory copy of the cells
    instead of three. Numerically identical: same per-cell vectors, lazily
    materialized. Controls stay dense (small). Nothing outside BalancedDataset
    reads .pert_adata/.true_expr, and Scouter.train() never touches .all_adata,
    so this is safe."""
    from scouter import _datasets as _sd

    def __init__(self, adata, key_label, key_embd_index, shuffle=True, seed=24):
        np.random.seed(seed)
        lab = np.asarray(adata.obs[key_label].values)
        self._X = adata.X.tocsr()                               # reference to parent CSR (no copy)
        self.ctrl_expr = np.asarray(adata[lab == "ctrl"].X.todense())   # dense ctrl (small)
        self._pert_rows = np.nonzero(lab != "ctrl")[0]          # pert-cell positions in adata.X
        self.pert_idx = np.asarray(adata.obs[key_embd_index].values)[self._pert_rows]
        self.conditions = lab[self._pert_rows]
        self.bcode = adata.obs.index.to_numpy()[self._pert_rows].tolist()
        self.n_pert_cells = self._pert_rows.shape[0]
        self.n_ctrl_cells = self.ctrl_expr.shape[0]
        self.PertCellIndx = np.arange(self.n_pert_cells)
        self.CtrlCellIndx = self._GetCtrlIndx(self.n_pert_cells)
        if shuffle:
            np.random.shuffle(self.PertCellIndx)
            np.random.shuffle(self.CtrlCellIndx)
        if len(self.PertCellIndx) != len(self.CtrlCellIndx):
            raise ValueError("Number of Perturbed cells and Control cells must be equal.")

    def __getitem__(self, idx):
        prow = self._pert_rows[self.PertCellIndx[idx]]
        te = np.asarray(self._X[prow].todense()).ravel()       # densify one perturbed cell
        return (torch.tensor(self.ctrl_expr[self.CtrlCellIndx[idx]], dtype=torch.float32),
                torch.tensor(te, dtype=torch.float32),
                torch.tensor(self.pert_idx[self.PertCellIndx[idx]], dtype=torch.long),
                self.bcode[self.PertCellIndx[idx]])

    _sd.BalancedDataset.__init__ = __init__
    _sd.BalancedDataset.__getitem__ = __getitem__
    print("[train] patched scouter BalancedDataset (deep low-mem: parent-CSR ref, lazy densify)", flush=True)


_patch_scouter_lowmem()


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits_dir", required=True, type=Path,
                    help="data/splits/<size>/fold_N/")
    ap.add_argument("--config", required=True, type=Path,
                    help="configs/tasks/perturb-seq/<size>.yaml — task config")
    ap.add_argument("--model_config", type=Path, default=None,
                    help="configs/models/scouter.yaml (defaults to ./configs/models/scouter.yaml)")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Output dir for model.pt + config.yaml + cache/status")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gene-panel", dest="gene_panel", default="all",
                    choices=("all", "dataset"),
                    help="gene space for the per-dataset model (see config gene_panel)")
    ap.add_argument("--shared_genes", action="store_true",
                    help="Restrict the expression feature space to genes measured "
                         "in every dataset (varm feature_mask_per_dataset). Fixes "
                         "near-random logits from zero-padded union genes and cuts "
                         "RAM ~3x. Perturbations are unaffected (encoded via GenePT).")
    ap.add_argument("--ctrl-h5ad", dest="ctrl_h5ad", default=None, type=Path,
                    help="control h5ad; overrides the task config's ctrl_h5ad (the "
                         "pipeline passes config['ctrl_h5ad'])")
    ap.add_argument("--genept-path", dest="genept_path", default=None, type=Path,
                    help="the default GenePT {gene: vector} pickle; overrides the task "
                         "config's embedding_path (the pipeline passes "
                         "config['scouter_gene_embeddings']). --embedding-path wins over it.")
    ap.add_argument("--embedding-path", dest="embedding_path", default=None, type=Path,
                    help="override the task config's embedding_path. Must be a pickle "
                         "{gene: vector}. Used by the scouter_atoms arm to feed ExPert's "
                         "LEARNED class table in place of the GenePT text table.")
    ap.add_argument("--predict-perts", dest="predict_perts", default=None, type=Path,
                    help="newline file of classes to export fingerprints for, INCLUDING "
                         "ones with no training cells (padding mode). Their embedding rows "
                         "are added to the table BEFORE the model is built, so the frozen "
                         "`self.embd` parameter already contains them and nothing needs "
                         "patching at predict time. Training is unaffected: no training "
                         "cell carries those conditions, so the extra rows are never read "
                         "by the training loop.")
    ap.add_argument("--dataset", default=None,
                    help="PER-DATASET mode: train on ONLY this dataset's cells + its "
                         "OWN gene panel (relieves the global-merge OOM), then save its "
                         "per-pert predicted fingerprints for the dataset-blind ensemble.")
    return ap.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    cfg = _load_yaml(args.config)
    model_cfg_path = args.model_config or (THIS_DIR.parent / "configs" / "models" / "scouter.yaml")
    model_cfg = _load_yaml(model_cfg_path)

    splits_dir = args.splits_dir.resolve()
    outdir = args.outdir.resolve()
    cache_dir = outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # ---- Resolve source paths from metadata + config -----------------------
    meta = read_metadata(splits_dir)
    source_path = Path(data_cfg.get("source_h5ad") or meta["source_path"]).resolve()
    ctrl_path = Path(args.ctrl_h5ad or data_cfg["ctrl_h5ad"]).resolve()
    embedding_path = Path(args.embedding_path or args.genept_path
                          or data_cfg["embedding_path"]).resolve()
    cls_col = data_cfg.get("condition_col", meta.get("cls_col", "perturbation"))
    ctrl_label_in = data_cfg.get("ctrl_label", "ctrl")
    ctx_col, held_out = held_out_from_metadata(splits_dir)
    fold_idx = fold_index_from_dir(splits_dir)

    print(f"[train] source       : {source_path}", flush=True)
    print(f"[train] ctrl         : {ctrl_path}", flush=True)
    print(f"[train] embedding    : {embedding_path}", flush=True)
    print(f"[train] cls_col      : {cls_col}", flush=True)
    print(f"[train] fold         : fold_{fold_idx}", flush=True)
    print(f"[train] held_out_ctx : {held_out}", flush=True)

    # ---- Load cells --------------------------------------------------------
    train_idx = splits_dir / "id_train.txt.gz"
    val_idx = splits_dir / "id_val.txt.gz"

    if args.dataset:
        # gene_cols=None means "all genes". Default gene_panel='all' keeps the full
        # var axis so every dataset's model shares ONE coordinate system (the panels
        # are hard zero-padded); 'dataset' restricts to D's own active panel.
        if args.gene_panel == "dataset":
            gene_cols = dataset_gene_cols(source_path, args.dataset)
        else:
            gene_cols = None
        print(f"[train] per-dataset {args.dataset} (gene_panel={args.gene_panel}): "
              f"{'all' if gene_cols is None else len(gene_cols)} genes", flush=True)
    else:
        gene_cols = shared_gene_cols(source_path) if args.shared_genes else None
        if gene_cols is not None:
            print(f"[train] shared-gene subset: {len(gene_cols)} genes measured in every "
                  f"dataset (from feature_mask_per_dataset)", flush=True)
        elif args.shared_genes:
            print("[train] --shared_genes set but no feature_mask_per_dataset in source; "
                  "keeping all genes", flush=True)

    # Per-dataset: load ONLY D's own train/val cells (this is what relieves the
    # global-merge OOM — was 892GB loading+merging all 12M cells at full).
    tr_idx = dataset_indices_at(source_path, train_idx, args.dataset) if args.dataset else None
    va_idx = dataset_indices_at(source_path, val_idx, args.dataset) if args.dataset else None

    print("[train] loading id_train cells...", flush=True)
    perturbed_train = load_cells_by_index(source_path, train_idx, gene_cols=gene_cols, indices=tr_idx)
    print(f"[train]   shape={perturbed_train.shape}", flush=True)

    print("[train] loading id_val cells...", flush=True)
    perturbed_val = load_cells_by_index(source_path, val_idx, gene_cols=gene_cols, indices=va_idx)
    print(f"[train]   shape={perturbed_val.shape}", flush=True)

    print("[train] loading controls...", flush=True)
    ctrl = load_ctrl(ctrl_path)
    if gene_cols is not None:                        # align ctrl to the same feature set
        ctrl = ctrl[:, gene_cols].copy()
    if args.dataset:                                 # D's own controls only (batch baseline)
        dcol = ctx_col if (ctx_col and ctx_col in ctrl.obs) else "dataset"
        ctrl = ctrl[ctrl.obs[dcol].astype(str) == args.dataset].copy()
    print(f"[train]   ctrl shape={ctrl.shape}", flush=True)

    # Same held-out filter cellot applies to controls — exclude ood-context
    # controls from the training pool so the AE doesn't see them.
    if held_out:
        n0 = ctrl.n_obs
        ctrl = filter_by_context(ctrl, ctx_col, held_out, keep_held_out=False)
        print(f"[train]   ctrl after held-out filter: {ctrl.n_obs} (was {n0})", flush=True)

    # ---- Merge into scouter-shaped adata ----------------------------------
    print("[train] merging controls + perturbed (train + val)...", flush=True)
    merged = merge_scouter_adata(
        perturbed_parts={"train": perturbed_train, "val": perturbed_val},
        ctrl=ctrl,
        cls_col=cls_col,
        ctrl_label_in=ctrl_label_in,
    )
    print(f"[train]   merged shape={merged.shape}, n_conds={merged.obs['condition'].nunique()}", flush=True)
    del perturbed_train, perturbed_val, ctrl        # now inside `merged`; free the copies
    gc.collect()

    # Scouter requires the perturbation label to be a gene present in var.
    # Multi-gene conditions like "GENE_A+GENE_B" pass when both are present.
    merged, dropped = keep_perturbations_with_var_match(merged)
    if dropped:
        print(f"[train] dropped {len(dropped)} conditions absent from var (first 10): {dropped[:10]}", flush=True)

    # ---- Embedding ---------------------------------------------------------
    print("[train] building GenePT embedding table...", flush=True)
    needed_genes = set()
    for cond in merged.obs["condition"].astype(str).unique():
        for g in cond.split("+"):
            needed_genes.add(g)
    predict_perts = None
    if args.predict_perts:
        predict_perts = [l.strip() for l in args.predict_perts.read_text().splitlines()
                         if l.strip()]
        before = len(needed_genes)
        needed_genes |= set(predict_perts)
        print(f"[train] padding mode: +{len(needed_genes) - before} unseen classes into the "
              f"embedding table ({len(predict_perts)} requested)", flush=True)
    embd = build_genept_embd(embedding_path, needed_genes)
    print(f"[train]   embd shape={embd.shape}", flush=True)

    merged, ctrl_idx = attach_embedding_indices(merged, embd, key="embd_index")
    print(f"[train]   ctrl_idx={ctrl_idx}", flush=True)

    # ---- ScouterData setup ON the merged adata ----------------------------
    # setup_ad / gene_ranks / get_dropout_non_zero_genes write into adata.uns;
    # the per-split adatas must be sliced AFTER those calls so the keys
    # (notably `gene_idx_non_zeros`) carry over via the .copy() below.
    scouter_data = ScouterData(
        adata=merged,
        embd=embd,
        key_label="condition",
        key_var_genename="gene_name",
    )
    scouter_data.setup_ad("embd_index")
    scouter_data.gene_ranks()
    scouter_data.get_dropout_non_zero_genes()

    # ScouterData copied `merged` (self.adata = adata.copy()); drop our reference so
    # we don't hold a redundant full-size (12M-cell) sparse copy during training.
    del merged; gc.collect()

    train_mask = (scouter_data.adata.obs["_split"].astype(str).isin(["train", "ctrl"])).to_numpy()
    val_mask = (scouter_data.adata.obs["_split"].astype(str).isin(["val", "ctrl"])).to_numpy()
    train_adata = scouter_data.adata[train_mask].copy()
    val_adata = scouter_data.adata[val_mask].copy()

    train_conds = sorted(set(train_adata.obs["condition"].astype(str).unique()) - {CTRL_LABEL})
    val_conds = sorted(set(val_adata.obs["condition"].astype(str).unique()) - {CTRL_LABEL})
    scouter_data.train_conds = train_conds
    scouter_data.val_conds = val_conds
    # Scouter's __init__ reads test_adata even when we never call .pred() on
    # it from train.py. Reuse val_adata as a placeholder — real test scoring
    # happens in predict.py against the actual id_test/ood_test indices.
    scouter_data.test_conds = list(val_conds)
    scouter_data.train_adata = train_adata
    scouter_data.val_adata = val_adata
    scouter_data.test_adata = val_adata
    # Deep low-mem: train() only ever reads train_adata/val_adata (+ their .uns);
    # ScouterData.adata (the full train+val+ctrl copy) feeds only .pred()/eval.
    # Global mode repoints it at train_adata to drop the full ~180GB copy. Per-
    # dataset mode KEEPS the full adata (with controls) because we call model.pred()
    # below to export fingerprints — and D's data is small, so memory is fine.
    if not args.dataset:
        scouter_data.adata = train_adata
    gc.collect()
    print(f"[train]   train_conds={len(train_conds)}  val_conds={len(val_conds)}", flush=True)

    # ---- Model init + train -------------------------------------------------
    model = Scouter(scouter_data, device="auto")
    arch = model_cfg["model"]
    loss_cfg = model_cfg["loss"]
    optim_cfg = model_cfg["optim"]
    model.model_init(
        n_encoder=tuple(arch["n_encoder"]),
        n_out_encoder=arch["n_out_encoder"],
        n_decoder=tuple(arch["n_decoder"]),
        use_batch_norm=arch.get("use_batch_norm", True),
        use_layer_norm=arch.get("use_layer_norm", False),
        dropout_rate=arch.get("dropout_rate", 0.0),
    )

    t0 = time.time()
    print("[train] starting Scouter.train()...", flush=True)
    model.train(
        nonzero_idx_key="gene_idx_non_zeros",
        batch_size=int(train_cfg.get("batch_size", 256)),
        loss_gamma=float(loss_cfg.get("loss_gamma", 0.0)),
        loss_lambda=float(loss_cfg.get("loss_lambda", 0.5)),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        sched_gamma=float(optim_cfg.get("sched_gamma", 0.9)),
        n_epochs=int(train_cfg.get("epochs", 40)),
        patience=int(train_cfg.get("patience", 5)),
    )
    train_seconds = time.time() - t0
    history = getattr(model, "loss_history", {"train_loss": [], "val_loss": []})
    print(f"[train]   trained in {train_seconds:.1f}s", flush=True)

    # ---- Save --------------------------------------------------------------
    print("[train] saving artifacts...", flush=True)
    torch.save(model.network.state_dict(), outdir / "model.pt")
    # pickle (not parquet) to dodge needing pyarrow as a runtime dep
    embd.to_pickle(outdir / "embd.pkl")
    with open(outdir / "gene_names.json", "w") as f:
        json.dump(scouter_data.adata.var["gene_name"].astype(str).tolist(), f)
    resolved_config = {
        "task": cfg,
        "model": model_cfg,
        "fold": int(fold_idx),
        "splits_dir": str(splits_dir),
        "source_path": str(source_path),
        "ctrl_path": str(ctrl_path),
        "embedding_path": str(embedding_path),
        "train_conds_count": len(train_conds),
        "val_conds_count": len(val_conds),
        "train_seconds": float(train_seconds),
        "n_params": int(sum(p.numel() for p in model.network.parameters())),
        "ctrl_idx": int(ctrl_idx),
    }
    with open(outdir / "config.yaml", "w") as f:
        yaml.safe_dump(resolved_config, f)
    with open(outdir / "loss_history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f)

    # ---- Per-dataset: export per-pert fingerprints for the dataset-blind ensemble.
    # μ_p = mean of model.pred([p]) over D's panel (log-norm space, like predict.py);
    # scouter_ensemble.py centers it by D's ctrl_mean and does the genepert combiner.
    if args.dataset:
        print("[train] exporting per-pert fingerprints (model.pred)...", flush=True)
        if predict_perts is None:
            pert_genes = sorted({(c[5:] if c.startswith("ctrl+") else c)
                                 for c in train_conds if c != CTRL_LABEL})
        else:
            # Export the EVALUATED class space, not the trained one: scouter's
            # perturbation input is a frozen embedding lookup, so a class it never saw
            # is predictable as long as it has a table row.
            pert_genes = sorted(set(predict_perts))
        missing = [p for p in pert_genes if p not in embd.index]
        pert_genes = [p for p in pert_genes if p in embd.index]
        if missing:
            print(f"[train]   {len(missing)} classes have no embedding row and are skipped "
                  f"(first 10: {missing[:10]})", flush=True)
        gene_names = scouter_data.adata.var["gene_name"].astype(str).tolist()
        mu = np.zeros((len(pert_genes), len(gene_names)), dtype=np.float32)
        for i, pert in enumerate(pert_genes):
            with torch.no_grad():
                pred = model.pred([pert], n_pred=300, seed=args.seed + i)[pert]
            mu[i] = np.asarray(pred, dtype=np.float32).mean(axis=0)
            if (i + 1) % 50 == 0 or i == len(pert_genes) - 1:
                print(f"[train]   fp {i + 1}/{len(pert_genes)}", flush=True)
        ad_all = scouter_data.adata
        cX = ad_all[ad_all.obs["condition"].astype(str) == CTRL_LABEL].X
        cX = cX.toarray() if sparse.issparse(cX) else np.asarray(cX)
        ctrl_mean = np.asarray(cX.mean(axis=0), dtype=np.float32).ravel()
        panel_cols = (gene_cols if gene_cols is not None
                      else np.arange(len(gene_names))).astype(np.int64)
        np.savez(outdir / "fingerprints.npz", P=mu, perts=np.array(pert_genes),
                 panel_cols=panel_cols, ctrl_mean=ctrl_mean, dataset=args.dataset)
        print(f"[train]   saved fingerprints.npz: {len(pert_genes)} perts x "
              f"{len(gene_names)} genes", flush=True)

    (cache_dir / "status").write_text("done")
    print(f"[train] done → {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
