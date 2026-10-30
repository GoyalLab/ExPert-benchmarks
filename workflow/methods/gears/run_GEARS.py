# GEARS perturbation model. Two stages so the expensive, fold-independent step is
# done ONCE per size instead of once per fold:
#   --stage prep   : build the per-cell PyG graph dataset (cell_graphs.pkl) into a
#                    SHARED PertData dir (dataset_name=<size>). CPU-only; this is
#                    the hours-long "Creating pyg object for each cell" loop, and
#                    it depends only on the (fold-independent) full adata.
#   --stage train  : per fold — cache-hit the shared cell_graphs, apply the fold's
#                    custom split (in-memory, read-only), train, and score both
#                    id_test + ood_test into logits.h5 (datasets logits/perts/
#                    obs_names — the scouter/cellot convention collect reads).
#
# Sharing is safe for concurrent folds: GEARS' custom-split path only reads the
# split dict, and load()/new_data_process() detect the existing cell_graphs.pkl
# and skip the rebuild. Per-fold writes (split.pkl, model, logits) go under
# results_root, never the shared dir. GEARS_PREP_WORKERS parallelises the build.
#
# All paths come from the CLI; the in-tree gears package is self-located.
import argparse
import gc
import gzip
import logging
import os
import pickle
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Vendored gears package: workflow/methods/gears/GEARS (sibling of this file).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GEARS"))
from gears import PertData, GEARS  # noqa: E402
from gears.utils import create_cell_graph_for_prediction  # noqa: E402

try:  # PyG >= 2.0 moved DataLoader to .loader (kept in .data until ~2.5)
    from torch_geometric.loader import DataLoader as GeoDataLoader  # noqa: E402
except ImportError:
    from torch_geometric.data import DataLoader as GeoDataLoader  # noqa: E402

logging.info("Imported libraries")

# Fixed normalization target so per-dataset fingerprints and the ensemble scorer
# agree on the log-normalized space (see gears_ensemble.py, same constant).
GEARS_TARGET_SUM = 1e4


# HELPER FUNCTIONS ============================================================
def to_dense(x):
    return np.asarray(x.toarray(), dtype=np.float32) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def pearson_matrix(A, B):
    """Pearson r of every row in A against every row in B -> (n_A, n_B)."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    An = np.linalg.norm(A, axis=1, keepdims=True); An[An == 0] = 1.0
    Bn = np.linalg.norm(B, axis=1, keepdims=True); Bn[Bn == 0] = 1.0
    return (A / An) @ (B / Bn).T


def predict_with_controls(model, ctrl_X, perturbation_names, batch_size=128, n_samples=300):
    """Run model on a random sample of ctrl_X for each perturbation."""
    rng = np.random.default_rng(42)
    n = min(n_samples, ctrl_X.shape[0])
    ctrl_X = ctrl_X[rng.choice(ctrl_X.shape[0], size=n, replace=False)]

    predicted_means = []
    model.best_model.eval()
    for pert in perturbation_names:
        pert_idx = [model.node_map_pert[pert]]
        graphs = [
            create_cell_graph_for_prediction(ctrl_X[i].reshape(1, -1), pert_idx, [pert])
            for i in range(ctrl_X.shape[0])
        ]
        loader = GeoDataLoader(graphs, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch.to(model.device)
                preds.append(model.best_model(batch).detach().cpu().numpy())
        predicted_means.append(np.concatenate(preds).mean(axis=0))
        logging.info("Predicted " + pert)
    return np.array(predicted_means, dtype=np.float32)


def prepare_full_set(source, ctrl):
    """Build the GEARS-compatible full adata (ctrl + perturbed, normalised, shared
    genes). Deterministic, so prep and every fold produce an identical adata — the
    key to the cell_graphs cache being valid across folds. Returns (full_set,
    perturbed_set); perturbed_set keeps its raw 'dataset' column for OOD held-out."""
    control_set = sc.read_h5ad(ctrl)
    # Every cell in ctrl_h5ad IS a control, but the harmonised DB keeps each study's
    # own control nomenclature (NTC in 13 datasets, Non-Targeting / non-targeting /
    # SAFE / NO-TARGET / control elsewhere). The condition rename below maps only the
    # literal "control" to "ctrl", so without this 18/22 datasets ended up with an
    # EMPTY ctrl set and GEARS died in pertdata.py with `ValueError: high <= 0`
    # (np.random.randint(0, len(self.ctrl_adata))).
    control_set.obs["perturbation"] = "control"
    perturbed_set = sc.read_h5ad(source)
    logging.info("Loaded the gene control and perturbed sets")

    mask_ctrl = np.asarray(control_set.varm["feature_mask_per_dataset"])
    mask_pert = np.asarray(perturbed_set.varm["feature_mask_per_dataset"])
    shared_genes_mask = (mask_ctrl == 1).all(axis=1) & (mask_pert == 1).all(axis=1)
    full_set = ad.concat(adatas=[control_set, perturbed_set])
    full_set.obs.drop(columns=["condition"], errors="ignore", inplace=True)
    full_set.obs.rename(columns={"perturbation": "condition"}, inplace=True)
    full_set.obs.rename(columns={"dataset": "cell_type"}, inplace=True)
    full_set.obs = full_set.obs.loc[:, ~full_set.obs.columns.duplicated(keep="last")]
    full_set.obs["condition"] = full_set.obs["condition"].apply(lambda x: "ctrl" if x == "control" else "ctrl+" + x)
    full_set.var["gene_name"] = full_set.var.index
    logging.info("Reassigned annotated data columns to be compatible with GEARS")

    sc.pp.normalize_total(full_set)
    sc.pp.log1p(full_set)
    full_set = full_set[:, shared_genes_mask]
    logging.info(f"Normalised the dataset. Kept {shared_genes_mask.sum()} genes shared across all datasets")
    return full_set, perturbed_set


def torch_device():
    """cuda when a GPU is actually present, else cpu. The per-dataset GEARS models
    are small (median ~25 min) and were pinned to `cuda`, which crashes outright on
    a CPU node; auto-detection lets them run wherever they are scheduled."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_dataset_set(source, ctrl, dataset, gene_panel="all", keep_rows=None):
    """Per-dataset GEARS adata: dataset D's OWN cells (ctrl+perturbed) — the
    single-dataset regime GEARS was designed for. Deterministic per D so prep and
    every fold share D's cell_graphs cache.

    gene_panel: 'all' (DEFAULT) keeps the full source var axis, so every dataset's
    model shares ONE coordinate system (panels are hard zero-padded, so the extra
    columns are structural zeros). 'dataset' restricts to D's own active panel
    (feature_mask_per_dataset[:, D]).

    Returns (full_set_D, perturbed_full, panel_cols):
      full_set_D    — D's ctrl+perturbed cells on D's panel (for PertData/training)
      perturbed_full— the FULL perturbed adata (for global split-index -> obs_name
                      mapping; the fold split files index the full source)
      panel_cols    — 0-based gene indices of D's panel into the source var axis

    `keep_rows` (padding mode) restricts D's PERTURBED cells to those source rows
    before graphing. Without it GEARS graphs every cell the dataset has, which is
    right for `small` (~30k cells/dataset -> ~1GB of cell_graphs) but ruinous on the
    full store (~890k cells/dataset -> tens of GB each, ~600-950GB over 20 datasets,
    far past the allocation). Since the padding run only ever trains on the capped
    train+val indices, graphing anything else is pure waste as well as unaffordable.
    Controls are always kept: GEARS samples from them at prediction time.
    """
    control_set = sc.read_h5ad(ctrl)
    # Every cell in ctrl_h5ad IS a control, but the harmonised DB keeps each study's
    # own control nomenclature (NTC in 13 datasets, Non-Targeting / non-targeting /
    # SAFE / NO-TARGET / control elsewhere). The condition rename below maps only the
    # literal "control" to "ctrl", so without this 18/22 datasets ended up with an
    # EMPTY ctrl set and GEARS died in pertdata.py with `ValueError: high <= 0`
    # (np.random.randint(0, len(self.ctrl_adata))).
    control_set.obs["perturbation"] = "control"
    perturbed_full = sc.read_h5ad(source)
    logging.info("Loaded control + perturbed sets (per-dataset: %s)", dataset)

    d2i = {str(k): int(v) for k, v in dict(perturbed_full.uns["dataset_to_mask_idx"]).items()}
    if dataset not in d2i:
        raise ValueError(f"dataset {dataset!r} not in uns['dataset_to_mask_idx']")
    col = d2i[dataset]
    if str(gene_panel).lower() == "dataset":
        mask_ctrl = np.asarray(control_set.varm["feature_mask_per_dataset"])[:, col]
        mask_pert = np.asarray(perturbed_full.varm["feature_mask_per_dataset"])[:, col]
        panel_mask = (mask_ctrl == 1) & (mask_pert == 1)
    else:
        panel_mask = np.ones(perturbed_full.n_vars, dtype=bool)
    panel_cols = np.where(panel_mask)[0]

    # D's own cells only (ctrl + perturbed).
    ctrl_D = control_set[control_set.obs["dataset"].astype(str) == dataset]
    is_D = perturbed_full.obs["dataset"].astype(str).to_numpy() == dataset
    if keep_rows is not None:
        keep = np.zeros(perturbed_full.n_obs, dtype=bool)
        keep[np.asarray(keep_rows, dtype=np.int64)] = True
        n_all = int(is_D.sum())
        is_D &= keep
        logging.info("[%s] row subset: %d of D's %d perturbed cells kept for graphing",
                     dataset, int(is_D.sum()), n_all)
        if not is_D.any():
            raise SystemExit(f"[{dataset}] no perturbed cell of this dataset is in the "
                             "requested index subset — check --splits-dir/--train-index")
    pert_D = perturbed_full[is_D]
    full_set = ad.concat(adatas=[ctrl_D, pert_D])
    full_set.obs.drop(columns=["condition"], errors="ignore", inplace=True)
    full_set.obs.rename(columns={"perturbation": "condition"}, inplace=True)
    full_set.obs.rename(columns={"dataset": "cell_type"}, inplace=True)
    full_set.obs = full_set.obs.loc[:, ~full_set.obs.columns.duplicated(keep="last")]
    full_set.obs["condition"] = full_set.obs["condition"].apply(
        lambda x: "ctrl" if x == "control" else "ctrl+" + x)
    full_set.var["gene_name"] = full_set.var.index

    # Fixed target_sum (NOT median) so P lives in the SAME normalized space the
    # dataset-blind ensemble scorer re-normalizes test cells into — a per-population
    # median would differ between training here and mixed-dataset scoring.
    sc.pp.normalize_total(full_set, target_sum=GEARS_TARGET_SUM)
    sc.pp.log1p(full_set)
    full_set = full_set[:, panel_mask].copy()
    logging.info("[%s] %d cells on %d-gene panel (gene_panel=%s)",
                 dataset, full_set.n_obs, int(panel_mask.sum()), gene_panel)
    return full_set, perturbed_full, panel_cols


def _pd_kwargs(pert_gene_set):
    """PertData kwargs: a custom gene_set_path switches GEARS off its default
    perturbation graph (see cmd_prep). Empty dict = unchanged upstream behaviour."""
    if not pert_gene_set:
        return {}
    return {"gene_set_path": pert_gene_set, "default_pert_graph": False}


def cmd_prep(size, source, ctrl, pertdata_dir, dataset=None, gene_panel="all",
             pert_gene_set=None, keep_rows=None):
    """Stage 1: build/cache cell_graphs.pkl. Per-dataset (dataset given) builds D's
    graphs on D's panel under a D-scoped PertData dir; else the shared global set.

    `pert_gene_set` (padding mode) overrides GEARS' DEFAULT perturbation graph. That
    default gene set covers only 961 of the 10,245 zero-shot perturbations, so
    without an override GEARS is structurally unable to express 90.6% of the padding
    class space. It also changes `len(pert_list)`, hence `num_perts`, hence the shape
    of `nn.Embedding(num_perts, hidden_size)` -- which is why a padded GEARS must be
    retrained rather than re-scored.
    """
    if dataset is not None:
        full_set, _, _ = prepare_dataset_set(source, ctrl, dataset, gene_panel,
                                             keep_rows=keep_rows)
        os.makedirs(pertdata_dir, exist_ok=True)
        pert_data = PertData(pertdata_dir, **_pd_kwargs(pert_gene_set))
        pert_data.new_data_process(dataset_name=size, adata=full_set, save_anndata=False)
        logging.info("Per-dataset prep complete (%s): %s", dataset,
                     os.path.join(pertdata_dir, size))
        return
    full_set, _ = prepare_full_set(source, ctrl)
    os.makedirs(pertdata_dir, exist_ok=True)
    pert_data = PertData(pertdata_dir, **_pd_kwargs(pert_gene_set))
    # dataset_name=<size> (shared across folds). new_data_process builds
    # cell_graphs.pkl, or skips if it already exists (GEARS' "Irish" guard).
    pert_data.new_data_process(dataset_name=size, adata=full_set, save_anndata=False)
    logging.info("Prep complete: cell_graphs ready under " + os.path.join(pertdata_dir, size))


def cmd_train(fold, size, source, ctrl, splits_dir, results_root, pertdata_dir,
              pert_gene_set=None):
    """Stage 2 (per fold): cache-hit the shared graphs, apply the fold split,
    train, and score both splits to logits.csv."""
    full_set, perturbed_set = prepare_full_set(source, ctrl)

    def _load_idx(name):
        with gzip.open(os.path.join(splits_dir, name), mode="rt") as fh:
            return [int(line.rstrip()) for line in fh]

    training_indices = _load_idx("id_train.txt.gz")
    id_testing_indices = _load_idx("id_test.txt.gz")
    id_val_indices = _load_idx("id_val.txt.gz")
    ood_testing_indices = _load_idx("ood_test.txt.gz")
    logging.info("Collected indices")

    held_out_dataset = perturbed_set[ood_testing_indices].obs["dataset"].unique()
    print("Held out dataset: " + str(held_out_dataset))

    perturbed_obs_names = perturbed_set.obs_names
    train_obs = perturbed_obs_names[training_indices]
    id_test_obs = perturbed_obs_names[id_testing_indices]
    id_val_obs = perturbed_obs_names[id_val_indices]
    ood_test_obs = perturbed_obs_names[ood_testing_indices]

    # Shared PertData: dataset_name=<size>. new_data_process cache-hits the
    # cell_graphs built by --stage prep (skips the hours-long build) but still runs
    # get_DE_genes, which annotates full_set.uns['rank_genes_groups_cov_all'].
    pert_data = PertData(pertdata_dir, **_pd_kwargs(pert_gene_set))
    pert_data.new_data_process(dataset_name=size, adata=full_set, save_anndata=False)
    # new_data_process loads cell_graphs into dataset_processed; load() reloads it.
    # Drop the first copy first so we hold ~one cell_graphs in RAM, not two (the
    # double-load OOM-killed the single-job baseline at 128GB).
    pert_data.dataset_processed = None
    gc.collect()
    pert_data.load(data_path=os.path.join(pertdata_dir, size), adata=full_set)

    train_conditions = list(set(full_set[train_obs].obs["condition"].tolist()))
    id_test_conditions = list(set(full_set[id_test_obs].obs["condition"].tolist()))
    id_val_conditions = list(set(full_set[id_val_obs].obs["condition"].tolist()))
    ood_test_conditions = list(set(full_set[ood_test_obs].obs["condition"].tolist()))

    split_dict = {"train": train_conditions, "val": id_val_conditions, "test": id_test_conditions}
    print("Train perturbations:", len(split_dict["train"]))
    print("Val (ID) perturbations:", len(split_dict["val"]))
    print("Test (ID) perturbations:", len(split_dict["test"]))

    # Per-fold scratch under results_root (never the shared dir → no fold races).
    fold_out = os.path.join(results_root, "gears", "perturb-seq", size, "fold_" + str(fold))
    os.makedirs(fold_out, exist_ok=True)
    split_path = os.path.join(fold_out, "split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(split_dict, f)

    pert_data.prepare_split(split="custom", split_dict_path=split_path)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    logging.info("Dataloaders ready for fold " + str(fold))

    # Add missing conditions to rank_genes_groups_cov_all so deeper_analysis doesn't crash
    all_genes = pert_data.adata.var_names.tolist()
    rank_genes = pert_data.adata.uns["rank_genes_groups_cov_all"]
    cond_to_full = dict(pert_data.adata.obs[["condition", "condition_name"]].values)
    for cond in id_test_conditions + ood_test_conditions:
        full_id = cond_to_full.get(cond)
        if full_id and full_id not in rank_genes:
            rank_genes[full_id] = all_genes[:20]
    pert_data.adata.uns["rank_genes_groups_cov_all"] = rank_genes

    # GEARS TRAINING
    model_folder = os.path.join(fold_out, "model")
    os.makedirs(model_folder, exist_ok=True)  # GEARS.train() np.saves epoch_losses here
    gears_model = GEARS(pert_data, device=torch_device(), model_save_folder=model_folder,
                        weight_bias_track=False, proj_name="pertnet", exp_name="pertnet")
    gears_model.model_initialize(hidden_size=64)

    missing = [c for c in train_conditions if c not in gears_model.dict_filter and c != "ctrl"]
    print(f"{len(missing)} training perturbations missing from dict_filter: {missing}")

    gears_model.train(epochs=20, lr=1e-3, n_eval_cells=500, force_retrain=True)
    logging.info("Finished training.")

    # SCORING
    train_pert_genes = {c.replace("ctrl+", "") for c in train_conditions if c != "ctrl"}
    perturbation_names = [p for p in gears_model.pert_list if p in train_pert_genes and p in gears_model.node_map_pert]
    pert_name_to_idx = {p: i for i, p in enumerate(perturbation_names)}

    all_ctrl = pert_data.adata.obs["condition"] == "ctrl"
    id_ctrl_obs = pert_data.adata.obs_names[all_ctrl & ~pert_data.adata.obs["cell_type"].isin(held_out_dataset)]
    ood_ctrl_obs = pert_data.adata.obs_names[all_ctrl & pert_data.adata.obs["cell_type"].isin(held_out_dataset)]

    logging.info("Started scoring.")
    for split_name, test_obs, ctrl_obs in [
        ("id_test", id_test_obs, id_ctrl_obs),
        ("ood_test", ood_test_obs, ood_ctrl_obs),
    ]:
        ctrl_X = to_dense(pert_data.adata[ctrl_obs].X)
        mean_expression = ctrl_X.mean(axis=0)

        predicted_means = predict_with_controls(gears_model, ctrl_X, perturbation_names)
        P_centered = predicted_means - mean_expression

        all_test_obs = list(test_obs)
        n_total = len(all_test_obs)
        full_test_obs_idx = {obs: i for i, obs in enumerate(all_test_obs)}

        test_adata = pert_data.adata[pert_data.adata.obs_names.isin(test_obs)]
        present_obs = list(test_adata.obs_names)
        present_global_idx = [full_test_obs_idx[obs] for obs in present_obs]

        GT_cells = to_dense(test_adata.X)
        GT_centered = GT_cells - mean_expression

        cell_perturbations = test_adata.obs["condition"].str.replace(r"^ctrl\+", "", regex=True).values

        valid_local_idx = [i for i, p in enumerate(cell_perturbations) if p in pert_name_to_idx]
        valid_cell_perts = cell_perturbations[valid_local_idx]
        valid_scores = pearson_matrix(GT_centered[valid_local_idx], P_centered)
        correct_pert_idx = [pert_name_to_idx[p] for p in valid_cell_perts]
        correct_scores = valid_scores[np.arange(len(valid_local_idx)), correct_pert_idx]

        cell_mse, cell_r2 = [], []
        for i, pert in enumerate(valid_cell_perts):
            j = pert_name_to_idx[pert]
            cell_mse.append(mean_squared_error(GT_centered[valid_local_idx[i]], P_centered[j]))
            cell_r2.append(r2_score(GT_centered[valid_local_idx[i]], P_centered[j]))

        score_matrix = np.full((n_total, len(perturbation_names)), np.nan, dtype=np.float32)
        for j, local_i in enumerate(valid_local_idx):
            score_matrix[present_global_idx[local_i]] = valid_scores[j]

        exported_results = {
            "SCORE_MATRIX": score_matrix,
            "CORRECT_PERT_SCORES": correct_scores,
            "CORRECT_PERT_MSE": cell_mse,
            "CORRECT_SCORE_R2": cell_r2,
            "PERTURBATION_NAMES": perturbation_names,
        }

        logits_save_folder = os.path.join(fold_out, split_name)
        os.makedirs(logits_save_folder, exist_ok=True)
        with open(os.path.join(logits_save_folder, f"fold_{fold}__type_{split_name}.pkl"), "wb") as f:
            pickle.dump(exported_results, f)
        # logits.h5 (cells x perts), columns sorted ascending. Datasets match the
        # scouter/cellot convention collect reads: logits / perts / obs_names.
        order = np.argsort(perturbation_names)
        sorted_perts = [perturbation_names[i] for i in order]
        out_path = os.path.join(logits_save_folder, "logits.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("logits", data=score_matrix[:, order], compression="lzf")
            f.create_dataset("perts", data=np.array(sorted_perts, dtype=h5py.string_dtype()))
            f.create_dataset("obs_names", data=np.array(list(all_test_obs), dtype=h5py.string_dtype()))
        logging.info(f"Saved {split_name}: {len(valid_local_idx)}/{n_total} valid cells -> {out_path}")

    logging.info("Scoring complete.")


def cmd_train_dataset(fold, size, source, ctrl, splits_dir, results_root,
                      pertdata_dir, dataset, gene_panel="all", pert_gene_set=None,
                      zs_perts=False, predict_perts=None, train_index="id_train",
                      val_index="id_val", keep_rows=None):
    """Per-dataset train: fit D's GEARS model on D's OWN train cells + panel, then
    save D's per-pert predicted fingerprints (P over D's panel, centered later by
    the ensemble). No scoring here — the dataset-blind ensemble eval reads every
    dataset's fingerprints.npz and combines them (genepert-style)."""
    full_set, perturbed_full, panel_cols = prepare_dataset_set(source, ctrl, dataset, gene_panel,
                                                               keep_rows=keep_rows)

    def _load_idx(name):
        with gzip.open(os.path.join(splits_dir, name), mode="rt") as fh:
            return [int(line.rstrip()) for line in fh if line.strip()]

    obs_full = perturbed_full.obs_names
    d_obs = set(full_set.obs_names)               # D's cells (ctrl+perturbed)
    def _conds(idx_name):
        obs = obs_full[_load_idx(idx_name)]
        obs = [o for o in obs if o in d_obs]       # restrict to D's cells
        if not obs:
            return []
        return list(set(full_set[obs].obs["condition"].tolist()))

    train_conditions = [c for c in _conds(f"{train_index}.txt.gz") if c != "ctrl"]
    val_conditions   = [c for c in _conds(f"{val_index}.txt.gz")   if c != "ctrl"]
    test_conditions  = [c for c in _conds("id_test.txt.gz")  if c != "ctrl"]
    if not train_conditions:
        raise RuntimeError(f"dataset {dataset} has no train conditions this fold")
    logging.info("[%s] train/val/test conds: %d/%d/%d", dataset,
                 len(train_conditions), len(val_conditions), len(test_conditions))

    ds_out = os.path.join(results_root, "gears", "perturb-seq", size,
                          f"fold_{fold}", "per_dataset", dataset)
    os.makedirs(ds_out, exist_ok=True)

    pert_data = PertData(pertdata_dir, **_pd_kwargs(pert_gene_set))
    pert_data.new_data_process(dataset_name=size, adata=full_set, save_anndata=False)
    pert_data.dataset_processed = None
    gc.collect()
    pert_data.load(data_path=os.path.join(pertdata_dir, size), adata=full_set)

    split_dict = {"train": train_conditions,
                  "val": val_conditions or train_conditions[:1],
                  "test": test_conditions or train_conditions[:1]}
    split_path = os.path.join(ds_out, "split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(split_dict, f)
    pert_data.prepare_split(split="custom", split_dict_path=split_path)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    # keep deeper_analysis from crashing on unseen conditions
    all_genes = pert_data.adata.var_names.tolist()
    rank_genes = pert_data.adata.uns["rank_genes_groups_cov_all"]
    cond_to_full = dict(pert_data.adata.obs[["condition", "condition_name"]].values)
    for cond in test_conditions:
        fid = cond_to_full.get(cond)
        if fid and fid not in rank_genes:
            rank_genes[fid] = all_genes[:20]
    pert_data.adata.uns["rank_genes_groups_cov_all"] = rank_genes

    model_folder = os.path.join(ds_out, "model")
    os.makedirs(model_folder, exist_ok=True)
    gears_model = GEARS(pert_data, device=torch_device(), model_save_folder=model_folder,
                        weight_bias_track=False, proj_name="pertnet", exp_name="pertnet")
    gears_model.model_initialize(hidden_size=64)
    gears_model.train(epochs=20, lr=1e-3, n_eval_cells=500, force_retrain=True)
    logging.info("[%s] training done", dataset)

    # Per-pert fingerprints P (predicted mean over D's panel).
    train_pert_genes = {c.replace("ctrl+", "") for c in train_conditions}
    if zs_perts:
        # PADDING MODE: emit fingerprints for perturbations this model never TRAINED
        # on. GEARS can do that for any gene that is a node in the perturbation graph
        # (node_map_pert) -- the GNN reads the node's neighbourhood, not a fitted
        # per-perturbation parameter. Dropping the `in train_pert_genes` clause is
        # therefore the whole change; the graph must already have been widened at
        # prep time (--pert-gene-set), or coverage collapses to ~9%.
        want = set(predict_perts) if predict_perts else None
        perts = sorted(p for p in gears_model.pert_list
                       if p in gears_model.node_map_pert and (want is None or p in want))
        missing = (sorted(want - set(perts))[:10] if want else [])
        logging.info("[%s] ZS mode: %d/%d requested classes are nodes in the pert graph%s",
                     dataset, len(perts), len(want) if want else -1,
                     f"; e.g. absent: {missing}" if missing else "")
        if not perts:
            raise SystemExit(
                f"[{dataset}] none of the requested classes are in the perturbation graph. "
                "Pass --pert-gene-set built by workflow/scripts/make_gears_geneset.py.")
    else:
        perts = sorted(p for p in gears_model.pert_list
                       if p in train_pert_genes and p in gears_model.node_map_pert)
    ctrl_X = to_dense(pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].X)
    ctrl_mean = ctrl_X.mean(axis=0).astype(np.float32)
    P = predict_with_controls(gears_model, ctrl_X, perts).astype(np.float32)   # raw predicted means

    np.savez(os.path.join(ds_out, "fingerprints.npz"),
             P=P, perts=np.array(perts), panel_cols=panel_cols.astype(np.int64),
             ctrl_mean=ctrl_mean, dataset=dataset)
    logging.info("[%s] saved fingerprints.npz: %d perts x %d genes", dataset, len(perts), P.shape[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run GEARS for one fold, or prep the shared cell-graph cache.")
    ap.add_argument("--stage", choices=["prep", "train"], default="train")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--size", required=True, help="Dataset size, e.g. small / full (also the shared dataset_name).")
    ap.add_argument("--source", required=True, help="Perturbed source .h5ad.")
    ap.add_argument("--ctrl", required=True, help="Control .h5ad.")
    ap.add_argument("--splits-dir", help="Fold dir with {id_train,id_val,id_test,ood_test}.txt.gz (train stage).")
    ap.add_argument("--results-root", help="RESULTS root; logits land under <root>/gears/perturb-seq/... (train stage).")
    ap.add_argument("--pertdata-dir", required=True, help="GEARS PertData dir (cell_graphs cache; per size, or per size+dataset).")
    ap.add_argument("--dataset", default=None, help="Per-dataset mode: train/prep only this dataset.")
    ap.add_argument("--gene-panel", dest="gene_panel", default="all",
                    choices=("all", "dataset"),
                    help="gene space for the per-dataset model (see config gene_panel)")
    # ---- padding mode (docs/padding_scope.md). All three flags default OFF, so the
    # main grid's prep/train behaviour is byte-for-byte unchanged without them.
    ap.add_argument("--pert-gene-set", dest="pert_gene_set", default=None,
                    help="pickle of gene symbols for the perturbation graph "
                         "(gene_set_path). Needed for unseen perturbations: GEARS' "
                         "default graph covers only 961/10,245 of them.")
    ap.add_argument("--zs-perts", dest="zs_perts", action="store_true",
                    help="emit fingerprints for perturbations with NO training cells "
                         "(drops the `in train_pert_genes` filter; graph nodes still required)")
    ap.add_argument("--predict-perts", dest="predict_perts", default=None,
                    help="newline file restricting --zs-perts output to this class space")
    ap.add_argument("--train-index", dest="train_index", default="id_train",
                    help="training index basename in --splits-dir")
    ap.add_argument("--val-index", dest="val_index", default="id_val",
                    help="validation index basename in --splits-dir")
    ap.add_argument("--graph-indices", dest="graph_indices", nargs="*", default=None,
                    help="index files (absolute .txt.gz paths) whose rows are the ONLY "
                         "perturbed cells to build cell graphs for. Pass the SAME list to "
                         "--stage prep and --stage train: the graph cache is shared, so a "
                         "mismatch means training cells without graphs. Unset = every cell "
                         "of the dataset (unchanged main-grid behaviour).")
    args = ap.parse_args()

    keep_rows = None
    if args.graph_indices:
        rows = []
        for f in args.graph_indices:
            with gzip.open(f, mode="rt") as fh:
                rows += [int(l) for l in fh if l.strip()]
        keep_rows = np.unique(np.asarray(rows, dtype=np.int64))
        logging.info("--graph-indices: %d unique source rows from %d file(s)",
                     keep_rows.size, len(args.graph_indices))

    predict_perts = None
    if args.predict_perts:
        with open(args.predict_perts) as fh:
            predict_perts = [l.strip() for l in fh if l.strip()]
    if args.zs_perts and not args.pert_gene_set:
        logging.warning("--zs-perts without --pert-gene-set: GEARS' DEFAULT perturbation "
                        "graph covers ~9%% of the zero-shot class space, so most classes "
                        "will be absent. Build one with workflow/scripts/make_gears_geneset.py.")

    if args.stage == "prep":
        cmd_prep(args.size.lower(), args.source, args.ctrl, args.pertdata_dir,
                 dataset=args.dataset, gene_panel=args.gene_panel,
                 pert_gene_set=args.pert_gene_set, keep_rows=keep_rows)
    else:
        if not args.splits_dir or not args.results_root:
            ap.error("--stage train requires --splits-dir and --results-root")
        if args.dataset is not None:
            cmd_train_dataset(args.fold, args.size.lower(), args.source, args.ctrl,
                              args.splits_dir, args.results_root, args.pertdata_dir,
                              args.dataset, gene_panel=args.gene_panel,
                              pert_gene_set=args.pert_gene_set, zs_perts=args.zs_perts,
                              predict_perts=predict_perts, train_index=args.train_index,
                              val_index=args.val_index, keep_rows=keep_rows)
        else:
            cmd_train(args.fold, args.size.lower(), args.source, args.ctrl,
                      args.splits_dir, args.results_root, args.pertdata_dir,
                      pert_gene_set=args.pert_gene_set)
