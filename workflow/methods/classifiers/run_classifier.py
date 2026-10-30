# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.
"""Supervised discriminative baselines for the benchmark: XGBoost and a linear
(SGD log-loss) classifier trained directly on cell -> perturbation.

Pipeline port of ExPert/expert/benchmarks/xgboost.py. The model, masking,
control-delta and scoring logic are carried over unchanged; what changed is only
what the pipeline needs:

  * --model {xgboost,linreg} replaces --linear (one script, two methods).
  * writes <results>/<method>/perturb-seq/<size>/<fold>/{id_test,ood_test}/logits.h5
    in the benchmark schema (logits / idx / columns) directly -- the step
    analysis/readout/export_xgb_logits.py used to do by hand. `idx` is the split
    index file itself (rows are X[split_idx], so split-file order), asserted
    against the row count rather than assumed.
  * classes are encoded on the TRAINING labels (was: the whole adata). Identical
    on the current DB -- every fold trains on all 200 perturbations -- but the
    sklearn XGBoost API rejects non-contiguous class ids, so a fold whose eval
    split carried a class absent from train would otherwise crash.
  * library-size normalisation takes --target_sum (default 1e6 = xgboost.py's --cpm,
    the documented small-200 setting); --cpm is kept as an alias.
  * --dataset_key defaults to `dataset` (was `ds_label`, not a column here).
  * optional --max_train_cells (default 0 = all) and per-round timing in the log.
"""

import os
import gzip
import json
import time
import logging
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import h5py
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _adata_from_split_dir(split_dir):
    """Read source_path from <split_dir>/../metadata.json."""
    parent = os.path.dirname(os.path.normpath(split_dir))
    mf = os.path.join(parent, "metadata.json")
    if not os.path.exists(mf):
        raise FileNotFoundError(f"No metadata.json in parent of split_dir: {mf}")
    with open(mf) as f:
        md = json.load(f)
    src = md.get("source_path")
    if not src:
        raise KeyError(f'metadata.json {mf} has no "source_path" entry.')
    return src


def _load_split_indices(split_dir):
    """Load train/val/test/ood_test indices from a fold directory (e.g. .../splits/full/fold_0)."""
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split_dir does not exist: {split_dir}")

    def _load(name):
        fp = os.path.join(split_dir, f"{name}.txt.gz")
        if not os.path.exists(fp):
            return np.array([], dtype=np.int64)
        return np.loadtxt(fp, dtype=np.int64, ndmin=1)

    return _load("id_train"), _load("id_val"), _load("id_test"), _load("ood_test")


def _normalise(adata, target_sum, log1p):
    if target_sum and target_sum > 0:
        sc.pp.normalize_total(adata, target_sum=float(target_sum))
    if log1p:
        sc.pp.log1p(adata)


def _ctrl_means(ctrl_path, var_names, dataset_key, target_sum, log1p):
    """Per-dataset CONTROL mean in exactly the normalisation applied to the query cells.

    CONTROL-CENTRED DELTAS put this baseline on the same footing as ExPert (encoder
    center_mode='ctrl') and the benchmark's delta methods (`centroid`, `neg_target`, cellot's
    own-control-centred delta): each cell is expressed relative to its OWN dataset's controls, so
    a held-out dataset's baseline shift is removed without using any test label.
    """
    log.info(f"Loading control cells from {ctrl_path}")
    c = sc.read_h5ad(ctrl_path)
    _normalise(c, target_sum, log1p)
    common = np.intersect1d(np.asarray(var_names).astype(str), np.asarray(c.var_names).astype(str))
    qpos = {g: i for i, g in enumerate(np.asarray(var_names).astype(str))}
    cpos = {g: i for i, g in enumerate(np.asarray(c.var_names).astype(str))}
    qi = np.array([qpos[g] for g in common]); ci = np.array([cpos[g] for g in common])
    log.info(f"  {c.n_obs:,} control cells; {len(common)}/{len(var_names)} query genes present in the control DB")
    ds = c.obs[dataset_key].astype(str).values
    out = {}
    for d in np.unique(ds):
        m = ds == d
        Xm = c.X[m][:, ci]
        mu = np.asarray(Xm.mean(0)).ravel().astype(np.float32)
        v = np.zeros(len(var_names), np.float32); v[qi] = np.nan_to_num(mu)
        out[d] = v
    return out


def load_and_prepare(
        adata_path,
        labels_key="perturbation",
        split_dir=None,
        target_sum=1e6,
        log1p=False,
        scale=False,
        ctrl_delta=None,
        dataset_key="dataset",
        max_train_cells=0,
        seed=0,
    ):
    """Load adata and extract splits."""
    log.info(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    _normalise(adata, target_sum, log1p)
    if scale:
        log.info("Applying scale transformation")
        sc.pp.scale(adata)

    # ZERO-COUNT CELLS BECOME NaN UNDER CPM (0/0), and the two baselines then see DIFFERENT data:
    # XGBoost treats NaN as a missing value and learns a default split direction, while
    # SGDClassifier raises outright. Zero-fill so both are scored on identical inputs -- a cell
    # with no counts carries no information either way, and dropping it instead would change the
    # eval composition between arms.
    _X = adata.X
    _dense = not sp.issparse(_X)
    _vals = _X if _dense else _X.data
    _bad = ~np.isfinite(_vals)
    _n = int(_bad.sum())
    if _n:
        log.warning(f"{_n:,} non-finite values after normalisation (zero-count cells under CPM) "
                    f"-> zero-filled, so XGBoost and SGD see identical inputs")
        if _dense:
            adata.X = np.nan_to_num(_X, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            _X.data = np.nan_to_num(_vals, nan=0.0, posinf=0.0, neginf=0.0)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32, copy=False)

    log.info(f"Loading split indices from {split_dir}")
    train_idx, val_idx, test_idx, ood_test_idx = _load_split_indices(split_dir)
    if max_train_cells and len(train_idx) > max_train_cells:
        rng = np.random.default_rng(seed)
        train_idx = np.sort(rng.choice(train_idx, int(max_train_cells), replace=False))
        log.warning(f"Subsampled id_train to {len(train_idx):,} cells (--max_train_cells)")

    labels = adata.obs[labels_key].astype(str).values
    # ENCODE ON TRAIN LABELS. XGBClassifier requires contiguous class ids 0..C-1 present in y;
    # a class seen only in an eval split can't be predicted by either model anyway.
    le = LabelEncoder().fit(labels[train_idx])
    known = set(le.classes_)
    for nm, ix in (("id_val", val_idx), ("id_test", test_idx), ("ood_test", ood_test_idx)):
        unseen = sorted(set(labels[ix]) - known)
        if unseen:
            log.warning(f"{nm}: {len(unseen)} classes absent from id_train (cannot be predicted): {unseen[:8]}")
    c2i = {c: i for i, c in enumerate(le.classes_)}
    y = np.fromiter((c2i.get(l, -1) for l in labels), dtype=np.int64, count=len(labels))

    log.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, ID-Test: {len(test_idx)}, OOD-Test: {len(ood_test_idx)}")
    log.info(f"Classes (from id_train): {len(le.classes_)}")

    ctrl_means = None
    if ctrl_delta:
        if scale:
            raise ValueError("--ctrl_delta with --scale: controls would need the query scaler; unsupported")
        ctrl_means = _ctrl_means(ctrl_delta, adata.var_names, dataset_key, target_sum, log1p)
        _ds = set(adata.obs[dataset_key].astype(str).unique())
        _miss = sorted(_ds - set(ctrl_means))
        if _miss:
            log.warning(f"  datasets with NO control cells (left uncentred): {_miss}")

    return {
        "ctrl_means": ctrl_means,
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "ood_test_idx": ood_test_idx,
        "le": le,
        "adata": adata,
    }


def mask_inactive_genes(X, fill_value=np.nan):
    """Set genes that are 0 across all cells to fill_value."""
    inactive = (X == 0).all(axis=0)
    X = X.copy()
    X[:, inactive] = fill_value
    return X


def _predict_scores(model, X):
    """Per-class scores for top-k ranking.

    `decision_function` is preferred over `predict_proba` where both exist: OvR probability
    normalisation divides by a sum that underflows to 0 on extreme rows and returns NaN, which
    silently poisons the ranking (hit on a previous baseline run).
    """
    for attr in ("decision_function", "predict_proba"):
        fn = getattr(model, attr, None)
        if fn is None:
            continue
        try:
            s = np.asarray(fn(X))
            return s if s.ndim == 2 else None
        except Exception:                                          # noqa: BLE001
            continue
    return None


def _write_logits_h5(path, scores, class_names, split_idx):
    """Benchmark schema: /logits (cells x classes), /idx (0-based source rows, split-file order),
    /columns (class names). /perts mirrors /columns for tools that look for it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if scores.shape[0] != len(split_idx):
        raise ValueError(f"{path}: {scores.shape[0]} score rows vs {len(split_idx)} split indices")
    if scores.shape[1] != len(class_names):
        raise ValueError(f"{path}: {len(class_names)} class names for {scores.shape[1]} columns")
    s = np.nan_to_num(scores.astype(np.float32), nan=-np.inf)
    str_dt = h5py.string_dtype()
    names = np.asarray([str(c) for c in class_names], dtype=object)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("logits", data=s, compression="gzip", compression_opts=4)
        fh.create_dataset("idx", data=np.asarray(split_idx, dtype=np.int64))
        fh.create_dataset("columns", data=names, dtype=str_dt)
        fh.create_dataset("perts", data=names, dtype=str_dt)
    log.info(f"wrote {path} {s.shape}")


class _RoundTimer:
    """XGBoost callback: log elapsed wall time every N rounds so a probe can be extrapolated."""
    def __init__(self, every=10):
        import xgboost as xgb
        self._base = xgb.callback.TrainingCallback
        self.every = every; self.t0 = None

    def make(self):
        outer = self

        class CB(outer._base):
            def before_training(self, model):
                outer.t0 = time.time(); return model

            def after_iteration(self, model, epoch, evals_log):
                if (epoch + 1) % outer.every == 0:
                    el = time.time() - outer.t0
                    log.info(f"[xgb] round {epoch + 1}: {el:.0f}s elapsed ({el / (epoch + 1):.1f}s/round)")
                return False
        return CB()


def train_and_evaluate(
    input_dict,
    model_name,
    out_dir,
    dataset_key="dataset",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=20,
    n_jobs=-1,
    colsample_bytree=1.0,
    subsample=1.0,
    max_bin=256,
):
    """Train XGBoost (or a linear SGD baseline) and evaluate on val/test/ood_test."""
    linear = model_name == "linreg"
    adata = input_dict["adata"]
    X = input_dict["X"]
    y = input_dict["y"]
    le = input_dict["le"]
    train_idx = input_dict["train_idx"]
    val_idx = input_dict["val_idx"]
    test_idx = input_dict["test_idx"]
    ood_test_idx = input_dict["ood_test_idx"]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_ood_test, y_ood_test = X[ood_test_idx], y[ood_test_idx]
    input_dict.pop("X", None); del X          # ~8GB dense; the split copies are all that's needed now
    # ALL-ZERO GENES ARE MASKED so a model can tell "outside this split's panel" from "measured at
    # zero". XGBoost encodes that natively as NaN and learns a default split direction; a linear
    # model cannot represent it at all and sklearn raises on NaN. Each baseline therefore gets the
    # encoding it can actually use -- NOT a shared one, which would either cripple XGBoost's
    # missing-value handling or crash SGD. Stated because it means the two arms see slightly
    # different inputs by design.
    _fill = 0.0 if linear else np.nan
    log.info(f'Masking all-zero genes with {"0.0 (linear)" if linear else "NaN (XGBoost native)"}')
    X_train = mask_inactive_genes(X_train, _fill)
    X_val = mask_inactive_genes(X_val, _fill)
    X_test = mask_inactive_genes(X_test, _fill)
    X_ood_test = mask_inactive_genes(X_ood_test, _fill)
    _cm = input_dict.get("ctrl_means")
    if _cm is not None:
        # AFTER masking, so the panel mask is decided on raw expression (a delta is never exactly
        # zero). NaN - mu stays NaN for XGBoost; the linear arm re-zeroes its masked genes so
        # "off-panel" keeps meaning 0 rather than becoming -mu.
        _dsv = adata.obs[dataset_key].astype(str).values
        for _Xs, _ix in ((X_train, train_idx), (X_val, val_idx), (X_test, test_idx),
                         (X_ood_test, ood_test_idx)):
            if len(_ix) == 0:
                continue
            _off = (_Xs == 0).all(axis=0) if linear else None
            _d = _dsv[_ix]
            for _u in np.unique(_d):
                if _u in _cm:
                    _r = np.flatnonzero(_d == _u)
                    _Xs[_r] -= _cm[_u][None, :]
            if linear and _off is not None:
                _Xs[:, _off] = 0.0
        log.info(f"Control-centred deltas applied ({len(_cm)} dataset control means)")

    n_classes = len(le.classes_)
    t_fit = time.time()
    # LINEAR BASELINE. SGDClassifier, NOT LogisticRegression: lbfgs multinomial IGNORES n_jobs and
    # ran 2.5 h single-threaded without converging. OvR + log_loss is parallel, and its
    # `decision_function` (preferred by _predict_scores) avoids the OvR probability normalisation
    # that underflows to NaN on extreme rows. Only construction and fit branch -- the evaluation
    # below is shared, so both baselines are scored by identical code.
    if linear:
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, tol=1e-4,
                              n_jobs=n_jobs, random_state=0)
        log.info("Linear baseline: SGDClassifier(log_loss, OvR), no early stopping")
        model.fit(X_train, y_train)
    else:
        import xgboost as xgb
        log.info(f"Training XGBoost with {n_jobs} jobs...")
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            max_bin=max_bin,
            n_jobs=n_jobs,
            eval_metric="mlogloss",
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[_RoundTimer(every=10).make()],
            verbosity=1,
        )
        _vs = y_val >= 0
        model.fit(X_train, y_train, eval_set=[(X_val[_vs], y_val[_vs])], verbose=50)
        log.info(f"Best iteration: {model.best_iteration}")
    log.info(f"fit done in {time.time() - t_fit:.0f}s")

    # model.classes_ are LabelEncoder integers -> decode to perturbation names for /columns
    class_names = le.inverse_transform(np.asarray(model.classes_, dtype=np.int64))

    results = {}
    for split_name, X_split, y_split, split_idx in [
        ("id_val", X_val, y_val, val_idx),
        ("id_test", X_test, y_test, test_idx),
        ("ood_test", X_ood_test, y_ood_test, ood_test_idx),
    ]:
        if len(split_idx) == 0:
            continue
        proba = _predict_scores(model, X_split)
        if proba is None:
            raise RuntimeError(f"{model_name}: no per-class scores for {split_name}")
        if split_name in ("id_test", "ood_test"):
            _write_logits_h5(os.path.join(out_dir, split_name, "logits.h5"),
                             proba, class_names, split_idx)

        # quick on-the-fly summary (collect_results.py is the authoritative scorer)
        y_pred = model.predict(X_split)
        seen = y_split >= 0
        metrics = {
            "n": int(len(split_idx)),
            "accuracy": accuracy_score(y_split[seen], y_pred[seen]),
            "f1_macro": f1_score(y_split[seen], y_pred[seen], average="macro"),
        }
        order = np.argsort(-np.nan_to_num(proba, nan=-np.inf), axis=1)
        for k in (1, 5, 10):
            kk = min(k, order.shape[1])
            hit = (order[seen, :kk] == y_split[seen][:, None]).any(1)
            metrics[f"recall@{k}"] = float(hit.mean())
        log.info(f"--- {split_name}: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in metrics.items()))
        results[split_name] = metrics

    pd.DataFrame(results).T.to_csv(os.path.join(out_dir, "results.csv"))
    return model, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["xgboost", "linreg"])
    p.add_argument("--split_dir", required=True,
                   help="fold dir with id_train/id_val/id_test/ood_test .txt.gz; source adata is "
                        "resolved from <split_dir>/../metadata.json")
    p.add_argument("--adata", default=None, help="override the metadata.json source_path")
    p.add_argument("--out_dir", required=True,
                   help="<results>/<method>/perturb-seq/<size>/<fold>; writes <split>/logits.h5")
    p.add_argument("--labels_key", default="perturbation")
    p.add_argument("--dataset_key", default="dataset")
    p.add_argument("--target_sum", type=float, default=1e6,
                   help="library-size normalisation target; 0 disables")
    p.add_argument("--cpm", action="store_true", help="shorthand for --target_sum 1e6")
    p.add_argument("--log1p", action="store_true")
    p.add_argument("--scale", action="store_true")
    p.add_argument("--ctrl_delta", default=None,
                   help="control h5ad; subtract each dataset's control mean (same normalisation)")
    p.add_argument("--max_train_cells", type=int, default=0)
    # SPEED KNOBS (xgboost only). Multiclass trains n_classes trees PER ROUND, so
    # colsample_bytree is the dominant lever.
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--early_stopping_rounds", type=int, default=20)
    p.add_argument("--colsample_bytree", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--max_bin", type=int, default=256)
    p.add_argument("--n_jobs", type=int, default=-1)
    a = p.parse_args()

    target_sum = 1e6 if a.cpm else a.target_sum
    adata_path = a.adata or _adata_from_split_dir(a.split_dir)
    log.info(f"model={a.model} adata={adata_path} target_sum={target_sum} log1p={a.log1p} "
             f"ctrl_delta={bool(a.ctrl_delta)} out={a.out_dir}")
    os.makedirs(a.out_dir, exist_ok=True)

    d = load_and_prepare(
        adata_path, a.labels_key, split_dir=a.split_dir,
        target_sum=target_sum, log1p=a.log1p, scale=a.scale,
        ctrl_delta=a.ctrl_delta, dataset_key=a.dataset_key,
        max_train_cells=a.max_train_cells,
    )
    train_and_evaluate(
        d, a.model, a.out_dir,
        dataset_key=a.dataset_key,
        n_estimators=a.n_estimators, max_depth=a.max_depth, learning_rate=a.lr,
        early_stopping_rounds=a.early_stopping_rounds, n_jobs=a.n_jobs,
        colsample_bytree=a.colsample_bytree, subsample=a.subsample, max_bin=a.max_bin,
    )


if __name__ == "__main__":
    main()
