"""Loader / tidy-frame utilities for the ExPert perturbation-classification benchmarks.

All data lives under ``small/`` (sibling of this ``analysis/`` dir). Three efficiency
levels are stored in per-threshold dirs (``none``, ``e:1``, ``e:2``, ``e:3``); top-level
files hold the efficiency sweep and the train-side covariates.

Primary metric for ranking is ``recall@5`` (per project decision). recall@1 == accuracy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
HERE = Path(__file__).resolve().parent
# Env-overridable so `collect` points the loader at the freshly-collected metrics
# (EXPERT_METRICS_DIR=metrics/<size>) and the plot dir (EXPERT_ANALYSIS_OUT).
# Defaults keep the standalone results/<size>/{analysis,small} layout unchanged.
SMALL = Path(os.environ.get("EXPERT_METRICS_DIR") or (HERE.parent / "small"))
OUT = Path(os.environ.get("EXPERT_ANALYSIS_OUT") or (HERE / "out"))
OUT.mkdir(parents=True, exist_ok=True)

# efficiency level name -> numeric min_efficiency (none == 0 == no threshold)
LEVELS = {"none": 0, "e:1": 1, "e:2": 2, "e:3": 3}

# PREFERRED ORDER ONLY -- never a filter. Anything scored but absent here still gets
# plotted, appended alphabetically by order_methods(). Using this list as a whitelist
# silently dropped every method added after it was written (scgpt, xgboost, linreg,
# baseline_centroid) from the radar.
METHODS = [
    "ExPert",
    "baseline_neg_target", "baseline_prior", "baseline_random", "baseline_centroid_notarget",
    "cellot", "fingerprint_perdataset", "fingerprint_merged", "gears", "genepert", "scouter",
    "scgpt", "xgboost_capped", "xgboost", "linreg",
]

# Scored and kept in collect's tables, but dropped from EVERY analysis figure (and the
# CSVs the figures are drawn from): baseline_centroid is replaced by
# baseline_centroid_notarget, shown as "baseline (centroid)". Mirror of
# collect_results.PLOT_HIDE -- keep the two in step.
PLOT_HIDE = {"baseline_centroid"}


def _visible(df: pd.DataFrame) -> pd.DataFrame:
    return _scaled_rank(df[~df["method"].astype(str).isin(PLOT_HIDE)] if "method" in df.columns else df)


_N_CLASSES = None


def n_classes() -> int:
    """Size of the scored class space (one row per trained class in collect's
    train_class_support.csv -- the selftest checks this equals the scored classes)."""
    global _N_CLASSES
    if _N_CLASSES is None:
        _N_CLASSES = len(pd.read_csv(SMALL / "train_class_support.csv"))
    return _N_CLASSES


def _scaled_rank(df: pd.DataFrame) -> pd.DataFrame:
    """median_rank_scaled = 1 - (median_rank - 1)/(C - 1): 1 at rank 1, 0.5 at chance.
    Newer collect runs score it directly; derived here for tables written before that."""
    for raw, scaled in (("median_rank", "median_rank_scaled"),
                        ("weighted_median_rank", "weighted_median_rank_scaled")):
        if raw in df.columns and scaled not in df.columns:
            df = df.assign(**{scaled: 1.0 - (df[raw] - 1.0) / (n_classes() - 1)})
    return df


def order_methods(index) -> list:
    """Every method in `index`, known ones first in METHODS order, rest appended.
    Use this instead of filtering by METHODS so new methods can never be dropped."""
    seen = list(dict.fromkeys(str(m) for m in index))
    known = [m for m in METHODS if m in seen]
    return known + sorted(m for m in seen if m not in set(METHODS))


def is_baseline(m) -> bool:
    return str(m).startswith("baseline")


# derived, so a new baseline_<x> is picked up automatically
BASELINES = [m for m in METHODS if is_baseline(m)]
def _resolve_primary() -> str:
    """Primary metric = the recall flavour `collect` reported (EXPERT_RECALL_MODE),
    falling back to cell-pooled recall@5 when the balanced column is absent -- which
    is the case for any metrics dir collected before balanced_recall@k existed."""
    want = ("balanced_recall@5"
            if os.environ.get("EXPERT_RECALL_MODE", "balanced") == "balanced"
            else "recall@5")
    for csv in (SMALL / "none" / "all_metrics.csv", SMALL / "none" / "all_metrics_per_fold.csv"):
        try:
            cols = pd.read_csv(csv, nrows=0).columns
        except Exception:
            continue
        if want in cols:
            return want
        print(f"[load] {want} not in {csv.name} — falling back to recall@5")
        return "recall@5"
    return want


PRIMARY = _resolve_primary()
# Per-class tables carry the same value under both names (see load_threshold); this
# alias exists so a reader can tell which granularity a call site is working at.
PRIMARY_PERCLASS = PRIMARY
METRICS = ["accuracy", "recall@1", "recall@5", "recall@10", "macro_f1", "macro_f1_no_ds_bias",
           "balanced_recall@1", "balanced_recall@5", "balanced_recall@10"]


# ---------------------------------------------------------------- split parsing
def split_parts(split: pd.Series) -> pd.DataFrame:
    """``id_test:Marson2025_D1_Rest`` -> regime=id, dataset=Marson2025_D1_Rest."""
    regime = split.str.split(":").str[0].str.replace("_test", "", regex=False)
    dataset = split.str.split(":", n=1).str[1]
    return pd.DataFrame({"regime": regime, "dataset": dataset}, index=split.index)


def dataset_meta(dataset: pd.Series) -> pd.DataFrame:
    """Parse condition metadata out of dataset names (Marson donor/stim, cell line)."""
    d = dataset.fillna("")
    study = d.str.split("_").str[0].str.replace(r"\d+$", "", regex=True)
    donor = d.str.extract(r"_(D\d)")[0]
    stim = np.where(d.str.contains("Stim48"), "Stim48hr",
            np.where(d.str.contains("Stim8"), "Stim8hr",
            np.where(d.str.contains("Rest"), "Rest", "NA")))
    return pd.DataFrame({"study": study.values, "donor": donor.values, "stim": stim},
                        index=dataset.index)


def add_split_cols(df: pd.DataFrame) -> pd.DataFrame:
    derived = ["regime", "dataset", "study", "donor", "stim"]
    if "split" in df.columns:
        df = df.drop(columns=[c for c in derived if c in df.columns])
        df = pd.concat([df, split_parts(df["split"])], axis=1)
        df = pd.concat([df, dataset_meta(df["dataset"])], axis=1)
    elif "dataset" in df.columns:  # perdataset_class: dataset but no split
        df = df.drop(columns=[c for c in ["study", "donor", "stim"] if c in df.columns])
        df = pd.concat([df, dataset_meta(df["dataset"])], axis=1)
    return df


# ----------------------------------------------------------------- per-threshold
def _level_dir(level: str) -> Path:
    return SMALL / level


def load_threshold(level: str) -> dict[str, pd.DataFrame]:
    """Load the six per-dir frames for one efficiency level; tag with min_efficiency."""
    d = _level_dir(level)
    me = LEVELS[level]
    frames = {
        "metrics": pd.read_csv(d / "all_metrics.csv"),
        "metrics_per_fold": pd.read_csv(d / "all_metrics_per_fold.csv"),
        "metrics_group": pd.read_csv(d / "all_metrics_group.csv"),
        "perdataset": pd.read_csv(d / "all_perdataset_metrics.csv"),
        "perclass": pd.read_parquet(d / "all_perclass_metrics.parquet"),
        "perdataset_class": pd.read_parquet(d / "all_perdataset_class_metrics.parquet"),
    }
    for name, f in frames.items():
        f["min_efficiency"] = me
        f["level"] = level
        # PER-CLASS rows are one class each, so "balanced recall" (the mean of the
        # per-class recalls) and plain recall are the SAME number there -- balancing
        # is what averaging these rows DOES. collect only emits the balanced_* columns
        # on the aggregate tables, so alias them here; otherwise PRIMARY resolves to
        # balanced_recall@k and every per-class consumer (axis2, axis3, synthesis)
        # dies with a KeyError and its plots silently go missing.
        if name in ("perclass", "perdataset_class"):
            for c in [c for c in f.columns if c.startswith("recall@")]:
                if f"balanced_{c}" not in f.columns:
                    f[f"balanced_{c}"] = f[c]
        frames[name] = add_split_cols(_visible(f))
    return frames


def load_all_thresholds() -> dict[str, pd.DataFrame]:
    """Concatenate each frame type across all four levels into long frames."""
    per_level = {lv: load_threshold(lv) for lv in LEVELS}
    keys = next(iter(per_level.values())).keys()
    return {k: pd.concat([per_level[lv][k] for lv in LEVELS], ignore_index=True) for k in keys}


# --------------------------------------------------------------------- top-level
def load_sweep(per_fold: bool = False) -> pd.DataFrame:
    f = "sweep_metrics_per_fold.csv" if per_fold else "sweep_metrics.csv"
    df = pd.read_csv(SMALL / f)
    df["min_efficiency"] = df["min_efficiency"].fillna(0)  # blank == no threshold
    return add_split_cols(_visible(df))


def load_cells_per_perturbation() -> pd.DataFrame:
    df = pd.read_parquet(SMALL / "sweep_cells_per_perturbation.parquet")
    df["min_efficiency"] = df["min_efficiency"].fillna(0)
    return add_split_cols(_visible(df))


def load_weighted() -> pd.DataFrame:
    return add_split_cols(_visible(pd.read_csv(SMALL / "weighted_metrics.csv")))


def load_train_covariates() -> pd.DataFrame:
    """One row per perturbation: train support + class stats covariates."""
    stats = pd.read_csv(SMALL / "train_class_stats.csv").rename(columns={"class": "perturbation"})
    support = pd.read_csv(SMALL / "train_class_support.csv")
    cov = stats.merge(support, on="perturbation", how="outer")
    cov["log_support"] = np.log10(cov["train_support"].clip(lower=1))
    return cov


def load_train_stats() -> pd.DataFrame:
    df = pd.read_csv(SMALL / "train_stats.csv")
    df["cells_per_pert"] = df["n_cells"] / df["n_perturbations"]
    return df


_CELL_META = None


def load_cell_meta() -> pd.DataFrame:
    """Per-cell metadata (dataset, efficiency, perturbation) from small/data/source.obs.csv.

    Values are stored as byte-string reprs (b'...') — stripped here. Cached per process.
    """
    global _CELL_META
    if _CELL_META is None:
        df = pd.read_csv(SMALL / "data" / "source.obs.csv", index_col=0)
        for c in ("dataset", "perturbation"):
            df[c] = df[c].str.replace(r"^b'(.*)'$", r"\1", regex=True)
        df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")
        _CELL_META = df
    return _CELL_META


def dataset_efficiency_stats() -> pd.DataFrame:
    """Per-dataset efficiency distribution summary + cells/perturbation."""
    m = load_cell_meta()
    g = m.groupby("dataset").agg(
        n_cells=("efficiency", "size"),
        n_perts=("perturbation", "nunique"),
        eff_med=("efficiency", "median"),
        eff_mean=("efficiency", "mean"),
        frac_ge1=("efficiency", lambda x: (x >= 1).mean()),
        frac_ge2=("efficiency", lambda x: (x >= 2).mean()),
        frac_ge3=("efficiency", lambda x: (x >= 3).mean()),
    ).reset_index()
    g["cells_per_pert"] = g["n_cells"] / g["n_perts"]
    return g


# -------------------------------------------------------------- derived helpers
def fold_aggregate(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Mean + std across folds for the given metrics."""
    g = df.groupby(group_cols, dropna=False)[metric_cols]
    out = g.mean().add_suffix("_mean")
    out = out.join(g.std().add_suffix("_std"))
    return out.reset_index()


def chance_rate(perdataset: pd.DataFrame) -> pd.DataFrame:
    """Per (split,dataset) chance = 1 / n_classes_present (from any one method row)."""
    c = perdataset.drop_duplicates(["split", "dataset"])[["split", "dataset", "n_classes_present"]].copy()
    c["chance"] = 1.0 / c["n_classes_present"]
    return c


def add_lift(df: pd.DataFrame, metric: str, baseline: str = "baseline_prior",
             key: list[str] | None = None) -> pd.DataFrame:
    """Add ``<metric>_lift`` = method metric minus the baseline method's metric on same key.

    ``key`` identifies a comparison cell (default: split). The baseline rows are matched
    and subtracted; baseline rows themselves get lift relative to themselves (~0).
    """
    key = key or ["split"]
    base = (df[df["method"] == baseline]
            .groupby(key, dropna=False)[metric].mean().rename(f"{metric}__base"))
    out = df.merge(base, on=key, how="left")
    out[f"{metric}_lift"] = out[metric] - out[f"{metric}__base"]
    return out


# ------------------------------------------------------------------- self-test
def selftest() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")

    print("Loading all thresholds...")
    allt = load_all_thresholds()
    pc = allt["perclass"]
    # SHAPE-DERIVED, NOT HARD-CODED. These were pinned to the run that existed when the
    # selftest was written (9 methods, 6 folds, 21 datasets) and went red the moment
    # methods and folds were added -- a permanently-failing selftest hides real
    # regressions, which is the opposite of the point.
    n_meth = pc["method"].nunique()
    check("perclass methods >= 2", n_meth >= 2, f"got {n_meth}")
    # class count is the size's label space (200 small, 1,157 ctx7, ~9.9k zs_f4) -- was
    # pinned to small's ~200 and failed on every other size.
    n_cls = pc["class"].nunique()
    check("perclass classes >= 10", n_cls >= 10, f"got {n_cls}")
    # compare fold NUMBERS: sorted() on the names is lexical (fold_10 before fold_2)
    fnums = sorted(int(str(f).split("_")[1]) for f in pc["fold"].unique())
    check("folds are a contiguous fold_0..N", fnums == list(range(len(fnums))),
          f"got {len(fnums)}: {fnums}")
    check("efficiency levels none/1/2/3", sorted(pc["min_efficiency"].unique()) == [0, 1, 2, 3])
    check("regimes id+ood", set(pc["regime"].unique()) == {"id", "ood"})
    check("datasets >= 10", pc["dataset"].nunique() >= 10, f"got {pc['dataset'].nunique()}")
    check("recall@5 >= recall@1 always", bool((pc["recall@5"] >= pc["recall@1"] - 1e-9).all()))

    # Per-method coverage must be REPORTED, not silently dropped. Under-coverage is not
    # itself a failure -- ExPert legitimately has fewer folds (only some benchmark folds
    # have a model whose training excluded their OOD datasets) -- but a method with ZERO
    # coverage means it silently vanished from the table, which is.
    m = allt["metrics"]
    cov = m.groupby("method")["split"].nunique()
    for meth, c in cov.sort_values().items():
        if c < cov.max():
            print(f"  [INFO] {meth}: {c}/{cov.max()} method-splits (under-covered)")
    check("every method has non-zero coverage", bool((cov > 0).all()),
          f"zero-coverage: {[k for k, v in cov.items() if v == 0]}")
    check("all scored methods are plottable", set(order_methods(cov.index)) == set(cov.index))

    sweep = load_sweep()
    check("sweep min_efficiency 0/1/2/3", sorted(sweep["min_efficiency"].unique()) == [0, 1, 2, 3])
    cov = load_train_covariates()
    # one covariate row per scored class (+/- control and a few train-only perts)
    check("train covariates ~= scored classes", abs(len(cov) - n_cls) <= 5,
          f"got {len(cov)} vs {n_cls} scored classes")
    check("covariate cols present",
          {"train_support", "dataset_entropy", "cells_per_dataset", "n_train_datasets"}
          .issubset(cov.columns))
    ts = load_train_stats()
    check("train_stats >= 10 datasets", len(ts) >= 10, f"got {len(ts)}")

    print("\nSELFTEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    print("Loaders available:", [n for n in dir() if n.startswith("load")])
