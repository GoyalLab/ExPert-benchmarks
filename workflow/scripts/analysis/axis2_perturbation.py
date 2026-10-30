"""Axis 2 — per-perturbation retrieval.

Which perturbations are reliably recovered vs hard, what drives that, which look
confusable (in-neighborhood but not top-1), and which fail to generalize (ID->OOD).
Per-class metrics are support-weighted when aggregating across splits/folds, since the
same perturbation appears in many datasets with very different cell counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load
import common as C

PRIMARY = load.PRIMARY  # recall@5
# PRIMARY must be in METRICS: per_class_summary() emits one <metric>_w column per
# entry and downstream reads PRIMARY_w back out.
METRICS = list(dict.fromkeys([PRIMARY, "recall@5", "recall@1", "f1"]))


def _wmean(df, val, w="support"):
    sw = df[w].sum()
    return np.average(df[val], weights=df[w]) if sw > 0 else np.nan


def per_class_summary(perclass: pd.DataFrame, regime: str | None = None) -> pd.DataFrame:
    """Support-weighted per-class metrics aggregated over folds & splits (optionally one regime)."""
    pc = perclass[perclass["method"] == "ExPert"].copy()
    if regime:
        pc = pc[pc["regime"] == regime]
    pc = pc[pc["support"] > 0]
    rows = []
    for cls, g in pc.groupby("class"):
        row = {"class": cls, "support_total": int(g["support"].sum()),
               "n_obs": len(g), "n_datasets": g["dataset"].nunique()}
        for m in METRICS:
            row[f"{m}_w"] = _wmean(g, m)
        # unweighted dispersion across datasets (mean per dataset first)
        per_ds = g.groupby("dataset").apply(lambda x: _wmean(x, PRIMARY), include_groups=False)
        row["recall@5_ds_std"] = per_ds.std()
        row["recall@5_ds_min"] = per_ds.min()
        row["recall@5_ds_max"] = per_ds.max()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["confusable_gap"] = out["recall@5_w"] - out["recall@1_w"]  # in-neighborhood, not top-1
    return out


def add_drivers(summary: pd.DataFrame) -> pd.DataFrame:
    cov = load.load_train_covariates()
    return summary.merge(cov, left_on="class", right_on="perturbation", how="left")


def driver_analysis(df: pd.DataFrame):
    """Drivers of per-class recall@5.

    Reports: univariate Spearman per feature; a cross-validated *linear* (RidgeCV on
    standardised features) fit with its standardised coefficients (honest, interpretable
    importance); and a GBT cross-val R2 as a nonlinearity check. With ~200 correlated
    covariates the linear model is the trustworthy one — a low CV R2 is itself a finding
    (training volume only partly explains which perturbations are recoverable).
    """
    feats = ["log_support", "n_train_cells", "n_train_datasets",
             "dataset_entropy", "cells_per_dataset"]
    d = df.dropna(subset=feats + [f"{PRIMARY}_w"]).copy()
    sp = []
    for f in feats:
        rho, pv = spearmanr(d[f], d[f"{PRIMARY}_w"])
        sp.append({"feature": f, "spearman_rho": rho, "p_value": pv})
    sp = pd.DataFrame(sp)

    X, y = d[feats].values, d[f"{PRIMARY}_w"].values
    # linear (standardised) — cross-validated R2 + standardised coefficients
    ridge = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-3, 3, 25)))
    lin_r2 = r2_score(y, cross_val_predict(ridge, X, y, cv=5))
    ridge.fit(X, y)
    insample_r2 = r2_score(y, ridge.predict(X))
    coefs = ridge[-1].coef_
    # GBT nonlinearity check
    gbt = HistGradientBoostingRegressor(max_depth=2, max_iter=200, learning_rate=0.05,
                                        min_samples_leaf=20, random_state=0)
    gbt_r2 = r2_score(y, cross_val_predict(gbt, X, y, cv=5))

    sp["ridge_std_coef"] = coefs
    sp = sp.reindex(sp["ridge_std_coef"].abs().sort_values(ascending=False).index)
    return sp, {"linear_insample_r2": insample_r2, "linear_cv_r2": lin_r2,
                "gbt_cv_r2": gbt_r2}, len(d)


def ood_drop(perclass: pd.DataFrame) -> pd.DataFrame:
    """Per-class ID vs OOD recall@5 (support-weighted) and the drop."""
    ids = per_class_summary(perclass, "id")[["class", f"{PRIMARY}_w", "support_total"]]
    oods = per_class_summary(perclass, "ood")[["class", f"{PRIMARY}_w"]]
    m = ids.merge(oods, on="class", suffixes=("_id", "_ood"))
    m["ood_drop"] = m[f"{PRIMARY}_w_id"] - m[f"{PRIMARY}_w_ood"]
    return m.sort_values("ood_drop", ascending=False)


def crossmethod_perclass(perclass: pd.DataFrame, regime: str = "id",
                         metric: str = PRIMARY) -> pd.DataFrame:
    """Per class: ExPert `metric` vs mean/best-of-competitors, to find ExPert-unique wins.

    Per-method values are support-weighted across folds/splits. Derived columns are
    metric-agnostic (`best_competitor_score`, `expert_minus_best`).
    """
    pc = perclass[perclass["regime"] == regime]
    pc = pc[pc["support"] > 0]
    agg = (pc.groupby(["class", "method"])
             .apply(lambda g: _wmean(g, metric), include_groups=False)
             .rename(metric).reset_index())
    wide = agg.pivot(index="class", columns="method", values=metric)
    comp = [c for c in wide.columns if c not in ("ExPert",) and not c.startswith("baseline")]
    wide["competitor_mean"] = wide[comp].mean(axis=1)
    wide["best_competitor"] = wide[comp].idxmax(axis=1)        # which other method is best
    wide["best_competitor_score"] = wide[comp].max(axis=1)     # its value of `metric`
    wide["expert_minus_competitors"] = wide["ExPert"] - wide["competitor_mean"]
    wide["expert_minus_best"] = wide["ExPert"] - wide["best_competitor_score"]
    wide.attrs["metric"] = metric
    return wide.reset_index().sort_values("expert_minus_best", ascending=False)


# discrete colour per competitor method (only the real competitors, baselines excluded)
COMPETITORS = ["cellot", "fingerprint_perdataset", "fingerprint_merged", "gears", "genepert", "scouter"]
_CMAP = dict(zip(COMPETITORS, C.plt.cm.tab10.colors))


def perdataset_soft_competitor(metric_col: str, regime: str) -> pd.DataFrame:
    """Per dataset: ExPert vs best competitor on a true soft (efficiency-weighted) metric
    from weighted_metrics.csv (per method×split, so dataset-level — the finest grain at
    which soft metrics exist)."""
    w = load.load_weighted()
    w = w[w.regime == regime]
    wide = w.groupby(["dataset", "method"])[metric_col].mean().unstack("method")
    comp = [c for c in wide.columns if c != "ExPert" and not c.startswith("baseline")]
    out = pd.DataFrame({
        "dataset": wide.index,
        "ExPert": wide["ExPert"].values,
        "best_competitor": wide[comp].idxmax(axis=1).values,
        "best_competitor_score": wide[comp].max(axis=1).values,
    })
    out["expert_minus_best"] = out["ExPert"] - out["best_competitor_score"]
    return out.sort_values("expert_minus_best")


def plot_perdataset_soft_competitor(out: pd.DataFrame, metric_label: str, regime: str, fname: str):
    """Per-dataset ExPert vs best competitor on a soft metric; above diagonal = ExPert wins."""
    d = out.dropna(subset=["ExPert", "best_competitor_score"]).copy()
    fig, ax = C.plt.subplots(figsize=(7.0, 7.2))
    lim = max(d["ExPert"].max(), d["best_competitor_score"].max()) * 1.08
    ax.plot([0, lim], [0, lim], "--", color="0.6", lw=1, zorder=0)
    ax.fill_between([0, lim], [0, lim], lim, color="#2a9d8f", alpha=0.05, zorder=0)
    for meth in COMPETITORS:
        sub = d[d["best_competitor"] == meth]
        if not sub.empty:
            ax.scatter(sub["best_competitor_score"], sub["ExPert"], s=70, color=_CMAP[meth],
                       edgecolor="k", lw=0.4, label=f"{C.disp(meth)} (n={len(sub)})", zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["dataset"].replace("Marson2025_", "M_").replace("ReplogleWeissman", "RW")
                    .replace("XAtlas2025_", "X_"),
                    (r["best_competitor_score"], r["ExPert"]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(f"best of other methods — {metric_label} ({regime.upper()})", fontsize=14)
    ax.set_ylabel(f"ExPert — {metric_label} ({regime.upper()})", fontsize=14)
    ax.set_title(f"Per-dataset: ExPert vs best competitor — {metric_label}, {regime.upper()}\n"
                 "(above diagonal = ExPert wins; colour = best competitor)", fontsize=15)
    ax.legend(title="best competitor", fontsize=11, ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.10))
    C.save(fig, fname)


def plot_crossmethod_scatter(wide: pd.DataFrame, regime: str = "id",
                             metric_label: str = "recall@5",
                             fname: str = "axis2_expert_vs_best_competitor.svg"):
    """ExPert `metric` (y) vs best-of-other-methods (x), coloured by which method is best.

    Points above the diagonal = ExPert wins; below = a competitor beats ExPert. Far
    upper-left = perturbations effectively *only* ExPert recovers.
    """
    d = wide.dropna(subset=["ExPert", "best_competitor_score"]).copy()
    fig, ax = C.plt.subplots(figsize=(7.0, 7.2))
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, zorder=0)
    ax.fill_between([0, 1], [0, 1], 1, color="#2a9d8f", alpha=0.05, zorder=0)  # ExPert-wins region
    for meth in COMPETITORS:
        sub = d[d["best_competitor"] == meth]
        if sub.empty:
            continue
        ax.scatter(sub["best_competitor_score"], sub["ExPert"],
                   s=34, alpha=0.8, edgecolor="k", lw=0.3,
                   color=_CMAP[meth], label=f"{C.disp(meth)} (n={len(sub)})")
    # annotate the most ExPert-unique perturbations (big positive gap, low competitor score)
    uniq = d.sort_values("expert_minus_best", ascending=False).head(8)
    for _, r in uniq.iterrows():
        ax.annotate(r["class"], (r["best_competitor_score"], r["ExPert"]),
                    fontsize=9, xytext=(3, 3), textcoords="offset points")
    # annotate perturbations where a competitor beats ExPert (below diagonal)
    lose = d[d["expert_minus_best"] < 0].sort_values("expert_minus_best").head(6)
    for _, r in lose.iterrows():
        ax.annotate(r["class"], (r["best_competitor_score"], r["ExPert"]),
                    fontsize=9, color="#b00020", xytext=(3, -8), textcoords="offset points")
    lim = max(d["ExPert"].max(), d["best_competitor_score"].max()) * 1.05
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    rg = regime.upper()
    ax.tick_params(labelsize=12)
    ax.set_xlabel(f"best of other methods — {metric_label} ({rg})", fontsize=14)
    ax.set_ylabel(f"ExPert — {metric_label} ({rg})", fontsize=14)
    ax.set_title(f"Per-perturbation: ExPert vs best competitor — {metric_label}, {rg}\n"
                 "(above diagonal = ExPert wins; colour = best competitor)", fontsize=15)
    ax.legend(title="best competitor", fontsize=11, ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.10))
    C.save(fig, fname)


def plot_dataset_variance(summary: pd.DataFrame):
    d = summary.dropna(subset=["recall@5_ds_std"]).copy()
    fig, ax = C.plt.subplots(figsize=(7, 5))
    sc = ax.scatter(d[f"{PRIMARY}_w"], d["recall@5_ds_std"],
                    c=np.log10(d["support_total"].clip(lower=1)), cmap="cividis",
                    s=35, edgecolor="k", lw=0.3)
    ax.set_xlabel(f"mean {PRIMARY} (support-weighted)")
    ax.set_ylabel("std of recall@5 across datasets")
    ax.set_title("Per-perturbation: difficulty vs cross-dataset consistency")
    fig.colorbar(sc, ax=ax, label="log10 total support")
    C.save(fig, "axis2_perclass_dataset_variance.svg")


def plot_driver(sp: pd.DataFrame):
    fig, ax = C.plt.subplots(figsize=(6, 4))
    colors = [C.EXPERT_C if v >= 0 else C.FIELD_C for v in sp["ridge_std_coef"]]
    ax.barh(sp["feature"], sp["ridge_std_coef"], color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("standardised ridge coefficient")
    ax.set_title("Drivers of per-perturbation recall@5 (linear)")
    ax.invert_yaxis()
    C.save(fig, "axis2_perclass_drivers.svg")


def main():
    print("Axis 2 — per-perturbation retrieval")
    perclass = load.load_threshold("none")["perclass"]

    summ = add_drivers(per_class_summary(perclass, regime="id"))
    summ = summ.sort_values(f"{PRIMARY}_w", ascending=False)
    ranked = summ[["class", f"{PRIMARY}_w", "recall@1_w", "f1_w", "confusable_gap",
                   "n_datasets", "recall@5_ds_std", "support_total",
                   "train_support", "n_train_datasets", "dataset_entropy"]].round(4)
    C.dump(ranked, "axis2_perturbation_difficulty_ranked.csv")

    sp, r2, n = driver_analysis(summ)
    C.dump(sp.round(4), "axis2_perclass_drivers.csv")
    plot_driver(sp)
    plot_dataset_variance(summ)

    conf = summ.sort_values("confusable_gap", ascending=False)[
        ["class", "confusable_gap", "recall@1_w", "recall@5_w", "support_total"]].round(4)
    C.dump(conf, "axis2_confusable_candidates.csv")

    od = ood_drop(perclass).round(4)
    C.dump(od, "axis2_ood_drop_per_class.csv")

    # cross-method per-perturbation. True soft (efficiency-weighted) F1 is not available
    # per-class, so the efficiency-aware view uses F1 at the e:2 hard threshold as a proxy.
    perclass_e2 = load.load_threshold("e:2")["perclass"]
    # (metric_col, metric_label, slug, source_perclass)
    views = [
        ("recall@5", "recall@5", "recall5", perclass),
        ("recall@5", "recall@5 (efficiency≥2)", "recall5e2", perclass_e2),
        ("f1", "F1", "f1", perclass),
        ("f1", "F1 (efficiency≥2)", "f1e2", perclass_e2),
    ]
    for metric_col, metric_label, slug, src in views:
        for regime in ("id", "ood"):
            xm = crossmethod_perclass(src, regime=regime, metric=metric_col)
            fig_name = f"axis2_expert_vs_best_competitor_{slug}_{regime}.svg"
            csv_name = f"axis2_crossmethod_perclass_{slug}_{regime}.csv"
            plot_crossmethod_scatter(xm, regime=regime, metric_label=metric_label, fname=fig_name)
            C.dump(xm.round(4), csv_name)
            expert_only = xm[(xm["best_competitor_score"] < 0.1) & (xm["ExPert"] >= 0.3)]
            expert_loses = xm[xm["expert_minus_best"] < 0]
            print(f"\n[{metric_label} {regime.upper()}] ExPert beats best competitor on "
                  f"{(xm['expert_minus_best'] > 0).sum()}/{len(xm)}; "
                  f"loses on {len(expert_loses)} ({list(expert_loses['class'][:5])}); "
                  f"ExPert-only={len(expert_only)}; "
                  f"toughest: {', '.join(f'{m}={c}' for m, c in xm['best_competitor'].value_counts().head(2).items())}")

    print(f"\nR² predicting recall@5 from train covariates (n={n}): "
          f"linear in-sample={r2['linear_insample_r2']:.3f}, "
          f"linear CV={r2['linear_cv_r2']:.3f}, gbt CV={r2['gbt_cv_r2']:.3f}")
    print("Driver ranking (|standardised ridge coef|):")
    print(sp[["feature", "spearman_rho", "ridge_std_coef"]].to_string(index=False))
    print("\nEasiest 5:", list(ranked.head(5)["class"]))
    print("Hardest 5:", list(ranked.tail(5)["class"]))
    print("Top confusable-gap 8:", list(conf.head(8)["class"]))
    print("Biggest OOD drop 8:", list(od.head(8)["class"]))


if __name__ == "__main__":
    main()
