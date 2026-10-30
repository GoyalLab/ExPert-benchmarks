"""Axis 1 — dataset signal.

Which datasets carry good vs poor signal, and which of the poor ones are genuinely
low-signal (low ID) vs a failure to generalize (high ID, low OOD)? Primary metric
recall@5. All cross-dataset numbers are also expressed as *lift over baseline_prior*
because datasets have very different class counts (chance rates).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load
import common as C

PRIMARY = load.PRIMARY  # recall@5
# PRIMARY must be in SHOW: dataset_table() pivots SHOW into <metric>__id/__ood and
# then reads PRIMARY__id / PRIMARY__ood back out.
SHOW = list(dict.fromkeys([PRIMARY, "recall@5", "recall@1", "macro_f1_no_ds_bias"]))


def dataset_table() -> pd.DataFrame:
    """One row per (method, dataset) with id/ood metrics and lift over baseline_prior."""
    m = load.load_threshold("none")["metrics"]  # fold-aggregated, no-threshold
    m = load.add_lift(m, PRIMARY, baseline="baseline_prior", key=["split"])

    # pivot id vs ood per method/dataset
    keep = ["method", "dataset", "regime"] + SHOW + [f"{PRIMARY}_lift"]
    long = m[keep]
    wide = long.pivot_table(index=["method", "dataset"], columns="regime",
                            values=SHOW + [f"{PRIMARY}_lift"])
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    # number of classes per dataset (from id_test rows)
    nc = (load.load_threshold("none")["perdataset"]
          .query("regime == 'id'")[["dataset", "n_classes_present"]].drop_duplicates())
    wide = wide.merge(nc, on="dataset", how="left")
    wide["id_ood_gap"] = wide[f"{PRIMARY}__id"] - wide[f"{PRIMARY}__ood"]
    return wide


def classify_signal(df_expert: pd.DataFrame) -> pd.DataFrame:
    """Quadrant labels using median ID / OOD recall@5 as thresholds."""
    d = df_expert.copy()
    id_hi = d[f"{PRIMARY}__id"] >= d[f"{PRIMARY}__id"].median()
    ood_hi = d[f"{PRIMARY}__ood"] >= d[f"{PRIMARY}__ood"].median()
    d["signal_class"] = np.select(
        [id_hi & ood_hi, id_hi & ~ood_hi, ~id_hi & ood_hi, ~id_hi & ~ood_hi],
        ["strong+generalizes", "memorizes (ID>OOD)", "robust-but-low-ID", "low-signal"],
        default="?",
    )
    return d


def plot_id_vs_ood(d: pd.DataFrame):
    fig, ax = C.plt.subplots(figsize=(7, 6))
    lo = min(d[f"{PRIMARY}__id"].min(), d[f"{PRIMARY}__ood"].min())
    hi = max(d[f"{PRIMARY}__id"].max(), d[f"{PRIMARY}__ood"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1, label="id = ood")
    sc = ax.scatter(d[f"{PRIMARY}__id"], d[f"{PRIMARY}__ood"],
                    c=d["id_ood_gap"], cmap="viridis_r", s=70, edgecolor="k", lw=0.4)
    for _, r in d.iterrows():
        ax.annotate(r["dataset"].replace("Marson2025_", "M_").replace("ReplogleWeissman", "RW")
                    .replace("XAtlas2025_", "X_"),
                    (r[f"{PRIMARY}__id"], r[f"{PRIMARY}__ood"]),
                    fontsize=9, xytext=(3, 3), textcoords="offset points")
    ax.axvline(d[f"{PRIMARY}__id"].median(), color="0.8", lw=0.8)
    ax.axhline(d[f"{PRIMARY}__ood"].median(), color="0.8", lw=0.8)
    fig.colorbar(sc, ax=ax, label="ID − OOD gap")
    ax.set_xlabel(f"ID test {PRIMARY}")
    ax.set_ylabel(f"OOD test {PRIMARY}")
    ax.set_title("ExPert dataset signal: ID vs OOD (recall@5)")
    ax.legend(loc="upper left", fontsize=11)
    C.save(fig, "axis1_id_vs_ood_scatter.svg")


def correlate_properties(d_expert: pd.DataFrame) -> pd.DataFrame:
    """Spearman of ExPert ID recall@5 (and lift) vs dataset training properties."""
    ts = load.load_train_stats()
    merged = d_expert.merge(ts, on="dataset", how="left")
    rows = []
    targets = [f"{PRIMARY}__id", f"{PRIMARY}_lift__id", "id_ood_gap"]
    props = ["n_cells", "n_perturbations", "cells_per_pert"]
    sub = merged.dropna(subset=props)  # only datasets present in train_stats
    for t in targets:
        for p in props:
            rho, pv = spearmanr(sub[p], sub[t])
            rows.append({"target": t, "property": p, "spearman_rho": rho,
                         "p_value": pv, "n": len(sub)})
    return pd.DataFrame(rows)


def crossmethod_rank(table: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: ExPert recall@5 (id) vs best competitor, and ExPert's rank."""
    t = table[["method", "dataset", f"{PRIMARY}__id"]].dropna()
    rows = []
    for ds, g in t.groupby("dataset"):
        g = g.sort_values(f"{PRIMARY}__id", ascending=False).reset_index(drop=True)
        expert = g[g["method"] == "ExPert"]
        if expert.empty:
            continue
        e_val = expert[f"{PRIMARY}__id"].iloc[0]
        rank = int(g.index[g["method"] == "ExPert"][0]) + 1
        others = g[g["method"] != "ExPert"]
        best_other = others.iloc[0]
        rows.append({"dataset": ds, "expert_recall@5": e_val, "expert_rank": rank,
                     "n_methods": len(g), "best_competitor": best_other["method"],
                     "best_competitor_recall@5": best_other[f"{PRIMARY}__id"],
                     "expert_minus_best_competitor": e_val - best_other[f"{PRIMARY}__id"]})
    return pd.DataFrame(rows).sort_values("expert_minus_best_competitor")


def recall_trajectory(regime: str = "id") -> pd.DataFrame:
    """Per-dataset ExPert recall@5 at each threshold, RAW vs COMMON-SET, for one regime.

    Raw = support-weighted over all classes evaluable at that threshold. Common-set =
    only classes present at ALL four thresholds, so the efficiency filter cannot inflate
    recall@5 by dropping hard classes — it isolates the genuine within-class effect of
    high-efficiency cells classifying better. ``gain_from_class_drop`` = raw − common gain.
    """
    pc = load.load_all_thresholds()["perclass"]
    pc = pc[(pc.method == "ExPert") & (pc.regime == regime) & (pc.support > 0)]

    def wmean(g):
        return np.average(g["recall@5"], weights=g["support"]) if g["support"].sum() else np.nan

    rows = []
    for ds, g in pc.groupby("dataset"):
        nlev = g.groupby("class")["min_efficiency"].nunique()
        common = set(nlev[nlev == 4].index)
        rec = {"dataset": ds, "n_common_classes": len(common)}
        for me in (0, 1, 2, 3):
            sub = g[g.min_efficiency == me]
            rec[f"raw_r5_{me}"] = wmean(sub)
            rec[f"com_r5_{me}"] = wmean(sub[sub["class"].isin(common)])
            rec[f"n_classes_{me}"] = sub["class"].nunique()
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["n_classes_dropped"] = d["n_classes_0"] - d["n_classes_3"]
    d["raw_gain"] = d["raw_r5_3"] - d["raw_r5_0"]
    d["common_gain"] = d["com_r5_3"] - d["com_r5_0"]
    d["gain_from_class_drop"] = d["raw_gain"] - d["common_gain"]
    return d


def efficiency_decomposition(expert: pd.DataFrame) -> pd.DataFrame:
    """Merge dataset signal with per-cell efficiency stats + thresholding recoverability.

    Separates two low-signal failure modes:
      - 'low-eff (recoverable)': baseline low but high-efficiency cells classify well.
      - 'intrinsically low': even high-efficiency cells (e:3) stay near the floor.
    """
    eff = load.dataset_efficiency_stats()
    traj_id = recall_trajectory("id")
    traj_ood = recall_trajectory("ood").set_index("dataset").add_suffix("_ood").reset_index()
    d = (expert[["dataset", f"{PRIMARY}__id", f"{PRIMARY}__ood", "n_classes_present"]]
         .merge(eff, on="dataset", how="left")
         .merge(traj_id, on="dataset", how="left")
         .merge(traj_ood, on="dataset", how="left"))
    # classify on the ID COMMON-SET trajectory (cleanest place to judge intrinsic-ness;
    # OOD adds a generalization confound). The class is a dataset property reused for OOD.
    intrinsic = d["com_r5_3"] < 0.35             # high-eff cells STILL don't classify (ID)
    recoverable = (~intrinsic) & (d["com_r5_0"] < 0.55) & (d["common_gain"] >= 0.20)
    high = d["com_r5_0"] >= 0.6
    d["efficiency_class"] = np.select(
        [high, intrinsic, recoverable],
        ["high signal", "intrinsically low", "low-eff (recoverable)"],
        default="moderate")
    return d.sort_values("eff_med")


def plot_signal_vs_efficiency(d: pd.DataFrame, regime: str = "id",
                              fname: str = "axis1_signal_vs_efficiency.svg"):
    sig = f"{PRIMARY}__{regime}"
    rho, pv = spearmanr(d["eff_med"], d[sig])
    fig, ax = C.plt.subplots(figsize=(7.2, 5.5))
    cmap = {"high signal": "#2a9d8f", "low-eff (recoverable)": "#e9c46a",
            "intrinsically low": "#b00020", "moderate": "#6c757d"}
    for cls, sub in d.groupby("efficiency_class"):
        ax.scatter(sub["eff_med"], sub[sig], s=70, color=cmap[cls],
                   edgecolor="k", lw=0.4, label=cls, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["dataset"].replace("Marson2025_", "M_").replace("ReplogleWeissman", "RW")
                    .replace("XAtlas2025_", "X_"),
                    (r["eff_med"], r[sig]), fontsize=9,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("per-cell perturbation efficiency (dataset median)")
    ax.set_ylabel(f"ExPert {regime.upper()} {PRIMARY} (no threshold)")
    ax.set_title(f"Dataset signal vs efficiency — {regime.upper()} (Spearman ρ={rho:.2f})")
    ax.legend(fontsize=11, title="failure mode (ID-derived)", loc="lower right")
    C.save(fig, fname)


def plot_recoverability(d: pd.DataFrame, regime: str = "id",
                        fname: str = "axis1_efficiency_recoverability.svg"):
    """Common-set recall@5 trajectory none->e:3 per dataset (classes fixed across thresholds,
    so the rise can't be a class-dropping artifact); intrinsically-low ones highlighted.
    Failure-mode colours are ID-derived; OOD just reuses them on the OOD trajectory."""
    sfx = "" if regime == "id" else "_ood"
    cols = [f"com_r5_{i}{sfx}" for i in range(4)]
    end = f"com_r5_3{sfx}"
    fig, ax = C.plt.subplots(figsize=(7.5, 5.5))
    x = [0, 1, 2, 3]
    for _, r in d.iterrows():
        y = [r[c] for c in cols]
        if r["efficiency_class"] == "intrinsically low":
            ax.plot(x, y, "-", color="#b00020", lw=2.2, zorder=4, marker="o", markersize=4)
            ax.annotate(r["dataset"].replace("XAtlas2025_", "X_"), (3, y[-1]),
                        color="#b00020", fontsize=11, xytext=(4, 0), textcoords="offset points")
        else:
            ax.plot(x, y, "-", color="0.7", lw=1, alpha=0.7, marker="o", markersize=3)
    for name in ["Wong2026", "ReplogleWeissman2022_K562_gwps"]:
        if name in d["dataset"].values:
            r = d[d["dataset"] == name].iloc[0]
            ax.plot(x, [r[c] for c in cols], "-", color="#e9c46a", lw=2.2,
                    marker="o", markersize=4, zorder=3)
            ax.annotate(name.replace("ReplogleWeissman2022_", "RW_"), (3, r[end]),
                        color="#9c7a00", fontsize=11, xytext=(4, 0), textcoords="offset points")
    ax.set_xticks(x)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("efficiency threshold (none → e:3)", fontsize=14)
    ax.set_ylabel(f"ExPert {regime.upper()} {PRIMARY} — common class set", fontsize=14)
    ax.set_title(f"Recoverability (common-set, no class-dropping) — {regime.upper()}\n"
                 "red = intrinsically low (thresholding fails); yellow = low-eff but recoverable",
                 fontsize=14)
    C.save(fig, fname)


def support_vs_metric_tables():
    """Per-dataset (median cells/perturbation vs recall@5) and per-perturbation
    (train support vs per-class recall@5) support-vs-metric relationships."""
    meta = load.load_cell_meta()
    cpp = meta.groupby(["dataset", "perturbation"]).size().rename("n").reset_index()
    med = cpp.groupby("dataset")["n"].median().rename("median_cells_per_pert").reset_index()

    pc = load.load_threshold("none")["perclass"]
    pc = pc[(pc.method == "ExPert") & (pc.support > 0)]
    out = {}
    for regime in ("id", "ood"):
        g = pc[pc.regime == regime].groupby("dataset")
        r5 = g.apply(lambda x: np.average(x["recall@5"], weights=x["support"]),
                     include_groups=False).rename(f"recall5_{regime}")
        med = med.merge(r5, on="dataset", how="left")
    # per-perturbation (ID): per-class recall@5 vs train support
    gp = pc[pc.regime == "id"].groupby("class")
    perc = pd.DataFrame({
        "recall5": gp.apply(lambda x: np.average(x["recall@5"], weights=x["support"]),
                            include_groups=False),
        "n_cells_total": gp["support"].sum()}).reset_index()
    cov = load.load_train_covariates()[["perturbation", "train_support"]]
    perc = perc.merge(cov, left_on="class", right_on="perturbation", how="left")
    return med, perc


def plot_support_vs_metric(med: pd.DataFrame, perc: pd.DataFrame, eff: pd.DataFrame):
    cls_by_ds = dict(zip(eff["dataset"], eff["efficiency_class"]))
    cmap = {"high signal": "#2a9d8f", "low-eff (recoverable)": "#e9c46a",
            "intrinsically low": "#b00020", "moderate": "#6c757d"}
    fig, (axA, axB) = C.plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A — per dataset: median cells/perturbation vs ExPert ID recall@5
    rhoA, _ = spearmanr(med["median_cells_per_pert"], med["recall5_id"])
    for _, r in med.iterrows():
        axA.scatter(r["median_cells_per_pert"], r["recall5_id"], s=70,
                    color=cmap.get(cls_by_ds.get(r["dataset"], "moderate"), "#6c757d"),
                    edgecolor="k", lw=0.4, zorder=3)
        axA.annotate(r["dataset"].replace("Marson2025_", "M_").replace("ReplogleWeissman", "RW")
                     .replace("XAtlas2025_", "X_"),
                     (r["median_cells_per_pert"], r["recall5_id"]), fontsize=9,
                     xytext=(3, 3), textcoords="offset points")
    axA.set_xscale("log")
    axA.set_xlabel("median cells per perturbation (per dataset, log)")
    axA.set_ylabel(f"ExPert ID {PRIMARY}")
    axA.set_title(f"Per dataset (Spearman ρ={rhoA:.2f})")
    handles = [C.plt.Line2D([0], [0], marker="o", ls="", color=c, label=k, markeredgecolor="k")
               for k, c in cmap.items()]
    axA.legend(handles=handles, fontsize=11, title="failure mode", loc="lower right")

    # Panel B — per perturbation: train support vs per-class recall@5
    p = perc.dropna(subset=["train_support", "recall5"])
    rhoB, _ = spearmanr(p["train_support"], p["recall5"])
    axB.scatter(p["train_support"], p["recall5"], s=20, alpha=0.6,
                color=C.EXPERT_C, edgecolor="none")
    axB.set_xscale("log")
    axB.set_xlabel("train support per perturbation (cells, log)")
    axB.set_ylabel(f"per-class ID {PRIMARY} (support-weighted)")
    axB.set_title(f"Per perturbation, n={len(p)} (Spearman ρ={rhoB:.2f})")
    fig.suptitle(f"Support vs retrieval: matters per-perturbation (ρ={rhoB:.2f}) but NOT "
                 f"per-dataset (ρ={rhoA:.2f}) — dataset signal is efficiency, not cell count",
                 fontsize=15)
    C.save(fig, "axis1_support_vs_metric.svg")
    return rhoA, rhoB


def crossmethod_efficiency(value_col: str = "recall@5",
                           weighted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (method, dataset) ID metric (column ``value``) merged with dataset median
    efficiency, + per-method Spearman(efficiency, metric). ``weighted=True`` pulls the soft
    efficiency-weighted metrics from weighted_metrics.csv (e.g. ``weighted_macro_f1``).
    Tests whether the efficiency→performance link is ExPert-specific or holds across methods."""
    m = load.load_weighted() if weighted else load.load_threshold("none")["metrics"]
    m = m[m.regime == "id"]
    eff = load.dataset_efficiency_stats()[["dataset", "eff_med"]]
    perf = (m.groupby(["method", "dataset"])[value_col].mean().reset_index()
            .rename(columns={value_col: "value"}).merge(eff, on="dataset"))
    sp = (perf.groupby("method")
          .apply(lambda g: spearmanr(g["eff_med"], g["value"])[0], include_groups=False)
          .rename("spearman_rho").reset_index()
          .sort_values("spearman_rho", ascending=False))
    return perf, sp


def _method_colors(methods):
    """viridis (green=best→violet=worst) for real methods, grey for baselines — matches radar."""
    real = [m for m in methods if not m.startswith("baseline")]
    base = [m for m in methods if m.startswith("baseline")]
    col = {m: C.plt.cm.viridis(0.78 - j / max(len(real) - 1, 1) * 0.73) for j, m in enumerate(real)}
    for k, m in enumerate(base):
        col[m] = str(round(0.45 + k / max(len(base) - 1, 1) * 0.3, 2))
    return col


def plot_crossmethod_efficiency(perf: pd.DataFrame, sp: pd.DataFrame, metric_label: str,
                                fname: str):
    """Small multiples: efficiency vs metric per method (own y-scale), with Spearman ρ —
    proves the efficiency→performance link holds for every method, not just ExPert."""
    methods = list(sp["method"])                       # best-correlated first
    rho_by = dict(zip(sp["method"], sp["spearman_rho"]))
    n = len(methods)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = C.plt.subplots(nrow, ncol, figsize=(11, 3.1 * nrow))
    for ax, meth in zip(axes.flat, methods):
        g = perf[perf.method == meth]
        is_base = meth.startswith("baseline")
        ax.scatter(g["eff_med"], g["value"],
                   s=28, color="0.6" if is_base else (C.EXPERT_C if meth == "ExPert" else "#4c72b0"),
                   edgecolor="k", lw=0.3, alpha=0.85)
        if g["value"].std() > 0:                         # linear trend guide
            b, a = np.polyfit(g["eff_med"], g["value"], 1)
            xs = np.array([g["eff_med"].min(), g["eff_med"].max()])
            ax.plot(xs, a + b * xs, "--", color="0.4", lw=1)
        ttl = f"{meth}{' (baseline)' if is_base else ''}  (ρ={rho_by[meth]:.2f})"
        ax.set_title(ttl, fontsize=10, color="0.45" if is_base else "k")
        ax.tick_params(labelsize=9)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supxlabel("dataset median perturbation efficiency", fontsize=13)
    fig.supylabel(f"{metric_label} (ID)", fontsize=13)
    fig.suptitle(f"Efficiency → performance is not ExPert-specific ({metric_label}): it holds\n"
                 "across the learning methods (strong→weak), absent for signal-free baselines",
                 fontsize=14)
    fig.tight_layout()
    C.save(fig, fname)


def plot_crossmethod_heatmap(perf: pd.DataFrame, eff: pd.DataFrame, col_order: list,
                             metric_label: str, fname: str):
    """Heatmap datasets (rows, high efficiency at top) × methods (cols, ordered by how strongly
    that method's dataset ranking follows efficiency). Cells = per-method-normalised metric
    (0=that method's worst dataset, 1=its best). The clean top→bottom gradient on the left
    (learning methods) shows they all rank the datasets the same way — low-efficiency datasets
    worst — while it dissolves into noise for the signal-free baselines on the right."""
    wide = perf.pivot(index="dataset", columns="method", values="value")
    # dedupe the dataset index (eff can carry >1 row/dataset here) so eff_med[r] is
    # a scalar, not a Series (else the yticklabel f-string TypeErrors).
    eff = eff.drop_duplicates("dataset")
    eff_med = eff.set_index("dataset")["eff_med"]
    cls = eff.set_index("dataset")["efficiency_class"]
    rows = eff_med.sort_values(ascending=False).index.tolist()       # high eff at top, low bottom
    cols = [c for c in col_order if c in wide.columns]               # by efficiency-correlation
    norm = (wide - wide.min()) / (wide.max() - wide.min())           # per-method (column) min-max
    M = norm.loc[rows, cols]

    cmapcol = {"high signal": "#2a9d8f", "low-eff (recoverable)": "#e9c46a",
               "intrinsically low": "#b00020", "moderate": "#6c757d"}
    fig, ax = C.plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(M.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{C.disp(c)}{'*' if c.startswith('baseline') else ''}" for c in cols],
                       rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r.replace('Marson2025_','M_').replace('ReplogleWeissman','RW').replace('XAtlas2025_','X_')}  (eff {eff_med[r]:.1f})" for r in rows],
                       fontsize=9)
    for t, r in zip(ax.get_yticklabels(), rows):       # colour dataset labels by failure mode
        t.set_color(cmapcol.get(cls[r], "k"))
    ax.set_xlabel(f"method — ordered by Spearman(efficiency, {metric_label});  * = baseline",
                  fontsize=11)
    ax.set_title(f"Dataset × method, {metric_label} (per-method normalised)\n"
                 "high-eff datasets (top) best, low-eff (bottom) worst — for every learning method",
                 fontsize=14)
    fig.colorbar(im, ax=ax, label=f"{metric_label} (normalised within method)", shrink=0.7)
    C.save(fig, fname)


def plot_crossmethod_lines(perf: pd.DataFrame, eff: pd.DataFrame, metric_label: str, fname: str):
    """One line per method across datasets (x), y = within-method-normalised metric. Datasets
    ordered by the *learning*-method mean, descending — high-quality left, low-quality
    (XAtlas/Wong) right; baselines greyed/dashed and excluded from the ordering. All learning
    lines declining together = every method ranks the datasets the same way (low-quality worst);
    the baseline lines stay flat/scrambled."""
    wide = perf.pivot(index="dataset", columns="method", values="value")
    norm = (wide - wide.min()) / (wide.max() - wide.min())          # within-method min-max
    methods = list(norm.columns)
    learners = sorted([m for m in methods if not m.startswith("baseline")],
                      key=lambda m: -wide[m].mean())
    bases = [m for m in methods if m.startswith("baseline")]
    order = norm[learners].mean(axis=1).sort_values(ascending=False).index.tolist()
    col = _method_colors(methods)
    effc = eff.set_index("dataset")["efficiency_class"]
    cmapcol = {"high signal": "#2a9d8f", "low-eff (recoverable)": "#e9c46a",
               "intrinsically low": "#b00020", "moderate": "#6c757d"}

    fig, ax = C.plt.subplots(figsize=(14, 6))
    x = list(range(len(order)))
    # consensus: mean across learning methods (+min-max band) — the shared declining trend
    learn_vals = norm.loc[order, learners]
    ax.fill_between(x, learn_vals.min(axis=1), learn_vals.max(axis=1),
                    color="0.7", alpha=0.25, zorder=1, label="learning-method range")
    ax.plot(x, learn_vals.mean(axis=1), "-", color="k", lw=3, zorder=6,
            label="learning-method mean")
    for m in learners:
        ax.plot(x, [norm.loc[ds, m] for ds in order], "-o", color=col[m], lw=1.2, ms=3.5,
                alpha=0.8, label=m, zorder=4 if m == "ExPert" else 3)
    for m in bases:
        ax.plot(x, [norm.loc[ds, m] for ds in order], "--", color=col[m], lw=1.1, ms=3,
                marker="x", alpha=0.55, label=m + " *", zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ds.replace("Marson2025_", "M_").replace("ReplogleWeissman", "RW")
                        .replace("XAtlas2025_", "X_") for ds in order],
                       rotation=45, ha="right", fontsize=9)
    for t, ds in zip(ax.get_xticklabels(), order):
        t.set_color(cmapcol.get(effc.get(ds, "moderate"), "k"))
    ax.margins(x=0.01)
    ax.set_ylabel(f"{metric_label}\n(normalised within method)", fontsize=12)
    ax.set_xlabel("dataset — ordered by learning-method mean (high → low quality)", fontsize=12)
    ax.set_title(f"Every method ranks datasets the same way ({metric_label})\n"
                 "lines decline together high→low quality; * = baseline (flat/scrambled)",
                 fontsize=14)
    ax.legend(ncol=3, fontsize=9, loc="upper right", framealpha=0.9)
    C.save(fig, fname)


def plot_efficiency_distributions(d: pd.DataFrame):
    """Boxplot of per-cell efficiency per dataset, sorted by median, coloured by failure mode."""
    meta = load.load_cell_meta()
    order = d.sort_values("eff_med")["dataset"].tolist()
    cmap = {"high signal": "#2a9d8f", "low-eff (recoverable)": "#e9c46a",
            "intrinsically low": "#b00020", "moderate": "#6c757d"}
    cls_by_ds = dict(zip(d["dataset"], d["efficiency_class"]))
    data = [meta.loc[meta["dataset"] == ds, "efficiency"].clip(upper=8).values for ds in order]
    fig, ax = C.plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k"))
    for patch, ds in zip(bp["boxes"], order):
        patch.set_facecolor(cmap[cls_by_ds[ds]])
        patch.set_alpha(0.8)
    ax.set_yticklabels([ds.replace("Marson2025_", "M_").replace("ReplogleWeissman", "RW")
                        .replace("XAtlas2025_", "X_") for ds in order], fontsize=11)
    for thr in (1, 2, 3):
        ax.axvline(thr, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("per-cell perturbation efficiency (clipped at 8)")
    ax.set_title("Efficiency distribution by dataset (sorted by median)")
    handles = [C.plt.Line2D([0], [0], color=c, lw=6, alpha=0.8) for c in cmap.values()]
    ax.legend(handles, cmap.keys(), fontsize=11, title="failure mode", loc="lower right")
    C.save(fig, "axis1_efficiency_distributions.svg")


def main():
    print("Axis 1 — dataset signal")
    table = dataset_table()
    expert = classify_signal(table[table["method"] == "ExPert"].copy())
    expert = expert.sort_values(f"{PRIMARY}__id", ascending=False)

    ranked = expert[["dataset", "n_classes_present", "signal_class",
                     f"{PRIMARY}__id", f"{PRIMARY}__ood", "id_ood_gap",
                     f"{PRIMARY}_lift__id", f"{PRIMARY}_lift__ood",
                     "recall@1__id", "macro_f1_no_ds_bias__id"]].round(4)
    C.dump(ranked, "axis1_dataset_signal_ranked.csv")

    plot_id_vs_ood(expert)

    corr = correlate_properties(expert)
    C.dump(corr.round(4), "axis1_signal_vs_dataset_props.csv")

    xm = crossmethod_rank(table)
    C.dump(xm.round(4), "axis1_crossmethod_rank.csv")

    # efficiency decomposition (per-cell metadata)
    eff = efficiency_decomposition(expert)
    C.dump(eff[["dataset", "efficiency_class", "eff_med", "frac_ge2", "frac_ge3",
                "cells_per_pert", "n_perts",
                "raw_r5_0", "raw_r5_3", "com_r5_0", "com_r5_3",
                "raw_gain", "common_gain", "gain_from_class_drop",
                "n_classes_dropped", "n_common_classes",
                "com_r5_0_ood", "com_r5_3_ood", "common_gain_ood", "gain_from_class_drop_ood",
                f"{PRIMARY}__id", f"{PRIMARY}__ood"]].round(4),
           "axis1_dataset_efficiency.csv")
    plot_efficiency_distributions(eff)
    med, perc = support_vs_metric_tables()
    C.dump(med.round(4), "axis1_support_per_dataset.csv")
    rhoA, rhoB = plot_support_vs_metric(med, perc, eff)
    print(f"\nSupport vs metric: per-dataset ρ={rhoA:.2f} (median cells/pert vs recall@5); "
          f"per-perturbation ρ={rhoB:.2f} (train support vs per-class recall@5)")
    for regime in ("id", "ood"):
        sfx = "" if regime == "id" else "_ood"
        plot_signal_vs_efficiency(eff, regime, f"axis1_signal_vs_efficiency{sfx}.svg")
        plot_recoverability(eff, regime, f"axis1_efficiency_recoverability{sfx}.svg")
    rho_eff, _ = spearmanr(eff["eff_med"], eff[f"{PRIMARY}__id"])
    rho_eff_ood, _ = spearmanr(eff["eff_med"], eff[f"{PRIMARY}__ood"])

    # cross-method: does the efficiency->performance link hold beyond ExPert?
    # one row per dataset (eff can carry duplicates here) so the downstream
    # set_index("dataset") lookups return scalars, not Series.
    eff_meta = eff[["dataset", "eff_med", "efficiency_class"]].drop_duplicates("dataset")
    # soft (efficiency-weighted) metrics only: weighting toward actually-perturbed cells
    # decouples the dataset-quality finding from the low-efficiency cells themselves.
    for value_col, weighted, label, slug in [
        ("weighted_recall@5", True, "soft eff-weighted recall@5", "softr5"),
        ("weighted_macro_f1", True, "soft eff-weighted F1", "softf1"),
    ]:
        perf, sp_cm = crossmethod_efficiency(value_col, weighted)
        C.dump(sp_cm.round(4), f"axis1_crossmethod_efficiency_spearman_{slug}.csv")
        plot_crossmethod_efficiency(perf, sp_cm, label, f"axis1_crossmethod_efficiency_{slug}.svg")
        plot_crossmethod_heatmap(perf, eff_meta, list(sp_cm["method"]), label,
                                 f"axis1_crossmethod_heatmap_{slug}.svg")
        plot_crossmethod_lines(perf, eff_meta, label, f"axis1_crossmethod_lines_{slug}.svg")
        print(f"\nCross-method Spearman(efficiency, {label}) per method:")
        print(sp_cm.round(3).to_string(index=False))

    # console summary
    print("\nSignal-class counts:")
    print(expert["signal_class"].value_counts().to_string())
    print("\nTop driver correlations (|rho| desc):")
    print(corr.reindex(corr["spearman_rho"].abs().sort_values(ascending=False).index)
          .head(5).to_string(index=False))
    print(f"\nExPert mean rank across datasets: {xm['expert_rank'].mean():.2f} "
          f"(1=best of {int(xm['n_methods'].max())}); "
          f"wins (rank1): {(xm['expert_rank']==1).sum()}/{len(xm)}")
    print(f"\nEfficiency decomposition: Spearman(median efficiency, recall@5) = "
          f"{rho_eff:.2f} (ID), {rho_eff_ood:.2f} (OOD)")
    print("Efficiency-class counts:")
    print(eff["efficiency_class"].value_counts().to_string())
    print("Intrinsically low (thresholding fails):",
          list(eff[eff.efficiency_class == "intrinsically low"]["dataset"]))


if __name__ == "__main__":
    main()
