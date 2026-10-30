"""Axis 3 — efficiency thresholding.

Thresholding cells by perturbation efficiency (min_efficiency 0->1->2->3) raises
recall@5 but discards cells and whole perturbations. Quantify the tradeoff/knee, check
the gain is not just an artefact of dropping hard classes (common-set comparison), see
whether the gain concentrates on previously-hard perturbations, and compare hard
thresholding to soft efficiency-weighting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load
import common as C

PRIMARY = load.PRIMARY  # recall@5
LEVELS = [0, 1, 2, 3]


def sweep_curve() -> pd.DataFrame:
    """ExPert recall@5 + cells retained vs threshold, aggregated over splits/folds (ID)."""
    sw = load.load_sweep(per_fold=False)
    sw = sw[(sw["method"] == "ExPert") & (sw["regime"] == "id")]
    # cell-weighted mean recall across splits (n = cells at that threshold)
    rows = []
    for me, g in sw.groupby("min_efficiency"):
        rows.append({
            "min_efficiency": me,
            "recall@5": np.average(g[PRIMARY], weights=g["n"]),
            "recall@1": np.average(g["recall@1"], weights=g["n"]),
            "macro_f1_no_ds_bias": g["macro_f1_no_ds_bias"].mean(),
            "total_cells": g["n"].sum(),
        })
    cur = pd.DataFrame(rows).sort_values("min_efficiency")
    base_cells = cur.loc[cur["min_efficiency"] == 0, "total_cells"].iloc[0]
    cur["cells_retained_frac"] = cur["total_cells"] / base_cells
    cur["recall@5_gain_vs_none"] = cur["recall@5"] - cur["recall@5"].iloc[0]
    cur["recall@5_step_gain"] = cur["recall@5"].diff()
    return cur


def coverage_curve() -> pd.DataFrame:
    """# perturbations still evaluable (>0 cells) vs threshold, from the cells parquet (ID)."""
    cells = load.load_cells_per_perturbation()
    cells = cells[cells["regime"] == "id"]
    # per (split, label, threshold): total cells over folds
    agg = (cells.groupby(["split", "label", "min_efficiency"])["n_cells"].sum()
           .reset_index())
    rows = []
    for me, g in agg.groupby("min_efficiency"):
        present = g[g["n_cells"] > 0]
        rows.append({"min_efficiency": me,
                     "n_pert_split_present": len(present),
                     "median_cells_per_pert": present["n_cells"].median()})
    cov = pd.DataFrame(rows).sort_values("min_efficiency")
    base = cov.loc[cov["min_efficiency"] == 0, "n_pert_split_present"].iloc[0]
    cov["pert_coverage_frac"] = cov["n_pert_split_present"] / base
    return cov


def common_set_gain() -> pd.DataFrame:
    """Fair gain: recall@5 only on (class,dataset) pairs evaluable at ALL thresholds (ID).

    Uses per-class metrics from each e:N dir so the comparison set is identical.
    """
    pc = load.load_all_thresholds()["perclass"]
    pc = pc[(pc["method"] == "ExPert") & (pc["regime"] == "id") & (pc["support"] > 0)]
    # key = (split, class); keep keys present at all 4 thresholds
    key_levels = pc.groupby(["split", "class"])["min_efficiency"].nunique()
    common = key_levels[key_levels == len(LEVELS)].index
    pc = pc.set_index(["split", "class"]).loc[common].reset_index()
    rows = []
    for me, g in pc.groupby("min_efficiency"):
        rows.append({"min_efficiency": me,
                     "recall@5_commonset": np.average(g[PRIMARY], weights=g["support"]),
                     "n_keys": g[["split", "class"]].drop_duplicates().shape[0]})
    cs = pd.DataFrame(rows).sort_values("min_efficiency")
    cs["recall@5_gain_commonset"] = cs["recall@5_commonset"] - cs["recall@5_commonset"].iloc[0]
    return cs


def differential_gain() -> pd.DataFrame:
    """Per-class recall@5 gain (none->e:3) stratified by baseline (none) difficulty tercile (ID)."""
    pc = load.load_all_thresholds()["perclass"]
    pc = pc[(pc["method"] == "ExPert") & (pc["regime"] == "id") & (pc["support"] > 0)]
    per = (pc.groupby(["class", "min_efficiency"])
             .apply(lambda g: np.average(g[PRIMARY], weights=g["support"]),
                    include_groups=False)
             .rename(PRIMARY).reset_index())
    wide = per.pivot(index="class", columns="min_efficiency", values=PRIMARY).dropna()
    wide["gain_none_to_e3"] = wide[3] - wide[0]
    wide["baseline_tercile"] = pd.qcut(wide[0], 3, labels=["hard", "medium", "easy"])
    strat = (wide.groupby("baseline_tercile", observed=True)
                 .agg(n=("gain_none_to_e3", "size"),
                      baseline_recall5=(0, "mean"),
                      mean_gain=("gain_none_to_e3", "mean")).reset_index())
    return wide.reset_index(), strat


def soft_vs_hard() -> pd.DataFrame:
    """Efficiency-weighted (soft) metrics vs hard thresholds, ExPert ID, cell-weighted."""
    w = load.load_weighted()
    w = w[(w["method"] == "ExPert") & (w["regime"] == "id")]
    soft = {"approach": "soft (obs_efficiency weighted, all cells kept)",
            "recall@5": np.average(w["weighted_recall@5"], weights=w["n"]),
            "cells_retained_frac": 1.0}
    cur = sweep_curve()
    rows = [soft]
    for _, r in cur.iterrows():
        label = "none (no threshold)" if r["min_efficiency"] == 0 else f"hard e:{int(r['min_efficiency'])}"
        rows.append({"approach": label, "recall@5": r["recall@5"],
                     "cells_retained_frac": r["cells_retained_frac"]})
    return pd.DataFrame(rows)


def plot_knee(cur: pd.DataFrame, cov: pd.DataFrame, cs: pd.DataFrame):
    fig, ax = C.plt.subplots(figsize=(7, 5))
    ax.plot(cur["min_efficiency"], cur["recall@5"], "o-", color=C.EXPERT_C,
            label="recall@5 (raw, all evaluable)")
    ax.plot(cs["min_efficiency"], cs["recall@5_commonset"], "s--", color="#2a9d8f",
            label="recall@5 (common-set, fair)")
    ax.set_xlabel("min_efficiency threshold")
    ax.set_ylabel("recall@5")
    ax.set_xticks(LEVELS)
    ax2 = ax.twinx()
    ax2.plot(cur["min_efficiency"], cur["cells_retained_frac"], "^:", color=C.FIELD_C,
             label="cells retained")
    ax2.plot(cov["min_efficiency"], cov["pert_coverage_frac"], "v:", color="#e9c46a",
             label="pert coverage")
    ax2.set_ylabel("fraction retained")
    ax2.grid(False)
    ax2.set_ylim(0, 1.05)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=11, loc="center right")
    ax.set_title("Efficiency thresholding: gain vs cost (ExPert, ID)")
    C.save(fig, "axis3_efficiency_knee.svg")


def plot_differential(strat: pd.DataFrame):
    fig, ax = C.plt.subplots(figsize=(6.5, 4.5))
    ax.bar(strat["baseline_tercile"].astype(str), strat["mean_gain"], color=C.EXPERT_C, alpha=0.85)
    ax.set_ylim(0, strat["mean_gain"].max() * 1.32)   # headroom so labels clear the title
    for i, r in strat.iterrows():
        ax.text(i, r["mean_gain"] + 0.005, f"{r['mean_gain']:.3f}\n(base {r['baseline_recall5']:.2f})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("mean recall@5 gain (none → e:3)")
    ax.set_xlabel("baseline difficulty tercile (at none)")
    ax.set_title("Does thresholding help hard perturbations more?", pad=10)
    C.save(fig, "axis3_gain_by_difficulty.svg")


def eff_readout_correlation(metric: str = "recall@5", min_support: int = 2) -> pd.DataFrame:
    """Spearman rho between a perturbation's |efficiency| and its readout, per method.

    Unit = (fold x dataset x perturbation): median |efficiency| over that group's cells vs
    the per-class recall the method achieved on it. ABSOLUTE efficiency because the column
    is signed (10% of cells are negative here, min -4.5) -- what should predict a readable
    readout is the MAGNITUDE of the knockdown, not its direction.

    Reported separately for ID and OOD: a method can track efficiency well in-distribution
    and lose that on held-out datasets, and that difference is the interesting part.
    """
    from scipy.stats import spearmanr

    pc = load.load_threshold("none")["perclass"]
    pc = pc[pc["support"] >= min_support]
    cm = load.load_cell_meta()
    eff = (cm.assign(abs_eff=cm["efficiency"].abs())
             .dropna(subset=["abs_eff"])
             .groupby(["dataset", "perturbation"])["abs_eff"].median()
             .rename("abs_eff").reset_index()
             .rename(columns={"perturbation": "class"}))
    df = pc.merge(eff, on=["dataset", "class"], how="inner").dropna(subset=[metric, "abs_eff"])
    rows = []
    for (meth, regime), g in df.groupby(["method", "regime"]):
        if len(g) < 20 or g["abs_eff"].nunique() < 3:
            continue
        rho, pv = spearmanr(g["abs_eff"], g[metric])
        rows.append({"method": meth, "regime": regime, "rho": float(rho),
                     "p": float(pv), "n": int(len(g))})
    return pd.DataFrame(rows)


def plot_eff_readout_correlation(corr: pd.DataFrame, metric: str = "recall@5"):
    """Dot plot: one row per method, ID and OOD rho side by side."""
    if corr.empty:
        print("  (no correlation rows — skipping)")
        return
    wide = corr.pivot_table(index="method", columns="regime", values="rho")
    wide = wide.reindex(load.order_methods(wide.index))
    y = np.arange(len(wide))[::-1]
    fig, ax = C.plt.subplots(figsize=(6.0, max(3.0, 0.34 * len(wide) + 1.2)))
    ax.axvline(0, color="0.6", lw=1.0, ls="--", zorder=1)
    for regime, colr, mkr in (("id", "#3b7dd8", "o"), ("ood", "#d1495b", "D")):
        if regime not in wide.columns:
            continue
        ax.scatter(wide[regime].values, y, s=62, color=colr, marker=mkr,
                   label=f"{regime.upper()} test", zorder=3,
                   edgecolor="white", linewidth=0.8)
    for yi, (_, r) in zip(y, wide.iterrows()):
        if {"id", "ood"} <= set(wide.columns) and pd.notna(r.get("id")) and pd.notna(r.get("ood")):
            ax.plot([r["id"], r["ood"]], [yi, yi], color="0.75", lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([C.disp(m) for m in wide.index], fontsize=9)
    ax.set_xlabel(f"Spearman rho  (|efficiency|  vs  per-class {metric})")
    ax.set_title("Does perturbation efficiency predict the readout?\n"
                 "unit = fold x dataset x perturbation; |eff| = median abs efficiency",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    C.save(fig, "axis3_eff_readout_correlation.svg")


def main():
    print("Axis 3 — efficiency thresholding")
    cur = sweep_curve()
    cov = coverage_curve()
    cs = common_set_gain()
    tradeoff = cur.merge(cov[["min_efficiency", "n_pert_split_present", "pert_coverage_frac",
                              "median_cells_per_pert"]], on="min_efficiency") \
                  .merge(cs[["min_efficiency", "recall@5_commonset", "recall@5_gain_commonset",
                             "n_keys"]], on="min_efficiency")
    C.dump(tradeoff.round(4), "axis3_efficiency_tradeoff.csv")

    wide, strat = differential_gain()
    C.dump(strat.round(4), "axis3_gain_by_difficulty.csv")
    svh = soft_vs_hard()
    C.dump(svh.round(4), "axis3_soft_vs_hard.csv")

    plot_knee(cur, cov, cs)
    plot_differential(strat)

    corr = eff_readout_correlation()
    C.dump(corr.round(4), "axis3_eff_readout_correlation.csv")
    plot_eff_readout_correlation(corr)

    print("\nTradeoff (ID, ExPert):")
    print(tradeoff[["min_efficiency", "recall@5", "recall@5_gain_vs_none",
                    "recall@5_commonset", "recall@5_gain_commonset",
                    "cells_retained_frac", "pert_coverage_frac"]].round(3).to_string(index=False))
    print("\nDifferential gain by baseline difficulty:")
    print(strat.round(3).to_string(index=False))
    print("\nSoft vs hard:")
    print(svh.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
