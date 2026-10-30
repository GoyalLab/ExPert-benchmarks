"""Synthesis — joint model of what makes a (perturbation, dataset, efficiency) recoverable,
and assembly of REPORT.md from the per-axis outputs.

Joint model: standardised OLS of per-(class, dataset, threshold) recall@5 against the
levers from all three axes, so their relative effect sizes are directly comparable.
Predictors are deliberately exogenous (training volume / dataset shape / threshold /
regime) — the dataset *outcome* is not used as a feature, to avoid circularity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load
import common as C

PRIMARY = load.PRIMARY


def build_joint_frame() -> pd.DataFrame:
    pc = load.load_all_thresholds()["perclass"]
    pc = pc[(pc["method"] == "ExPert") & (pc["support"] > 0)].copy()
    keep = ["class", "dataset", "regime", "min_efficiency", PRIMARY, "support"]
    pc = pc[keep]

    cov = load.load_train_covariates()[["perturbation", "log_support", "dataset_entropy",
                                        "n_train_datasets"]]
    pc = pc.merge(cov, left_on="class", right_on="perturbation", how="left")

    ts = load.load_train_stats()[["dataset", "n_perturbations"]]
    pc = pc.merge(ts, on="dataset", how="left")
    pc["log_n_classes"] = np.log10(pc["n_perturbations"].fillna(pc["n_perturbations"].median()))
    pc["is_ood"] = (pc["regime"] == "ood").astype(float)
    return pc


def joint_model(df: pd.DataFrame):
    feats = ["min_efficiency", "is_ood", "log_support", "dataset_entropy",
             "n_train_datasets", "log_n_classes"]
    d = df.dropna(subset=feats + [PRIMARY]).copy()
    # standardise predictors so coefficients are comparable effect sizes
    Z = (d[feats] - d[feats].mean()) / d[feats].std()
    X = sm.add_constant(Z)
    model = sm.WLS(d[PRIMARY].values, X, weights=d["support"].values).fit()
    res = pd.DataFrame({
        "feature": ["intercept"] + feats,
        "std_coef": model.params.values,
        "p_value": model.pvalues.values,
    })
    res["abs_coef"] = res["std_coef"].abs()
    body = res[res["feature"] != "intercept"].sort_values("abs_coef", ascending=False)
    return body, model.rsquared, len(d)


def support_reconciliation(df: pd.DataFrame) -> dict:
    """Quantify the 'big lever vs intrinsic' tension: support's effect SIZE vs the
    VARIANCE it explains. Returns numbers used in the TL;DR reconciliation."""
    from scipy.stats import pearsonr, spearmanr

    feats = ["min_efficiency", "is_ood", "log_support", "dataset_entropy",
             "n_train_datasets", "log_n_classes"]
    d = df.dropna(subset=feats + [PRIMARY]).copy()

    def wls_r2(cols):
        Z = (d[cols] - d[cols].mean()) / d[cols].std()
        return sm.WLS(d[PRIMARY], sm.add_constant(Z), weights=d["support"]).fit().rsquared

    full = wls_r2(feats)
    no_sup = wls_r2([c for c in feats if c != "log_support"])
    only_sup = wls_r2(["log_support"])

    # per-class (one row per class, support-weighted recall@5, ID only)
    pc = load.load_threshold("none")["perclass"]
    pc = pc[(pc.method == "ExPert") & (pc.regime == "id") & (pc.support > 0)]
    g = pc.groupby("class")
    cls = pd.DataFrame({
        "recall5": g.apply(lambda x: np.average(x[PRIMARY], weights=x["support"]),
                           include_groups=False),
        "support": g["support"].sum()})
    cls["log_support"] = np.log10(cls["support"])
    r, _ = pearsonr(cls.log_support, cls.recall5)
    rho, _ = spearmanr(cls.log_support, cls.recall5)
    cls["q"] = pd.qcut(cls.support, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    q = cls.groupby("q", observed=True)["recall5"].agg(["mean", "min", "max"])
    return {
        "joint_full_r2": full, "joint_unique_support_r2": full - no_sup,
        "joint_only_support_r2": only_sup, "joint_unexplained": 1 - full,
        "perclass_pearson_r": r, "perclass_r2": r ** 2, "perclass_spearman": rho,
        "q1_mean": q.loc["Q1", "mean"], "q1_min": q.loc["Q1", "min"], "q1_max": q.loc["Q1", "max"],
        "q4_mean": q.loc["Q4", "mean"], "q4_min": q.loc["Q4", "min"], "q4_max": q.loc["Q4", "max"],
    }


def _above_chance(scaled: pd.Series) -> pd.Series:
    """scaled median rank (1 = first, 0.5 = chance) -> 0 at chance or worse, 0.5 at rank 1."""
    return (scaled - 0.5).clip(lower=0.0)


def method_radar():
    """Per-method fingerprint on the three headline metrics, ID and OOD:

      * soft efficiency-WEIGHTED metrics (obs_efficiency, weighted_metrics.csv), one row
        per fold x dataset;
      * averaged over a MATCHED cell set -- only the (fold, split) cells that EVERY
        method has, as in collect's ood_sweep_matched. A method that does not cover all
        folds (gears mid-run: 9 of 12) would otherwise be averaged over a different set
        of datasets, and part of the gap would be fold composition. With full coverage
        matched == all cells, so this costs nothing once a run is complete.

    Returns the raw table (median rank already shifted to above-chance for the radar).
    """
    s = load.load_weighted()
    s = s[s["regime"].isin(["id", "ood"])]
    key = ["fold", "split"]
    cells = None
    for _, g in s.groupby("method"):
        c = set(map(tuple, g[key].astype(str).to_numpy()))
        cells = c if cells is None else cells & c
    n_all = s[key].astype(str).drop_duplicates().shape[0]
    s = s[[tuple(r) in cells for r in s[key].astype(str).to_numpy()]]
    print(f"  [radar] matched on {len(cells)} of {n_all} (fold x dataset) cells "
          f"({s[s.regime == 'id'][key].drop_duplicates().shape[0]} ID, "
          f"{s[s.regime == 'ood'][key].drop_duplicates().shape[0]} OOD)")
    r10 = _wcol(s, 10)                           # reported recall flavour, weighted

    def agg(regime, col):
        return s[s["regime"] == regime].groupby("method")[col].mean()

    raw = pd.DataFrame({
        "MRR\n(ID)": agg("id", "weighted_mrr"),
        "recall@10\n(ID)": agg("id", r10),
        # radar-only: scaled median rank minus chance (0.5), floored at 0 -- so chance is
        # 0 on EVERY spoke. Unshifted, these spokes sat at 0.5-0.98 while recall/MRR sat
        # below 0.45, and the shared radius squeezed every polygon into the middle.
        "median rank\n(above chance, ID)": _above_chance(agg("id", "weighted_median_rank_scaled")),
        "median rank\n(above chance, OOD)": _above_chance(agg("ood", "weighted_median_rank_scaled")),
        "recall@10\n(OOD)": agg("ood", r10),
        "MRR\n(OOD)": agg("ood", "weighted_mrr"),
    })
    # ALL scored methods, known ones first (METHODS is an order, not a whitelist).
    return raw.loc[load.order_methods(raw.index)]


# The three headline metrics, each asking a different question of the ranking:
#   MRR                   precision at the top (rank 1-3 counts most, degrades gracefully)
#   recall@10             is the true driver on the shortlist someone would validate?
#   median rank           where does it land in the FULL ranking -- the disease readout;
#                         plotted ABOVE CHANCE: 1 - (median rank - 1)/(C - 1) - 0.5, floored
#                         at 0, so 0 = chance (as on every other spoke) and 0.5 = rank 1
# Corner order (clockwise from upper-left): ID trio across the top L->R, OOD trio mirrored
# beneath it, so each metric's ID/OOD stack vertically (MRR left, median rank right).
# Replaces macro-F1 / recall@5 / recall@10: F1 scores a hard single-class call the
# readout never makes, and recall@5 asked nearly the same question as recall@10.
RADAR_AXES = ["MRR\n(ID)", "recall@10\n(ID)", "median rank\n(above chance, ID)",
              "median rank\n(above chance, OOD)", "recall@10\n(OOD)", "MRR\n(OOD)"]


def plot_method_radar(raw: pd.DataFrame):
    axes = RADAR_AXES
    rr = raw[axes]                              # absolute scores, all already on [0, 1]
    N = len(axes)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0.0]
    # order methods best->worst by mean metric. Real methods: viridis (green=best, violet=worst).
    # Baselines: shades of grey (dark=better) so they read as reference, not contenders.
    order = rr.mean(axis=1).sort_values(ascending=False).index.tolist()
    real = [m for m in order if not m.startswith("baseline")]
    base = [m for m in order if m.startswith("baseline")]
    color = {m: C.plt.cm.viridis(0.78 - (j / (max(len(real) - 1, 1))) * 0.73)
             for j, m in enumerate(real)}
    for k, m in enumerate(base):
        color[m] = str(round(0.45 + k / max(len(base) - 1, 1) * 0.3, 2))  # 0.45->0.75 grey

    rmax = float(rr.values.max()) * 1.1            # zoom radial axis to ~10% past the top corner

    fig, ax = C.plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.radians(150))   # axis 0 (MRR ID) at upper-left (flat-top hexagon)
    ax.set_theta_direction(-1)             # clockwise: ID trio over the top, OOD mirrored beneath

    # ID / OOD halves. Axes sit at data-angles 0/60/120 (ID) and 180/240/300 (OOD), so the
    # split falls midway between the trios: 150° and 330°. Two near-white greys — they have to
    # stay lighter than the lightest baseline line (0.75 grey) or the dashed baselines vanish.
    for lo, hi, shade in ((-30, 150, "0.965"), (150, 330, "0.915")):
        th = np.radians(np.linspace(lo, hi, 181))
        ax.fill_between(th, 0, rmax, color=shade, zorder=0, linewidth=0)
    for rank, meth in enumerate(order):                 # plot best first -> on top via zorder
        vals = rr.loc[meth].tolist() + [rr.loc[meth].iloc[0]]
        is_expert = meth == "ExPert"
        is_base = meth.startswith("baseline")
        ax.plot(angles, vals, color=color[meth],
                lw=3 if is_expert else (1.1 if is_base else 1.6),
                ls="--" if is_base else "-",
                label=C.disp(meth), zorder=100 - rank, alpha=0.9)
        if is_expert:
            ax.fill(angles, vals, color=color[meth], alpha=0.15, zorder=1)

    ticks = [round(t, 1) for t in np.arange(0.1, rmax + 1e-9, 0.1)]  # round refs: 0.1, 0.2, ...
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=13)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])                          # hide default; draw on both recall@1 spokes
    ax.set_ylim(0, rmax)
    # radial scale numbers mirrored on both recall@10 spokes (data-angles 60° top, 240° bottom)
    for a in (np.radians(60), np.radians(240)):
        for t in ticks:
            ax.text(a, t, f"{t:g}", fontsize=11, color="0.4", ha="center", va="center",
                    zorder=200, bbox=dict(facecolor="white", edgecolor="none",
                                          alpha=0.65, pad=0.4))
    # No title and no half labels: the spoke labels already carry "(ID)" / "(OOD)", so the
    # shading is a redundant grouping cue rather than the only way to read the split.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=11,
              frameon=False, columnspacing=1.2)
    C.save(fig, "synthesis_method_radar.svg")


def plot_joint(body: pd.DataFrame):
    fig, ax = C.plt.subplots(figsize=(6.5, 4))
    colors = [C.EXPERT_C if v >= 0 else C.FIELD_C for v in body["std_coef"]]
    ax.barh(body["feature"], body["std_coef"], color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("standardised WLS coefficient (effect on recall@5)")
    ax.set_title("Joint model: relative levers on recall@5")
    C.save(fig, "synthesis_joint_levers.svg")


def _wcol(df, k: int) -> str:
    """Efficiency-weighted recall@k column matching the reported flavour."""
    want = ("balanced_weighted_recall@" if PRIMARY.startswith("balanced_")
            else "weighted_recall@") + str(k)
    return want if want in df.columns else f"weighted_recall@{k}"


def _read(name):
    return pd.read_csv(C.OUT / name)


def write_report(body, r2, n, rec):
    a1 = _read("axis1_dataset_signal_ranked.csv")
    a1corr = _read("axis1_signal_vs_dataset_props.csv")
    a1xm = _read("axis1_crossmethod_rank.csv")
    a2 = _read("axis2_perturbation_difficulty_ranked.csv")
    a2d = _read("axis2_perclass_drivers.csv")
    a2c = _read("axis2_confusable_candidates.csv")
    a2o = _read("axis2_ood_drop_per_class.csv")
    a3 = _read("axis3_efficiency_tradeoff.csv")
    a3s = _read("axis3_gain_by_difficulty.csv")
    a3sh = _read("axis3_soft_vs_hard.csv")

    sc = a1["signal_class"].value_counts()
    win = (a1xm["expert_rank"] == 1).sum()
    md = []
    md.append("# ExPert benchmark — analytical findings (200-perturbation subset)\n")
    md.append("> Primary metric **recall@5**. ID = held-out cells of a seen dataset; "
              "OOD = held-out dataset. Cross-dataset comparisons also use *lift over "
              "`baseline_prior`* because datasets differ in class count (chance rate). "
              "All figures/tables in `analysis/out/`.\n")
    md.append("> Caveat: 200 / >9k perturbations. Treat every ranking as a hypothesis to "
              "re-test at scale; per-class tails are the most sample-size sensitive.\n")

    eff_coef = body.loc[body.feature == "min_efficiency", "std_coef"].iloc[0]
    sup_coef = body.loc[body.feature == "log_support", "std_coef"].iloc[0]
    md.append("\n## TL;DR refinement levers\n")
    md.append(f"1. **Efficiency is the most useful lever you control at inference.** In the "
              f"joint model efficiency (std-coef `{eff_coef:+.3f}`) and per-class training "
              f"support (`{sup_coef:+.3f}`) have the two largest *slopes*, but **soft "
              f"efficiency-weighting recovers almost all of the hard-threshold gain with "
              f"zero cells dropped** "
              f"(soft {a3sh.loc[a3sh.approach.str.startswith('soft'),'recall@5'].iloc[0]:.3f} "
              f"≈ hard e:1 {a3sh.loc[a3sh.approach=='hard e:1','recall@5'].iloc[0]:.3f} vs "
              f"none {a3sh.loc[a3sh.approach=='none (no threshold)','recall@5'].iloc[0]:.3f}). "
              "Prefer weighting/efficiency-aware loss over discarding cells.\n")
    md.append("2. **Per-perturbation difficulty is mostly intrinsic, not a data-volume "
              "problem.** Support has the largest slope *of the knobs measured*, but those "
              "knobs are weak in absolute terms: support explains only "
              f"~{rec['perclass_r2']*100:.0f}% of between-class difficulty (Pearson "
              f"r²={rec['perclass_r2']:.2f}) and adds just "
              f"~{rec['joint_unique_support_r2']*100:.0f}% *unique* variance in the joint "
              f"model ({rec['joint_unexplained']*100:.0f}% stays unexplained). At matched "
              f"support, recall@5 still swings the full range (top quartile "
              f"{rec['q4_min']:.2f}→{rec['q4_max']:.2f}). So **more cells nudge any class up "
              "a little but won't turn a hard perturbation into an easy one** — gains come "
              "from features/architecture that separate confusable gene families. "
              "(See *Why these aren't contradictory* below.)\n")
    md.append("3. **A clear confusable-family signature exists**: the largest recall@1→@5 "
              "gaps are co-functional complexes (e.g. mito-ribosomal `MRPS22/26/28`, "
              "amino-acid transporters `SLC1A5/SLC7A5`). The model finds the right "
              "neighbourhood but not the exact gene — a candidate for contrastive / "
              "hard-negative training.\n")
    a1e_t = _read("axis1_dataset_efficiency.csv")
    rho_e, _ = spearmanr(a1e_t["eff_med"], a1e_t[f"{PRIMARY}__id"])
    intrins = a1e_t[a1e_t.efficiency_class == "intrinsically low"]["dataset"].tolist()
    wong = a1e_t[a1e_t.dataset == "Wong2026"].iloc[0]
    md.append(f"4. **\"Low-signal\" mostly means \"low perturbation efficiency\" — except "
              f"XAtlas.** Per-cell efficiency explains dataset signal almost entirely "
              f"(Spearman ρ={rho_e:.2f}). Restricting to high-efficiency cells — on a *fixed "
              f"class set*, so this isn't a class-dropping artifact — rescues most low-signal "
              f"datasets (Wong2026 {wong['com_r5_0']:.2f}→{wong['com_r5_3']:.2f}), but **not "
              + ", ".join(d.replace('XAtlas2025_', 'XAtlas ') for d in intrins)
              + "** (≈0.10→0.20 even at e:3) — their high-efficiency cells still don't classify, "
              "so the issue is intrinsic (miscalibrated efficiency / unreliable assay), not the "
              "model. Quarantine them rather than tuning to them. (Axis 1.)\n")
    md.append(f"5. **OOD is a generalization wall, not just low signal**: ExPert wins "
              f"{win}/{len(a1xm)} datasets on ID yet OOD recall@5 collapses on most "
              "(see Axis 1). The id→ood drop is concentrated in specific perturbations "
              "(Axis 2) — domain-adaptation should target those.\n")
    f1_id = _read("axis2_crossmethod_perclass_f1_id.csv")
    r5_id = _read("axis2_crossmethod_perclass_recall5_id.csv")
    f1_win = int((f1_id["expert_minus_best"] > 0).sum())
    r5_win = int((r5_id["expert_minus_best"] > 0).sum())
    md.append(f"6. **ExPert's edge is precision, not just retrieval**: per-perturbation it beats "
              f"the best competitor on **{f1_win}/{len(f1_id)}** classes by **F1** but only "
              f"**{r5_win}/{len(r5_id)}** by **recall@5** (ID). The field can often surface the "
              "right perturbation in its top-5 (closer recall@5), but ExPert is far better at "
              "committing to the correct top-1 with clean precision — the cross-method gap is "
              "widest on F1 and widest still on F1 at efficiency≥2 (Axis 2).\n")

    # Cross-method fingerprint (radar)
    radar = _read("synthesis_method_radar.csv")
    md.append("\n## Methods at a glance (radar)\n")
    md.append(f"Per-method fingerprint on the three headline metrics, **soft "
              "efficiency-weighted** (`obs_efficiency`) and averaged over a **matched** cell set "
              "(only the fold × dataset cells every method covers, as in "
              "`plots/ood_sweep_matched`): **MRR** "
              "(precision at the top), **recall@10** (is the true driver on a validation "
              "shortlist) and the **median rank above chance**, 1 − (median rank − 1)/(C − 1) − 0.5 "
              f"floored at 0 (where it lands in the full ranking of C = {load.n_classes():,} "
              "classes; 0.5 = ranked first, 0 = chance or worse). Chance is 0 on every spoke. "
              "The **ID** trio runs across the top and the **OOD** trio is "
              "mirrored beneath, so each metric's ID/OOD stack vertically and the top→bottom "
              "squash of a polygon is its generalization drop. Absolute values, radius = the "
              "metric itself. Real methods are coloured by overall rank (viridis, "
              "green = best → violet = worst); baselines are dashed greys. Raw values below and "
              "in `synthesis_method_radar.csv`.\n\n")
    md.append("![Method fingerprint radar](out/synthesis_method_radar.svg)\n\n")
    md.append(radar.round(3).to_markdown(index=False) + "\n")

    # Axis 1
    md.append("\n## Axis 1 — dataset signal\n")
    md.append("Signal-class split (median ID/OOD recall@5 thresholds): "
              + ", ".join(f"{k} = {v}" for k, v in sc.items()) + ".\n\n")
    # column names follow PRIMARY (axis1 pivots THAT metric into <name>__id/__ood),
    # so this report tracks whichever recall flavour collect was told to report
    _cols = ["dataset", f"{PRIMARY}__id", f"{PRIMARY}__ood", "id_ood_gap", "signal_class"]
    md.append(f"**Highest-signal (ID {PRIMARY}):**\n\n")
    md.append(a1.head(5)[_cols].to_markdown(index=False) + "\n\n")
    md.append("**Lowest-signal:**\n\n")
    md.append(a1.tail(5)[_cols].to_markdown(index=False) + "\n\n")
    topc = a1corr.reindex(a1corr["spearman_rho"].abs().sort_values(ascending=False).index).head(3)
    md.append("Dataset-property correlations (Spearman, n=17 train datasets): "
              + "; ".join(f"{r['target']} vs {r['property']} ρ={r['spearman_rho']:.2f} "
                          f"(p={r['p_value']:.3f})" for _, r in topc.iterrows()) + ".\n")
    md.append(f"\nExPert is rank-1 on **{win}/{len(a1xm)}** datasets "
              f"(mean rank {a1xm['expert_rank'].mean():.2f} of {int(a1xm['n_methods'].max())}).\n\n")
    md.append("![ID vs OOD dataset signal](out/axis1_id_vs_ood_scatter.svg)\n")

    # Axis 1b — efficiency decomposition (per-cell metadata)
    a1e = _read("axis1_dataset_efficiency.csv")
    rho_eff, _ = spearmanr(a1e["eff_med"], a1e[f"{PRIMARY}__id"])
    cls = a1e["efficiency_class"].value_counts()
    intrinsic = a1e[a1e.efficiency_class == "intrinsically low"]["dataset"].tolist()
    recov = a1e[a1e.efficiency_class == "low-eff (recoverable)"].sort_values("common_gain",
                                                                            ascending=False)
    rho_eff_ood, _ = spearmanr(a1e["eff_med"], a1e[f"{PRIMARY}__ood"])
    md.append("\n### Is low signal just low perturbation efficiency?\n")
    md.append("Using per-cell perturbation efficiency (`small/data/source.obs.csv`), a dataset's "
              f"median efficiency **almost fully explains its signal** — Spearman(median "
              f"efficiency, ExPert recall@5) = **{rho_eff:.2f}** (ID), **{rho_eff_ood:.2f}** "
              "(OOD). Mostly, \"low-signal\" means \"the perturbation barely took effect in most "
              "cells.\"\n\n")
    md.append("But efficiency is not the whole story — restricting each dataset to its "
              "*high-efficiency* cells (threshold e:3) splits the low-signal datasets in two:\n\n")
    md.append(f"- **Low-eff but recoverable ({int(cls.get('low-eff (recoverable)', 0))} datasets):** "
              "baseline low, yet high-efficiency cells classify well — e.g. "
              + ", ".join(f"{r['dataset'].replace('ReplogleWeissman','RW')} "
                          f"({r['com_r5_0']:.2f}→{r['com_r5_3']:.2f})"
                          for _, r in recov.head(3).iterrows())
              + ". For these the signal is real; it's diluted by weak-perturbation cells, so "
              "efficiency weighting / thresholding (Axis 3) is the fix.\n")
    md.append(f"- **Intrinsically low ({int(cls.get('intrinsically low', 0))} datasets — "
              f"{', '.join(intrinsic)}):** even the high-efficiency cells stay near the floor "
              + "; ".join(f"{r['dataset'].replace('XAtlas2025_','X_')} "
                          f"{r['com_r5_0']:.2f}→{r['com_r5_3']:.2f}"
                          for _, r in a1e[a1e.efficiency_class == 'intrinsically low'].iterrows())
              + ". Thresholding does **not** rescue them — so the problem is not low efficiency "
              "but something intrinsic (likely a miscalibrated efficiency score or an unreliable "
              "assay/readout for XAtlas). These should be quarantined, not blamed on the model.\n\n")
    md.append("\n**Is the recoverability rise just the efficiency filter dropping hard "
              "classes?** No. Recomputing the trajectory on the *common class set* (classes "
              "present at every threshold, so none can drop out) isolates the genuine "
              "within-class effect. Class-dropping contributes only a small slice of the gain, "
              "and the intrinsic/recoverable split is unchanged:\n\n")
    rep = a1e[a1e.dataset.isin(["XAtlas2025_HEK293T", "XAtlas2025_HCT116", "Wong2026",
                                "ReplogleWeissman2022_K562_gwps", "Zhuang2026",
                                "Marson2025_D2_Stim8hr"])].copy()
    rep["dataset"] = rep["dataset"].str.replace("ReplogleWeissman2022_", "RW_") \
        .str.replace("XAtlas2025_", "X_").str.replace("Marson2025_", "M_")
    rep = rep[["dataset", "efficiency_class", "raw_gain", "common_gain",
               "gain_from_class_drop", "n_classes_dropped"]].rename(columns={
        "raw_gain": "gain (raw)", "common_gain": "gain (common-set, real)",
        "gain_from_class_drop": "from class-drop", "n_classes_dropped": "#classes dropped"})
    md.append(rep.round(3).to_markdown(index=False) + "\n\n")
    md.append("XAtlas's common-set gain is ~0.06–0.11 (vs 0.23–0.40 for recoverable datasets) "
              "and it drops barely any classes — so its flatness is real, not a filtering "
              "artifact. (The recoverability figures use the common-set trajectory.)\n\n")
    # OOD: same structure, muted by the generalization wall
    iz = a1e[a1e.efficiency_class == "intrinsically low"]
    rz = a1e[a1e.efficiency_class == "low-eff (recoverable)"]
    md.append(f"**Does it hold OOD?** Yes, with the generalization wall stacked on top. "
              f"Efficiency still tracks OOD signal (ρ={rho_eff_ood:.2f}), and thresholding still "
              "splits the datasets the same way — but every ceiling is lower. Recoverable "
              "datasets rise to only ~0.27–0.37 OOD (vs 0.6–0.84 ID): e.g. Wong2026 "
              + "→".join(f"{rz[rz.dataset=='Wong2026'][c].iloc[0]:.2f}"
                         for c in ["com_r5_0_ood", "com_r5_3_ood"]) + ", K562_gwps "
              + "→".join(f"{rz[rz.dataset.str.contains('K562_gwps')][c].iloc[0]:.2f}"
                         for c in ["com_r5_0_ood", "com_r5_3_ood"])
              + ". The two XAtlas datasets stay pinned to the floor OOD too ("
              + "; ".join(f"{r['dataset'].replace('XAtlas2025_','X_')} "
                          f"{r['com_r5_0_ood']:.2f}→{r['com_r5_3_ood']:.2f}"
                          for _, r in iz.iterrows())
              + ") — intrinsic, not a distribution-shift effect.\n\n")
    md.append("| | ID | OOD |\n|:--|:--:|:--:|\n")
    md.append("| signal vs efficiency | ![](out/axis1_signal_vs_efficiency.svg) "
              "| ![](out/axis1_signal_vs_efficiency_ood.svg) |\n")
    md.append("| recoverability (common-set) | ![](out/axis1_efficiency_recoverability.svg) "
              "| ![](out/axis1_efficiency_recoverability_ood.svg) |\n\n")
    md.append("![efficiency distribution by dataset](out/axis1_efficiency_distributions.svg)\n")

    # support vs metric (per dataset and per perturbation)
    sd = _read("axis1_support_per_dataset.csv")
    rho_ds, _ = spearmanr(sd["median_cells_per_pert"], sd["recall5_id"])
    md.append("\n**Is it cell *count* rather than efficiency?** No — and this is a clean "
              "double dissociation. **More cells per perturbation does *not* make a dataset "
              f"higher-signal** (per-dataset Spearman ρ={rho_ds:.2f}, essentially zero — "
              "efficiency, ρ≈0.93, is what matters). But **at the perturbation level, more "
              "*training* cells do help** (per-class recall@5 vs train support ρ≈0.50). So "
              "support is a per-perturbation lever, not a per-dataset one: a low-signal dataset "
              "won't be fixed by more cells, but an individual rare perturbation benefits from "
              "them.\n\n")
    md.append("![support vs metric (per dataset and per perturbation)](out/axis1_support_vs_metric.svg)\n")

    # cross-method: is the efficiency->performance link ExPert-specific?
    cmr = _read("axis1_crossmethod_efficiency_spearman_softr5.csv").rename(
        columns={"spearman_rho": "ρ (soft recall@5)"})
    cmf = _read("axis1_crossmethod_efficiency_spearman_softf1.csv").rename(
        columns={"spearman_rho": "ρ (soft F1)"})
    both = cmr.merge(cmf, on="method")
    both = both.assign(method=lambda x: x.method.where(~x.method.str.startswith("baseline"),
                                                       x.method + " *"))
    md.append("\n**Does the efficiency→performance link generalize beyond ExPert?** Yes — it is "
              "a property of *extracting real signal*, not an ExPert artifact. Here we use the "
              "**soft (efficiency-weighted) metrics**, which weight each cell by its perturbation "
              "efficiency — so the metric reflects performance on actually-perturbed cells and "
              "the finding can't be a mechanical low-efficiency-cell artifact. Per-method "
              "Spearman(dataset median efficiency, soft metric) across the 21 datasets:\n\n")
    md.append(both.round(2).to_markdown(index=False) + "\n\n")
    md.append("Every *learning* method ranks the datasets the same way — high-efficiency best, "
              "low-efficiency (XAtlas, Wong2026, K562_gwps) worst — strongest for ExPert and "
              "fingerprint, clearly present for scouter/cellot/gears; only genepert (which "
              "barely learns) is weak. The signal-free baselines (`*`) do **not** track "
              "efficiency, exactly as expected if low efficiency means less real perturbation "
              "signal. So the 'low-quality dataset' effect is intrinsic to the data, not to our "
              "model.\n\n")
    md.append("Every learning method's soft metric still falls as efficiency falls, and the "
              "pure baselines sit negative — confirming the dataset-quality effect survives "
              "efficiency weighting (it is not just that low-efficiency cells score low). Small "
              "multiples (each method's trend) and the dataset×method heatmap (shared "
              "high-eff-top→low-eff-bottom ranking) below, for both soft metrics.\n\n")
    for lbl, slug in [("soft eff-weighted recall@5", "softr5"),
                      ("soft eff-weighted F1", "softf1")]:
        md.append(f"*{lbl}:*\n\n")
        md.append(f"![efficiency vs {lbl} per method](out/axis1_crossmethod_efficiency_{slug}.svg)\n\n")
        md.append(f"![dataset x method heatmap, {lbl}](out/axis1_crossmethod_heatmap_{slug}.svg)\n\n")
    md.append("The same data as lines — one line per method across datasets, datasets ordered "
              "left→right by learning-method mean (high→low quality), y = within-method-normalised "
              "metric. The bold black **learning-method mean** (with the min–max band) declines "
              "monotonically high→low quality, and the individual learning-method lines track it; "
              "baselines (`*`, dashed grey) stay flat/scrambled. Low-quality datasets (XAtlas, "
              "Wong2026 — red labels, right) are worst for every method:\n\n")
    md.append("![per-dataset lines, soft recall@5](out/axis1_crossmethod_lines_softr5.svg)\n\n")
    md.append("![per-dataset lines, soft-F1](out/axis1_crossmethod_lines_softf1.svg)\n")

    # Axis 2
    md.append("\n## Axis 2 — per-perturbation retrieval\n")
    md.append("**Easiest perturbations:** " + ", ".join(a2.head(8)["class"]) + ".\n\n")
    md.append("**Hardest perturbations:** " + ", ".join(a2.tail(8)["class"][::-1]) + ".\n\n")
    md.append("**Drivers of per-class recall@5** (univariate Spearman | standardised ridge):\n\n")
    md.append(a2d[["feature", "spearman_rho", "ridge_std_coef"]].to_markdown(index=False) + "\n\n")
    md.append("**Confusable families** (largest recall@1→@5 gap = right neighbourhood, "
              "wrong top-1):\n\n")
    md.append(a2c.head(8)[["class", "confusable_gap", "recall@1_w", "recall@5_w"]]
              .to_markdown(index=False) + "\n\n")
    md.append("**Biggest ID→OOD drop** (fails to generalize across contexts): "
              + ", ".join(a2o.head(8)["class"]) + ".\n\n")
    md.append("![Drivers of per-class recall@5](out/axis2_perclass_drivers.svg)\n\n")
    md.append("![Difficulty vs cross-dataset consistency](out/axis2_perclass_dataset_variance.svg)\n")

    # cross-method per-perturbation — 3 metrics × {ID, OOD}
    def _xm_stats(t):
        wins = int((t["expert_minus_best"] > 0).sum())
        loses = t[t["expert_minus_best"] < 0].sort_values("expert_minus_best")
        only = t[(t["best_competitor_score"] < 0.1) & (t["ExPert"] >= 0.3)]
        freq = t["best_competitor"].value_counts()
        return wins, loses, only, freq

    md.append("\n**ExPert vs best competitor, per perturbation.** Points above the diagonal "
              "are ExPert wins; colour marks the closest competitor for that perturbation; "
              "the legend (below each plot) gives how many perturbations each method is the "
              "toughest on. Shown for three metrics: **recall@5**, **F1**, and **F1 at "
              "efficiency≥2** (the closest per-perturbation proxy for efficiency-weighted F1 — "
              "true soft F1 is only available at the split level in `weighted_metrics.csv`).\n")
    metric_blocks = [
        ("recall@5", "recall5", "Top-5 retrieval."),
        ("recall@5 (efficiency≥2)", "recall5e2",
         "Top-5 retrieval, efficiency-aware proxy (only high-efficiency cells kept)."),
        ("F1", "f1", "Precision×recall — ExPert's lead widens here vs recall@5."),
        ("F1 (efficiency≥2)", "f1e2", "Efficiency-aware proxy; only high-efficiency cells kept."),
    ]
    for label, slug, note in metric_blocks:
        idd = _read(f"axis2_crossmethod_perclass_{slug}_id.csv")
        ood = _read(f"axis2_crossmethod_perclass_{slug}_ood.csv")
        w, l, o, f = _xm_stats(idd)
        wo, lo, oo, fo = _xm_stats(ood)
        md.append(f"\n*{label}.* {note} ID: ExPert beats the best other method on "
                  f"**{w}/{len(idd)}** (loses {len(l)}, {len(o)} ExPert-only); toughest "
                  f"`{f.index[0]}` ({int(f.iloc[0])}). OOD: **{wo}/{len(ood)}** "
                  f"(loses {len(lo)}, {len(oo)} ExPert-only); toughest "
                  f"`{fo.index[0]}` ({int(fo.iloc[0])}).\n\n")
        # ID | OOD side by side (markdown table renders images in columns)
        md.append("| ID | OOD |\n|:--:|:--:|\n")
        md.append(f"| ![{label} ID](out/axis2_expert_vs_best_competitor_{slug}_id.svg) "
                  f"| ![{label} OOD](out/axis2_expert_vs_best_competitor_{slug}_ood.svg) |\n")

    # Axis 3
    md.append("\n## Axis 3 — efficiency thresholding\n")
    md.append("Tradeoff (ExPert, ID; raw vs fair common-set; cost = cells/coverage retained):\n\n")
    md.append(a3[["min_efficiency", "recall@5", "recall@5_gain_vs_none",
                  "recall@5_commonset", "recall@5_gain_commonset",
                  "cells_retained_frac", "pert_coverage_frac"]].round(3)
              .to_markdown(index=False) + "\n\n")
    e1 = a3[a3.min_efficiency == 1].iloc[0]
    md.append(f"**Knee at e:1**: +{e1['recall@5_gain_vs_none']:.3f} raw "
              f"(+{e1['recall@5_gain_commonset']:.3f} fair, common-set) for "
              f"{e1['cells_retained_frac']:.0%} cells retained. Roughly "
              f"{e1['recall@5_gain_commonset']/e1['recall@5_gain_vs_none']:.0%} of the raw "
              "gain is genuine improvement; the rest is from dropping hard classes.\n\n")
    md.append("Gain by baseline difficulty (does thresholding help hard perts more?):\n\n")
    md.append(a3s.to_markdown(index=False) + "\n\n")
    md.append("**Soft vs hard** — efficiency-weighting keeps all cells:\n\n")
    md.append(a3sh.to_markdown(index=False) + "\n\n")
    md.append("![Efficiency thresholding: gain vs cost](out/axis3_efficiency_knee.svg)\n\n")
    md.append("![Gain by baseline difficulty](out/axis3_gain_by_difficulty.svg)\n")

    # Joint
    md.append("\n## Joint model — relative levers\n")
    md.append(f"Standardised WLS (support-weighted) of per-(class, dataset, threshold) "
              f"recall@5 on exogenous levers. R²={r2:.3f}, n={n} rows. "
              "Coefficients are comparable effect sizes (predictor SD → recall@5 units):\n\n")
    md.append(body[["feature", "std_coef", "p_value"]].round(4).to_markdown(index=False) + "\n\n")
    md.append("![Joint model levers](out/synthesis_joint_levers.svg)\n")

    # Reconciliation callout
    md.append("\n### Why \"support is a big lever\" and \"difficulty is intrinsic\" aren't "
              "contradictory\n")
    md.append("They answer two different questions — *effect size* vs *variance explained*:\n\n")
    md.append(f"- **Effect size (slope):** among the knobs measured, support has one of the "
              f"largest coefficients (+1 SD log_support → +{sup_coef:.3f} recall@5). Real and "
              "reliable — but it's the strongest of a deliberately limited menu "
              "(training-volume / dataset-shape features only; no biology).\n")
    md.append(f"- **Variance explained (absolute):** those knobs are weak overall. Support "
              f"explains only ~{rec['perclass_r2']*100:.0f}% of between-class difficulty "
              f"(per-class Pearson r²={rec['perclass_r2']:.2f}; Spearman "
              f"{rec['perclass_spearman']:.2f}), and adds just "
              f"~{rec['joint_unique_support_r2']*100:.0f}% *unique* variance in the joint "
              f"model (full R²={rec['joint_full_r2']:.2f}, support-only "
              f"R²={rec['joint_only_support_r2']:.2f}); "
              f"~{rec['joint_unexplained']*100:.0f}% stays unexplained.\n")
    md.append("- **The clincher — hold support fixed, difficulty still swings the full "
              "range** (per-class recall@5 by support quartile):\n\n")
    md.append(pd.DataFrame({
        "support quartile": ["Q1 (low)", "Q4 (high)"],
        "mean recall@5": [round(rec["q1_mean"], 2), round(rec["q4_mean"], 2)],
        "min → max": [f"{rec['q1_min']:.2f} → {rec['q1_max']:.2f}",
                       f"{rec['q4_min']:.2f} → {rec['q4_max']:.2f}"],
    }).to_markdown(index=False) + "\n\n")
    md.append("A high-support class can still sit near the floor and a low-support class can "
              "land mid-pack — so support *nudges* difficulty but does not *determine* it. "
              "**Takeaway:** adding cells to a hard perturbation helps a little but won't make "
              "it easy; the leverage is in separating confusable gene families.\n")

    md.append("\n## Method / reproduction\n")
    md.append("`bash analysis/run.sh all` (env: `.venv-analysis`, built with uv). "
              "Loader self-test: `analysis/run.sh load.py`. Outputs land in `analysis/out/` "
              "and are logged to `analysis/out/analysis.log`.\n")

    path = C.OUT.parent / "REPORT.md"
    path.write_text("".join(md))
    print(f"  wrote {path.relative_to(C.OUT.parent.parent)}")


def main():
    print("Synthesis — joint model + report")
    df = build_joint_frame()
    body, r2, n = joint_model(df)
    C.dump(body[["feature", "std_coef", "p_value", "abs_coef"]].round(4),
           "synthesis_joint_levers.csv")
    plot_joint(body)
    radar = method_radar()
    C.dump(radar.reset_index().round(4), "synthesis_method_radar.csv")
    plot_method_radar(radar)
    rec = support_reconciliation(df)
    print(f"\nJoint WLS R²={r2:.3f} (n={n}). Levers by |effect|:")
    print(body[["feature", "std_coef", "p_value"]].round(4).to_string(index=False))
    print(f"Support reconciliation: per-class r²={rec['perclass_r2']:.2f}, "
          f"joint unique ΔR²={rec['joint_unique_support_r2']:.3f}, "
          f"unexplained={rec['joint_unexplained']:.0%}")
    write_report(body, r2, n, rec)


if __name__ == "__main__":
    main()
