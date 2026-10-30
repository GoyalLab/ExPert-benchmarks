#!/usr/bin/env python3
"""Scalability: performance and resources as the class space grows (small -> ctx7 -> ...).

Three figures, x = number of perturbations C in the class space (log scale):

  scaling_performance.{png,svg}
      fold-improvement over random ("x over random", log y) for recall@10, MRR and median
      rank, ID and OOD. Random is analytical: recall@10 = 10/C, MRR = H_C/C, median rank
      = (C+1)/2 (the median-rank ratio is random/method, so higher is better everywhere).
      Efficiency-weighted and averaged over a MATCHED (fold, split) cell set per size,
      over the methods present at EVERY size (the radar's construction).
  scaling_resources.{png,svg}
      EVAL (inference) cost only, per benchmark fold: compute (CPU core-h + GPU-h, the
      cluster's own 1:1 pricing, see cost.py; linear, broken axis), SLURM jobs (the
      parallelisation burden) and peak per-job RAM. XGBoost / linreg / genepert train
      and score in ONE job, so their eval cannot be separated: they are drawn as
      train+eval upper bounds (hollow markers).
  scaling_slopes.{png,svg}
      the scaling exponent itself, per method: b in cost ~ C^b, i.e. the log-log slope
      between the smallest and largest size (0 = flat, 1 = linear in C), for compute per
      fold and for peak RAM; bars annotated with the raw fold-change.


Left out: scGPT (small was scored in the legacy label-conditioned mode, not comparable
with ctx7's classifier readout) and GEARS (CPU at small, GPU at ctx7: its cost change is
a hardware change, not scaling). Colours are collect_results.py's own performance-ranked
palette (viridis for methods, greys for baselines), ranked on the largest size.

  python utils/plot_scaling.py --out metrics/ctx7_preview/scaling \\
      --size small=/path/metrics_small:/path/slurm_costs.csv \\
      --size ctx7=/path/metrics_ctx7:/path/slurm_costs_ctx7.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms  # noqa: E401
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "workflow" / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "workflow" / "scripts"))
from cost import apply_overrides  # noqa: E402
from collect_results import _method_palette, _rank_methods, _disp  # noqa: E402

HIDE = {"baseline_centroid"}                 # superseded by baseline_centroid_notarget
DROP = {"scgpt", "scgpt_masked", "scgpt_zf", "gears"}   # see docstring
# the capped twin is the same method at small (500k cap > small's 491k train cells)
ALIAS = {"xgboost": "xgboost_capped"}
# cost-table method -> metrics method (fingerprint's per-dataset rules book as "fingerprint")
COST_TO_METRIC = {"fingerprint": "fingerprint_perdataset", "xgboost": "xgboost_capped"}
REC = "balanced_weighted_recall@10"          # collect's default --recall-mode balanced
PERF = [  # (column, label, chance(C), higher_is_better)
    (REC, "recall@10", lambda C: 10 / C, True),
    ("weighted_mrr", "MRR", lambda C: np.sum(1 / np.arange(1, C + 1)) / C, True),
    ("weighted_median_rank", "median rank", lambda C: (C + 1) / 2, False),
]


def n_classes(mdir: Path) -> int:
    return len(pd.read_csv(mdir / "train_class_support.csv"))


def _weighted(mdir: Path) -> pd.DataFrame:
    w = pd.read_csv(mdir / "weighted_metrics.csv")
    w["method"] = w["method"].replace(ALIAS)
    return w[~w["method"].isin(HIDE | DROP)]


def performance(mdir: Path, C: int, methods: set) -> pd.DataFrame:
    """x-over-random per (method, regime) on matched cells, over `methods` only."""
    w = _weighted(mdir)
    w = w[w["method"].isin(methods)].copy()
    w["regime"] = w["split"].astype(str).str.split(":").str[0].str.replace("_test", "", regex=False)
    w = w[w["regime"].isin(["id", "ood"])]
    key = ["fold", "split"]
    cells = None
    for _, g in w.groupby("method"):
        c = set(map(tuple, g[key].astype(str).to_numpy()))
        cells = c if cells is None else cells & c
    w = w[[tuple(r) in cells for r in w[key].astype(str).to_numpy()]]
    out = w.groupby(["method", "regime"])[[c for c, *_ in PERF]].mean()
    for col, lab, chance, hib in PERF:
        out[f"{lab} x random"] = out[col] / chance(C) if hib else chance(C) / out[col]
    out.attrs["matched"] = len(cells)
    return out.reset_index()


def resources(costs: Path, size: str, n_folds: int) -> pd.DataFrame:
    """EVAL compute-hours per fold (CPU core-h + GPU-h), eval jobs per fold and peak eval
    RAM per method. Single-job train+eval methods are kept and flagged (`combined`)."""
    c = pd.read_csv(costs)
    c = c[c["size"].astype(str) == size].copy()
    c = apply_overrides(c, size)          # measured ExPert eval RAM etc. (config/cost_overrides.yaml)
    c["method"] = c["method"].map(lambda m: COST_TO_METRIC.get(m, m))
    c = c[~c["method"].isin(DROP) & c["stage"].isin(["eval", "train+eval"])]
    c["compute_h"] = c["cpu_hours"].fillna(0) + c["gpu_hours"].fillna(0)
    g = c.groupby("method").agg(compute_h=("compute_h", "sum"), ram=("max_ram_GB", "max"),
                                n_jobs=("n_jobs", "sum"),
                                combined=("stage", lambda x: (x == "train+eval").any()))
    g["per_fold"] = g["compute_h"] / n_folds
    # SLURM jobs per fold (slurm_costs.py counts each rule+wildcards job once): the
    # parallelisation burden -- thousands of chunk jobs are only feasible on a cluster.
    g["jobs_per_fold"] = g["n_jobs"] / n_folds
    if "ExPert" in g.index:
        g.loc["ExPert", "jobs_per_fold"] = 1   # one eval job per fold, by design
    return g.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", action="append", required=True,
                    help="NAME=METRICS_DIR:COSTS_CSV (repeat, smallest first)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    specs = [(n, *r.split(":", 1)) for n, r in (s.split("=", 1) for s in a.size)]
    common = set.intersection(*(set(_weighted(Path(m))["method"]) for _, m, _ in specs))
    print("methods at every size:", ", ".join(sorted(common)))
    perf, res, Cs = [], [], {}
    for name, mdir, costs in specs:
        C = n_classes(Path(mdir)); Cs[name] = C
        p = performance(Path(mdir), C, common); p["size"], p["C"] = name, C; perf.append(p)
        print(f"{name}: C={C}, matched cells={p.attrs.get('matched')}")
        r = resources(Path(costs), name, a.folds); r["size"], r["C"] = name, C
        res.append(r)
    perf = pd.concat(perf, ignore_index=True); res = pd.concat(res, ignore_index=True)
    perf.to_csv(out / "scaling_performance.csv", index=False)
    res.to_csv(out / "scaling_resources.csv", index=False)
    xs = sorted(Cs.values())

    # collect_results' palette: ranked on the LARGEST size's sweep, as collect does
    big = Path(specs[-1][1])
    sw = pd.read_csv(big / "sweep_metrics.csv")
    sw["method"] = sw["method"].replace(ALIAS)
    ranked = _rank_methods(sw, "balanced_recall@")
    pal = _method_palette(ranked, baselines=tuple(m for m in ranked if "baseline" in m.lower()))
    order = [m for m in pal if m not in HIDE | DROP]
    COST_BASE = "baselines"                    # the cost table books all baselines as one
    pal.setdefault(COST_BASE, pal.get("baseline_centroid_notarget", (0.5, 0.5, 0.5, 1)))

    def style(m):
        st = dict(color=pal.get(m, "0.4"), marker="o", zorder=5)
        if m == "ExPert":
            st.update(lw=2.6, ms=7, zorder=10)
        elif "baseline" in m:
            st.update(lw=1.2, ms=4, ls="--", zorder=3)
        else:
            st.update(lw=1.5, ms=5)
        return st

    combined = set(res.loc[res["combined"].astype(bool), "method"])

    def label(m, res_panel=False):
        if m == COST_BASE:
            return "baselines (all)"
        return _disp(m) + (" (train+eval)" if res_panel and m in combined else "")

    def rstyle(m):
        st = style(m)
        if m in combined:                      # upper bound: eval not separable
            st.update(mfc="white", mew=1.4)
        return st

    def legend_below(fig, axes, ncol=5):
        h, l = [], []
        for ax in np.ravel(axes):
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh); l.append(ll)
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol,
                   frameon=False, fontsize=9)

    def xaxis(ax):
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}" for x in xs])
        ax.minorticks_off(); ax.grid(alpha=0.3)

    # ---- performance: x over random ------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for i, reg in enumerate(("id", "ood")):
        d = perf[(perf["regime"] == reg) & (perf["method"] != "baseline_random")]
        for j, (_, lab, _, _) in enumerate(PERF):
            ax = axes[i, j]
            for m in order:
                g = d[d["method"] == m].sort_values("C")
                if len(g):
                    ax.plot(g["C"], g[f"{lab} x random"], label=label(m), **style(m))
            ax.axhline(1, color="0.3", lw=1, ls=":", zorder=1)
            ax.text(0.99, 1, "random ", fontsize=8, color="0.3", va="top", ha="right",
                    transform=ax.get_yaxis_transform())
            ax.set_yscale("log")
            xaxis(ax)
            ax.set_title(f"{lab} — {reg.upper()}")
            if j == 0:
                ax.set_ylabel("× over random")
            if i == 1:
                ax.set_xlabel("perturbations in the class space")
    fig.suptitle("Performance over random as the class space grows "
                 "(efficiency-weighted, matched cells)", y=1.0)
    fig.tight_layout()
    legend_below(fig, axes)
    for ext in ("png", "svg"):
        fig.savefig(out / f"scaling_performance.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- resources ------------------------------------------------------------------
    # compute: LINEAR y with a broken axis (cost.py's eval-cost style) -- a log axis made
    # 5,000 vs 2,000 core-h look close; jobs: log y (1 .. 10^3); RAM: linear.
    r_order = [m for m in order if m in set(res["method"])] + [COST_BASE]
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(17, 5.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.28)
    vals = np.sort(res["per_fold"].to_numpy())
    mx = vals[-1]
    # break at the widest gap (>2x) whose upper side is a real share of the maximum
    gaps = [(vals[i + 1] / vals[i], i) for i in range(len(vals) - 1)
            if vals[i] > 0 and vals[i + 1] > 2 * vals[i] and vals[i + 1] >= 0.15 * mx]
    if gaps:
        _, ib = max(gaps)
        sub_gs = gs[0, 0].subgridspec(2, 1, height_ratios=[1, 1.6], hspace=0.07)
        axt = fig.add_subplot(sub_gs[0]); axb = fig.add_subplot(sub_gs[1], sharex=axt)
        comp_axes = (axt, axb)
    else:
        axb = fig.add_subplot(gs[0, 0]); axt = None
        comp_axes = (axb,)
    ax_jobs = fig.add_subplot(gs[0, 1], sharex=axb)
    ax_ram = fig.add_subplot(gs[0, 2], sharex=axb)
    for ax in comp_axes:
        for m in r_order:
            g = res[res["method"] == m].sort_values("C")
            if len(g):
                ax.plot(g["C"], g["per_fold"], label=label(m, True), **rstyle(m))
        xaxis(ax)
    if axt is not None:
        axt.set_ylim(vals[ib + 1] * 0.85, mx * 1.06)
        axb.set_ylim(0, vals[ib] * 1.18)
        axt.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="lower"))
        axb.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        axt.spines["bottom"].set_visible(False); axb.spines["top"].set_visible(False)
        axt.tick_params(labelbottom=False, bottom=False)
        d, kw = .012, dict(color="k", clip_on=False, lw=1)
        axt.plot((-d, d), (-d * 1.6, d * 1.6), transform=axt.transAxes, **kw)
        axt.plot((1 - d, 1 + d), (-d * 1.6, d * 1.6), transform=axt.transAxes, **kw)
        axb.plot((-d, d), (1 - d, 1 + d), transform=axb.transAxes, **kw)
        axb.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axb.transAxes, **kw)
        axt.set_title("Eval compute (axis broken)")
    else:
        axb.set_title("Eval compute")
    axb.set_ylabel("eval compute-hours per fold\n(CPU core-h + GPU-h)")
    axb.set_xlabel("perturbations in the class space")
    for ax, col, ylab, title, log in (
            (ax_jobs, "jobs_per_fold", "eval SLURM jobs per fold",
             "Parallelisation burden", True),
            (ax_ram, "ram", "peak RAM per eval job (GB)", "Memory footprint", False)):
        for m in r_order:
            g = res[res["method"] == m].sort_values("C")
            if len(g):
                ax.plot(g["C"], g[col], label=label(m, True), **rstyle(m))
        xaxis(ax)
        if log:
            ax.set_yscale("log")
        if col == "jobs_per_fold":             # several methods share exactly 1 job: say so
            one = [label(m, True) for m in r_order
                   if np.allclose(res.loc[res["method"] == m, col], 1)]
            if len(one) > 1:
                import textwrap
                ax.set_ylim(0.3, None)         # room under the shared line for the note
                ax.text(0.5, 0.88, textwrap.fill("1 job per fold: " + ", ".join(one), 55),
                        fontsize=7.5, linespacing=1.1,
                        color="0.3", ha="center", va="top",
                        transform=matplotlib.transforms.blended_transform_factory(
                            ax.transAxes, ax.transData))
        ax.set_xlabel("perturbations in the class space"); ax.set_ylabel(ylab)
        ax.set_title(title)
    legend_below(fig, [*comp_axes, ax_jobs, ax_ram], ncol=6)
    fig.text(0.5, -0.17, "Eval (inference) only. Hollow markers: methods that train and score "
             "in one SLURM job (eval not separable) -- train+eval upper bounds. Jobs = distinct "
             "SLURM jobs of one clean pass (re-runs counted once).",
             ha="center", fontsize=9, color="0.3")
    for ext in ("png", "svg"):
        fig.savefig(out / f"scaling_resources.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- slopes: the scaling exponent itself ----------------------------------------
    # Burden = eval compute-h per fold x eval SLURM jobs per fold: what it costs AND how
    # much cluster it takes to spread that cost. Its exponent is b_compute + b_jobs, so a
    # method that keeps compute flat by fanning out into ever more jobs still scores high.
    lo, hi = min(xs), max(xs)
    res["burden"] = res["per_fold"] * res["jobs_per_fold"]
    rows = []
    for m in r_order:
        g = res[res["method"] == m].set_index("C")
        if lo in g.index and hi in g.index:
            for col, lab in (("per_fold", "compute"), ("jobs_per_fold", "jobs"),
                             ("burden", "burden"), ("ram", "peak RAM")):
                fc = g.loc[hi, col] / g.loc[lo, col]
                rows.append(dict(method=m, what=lab, fold_change=fc,
                                 exponent=np.log(fc) / np.log(hi / lo)))
    sl = pd.DataFrame(rows)
    sl.to_csv(out / "scaling_slopes.csv", index=False)
    fc_of = sl.set_index(["method", "what"])["fold_change"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.5 * len(r_order) + 1.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ylab = [label(m, True) for m in r_order]
    for ax, what in zip(axes, ("burden", "peak RAM")):
        d = sl[sl["what"] == what].set_index("method").reindex(r_order)
        y = np.arange(len(r_order))
        ax.barh(y, d["exponent"], color=[pal.get(m, "0.5") for m in r_order],
                edgecolor="black", lw=0.4)
        for yy, (m, e, fc) in enumerate(zip(r_order, d["exponent"], d["fold_change"])):
            if not np.isfinite(e):
                continue
            txt = f"×{fc:.2g}"
            if what == "burden":
                txt += (f"  (compute ×{fc_of[(m, 'compute')]:.2g} · "
                        f"jobs ×{fc_of[(m, 'jobs')]:.2g})")
            # negative bars: label on the positive side so it never meets the tick labels
            ax.text(max(e, 0) + 0.04, yy, txt, va="center", ha="left", fontsize=8)
        for ref, txt in ((0, "flat"), (1, "linear in C")):
            ax.axvline(ref, color="0.3", lw=1, ls=":" if ref else "-", zorder=0)
            ax.text(ref, len(r_order) - 0.4, f" {txt}", fontsize=8, color="0.3", va="bottom")
        ax.set_yticks(y); ax.set_yticklabels(ylab); ax.invert_yaxis()
        ax.set_xlabel(f"scaling exponent b  (∝ C^b, C = {lo:,} → {hi:,})")
        ax.set_title({"burden": "Eval burden: compute-hours × SLURM jobs per fold",
                      "peak RAM": "Peak RAM per eval job"}[what])
        ax.grid(alpha=0.3, axis="x")
        lo_x = min(-0.2, np.nanmin(d["exponent"]) - 0.35)
        hi_x = max(1.3, np.nanmax(d["exponent"]) + (1.15 if what == "burden" else 0.45))
        ax.set_xlim(lo_x, hi_x)
    fig.text(0.5, -0.03, "Eval (inference) only. Burden exponent = compute exponent + jobs "
             "exponent. (train+eval) = one job that trains and scores, eval not separable.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"scaling_slopes.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", *(out / f"scaling_{k}.png" for k in ("performance", "resources", "slopes")))


if __name__ == "__main__":
    main()
