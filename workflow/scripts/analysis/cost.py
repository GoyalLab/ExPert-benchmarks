"""Resource-cost axis — what each method cost to produce.

Reads the run's ``slurm_costs.csv`` (emitted by the onsuccess/onerror hook via
utils/slurm_costs.py: one row per method x stage with cpu/gpu/node-hours + peak
RAM) and turns it into three views:

  1. compute_cost_by_method  — CPU-hours (+ GPU-hours) per method, stage-split, log x
  2. peak_ram_by_method      — peak per-job RAM per method x stage
  3. cost_vs_performance     — total compute-hours (log) vs headline id recall@5

The point of (3) is the bang-for-buck read: fingerprint's per-dataset ensemble is
enormously expensive per unit accuracy, genepert/baselines are ~free. ExPert has no
cost row here (trained outside this pipeline), so it's annotated, not plotted.

Non-fatal by design: if slurm_costs.csv is absent (e.g. a run that never triggered
the hook) the script prints a note and exits 0 so it can't break collect.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

# cost.method (from RULE_MAP / --external) -> metric.method (in all_metrics.csv).
# fingerprint's per-dataset rules are booked as "fingerprint"; the metrics call it
# perdataset. ExPert is booked via --external (its training timed-out-with-best-ckpt).
COST_TO_METRIC = {
    "fingerprint": "fingerprint_perdataset", "fingerprint_merged": "fingerprint_merged",
    "cellot": "cellot", "gears": "gears", "scouter": "scouter", "genepert": "genepert",
    "scgpt": "scgpt", "ExPert": "ExPert",
}
STAGE_C = {"train": "#4e79a7", "eval": "#f28e2b", "train+eval": "#59a14f"}
GPU_C = "#76b7b2"

# CPU and GPU combined into ONE compute figure: compute_hours = CPU core-hours +
# GPU-hours. That 1:1 exchange is the cluster's own: the cluster's TRESBillingWeights
# price a CPU core and a GPU identically (CPU=4.5, gres/gpu=4.5). The earlier
# CPU-hours-only plots left GPU work invisible (scGPT's eval is 116 GPU-h at ctx7).
COMPUTE_LABEL = "compute-hours (CPU core-h + GPU-h)"


def _with_compute(costs: pd.DataFrame) -> pd.DataFrame:
    out = costs.copy()
    out["gpu_hours"] = out.get("gpu_hours", 0.0)
    out["compute_hours"] = out["cpu_hours"].fillna(0) + out["gpu_hours"].fillna(0)
    return out


def _clabel(m):
    """Display label for a COST method key: normalise cost-name -> metric-name (so the
    ensemble, booked as 'fingerprint', reads 'fingerprint (ensemble)') then prettify.
    Accepts a scalar or a pandas Index/Series."""
    if hasattr(m, "map"):
        return m.map(lambda x: C.disp(COST_TO_METRIC.get(x, x)))
    return C.disp(COST_TO_METRIC.get(m, m))


OVERRIDES = Path(__file__).resolve().parents[3] / "config" / "cost_overrides.yaml"


def apply_overrides(costs: pd.DataFrame, size: str | None = None,
                    path: Path = OVERRIDES) -> pd.DataFrame:
    """Replace sacct-booked values that don't reflect the method with measured ones
    (config/cost_overrides.yaml: method -> stage -> column -> {size: value}). Each
    replaced cell is logged; rows/sizes without an override pass through untouched."""
    if not path.is_file():
        return costs
    import yaml
    ov = yaml.safe_load(path.read_text()) or {}
    out = costs.copy()
    for m, stages in ov.items():
        for st, cols in (stages or {}).items():
            for col, by_size in (cols or {}).items():
                if col not in out.columns or not isinstance(by_size, dict):
                    continue
                for sz, v in by_size.items():
                    sel = (out["method"] == m) & (out["stage"] == st)
                    if "size" in out.columns:
                        sel &= out["size"].astype(str) == str(sz)
                    elif size is not None and str(sz) != str(size):
                        continue
                    if sel.any():
                        print(f"  [override] {m}/{st}/{sz} {col}: "
                              f"{out.loc[sel, col].tolist()} -> {v}  ({path.name})")
                        out.loc[sel, col] = v
    return out


def _costs_csv() -> Path | None:
    """slurm_costs.csv lives one level above the per-size metrics dir
    (metrics/slurm_costs.csv). EXPERT_COSTS_CSV overrides; None if not found."""
    env = os.environ.get("EXPERT_COSTS_CSV")
    cands = []
    if env:
        cands.append(Path(env))
    md = os.environ.get("EXPERT_METRICS_DIR")
    if md:
        cands.append(Path(md).parent / "slurm_costs.csv")
    cands.append(Path(__file__).resolve().parents[3] / "metrics" / "slurm_costs.csv")
    for c in cands:
        if c.is_file():
            return c
    return None


def _headline(split: str = "ood_test") -> tuple[pd.DataFrame, str]:
    """Per-method headline recall@5 on `split`, cell-weighted over sub-splits.

    Returns (frame, column-name). The column follows the reported recall flavour
    (EXPERT_RECALL_MODE, as everywhere else) and falls back to the cell-pooled one for
    metrics dirs written before balanced_recall@k existed.
    """
    md = os.environ.get("EXPERT_METRICS_DIR")
    p = (Path(md) if md else Path(__file__).resolve().parents[3] / "metrics" / "small") / "none" / "all_metrics.csv"
    want = ("balanced_recall@5"
            if os.environ.get("EXPERT_RECALL_MODE", "balanced") == "balanced" else "recall@5")
    if not p.is_file():
        return pd.DataFrame(columns=["method", want]), want
    m = pd.read_csv(p)
    if want not in m.columns:
        want = "recall@5"
    m = m[m["split"].str.startswith(split)].copy()
    m["_wr"] = m[want] * m["n"]
    g = m.groupby("method").agg(_wr=("_wr", "sum"), _n=("n", "sum"))
    g[want] = np.where(g["_n"] > 0, g["_wr"] / g["_n"], np.nan)
    return g[[want]].reset_index(), want


def plot_compute(costs: pd.DataFrame):
    """Horizontal grouped bars: CPU-hours per method, one bar per stage. Log x
    (fingerprint eval ~10^5 h dwarfs genepert ~10^0)."""
    piv = costs.pivot_table(index="method", columns="stage", values="compute_hours",
                            aggfunc="sum", fill_value=0.0)
    order = piv.sum(axis=1).sort_values().index
    piv = piv.loc[order]
    piv.index = _clabel(piv.index)
    stages = [s for s in ("train", "eval", "train+eval") if s in piv.columns]
    n, h = len(piv), 0.8 / max(len(stages), 1)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * n + 1)))
    y = np.arange(n)
    for i, s in enumerate(stages):
        vals = piv[s].where(piv[s] > 0, np.nan)   # NaN -> no bar (skip stages a method lacks)
        ax.barh(y + (i - (len(stages) - 1) / 2) * h, vals, height=h,
                label=s, color=STAGE_C.get(s, "#888"))
    ax.set(yticks=y, yticklabels=piv.index, xlabel=f"{COMPUTE_LABEL}, log", xscale="log",
           title="Compute cost by method — CPU + GPU hours per stage")
    for i, s in enumerate(stages):
        for j, v in enumerate(piv[s]):
            if v > 0:
                ax.text(v, y[j] + (i - (len(stages) - 1) / 2) * h, f" {v:,.0f}",
                        va="center", ha="left", fontsize=8)
    ax.legend(title="stage", loc="lower right")
    C.save(fig, "cost_compute_by_method.svg")


def plot_eval_cost_linear(costs: pd.DataFrame):
    """Vertical bars, LINEAR y, of eval (inference) compute per method -- the actual
    magnitude the log plot hides. Each bar is CPU core-hours with GPU-hours stacked on
    top (combined 1:1, the cluster's own pricing; see COMPUTE_LABEL), so a GPU method
    is no longer drawn as if its accelerator were free. fingerprint's per-cell susie
    eval dwarfs everything, so a broken y-axis (thin top panel for the outlier, wide
    bottom panel for the rest) is used when max >> 2nd; otherwise a single axis."""
    ev = costs[costs["stage"].isin(["eval", "train+eval"])]
    g = ev.groupby("method")[["cpu_hours", "gpu_hours"]].sum()
    if g.empty:
        return
    g["total"] = g["cpu_hours"] + g["gpu_hours"]
    g = g.sort_values("total", ascending=False)
    methods, vals = list(g.index), g["total"].values
    cpu, gpu = g["cpu_hours"].values, g["gpu_hours"].values
    colors = [C.EXPERT_C if m == "ExPert" else C.FIELD_C for m in methods]
    x = np.arange(len(methods))
    mx = float(vals[0])
    rest = float(vals[1]) if len(vals) > 1 else mx
    any_gpu = bool((gpu > 0).any())
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def _bars(ax):
        ax.bar(x, cpu, color=colors, width=0.7)
        if any_gpu:
            ax.bar(x, gpu, bottom=cpu, color=GPU_C, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([_clabel(m) for m in methods], rotation=30, ha="right")

    def _legend(ax):
        if any_gpu:
            ax.legend(handles=[Patch(color=C.FIELD_C, label="CPU core-hours"),
                               Patch(color=GPU_C, label="GPU-hours")],
                      loc="upper right", fontsize=9)

    def _label(ax, i, v):
        txt = f"{v:,.1f}" + (f"\n({gpu[i]:,.0f} GPU-h)" if gpu[i] > 0 else "")
        ax.text(i, v, txt, ha="center", va="bottom", fontsize=9,
                fontweight="bold" if methods[i] == "ExPert" else "normal")

    title = "Eval (inference) cost by method — CPU + GPU hours, linear"
    if mx > 3 * max(rest, 1e-9) and len(vals) > 1:
        fig, (axt, axb) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                       gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.06})
        _bars(axt); _bars(axb)
        top_lo = mx * 0.985                       # thin band around the outlier
        axt.set_ylim(top_lo, mx * 1.03)
        axb.set_ylim(0, rest * 1.35)              # the rest, at readable scale
        # the panels sit 0.06 apart, so the top panel's lowest tick label and the
        # bottom panel's highest ran into each other: drop the tick on each side
        from matplotlib.ticker import MaxNLocator
        axt.yaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))
        axb.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))
        axt.spines["bottom"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axt.tick_params(labelbottom=False, bottom=False)
        d = .012                                  # diagonal break marks
        kw = dict(color="k", clip_on=False, lw=1)
        axt.plot((-d, d), (-d * 2.4, d * 2.4), transform=axt.transAxes, **kw)
        axt.plot((1 - d, 1 + d), (-d * 2.4, d * 2.4), transform=axt.transAxes, **kw)
        axb.plot((-d, d), (1 - d, 1 + d), transform=axb.transAxes, **kw)
        axb.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axb.transAxes, **kw)
        for i, v in enumerate(vals):
            _label(axt if v >= top_lo else axb, i, v)
        _legend(axb)
        axt.set_title(title.replace(", linear", ", linear (axis broken)"))
        fig.supylabel(f"eval {COMPUTE_LABEL}")
        C.save(fig, "cost_eval_linear.svg")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        _bars(ax)
        for i, v in enumerate(vals):
            _label(ax, i, v)
        _legend(ax)
        ax.set(ylabel=f"eval {COMPUTE_LABEL}", title=title)
        C.save(fig, "cost_eval_linear.svg")

def plot_gpu(costs: pd.DataFrame):
    """GPU-hours per method (only methods that used a GPU)."""
    g = costs.groupby("method")["gpu_hours"].sum()
    g = g[g > 0].sort_values()
    if g.empty:
        print("  (no GPU-hours recorded — skipping GPU plot)")
        return
    g.index = _clabel(g.index)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(g) + 1)))
    ax.barh(g.index, g.values, color="#76b7b2")
    for i, v in enumerate(g.values):
        ax.text(v, i, f" {v:,.1f}", va="center", fontsize=9)
    ax.set(xlabel="GPU-hours", title="GPU cost by method")
    C.save(fig, "cost_gpu_by_method.svg")


def plot_ram(costs: pd.DataFrame):
    """Peak per-job RAM (GB) per method x stage."""
    piv = costs.pivot_table(index="method", columns="stage", values="max_ram_GB",
                            aggfunc="max", fill_value=0.0)
    order = piv.max(axis=1).sort_values().index
    piv = piv.loc[order]
    piv.index = _clabel(piv.index)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(piv) + 1)))
    piv.plot.barh(ax=ax, color=[STAGE_C.get(c, "#888") for c in piv.columns])
    ax.set(xlabel="peak per-job RAM (GB)", title="Peak memory footprint by method x stage")
    ax.legend(title="stage", loc="lower right")
    C.save(fig, "cost_peak_ram.svg")


def plot_cost_vs_perf(costs: pd.DataFrame):
    """EVAL compute-hours (x) vs headline OOD recall@5 (y), one point per method.

    Eval-only, not train+eval: inference cost is what a user pays per new dataset,
    whereas training is amortised. Three methods (genepert, xgboost, linreg) run train
    and eval in ONE job and so only report a combined figure; they are plotted as hollow
    markers because their x is an UPPER BOUND on eval cost, not a measurement.

    x uses the same BROKEN LINEAR scale as plot_eval_cost_linear rather than a log axis:
    fingerprint's per-cell susie eval is ~10x the next method, and a log axis makes that
    gap look like a modest step instead of the order of magnitude it is.
    """
    ev = costs[costs["stage"].astype(str).str.contains("eval")]
    if ev.empty:
        print("  (no eval-stage rows — skipping cost-vs-perf plot)")
        return
    combined = set(ev.loc[ev["stage"] == "train+eval", "method"])
    tot = ev.groupby("method")[["cpu_hours", "gpu_hours", "compute_hours"]].sum()
    tot["compute_h"] = tot["compute_hours"]
    tot = tot.reset_index()
    # .map(dict) yields NaN for unmapped keys and the dropna below then deletes them --
    # that silently dropped xgboost/linreg (whose cost name already equals their metric
    # name) from this plot. Identity fallback, like _clabel already does.
    tot["metric_method"] = tot["method"].map(lambda x: COST_TO_METRIC.get(x, x))
    hl, MCOL = _headline("ood_test")
    df = tot.merge(hl, left_on="metric_method", right_on="method", how="left",
                   suffixes=("", "_m")).dropna(subset=[MCOL])
    if df.empty:
        print("  (no cost<->score overlap — skipping cost-vs-perf plot)")
        return
    import matplotlib.pyplot as plt

    is_exp = df["method"] == "ExPert"
    is_ub = df["method"].isin(combined)
    ylab = f"OOD test recall@5{' (balanced)' if MCOL.startswith('balanced_') else ''}"

    def _pts(ax):
        for mask, kw in (((~is_exp) & (~is_ub), dict(s=90, color=C.FIELD_C, zorder=3)),
                         ((~is_exp) & is_ub, dict(s=90, facecolor="none",
                                                  edgecolor=C.FIELD_C, linewidth=1.6, zorder=3)),
                         (is_exp, dict(s=170, color=C.EXPERT_C, edgecolor="white",
                                       linewidth=1.2, zorder=4))):
            if mask.any():
                ax.scatter(df.loc[mask, "compute_h"], df.loc[mask, MCOL], **kw)

    # De-collide labels PER PANEL. Clustering has to use each panel's own x-range: the
    # broken axis means a single global span is dominated by the outlier, which lumped
    # every cheap method into one cluster and pushed cellot's label off the axes.
    yspan = max(df[MCOL].max() - df[MCOL].min(), 1e-9)

    def _labels(ax, lo, hi):
        sub = df[(df["compute_h"] >= lo) & (df["compute_h"] <= hi)].copy()
        if sub.empty:
            return
        xlo, xhi = ax.get_xlim()
        span = max(xhi - xlo, 1e-9)
        sub["_nx"] = (sub["compute_h"] - xlo) / span
        sub["_ny"] = sub[MCOL] / yspan
        sub = sub.sort_values(["_nx", "_ny"])
        clusters: list[list] = []
        for i, r in sub.iterrows():
            for cl in clusters:
                j = cl[0]
                if (abs(r["_nx"] - sub.loc[j, "_nx"]) < 0.08
                        and abs(r["_ny"] - sub.loc[j, "_ny"]) < 0.07):
                    cl.append(i)
                    break
            else:
                clusters.append([i])
        # alternate up/down around the point so a 3-way cluster spreads
        # +9 / -9 / -25 rather than 4 / -9 / -22 (which still overlapped)
        dy = {i: (9 if k == 0 else 7 - 16 * k)
              for cl in clusters for k, i in enumerate(cl)}
        for i, r in sub.iterrows():
            lab = _clabel(r["method"]) + ("*" if r["method"] in combined else "")
            # long labels near a panel's right edge would run past it -> flip them left
            right = r["_nx"] > 0.62
            ax.annotate(lab, (r["compute_h"], r[MCOL]),
                        xytext=(-7 if right else 7, dy[i]), textcoords="offset points",
                        ha="right" if right else "left",
                        fontsize=11 if r["method"] == "ExPert" else 10,
                        fontweight="bold" if r["method"] == "ExPert" else "normal")

    vals = df["compute_h"].sort_values(ascending=False).values
    mx = float(vals[0])
    rest = float(vals[1]) if len(vals) > 1 else mx
    broken = mx > 3 * max(rest, 1e-9) and len(vals) > 1

    if broken:
        fig, (axl, axr) = plt.subplots(1, 2, sharey=True, figsize=(9, 5.5),
                                       gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.06})
        _pts(axl); _pts(axr)
        axl.set_xlim(0, rest * 1.35)              # the bulk, at readable scale
        axr.set_xlim(mx * 0.97, mx * 1.06)        # thin band around the outlier
        # same break-gap collision as cost_eval_linear, horizontally
        from matplotlib.ticker import MaxNLocator
        axl.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        axr.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))
        _labels(axl, -np.inf, rest * 1.35)
        _labels(axr, rest * 1.35, np.inf)
        axl.spines["right"].set_visible(False)
        axr.spines["left"].set_visible(False)
        axr.tick_params(labelleft=False, left=False)
        dd = .012                                 # diagonal break marks
        kw = dict(color="k", clip_on=False, lw=1)
        axl.plot((1 - dd / 2.4, 1 + dd / 2.4), (-dd, dd), transform=axl.transAxes, **kw)
        axl.plot((1 - dd / 2.4, 1 + dd / 2.4), (1 - dd, 1 + dd), transform=axl.transAxes, **kw)
        axr.plot((-dd, dd), (-dd, dd), transform=axr.transAxes, **kw)
        axr.plot((-dd, dd), (1 - dd, 1 + dd), transform=axr.transAxes, **kw)
        for ax in (axl, axr):
            ax.grid(axis="y", alpha=0.3)
        if is_exp.any():
            for ax in (axl, axr):
                ax.axhline(df.loc[is_exp, MCOL].iloc[0], ls="--", color=C.EXPERT_C,
                           lw=1.0, alpha=0.5, zorder=1)
        axl.set_ylabel(ylab)
        fig.supxlabel("eval compute-hours (CPU+GPU, linear — axis broken)")
        axl.set_title("Cost vs performance — bang for buck", loc="left")
        tgt = axl
    else:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _pts(ax); _labels(ax, -np.inf, np.inf)
        if is_exp.any():
            ax.axhline(df.loc[is_exp, MCOL].iloc[0], ls="--", color=C.EXPERT_C, lw=1.0, alpha=0.5)
        ax.set(xlabel="eval compute-hours (CPU+GPU)", ylabel=ylab,
               title="Cost vs performance — bang for buck")
        tgt = ax
    if is_ub.any():
        tgt.text(0.99, 0.90, "* train+eval in one job — x is an upper bound on eval cost",
                 transform=tgt.transAxes, ha="right", va="top", fontsize=8, color="0.45")
    C.save(fig, "cost_vs_performance.svg")


def plot_util(costs: pd.DataFrame):
    """CPU utilisation (TotalCPU / allocated core-seconds) per method x stage — how
    much of the reserved cores were actually busy. Low bars = over-allocated / serial
    work (right-sizing target). Skipped if the column is absent (older CSV)."""
    if "cpu_util_pct" not in costs.columns:
        return
    piv = costs.pivot_table(index="method", columns="stage", values="cpu_util_pct",
                            aggfunc="mean", fill_value=np.nan)
    piv = piv.loc[piv.max(axis=1).sort_values().index]
    piv.index = _clabel(piv.index)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(piv) + 1)))
    piv.plot.barh(ax=ax, color=[STAGE_C.get(c, "#888") for c in piv.columns])
    ax.axvline(100, ls=":", color="#444", lw=1)
    ax.set(xlabel="CPU utilisation (%)", xlim=(0, max(105, np.nanmax(piv.values) + 5)),
           title="CPU utilisation by method x stage (reserved cores actually used)")
    ax.legend(title="stage", loc="lower right")
    C.save(fig, "cost_cpu_utilisation.svg")


def write_report(costs: pd.DataFrame, size):
    """A compact markdown cost table (+ totals) beside the analysis REPORT.md."""
    cols = [c for c in ["method", "stage", "n_jobs", "cpu_hours", "gpu_hours",
                        "wall_hours", "cpu_util_pct", "max_ram_GB"] if c in costs.columns]
    tbl = costs[cols].sort_values(["method", "stage"]).copy()
    tbl["method"] = _clabel(tbl["method"])
    tot = {"method": "TOTAL", "stage": "",
           "n_jobs": int(costs["n_jobs"].sum()) if "n_jobs" in costs else "",
           "cpu_hours": round(costs["cpu_hours"].sum(), 1) if "cpu_hours" in costs else "",
           "gpu_hours": round(costs["gpu_hours"].sum(), 1) if "gpu_hours" in costs else ""}
    tbl = pd.concat([tbl, pd.DataFrame([tot])], ignore_index=True)
    p = C.OUT.parent / "COST.md"
    lines = [f"# Resource cost — size `{size or 'all'}`  (COMPLETED jobs only)", "",
             "CPU/GPU-hours = sum(elapsed x allocated units); wall_hours = stage span;",
             "cpu_util_pct = TotalCPU / allocated core-seconds (low = over-reserved cores).",
             "ExPert is trained outside this pipeline, so it has no cost row.", ""]
    try:
        lines.append(tbl.to_markdown(index=False))
    except Exception:
        lines.append(tbl.to_string(index=False))
    p.write_text("\n".join(lines) + "\n")
    print(f"  wrote {p.relative_to(C.OUT.parent)}")


def main() -> int:
    csv = _costs_csv()
    if csv is None:
        print("cost.py: no slurm_costs.csv found (hook may not have run) — skipping.")
        return 0
    costs = pd.read_csv(csv)
    # Scope to THIS collector's size (the metrics-dir basename); the CSV is global
    # and size-keyed, so small's collect shows small's costs. Fall back to all rows
    # if there's no size column (older CSV) or no match for this size.
    import re
    name = Path(os.environ.get("EXPERT_METRICS_DIR", "")).name
    size = os.environ.get("EXPERT_SIZE")
    if not size and "size" in costs.columns:
        # the size is part of the dir name (metrics/ctx7, metrics_ctx7_preview, ...):
        # match it rather than the whole name, which left a preview dir unfiltered
        known = sorted(costs["size"].dropna().astype(str).unique(), key=len, reverse=True)
        size = next((k for k in known if re.search(rf"(^|[^a-z0-9]){re.escape(k)}($|[^a-z0-9])", name)), None)
    if "size" in costs.columns and size and (costs["size"].astype(str) == size).any():
        costs = costs[costs["size"].astype(str) == size].copy()
        print(f"cost.py: {csv} — {len(costs)} rows for size={size}")
    elif "size" in costs.columns and costs["size"].nunique() > 1:
        # summing small + ctx7 + full rows into one bar per method is meaningless
        print(f"cost.py: {csv} has sizes {sorted(costs['size'].unique())} and none matches "
              f"'{name}' -- set EXPERT_SIZE. Skipping cost plots rather than mixing sizes.")
        return 0
    else:
        print(f"cost.py: {csv} — {len(costs)} rows (single size)")
    costs = _with_compute(apply_overrides(costs, size))
    C.dump(costs.sort_values(["method", "stage"]), "cost_by_method_stage.csv")
    plot_compute(costs)
    plot_eval_cost_linear(costs)
    plot_gpu(costs)
    plot_ram(costs)
    plot_util(costs)
    plot_cost_vs_perf(costs)
    write_report(costs, size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
