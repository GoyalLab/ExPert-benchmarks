"""Shared plotting style + small helpers for the analysis scripts."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# OUT (where plots/CSVs land) is env-overridable so `collect` can point it at
# metrics/<size>/analysis/out; default keeps the standalone `analysis/out` layout.
OUT = Path(os.environ.get("EXPERT_ANALYSIS_OUT") or (Path(__file__).resolve().parent / "out"))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    # moderate default fonts (the three large figures — radar, competitor, recovery —
    # pin their own larger explicit sizes; these defaults keep the smaller plots tidy)
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "figure.titlesize": 14,
})

# stable colour for the subject method vs the field
EXPERT_C = "#d1495b"
FIELD_C = "#6c757d"


def _drawn_ticklabels(axis):
    try:
        ticks = axis._update_ticks()              # exactly the ticks that get drawn
    except Exception:
        ticks = axis.get_major_ticks()
    return [t for tk in ticks for t in (tk.label1, tk.label2)
            if t.get_visible() and t.get_text().strip()]


def _wrap_lines(text: str, width: int) -> str:
    import textwrap
    return "\n".join(w for line in text.split("\n")
                     for w in (textwrap.wrap(line, max(8, width)) or [""]))


def fit_titles(fig) -> bool:
    """Wrap axes titles wider than their axes (and the suptitle wider than the
    figure). A title that overhangs its axes lands on the neighbour's tick labels or
    on its own y label -- several per-panel titles here run ~60 characters."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    changed = False
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        room = ax.get_window_extent(r).width
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is None or not t.get_text().strip():
                continue
            w = t.get_window_extent(r).width
            if w > room * 1.02:
                longest = max(len(x) for x in t.get_text().split("\n"))
                t.set_text(_wrap_lines(t.get_text(), int(longest * room / w * 0.95)))
                changed = True
    st = getattr(fig, "_suptitle", None)
    if st is not None and st.get_text().strip():
        w, room = st.get_window_extent(r).width, fig.bbox.width
        if w > room * 0.98:
            longest = max(len(x) for x in st.get_text().split("\n"))
            st.set_text(_wrap_lines(st.get_text(), int(longest * room / w * 0.95)))
            changed = True
    return changed


def make_room_for_suptitle(fig, gap_pt: float = 6.0) -> None:
    """Lower the subplot grid until no axes (incl. its title) reaches the suptitle.
    A suptitle that wraps to two lines grows DOWN into the panel titles; figures that
    never run a layout pass keep that collision. Only for figures without a layout
    engine -- constrained/tight engines re-solve this themselves at draw time."""
    st = getattr(fig, "_suptitle", None)
    if st is None or not st.get_text().strip() or fig.get_layout_engine() is not None:
        return
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        floor = st.get_window_extent(r).y0 - gap_pt * fig.dpi / 72.0
        tops = [ax.get_tightbbox(r).y1 for ax in fig.axes if ax.get_visible()]
        over = max(tops) - floor if tops else 0.0
        if over <= 0:
            return
        top = fig.subplotpars.top - over / fig.bbox.height
        if top <= fig.subplotpars.bottom + 0.2:            # would crush the panels
            return
        fig.subplots_adjust(top=top)


def place_legends(fig) -> None:
    """For each legend drawn INSIDE its axes, pick the standard position that covers
    the fewest data points and point labels. A fixed loc= (e.g. "lower right") can
    sit on exactly the corner where labelled points cluster, leaving those labels
    nowhere to go. Legends placed outside the axes are left alone."""
    import numpy as np
    from matplotlib.text import Text as _TX
    locs = ["upper left", "upper right", "lower left", "lower right",
            "center left", "center right", "upper center", "lower center"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not ax.get_visible():
            continue
        axb, lb = ax.get_window_extent(r), leg.get_window_extent(r)
        if not (lb.x0 >= axb.x0 - 1 and lb.x1 <= axb.x1 + 1
                and lb.y0 >= axb.y0 - 1 and lb.y1 <= axb.y1 + 1):
            continue                                      # outside the axes: deliberate
        pts = [ax.transData.transform(c.get_offsets()) for c in ax.collections
               if hasattr(c, "get_offsets") and len(c.get_offsets())]
        pts = np.vstack(pts) if pts else np.empty((0, 2))
        labels = [_TX.get_window_extent(t, r) for t in ax.texts
                  if t.get_visible() and t.get_text().strip()]

        # bars / filled patches count too: scoring only points and labels, a bar chart
        # looked empty everywhere and the legend landed on its tallest bar
        # rendered extent, not get_width(): boxplots/violins/fills are PathPatches
        patches = [b for b in (pa.get_window_extent(r) for pa in ax.patches if pa.get_visible())
                   if b.width > 0 and b.height > 0]

        def cost(bb):
            inside = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1)
                      & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum() if len(pts) else 0
            hit = sum(1 for t in labels if min(t.x1, bb.x1) > max(t.x0, bb.x0)
                      and min(t.y1, bb.y1) > max(t.y0, bb.y0))
            area = max(bb.width * bb.height, 1.0)
            covered = sum(max(0.0, min(pa.x1, bb.x1) - max(pa.x0, bb.x0))
                          * max(0.0, min(pa.y1, bb.y1) - max(pa.y0, bb.y0)) for pa in patches)
            return 3 * hit + inside + 20.0 * covered / area   # fraction of legend over bars
        best, best_cost = None, None
        for loc in locs:
            leg.set_loc(loc)
            fig.canvas.draw()
            c = cost(leg.get_window_extent(r))
            if best_cost is None or c < best_cost:
                best, best_cost = loc, c
        leg.set_loc(best)


def declutter_tick_corners(fig, rounds: int = 3) -> None:
    """Where an axes' x tick label and y tick label meet at a corner (small panels
    with their own y range put a tick right at the corner), push that axes' x tick
    labels down by just the overlap. Nothing is hidden or moved sideways."""
    ppp = fig.dpi / 72.0
    for _ in range(rounds):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        moved = False
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            xs = [t.get_window_extent(r) for t in _drawn_ticklabels(ax.xaxis)]
            ys = [t.get_window_extent(r) for t in _drawn_ticklabels(ax.yaxis)]
            need = 0.0
            for bx in xs:
                for by in ys:
                    ox = min(bx.x1, by.x1) - max(bx.x0, by.x0)
                    oy = min(bx.y1, by.y1) - max(bx.y0, by.y0)
                    if ox > 0 and oy > 0:
                        need = max(need, oy)
            if need > 0:
                pad = ax.xaxis.get_major_ticks()[0].get_pad() if ax.xaxis.get_major_ticks() else 3.5
                ax.tick_params(axis="x", pad=pad + need / ppp + 1.0)
                moved = True
        if not moved:
            return


def repel_annotations(fig, max_iter: int = 400, pad_px: float = 2.0,
                      leader_pt: float = 9.0) -> int:
    """Push apart colliding point labels, then return how many still collide.

    Acts on every `ax.annotate(..., textcoords="offset points")` in the figure -- the
    form all point labels in this suite use -- by changing only the OFFSET, so each
    label stays tied to its point. Pairs are separated along their axis of least
    overlap (usually vertically, which keeps a label beside its point). A label that
    ends up more than `leader_pt` from its point gets a thin leader line back to it.
    Written here because adjustText is not in the collect env.
    """
    import math
    from matplotlib.text import Annotation, Text as _TX       # _TX: text-only extent
    from matplotlib.patches import FancyArrowPatch
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ppp = fig.dpi / 72.0                               # display px per point
    by_ax: dict = {}
    for a in fig.findobj(Annotation):
        if (a.get_visible() and a.get_text().strip() and a.axes is not None
                and a.anncoords == "offset points" and a.arrowprops is None):
            by_ax.setdefault(id(a.axes), []).append(a)
    remaining = 0
    for anns in by_ax.values():
        ax = anns[0].axes
        mine = {id(a) for a in anns}
        obst = [t for t in (ax.title, getattr(ax, "_left_title", None),
                            getattr(ax, "_right_title", None), ax.xaxis.label, ax.yaxis.label,
                            *_drawn_ticklabels(ax.xaxis), *_drawn_ticklabels(ax.yaxis),
                            *[t for t in ax.texts if id(t) not in mine])
                if t is not None and t.get_visible() and t.get_text().strip()]
        obst_boxes = [t.get_window_extent(r) for t in obst]
        if ax.get_legend() is not None:
            obst_boxes.append(ax.get_legend().get_window_extent(r))
        start = {id(a): tuple(a.xyann) for a in anns}
        for _ in range(max_iter):
            boxes = [_TX.get_window_extent(a, r) for a in anns]
            moved = False
            for i in range(len(anns)):                  # fixed obstacles: only the label moves
                for ob in obst_boxes:
                    bi = boxes[i]
                    ox = min(bi.x1, ob.x1) - max(bi.x0, ob.x0)
                    oy = min(bi.y1, ob.y1) - max(bi.y0, ob.y0)
                    if ox <= -pad_px or oy <= -pad_px:
                        continue
                    moved = True
                    xi, yi = anns[i].xyann
                    if oy <= ox:
                        sgn = 1.0 if (bi.y0 + bi.y1) >= (ob.y0 + ob.y1) else -1.0
                        anns[i].xyann = (xi, yi + sgn * ((oy + pad_px) / ppp + 0.25))
                    else:
                        sgn = 1.0 if (bi.x0 + bi.x1) >= (ob.x0 + ob.x1) else -1.0
                        anns[i].xyann = (xi + sgn * ((ox + pad_px) / ppp + 0.25), yi)
                    boxes[i] = _TX.get_window_extent(anns[i], r)
            for i in range(len(anns)):
                for j in range(i + 1, len(anns)):
                    bi, bj = boxes[i], boxes[j]
                    ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                    oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                    if ox <= -pad_px or oy <= -pad_px:
                        continue
                    moved = True
                    if oy <= ox:                        # separate vertically
                        sgn = 1.0 if (bi.y0 + bi.y1) >= (bj.y0 + bj.y1) else -1.0
                        d = (oy + pad_px) / 2 / ppp + 0.25
                        xi, yi = anns[i].xyann; anns[i].xyann = (xi, yi + sgn * d)
                        xj, yj = anns[j].xyann; anns[j].xyann = (xj, yj - sgn * d)
                    else:                               # separate horizontally
                        sgn = 1.0 if (bi.x0 + bi.x1) >= (bj.x0 + bj.x1) else -1.0
                        d = (ox + pad_px) / 2 / ppp + 0.25
                        xi, yi = anns[i].xyann; anns[i].xyann = (xi + sgn * d, yi)
                        xj, yj = anns[j].xyann; anns[j].xyann = (xj - sgn * d, yj)
                    boxes[i] = _TX.get_window_extent(anns[i], r)
                    boxes[j] = _TX.get_window_extent(anns[j], r)
            # keep every label inside its own axes: near a panel edge the push-apart
            # otherwise walked labels across the gap onto the NEXT panel's labels
            axb = ax.get_window_extent(r)
            for i, a in enumerate(anns):
                b = _TX.get_window_extent(a, r)
                dx = (max(0.0, axb.x0 - b.x0) - max(0.0, b.x1 - axb.x1)) / ppp
                dy = (max(0.0, axb.y0 - b.y0) - max(0.0, b.y1 - axb.y1)) / ppp
                if dx or dy:
                    xi, yi = a.xyann
                    a.xyann = (xi + dx, yi + dy)
            if not moved:
                break
        boxes = [_TX.get_window_extent(a, r) for a in anns]
        remaining += sum(
            1 for i in range(len(anns)) for j in range(i + 1, len(anns))
            if min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0) > 0
            and min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0) > 0)
        for a in anns:                                  # leader for displaced labels
            (x0, y0), (x1, y1) = start[id(a)], a.xyann
            if math.hypot(x1 - x0, y1 - y0) > leader_pt:
                try:
                    props = dict(arrowstyle="-", color="0.55", lw=0.6,
                                 shrinkA=0, shrinkB=2)
                    a.arrowprops = props
                    a._arrow_relpos = (0.5, 0.5)
                    a.arrow_patch = FancyArrowPatch((0, 0), (1, 1), **props)
                except Exception:
                    pass
    return remaining


def save(fig, name: str, png_dpi: int = 300):
    """Save as SVG (vector) and a high-res PNG with the same stem. Point labels are
    de-collided first (repel_annotations)."""
    fit_titles(fig)
    make_room_for_suptitle(fig)
    place_legends(fig)
    declutter_tick_corners(fig)
    left = repel_annotations(fig)
    if left:
        print(f"  NOTE {name}: {left} point-label pair(s) still touching after repel")
    svg = OUT / name
    fig.savefig(svg)
    png = svg.with_suffix(".png")
    fig.savefig(png, dpi=png_dpi)
    plt.close(fig)
    print(f"  wrote {svg.relative_to(OUT.parent)} + {png.name} ({png_dpi} dpi)")


def dump(df: pd.DataFrame, name: str, index: bool = False):
    p = OUT / name
    df.to_csv(p, index=index)
    print(f"  wrote {p.relative_to(OUT.parent)} ({len(df)} rows)")
    return df


# Display names for plots ONLY — data, paths, CSV keys keep the raw method names.
DISPLAY = {
    "fingerprint_perdataset": "fingerprint (ensemble)",
    "fingerprint_merged": "fingerprint (merged)",
    "baseline_prior": "baseline (prior)",
    "baseline_random": "baseline (random)",
    "baseline_neg_target": "baseline (neg target)",
    # plotted centroid baseline = the target-masked one; see load.PLOT_HIDE
    "baseline_centroid_notarget": "baseline (centroid)",
    "baseline_centroid": "baseline (centroid, incl. target)",
    "xgboost": "XGBoost",
    "xgboost_capped": "XGBoost",
    "linreg": "linreg (SGD)",          # was only in collect's map; the two had drifted
}


def disp(method):
    """Pretty label for a method key (identity for unmapped). Accepts a scalar or a
    pandas Index/Series (element-wise), so plotters can `idx.map(C.disp)`."""
    if hasattr(method, "map"):
        return method.map(lambda m: DISPLAY.get(m, m))
    return DISPLAY.get(method, method)
