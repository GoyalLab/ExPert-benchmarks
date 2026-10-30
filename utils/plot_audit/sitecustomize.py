"""Opt-in overlap audit for every matplotlib figure a process saves.

Put this directory on PYTHONPATH and set EXPERT_PLOT_AUDIT=<report.jsonl>; Python
imports `sitecustomize` at start-up, so collect_results.py and the analysis scripts
are audited without touching their code. Before each savefig the figure is drawn and
every visible text element (titles, axis labels, tick labels, legend entries,
annotations) plus each legend frame is boxed in display pixels; any two that
intersect by more than PAD px in both directions are reported, one JSON line per
saved file. Unset the variable and this module does nothing.
"""
import os

_OUT = os.environ.get("EXPERT_PLOT_AUDIT")
PAD = 1.0          # px of slack, so boxes that merely touch are not reported

if _OUT:
    import json

    def _drawn_ticklabels(fig):
        """ids of tick labels matplotlib will actually DRAW. Axis objects keep Tick
        Texts for locations outside the view interval (e.g. a '-0.2' on an axis that
        starts at 0) and on hidden twin axes; those report get_visible() True but are
        never rendered, and counting them produced phantom tick collisions."""
        drawn = set()
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if not axis.get_visible():
                    continue
                try:
                    ticks = axis._update_ticks()          # exactly the ticks that get drawn
                except Exception:
                    ticks = axis.get_major_ticks()
                for tk in ticks:
                    for lab in (tk.label1, tk.label2):
                        if lab.get_visible():
                            drawn.add(id(lab))
        return drawn

    def _roles(fig):
        roles = {}
        for ax in fig.axes:
            for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
                if t is not None:
                    roles[id(t)] = "axes-title"
            roles[id(ax.xaxis.label)] = "xlabel"
            roles[id(ax.yaxis.label)] = "ylabel"
            for t in ax.get_xticklabels():
                roles[id(t)] = "xtick"
            for t in ax.get_yticklabels():
                roles[id(t)] = "ytick"
        if getattr(fig, "_suptitle", None) is not None:
            roles[id(fig._suptitle)] = "suptitle"
        for leg in fig.findobj(lambda a: a.__class__.__name__ == "Legend"):
            for t in leg.get_texts():
                roles[id(t)] = "legend"
            roles[id(leg.get_title())] = "legend-title"
        return roles

    def _quad(t, bb, r):
        """Corners of the text's ROTATED glyph block, in display px. The axis-aligned
        window extent of a diagonal label is far larger than the label, so AABB tests
        report neighbouring rotated tick labels as colliding when they do not."""
        import numpy as np
        ang = (t.get_rotation() or 0.0) % 180.0
        cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
        if ang in (0.0, 90.0):
            return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
        # unrotated size from an upright copy (Text._get_layout returns the ROTATED
        # extent, which made widely spaced 45-degree labels register as colliding)
        from matplotlib.text import Text as _T
        u = _T(text=t.get_text(), fontproperties=t.get_fontproperties(),
               linespacing=getattr(t, "_linespacing", 1.2))
        u.set_figure(t.figure)
        ub = u.get_window_extent(r)
        w, h = ub.width, ub.height
        a = np.deg2rad(ang)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        return c @ R.T + [cx, cy]

    def _sat(p, q, pad):
        """True if convex quads p, q overlap by more than `pad` along every axis."""
        import numpy as np
        for poly in (p, q):
            for k in range(4):
                e = poly[(k + 1) % 4] - poly[k]
                n = np.array([-e[1], e[0]])
                nn = np.hypot(*n)
                if nn == 0:
                    continue
                n = n / nn
                pp, qq = p @ n, q @ n
                if min(pp.max(), qq.max()) - max(pp.min(), qq.min()) <= pad:
                    return False
        return True

    def _audit(fig, fname):
        import numpy as np
        from matplotlib.text import Text
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        roles = _roles(fig)
        drawn_ticks = _drawn_ticklabels(fig)
        legends = fig.findobj(lambda a: a.__class__.__name__ == "Legend")
        owner = {}
        for leg in legends:
            for t in list(leg.get_texts()) + [leg.get_title()]:
                owner[id(t)] = id(leg)
        items = []                           # (label, role, owner-legend-id, bbox, quad)
        for t in fig.findobj(Text):
            s = t.get_text().strip()
            if not s or not t.get_visible() or (t.axes is not None and not t.axes.get_visible()):
                continue
            if roles.get(id(t)) in ("xtick", "ytick") and id(t) not in drawn_ticks:
                continue                                  # created but never rendered
            try:
                # Text-only extent: an Annotation's own get_window_extent also covers its
                # leader line, so a diagonal leader counted as a "text overlap"
                bb = Text.get_window_extent(t, r)
            except Exception:
                continue
            try:
                q = _quad(t, bb, r)
            except Exception:
                # never drop a text silently -- fall back to its plain box, which can
                # only OVER-report. (A failing _quad once hid every rotated label.)
                q = np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
            if bb.width < 1 or bb.height < 1:
                continue
            items.append((s[:40], roles.get(id(t), "text"), owner.get(id(t)), bb, q))
        for leg in legends:
            if leg.get_visible():
                bb = leg.get_window_extent(r)
                q = np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
                items.append(("<legend frame>", "legend-frame", id(leg), bb, q))
        if len(items) < 2:
            return {"file": str(fname), "n_text": len(items), "overlaps": 0, "examples": []}
        b = np.array([[i[3].x0, i[3].y0, i[3].x1, i[3].y1] for i in items])
        ox = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
        oy = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
        cand = (ox > PAD) & (oy > PAD)                # cheap AABB pre-filter
        own = np.array([i[2] if i[2] is not None else -1 for i in items])
        cand &= ~((own[:, None] == own[None, :]) & (own[:, None] != -1))
        cand = np.triu(cand, 1)
        pairs = [(a, c) for a, c in np.argwhere(cand) if _sat(items[a][4], items[c][4], PAD)]
        kinds = {}
        for a, c in pairs:
            k = " x ".join(sorted((items[a][1], items[c][1])))
            kinds[k] = kinds.get(k, 0) + 1
        ex = [f"{items[a][1]}:'{items[a][0]}' x {items[c][1]}:'{items[c][0]}'" for a, c in pairs[:6]]
        return {"file": str(fname), "n_text": len(items), "overlaps": int(len(pairs)),
                "by_kind": kinds, "examples": ex}

    def _install():
        from matplotlib.figure import Figure
        orig = Figure.savefig
        done = set()

        def savefig(self, fname, *args, **kwargs):
            key = (id(self), os.path.splitext(str(fname))[0])   # png+svg of one figure: audit once
            if key not in done:
                done.add(key)
                try:
                    rec = _audit(self, fname)
                except Exception as e:                            # never break the caller
                    rec = {"file": str(fname), "error": repr(e)}
                with open(_OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            return orig(self, fname, *args, **kwargs)

        Figure.savefig = savefig

    try:
        _install()
    except Exception:
        pass
