#!/usr/bin/env python3
"""Re-render the weighted-metrics figures from a metrics dir's CACHED csvs --
no 30-minute collect rerun.

    python utils/replot_weighted.py metrics/small_preview

Writes, into <metrics_dir>/plots/:
    weighted_metrics.<ext>              id_test vs ood_test        (unchanged view)
    ID_weighted_metrics.<ext>           id_test  by shift type
    OOD_weighted_metrics.<ext>          ood_test by shift type
    *_weighted_metrics_f1_r5_r10.<ext>  the same three panels, F1 / r@5 / r@10 only
    efficiency_sweep.<ext>              metric vs min-efficiency threshold
    efficiency_sweep_f1_r5_r10.<ext>    the same, F1 / r@5 / r@10 only

The recall flavour follows --recall-mode / EXPERT_RECALL_MODE, exactly as collect
does, so these match the figures a full run would emit.

Imports collect_results, so the plot function, the shift classifier AND the
palette are the pipeline's own code -- this only skips the scoring pass that
already wrote weighted_metrics.csv. The palette is rebuilt from
sweep_metrics.csv exactly as main() does (_rank_methods -> _method_palette),
so method colours and ordering match every other figure.
"""
import argparse, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflow" / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_dir", type=Path)
    ap.add_argument("--ext", default="svg", choices=["svg", "png", "pdf"])
    ap.add_argument("--recall-mode", default=os.environ.get("EXPERT_RECALL_MODE", "balanced"),
                    choices=("balanced", "cells"))
    a = ap.parse_args()

    import pandas as pd
    import collect_results as C

    wcsv = a.metrics_dir / "weighted_metrics.csv"
    scsv = a.metrics_dir / "sweep_metrics.csv"
    for f in (wcsv, scsv):
        if not f.exists():
            print(f"ERROR: {f} not found", file=sys.stderr); return 1
    w = pd.read_csv(wcsv)
    sweep = pd.read_csv(scsv)
    print(f"  weighted rows: {len(w):,}   sweep rows: {len(sweep):,}")

    # identical palette/order to main()
    ranked = C._rank_methods(sweep)
    baselines = tuple(m for m in ranked if "baseline" in m.lower())
    palette = C._method_palette(ranked, baselines=baselines)
    print(f"  method order: {', '.join(ranked)}")

    out_dir = a.metrics_dir / "plots"; out_dir.mkdir(parents=True, exist_ok=True)
    RP, WRP = C._recall_prefixes(a.recall_mode)
    print(f"  recall flavour: {a.recall_mode}  ({RP} / {WRP})")
    w_trio = ["weighted_macro_f1", f"{WRP}5", f"{WRP}10"]
    s_trio = ["macro_f1", f"{RP}5", f"{RP}10"]

    top = w.copy()
    top["split"] = top["split"].astype(str).str.split(":", n=1).str[0]
    C.plot_weighted_metrics(top, out_dir / f"weighted_metrics.{a.ext}",
                            palette=palette, recall_prefix=WRP)
    C.plot_weighted_metrics(top, out_dir / f"weighted_metrics_f1_r5_r10.{a.ext}",
                            palette=palette, recall_prefix=WRP, metrics=w_trio)
    print(f"  wrote weighted_metrics[_f1_r5_r10].{a.ext}  ({', '.join(top['split'].unique())})")

    shifted = C.canon_split_shift(w)
    for tag, prefix, group_order in (
        ("ID", "id_", C.ID_GROUP_ORDER), ("OOD", "ood_", C.OOD_GROUP_ORDER),
    ):
        sub = shifted[shifted["split"].astype(str).str.startswith(prefix)]
        if sub.empty:
            print(f"  (no {tag} rows)"); continue
        order = [g for g in group_order if g in set(sub["split"])]
        sub = sub.copy()
        sub["split"] = pd.Categorical(sub["split"], categories=order, ordered=True)
        sub = sub.sort_values("split")
        C.plot_weighted_metrics(sub, out_dir / f"{tag}_weighted_metrics.{a.ext}",
                                palette=palette, recall_prefix=WRP)
        C.plot_weighted_metrics(sub, out_dir / f"{tag}_weighted_metrics_f1_r5_r10.{a.ext}",
                                palette=palette, recall_prefix=WRP, metrics=w_trio)
        n = sub.groupby("split", observed=True).size()
        print(f"  wrote {tag}_weighted_metrics[_f1_r5_r10].{a.ext}  " +
              ", ".join(f"{k}={v}" for k, v in n.items()))

    if sweep["min_efficiency"].nunique() > 1:
        top_s = sweep.copy()
        top_s["split"] = top_s["split"].astype(str).str.split(":", n=1).str[0]
        C.plot_efficiency_sweep(top_s, out_dir / f"efficiency_sweep.{a.ext}", palette=palette)
        C.plot_efficiency_sweep(top_s, out_dir / f"efficiency_sweep_f1_r5_r10.{a.ext}",
                                palette=palette, metrics=s_trio)
        print(f"  wrote efficiency_sweep[_f1_r5_r10].{a.ext}")
    else:
        print("  (single efficiency threshold — no sweep plots)")

    pf = a.metrics_dir / "sweep_metrics_per_fold.csv"
    if pf.exists():
        C.plot_ood_matched_sweep(pd.read_csv(pf),
                                 out_dir / f"ood_sweep_matched.{a.ext}",
                                 metrics=s_trio, palette=palette)
        print(f"  wrote ood_sweep_matched.{a.ext}")
    else:
        print(f"  ({pf.name} not found — no matched OOD sweep)")

    for ext in ("svg", "png"):                     # shared legend, published once
        C.save_method_legend(palette, out_dir / f"legend.{ext}")
    print("  wrote legend.svg + legend.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
