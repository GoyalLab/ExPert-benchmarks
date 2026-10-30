"""Plot training / eval curves across all per-perturbation cellot models.

Reads the HDF5 scalar logs under <models_root>/<pert>/cache/scalars and writes
a diagnostic figure showing:
    1. eval MMD per step, one line per pert (alpha-faded) + median & IQR band
    2. histogram of the final best (min) eval MMD per pert
    3. train gloss per step, median & IQR band
    4. early-stop step per pert (last logged step vs. configured n_iters)
    5. per-pert support (n val cells) vs min mmd — auto-enabled if the source
       h5ad + val indices referenced by any model's config are reachable

Usage:
    python scripts/plot_curves.py \\
        --models_root ./results/perturb-seq/small/fold_0/models \\
        --out ./results/perturb-seq/small/fold_0/_curves.png
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_scalars(path):
    """Return (train_df, eval_df). Empty DataFrames if missing/unreadable."""
    train = eval_ = pd.DataFrame()
    try:
        train = pd.read_hdf(path, "train")
    except (KeyError, FileNotFoundError, OSError):
        pass
    try:
        eval_ = pd.read_hdf(path, "eval")
    except (KeyError, FileNotFoundError, OSError):
        pass
    return train, eval_


def read_status(mdir):
    f = mdir / "cache" / "status"
    return f.read_text().strip() if f.exists() else "unknown"


def try_load_val_support(first_cfg_path):
    """Best-effort: read per-pert val-cell count from the source h5ad + splits
    referenced in a model's config. Returns {pert_name: n_val_cells} or None."""
    try:
        import anndata
        import yaml
        cfg = yaml.load(first_cfg_path.read_text(), Loader=yaml.UnsafeLoader)
        source = cfg["data"]["path"]
        splits_dir = Path(cfg["datasplit"]["path"])
        condition = cfg["data"].get("condition", "perturbation")
        val_idx = np.loadtxt(splits_dir / "id_val.txt.gz", dtype=int)
        adata = anndata.read(source, backed="r")
        labels = adata.obs[condition].iloc[val_idx]
        return labels.value_counts().to_dict()
    except Exception as e:
        print(f"(support panel disabled: {type(e).__name__}: {e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_step", type=int, default=None,
                    help="Clip x-axis; useful if some models ran much longer")
    args = ap.parse_args()

    cellot_root = Path(args.models_root)
    out = Path(args.out) if args.out else cellot_root / "_diagnose" / "_curves.png"
    if not (cellot_root / "models").exists():
        raise FileNotFoundError(f"No models/ under {cellot_root}")
    models_root = cellot_root / "models"

    model_dirs = sorted(p.parent.parent for p in models_root.glob("*/cache/scalars"))
    if not model_dirs:
        raise RuntimeError(f"No scalar logs under {models_root}")
    print(f"Found {len(model_dirs)} models with scalar logs")

    rows = []
    for mdir in model_dirs:
        train_df, eval_df = load_scalars(mdir / "cache" / "scalars")
        rows.append({
            "pert": mdir.name,
            "status": read_status(mdir),
            "train": train_df,
            "eval": eval_df,
            "last_step": int(max(
                train_df.index.max() if len(train_df) else 0,
                eval_df.index.max() if len(eval_df) else 0,
            )),
            "min_mmd": float(eval_df["mmd"].min()) if "mmd" in eval_df else float("nan"),
        })

    support_map = try_load_val_support(model_dirs[0] / "config.yaml")
    if support_map is not None:
        fig, axes = plt.subplots(2, 3, figsize=(19, 9))
        ax_mmd, ax_hist, ax_support, ax_gloss, ax_stop, ax_empty = axes.ravel()
        ax_empty.set_visible(False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax_mmd, ax_hist, ax_gloss, ax_stop = axes.ravel()
        ax_support = None

    # 1) eval MMD curves
    mmd_frames = []
    for r in rows:
        if "mmd" not in r["eval"]:
            continue
        s = r["eval"]["mmd"]
        ax_mmd.plot(s.index, s.values, alpha=0.2, linewidth=0.7, color="steelblue")
        mmd_frames.append(s.rename(r["pert"]))
    if mmd_frames:
        mmd_wide = pd.concat(mmd_frames, axis=1)
        ax_mmd.plot(mmd_wide.index, mmd_wide.median(axis=1), color="black",
                    linewidth=2, label="median")
        ax_mmd.fill_between(
            mmd_wide.index,
            mmd_wide.quantile(0.25, axis=1),
            mmd_wide.quantile(0.75, axis=1),
            color="black", alpha=0.15, label="IQR",
        )
        ax_mmd.legend()
    if args.max_step:
        ax_mmd.set_xlim(0, args.max_step)
    ax_mmd.set(title="eval MMD over step (per pert)", xlabel="step", ylabel="mmd")

    # 2) final min-mmd histogram
    mins = np.array([r["min_mmd"] for r in rows if not np.isnan(r["min_mmd"])])
    if len(mins):
        ax_hist.hist(mins, bins=30, color="steelblue", edgecolor="black")
        ax_hist.axvline(np.median(mins), color="red", linestyle="--",
                        label=f"median={np.median(mins):.4g}")
        ax_hist.legend()
    ax_hist.set(title="final min eval MMD per pert", xlabel="min mmd", ylabel="count")

    # 2b) per-pert support (n val cells) vs min mmd — requires loaded support map
    if ax_support is not None:
        groups = {"done": ([], []), "in-progress": ([], [])}
        for r in rows:
            n = support_map.get(r["pert"])
            if n is None or np.isnan(r["min_mmd"]):
                continue
            key = "done" if r["status"] == "done" else "in-progress"
            groups[key][0].append(int(n))
            groups[key][1].append(float(r["min_mmd"]))
        for label, (xs, ys), color in [
            ("done", groups["done"], "steelblue"),
            ("in-progress", groups["in-progress"], "lightgray"),
        ]:
            if xs:
                ax_support.scatter(xs, ys, alpha=0.6, color=color,
                                   label=f"{label} (n={len(xs)})")
        ax_support.set_xscale("log")
        ax_support.legend()
        ax_support.set(
            title="support vs min eval MMD (per pert)",
            xlabel="n val cells (log)", ylabel="min mmd",
        )

    # 3) train gloss curves (log y-scale; values can be huge early)
    gloss_frames = []
    for r in rows:
        if "gloss" not in r["train"]:
            continue
        s = r["train"]["gloss"].abs()  # absolute value for log plot
        ax_gloss.plot(s.index, s.values, alpha=0.15, linewidth=0.7, color="darkorange")
        gloss_frames.append(s.rename(r["pert"]))
    if gloss_frames:
        g_wide = pd.concat(gloss_frames, axis=1)
        ax_gloss.plot(g_wide.index, g_wide.median(axis=1), color="black",
                      linewidth=2, label="median")
        ax_gloss.legend()
    ax_gloss.set_yscale("log")
    if args.max_step:
        ax_gloss.set_xlim(0, args.max_step)
    ax_gloss.set(title="train |gloss| over step (per pert)", xlabel="step",
                 ylabel="|gloss| (log)")

    # 4) last-step per pert (proxy for early-stop behaviour)
    last_steps = sorted(r["last_step"] for r in rows)
    ax_stop.plot(range(len(last_steps)), last_steps, "o", color="steelblue")
    ax_stop.set(title="final logged step per pert (sorted)",
                xlabel="pert index (sorted)", ylabel="last step")

    # Create output dir if needed
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")

    # Print a small summary to stdout
    n_done = sum(r["status"] == "done" for r in rows)
    print(f"status: {n_done}/{len(rows)} done")
    if len(mins):
        print(f"min_mmd: median={np.median(mins):.4g}  "
              f"p25={np.quantile(mins, 0.25):.4g}  "
              f"p75={np.quantile(mins, 0.75):.4g}  "
              f"min={mins.min():.4g}  max={mins.max():.4g}")


if __name__ == "__main__":
    main()
