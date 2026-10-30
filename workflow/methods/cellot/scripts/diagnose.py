"""Diagnose whether per-pert cellot models are collapsing or the pert effects
are just too small for kNN-based classification.

Computes for each trained pert P:
    baseline_mmd  = MMD(ctrl, target_P)         — no model; how far apart
                                                   controls and pert cells
                                                   already are in expression
                                                   space
    model_mmd     = MMD(g_P(ctrl), target_P)    — how close the transport
                                                   map gets us

And across all perts jointly:
    centroid L2 distances between g_P(ctrl) clouds for every pair (P, Q)
    — if all centroids are ~identical, different g's are collapsing to the
    same mapping. Spread = models are distinct.

Outputs under <models_root>/..:
    _diagnose/mmd.csv           pert, n_target, baseline_mmd, model_mmd
    _diagnose/centroids.npy     (n_perts, n_genes) array of g_P(ctrl) means
    _diagnose/centroid_dist.npy (n_perts, n_perts) pairwise L2
    _diagnose/perts.txt         row/col labels
    _diagnose/_diagnose.png     3-panel summary figure
"""
import argparse
from pathlib import Path

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from cellot.data.cell import _ensure_normalized, _held_out_context_from_metadata
from cellot.losses.mmd import compute_scalar_mmd
from cellot.models import load_cellot_model
from cellot.utils import load_config

from predict import (
    determine_kept_genes,
    load_controls,
    load_targets,
    maybe_encode_with_ae,
    pushforward,
    stream_load_encode_targets,
)


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True,
                    help="Dir with per-pert subdirs (same layout predict.py uses)")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", choices=["id", "ood"], default="id")
    ap.add_argument("--n_mmd_samples", type=int, default=1000,
                    help="Per-side sample size for MMD (O(n^2) memory).")
    ap.add_argument("--n_cloud_samples", type=int, default=1000,
                    help="Controls pushed through g for centroid estimation.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cellot_root = Path(args.models_root)
    if not (cellot_root / "models").exists():
        raise FileNotFoundError(f"No models/ under {cellot_root}")
    models_root = cellot_root / "models"
    splits_dir = Path(args.splits_dir)
    outdir = cellot_root / "_diagnose"
    outdir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(p.parent.parent for p in models_root.glob("*/cache/model.pt"))
    if not model_dirs:
        raise RuntimeError(f"No trained models in {models_root}")
    perts = [d.name for d in model_dirs]
    print(f"Found {len(model_dirs)} trained models")

    first_config = load_config(model_dirs[0] / "config.yaml")
    source_file = first_config.data.path
    ctrl_file = first_config.data.path_ctrl
    condition_col = first_config.data.condition
    ctx_col, held_out = _held_out_context_from_metadata(splits_dir)

    # Controls are small (~5GB) — load eagerly. Targets stream when AE is in
    # use so we never hold n_test × n_genes in memory at full-atlas scale.
    controls = load_controls(ctrl_file, splits_dir, args.split, ctx_col, held_out)
    kept_genes = determine_kept_genes(controls, source_file, first_config, held_out)
    controls = controls[:, kept_genes].copy()

    ctrl_X = to_dense(controls.X)
    ctrl_X, encoded = maybe_encode_with_ae(ctrl_X, first_config)

    if encoded:
        target_X, labels = stream_load_encode_targets(
            source_file, splits_dir, args.split, condition_col,
            kept_genes=kept_genes,
            ae_path=first_config.data.ae_emb.path,
        )
        print(f"AE-encoded to latent dim={ctrl_X.shape[1]}  "
              f"ctrl n={len(ctrl_X)}  targets n={len(target_X)}")
    else:
        targets, labels = load_targets(source_file, splits_dir, args.split,
                                       condition_col)
        targets = targets[:, kept_genes].copy()
        target_X = to_dense(targets.X)
        print(f"ctrl n={len(ctrl_X)}  targets n={len(target_X)}  "
              f"dim={ctrl_X.shape[1]} (gene space, no AE)")

    # Fixed ctrl sample for both MMD and cloud computations (common baseline).
    k_ctrl = min(args.n_cloud_samples, len(ctrl_X))
    ctrl_idx = rng.choice(len(ctrl_X), k_ctrl, replace=False)
    ctrl_sample = ctrl_X[ctrl_idx]

    # Per-pert: split target cells by true label
    target_by_pert = {p: np.where(labels == p)[0] for p in perts}

    rows = []
    centroids = np.zeros((len(perts), ctrl_X.shape[1]), dtype=np.float32)

    for col, (pert, mdir) in enumerate(tqdm(list(zip(perts, model_dirs)),
                                            desc="diagnose", unit="pert")):
        tgt_idx = target_by_pert.get(pert, np.array([], dtype=int))
        n_tgt = len(tgt_idx)
        if n_tgt == 0:
            rows.append({"pert": pert, "n_target": 0,
                         "baseline_mmd": np.nan, "model_mmd": np.nan})
        else:
            # Sample target cells to match ctrl sample size (bounded by availability)
            k_tgt = min(args.n_mmd_samples, n_tgt, k_ctrl)
            sel = rng.choice(tgt_idx, k_tgt, replace=False)
            tgt_sample = target_X[sel]

            ctrl_for_mmd = ctrl_sample[:k_tgt]
            baseline_mmd = float(compute_scalar_mmd(tgt_sample, ctrl_for_mmd))

            cfg = load_config(mdir / "config.yaml")
            (_, g_net), _ = load_cellot_model(
                cfg, restore=mdir / "cache" / "model.pt", input_dim=ctrl_X.shape[1]
            )
            cloud = pushforward(g_net, ctrl_sample)
            centroids[col] = cloud.mean(axis=0)
            cloud_for_mmd = cloud[:k_tgt]
            model_mmd = float(compute_scalar_mmd(tgt_sample, cloud_for_mmd))

            rows.append({"pert": pert, "n_target": int(n_tgt),
                         "baseline_mmd": baseline_mmd, "model_mmd": model_mmd})
            continue

        # If we skipped the model (no target cells), still compute its cloud
        # centroid so the pairwise plot isn't full of missing rows.
        cfg = load_config(mdir / "config.yaml")
        (_, g_net), _ = load_cellot_model(
            cfg, restore=mdir / "cache" / "model.pt", input_dim=ctrl_X.shape[1]
        )
        centroids[col] = pushforward(g_net, ctrl_sample).mean(axis=0)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "mmd.csv", index=False)

    # Pairwise centroid distance
    diffs = centroids[:, None, :] - centroids[None, :, :]
    cd = np.linalg.norm(diffs, axis=-1).astype(np.float32)
    np.save(outdir / "centroids.npy", centroids)
    np.save(outdir / "centroid_dist.npy", cd)
    (outdir / "perts.txt").write_text("\n".join(perts) + "\n")

    # Print quick summary
    m = df.dropna()
    if len(m):
        print(f"\nbaseline_mmd  median={m.baseline_mmd.median():.4g}  "
              f"p25={m.baseline_mmd.quantile(0.25):.4g}  "
              f"p75={m.baseline_mmd.quantile(0.75):.4g}")
        print(f"model_mmd     median={m.model_mmd.median():.4g}  "
              f"p25={m.model_mmd.quantile(0.25):.4g}  "
              f"p75={m.model_mmd.quantile(0.75):.4g}")
        ratio = (m.model_mmd / m.baseline_mmd).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"model/baseline ratio  median={ratio.median():.3f}  "
              f"(<1 = model helps, ≈1 = no help, >1 = model hurts)")
    off_diag = cd[~np.eye(len(perts), dtype=bool)]
    print(f"centroid distances between perts: "
          f"median={np.median(off_diag):.4g}  "
          f"p25={np.quantile(off_diag, 0.25):.4g}  "
          f"p75={np.quantile(off_diag, 0.75):.4g}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_scatter, ax_bar, ax_heat = axes

    if len(m):
        ax_scatter.scatter(m.baseline_mmd, m.model_mmd, alpha=0.7)
        lims = [0, max(m.baseline_mmd.max(), m.model_mmd.max()) * 1.05]
        ax_scatter.plot(lims, lims, "k--", alpha=0.5, label="y = x")
        ax_scatter.set(xlabel="baseline MMD (ctrl vs target)",
                       ylabel="model MMD (g(ctrl) vs target)",
                       title="baseline vs model MMD (per pert)",
                       xlim=lims, ylim=lims)
        ax_scatter.legend()

        order = m.sort_values("baseline_mmd").index
        x = np.arange(len(order))
        w = 0.4
        ax_bar.bar(x - w/2, m.loc[order, "baseline_mmd"], w,
                   label="baseline", color="gray")
        ax_bar.bar(x + w/2, m.loc[order, "model_mmd"], w,
                   label="model", color="steelblue")
        ax_bar.set(xlabel="pert (sorted by baseline mmd)", ylabel="MMD",
                   title="per-pert baseline vs model MMD")
        ax_bar.legend()
        ax_bar.set_xticks([])

    im = ax_heat.imshow(cd, cmap="viridis", aspect="auto")
    ax_heat.set(title="pairwise centroid L2 distance between g_P clouds",
                xlabel="pert index", ylabel="pert index")
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(outdir / "_diagnose.png", dpi=120)
    print(f"\nWrote {outdir}/")


if __name__ == "__main__":
    main()
