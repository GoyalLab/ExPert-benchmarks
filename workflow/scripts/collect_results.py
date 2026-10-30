"""Collect logits from a source tree and score methods.

Layout assumption:
    <source>/<method-segments...>/.../<split>/logits.{csv,csv.gz,npy,parquet,h5,hdf5}
    # logits.npy: a sibling column-names file is required (logits.cols.txt[.gz]
    #             or column_names.txt[.gz], one class name per line).
    # logits.h5/.hdf5: try pandas HDF first; otherwise h5py looks for a 2D
    #             dataset under "logits"/"data"/"matrix"/"X" plus class names
    #             under "columns"/"classes"/"gene_names"/"var_names"/"cols".
    #             Falls back to the sibling-text-file convention.

Per-row labels are read directly from each logits file:
    - HDF5: a 1D string-like dataset whose length matches the row axis
      (the picker also drops it from the class-names candidate list).
    - CSV / parquet / pandas-HDF5: the DataFrame's index (default
      `Unnamed: 0` index columns or any non-numeric column is promoted to
      the index by the reader). No sibling labels.csv is required.

- <split>     = basename of the parent directory of each logits.csv
                (e.g. id_val, id_test, ood_test, eval_iid, eval_ood, ...)
- <direction> = first path segment under <source> when --method-depth >= 2
                (e.g. forward, reverse), else NA
- <method>    = remaining captured segments (1..method_depth-1) joined with "/".
                With --method-depth=1 the method is just the first segment.
- <fold>      = first fold_N ancestor of each logits.csv (when --splits-dir is set).

When --splits-dir is set (e.g. data/splits), per-cell labels and metadata are
pulled from the source adata referenced by <splits-dir>/<dataset-size>/metadata.json.
For each logits file we read <splits-dir>/<dataset-size>/<fold>/<canonical_split>.txt.gz
(aliases: eval_iid→id_test, eval_ood→ood_test, val→id_val, test→id_test,
train→id_train), open the adata in backed mode, slice obs by those indices,
and attach every obs column as obs_<col>. Embedded row labels (file index)
are cross-checked against adata.obs[--label-col] when present.

--dataset-size (default small) filters the logits tree to paths that contain
the size segment, and points metadata/indices at the matching subtree.
--fold restricts to a specific fold (e.g. fold_0); omit it to include all folds.

Example — cover forward + reverse in one run:
    python collect_results.py \
        --source benchmarks/context \
        --splits-dir data/splits \
        --dataset-size small \
        --method-depth 2 \
        --output-dir benchmarks/context/results/small \
        --efficiency-sweep none 1 2 4 \
        --splits id_test ood_test eval_iid eval_ood \
        --knn

Output layout:
    <output_dir>/
        all_logits.parquet         # unfiltered stacked log-probs
        none/
            all_metrics.csv
            all_perclass_metrics.parquet
            plots/{topk_recall,macro_f1,perclass_f1_boxen,perclass_topk_boxen}.png
        e:1/
            ...
        e:2/
            ...
        e:4/
            ...
        sweep_metrics.csv          # only when --efficiency-sweep is set
        plots/efficiency_sweep.png # only when >1 thresholds run

Within each (method, split) all files must share shape + columns; within each
method the logit columns (label space) must match across splits. Labels.csv
must exist next to each logits.csv and have the same row count.

The stacked frame adds: label, split, direction, method, source_path.

Plots (under each threshold subdir):
  - topk_recall.png         : bar plot of recall@k, hue=method, facet=split
  - topk_recall_balanced.png: same, but mean-over-classes (balanced) recall@k
  - macro_f1.png            : bar plot of macro F1, x=split, hue=method
  - perclass_f1_boxen.png   : boxen of per-class F1 across classes, facet=split
  - perclass_topk_boxen.png : boxen of per-class recall@k, rows=split, cols=k
  - perdataset_f1_boxen.png : boxen of per-class F1 by dataset (id+ood share x),
                              hue=method, val splits excluded (requires --splits-dir).
                              Top panels show training n_cells and n_perts per dataset.
  - support_vs_f1.png       : scatter of per-class F1 vs test-split support, log x,
                              hue=method, col=split (val excluded)
Sweep plots (at <output_dir>/plots/ when multiple thresholds ran):
  - efficiency_sweep.png             : metric vs min-efficiency, row=metric, col=split, hue=method
  - cells_per_perturbation_sweep.png : top=n_perts with ≥1 cell, bottom=boxen of
                                       cells/perturbation; both vs min-efficiency,
                                       col=split. Sanity-check for whether metric
                                       gains track class drop-outs.

Metrics (per method x split):
  - accuracy      : top-1 correctness
  - macro_f1      : F1 averaged across classes (fairness across perturbations)
  - recall@k      : top-k retrieval recall (--ks, default 1 5 10). On read,
                    each logits file is auto-normalised to log-probs (detected
                    as: already log-probs if row logsumexp ≈ 0; probabilities
                    if rows in [0,1] sum to 1; otherwise raw logits → log-softmax).
  - knn_acc       : kNN (fit on each method's val split) accuracy on non-val
                    splits. Only computed when --knn is set and the method has
                    exactly one val-like split (name matches (^|[_-])val($|[_-])).
"""

import argparse
import gzip
import json
import logging
import os
import re
from pathlib import Path
# tqdm is NOT in workflow/envs/collect.yaml and never has been -- this import has
# sat here since the initial scaffold and would abort the whole `collect` rule
# (its one use below is a cosmetic progress bar). Degrade to a no-op instead of
# taking down the scorer. Adding tqdm to collect.yaml would change the env hash
# and force a full rebuild of the collect env mid-run, which is the riskier fix
# while a controller is live; do that between runs if you want the bar back.
try:
    from tqdm import tqdm
except ModuleNotFoundError:                     # pragma: no cover
    def tqdm(it, *_a, **_kw):
        return it

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

META_COLS = ("label", "split", "direction", "method", "source_path", "fold")
OBS_PREFIX = "obs_"
VAL_SPLIT_RE = re.compile(r"(^|[_-])val($|[_-])", re.IGNORECASE)
FOLD_RE = re.compile(r"^fold_(\d+)$")

# map per-method split-dir names onto the canonical index files under split_dir/fold_N/.
# Dashes are normalised to underscores first (so "id-test", "ood-test", etc. collapse
# to the underscore forms) — list aliases here in their underscore form only.
SPLIT_ALIASES: dict[str, str] = {
    "eval_iid": "id_test",
    "eval_ood": "ood_test",
    "val": "id_val",
    "test": "id_test",
    "train": "id_train",
}


def _canonical_split(name: str) -> str:
    """Canonicalise a split name: dashes→underscores, then alias lookup."""
    norm = str(name).replace("-", "_")
    return SPLIT_ALIASES.get(norm, norm)


def _resolve_index_file(fold_dir: Path, split: str) -> Path:
    canonical = _canonical_split(split)
    p = fold_dir / f"{canonical}.txt.gz"
    if not p.exists():
        raise FileNotFoundError(
            f"no index file for split={split} (resolved to {canonical}, tried {p})"
        )
    return p


def _load_indices(p: Path) -> np.ndarray:
    with gzip.open(p, "rt") as f:
        return np.fromiter((int(line) for line in f if line.strip()), dtype=np.int64)


LOGIT_PATTERNS = (
    "logits.csv", "logits.csv.gz", "logits.npy",
    "logits.parquet", "logits.h5", "logits.hdf5",
)
LOGIT_COL_CANDIDATES = (
    "{stem}.cols.txt",
    "{stem}.cols.txt.gz",
    "column_names.txt",
    "column_names.txt.gz",
    "columns.txt",
    "columns.txt.gz",
    "labels.txt",
    "labels.txt.gz"
)
H5_MATRIX_KEYS = ("logits", "data", "matrix", "X")
H5_COLUMN_KEYS = ("columns", "classes", "gene_names", "var_names", "cols", "labels")


def _load_external_column_names(path: Path, ncols: int | None = None) -> list[str]:
    """Find a sibling text file with column names (one per line) for a logits matrix.

    `ncols` is optional; if given, the number of names must match (caller does
    the orientation check otherwise).
    """
    stem = path.with_suffix("").name
    candidates = [
        path.parent / template.format(stem=stem) for template in LOGIT_COL_CANDIDATES
    ]
    for c in candidates:
        if not c.exists():
            continue
        opener = gzip.open if c.suffix == ".gz" else open
        with opener(c, "rt") as fh:
            names = [line.strip() for line in fh if line.strip()]
        if ncols is not None and len(names) != ncols:
            raise ValueError(
                f"{c} has {len(names)} names but {path} has {ncols} columns"
            )
        return names
    raise FileNotFoundError(
        f"no column-names file next to {path} (tried "
        f"{[c.name for c in candidates]}). Save logit class names alongside the matrix."
    )


def _orient_to_classes_as_cols(arr: np.ndarray, n_classes: int, path: Path) -> np.ndarray:
    """Ensure the columns axis equals n_classes; transpose if it's the rows axis."""
    if arr.shape[1] == n_classes:
        return arr
    if arr.shape[0] == n_classes:
        log.info(
            "[orient] %s: transposing %s -> %s (classes -> columns axis)",
            path, arr.shape, arr.shape[::-1],
        )
        return arr.T
    raise ValueError(
        f"{path}: matrix shape {arr.shape} matches neither axis to {n_classes} classes"
    )


def _drop_leading_index(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric (logit) columns; promote the first non-numeric column
    to the DataFrame index so callers can recover embedded row labels later.

    Drops:
      - leading "Unnamed: 0" columns (default `df.to_csv(... index=True)` output)
      - any other non-numeric column (string indices like 'cell_id', method-
        embedded label columns like scGPT's predicted-class column, etc.)

    The first non-numeric column found becomes the index (as strings), so a
    caller checking `raw.index` instead of a default `RangeIndex` can read the
    per-row labels without re-reading the file.
    """
    if len(df.columns) == 0:
        return df
    keep: list = []
    drop: list = []
    index_col: str | None = None
    for i, c in enumerate(df.columns):
        if i == 0 and str(c).startswith("Unnamed"):
            drop.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
        else:
            if index_col is None:
                index_col = c
            drop.append(c)
    if not drop:
        return df
    log.info(
        "[load] dropping %d non-numeric col(s): %s%s",
        len(drop), drop[:5], "..." if len(drop) > 5 else "",
    )
    out = df[keep]
    if index_col is not None:
        out = out.set_axis(
            pd.Index(df[index_col].astype(str).to_numpy(), name="row_label"),
            axis=0,
        )
    return out


def _h5_pick_names_and_shape(path: Path) -> tuple[str, list[str], tuple[int, int]]:
    """Open an HDF5 file briefly and return (matrix_dataset_key, class names, shape).

    Doesn't load the matrix — used both for full reads and for cheap column
    peeking during the discovery phase.
    """
    import h5py

    with h5py.File(path, "r") as fh:
        keys = list(fh.keys())
        mat_key = next((k for k in H5_MATRIX_KEYS if k in fh), None)
        if mat_key is None:
            mat_key = next(
                (k for k in keys
                 if isinstance(fh[k], h5py.Dataset) and fh[k].ndim == 2),
                None,
            )
        if mat_key is None:
            raise ValueError(f"{path}: no 2D dataset found (keys: {keys})")
        mat_shape = tuple(fh[mat_key].shape)
        if len(mat_shape) != 2:
            raise ValueError(f"{path}/{mat_key}: expected 2D, got shape {mat_shape}")

        candidates: list[tuple[int, int, str]] = []  # (preference, length, key)
        for k in keys:
            if k == mat_key:
                continue
            ds = fh[k]
            if not isinstance(ds, h5py.Dataset) or ds.ndim != 1:
                continue
            L = ds.shape[0]
            if L != mat_shape[0] and L != mat_shape[1]:
                continue
            pref = H5_COLUMN_KEYS.index(k) if k in H5_COLUMN_KEYS else len(H5_COLUMN_KEYS)
            candidates.append((pref, L, k))
        candidates.sort(key=lambda t: (t[1], t[0]))

        if candidates:
            _, n_classes, cols_key = candidates[0]
            raw = fh[cols_key][:]
            cols = [c.decode() if isinstance(c, bytes) else str(c) for c in raw]
            if cols_key not in H5_COLUMN_KEYS:
                log.info(
                    "[h5] %s: picked %r as class names (length %d)",
                    path, cols_key, n_classes,
                )
        else:
            cols = _load_external_column_names(path)
    return mat_key, cols, mat_shape


def _read_h5_logits(path: Path) -> pd.DataFrame:
    """Load logits from an HDF5 file. Tries pandas HDF first, then h5py with conventions."""
    try:
        obj = pd.read_hdf(path)
        if isinstance(obj, pd.DataFrame):
            return _drop_leading_index(obj)
    except (ImportError, ValueError, TypeError, KeyError, OSError):
        # ImportError covers missing pytables — fall through to h5py.
        pass

    import h5py

    mat_key, cols, _ = _h5_pick_names_and_shape(path)
    n_classes = len(cols)
    with h5py.File(path, "r") as fh:
        arr = fh[mat_key][:]
        # determine post-orient row count to find a matching 1D string dataset
        if arr.shape[1] == n_classes:
            n_final_rows = arr.shape[0]
        elif arr.shape[0] == n_classes:
            n_final_rows = arr.shape[1]
        else:
            n_final_rows = arr.shape[0]  # let _orient raise the clearer error
        row_labels: np.ndarray | None = None
        for k in fh.keys():
            if k == mat_key:
                continue
            ds = fh[k]
            if not isinstance(ds, h5py.Dataset) or ds.ndim != 1:
                continue
            if ds.shape[0] != n_final_rows:
                continue
            if ds.dtype.kind not in ("U", "S", "O"):
                continue
            raw_lbl = ds[:]
            row_labels = np.array(
                [c.decode() if isinstance(c, bytes) else str(c) for c in raw_lbl]
            )
            break
    arr = _orient_to_classes_as_cols(arr, n_classes, path)
    df = pd.DataFrame(arr, columns=cols)
    if row_labels is not None and len(row_labels) == df.shape[0]:
        df = df.set_axis(pd.Index(row_labels, name="row_label"), axis=0)
    return df


def _peek_columns(path: Path) -> list[str]:
    """Read just the class-name list for a logits file without loading the matrix."""
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return _load_external_column_names(path)
    if suffixes.endswith(".parquet"):
        import pyarrow.parquet as pq
        import pyarrow.types as pat
        schema = pq.read_schema(str(path))
        cols = []
        for i, name in enumerate(schema.names):
            # drop leading default-index columns
            if i == 0 and (str(name).startswith("Unnamed") or name in ("", "index")):
                continue
            # only numeric columns count as class logits
            if pat.is_integer(schema.field(name).type) or \
               pat.is_floating(schema.field(name).type):
                cols.append(name)
        return cols
    if suffixes.endswith((".h5", ".hdf5")):
        try:
            obj = pd.read_hdf(path, stop=1)
            if isinstance(obj, pd.DataFrame):
                return list(_drop_leading_index(obj).columns)
        except (ImportError, ValueError, TypeError, KeyError, OSError):
            pass
        _, cols, _ = _h5_pick_names_and_shape(path)
        return cols
    # CSV / CSV.GZ — read header + 1 data row so we can spot non-numeric
    # columns (cell_id, true_perturbation, etc.) the same way _drop_leading_index
    # will at full load time. Without this peek, those non-class columns sneak
    # into the cross-method union and become bogus -inf classes downstream.
    head = pd.read_csv(path, nrows=1)
    return list(_drop_leading_index(head).columns)


def _read_logits_idx(path: Path, n_rows: int) -> np.ndarray | None:
    """Read /idx from a fingerprint-style logits.h5 — 0-based source-adata
    positions, one per row of /logits. Used to recover obs alignment when the
    method dropped cells (e.g. fingerprint's old subset=TRUE pre-filter)
    without falling back to label-only scoring.

    Returns None if the file isn't HDF5, the dataset is missing, or its
    length doesn't match n_rows.
    """
    suffixes = "".join(path.suffixes).lower()
    if not suffixes.endswith((".h5", ".hdf5")):
        return None
    try:
        import h5py
        with h5py.File(path, "r") as fh:
            if "idx" not in fh:
                return None
            ds = fh["idx"]
            if ds.ndim != 1 or ds.shape[0] != n_rows:
                return None
            return np.asarray(ds[:], dtype=np.int64)
    except (OSError, KeyError, ValueError) as e:
        log.warning("[row-idx] %s: h5 lookup failed (%s)", path, e)
        return None


def _read_logits(path: Path) -> pd.DataFrame:
    """Load logits into a DataFrame with named columns.

    Supported formats:
      - .csv / .csv.gz : header row provides class names
      - .parquet       : column names from parquet schema
      - .npy           : 2D array; expects sibling column-names file
                         (see LOGIT_COL_CANDIDATES)
      - .h5 / .hdf5    : pandas HDF5 if saved that way; otherwise h5py with
                         conventional matrix keys (logits, data, matrix, X)
                         and column-name keys (columns, classes, gene_names,
                         var_names, cols, labels). Falls back to the sibling
                         text file if no in-file names found.

    A leading index column ('Unnamed: 0' or non-numeric like cell_id) is dropped
    for CSV/parquet/HDF-via-pandas inputs.
    """
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected a 2D array, got shape {arr.shape}")
        cols = _load_external_column_names(path)
        arr = _orient_to_classes_as_cols(arr, len(cols), path)
        return pd.DataFrame(arr, columns=cols)
    if suffixes.endswith(".parquet"):
        return _drop_leading_index(pd.read_parquet(path))
    if suffixes.endswith((".h5", ".hdf5")):
        return _read_h5_logits(path)
    # CSV (pandas auto-decompresses .csv.gz from the extension)
    return _drop_leading_index(pd.read_csv(path))


def _detect_fold(path: Path) -> str | None:
    for anc in path.parents:
        if FOLD_RE.match(anc.name):
            return anc.name
    return None


from dataclasses import dataclass


@dataclass(frozen=True)
class FileMeta:
    path: Path
    method: str
    direction: str | None
    split: str
    fold: str | None


def _list_logits_files(source: Path) -> list[Path]:
    pattern_set = set(LOGIT_PATTERNS)
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(source, followlinks=True):
        for fn in filenames:
            if fn in pattern_set:
                out.append(Path(dirpath) / fn)
    return sorted(set(out))


def discover_files(
    source: Path,
    method_depth: int = 1,
    dataset_size: str = "small",
    fold: str | None = None,
) -> tuple[list[FileMeta], dict[Path, list[str]]]:
    """Find logits files matching --dataset-size / --fold, peek at their column names.

    Returns the list of FileMeta plus a {path: cols} dict (cheap to compute,
    needed for the consistency phase before any matrix is loaded).
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    all_files = _list_logits_files(source)
    if not all_files:
        raise FileNotFoundError(
            f"no {'/'.join(LOGIT_PATTERNS)} under {source}"
        )

    metas: list[FileMeta] = []
    for f in all_files:
        if dataset_size and dataset_size not in f.parts:
            continue
        file_fold = _detect_fold(f)
        if fold is not None and file_fold != fold:
            continue
        rel_parts = f.relative_to(source).parts
        if len(rel_parts) < method_depth + 2:
            raise ValueError(
                f"{f} is too shallow for --method-depth={method_depth} "
                f"(relative parts: {rel_parts})"
            )
        if method_depth >= 2:
            direction = rel_parts[0]
            method = "/".join(rel_parts[1:method_depth])
        else:
            direction = None
            method = rel_parts[0]
        metas.append(FileMeta(
            path=f, method=method, direction=direction,
            split=_canonical_split(f.parent.name), fold=file_fold,
        ))
    if not metas:
        raise FileNotFoundError(
            f"no logits files matched dataset_size={dataset_size!r} fold={fold!r} under {source}"
        )

    cols_by_path: dict[Path, list[str]] = {}
    for m in metas:
        cols_by_path[m.path] = _peek_columns(m.path)

    log.info(
        "[discover] %d/%d logits files match (size=%s, fold=%s)",
        len(metas), len(all_files), dataset_size, fold,
    )
    return metas, cols_by_path


def apply_consistency(
    metas: list[FileMeta],
    cols_by_path: dict[Path, list[str]],
) -> tuple[list[FileMeta], dict[tuple[str | None, str], list[str]]]:
    """Drop offending files/methods using only column-name metadata.

    - Within (method, split, direction, fold): pick the first file as canonical;
      warn + drop subsequent files whose column SET differs.
    - Within (direction, method) across splits: compute the union of all class
      sets — surviving files will be padded to this union at load time.
    - Across methods: drop any method whose union differs from the reference
      method's union (apples-to-apples).

    Returns the surviving FileMeta list and the per-(direction, method)
    sorted-list union to use for padding.
    """
    by_fold_seen: dict[tuple[str, str, str | None, str | None], FileMeta] = {}
    keep1: list[FileMeta] = []
    for m in metas:
        key = (m.method, m.split, m.direction, m.fold)
        if key not in by_fold_seen:
            by_fold_seen[key] = m
            keep1.append(m)
            continue
        ref = by_fold_seen[key]
        if set(cols_by_path[m.path]) != set(cols_by_path[ref.path]):
            log.warning(
                "[drop file] %s vs %s — column-set mismatch, dropping latter",
                ref.path, m.path,
            )
        # silently dedupe identical-column duplicates

    method_union_sets: dict[tuple[str | None, str], set[str]] = {}
    for m in keep1:
        method_union_sets.setdefault((m.direction, m.method), set()).update(
            cols_by_path[m.path]
        )

    # Cross-method label space: take the global union across all methods so
    # everyone scores against the same class set. Methods that don't predict
    # some classes get those columns padded with -inf at load time (zero
    # probability — they just can't pick those classes). Macro-F1 will reflect
    # that legitimately rather than excluding the method entirely.
    global_union: set[str] = set()
    for s in method_union_sets.values():
        global_union.update(s)
    # --classes: pin the label space instead of taking the cross-method union.
    # Needed whenever methods legitimately cover very different fractions of the
    # space (the padding benchmark: 9.4%-100% of 10,245 classes). With a union the
    # space would silently depend on WHICH methods happened to run, so macro-F1
    # would not be comparable across runs; with a fixed list every method is padded
    # to the same columns and its coverage is reported instead of absorbed.
    if FIXED_CLASSES is not None:
        fixed = set(FIXED_CLASSES)
        extra = global_union - fixed
        if extra:
            log.warning(
                "[classes] %d emitted classes are NOT in --classes and will be DROPPED "
                "(e.g. %s). This is expected when a method emits its training vocabulary "
                "alongside the evaluated one.", len(extra), sorted(extra)[:10])
        missing_all = fixed - global_union
        if missing_all:
            log.warning(
                "[classes] %d/%d classes in --classes are emitted by NO method — they stay "
                "-inf for everyone (chance is computed over the full fixed space). "
                "Examples: %s", len(missing_all), len(fixed), sorted(missing_all)[:10])
        global_union = fixed
        method_union_sets = {k: (v & fixed) for k, v in method_union_sets.items()}
    # Class column ORDER decides tie-breaking downstream: argpartition/argsort in
    # _argtopk_chunked resolve equal scores by column index. Sorting alphabetically
    # therefore hands every tie to the alphabetically-earliest perturbation — a real
    # bias for methods that emit many exactly-equal scores (fingerprint scores whole
    # susie credible sets, leaving every non-set perturbation at exactly 0.0, so its
    # top-1 would otherwise measure alphabetical luck). Shuffle once with a fixed
    # seed: reproducible, costs nothing, and makes tie-breaking independent of the
    # perturbation NAME. Ties are still resolved deterministically, just not
    # systematically in one method's favour; `n_tied_at_top` reports how often it
    # mattered. --tie-break-seed -1 restores the old alphabetical order.
    global_sorted = sorted(global_union)
    if TIE_BREAK_SEED >= 0:
        _perm = np.random.default_rng(TIE_BREAK_SEED).permutation(len(global_sorted))
        global_sorted = [global_sorted[i] for i in _perm]

    method_seen: set[str] = set()
    for m in keep1:
        if m.method in method_seen:
            continue
        method_seen.add(m.method)
        own = method_union_sets[(m.direction, m.method)]
        missing = global_union - own
        if missing:
            log.warning(
                "[pad method] method=%s missing %d/%d classes from global union — "
                "will pad with -inf at load time. Examples: %s%s",
                m.method, len(missing), len(global_union),
                sorted(missing)[:10], "..." if len(missing) > 10 else "",
            )

    for m in keep1:
        own = method_union_sets[(m.direction, m.method)]
        present = set(cols_by_path[m.path])
        missing = own - present
        if missing:
            log.warning(
                "[pad split] method=%s split=%s direction=%s fold=%s: missing %d/%d "
                "classes from this method's union — will pad with -inf at load time",
                m.method, m.split, m.direction, m.fold, len(missing), len(own),
            )

    method_unions = {k: global_sorted for k in method_union_sets}
    return keep1, method_unions


def load_obs(
    splits_dir: Path,
    dataset_size: str,
    label_col: str,
) -> tuple[pd.DataFrame, Path, Path]:
    """Open the source adata in backed mode; return (obs DataFrame, size_dir, source_h5ad)."""
    import anndata as ad

    splits_dir = Path(splits_dir).resolve()
    size_dir = splits_dir / dataset_size
    meta = json.loads((size_dir / "metadata.json").read_text())
    source_h5ad = Path(meta["source_path"])
    adata = ad.read_h5ad(source_h5ad, backed="r")
    obs = adata.obs
    if label_col not in obs.columns:
        raise ValueError(
            f"label_col={label_col!r} missing from obs (have: {list(obs.columns)})"
        )
    log.info(
        "[load_obs] %s (%d rows, %d cols); size=%s label_col=%s",
        source_h5ad, obs.shape[0], obs.shape[1], dataset_size, label_col,
    )
    return obs, size_dir, source_h5ad


@dataclass
class FileData:
    """Per-file scoring view that keeps the big logits matrix as raw numpy.

    pandas wrapping of a (1.3M × 7k) float32 matrix triggers block consolidation
    on every operation (groupby, column-select, boolean-mask) and balloons memory
    + time. We keep the matrix as a flat ndarray and only pandas-wrap the small
    obs metadata DataFrame.
    """
    log_probs: np.ndarray            # (n, n_classes) float32, columns aligned to `classes`
    classes: list[str]               # length n_classes — the per-method-padded global union
    labels: np.ndarray               # (n,) object dtype — true class names
    obs: pd.DataFrame                # (n, k) — only the small obs cols we attached
    meta: FileMeta

    @property
    def n_rows(self) -> int:
        return self.log_probs.shape[0]

    def filter_mask(self, mask: np.ndarray, meta: "FileMeta | None" = None) -> "FileData":
        return FileData(
            log_probs=self.log_probs[mask],
            classes=self.classes,
            labels=self.labels[mask],
            obs=self.obs.iloc[mask].reset_index(drop=True) if len(self.obs.columns) else self.obs,
            meta=meta if meta is not None else self.meta,
        )


def _split_by_dataset(fd: "FileData") -> list["FileData"]:
    """Return one FileData per dataset present in obs[OBS_PREFIX+'dataset'].

    Each dataset is treated as its own "split-like" data point: the returned
    FileData carries a split tag of `<original_split>:<dataset>` so every
    downstream group-by (method, split, fold) bucket counts it separately.
    Applies to every split (id / ood / val) for symmetric per-dataset metrics.
    Files with no obs metadata or only one dataset are returned unchanged as a
    single-element list.
    """
    ds_col = f"{OBS_PREFIX}dataset"
    if ds_col not in fd.obs.columns:
        return [fd]
    datasets = fd.obs[ds_col].to_numpy()
    unique = pd.unique(datasets)
    if len(unique) <= 1:
        return [fd]
    log.info(
        "[ds split] %s [%s]: %d cells split across %d datasets: %s",
        fd.meta.path, fd.meta.split, fd.n_rows, len(unique), list(unique),
    )
    out = []
    for ds in unique:
        mask = datasets == ds
        sub_meta = FileMeta(
            path=fd.meta.path,
            method=fd.meta.method,
            direction=fd.meta.direction,
            split=f"{fd.meta.split}:{ds}",
            fold=fd.meta.fold,
        )
        out.append(fd.filter_mask(mask, meta=sub_meta))
    return out


def load_one_file(
    meta: FileMeta,
    obs: pd.DataFrame | None,
    size_dir: Path | None,
    label_col: str,
    obs_cols: tuple[str, ...] | None,
    union_sorted: list[str],
) -> FileData | None:
    """Stream-load one logits file. Returns a FileData (no big pandas wrapping)."""
    raw = _read_logits(meta.path)
    raw_cols = list(raw.columns)
    # embedded row labels = a non-default index set by _drop_leading_index /
    # _read_h5_logits when the file ships per-row identifiers.
    embedded_row_labels: np.ndarray | None = (
        None if isinstance(raw.index, pd.RangeIndex)
        else raw.index.astype(str).to_numpy()
    )
    raw_arr = raw.to_numpy(copy=False)
    del raw

    # cast to float32 only if we have to (avoid silent 2x copy of a 36GB array)
    if raw_arr.dtype != np.float32:
        log.info("[load] %s: casting %s -> float32", meta.path, raw_arr.dtype)
        mat = raw_arr.astype(np.float32)
    else:
        mat = raw_arr  # view of the original
    del raw_arr

    kind = _to_log_probs_inplace(mat)

    raw_set = set(raw_cols)
    union_set = set(union_sorted)
    if raw_set == union_set and raw_cols == union_sorted:
        log_probs = mat
    elif raw_set == union_set:
        # reorder columns only — np takes a contiguous copy here, unavoidable
        col_to_idx = {c: i for i, c in enumerate(raw_cols)}
        new_idx = np.array([col_to_idx[c] for c in union_sorted], dtype=np.int64)
        log_probs = mat[:, new_idx]
        del mat
    else:
        # need padding → full (n, |union|) allocation
        col_to_idx = {c: i for i, c in enumerate(raw_cols)}
        log_probs = np.full((mat.shape[0], len(union_sorted)),
                            float("-inf"), dtype=np.float32)
        for j, c in enumerate(union_sorted):
            i = col_to_idx.get(c)
            if i is not None:
                log_probs[:, j] = mat[:, i]
        del mat

    log.info("[load] %s: %s (fold=%s, shape=%s, dtype=%s)",
             meta.path, kind, meta.fold, log_probs.shape, log_probs.dtype)

    obs_small = pd.DataFrame()
    if obs is not None:
        if meta.fold is None:
            raise ValueError(f"{meta.path} has no fold_N ancestor; required when --splits-dir is set")
        idx_path = _resolve_index_file(size_dir / meta.fold, meta.split)
        indices = _load_indices(idx_path)

        # /idx (when present) is authoritative for the row→source-position
        # mapping. Used both when the method dropped cells (fingerprint) and
        # when the method wrote rows in a different order than the split file
        # for write-time efficiency (baselines stream sorted reads).
        recovered_idx = _read_logits_idx(meta.path, log_probs.shape[0])
        if recovered_idx is not None:
            if len(indices) != log_probs.shape[0]:
                log.info(
                    "[idx] %s: split has %d cells but logits has %d; "
                    "using /idx (%d positions) for obs alignment",
                    meta.path, len(indices), log_probs.shape[0], len(recovered_idx),
                )
            indices = recovered_idx
        elif len(indices) != log_probs.shape[0]:
            # Method dropped (or reordered) cells. Look up the embedded row
            # labels as cell IDs in adata.obs.index — that recovers full obs
            # alignment so the file still participates in efficiency-filtered
            # and per-dataset analyses.
            if embedded_row_labels is None or len(embedded_row_labels) != log_probs.shape[0]:
                log.warning(
                    "[row mismatch] %s: split has %d cells but logits has %d; "
                    "no embedded row labels to align — skipping",
                    meta.path, len(indices), log_probs.shape[0],
                )
                return None
            log.warning(
                "[row mismatch] %s: split has %d cells but logits has %d; "
                "aligning via embedded row labels against adata.obs.index",
                meta.path, len(indices), log_probs.shape[0],
            )
            positions = obs.index.get_indexer(embedded_row_labels)
            missing = positions < 0
            n_missing = int(missing.sum())
            if n_missing == len(positions):
                log.warning(
                    "[row mismatch] %s: 0/%d embedded labels found in adata.obs.index "
                    "(they may be class names rather than cell IDs) — skipping",
                    meta.path, len(positions),
                )
                return None
            if n_missing > 0:
                log.warning(
                    "[row mismatch] %s: %d/%d embedded labels not in adata.obs.index "
                    "— dropping those rows",
                    meta.path, n_missing, len(positions),
                )
                keep = ~missing
                log_probs = log_probs[keep]
                positions = positions[keep]
                embedded_row_labels = embedded_row_labels[keep]
            indices = positions

        obs_slice = obs.iloc[indices].reset_index(drop=True)

        # Shapes match → split indices are authoritative; no need to verify
        # against embedded labels (which may be a method-specific 1D dataset
        # that isn't a cell ID list, e.g. cellot stores per-cell perturbation
        # labels in its h5 alongside the logits).
        labels_arr = obs_slice[label_col].to_numpy()
        cols_to_attach = (
            [c for c in obs_cols if c in obs_slice.columns]
            if obs_cols is not None
            else list(obs_slice.columns)
        )
        if cols_to_attach:
            obs_small = obs_slice[cols_to_attach].rename(
                columns={c: f"{OBS_PREFIX}{c}" for c in cols_to_attach}
            )
    else:
        # Embedded row labels in logits files are cell IDs (e.g. cell barcodes),
        # not class labels — we need the adata referenced by --splits-dir to
        # translate them into per-cell class labels. There is no fallback here.
        raise ValueError(
            f"--splits-dir is required to score {meta.path}: embedded row labels are "
            "cell IDs, not class labels, and the adata is needed to map them"
        )

    return FileData(
        log_probs=log_probs,
        classes=list(union_sorted),
        labels=labels_arr,
        obs=obs_small,
        meta=meta,
    )


def _filedata_efficiency_mask(fd: FileData, col: str, min_val: float) -> np.ndarray:
    """Return a boolean row-mask for fd matching the efficiency filter."""
    resolved = col if col in fd.obs.columns else f"{OBS_PREFIX}{col}"
    if resolved not in fd.obs.columns:
        raise KeyError(
            f"filter column {col!r} not found in obs (also tried {resolved!r}); "
            f"available: {list(fd.obs.columns)}"
        )
    vals = pd.to_numeric(fd.obs[resolved], errors="coerce").to_numpy()
    mask = np.isfinite(vals) & (vals >= min_val)
    return mask


def _meta_row(fd: FileData) -> dict:
    return {
        "method": fd.meta.method,
        "split": fd.meta.split,
        "direction": fd.meta.direction,
        "fold": fd.meta.fold,
    }


def _true_rank(log_probs: np.ndarray, true_idx: np.ndarray,
               chunk: int | None = None) -> np.ndarray:
    """0-based rank of the TRUE class in each row, averaged over ties.

    rank = #{classes scoring above the truth} + 0.5 * #{classes tied with it}.
    Tie-splitting is what makes this honest for methods that emit set-level or
    sparse scores: fingerprint leaves every perturbation outside its credible set
    at exactly 0.0, so hundreds of classes tie with the truth. Taking the optimistic
    end of a tie would hand those methods a rank they did not earn (cf.
    frac_tied_at_top in score_one).
    """
    n = log_probs.shape[0]
    chunk = chunk or _LSE_CHUNK        # defined below this point in the module
    ranks = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        blk = log_probs[sl]
        truth = np.take_along_axis(blk, true_idx[sl, None], axis=1)
        above = (blk > truth).sum(axis=1)
        tied = (blk == truth).sum(axis=1) - 1          # minus the true class itself
        ranks[sl] = above + 0.5 * tied
    return ranks


def add_rank_metrics(out: dict, log_probs: np.ndarray, true_idx: np.ndarray,
                     n_classes: int) -> dict:
    """Rank-based scores, which top-k and F1 cannot express.

    recall@k asks one yes/no question at a fixed cut-off, so on a 1,157-class space
    every method that misses the top 10 scores identically 0 -- a method that ranks
    the truth 11th is indistinguishable from one that ranks it 1,100th. These use the
    WHOLE ranking:

      mean_auroc   mean over cells of the share of other classes ranked below the
                   truth. Identical to one-vs-rest ROC-AUC with a single positive per
                   cell, so 0.5 is chance regardless of how many classes there are --
                   directly comparable across small/ctx7/zs, unlike recall@k.
      balanced_mean_auroc
                   the same, averaged over CLASSES rather than cells, matching the
                   balanced_recall@k convention: ood_test's class distribution is
                   the natural one, so the cell-weighted form pays a method for
                   doing well on frequent perturbations.
      mrr          mean reciprocal rank (1-based). Top-weighted, so it rewards
                   getting the truth to rank 1-5 rather than merely out of the tail.
      median_rank  the plain readable number: "the true perturbation sits Nth of C".
    """
    n = log_probs.shape[0]
    if n == 0 or n_classes < 2:
        return out
    ranks = _true_rank(log_probs, true_idx)
    auroc = 1.0 - ranks / (n_classes - 1)
    out["mean_auroc"] = float(auroc.mean())
    out["mrr"] = float((1.0 / (ranks + 1.0)).mean())
    out["median_rank"] = float(np.median(ranks) + 1.0)
    # 0-1 form: 1 at rank 1, 0 at rank C, 0.5 at chance -- the MEDIAN per-cell AUROC
    out["median_rank_scaled"] = float(1.0 - np.median(ranks) / (n_classes - 1))
    sup = np.bincount(true_idx, minlength=n_classes)
    present = sup > 0
    if present.any():
        per_class = np.bincount(true_idx, weights=auroc, minlength=n_classes)
        out["balanced_mean_auroc"] = float((per_class[present] / sup[present]).mean())
    return out


def score_one(fd: FileData, ks: tuple[int, ...]) -> dict:
    col_idx = {c: i for i, c in enumerate(fd.classes)}
    n_classes = len(fd.classes)
    labels = fd.labels
    true_idx = np.fromiter((col_idx.get(c, -1) for c in labels),
                           count=len(labels), dtype=np.int64)
    valid = true_idx >= 0
    log_probs = fd.log_probs if valid.all() else fd.log_probs[valid]
    true_idx = true_idx[valid]
    n = log_probs.shape[0]

    out = _meta_row(fd) | {"n": n}
    if n == 0:
        return out
    max_k = max(ks) if ks else 1
    order = _argtopk_chunked(log_probs, max(max_k, 1))
    out["accuracy"] = float((order[:, 0] == true_idx).mean())
    # How often top-1 was decided by a TIE rather than by the score. Methods that
    # emit set-level or sparse scores (fingerprint: every perturbation outside a
    # susie credible set sits at exactly 0.0) can have hundreds of classes tied at
    # the top, making their argmax an artifact of column order rather than a
    # prediction. Reported so the limitation is visible instead of hidden.
    top_score = np.take_along_axis(log_probs, order[:, :1], axis=1).ravel()
    n_tied = (log_probs == top_score[:, None]).sum(axis=1)
    out["frac_tied_at_top"] = float((n_tied > 1).mean())
    out["mean_n_tied_at_top"] = float(n_tied.mean())
    # recall@k has TWO flavours and they disagree sharply on ood_test:
    #   recall@k          -- over cells, i.e. weighted by each class's cell count
    #   balanced_recall@k -- mean of per-class recall@k over the classes present
    # ood_test is every cell of the held-out datasets, so its class distribution is the
    # natural one (max/median cells per class ~14x at 200 perturbations, ~33x at ctx7).
    # Cell-weighted recall@1 therefore pays a method for concentrating its predictions on
    # frequent classes: measured on fold_0, xgboost scores 2.72x and linreg 1.87x higher
    # cell-weighted than balanced, while ExPert moves 1.19x and centroid not at all.
    # The balanced form is also what a per-group-capped ood_test (benchmark_ood_cap, used
    # from ctx7 on) approximately measures, so it is the metric that lines the tiers up.
    sup = np.bincount(true_idx, minlength=n_classes)
    present = sup > 0
    for k in ks:
        kk = min(k, order.shape[1])
        hits = (order[:, :kk] == true_idx[:, None]).any(axis=1)
        out[f"recall@{k}"] = float(hits.mean())
        per_class = np.bincount(true_idx[hits], minlength=n_classes)[present] / sup[present]
        out[f"balanced_recall@{k}"] = float(per_class.mean()) if present.any() else float("nan")
    out["macro_f1"] = _macro_f1_from_idx(true_idx, order[:, 0], n_classes)
    out["n_classes_present"] = int(present.sum())
    add_rank_metrics(out, log_probs, true_idx, n_classes)

    if OVR_AUROC:
        out["auroc_macro_ovr"], out["auroc_macro_ovr_n_classes"] = _ovr_macro_auroc(
            log_probs, true_idx, n_classes)

    # macro_f1_no_ds_bias: mean of per-dataset present-only macro-F1s.
    # Each dataset's term only averages over the classes that actually
    # appear in its y_true, so panels missing a chunk of the global
    # perturbation pool aren't penalised for the absences. The split-level
    # `macro_f1` above is the union-pool version (matches sklearn default).
    ds_col = f"{OBS_PREFIX}dataset"
    if ds_col in fd.obs.columns:
        datasets = fd.obs[ds_col].to_numpy()
        if len(datasets) != log_probs.shape[0]:
            datasets = datasets[valid]
        pred_idx = order[:, 0]
        per_ds_f1 = []
        for ds in pd.unique(datasets):
            rmask = datasets == ds
            if not rmask.any():
                continue
            per_ds_f1.append(_macro_f1_from_idx(
                true_idx[rmask], pred_idx[rmask],
                n_classes, present_only=True,
            ))
        if per_ds_f1:
            out["macro_f1_no_ds_bias"] = float(np.nanmean(per_ds_f1))
    return out


def add_over_random(out: dict, n_classes: int, ks: tuple[int, ...]) -> dict:
    """Append `<metric>_over_random` for the retrieval metrics.

    On a ~9.9k-class space chance recall@1 is 1.0e-4, so an honest result reads as
    0.0012 and is impossible to eye. The ratio to the analytic uniform floor turns
    that into "12x chance". Analytic rather than the measured `baseline_random`
    method so the column exists even when that method was not run, and so it cannot
    move between runs; `baseline_random` remains the empirical check that the
    analytic floor is right.
    """
    if not n_classes:
        return out
    chance1 = 1.0 / n_classes
    # A uniformly random ranking puts the truth at expected rank (C+1)/2, i.e.
    # auroc 0.5 exactly; MRR's floor is the harmonic number over C, H_C/C.
    harmonic = float(np.log(n_classes) + 0.5772156649) / n_classes
    for key, floor in (("mean_auroc", 0.5), ("balanced_mean_auroc", 0.5),
                       ("mrr", harmonic)):
        if key in out and out[key] == out[key]:
            out[f"{key}_over_random"] = float(out[key] / floor)
    for key, floor in (("accuracy", chance1), ("macro_f1", chance1)):
        if key in out and out[key] == out[key]:          # skip NaN
            out[f"{key}_over_random"] = float(out[key] / floor)
    for k in ks:
        floor_k = min(k / n_classes, 1.0)
        for key in (f"recall@{k}", f"balanced_recall@{k}"):
            if key in out and out[key] == out[key]:
                out[f"{key}_over_random"] = float(out[key] / floor_k)
    out["n_classes_scored"] = int(n_classes)
    out["chance_recall@1"] = float(chance1)
    out["chance_mrr"] = float(harmonic)
    return out


def score_grouped_one(fd: FileData, ks: tuple[int, ...], group_cols: list[str]) -> dict:
    """Score GROUPS of cells instead of single cells.

    A group is one (dataset, perturbation) -- the unit the eval set is stratified on.
    Its score vector is the MEAN LOG-PROB over its cells, i.e. the product of the
    per-cell likelihoods, which is the natural pooling for independent observations
    of the same perturbation in the same context.

    Why this is the primary readout on the padding benchmark: the arm carries ~2
    cells per (dataset, pert) and a median of 6 per perturbation across ~9.9k
    classes, so per-cell balanced recall is dominated by sampling noise. There are
    68,574 groups, and a group is what a biologist would actually call one
    observation of a perturbation in a context.

    Returns {} when the obs columns are absent, so callers can treat it as optional.
    """
    if fd.obs is None or any(c not in fd.obs.columns for c in group_cols):
        return {}
    col_idx = {c: i for i, c in enumerate(fd.classes)}
    true_idx = np.fromiter((col_idx.get(c, -1) for c in fd.labels),
                           count=len(fd.labels), dtype=np.int64)
    valid = true_idx >= 0
    if not valid.any():
        return {}
    lp = fd.log_probs[valid]
    ti = true_idx[valid]
    keys = fd.obs.loc[valid, group_cols].astype(str).agg("\x00".join, axis=1).to_numpy()

    uniq, inv = np.unique(keys, return_inverse=True)
    n_groups, n_classes = len(uniq), len(fd.classes)
    sums = np.zeros((n_groups, n_classes), dtype=np.float64)
    np.add.at(sums, inv, lp.astype(np.float64))
    counts = np.bincount(inv, minlength=n_groups).astype(np.float64)
    gl = (sums / counts[:, None]).astype(np.float32)          # mean log-prob per group
    # a group's truth is constant by construction (perturbation is a group key)
    gt = np.zeros(n_groups, dtype=np.int64)
    gt[inv] = ti

    out = _meta_row(fd) | {"n_groups": n_groups, "n_cells": int(valid.sum()),
                           "group_by": "+".join(group_cols)}
    max_k = max(ks) if ks else 1
    order = _argtopk_chunked(gl, max(max_k, 1))
    out["accuracy"] = float((order[:, 0] == gt).mean())
    add_rank_metrics(out, gl, gt, n_classes)
    for k in ks:
        hits = (order[:, :k] == gt[:, None]).any(axis=1)
        out[f"recall@{k}"] = float(hits.mean())
        per_class = np.full(n_classes, np.nan)
        present = np.bincount(gt, minlength=n_classes) > 0
        if present.any():
            num = np.bincount(gt[hits], minlength=n_classes)[present]
            den = np.bincount(gt, minlength=n_classes)[present]
            per_class[present] = num / den
        out[f"balanced_recall@{k}"] = float(np.nanmean(per_class)) if present.any() else float("nan")
    out["macro_f1"] = _macro_f1_from_idx(gt, order[:, 0], n_classes)
    return add_over_random(out, n_classes, ks)


def score_weighted_one(
    fd: FileData,
    ks: tuple[int, ...],
    weight_col: str,
) -> dict | None:
    """Per-file cell-weighted metrics (weight = fd.obs[weight_col]).

    Computed once over all cells — weighting subsumes the discrete efficiency
    thresholding (a cell with low efficiency contributes proportionally to its
    weight rather than being kept/dropped at a hard cutoff).

    Returns None if the weight column isn't attached (e.g. row-mismatched files).
    """
    resolved = weight_col if weight_col in fd.obs.columns else f"{OBS_PREFIX}{weight_col}"
    if resolved not in fd.obs.columns:
        return None
    weights_all = pd.to_numeric(fd.obs[resolved], errors="coerce").to_numpy()

    col_idx = {c: i for i, c in enumerate(fd.classes)}
    n_classes = len(fd.classes)
    true_idx_all = np.fromiter((col_idx.get(c, -1) for c in fd.labels),
                                count=len(fd.labels), dtype=np.int64)
    valid = (true_idx_all >= 0) & np.isfinite(weights_all) & (weights_all > 0)
    if not valid.any():
        return None

    log_probs = fd.log_probs[valid]
    true_idx = true_idx_all[valid]
    weights = weights_all[valid].astype(np.float64, copy=False)
    n = log_probs.shape[0]

    max_k = max(ks) if ks else 1
    order = _argtopk_chunked(log_probs, max(max_k, 1))

    out = _meta_row(fd) | {
        "n": n,
        "weight_sum": float(weights.sum()),
        "weight_col": resolved,
    }
    correct = (order[:, 0] == true_idx).astype(np.float64)
    out["weighted_accuracy"] = float(np.average(correct, weights=weights))
    out["weighted_macro_f1"] = _weighted_macro_f1_from_idx(
        true_idx, order[:, 0], weights, n_classes,
    )
    # Same two flavours as score_one: weighted_recall@k pools cells (so frequent
    # classes dominate), balanced_weighted_recall@k averages the per-class weighted
    # recalls. weighted_macro_f1 above is already class-balanced.
    wsum = np.bincount(true_idx, weights=weights, minlength=n_classes)
    present = wsum > 0
    for k in ks:
        kk = min(k, order.shape[1])
        hits = (order[:, :kk] == true_idx[:, None]).any(axis=1).astype(np.float64)
        out[f"weighted_recall@{k}"] = float(np.average(hits, weights=weights))
        hit_w = np.bincount(true_idx, weights=hits * weights, minlength=n_classes)
        out[f"balanced_weighted_recall@{k}"] = (
            float((hit_w[present] / wsum[present]).mean()) if present.any() else float("nan"))
    # Rank metrics in the weighted table too (add_rank_metrics covers the unweighted
    # one in score_one). Same definitions, cell weights applied.
    if n_classes > 1:
        ranks = _true_rank(log_probs, true_idx)
        auroc = 1.0 - ranks / (n_classes - 1)
        rr = 1.0 / (ranks + 1.0)
        out["weighted_mean_auroc"] = float(np.average(auroc, weights=weights))
        out["weighted_mrr"] = float(np.average(rr, weights=weights))
        num = np.bincount(true_idx, weights=auroc * weights, minlength=n_classes)
        if present.any():
            out["balanced_weighted_mean_auroc"] = float((num[present] / wsum[present]).mean())
        # weighted median: the rank at which half the total cell weight is reached
        o = np.argsort(ranks, kind="stable")
        cw = np.cumsum(weights[o])
        out["weighted_median_rank"] = float(ranks[o][np.searchsorted(cw, 0.5 * cw[-1])] + 1.0)
        out["weighted_median_rank_scaled"] = float(
            1.0 - (out["weighted_median_rank"] - 1.0) / (n_classes - 1))
    return out


def score_per_class_one(
    fd: FileData,
    ks: tuple[int, ...],
    drop_zero_support: bool = True,
) -> pd.DataFrame:
    col_idx = {c: i for i, c in enumerate(fd.classes)}
    n_classes = len(fd.classes)
    true_idx = np.fromiter((col_idx.get(c, -1) for c in fd.labels),
                           count=len(fd.labels), dtype=np.int64)
    valid = true_idx >= 0
    log_probs = fd.log_probs if valid.all() else fd.log_probs[valid]
    true_idx = true_idx[valid]
    if log_probs.shape[0] == 0:
        return pd.DataFrame()

    max_k = max(ks) if ks else 1
    order = _argtopk_chunked(log_probs, max(max_k, 1))
    precision, recall, f1, support = _perclass_prf_from_idx(
        true_idx, order[:, 0], n_classes,
    )

    topk = {}
    for k in ks:
        kk = min(k, order.shape[1])
        hits = (order[:, :kk] == true_idx[:, None]).any(axis=1)
        cls_hits = np.bincount(true_idx[hits], minlength=n_classes)
        cls_total = np.bincount(true_idx, minlength=n_classes)
        rk = np.zeros(n_classes, dtype=np.float64)
        np.divide(cls_hits, cls_total, out=rk, where=cls_total > 0)
        topk[k] = rk

    base = _meta_row(fd)
    rows = []
    for ci, cls in enumerate(fd.classes):
        ss = int(support[ci])
        if drop_zero_support and ss == 0:
            continue
        rec = base | {
            "class": cls,
            "precision": float(precision[ci]),
            "recall": float(recall[ci]),
            "f1": float(f1[ci]),
            "support": ss,
        }
        for k in ks:
            rec[f"recall@{k}"] = float(topk[k][ci]) if ss > 0 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def score_per_dataset_class_one(
    fd: FileData, dataset_col: str = "dataset",
) -> pd.DataFrame:
    if VAL_SPLIT_RE.search(fd.meta.split):
        return pd.DataFrame()
    resolved = f"{OBS_PREFIX}{dataset_col}"
    if resolved not in fd.obs.columns:
        return pd.DataFrame()

    col_idx = {c: i for i, c in enumerate(fd.classes)}
    n_classes = len(fd.classes)
    datasets = fd.obs[resolved].to_numpy()

    rows = []
    for ds in pd.unique(datasets):
        rmask = datasets == ds
        sub_labels = fd.labels[rmask]
        sub_lp = fd.log_probs[rmask]
        true_idx = np.fromiter((col_idx.get(c, -1) for c in sub_labels),
                               count=len(sub_labels), dtype=np.int64)
        valid = true_idx >= 0
        if not valid.any():
            continue
        sub_lp = sub_lp[valid]
        true_idx = true_idx[valid]
        pred_idx = sub_lp.argmax(axis=1)
        precision, recall, f1, support = _perclass_prf_from_idx(
            true_idx, pred_idx, n_classes,
        )
        for ci, cls in enumerate(fd.classes):
            ss = int(support[ci])
            if ss == 0:
                continue
            rows.append({
                "method": fd.meta.method,
                "dataset": str(ds),
                "class": cls,
                "precision": float(precision[ci]),
                "recall": float(recall[ci]),
                "f1": float(f1[ci]),
                "support": ss,
            })
    return pd.DataFrame(rows)


def score_per_dataset_one(
    fd: FileData, dataset_col: str = "dataset",
) -> pd.DataFrame:
    """Per (method, dataset) accuracy + macro-F1 on non-val cells.

    macro-F1 is computed with present_only=True, so the mean is taken only
    over classes that actually appear in the dataset's y_true. Without that,
    a dataset with 5k of the global 7k perts gets penalized for the 2k it
    never had — every absent class contributes F1=0 from FPs (cells the
    model still predicted to that class), which drags the average down on
    smaller-panel datasets even when the surviving-class predictions are
    fine."""
    if VAL_SPLIT_RE.search(fd.meta.split):
        return pd.DataFrame()
    resolved = f"{OBS_PREFIX}{dataset_col}"
    if resolved not in fd.obs.columns:
        return pd.DataFrame()

    col_idx = {c: i for i, c in enumerate(fd.classes)}
    n_classes = len(fd.classes)
    datasets = fd.obs[resolved].to_numpy()

    rows = []
    for ds in pd.unique(datasets):
        rmask = datasets == ds
        sub_labels = fd.labels[rmask]
        sub_lp = fd.log_probs[rmask]
        true_idx = np.fromiter((col_idx.get(c, -1) for c in sub_labels),
                               count=len(sub_labels), dtype=np.int64)
        valid = true_idx >= 0
        if not valid.any():
            continue
        sub_lp = sub_lp[valid]
        true_idx = true_idx[valid]
        pred_idx = sub_lp.argmax(axis=1)
        rows.append({
            "method": fd.meta.method,
            "split": fd.meta.split,
            "dataset": str(ds),
            "n": int(valid.sum()),
            "n_classes_present": int(np.unique(true_idx).size),
            "accuracy": float((pred_idx == true_idx).mean()),
            "macro_f1": _macro_f1_from_idx(
                true_idx, pred_idx, n_classes, present_only=True,
            ),
        })
    return pd.DataFrame(rows)


def cells_per_perturbation_one(fd: FileData) -> pd.DataFrame:
    s = pd.Series(fd.labels).value_counts()
    return pd.DataFrame({
        "label": s.index.to_numpy(),
        "n_cells": s.values,
        "split": fd.meta.split,
    })


def score_group_level_one(
    fd: FileData,
    ks: tuple[int, ...],
    aggregation: str = "mean_log_prob",
    dataset_col: str = "dataset",
) -> dict | None:
    resolved = f"{OBS_PREFIX}{dataset_col}"
    if resolved not in fd.obs.columns:
        return None
    col_idx = {c: i for i, c in enumerate(fd.classes)}
    n_classes = len(fd.classes)

    grp_df = pd.DataFrame({"d": fd.obs[resolved].to_numpy(), "l": fd.labels})
    agg_vecs: list[np.ndarray] = []
    true_labels: list[str] = []
    for (_, lbl), idx in grp_df.groupby(["d", "l"], sort=False, observed=True).groups.items():
        rows = idx.to_numpy()
        sub_lp = fd.log_probs[rows]
        if aggregation == "mean_log_prob":
            vec = sub_lp.mean(axis=0)
        elif aggregation == "mean_prob":
            probs = np.exp(sub_lp)
            vec = np.log(np.clip(probs.mean(axis=0), 1e-30, None))
        elif aggregation == "majority":
            top1 = sub_lp.argmax(axis=1)
            vec = np.bincount(top1, minlength=n_classes).astype(np.float32)
        else:
            raise ValueError(f"unknown aggregation: {aggregation!r}")
        agg_vecs.append(vec)
        true_labels.append(str(lbl))
    if not agg_vecs:
        return None
    agg_mat = np.stack(agg_vecs)
    true_idx = np.array([col_idx.get(t, -1) for t in true_labels])
    valid = true_idx >= 0
    if not valid.any():
        return None
    agg_mat = agg_mat[valid]
    true_idx = true_idx[valid]

    order = (-agg_mat).argsort(axis=1)
    out = _meta_row(fd) | {
        "n_groups": int(valid.sum()),
        "accuracy": float((order[:, 0] == true_idx).mean()),
        "macro_f1": _macro_f1_from_idx(true_idx, order[:, 0], n_classes),
    }
    for k in ks:
        kk = min(k, order.shape[1])
        out[f"recall@{k}"] = float((order[:, :kk] == true_idx[:, None]).any(axis=1).mean())
    # rank metrics on the pooled group vectors too, so auroc_group / median_rank_group
    # sit beside macro_f1_group
    add_rank_metrics(out, agg_mat, true_idx, n_classes)
    return out


def collect(
    source: Path,
    method_depth: int = 1,
    splits_dir: Path | None = None,
    dataset_size: str = "small",
    fold: str | None = None,
    label_col: str = "perturbation",
    obs_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    source = Path(source).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)

    # os.walk(followlinks=True) instead of Path.rglob: result dirs may be
    # symlinks (e.g. to scratch storage) and rglob does not follow
    # symlinks across boundaries.
    pattern_set = set(LOGIT_PATTERNS)
    all_files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(source, followlinks=True):
        for fn in filenames:
            if fn in pattern_set:
                all_files.append(Path(dirpath) / fn)
    all_files = sorted(set(all_files))
    if not all_files:
        raise FileNotFoundError(
            f"no {'/'.join(LOGIT_PATTERNS)} under {source}"
        )

    # filter by dataset_size (must appear as a path segment) and --fold if set
    filtered: list[tuple[Path, str | None]] = []
    for f in all_files:
        if dataset_size and dataset_size not in f.parts:
            continue
        file_fold = _detect_fold(f)
        if fold is not None and file_fold != fold:
            continue
        filtered.append((f, file_fold))
    if not filtered:
        raise FileNotFoundError(
            f"no logits.csv matched dataset_size={dataset_size!r} fold={fold!r} under {source}"
        )
    log.info(
        "[collect] %d/%d logits.csv files match (size=%s, fold=%s)",
        len(filtered), len(all_files), dataset_size, fold,
    )

    obs: pd.DataFrame | None = None
    size_dir: Path | None = None
    if splits_dir is not None:
        splits_dir = Path(splits_dir).resolve()
        size_dir = splits_dir / dataset_size
        meta = json.loads((size_dir / "metadata.json").read_text())
        source_h5ad = Path(meta["source_path"])
        import anndata as ad  # lazy: only needed when splits_dir is set
        adata = ad.read_h5ad(source_h5ad, backed="r")
        obs = adata.obs
        if label_col not in obs.columns:
            raise ValueError(
                f"label_col={label_col!r} missing from obs (have: {list(obs.columns)})"
            )
        log.info(
            "[collect] loaded obs from %s (%d rows, %d cols); size=%s label_col=%s",
            source_h5ad, obs.shape[0], obs.shape[1], dataset_size, label_col,
        )

    groups: dict[tuple[str, str, str | None], list[tuple[Path, pd.DataFrame, str | None]]] = {}
    for f, file_fold in filtered:
        rel_parts = f.relative_to(source).parts
        if len(rel_parts) < method_depth + 2:
            raise ValueError(
                f"{f} is too shallow for --method-depth={method_depth} "
                f"(relative parts: {rel_parts})"
            )
        if method_depth >= 2:
            direction = rel_parts[0]
            method = "/".join(rel_parts[1:method_depth])
        else:
            direction = None
            method = rel_parts[0]
        split = _canonical_split(f.parent.name)

        logits_df = _read_logits(f)
        embedded_labels = (
            None if isinstance(logits_df.index, pd.RangeIndex)
            else logits_df.index.astype(str).to_numpy()
        )
        mat = logits_df.to_numpy(dtype=np.float32)
        normalised, kind = _to_log_probs(mat)
        log.info("[collect] %s: detected %s (fold=%s)", f, kind, file_fold)
        # store as float32 — log-probs don't need 8-byte precision and float32
        # halves the memory footprint of the stacked frame, which matters a lot
        # at full-dataset scale (millions of cells × thousands of classes).
        logits_df = pd.DataFrame(
            normalised.astype(np.float32, copy=False),
            columns=logits_df.columns,
        )

        if obs is not None:
            if file_fold is None:
                raise ValueError(
                    f"{f} has no fold_N ancestor; required when --splits-dir is set"
                )
            fold_dir = size_dir / file_fold
            idx_path = _resolve_index_file(fold_dir, split)
            indices = _load_indices(idx_path)
            if len(indices) != len(logits_df):
                raise ValueError(
                    f"split index count {len(indices)} (from {idx_path}) != logits rows "
                    f"{len(logits_df)} for {f}"
                )
            obs_slice = obs.iloc[indices].reset_index(drop=True)

            # cross-check embedded row labels against adata.obs (no labels.csv reliance)
            # Shapes match → trust split indices; no embedded-labels cross-check.
            logits_df["label"] = obs_slice[label_col].astype(str).to_numpy()
            cols_to_attach = (
                [c for c in obs_cols if c in obs_slice.columns]
                if obs_cols is not None
                else list(obs_slice.columns)
            )
            for col in cols_to_attach:
                logits_df[f"{OBS_PREFIX}{col}"] = obs_slice[col].to_numpy()
        else:
            # See load_one_file for the same reasoning: embedded row labels are
            # cell IDs, so the adata is required to resolve their class labels.
            raise ValueError(
                f"--splits-dir is required to score {f}: embedded row labels are "
                "cell IDs, not class labels, and the adata is needed to map them"
            )

        groups.setdefault((method, split, direction), []).append((f, logits_df, file_fold))

    # shape + column consistency within each (method, split, fold).
    # Across folds, row counts can differ legitimately — only column consistency
    # is enforced at the (method, split) level by the loop below.
    by_fold: dict[tuple[str, str, str | None, str | None], tuple[Path, pd.DataFrame]] = {}
    drop_files: set[tuple] = set()  # (method, split, direction, fold, source_path)
    for (method, split, direction), items in groups.items():
        for p, df, file_fold in items:
            k = (method, split, direction, file_fold)
            if k not in by_fold:
                by_fold[k] = (p, df)
                continue
            ref_path, ref_df = by_fold[k]
            if df.shape != ref_df.shape:
                log.warning(
                    "[drop file] shape mismatch for method=%s split=%s fold=%s: "
                    "%s is %s but %s is %s — dropping the latter",
                    method, split, file_fold, ref_path, ref_df.shape, p, df.shape,
                )
                drop_files.add((method, split, direction, file_fold, p))
                continue
            if df.columns.tolist() != ref_df.columns.tolist():
                log.warning(
                    "[drop file] column mismatch for method=%s split=%s fold=%s: "
                    "%s vs %s — dropping the latter",
                    method, split, file_fold, ref_path, p,
                )
                drop_files.add((method, split, direction, file_fold, p))
    if drop_files:
        for (m, s, d), items in list(groups.items()):
            kept = [(p, df, f) for (p, df, f) in items
                    if (m, s, d, f, p) not in drop_files]
            if kept:
                groups[(m, s, d)] = kept
            else:
                groups.pop((m, s, d))

    # label-space consistency across splits within a method (scoped by direction).
    # If splits within a (direction, method) disagree on logit columns, pad each
    # group to the union with -inf so missing classes get zero probability after
    # softmax (rather than dropping the offending split). Stored values are
    # already log-probs, so -inf doesn't break normalisation.
    PAD_VALUE = float("-inf")

    def _logit_set(df: pd.DataFrame) -> set[str]:
        return {c for c in df.columns if c not in META_COLS and not c.startswith(OBS_PREFIX)}

    method_unions: dict[tuple[str | None, str], set[str]] = {}
    for (method, _split, direction), items in groups.items():
        method_unions.setdefault((direction, method), set()).update(_logit_set(items[0][1]))

    for (method, split, direction), items in list(groups.items()):
        union = method_unions[(direction, method)]
        union_sorted = sorted(union)
        new_items = []
        for p, df, fold in items:
            present = _logit_set(df)
            missing = union - present
            if missing:
                log.warning(
                    "[pad] method=%s split=%s direction=%s fold=%s: padding %d/%d "
                    "missing classes with -inf (log-prob = 0): %s%s",
                    method, split, direction, fold,
                    len(missing), len(union),
                    sorted(missing)[:5], "..." if len(missing) > 5 else "",
                )
                pad = pd.DataFrame(
                    PAD_VALUE, index=df.index, columns=sorted(missing),
                )
                df = pd.concat([df, pad], axis=1, copy=False)
                meta_present = [c for c in df.columns
                                if c in META_COLS or c.startswith(OBS_PREFIX)]
                df = df[union_sorted + meta_present]
            new_items.append((p, df, fold))
        groups[(method, split, direction)] = new_items

    # cross-method label-space consistency: every method has to predict over the
    # same set of classes, otherwise per-method scoring is comparing apples to
    # oranges (and concat would inject NaN columns). Order may differ; only the
    # set is enforced. Mismatched methods are dropped wholesale (all of their
    # (split, direction) groups removed).
    ref_set: set[str] | None = None
    ref_origin: tuple[str, Path] | None = None
    drop_methods: set[str] = set()
    for (method, split, direction), items in groups.items():
        if method in drop_methods:
            continue
        p, df, _ = items[0]
        logit_cols = {c for c in df.columns if c not in META_COLS and not c.startswith(OBS_PREFIX)}
        if ref_set is None:
            ref_set = logit_cols
            ref_origin = (method, p)
            continue
        if logit_cols != ref_set:
            only_ref = sorted(ref_set - logit_cols)
            only_new = sorted(logit_cols - ref_set)
            ref_method, ref_path = ref_origin
            log.warning(
                "[drop method] logit column-name mismatch — dropping method=%s.\n"
                "  reference: method=%s (%s) — %d classes\n"
                "  this:      method=%s (%s) — %d classes\n"
                "  only in reference (%d): %s%s\n"
                "  only in this      (%d): %s%s",
                method,
                ref_method, ref_path, len(ref_set),
                method, p, len(logit_cols),
                len(only_ref), only_ref[:10], "..." if len(only_ref) > 10 else "",
                len(only_new), only_new[:10], "..." if len(only_new) > 10 else "",
            )
            drop_methods.add(method)
    if drop_methods:
        for k in [k for k in groups if k[0] in drop_methods]:
            groups.pop(k)

    if not groups:
        raise RuntimeError(
            "no logits left after consistency filtering; check warnings above"
        )

    frames = []
    for (method, split, direction), items in groups.items():
        for p, df, fold in items:
            tagged = df.copy()
            tagged["split"] = split
            tagged["direction"] = direction
            tagged["method"] = method
            tagged["source_path"] = str(p)
            tagged["fold"] = fold
            frames.append(tagged)
    return pd.concat(frames, ignore_index=True)


def _logit_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not c.startswith(OBS_PREFIX)]


_LSE_CHUNK = 4096  # rows per chunk for memory-bounded reductions


def _log_softmax_inplace(x: np.ndarray, chunk: int = _LSE_CHUNK) -> None:
    """Row-wise log-softmax, modifying `x` in place. Peak temp ≈ chunk × n_cols."""
    for i in range(0, x.shape[0], chunk):
        sl = slice(i, min(i + chunk, x.shape[0]))
        block = x[sl]
        m = block.max(axis=1, keepdims=True)
        block -= m                              # in-place
        e = np.exp(block)                       # one chunk-sized temp
        s = e.sum(axis=1, keepdims=True)
        np.log(s, out=s)
        block -= s                              # in-place


def _log_softmax(x: np.ndarray, chunk: int = _LSE_CHUNK) -> np.ndarray:
    """Allocating wrapper around `_log_softmax_inplace`."""
    out = np.array(x, copy=True)
    _log_softmax_inplace(out, chunk=chunk)
    return out


def _row_logsumexp(x: np.ndarray, chunk: int = _LSE_CHUNK) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(0, x.shape[0], chunk):
        sl = slice(i, min(i + chunk, x.shape[0]))
        block = x[sl]
        m = block.max(axis=1, keepdims=True)
        e = np.exp(block - m)
        out[sl] = (m + np.log(e.sum(axis=1, keepdims=True))).ravel()
    return out


def _to_log_probs(mat: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, str]:
    """Return (log-prob matrix, detected kind).

    - already log-probs   → row logsumexp ≈ 0
    - probabilities       → all values in [0, 1] and row sums ≈ 1 → take log
    - otherwise raw logits → apply log-softmax
    """
    lse = _row_logsumexp(mat)
    if np.max(np.abs(lse)) < tol:
        return mat, "log_probs"
    if mat.min() >= -tol and mat.max() <= 1 + tol:
        row_sums = mat.sum(axis=1)
        if np.max(np.abs(row_sums - 1.0)) < tol:
            return np.log(np.clip(mat, 1e-30, None)), "probs"
    return _log_softmax(mat), "raw_logits"


def _macro_f1_from_idx(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    n_classes: int,
    present_only: bool = False,
) -> float:
    """Macro F1 from integer class indices — orders of magnitude faster than
    sklearn.f1_score(..., average='macro') when n_classes is in the thousands.

    By default matches sklearn semantics: classes with 0 support contribute
    F1=0 to the mean (denominator = n_classes).

    With present_only=True, the mean is taken over classes that have at
    least one true-label cell in this y_true — i.e., classes actually present
    in the slice. Use this when scoring a subset that doesn't contain the
    full class pool (e.g. a per-dataset slice when datasets carry different
    perturbation panels) and you don't want absent classes to drag the mean
    down. False positives still penalize the F1 of present classes; only
    classes with support == 0 are excluded from the average.
    """
    correct = y_true_idx == y_pred_idx
    tp = np.bincount(y_true_idx[correct], minlength=n_classes)
    sup = np.bincount(y_true_idx, minlength=n_classes)
    pred = np.bincount(y_pred_idx, minlength=n_classes)
    fp = pred - tp
    fn = sup - tp
    denom = 2 * tp + fp + fn
    f1 = np.zeros(n_classes, dtype=np.float64)
    nz = denom > 0
    f1[nz] = (2 * tp[nz]) / denom[nz]
    if present_only:
        keep = sup > 0
        if not keep.any():
            return float("nan")
        return float(f1[keep].mean())
    return float(f1.mean())


def _perclass_prf_from_idx(
    y_true_idx: np.ndarray, y_pred_idx: np.ndarray, n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class (precision, recall, F1, support) via numpy bincount."""
    correct = y_true_idx == y_pred_idx
    tp = np.bincount(y_true_idx[correct], minlength=n_classes).astype(np.int64)
    sup = np.bincount(y_true_idx, minlength=n_classes).astype(np.int64)
    pred = np.bincount(y_pred_idx, minlength=n_classes).astype(np.int64)
    fp = pred - tp
    fn = sup - tp
    precision = np.zeros(n_classes, dtype=np.float64)
    recall = np.zeros(n_classes, dtype=np.float64)
    f1 = np.zeros(n_classes, dtype=np.float64)
    pred_pos = tp + fp
    real_pos = tp + fn
    np.divide(tp, pred_pos, out=precision, where=pred_pos > 0)
    np.divide(tp, real_pos, out=recall, where=real_pos > 0)
    pr_sum = precision + recall
    np.divide(2 * precision * recall, pr_sum, out=f1, where=pr_sum > 0)
    return precision, recall, f1, sup


def _weighted_perclass_prf_from_idx(
    y_true_idx: np.ndarray, y_pred_idx: np.ndarray,
    weights: np.ndarray, n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision / recall / F1 / weighted-support with sample weights.

    Per-class TP/FP/FN are computed by summing weights instead of counts:
        wTP[c] = sum(w_i)  where y_true_i == c & y_pred_i == c
        wFP[c] = sum(w_i)  where y_true_i != c & y_pred_i == c
        wFN[c] = sum(w_i)  where y_true_i == c & y_pred_i != c
    The returned `support` is the weighted true-positive support per class
    (wTP + wFN) — useful as the secondary weight when macro-averaging.
    """
    weights = weights.astype(np.float64, copy=False)
    correct = y_true_idx == y_pred_idx
    wtp = np.bincount(y_true_idx[correct], weights=weights[correct], minlength=n_classes)
    wsup = np.bincount(y_true_idx, weights=weights, minlength=n_classes)
    wpred = np.bincount(y_pred_idx, weights=weights, minlength=n_classes)
    wfp = wpred - wtp
    wfn = wsup - wtp
    precision = np.zeros(n_classes, dtype=np.float64)
    recall = np.zeros(n_classes, dtype=np.float64)
    f1 = np.zeros(n_classes, dtype=np.float64)
    pred_pos = wtp + wfp
    real_pos = wtp + wfn
    np.divide(wtp, pred_pos, out=precision, where=pred_pos > 0)
    np.divide(wtp, real_pos, out=recall, where=real_pos > 0)
    pr_sum = precision + recall
    np.divide(2 * precision * recall, pr_sum, out=f1, where=pr_sum > 0)
    return precision, recall, f1, wsup


def _weighted_macro_f1_from_idx(
    y_true_idx: np.ndarray, y_pred_idx: np.ndarray,
    weights: np.ndarray, n_classes: int,
) -> float:
    """Class-weighted macro F1: per-class weighted F1 averaged by weighted support.

    Plain mean over classes would dilute the cell-weighting because every
    class gets one vote regardless of how much real perturbation effect it
    carries. Weighting by per-class total weighted support means high-effect
    classes dominate the score — matching the per-cell weighting intent.
    """
    _, _, f1, sup = _weighted_perclass_prf_from_idx(
        y_true_idx, y_pred_idx, weights, n_classes,
    )
    keep = (sup > 0) & np.isfinite(f1)
    if not keep.any():
        return 0.0
    return float(np.average(f1[keep], weights=sup[keep]))


TIE_BREAK_SEED = 0   # see global_sorted in apply_consistency()
# None = take the cross-method union (historical behaviour). A list pins the label
# space; set from --classes. See the FIXED_CLASSES block in apply_consistency().
FIXED_CLASSES: list[str] | None = None
# Macro one-vs-rest AUROC is a sort per class per file, so it is opt-in (--ovr-auroc).
# The per-cell rank metrics (add_rank_metrics: mean_auroc/mrr) are always on --
# they cost one comparison pass over the matrix the metrics already load.
OVR_AUROC: bool = False


def _argtopk_chunked(mat: np.ndarray, k: int, chunk: int = _LSE_CHUNK) -> np.ndarray:
    """Return (n_rows, k) top-k column indices in descending-score order, row-chunked.

    The naive `(-mat).argsort(axis=1)` allocates a full int64 array of the
    matrix shape (= 2× the float32 matrix size), which OOMs at full scale.
    This processes `chunk` rows at a time so peak temp ≈ chunk × n_cols × 8B.
    """
    n_rows, n_cols = mat.shape
    k = min(k, n_cols)
    out = np.empty((n_rows, k), dtype=np.int64)
    for i in range(0, n_rows, chunk):
        sl = slice(i, min(i + chunk, n_rows))
        block = mat[sl]
        # argpartition gets us top-k unsorted within the partition, then we
        # sort just those k columns — much cheaper than sorting all n_cols.
        if k < n_cols:
            part = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
        else:
            part = np.argsort(-block, axis=1)
            out[sl] = part
            continue
        scores = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-scores, axis=1)
        out[sl] = np.take_along_axis(part, order, axis=1)
    return out


def _ovr_macro_auroc(mat: np.ndarray, true_idx: np.ndarray, n_classes: int) -> tuple[float, int]:
    """Macro one-vs-rest AUROC: per class, do ITS cells score higher on ITS column
    than other cells do? Averaged over classes present.

    A different question from auc_rank, which ranks classes within one cell. This one
    ranks cells within one class, so it also penalises a method whose per-class scores
    are not comparable across cells. Opt-in (--ovr-auroc): it is a sort per class.
    """
    from scipy.stats import rankdata
    n = mat.shape[0]
    sup = np.bincount(true_idx, minlength=n_classes)
    aucs = []
    for c in range(n_classes):
        npos = int(sup[c])
        if npos == 0 or npos == n:
            continue
        r = rankdata(mat[:, c])                       # mid-ranks over cells
        s_pos = r[true_idx == c].sum()
        aucs.append((s_pos - npos * (npos + 1) / 2.0) / (npos * (n - npos)))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def _to_log_probs_inplace(mat: np.ndarray, tol: float = 1e-3) -> str:
    """Convert `mat` to log-probs in place. Returns the detected kind.

    Used in the streaming path where avoiding a 2× peak (input + output) for
    the full per-file matrix is the difference between fitting and OOMing.
    The "probabilities" branch can't avoid an allocation cleanly, so it falls
    back to a copy in that rare case.
    """
    lse = _row_logsumexp(mat)
    if np.max(np.abs(lse)) < tol:
        return "log_probs"
    if mat.min() >= -tol and mat.max() <= 1 + tol:
        row_sums = mat.sum(axis=1)
        if np.max(np.abs(row_sums - 1.0)) < tol:
            np.clip(mat, 1e-30, None, out=mat)
            np.log(mat, out=mat)
            return "probs"
    _log_softmax_inplace(mat)
    return "raw_logits"


def score(
    df: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10, 50),
    splits: tuple[str, ...] | None = None,
    knn: bool = False,
    knn_k: int = 5,
) -> pd.DataFrame:
    """Per (method, split): accuracy, macro F1, recall@k, NLL, optional kNN acc.

    `df` is expected to come from `collect()`, which normalises each file to
    log-probs on read, so NLL is just -log_prob[true_class].
    """
    from sklearn.neighbors import KNeighborsClassifier

    logit_cols = _logit_cols(df)
    col_idx = {c: i for i, c in enumerate(logit_cols)}

    # pre-compute kNN refs per method: the unique val-like split, if any
    knn_refs: dict[str, str] = {}
    if knn:
        for method, method_df in df.groupby("method", sort=False):
            vals = [s for s in method_df["split"].unique() if VAL_SPLIT_RE.search(s)]
            if len(vals) == 1:
                knn_refs[method] = vals[0]
            else:
                log.info(
                    "[knn] skipping method=%s: found %d val-like splits %s",
                    method, len(vals), vals,
                )

    has_direction = "direction" in df.columns

    rows = []
    for (method, split), group in df.groupby(["method", "split"], sort=False):
        if splits is not None and split not in splits:
            continue
        y_true = group["label"].to_numpy()
        missing = set(y_true) - col_idx.keys()
        if missing:
            raise ValueError(
                f"labels for method={method} split={split} include classes not in logit columns: "
                f"{sorted(missing)[:5]}..."
            )
        log_probs = group[logit_cols].to_numpy(dtype=np.float32, copy=False)
        true_idx = np.array([col_idx[c] for c in y_true])
        max_k = max(ks) if ks else 1
        order = _argtopk_chunked(log_probs, max(max_k, 1))

        out: dict[str, float | str | int | None] = {"method": method, "split": split, "n": len(group)}
        if has_direction:
            out["direction"] = group["direction"].iloc[0]
        out["accuracy"] = float((order[:, 0] == true_idx).mean())
        for k in ks:
            kk = min(k, order.shape[1])
            hits = (order[:, :kk] == true_idx[:, None]).any(axis=1)
            out[f"recall@{k}"] = float(hits.mean())

        out["macro_f1"] = _macro_f1_from_idx(true_idx, order[:, 0], len(logit_cols))

        if knn and method in knn_refs and split != knn_refs[method]:
            ref_split = knn_refs[method]
            ref_rows = df[(df["method"] == method) & (df["split"] == ref_split)]
            X_ref = ref_rows[logit_cols].to_numpy()
            y_ref = ref_rows["label"].to_numpy()
            clf = KNeighborsClassifier(n_neighbors=knn_k, n_jobs=-1)
            clf.fit(X_ref, y_ref)
            out["knn_acc"] = float((clf.predict(log_probs) == y_true).mean())
            out["knn_ref_split"] = ref_split
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    sort_cols = ["split", "direction", "method"] if has_direction else ["split", "method"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def score_per_fold(
    df: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10),
    splits: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Per (method, split, fold) macro F1, accuracy, recall@k — for cross-fold distribution plots."""
    if "fold" not in df.columns:
        raise ValueError("score_per_fold requires a 'fold' column on df")
    logit_cols = _logit_cols(df)
    col_idx = {c: i for i, c in enumerate(logit_cols)}
    has_direction = "direction" in df.columns
    max_k = max(ks) if ks else 1
    rows = []
    for (method, split, fold), group in df.groupby(["method", "split", "fold"], sort=False):
        if splits is not None and split not in splits:
            continue
        y_true = group["label"].astype(str).to_numpy()
        logits = group[logit_cols].to_numpy(dtype=np.float32, copy=False)
        true_idx = np.array([col_idx[c] for c in y_true])
        order = _argtopk_chunked(logits, max(max_k, 1))
        rec: dict[str, float | str | int | None] = {
            "method": method,
            "split": split,
            "fold": fold,
            "macro_f1": _macro_f1_from_idx(true_idx, order[:, 0], len(logit_cols)),
            "accuracy": float((order[:, 0] == true_idx).mean()),
            "n": len(group),
        }
        for k in ks:
            kk = min(k, order.shape[1])
            hits = (order[:, :kk] == true_idx[:, None]).any(axis=1)
            rec[f"recall@{k}"] = float(hits.mean())
        if has_direction:
            rec["direction"] = group["direction"].iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def score_per_class(
    df: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10, 100),
    splits: tuple[str, ...] | None = None,
    drop_zero_support: bool = True,
) -> pd.DataFrame:
    """Long-format per-class precision/recall/F1/support/recall@k.

    Per-class recall@k is the fraction of rows whose true class is in the top-k
    logits, computed only over rows where y_true == class (i.e. support > 0).
    Almost free: argsort is shared with the top-1 computation.
    """
    logit_cols = _logit_cols(df)
    col_idx = {c: i for i, c in enumerate(logit_cols)}
    has_direction = "direction" in df.columns
    max_k = max(ks) if ks else 1

    rows = []
    for (method, split), group in df.groupby(["method", "split"], sort=False):
        if splits is not None and split not in splits:
            continue
        y_true = group["label"].to_numpy()
        log_probs = group[logit_cols].to_numpy(dtype=np.float32, copy=False)
        order = _argtopk_chunked(log_probs, max(max_k, 1))  # (n, max_k)
        true_idx_arr = np.fromiter((col_idx.get(c, -1) for c in y_true),
                                    count=len(y_true), dtype=np.int64)
        # samples whose true class isn't in the logit space contribute to
        # accuracy/recall but not per-class P/R/F (no class to attribute to).
        valid = true_idx_arr >= 0
        n_classes = len(logit_cols)
        # vectorised per-class P/R/F
        precision, recall, f1, support = _perclass_prf_from_idx(
            true_idx_arr[valid], order[valid, 0], n_classes,
        )
        # per-class top-k recall: count how many of class c's samples have c in top-k
        topk_recall_per_class: dict[int, np.ndarray] = {}
        for k in ks:
            kk = min(k, order.shape[1])
            hits = (order[valid, :kk] == true_idx_arr[valid, None]).any(axis=1)
            cls_hits = np.bincount(true_idx_arr[valid][hits], minlength=n_classes)
            cls_total = np.bincount(true_idx_arr[valid], minlength=n_classes)
            tk = np.zeros(n_classes, dtype=np.float64)
            np.divide(cls_hits, cls_total, out=tk, where=cls_total > 0)
            topk_recall_per_class[k] = tk

        direction = group["direction"].iloc[0] if has_direction else None
        for ci, cls in enumerate(logit_cols):
            ss = int(support[ci])
            if drop_zero_support and ss == 0:
                continue
            rec = {
                "method": method,
                "split": split,
                "class": cls,
                "precision": float(precision[ci]),
                "recall": float(recall[ci]),
                "f1": float(f1[ci]),
                "support": ss,
            }
            for k in ks:
                rec[f"recall@{k}"] = (
                    float(topk_recall_per_class[k][ci]) if ss > 0 else float("nan")
                )
            if has_direction:
                rec["direction"] = direction
            rows.append(rec)
    return pd.DataFrame(rows)


def score_group_level(
    df: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10),
    splits: tuple[str, ...] | None = None,
    group_cols: tuple[str, ...] = (f"{OBS_PREFIX}dataset", "label"),
    aggregation: str = "mean_log_prob",
) -> pd.DataFrame:
    """Per (method, split) group-level metrics.

    Each group is one combination of `group_cols` (default: dataset × true label).
    The cells in a group are pooled and their log-probs aggregated into a single
    score vector per class; the group's prediction is argmax of that vector.
    Accuracy / recall@k / macro_f1 are then computed over groups, not cells —
    matching the "group-level assignment" evaluation in the fingerprinting paper.

    aggregation:
      - "mean_log_prob" (default): mean of log-probs per class across cells.
        Equivalent to product of probabilities under conditional independence
        (joint log-likelihood / n) — the principled choice when each cell is an
        i.i.d. observation of the same condition.
      - "mean_prob": mean of probabilities (softmax-then-average), then log.
        Less sensitive to confidently-wrong cells than mean_log_prob.
      - "majority": each cell votes its top-1; group prediction is the mode.
        Loses calibration info but matches a literal voting baseline.
    """
    if any(c not in df.columns for c in group_cols):
        return pd.DataFrame()  # nothing to do without dataset/label info

    logit_cols = _logit_cols(df)
    col_idx = {c: i for i, c in enumerate(logit_cols)}
    has_direction = "direction" in df.columns
    label_pos = list(group_cols).index("label") if "label" in group_cols else None
    if label_pos is None:
        raise ValueError("score_group_level requires 'label' to be in group_cols")
    n_classes = len(logit_cols)

    rows = []
    for (method, split), gdf in df.groupby(["method", "split"], sort=False):
        if splits is not None and split not in splits:
            continue

        agg_vecs: list[np.ndarray] = []
        true_labels: list[str] = []
        for gkey, sub in gdf.groupby(list(group_cols), sort=False, observed=True):
            mat = sub[logit_cols].to_numpy(dtype=np.float32, copy=False)
            if aggregation == "mean_log_prob":
                vec = mat.mean(axis=0)
            elif aggregation == "mean_prob":
                probs = np.exp(mat)
                vec = np.log(np.clip(probs.mean(axis=0), 1e-30, None))
            elif aggregation == "majority":
                top1 = mat.argmax(axis=1)
                vec = np.bincount(top1, minlength=n_classes).astype(np.float32)
            else:
                raise ValueError(f"unknown aggregation: {aggregation!r}")
            agg_vecs.append(vec)
            true_labels.append(str(gkey[label_pos] if isinstance(gkey, tuple) else gkey))

        if not agg_vecs:
            continue
        agg_mat = np.stack(agg_vecs)
        y_true = np.array(true_labels)
        true_idx = np.array([col_idx.get(t, -1) for t in y_true])
        valid = true_idx >= 0
        if not valid.any():
            continue
        agg_mat = agg_mat[valid]
        y_true = y_true[valid]
        true_idx = true_idx[valid]

        order = (-agg_mat).argsort(axis=1)
        top1 = np.array([logit_cols[i] for i in order[:, 0]])
        out: dict[str, float | str | int | None] = {
            "method": method,
            "split": split,
            "n_groups": len(y_true),
            "accuracy": float((top1 == y_true).mean()),
            "macro_f1": _macro_f1_from_idx(true_idx, order[:, 0], len(logit_cols)),
        }
        for k in ks:
            hits = (order[:, :k] == true_idx[:, None]).any(axis=1)
            out[f"recall@{k}"] = float(hits.mean())
        if has_direction:
            out["direction"] = gdf["direction"].iloc[0]
        rows.append(out)
    return pd.DataFrame(rows)


def score_per_dataset_class(
    df: pd.DataFrame,
    dataset_col: str = "dataset",
    exclude_val_splits: bool = True,
    drop_zero_support: bool = True,
) -> pd.DataFrame:
    """Per (method, dataset, class) precision/recall/F1/support on non-val cells.

    OOD held-out datasets become just another dataset on the x-axis — id_test
    and ood_test cells share the same dataset dimension via obs[dataset_col].
    """
    resolved = dataset_col if dataset_col in df.columns else f"{OBS_PREFIX}{dataset_col}"
    if resolved not in df.columns:
        raise KeyError(
            f"dataset column {dataset_col!r} not found (also tried {resolved!r}); "
            f"need --splits-dir so obs columns are attached"
        )

    if exclude_val_splits:
        keep = ~df["split"].astype(str).map(lambda s: bool(VAL_SPLIT_RE.search(s)))
        df = df[keep]

    logit_cols = _logit_cols(df)
    col_idx = {c: i for i, c in enumerate(logit_cols)}
    n_classes = len(logit_cols)
    rows = []
    for (method, dataset), group in df.groupby(["method", resolved], sort=False):
        y_true = group["label"].astype(str).to_numpy()
        logits = group[logit_cols].to_numpy(dtype=np.float32, copy=False)
        pred_idx = logits.argmax(axis=1)
        true_idx = np.fromiter((col_idx.get(c, -1) for c in y_true),
                               count=len(y_true), dtype=np.int64)
        valid = true_idx >= 0
        precision, recall, f1, support = _perclass_prf_from_idx(
            true_idx[valid], pred_idx[valid], n_classes,
        )
        for ci, cls in enumerate(logit_cols):
            ss = int(support[ci])
            if drop_zero_support and ss == 0:
                continue
            rows.append({
                "method": method,
                "dataset": str(dataset),
                "class": cls,
                "precision": float(precision[ci]),
                "recall": float(recall[ci]),
                "f1": float(f1[ci]),
                "support": ss,
            })
    return pd.DataFrame(rows)


def training_stats(
    splits_dir: Path,
    dataset_size: str = "small",
    fold: str | None = None,
    dataset_col: str = "dataset",
    label_col: str = "perturbation",
) -> pd.DataFrame:
    """Per-dataset n_cells and n_perturbations in the training set.

    If `fold` is None, uses the first fold_N dir found under
    <splits_dir>/<dataset_size>/ (training partitions are near-identical across
    folds for this purpose).
    """
    import anndata as ad

    size_dir = Path(splits_dir).resolve() / dataset_size
    if fold is None:
        folds = sorted(p.name for p in size_dir.iterdir() if FOLD_RE.match(p.name))
        if not folds:
            raise FileNotFoundError(f"no fold_N dirs under {size_dir}")
        fold = folds[0]
    fold_dir = size_dir / fold
    meta = json.loads((size_dir / "metadata.json").read_text())
    source_h5ad = Path(meta["source_path"])
    adata = ad.read_h5ad(source_h5ad, backed="r")
    indices = _load_indices(fold_dir / "id_train.txt.gz")
    obs = adata.obs.iloc[indices]
    return (
        obs.groupby(dataset_col, observed=True)
        .agg(n_cells=(label_col, "size"), n_perturbations=(label_col, "nunique"))
        .reset_index()
        .rename(columns={dataset_col: "dataset"})
    )


def training_class_support(
    splits_dir: Path,
    dataset_size: str = "small",
    fold: str | None = None,
    label_col: str = "perturbation",
) -> pd.Series:
    """Per-class training cell count from <splits_dir>/<size>/<fold>/id_train.txt.gz.

    If `fold` is None, picks the first available fold (training partitions are
    near-identical across folds).
    """
    import anndata as ad

    size_dir = Path(splits_dir).resolve() / dataset_size
    if fold is None:
        folds = sorted(p.name for p in size_dir.iterdir() if FOLD_RE.match(p.name))
        if not folds:
            raise FileNotFoundError(f"no fold_N dirs under {size_dir}")
        fold = folds[0]
    fold_dir = size_dir / fold
    meta = json.loads((size_dir / "metadata.json").read_text())
    source_h5ad = Path(meta["source_path"])
    adata = ad.read_h5ad(source_h5ad, backed="r")
    indices = _load_indices(fold_dir / "id_train.txt.gz")
    obs = adata.obs.iloc[indices]
    return obs[label_col].astype(str).value_counts().rename("train_support")


def training_class_stats(
    splits_dir: Path,
    dataset_size: str = "small",
    fold: str | None = None,
    label_col: str = "perturbation",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """Per-class training-set summary: cell count, dataset coverage,
    mean cells per dataset, and Shannon-entropy of the dataset distribution.

    Columns:
      n_train_cells       — total cells of this class across training datasets
      n_train_datasets    — number of distinct datasets the class appears in
      cells_per_dataset   — n_train_cells / n_train_datasets (mean depth)
      dataset_entropy     — bits — uniform spread = log2(n_train_datasets), single
                            dataset = 0; high entropy = balanced source coverage.

    All fields are computed against the same id_train indices as
    `training_class_support` so they line up row-by-row.
    """
    import anndata as ad

    size_dir = Path(splits_dir).resolve() / dataset_size
    if fold is None:
        folds = sorted(p.name for p in size_dir.iterdir() if FOLD_RE.match(p.name))
        if not folds:
            raise FileNotFoundError(f"no fold_N dirs under {size_dir}")
        fold = folds[0]
    fold_dir = size_dir / fold
    meta = json.loads((size_dir / "metadata.json").read_text())
    source_h5ad = Path(meta["source_path"])
    adata = ad.read_h5ad(source_h5ad, backed="r")
    indices = _load_indices(fold_dir / "id_train.txt.gz")
    obs = adata.obs.iloc[indices]

    labels = obs[label_col].astype(str).to_numpy()
    datasets = obs[dataset_col].astype(str).to_numpy()
    pairs = pd.DataFrame({"label": labels, "dataset": datasets})
    counts = (pairs.groupby(["label", "dataset"], observed=True)
                   .size()
                   .rename("n").reset_index())

    def _entropy_per_label(grp: pd.DataFrame) -> float:
        n = grp["n"].to_numpy(dtype=np.float64)
        if n.sum() <= 0:
            return 0.0
        p = n / n.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    per_class = (counts.groupby("label", observed=True)
                       .apply(lambda g: pd.Series({
                           "n_train_cells":    int(g["n"].sum()),
                           "n_train_datasets": int(g["dataset"].nunique()),
                           "dataset_entropy":  _entropy_per_label(g),
                       }))
                       .reset_index())
    per_class["cells_per_dataset"] = (
        per_class["n_train_cells"] / per_class["n_train_datasets"].clip(lower=1)
    )
    return per_class.rename(columns={"label": "class"})


def plot_perdataset_f1_boxen(
    perdataset: pd.DataFrame,
    out_path: Path,
    palette: dict[str, tuple] | None = None,
    train_stats: pd.DataFrame | None = None,
) -> None:
    """Boxen of per-class F1, x=dataset (id + ood on same axis), hue=method.

    If `train_stats` is given, adds two bar panels on top showing per-dataset
    training cell + perturbation counts on a shared x-axis.
    """
    perdataset, palette = _disp_plot(perdataset, palette)   # raw keys -> display names (plots only)
    import matplotlib.pyplot as plt
    import seaborn as sns

    datasets = sorted(perdataset["dataset"].unique())
    methods = (list(palette.keys()) if palette
               else sorted(perdataset["method"].unique()))
    width = max(8, 1.1 * len(datasets))

    if train_stats is not None:
        ts = train_stats.set_index("dataset").reindex(datasets).reset_index()
        fig, axes = plt.subplots(
            3, 1, figsize=(width, 9),
            gridspec_kw={"height_ratios": [1, 1, 4]}, sharex=True,
        )
        ax_cells, ax_perts, ax = axes
        ax_cells.bar(datasets, ts["n_cells"], color="slategray")
        ax_cells.set_ylabel("# cells\n(train)")
        ax_cells.grid(axis="y", alpha=0.3)
        ax_perts.bar(datasets, ts["n_perturbations"], color="slategray")
        ax_perts.set_ylabel("# perts\n(train)")
        ax_perts.grid(axis="y", alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(width, 5.5))

    sns.boxenplot(
        data=perdataset, x="dataset", y="f1", hue="method",
        order=datasets, hue_order=methods, palette=palette, ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 distribution by dataset (id + ood) and method")
    _method_ticks(ax)
    fig.tight_layout()
    if palette:
        _consolidated_legend(fig, palette)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_support_vs_performance(
    perclass: pd.DataFrame,
    out_path: Path,
    palette: dict[str, tuple] | None = None,
    metric: str = "f1",
    train_support: pd.Series | None = None,
) -> None:
    """Scatter of per-class metric vs *training* support, hue=method, col=split.

    OOD-test classes that aren't in training are dropped (no training support).
    Val-like splits are excluded. X is log-scaled.
    """
    perclass, palette = _disp_plot(perclass, palette)   # display names + PLOT_HIDE
    import matplotlib.pyplot as plt
    import seaborn as sns

    mask = ~perclass["split"].astype(str).map(lambda s: bool(VAL_SPLIT_RE.search(s)))
    df = perclass[mask].copy()
    if df.empty:
        return

    if train_support is None:
        x_col = "support"
        x_label = "test cells per class"
    else:
        df["train_support"] = df["class"].astype(str).map(train_support).astype(float)
        before = len(df)
        df = df[df["train_support"].notna() & (df["train_support"] > 0)]
        dropped = before - len(df)
        if dropped:
            log.info(
                "support_vs_f1: dropped %d perclass rows whose class is absent from training",
                dropped,
            )
        x_col = "train_support"
        x_label = "training cells per class"

    col_order = list(df["split"].drop_duplicates())
    hue_order = (list(palette.keys()) if palette
                 else sorted(df["method"].unique()))
    g = sns.relplot(
        data=df, x=x_col, y=metric, hue="method", col="split",
        col_order=col_order, hue_order=hue_order,
        kind="scatter", palette=palette, alpha=0.6,
        height=4, aspect=1.2,
    )
    for ax in g.axes.flat:
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
    g.figure.suptitle(f"Per-class {metric} vs {x_label}", y=1.03)
    if palette:
        _consolidated_legend(g, palette)
    g.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(g.figure)


def plot_class_training_diagnostics(
    perclass: pd.DataFrame,
    train_class_stats: pd.DataFrame,
    out_path: Path,
    palette: dict[str, tuple] | None = None,
    metric: str = "f1",
) -> None:
    """For each per-(method, class, split) row in `perclass`, scatter the metric
    against several class-level training-set features:

      1. # training cells per class                (log x)
      2. # training datasets per class             (linear x; integer counts)
      3. mean cells per dataset                    (log x)
      4. Shannon-entropy of training dataset distribution (linear x)

    Each panel is faceted by split, hued by method, and titled with the
    per-method Spearman ρ so you can see whether the model's success on a
    class tracks training abundance vs source diversity. Val-like splits are
    excluded (same logic as plot_support_vs_performance).
    """
    perclass, palette = _disp_plot(perclass, palette)   # display names + PLOT_HIDE
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    if perclass.empty or train_class_stats.empty:
        return
    if metric not in perclass.columns:
        return

    keep = ~perclass["split"].astype(str).map(lambda s: bool(VAL_SPLIT_RE.search(s)))
    df = perclass[keep].copy()
    if df.empty:
        return
    df = df.merge(
        train_class_stats, on="class", how="inner", validate="many_to_one",
    )
    if df.empty:
        return

    feature_specs = [
        ("n_train_cells",     "log",    "# training cells per class"),
        ("n_train_datasets",  "linear", "# training datasets per class"),
        ("cells_per_dataset", "log",    "mean cells / training dataset"),
        ("dataset_entropy",   "linear", "training dataset entropy (bits)"),
    ]
    feature_specs = [(c, s, lbl) for c, s, lbl in feature_specs if c in df.columns]
    if not feature_specs:
        return

    col_order = list(df["split"].drop_duplicates())
    hue_order = (list(palette.keys()) if palette
                 else sorted(df["method"].unique()))

    n_rows = len(feature_specs)
    n_cols = len(col_order)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        # +3in on the right for each row's rho table; without it tight_layout shrank
        # the panels to fit the table and their x labels ran into each other
        figsize=(max(6.0, 3.2 * n_cols) + 3.0, max(3.0, 3.0 * n_rows)),
        sharey="row", squeeze=False,
    )

    for i, (feat, scale, x_label) in enumerate(feature_specs):
        row_rho: dict[str, dict[str, float]] = {}      # method -> split -> rho
        for j, split in enumerate(col_order):
            ax = axes[i, j]
            sub = df[df["split"] == split]
            if sub.empty:
                ax.set_visible(False)
                continue
            sns.scatterplot(
                data=sub, x=feat, y=metric, hue="method", ax=ax,
                hue_order=hue_order, palette=palette,
                alpha=0.5, s=14, edgecolor="none", legend=False,
            )
            if scale == "log":
                # Some features (cells, cells_per_dataset) may include 0 from
                # rare edge cases; clip x to avoid log warnings.
                ax.set_xscale("symlog", linthresh=1)
                # counts are >= 0; symlog's default margin reached into negative
                # decades and drew "-10^1 -10^0" tick labels that ran together
                ax.set_xlim(left=0)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)

            # Per-method Spearman rho, collected for the row's side table.
            for m_name in hue_order:
                sm = sub[sub["method"] == m_name]
                if len(sm) >= 3 and sm[feat].nunique() >= 2 and sm[metric].nunique() >= 2:
                    rho, _ = spearmanr(sm[feat].to_numpy(), sm[metric].to_numpy())
                    if np.isfinite(rho):
                        row_rho.setdefault(m_name, {})[split] = float(rho)

            if i == 0:
                ax.set_title(split)
            else:
                ax.set_title("")
            ax.set_ylabel(_metric_label(metric) if j == 0 else "")
            # Every row plots a different feature — give each its own x-label,
            # not just the bottom row.
            ax.set_xlabel(x_label)

        # One rho table per row, to the RIGHT of the row: printed inside each panel
        # it was 14 lines covering half the scatter and the tick labels.
        if row_rho:
            w = max(len(str(m)) for m in row_rho)
            head = " " * w + "".join(f"{str(sp)[:8]:>9}" for sp in col_order)
            lines = [head] + [
                f"{str(m):<{w}}" + "".join(
                    f"{row_rho[m][sp]:>+9.2f}" if sp in row_rho[m] else f"{'':>9}"
                    for sp in col_order)
                for m in hue_order if m in row_rho]
            axes[i, -1].text(
                1.04, 1.0, "Spearman rho\n" + "\n".join(lines),
                transform=axes[i, -1].transAxes, ha="left", va="top",
                fontsize=6.5, family="monospace", clip_on=False,
            )

    fig.suptitle(
        f"Per-class {metric} vs training-set features (Spearman ρ per method)",
        y=1.01,
    )
    fig.tight_layout()
    if palette:
        _consolidated_legend(fig, palette)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def apply_filter(df: pd.DataFrame, col: str, min_val: float) -> pd.DataFrame:
    """Drop rows where df[col] < min_val. Accepts either 'efficiency' or 'obs_efficiency'."""
    resolved = col if col in df.columns else f"{OBS_PREFIX}{col}"
    if resolved not in df.columns:
        raise KeyError(
            f"filter column {col!r} not found (also tried {resolved!r}); "
            f"available obs cols: {[c for c in df.columns if c.startswith(OBS_PREFIX)]}"
        )
    vals = pd.to_numeric(df[resolved], errors="coerce")
    kept = vals.notna() & (vals >= min_val)
    out = df[kept].reset_index(drop=True)
    log.info(
        "filter %s >= %g: kept %d / %d rows (%.1f%%)",
        resolved, min_val, len(out), len(df), 100 * len(out) / max(len(df), 1),
    )
    # per (method, split) retention
    summary = (
        df.assign(_kept=kept.to_numpy())
        .groupby(["method", "split"])
        .agg(n_before=("_kept", "size"), n_after=("_kept", "sum"))
        .reset_index()
    )
    log.info("filter retention by (method, split):\n%s", summary.to_string(index=False))
    return out


def _recall_prefixes(mode: str) -> tuple[str, str]:
    """(split-level prefix, efficiency-weighted prefix) for the chosen recall flavour."""
    return (("balanced_recall@", "balanced_weighted_recall@") if mode == "balanced"
            else ("recall@", "weighted_recall@"))


def _auroc_ylim(values) -> tuple[float, float]:
    """AUROC axis: fixed top at 1, bottom at 0.4 unless a point sits lower. A fixed
    range is the point of the metric -- chance is 0.5 at any class count, so
    small/ctx7/zs figures are directly comparable, which recall@k plots are not."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    lo = min(0.4, float(np.floor(v.min() * 20) / 20)) if len(v) else 0.4
    return (lo, 1.0)


def _rank_ylim(values, chance: float) -> tuple[float, float]:
    """Median-rank axis: from 0 to just above whichever is larger, the worst
    method or chance ((C+1)/2) -- some baselines rank the truth WORSE than chance."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return (0.0, max(chance, float(v.max()) if len(v) else chance) * 1.08)


def _metric_label(m: str, wrap: bool = False) -> str:
    """Axis label for a metric column. The raw balanced_* / weighted_* names are long
    enough that three of them stacked as row labels collide, so spell the qualifier out
    in parentheses instead: balanced_weighted_recall@5 -> "recall@5 (balanced, weighted)".
    """
    s = str(m)
    quals = []
    if s.startswith("balanced_"):
        s, _ = s[len("balanced_"):], quals.append("balanced")
    if s.startswith("weighted_"):
        s, _ = s[len("weighted_"):], quals.append("weighted")
    s = {"macro_f1": "macro-F1", "macro_f1_no_ds_bias": "macro-F1 (no ds bias)",
         "accuracy": "accuracy", "mean_auroc": "AUROC", "mrr": "MRR",
         "median_rank": "median rank", "median_rank_scaled": "median rank (scaled)"}.get(s, s)
    if not quals:
        return s
    # wrap=True for stacked-row y labels: one row is ~2.6in, and
    # "recall@5 (balanced, weighted)" set rotated is longer than that
    return f"{s}\n({', '.join(quals)})" if wrap else f"{s} ({', '.join(quals)})"


def _rank_methods(metrics_df: pd.DataFrame, recall_prefix: str = "recall@") -> list[str]:
    """Rank methods best→worst by mean rank across available metrics.

    Higher-is-better: accuracy, macro_f1, knn_acc, recall@*, balanced_recall@*.
    Per-method values are averaged across splits, ranked within each column,
    then the per-column ranks are averaged. Tie-break: alphabetical.
    """
    if metrics_df.empty or "method" not in metrics_df.columns:
        return []
    # Rank on ONE recall flavour, not both: including both would double-weight recall
    # relative to accuracy/macro_f1 in the mean-rank.
    higher_better = [c for c in metrics_df.columns
                     if c in ("accuracy", "macro_f1", "knn_acc")
                     or c.startswith(recall_prefix)]
    rank_total: pd.Series | None = None
    n_used = 0
    for col in higher_better:
        avg = metrics_df.groupby("method")[col].mean()
        if avg.dropna().empty:
            continue
        ranks = avg.rank(ascending=True)  # higher value → larger rank
        rank_total = ranks if rank_total is None else rank_total.add(ranks, fill_value=0)
        n_used += 1
    if rank_total is None or n_used == 0:
        return sorted(metrics_df["method"].astype(str).unique())
    rank_total = rank_total.sort_index()  # alphabetical tie-break
    return list(rank_total.sort_values(ascending=False, kind="stable").index)


# Display names for PLOTS ONLY — the CSVs keep the raw method keys, because the
# analysis suite (workflow/scripts/analysis/load.py) filters them by raw name.
# Mirrors analysis/common.py DISPLAY so both sets of figures read the same.
PLOT_DISPLAY = {
    "fingerprint_perdataset": "fingerprint (ensemble)",
    "fingerprint_merged": "fingerprint (merged)",
    "baseline_prior": "baseline (prior)",
    "baseline_random": "baseline (random)",
    "baseline_neg_target": "baseline (neg target)",
    # the plotted centroid baseline is the target-masked one (see PLOT_HIDE)
    "baseline_centroid_notarget": "baseline (centroid)",
    "baseline_centroid": "baseline (centroid, incl. target)",
    "xgboost": "XGBoost",
    "xgboost_capped": "XGBoost",
    "ExPert": "ExPert",
    "linreg": "linreg (SGD)",
}

# Scored and kept in every TABLE, but left out of every figure. baseline_centroid is
# replaced in the figures by baseline_centroid_notarget (labelled "baseline
# (centroid)"): the unmasked centroid reads the perturbation's own knocked-down
# target gene, which alone is worth -33%/-37% balanced recall@1 on the 275 classes
# whose target is in the panel. Mirror of analysis/load.py PLOT_HIDE.
PLOT_HIDE = {"baseline_centroid"}


def _disp(name):
    return PLOT_DISPLAY.get(name, name)


def _disp_plot(df, palette=None, col: str = "method"):
    """Map raw method keys -> display names on a plot frame AND its palette keys
    (hue= reads the column, palette is keyed by the same values). Returns both."""
    if df is not None and hasattr(df, "columns") and col in df.columns:
        df = df[~df[col].astype(str).isin(PLOT_HIDE)]
        df = df.assign(**{col: df[col].map(_disp)})
    if isinstance(palette, dict):
        palette = {_disp(k): v for k, v in palette.items() if k not in PLOT_HIDE}
    return df, palette


def _method_palette(
    methods_best_to_worst,
    baselines: tuple[str, ...] | list[str] = (),
    sub_range: tuple[float, float] = (0.05, 0.75),
    grey_range: tuple[float, float] = (0.30, 0.75),
) -> dict[str, tuple]:
    """Performance-ordered palette.

    Non-baseline methods are coloured via viridis (best → lime-green, worst →
    dark purple). Baselines are coloured via greys (best → dark, worst → light)
    and grouped at the **end** of the returned dict's insertion order so the
    legend/plot lines them up together after the real methods.

    `methods_best_to_worst` is the rank from `_rank_methods()`; baselines are
    pulled out and re-ordered to follow the same rank within their own group.
    """
    import matplotlib as mpl

    baseline_set = {str(b) for b in baselines}
    ordered = list(dict.fromkeys(str(m) for m in methods_best_to_worst))
    non_b = [m for m in ordered if m not in baseline_set]
    bsl = [m for m in ordered if m in baseline_set]

    palette: dict[str, tuple] = {}

    if non_b:
        n = len(non_b)
        lo, hi = sub_range
        positions = [hi] if n == 1 else np.linspace(hi, lo, n)
        for m, p in zip(non_b, positions):
            palette[m] = mpl.cm.viridis(p)

    if bsl:
        n = len(bsl)
        lo, hi = grey_range
        positions = [(lo + hi) / 2] if n == 1 else np.linspace(hi, lo, n)
        for m, p in zip(bsl, positions):
            palette[m] = mpl.cm.Greys(p)

    return palette


def _consolidated_legend(
    fig_or_grid, palette: dict[str, tuple], max_rows: int = 3, show: bool = True,
) -> None:
    """Strip per-axes / per-FacetGrid legends and place a single one below the
    figure, wrapping into at most `max_rows` rows. Entries follow the `palette`
    dict's insertion order (so baselines stay together after the main methods).

    `show=False` only strips: used where the x axis already names every method, so a
    colour legend would duplicate the tick labels.

    Layout is redone AFTER stripping: seaborn attaches a 14-entry legend to every
    axes, and a tight_layout() run while those were attached reserved their space,
    which is what squashed efficiency_sweep's panels into slivers. The legend is then
    anchored just below the lowest drawn element (tick labels, x label) instead of at
    a fixed figure fraction, which put it on top of rotated method names in short
    figures and left a gap in tall ones.
    """
    import math
    import warnings
    import matplotlib.patches as mpatches

    fig = getattr(fig_or_grid, "figure", fig_or_grid)
    grid_leg = getattr(fig_or_grid, "_legend", None)
    if grid_leg is not None:
        try:
            grid_leg.remove()
        except (AttributeError, ValueError):
            pass
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except Exception:
            pass
    if not palette or not show:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(r) for ax in fig.axes if ax.get_visible()]
    bottom_px = min(b.y0 for b in boxes if b is not None) if boxes else 0.0
    gap_px = 10 * fig.dpi / 72
    handles = [mpatches.Patch(color=color, label=label)
               for label, color in palette.items()]
    labels = list(palette.keys())
    ncol = max(1, math.ceil(len(labels) / max_rows))
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, (bottom_px - gap_px) / fig.bbox.height),
        bbox_transform=fig.transFigure,
        ncol=ncol, frameon=False, handlelength=1.2, columnspacing=1.4,
    )


def _method_ticks(ax, fontsize: float | None = None) -> None:
    """Method (or dataset) names on a categorical x axis: 45 deg, anchored at the tick.
    At 30 deg with ha='right' adjacent long names ran into each other."""
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
        if fontsize:
            lab.set_fontsize(fontsize)


def class_counts(df: pd.DataFrame, splits: tuple[str, ...] | None = None) -> dict[str, tuple[int, int]]:
    """Per-split (n_queried, n_total): queried = distinct true labels present; total = label space size.

    Expects the stacked collect() output (with logit columns and a `label` column).
    """
    logit_cols = _logit_cols(df)
    n_total = len(logit_cols)
    out: dict[str, tuple[int, int]] = {}
    for split, group in df.groupby("split", sort=False):
        if splits is not None and split not in splits:
            continue
        out[str(split)] = (int(group["label"].nunique()), n_total)
    return out


def _fmt_split(split: str, counts: dict[str, tuple[int, int]] | None) -> str:
    if counts is None or split not in counts:
        return split
    q, t = counts[split]
    return f"{split}\n(classes: {q}/{t})"


def _draw_median_ticks(ax, df, x_col, y_col, hue_col, order, hue_order,
                        bar_total_width: float = 0.8) -> None:
    """Overlay a thick black horizontal tick at the median of each bar.

    Matches seaborn barplot's default dodge geometry (bar group spans
    `bar_total_width` centered on the integer x position).
    """
    medians = df.groupby([x_col, hue_col], observed=True)[y_col].median().to_dict()
    n_hue = len(hue_order)
    bar_width = bar_total_width / n_hue
    half = bar_width * 0.42
    for i, xv in enumerate(order):
        for j, h in enumerate(hue_order):
            m = medians.get((xv, h))
            if m is None or (isinstance(m, float) and np.isnan(m)):
                continue
            xpos = i + (j - (n_hue - 1) / 2) * bar_width
            ax.hlines(m, xpos - half, xpos + half,
                      colors="black", linewidth=2, zorder=10)


def _max_points_per_group(df, group_cols) -> int:
    """Largest number of rows in any (hue × x) cell — drives bar-vs-boxen choice."""
    cols = [c for c in group_cols if c in df.columns]
    if not cols or df.empty:
        return 0
    return int(df.groupby(cols, dropna=False, observed=True).size().max())


def _bar_strip_median(
    ax, df, x_col, y_col, hue_col, order, hue_order, palette,
    show_strip: bool, add_median: bool = False,
) -> None:
    """Bar (mean) + optional strip (per-fold dots) + median tick mark.

    Auto-upgrades to boxen + strip overlay when there are enough points per
    (hue, x) group (≥4) so the distribution shows in a single visual instead
    of squashing everything into a mean bar with dots beside it.
    """
    import seaborn as sns

    n_max = _max_points_per_group(df, [x_col, hue_col])
    # Boxen + strip whenever there's any distribution to show (≥2 points per
    # group). Bar+strip only kicks in when every (x, hue) cell has a single
    # value and the boxen would degenerate to a flat line.
    use_boxen = show_strip and n_max >= 2

    if use_boxen:
        sns.boxenplot(
            data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
            order=order, hue_order=hue_order, palette=palette,
            showfliers=False,
        )
    else:
        sns.barplot(
            data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
            order=order, hue_order=hue_order, palette=palette,
            estimator=np.mean, errorbar=None,
        )
    if show_strip:
        # When x == hue, stripplot would dodge dots away from the box centre
        # (the boxen has no second hue dim, so dodge mis-aligns by half a
        # group width). Drop dodge in that case so dots sit on the box.
        dodge_strip = x_col != hue_col
        sns.stripplot(
            data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
            order=order, hue_order=hue_order, palette=palette,
            dodge=dodge_strip, edgecolor="black", linewidth=0.4,
            jitter=0.18, alpha=0.7,
            size=3 if use_boxen else 4, legend=False,
        )
        # Boxen already draws its own median line — only add a manual tick on
        # top of the bar variant where the mean alone hides skew.
        if add_median and not use_boxen:
            _draw_median_ticks(ax, df, x_col, y_col, hue_col, order, hue_order)
        # dedupe legend (boxen/bar + stripplot can stack handles)
        handles, labels = ax.get_legend_handles_labels()
        seen: dict[str, object] = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), loc="best", frameon=False)


def plot_topk_recall(
    metrics: pd.DataFrame,
    out_path: Path,
    ks: tuple[int, ...] = (1, 5, 10, 100),
    palette: dict[str, tuple] | None = None,
    counts: dict[str, tuple[int, int]] | None = None,
    prefix: str = "recall@",
) -> None:
    """Bar plot of recall@k by method, faceted by split. For multi-fold runs the
    bar shows the mean across folds, with each fold drawn as a dot and the
    median marked with a horizontal tick.

    `prefix` selects the metric family: "recall@" (cell-weighted, the natural class
    distribution) or "balanced_recall@" (mean over classes). They differ a lot on
    ood_test -- see the comment in score_one."""
    metrics, palette = _disp_plot(metrics, palette)   # raw keys -> display names (plots only)
    import matplotlib.pyplot as plt

    cols = [f"{prefix}{k}" for k in ks if f"{prefix}{k}" in metrics.columns]
    if not cols:
        log.warning("[plot] no %s columns present — skipping %s", prefix, out_path.name)
        return
    id_vars = ["method", "split"] + (["fold"] if "fold" in metrics.columns else [])
    long = metrics[[*id_vars, *cols]].melt(
        id_vars=id_vars, value_vars=cols,
        var_name="k", value_name="recall",
    )
    long["k"] = long["k"].str.replace(prefix, "", regex=False).astype(int)
    col_order = list(metrics["split"].drop_duplicates())
    # palette dict order groups baselines after non-baselines — use it directly
    hue_order = (list(palette.keys()) if palette
                 else sorted(long["method"].unique()))
    k_order = sorted(long["k"].unique())
    # Show dots whenever there's > 1 row per (k, method, split) — covers
    # multi-fold AND the per-dataset split fan-out.
    n_max = _max_points_per_group(long, ["k", "method", "split"])
    show_strip = n_max > 1
    use_boxen = n_max >= 2

    n = len(col_order)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 4.4 * n), 4),
                             sharey=True, squeeze=False)
    for ax, split in zip(axes[0], col_order):
        sub = long[long["split"] == split]
        _bar_strip_median(
            ax, sub, x_col="k", y_col="recall", hue_col="method",
            order=k_order, hue_order=hue_order, palette=palette,
            show_strip=show_strip,
        )
        ax.set_ylim(0, 1)
        ax.set_title(_fmt_split(split, counts))
    if use_boxen:
        suffix = f"\n(boxen of {n_max} points per (k, method, split); dots = individual values)"
    elif show_strip:
        suffix = f"\n(mean across {n_max} points; dots = individual values)"
    else:
        suffix = ""
    fig.suptitle(f"Top-k recall by method{suffix}", y=1.03)
    fig.tight_layout()
    if palette:
        _consolidated_legend(fig, palette)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_macro_f1(
    metrics: pd.DataFrame,
    out_path: Path,
    palette: dict[str, tuple] | None = None,
    counts: dict[str, tuple[int, int]] | None = None,
    metric: str = "macro_f1",
    title_label: str | None = None,
    ylim: tuple[float, float] | None = None,
    ref: float | None = None,
    ref_label: str = "chance",
) -> None:
    """Macro F1 bar plot (also used for any single scalar metric via `metric`). For multi-fold runs the bar shows the mean across folds,
    each fold is drawn as a dot, and the median is marked with a horizontal tick.

    Pass `metric="macro_f1_no_ds_bias"` to plot the dataset-bias-corrected
    variant (mean of per-dataset present-only macro-F1s) on the same axes
    layout. `title_label` overrides the y-axis description in the title.
    """
    metrics, palette = _disp_plot(metrics, palette)   # raw keys -> display names (plots only)
    import matplotlib.pyplot as plt

    if metric not in metrics.columns:
        return
    title_label = title_label or metric
    n_max = _max_points_per_group(metrics, ["split", "method"])
    show_strip = n_max > 1
    use_boxen = n_max >= 2
    order = list(metrics["split"].drop_duplicates())
    hue_order = (list(palette.keys()) if palette
                 else sorted(metrics["method"].unique()))
    # width scales with methods x splits too: at 14 methods the old 6in squeezed each
    # box to a sliver, which matters now that AUROC/median rank are headline figures.
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(order), 0.45 * len(hue_order) * len(order)), 4.5))
    _bar_strip_median(
        ax, metrics, x_col="split", y_col=metric, hue_col="method",
        order=order, hue_order=hue_order, palette=palette,
        show_strip=show_strip,
    )
    title = f"{title_label} by method"
    if use_boxen:
        title += f"\n(boxen of {n_max} points per (split, method); dots = individual values)"
    elif show_strip:
        title += f"\n(mean across {n_max} points; dots = individual values)"
    ax.set_ylim(*(ylim or (0, max(0.05, metrics[metric].max() * 1.15))))
    if ref is not None:
        ax.axhline(ref, ls="--", lw=1, color="0.4", zorder=0)
        ax.annotate(f"{ref_label} {ref:g}", xy=(1.0, ref), xycoords=("axes fraction", "data"),
                    xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color="0.35")
    ax.set_ylabel(title_label)
    ax.set_title(title)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_fmt_split(s, counts) for s in order])
    if palette:
        _consolidated_legend(fig, palette)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_perclass_f1_boxen(
    perclass: pd.DataFrame,
    out_path: Path,
    palette: dict[str, tuple] | None = None,
    counts: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Boxen plot of per-class F1 across classes, x=method, faceted by split."""
    perclass, palette = _disp_plot(perclass, palette)   # raw keys -> display names (plots only)
    import matplotlib.pyplot as plt
    import seaborn as sns

    col_order = list(perclass["split"].drop_duplicates())
    order = (list(palette.keys()) if palette
             else sorted(perclass["method"].unique()))
    g = sns.catplot(
        data=perclass, x="method", y="f1", col="split",
        col_order=col_order, order=order, palette=palette,
        kind="boxen", height=4.5, aspect=max(1.1, 0.3 * max(1, len(order)) / 4.5), sharey=True,
    )
    g.set(ylim=(0, 1))
    for ax, split in zip(g.axes.flat, col_order):
        ax.set_title(_fmt_split(split, counts))
        _method_ticks(ax)
    g.figure.suptitle("Per-class F1 distribution by method", y=1.03)
    if palette:
        _consolidated_legend(g, palette, show=False)
    g.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(g.figure)


def plot_perclass_topk_boxen(
    perclass: pd.DataFrame,
    out_path: Path,
    ks: tuple[int, ...] = (1, 5, 10),
    palette: dict[str, tuple] | None = None,
    counts: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Boxen plot of per-class recall@k. rows=split, cols=k, x=method."""
    perclass, palette = _disp_plot(perclass, palette)   # raw keys -> display names (plots only)
    import matplotlib.pyplot as plt
    import seaborn as sns

    value_cols = [f"recall@{k}" for k in ks if f"recall@{k}" in perclass.columns]
    long = perclass[["method", "split", "class", *value_cols]].melt(
        id_vars=["method", "split", "class"],
        value_vars=value_cols,
        var_name="k", value_name="recall_at_k",
    ).dropna(subset=["recall_at_k"])
    long["k"] = long["k"].str.replace("recall@", "").astype(int)

    row_order = list(perclass["split"].drop_duplicates())
    order = (list(palette.keys()) if palette
             else sorted(perclass["method"].unique()))
    col_order = sorted(long["k"].unique())
    g = sns.catplot(
        data=long, x="method", y="recall_at_k",
        row="split", col="k",
        row_order=row_order, col_order=col_order, order=order, palette=palette,
        kind="boxen", height=4, aspect=max(1.1, 0.3 * max(1, len(order)) / 4), sharey=True, margin_titles=True,
    )
    g.set(ylim=(0, 1))
    for ax in g.axes.flat:
        _method_ticks(ax)
    # row titles (right margin) already render each split via margin_titles.
    # Inject the class counts into those margin labels by re-rendering them.
    if counts is not None:
        for i, split in enumerate(row_order):
            q, t = counts.get(split, (None, None))
            if q is None:
                continue
            # the last column in each row hosts the margin title text
            axes_row = g.axes[i]
            ax_last = axes_row[-1]
            if ax_last.texts:
                ax_last.texts[-1].set_text(f"{split}\nclasses: {q}/{t}")
    g.figure.suptitle("Per-class recall@k distribution by method", y=1.02)
    if palette:
        _consolidated_legend(g, palette, show=False)
    g.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(g.figure)


def _eff_subdir(thresh: float | None) -> str:
    return "none" if thresh is None else f"e:{thresh:g}"


def _eff_label(thresh) -> str:
    """Display label for the sweep x-axis. None / NaN render as '0' (no-filter baseline)."""
    if thresh is None or (isinstance(thresh, float) and np.isnan(thresh)):
        return "0"
    return f"{thresh:g}"


def _sweep_order(values) -> list[str]:
    return sorted({_eff_label(v) for v in values}, key=float)


def cells_per_perturbation(
    df: pd.DataFrame,
    splits: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Per (canonical_split, class) cell count, deduplicated across methods.

    Each method's logits refer to the same underlying cells in a given split
    (e.g. id_test == eval_iid), so we pick one method per canonical split.
    """
    df = df.copy()
    df["canonical_split"] = df["split"].map(_canonical_split)
    if splits is not None:
        wanted = {_canonical_split(s) for s in splits}
        df = df[df["canonical_split"].isin(wanted)]
    rows = []
    for csplit, g in df.groupby("canonical_split", sort=False):
        ref_method = g["method"].iloc[0]
        gm = g[g["method"] == ref_method]
        counts = gm.groupby("label", sort=False).size().reset_index(name="n_cells")
        counts["split"] = str(csplit)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["label", "n_cells", "split"]
    )


def plot_cells_per_perturbation_sweep(class_sweep: pd.DataFrame, out_path: Path) -> None:
    """Top: # perturbations with ≥1 cell; bottom: boxen of cells-per-perturbation.
    Both vs min-efficiency threshold, columns=canonical split. Companion to
    efficiency_sweep.png for checking whether perf gains track class drop-outs.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = class_sweep.copy()
    df["min_efficiency"] = df["min_efficiency"].apply(_eff_label)
    order = _sweep_order(class_sweep["min_efficiency"])

    # Per-fold and per-dataset rows multiply the same (split, label) — collapse
    # them by summing n_cells so each label has a single row per threshold.
    # Without this the top-panel count was n_unique_labels × n_folds × n_datasets,
    # and the bottom-panel boxen was a distribution over (fold × dataset × label)
    # instead of just labels.
    df = (df.groupby(["split", "label", "min_efficiency"], as_index=False)
            ["n_cells"].sum())

    summary = (
        df.groupby(["split", "min_efficiency"], as_index=False)
        .size()
        .rename(columns={"size": "n_perts"})
    )

    splits_u = sorted(df["split"].unique())
    ncols = max(len(splits_u), 1)
    fig, axes = plt.subplots(
        2, ncols, figsize=(max(6.0, 3.5 * ncols), 7.0),
        gridspec_kw={"height_ratios": [1, 2]}, sharex="col", sharey="row",
    )
    axes = np.atleast_2d(axes)
    if ncols == 1 and axes.shape != (2, 1):
        axes = axes.reshape(2, 1)

    for j, split in enumerate(splits_u):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]
        s_sum = summary[summary["split"] == split]
        sns.barplot(data=s_sum, x="min_efficiency", y="n_perts",
                    order=order, ax=ax_top, color="slategray")
        ax_top.set_title(split)
        ax_top.set_xlabel("")
        ax_top.set_ylabel("# perturbations\nwith ≥1 cell" if j == 0 else "")
        ax_top.grid(axis="y", alpha=0.3)

        s_df = df[df["split"] == split]
        sns.boxenplot(data=s_df, x="min_efficiency", y="n_cells",
                      order=order, ax=ax_bot, color="slategray")
        ax_bot.set_yscale("log")
        ax_bot.set_xlabel("min efficiency")
        ax_bot.set_ylabel("# cells / perturbation" if j == 0 else "")
        ax_bot.grid(axis="y", which="both", alpha=0.3)

    fig.suptitle("Cells per perturbation vs min-efficiency filter", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_method_legend(palette: dict[str, tuple], out_path: Path,
                       ncol: int = 5) -> None:
    """The shared method legend as its OWN figure.

    Method colours are identical in every figure, so repeating the legend inside each one
    costs a third of the canvas and says nothing new. Published once as legend.{svg,png};
    the compact per-figure panels then carry data only.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    items = [(mpatches.Patch(facecolor=c, edgecolor="none"), _disp(m))
             for m, c in palette.items() if m not in PLOT_HIDE]
    if not items:
        return
    rows = -(-len(items) // ncol)
    fig = plt.figure(figsize=(1.9 * ncol, 0.34 * rows + 0.2))
    fig.legend([h for h, _ in items], [l for _, l in items], loc="center",
               ncol=ncol, frameon=False, fontsize=10, handlelength=1.4,
               columnspacing=1.4, borderaxespad=0.0)
    fig.savefig(out_path, bbox_inches="tight", dpi=150, transparent=False)
    plt.close(fig)


def plot_ood_matched_sweep(
    sweep_per_fold: pd.DataFrame,
    out_path: Path,
    metrics: list[str],
    palette: dict[str, tuple] | None = None,
    baseline: str = "baseline_random",
) -> None:
    """OOD efficiency sweep as LINES, on a fold x dataset MATCHED cell set.

    Every point is a mean over (fold x held-out dataset x threshold) cells, and the cell
    set is the INTERSECTION across the plotted methods -- ExPert covers 11 of the 12
    small folds, so an unmatched mean would compare it on a different set of datasets
    than everyone else and the differences would partly be fold composition, not method.
    `baseline` is only used to define that matched set; its own line is drawn like any
    other method's. No legend: the method colours are shared across every figure and
    published once as legend.{svg,png}.
    """
    import matplotlib.pyplot as plt

    df = sweep_per_fold.copy()
    # per-dataset OOD rows. Folds that hold out a SINGLE dataset write a plain "ood_test"
    # row (nothing to split), and matching on "ood_test:" alone silently dropped them.
    sp = df["split"].astype(str)
    df = df[sp.str.startswith("ood_test:") | (sp == "ood_test")]
    if df.empty:
        log.warning("[plot] no per-dataset ood_test rows — skipping %s", out_path.name)
        return
    df["eff"] = pd.to_numeric(df["min_efficiency"], errors="coerce").fillna(0.0)
    metrics = [m for m in metrics if m in df.columns]
    if not metrics or baseline not in set(df["method"]):
        log.warning("[plot] missing metrics or %s — skipping %s", baseline, out_path.name)
        return

    key = ["fold", "split", "eff"]
    base = df[df["method"] == baseline].set_index(key)[metrics]
    others = df[df["method"] != baseline]
    # matched cells: present for the baseline AND for every plotted method
    cells = set(base.index)
    for m, g in others.groupby("method"):
        cells &= set(map(tuple, g[key].to_numpy()))
    if not cells:
        log.warning("[plot] no cells common to all methods — skipping %s", out_path.name)
        return
    n_before = others.set_index(key).index.nunique()
    log.info("[ood-delta] matched on %d of %d (fold x dataset x threshold) cells",
             len(cells), n_before)

    allm = df.set_index(key)
    keep = allm.loc[allm.index.isin(cells)]
    vals = keep[metrics].copy()
    vals["method"] = keep["method"].values
    delta = vals.reset_index()

    agg = delta.groupby(["method", "eff"])[metrics].agg(["mean", "sem"])
    present = set(delta["method"])
    order = [m for m in (palette or {}) if m in present] or sorted(present)

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(3.4, max(3.0, 2.4 * n)),
                             sharex=True, squeeze=False)
    xs = sorted(delta["eff"].unique())
    for i, metric in enumerate(metrics):
        ax = axes[i, 0]
        for meth in order:
            if meth not in agg.index.get_level_values(0):
                continue
            g = agg.loc[meth]
            mu = g[(metric, "mean")].reindex(xs)
            se = g[(metric, "sem")].reindex(xs)
            col = (palette or {}).get(meth, "0.4")
            lw = 2.6 if meth == "ExPert" else 1.5
            ax.plot(xs, mu.values, color=col, lw=lw, marker="o",
                    ms=7 if meth == "ExPert" else 5, label=_disp(meth), zorder=3)
            ax.fill_between(xs, (mu - se).values, (mu + se).values,
                            color=col, alpha=0.15, lw=0, zorder=2)
        ax.set_ylabel(_metric_label(metric), fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xticks(xs)
    axes[-1, 0].set_xlabel("min efficiency")
    axes[0, 0].set_title(f"OOD — fold x dataset matched\n({len(cells):,} cells; band = ±SEM)",
                         fontsize=9)
    axes[-1, 0].tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_efficiency_sweep(
    sweep_metrics: pd.DataFrame,
    out_path: Path,
    ks: tuple[int, ...] = (1, 5, 10),
    palette: dict[str, tuple] | None = None,
    metrics: list[str] | None = None,
) -> None:
    """Metric vs min-efficiency threshold. row=metric, col=split, x=threshold, hue=method.

    Auto-switches to boxen + scatter once there are ≥4 points per
    (threshold, method) cell (e.g. multi-fold × per-dataset rows).
    """
    sweep_metrics, palette = _disp_plot(sweep_metrics, palette)   # raw keys -> display names (plots only)
    import matplotlib.pyplot as plt

    df = sweep_metrics.copy()
    df["min_efficiency"] = df["min_efficiency"].apply(_eff_label)
    order = _sweep_order(sweep_metrics["min_efficiency"])

    recall_pick = f"recall@{ks[len(ks) // 2]}" if ks else None
    wanted = metrics or ("accuracy", "macro_f1", "macro_f1_no_ds_bias", recall_pick)
    metrics_to_plot = [m for m in wanted if m and m in df.columns]
    if not metrics_to_plot:
        return
    long = df.melt(
        id_vars=["method", "split", "min_efficiency"]
                + (["fold"] if "fold" in df.columns else []),
        value_vars=metrics_to_plot,
        var_name="metric", value_name="value",
    )
    col_order = list(df["split"].drop_duplicates())
    hue_order = (list(palette.keys()) if palette
                 else sorted(df["method"].unique()))
    metric_order = [m for m in metrics_to_plot if m in long["metric"].unique()]
    n_max = _max_points_per_group(long, ["min_efficiency", "method", "split", "metric"])
    show_strip = n_max > 1
    use_boxen = n_max >= 2

    n_rows = len(metric_order)
    n_cols = len(col_order)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6.0, 3.4 * n_cols), max(3.0, 2.8 * n_rows)),
        sharey="row", sharex=True, squeeze=False,
    )
    for i, metric in enumerate(metric_order):
        for j, split in enumerate(col_order):
            ax = axes[i, j]
            sub = long[(long["metric"] == metric) & (long["split"] == split)]
            _bar_strip_median(
                ax, sub,
                x_col="min_efficiency", y_col="value", hue_col="method",
                order=order, hue_order=hue_order, palette=palette,
                show_strip=show_strip,
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("min efficiency" if i == n_rows - 1 else "")
            ax.set_ylabel(_metric_label(metric) if j == 0 else "")
            ax.set_title(split if i == 0 else "")

    if use_boxen:
        suffix = f"\n(boxen of {n_max} points per (threshold, method); dots = individual values)"
    elif show_strip:
        suffix = f"\n(mean across {n_max} points; dots = individual values)"
    else:
        suffix = ""
    fig.suptitle(f"Metrics vs min-efficiency filter{suffix}", y=1.01)
    fig.tight_layout()
    if palette:
        _consolidated_legend(fig, palette)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


OOD_GROUP_ORDER = ("ood_cell_line", "ood_condition", "ood_donor")
ID_GROUP_ORDER = ("id_cell_line", "id_condition", "id_donor")


def _marson_parts(name):
    import re as _re
    m = _re.match(r"^(Marson\d*)_?(D\d+)_(.+)$", str(name))
    return (m.group(2), m.group(3)) if m else None


def _fold_holdouts(df, split_col="split"):
    """fold -> set of datasets that fold held out (read off its ood rows)."""
    held = {}
    ood = df[df[split_col].astype(str).str.startswith("ood")]
    for f, g in ood.groupby("fold"):
        held[f] = set(g[split_col].astype(str).str.split(":", n=1).str[-1])
    return held


def _fold_shift_type(held_set):
    """What KIND of holdout this fold is, from its held-out datasets.

    donor      : one Marson donor held out across all its conditions (folds 0-3)
    condition  : one Marson condition held out across all donors     (folds 9-11)
    cell_line  : a whole cell line / study held out                  (folds 4-8)
    """
    sib = [d for d in held_set if _marson_parts(d)]
    if sib:
        donors = {_marson_parts(d)[0] for d in sib}
        conds = {_marson_parts(d)[1] for d in sib}
        if len(donors) == 1 and len(conds) >= 2:
            return "donor"
        if len(conds) == 1 and len(donors) >= 2:
            return "condition"
    return "cell_line"


def canon_split_shift(df, split_col="split"):
    """Label every row by the DISTRIBUTION SHIFT its fold encodes, instead of
    collapsing everything into one id_test / ood_test pair.

      ood rows -> ood_{cell_line,condition,donor}, from the HELD-OUT dataset
                  itself (a fold can hold out a donor and a cell line at once,
                  so this must be per-dataset, not per-fold).
      id  rows -> id_{cell_line,condition,donor}, from the fold's holdout TYPE
                  (an id dataset is in training, so it has no shift of its own;
                  this asks whether in-distribution accuracy depends on what
                  was held out).

    Needs `fold`; without it splits collapse to plain id_test / ood_test.
    """
    if split_col not in df.columns or df.empty:
        return df
    out = df.copy()
    top = out[split_col].astype(str).str.split(":", n=1).str[0]
    ds = out[split_col].astype(str).str.split(":", n=1).str[-1]
    if "fold" not in out.columns:
        out[split_col] = top
        return out

    held = _fold_holdouts(out, split_col)
    ftype = {f: _fold_shift_type(h) for f, h in held.items()}

    def _one(t, f, d):
        t = str(t)
        if t.startswith("ood"):
            parts = _marson_parts(d)
            if parts is not None:
                donor, cond = parts
                sib = [x for x in held.get(f, ()) if _marson_parts(x)]
                same_donor = [x for x in sib if _marson_parts(x)[0] == donor]
                same_cond = [x for x in sib if _marson_parts(x)[1] == cond]
                if len(same_donor) >= 2 and len(same_cond) <= 1:
                    return "ood_donor"
                if len(same_cond) >= 2 and len(same_donor) <= 1:
                    return "ood_condition"
            return "ood_cell_line"
        return f"id_{ftype.get(f, 'cell_line')}"

    out[split_col] = [_one(t, f, d) for t, f, d in zip(top, out["fold"], ds)]
    return out


# kept for callers that only need the ood breakdown
def canon_split_ood(df, split_col="split"):
    return canon_split_shift(df, split_col)


def plot_weighted_metrics(
    weighted: pd.DataFrame,
    out_path: Path,
    ks: tuple[int, ...] = (1, 5, 10),
    palette: dict[str, tuple] | None = None,
    recall_prefix: str = "weighted_recall@",
    metrics: list[str] | None = None,
    refs: dict[str, float] | None = None,
    ylims: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
) -> None:
    """Cell-weighted metrics. row=metric, col=split, x=method.

    `refs` / `ylims` (keyed by metric) add a dashed chance line and a fixed y-range
    to that metric's row -- used by the rank-metric figures, where chance is not 0.

    Orthogonal to the efficiency_sweep view: instead of hard thresholding, each
    cell's contribution is scaled by its efficiency, so high-effect cells
    dominate. Per-fold × per-dataset rows make this a distribution per
    (method, split, metric), so we auto-switch to boxen + scatter once there
    are enough points to make the distribution informative.
    """
    weighted, palette = _disp_plot(weighted, palette)   # raw keys -> display names (plots only)
    import matplotlib.pyplot as plt

    recall_pick = f"{recall_prefix}{ks[len(ks) // 2]}" if ks else None
    wanted = metrics or ("weighted_accuracy", "weighted_macro_f1", recall_pick)
    metrics_to_plot = [m for m in wanted if m and m in weighted.columns]
    if not metrics_to_plot:
        return

    id_vars = ["method", "split"] + (["fold"] if "fold" in weighted.columns else [])
    long = weighted[[*id_vars, *metrics_to_plot]].melt(
        id_vars=id_vars, value_vars=metrics_to_plot,
        var_name="metric", value_name="value",
    )
    col_order = list(weighted["split"].drop_duplicates())
    hue_order = (list(palette.keys()) if palette
                 else sorted(weighted["method"].unique()))
    metric_order = [m for m in metrics_to_plot if m in long["metric"].unique()]

    n_max = _max_points_per_group(long, ["method", "split", "metric"])
    show_strip = n_max > 1
    use_boxen = n_max >= 2

    n_rows = len(metric_order)
    n_cols = len(col_order)
    # ~0.3in per method keeps 45-degree names apart (at 3in/panel, 14 names collided)
    panel_w = max(3.0, 0.3 * len(hue_order))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6.0, panel_w * n_cols), max(3.0, 2.6 * n_rows)),
        sharey="row", sharex=True, squeeze=False,
    )
    for i, metric in enumerate(metric_order):
        for j, split in enumerate(col_order):
            ax = axes[i, j]
            sub = long[(long["metric"] == metric) & (long["split"] == split)]
            _bar_strip_median(
                ax, sub, x_col="method", y_col="value", hue_col="method",
                order=hue_order, hue_order=hue_order, palette=palette,
                show_strip=show_strip,
            )
            ax.grid(True, alpha=0.3, axis="y")
            if ylims and metric in ylims:
                ax.set_ylim(*ylims[metric])
            if refs and metric in refs:
                ax.axhline(refs[metric], ls="--", lw=1, color="0.4", zorder=0)
            ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(_metric_label(metric, wrap=True))
            else:
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(split)
            else:
                ax.set_title("")
            _method_ticks(ax, fontsize=8)
            ax.tick_params(axis="y", labelsize=8)

    weight_col = (weighted["weight_col"].iloc[0]
                  if "weight_col" in weighted.columns else "weight")
    if use_boxen:
        suffix = f"\n(boxen of {n_max} points per (method, split); dots = individual values)"
    elif show_strip:
        suffix = f"\n(mean across {n_max} points; dots = individual values)"
    else:
        suffix = ""
    fig.suptitle(
        f"{title or 'Cell-weighted metrics'} (weights = {weight_col}){suffix}", y=1.01,
    )
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    # x already names every method, so a colour legend would only duplicate it
    _consolidated_legend(fig, palette, show=False)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


_NO_WRITE = False   # set by --replot: tables are inputs then, never rewritten


def _write(df: pd.DataFrame, path: Path) -> None:
    if _NO_WRITE:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix in (".csv", ".tsv"):
        df.to_csv(path, index=False, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        raise ValueError(f"unsupported output suffix: {path.suffix}")


def _derive_scaled_rank(df: pd.DataFrame | None, n_classes: int) -> pd.DataFrame | None:
    """Add median_rank_scaled / weighted_median_rank_scaled where only the raw median
    rank exists (tables written before the scaled form was scored). Exact: the scaled
    value is 1 - (median_rank - 1) / (C - 1), a function of the stored rank and C."""
    if df is None or n_classes < 2:
        return df
    for raw, scaled in (("median_rank", "median_rank_scaled"),
                        ("weighted_median_rank", "weighted_median_rank_scaled")):
        if raw in df.columns and scaled not in df.columns:
            df[scaled] = 1.0 - (df[raw] - 1.0) / (n_classes - 1)
    return df


def _headline_metrics(recall_prefix: str, n_classes: int, weighted: bool):
    """The three headline metrics and their chance levels: recall@10 (the shortlist a
    biologist would test), MRR (precision at the top, degrading gracefully) and the
    scaled median rank (where the true driver lands in the full ranking)."""
    w = "weighted_" if weighted else ""
    r10 = f"{recall_prefix}10"
    mets = [r10, f"{w}mrr", f"{w}median_rank_scaled"]
    refs = {r10: min(10 / n_classes, 1.0),
            f"{w}mrr": float(np.log(n_classes) + 0.5772156649) / n_classes,
            f"{w}median_rank_scaled": 0.5}
    return mets, refs


def _plot_rank_figures(df: pd.DataFrame, out_dir: Path, ext: str, palette, counts,
                       n_classes: int, recall_prefix: str, suffix: str = "") -> None:
    """auroc{suffix}.png (reported flavour), auroc_<other>{suffix}.png, and
    median_rank{suffix}.png -- one metric per figure, x=split, hue=method."""
    # recall_prefix is main()'s local, passed in -- it is not a module global.
    reported = ("balanced_mean_auroc" if recall_prefix.startswith("balanced")
                else "mean_auroc")
    other = "mean_auroc" if reported == "balanced_mean_auroc" else "balanced_mean_auroc"
    other_tag = "cells" if other == "mean_auroc" else "balanced"
    for key, name in ((reported, "auroc"), (other, f"auroc_{other_tag}")):
        if key in df.columns and df[key].notna().any():
            plot_macro_f1(df, out_dir / f"{name}{suffix}.{ext}", palette=palette,
                          counts=counts, metric=key, title_label=_metric_label(key),
                          ylim=_auroc_ylim(df[key]), ref=0.5)
    if "median_rank" in df.columns and df["median_rank"].notna().any():
        chance = (n_classes + 1) / 2
        plot_macro_f1(df, out_dir / f"median_rank{suffix}.{ext}", palette=palette,
                      counts=counts, metric="median_rank",
                      title_label=f"median rank (of {n_classes:,}; lower = better)",
                      ylim=_rank_ylim(df["median_rank"], chance), ref=chance)


def _load_plot_state(out: Path, thresholds) -> tuple:
    """Rebuild the scoring loop's frames from a finished run's tables (--replot).

    Exact for everything plotted, with one exception: runs written before
    sweep_group.csv existed kept only fold-MEANS of the group-level table, so their
    group figures redraw without per-fold points.
    """
    def eff(t):
        return np.nan if t is None else float(t)

    def per_threshold(name, reader):
        parts = []
        for t in thresholds:
            p = out / _eff_subdir(t) / name
            if p.exists():
                df = reader(p)
                df["min_efficiency"] = eff(t)
                parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else None

    def need(p: Path) -> Path:
        if not p.exists():
            raise SystemExit(f"--replot: {p} is missing -- was --output-dir a finished run "
                             "with the same --efficiency-sweep?")
        return p

    sweep_metrics = pd.read_csv(need(out / "sweep_metrics.csv"))
    pf = out / "sweep_metrics_per_fold.csv"
    sweep_per_fold = pd.read_csv(pf) if pf.exists() else None
    sweep_perclass = per_threshold("all_perclass_metrics.parquet", pd.read_parquet)
    if sweep_perclass is None:
        raise SystemExit("--replot: no all_perclass_metrics.parquet under any threshold dir")
    sweep_perdataset = per_threshold("all_perdataset_class_metrics.parquet", pd.read_parquet)
    sweep_cc = pd.read_parquet(need(out / "sweep_cells_per_perturbation.parquet"))
    sg = out / "sweep_group.csv"
    if sg.exists():
        sweep_group = pd.read_csv(sg)
    else:
        sweep_group = per_threshold("all_metrics_group.csv", pd.read_csv)
        if sweep_group is not None:
            log.warning("--replot: no sweep_group.csv (older run) -- group figures use "
                        "fold means, so they lose their per-fold points")
    wm = out / "weighted_metrics.csv"
    rows_weighted = [pd.read_csv(wm)] if wm.exists() else []
    ts = out / "train_stats.csv"
    train_stats_df = pd.read_csv(ts) if ts.exists() else None
    tcs = out / "train_class_support.csv"
    train_support = None
    if tcs.exists():
        d = pd.read_csv(tcs)
        train_support = d.set_index(d.columns[0])[d.columns[1]]
    tcst = out / "train_class_stats.csv"
    train_class_stats_df = pd.read_csv(tcst) if tcst.exists() else None
    log.info("--replot: rebuilt plotting frames from %s (%d metric rows, %d per-class rows)",
             out, len(sweep_metrics), len(sweep_perclass))
    return (sweep_metrics, sweep_per_fold, sweep_perclass, sweep_perdataset, None,
            sweep_cc, sweep_group, [], rows_weighted,
            train_stats_df, train_support, train_class_stats_df)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", type=Path, required=True, help="root dir to glob 'logits.csv' under")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="root output dir. Writes all_logits.parquet at the top and "
                         "<eff_subdir>/{all_metrics.csv,all_perclass_metrics.parquet,plots/}"
                         " per threshold. eff_subdir is 'none' or 'e:<val>'.")
    ap.add_argument(
        "--method-depth", type=int, default=1,
        help="number of path segments under --source to capture. First becomes 'direction' "
             "when >=2; remaining segments are joined as 'method' (default 1)",
    )
    ap.add_argument(
        "--splits-dir", type=Path, default=None,
        help="root of the splits tree, e.g. data/splits. When set, per-cell labels "
             "and metadata are pulled from the source adata referenced by "
             "<splits-dir>/<dataset-size>/metadata.json (backed mode) using "
             "<splits-dir>/<dataset-size>/<fold>/<split>.txt.gz indices. obs cols "
             "are attached as obs_<col>.",
    )
    ap.add_argument(
        # No `choices`: sizes are config (config.yaml `sizes` + the padding size), not a
        # fixed set, and the value is only used as a path segment filter and to pick the
        # splits subtree -- any size works. The old ("small","medium","full") list would
        # have failed `collect` for ctx7 and zs_f4 at the very END of a multi-day run,
        # since collect is the last job in the DAG.
        "--dataset-size", default="small",
        help="dataset-size path segment filter + which splits subtree to consult (default small)",
    )
    ap.add_argument(
        "--fold", default=None,
        help="restrict to a single fold (e.g. fold_0). When omitted, all fold_N "
             "dirs matching --dataset-size under --source are included.",
    )
    ap.add_argument(
        "--label-col", default="perturbation",
        help="adata.obs column to use as the classification label (default perturbation)",
    )
    ap.add_argument("--group-by", nargs="*", default=None,
                    help="obs columns defining a GROUP (e.g. dataset perturbation). When "
                         "given, each group is scored on the mean log-prob over its cells "
                         "and written to all_metrics_grouped.csv NEXT TO the per-cell table. "
                         "On a ~9.9k-class space with ~2 cells per (dataset,pert), the group "
                         "is the meaningful unit and the per-cell number is mostly noise.")
    ap.add_argument("--over-random", action="store_true",
                    help="also emit <metric>_over_random = metric / analytic uniform floor. "
                         "Absolute values are ~1e-3 at this class count; the ratio is what is "
                         "actually readable.")
    ap.add_argument("--classes", type=Path, default=None,
                    help="newline file pinning the label space (one class per line). "
                         "Without it, the space is the cross-method union as before.")
    ap.add_argument("--ovr-auroc", action="store_true",
                    help="also compute macro one-vs-rest AUROC (ranks CELLS within each "
                         "class, unlike auc_rank which ranks CLASSES within each cell). "
                         "One sort per class per file -- off by default.")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="restrict metrics to these splits (default: all). Typical: id_test ood_test eval_iid eval_ood")
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 5, 10], help="k values for recall@k")
    ap.add_argument("--recall-mode", choices=("balanced", "cells"), default="balanced",
                    help="which recall@k flavour the PLOTS and the method ranking report. "
                         "Both are always computed and both land in the CSVs; this only "
                         "picks the one that is shown. 'balanced' (default) averages the "
                         "per-class recalls; 'cells' pools cells, so classes with more "
                         "cells count for more. They diverge sharply on ood_test, which "
                         "carries the natural class distribution -- accuracy stays "
                         "cell-pooled either way (it is what it is), and macro_f1 is "
                         "already class-balanced.")
    ap.add_argument("--tie-break-seed", type=int, default=0,
                    help="seed for the class-order shuffle that makes score ties break "
                         "independently of perturbation name; -1 keeps alphabetical order")
    ap.add_argument("--knn", action="store_true", help="compute knn_acc (fit on each method's val split)")
    ap.add_argument("--knn-k", type=int, default=5, help="k for kNN classifier (default 5)")
    ap.add_argument("--min-efficiency", type=float, default=None,
                    help="single-run filter on --efficiency-col. Ignored if --efficiency-sweep is set.")
    ap.add_argument("--efficiency-col", default="efficiency",
                    help="column for efficiency filters (accepts 'efficiency' or 'obs_efficiency'; default 'efficiency')")
    ap.add_argument("--efficiency-sweep", nargs="*", default=None,
                    help="sweep over multiple thresholds; each gets its own subdir and all share one "
                         "efficiency_sweep.png plot. Use 'none' for the no-filter baseline. "
                         "Example: --efficiency-sweep none 1 2 4")
    ap.add_argument("--replot", action="store_true",
                    help="redraw every figure from the tables an earlier run wrote to "
                         "--output-dir (no logits read, no metrics recomputed, no tables "
                         "rewritten). Pass the same --efficiency-sweep / --recall-mode.")
    ap.add_argument("--top-plots-only", action="store_true",
                    help="with --replot: redraw only the top-level plots/ figures (weighted "
                         "comparisons, sweeps), not the per-threshold ones")
    ap.add_argument("--plot-format", default="svg", choices=("svg", "png", "pdf"),
                    help="output format for all plots (default svg)")
    ap.add_argument("--write-stacked", action="store_true",
                    help="also write all_logits.parquet (the full stacked log-prob frame). "
                         "Off by default — at full-dataset scale this file is hundreds of GB "
                         "and roughly doubles peak RAM during the parquet write.")
    ap.add_argument("--obs-cols", nargs="*", default=None,
                    help="restrict adata.obs columns attached as obs_<col> "
                         "(default: all). Useful at full-dataset scale to limit RAM.")
    ap.add_argument("--ignore-methods", nargs="*", default=None,
                    help="method names to skip entirely (matched against the resolved "
                         "method string from --method-depth, e.g. 'forward/cellot' or 'cellot'). "
                         "All of the method's files are dropped before any matrix is loaded.")
    ap.add_argument("--log-level", default="INFO",
                    help="logging level (default INFO; pass WARNING to silence per-file notes)")
    args = ap.parse_args(argv)
    RECALL_PREFIX, W_RECALL_PREFIX = _recall_prefixes(args.recall_mode)
    log.info("[recall] reporting the %r flavour (%s / %s); BOTH are written to the CSVs",
             args.recall_mode, RECALL_PREFIX, W_RECALL_PREFIX)
    global TIE_BREAK_SEED, FIXED_CLASSES, OVR_AUROC, _NO_WRITE
    _NO_WRITE = bool(args.replot)
    if args.top_plots_only and not args.replot:
        raise SystemExit("--top-plots-only only makes sense with --replot (a scoring run "
                         "would skip writing the per-threshold tables)")
    OVR_AUROC = bool(getattr(args, "ovr_auroc", False))
    TIE_BREAK_SEED = args.tie_break_seed
    if args.classes:
        FIXED_CLASSES = [l.strip() for l in args.classes.read_text().splitlines() if l.strip()]
        if not FIXED_CLASSES:
            raise SystemExit(f"--classes {args.classes} is empty")
        if len(set(FIXED_CLASSES)) != len(FIXED_CLASSES):
            raise SystemExit(f"--classes {args.classes} has duplicate entries")
        log.info("[classes] label space PINNED to %d classes from %s (chance %.3g)",
                 len(FIXED_CLASSES), args.classes, 1.0 / len(FIXED_CLASSES))

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # build the threshold list: sweep overrides single-value
    if args.efficiency_sweep is not None:
        thresholds: list[float | None] = []
        for v in args.efficiency_sweep:
            if str(v).lower() in ("none", "na", "null"):
                thresholds.append(None)
            else:
                thresholds.append(float(v))
        # dedupe while preserving order
        seen: set = set()
        thresholds = [t for t in thresholds if (t not in seen and not seen.add(t))]
    else:
        thresholds = [args.min_efficiency]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_stacked:
        log.warning("[stacked] --write-stacked is not supported in streaming mode; ignoring")
    if args.knn:
        log.warning("[knn] disabled in streaming mode (would need val + test in RAM together)")
    obs_cols = tuple(args.obs_cols) if args.obs_cols else None

    splits_filter = tuple(_canonical_split(s) for s in args.splits) if args.splits else None
    ks = tuple(args.ks)
    plot_ext = args.plot_format

    # Phase 1: discover files + peek column names + apply consistency rules.
    metas, cols_by_path = discover_files(
        args.source, method_depth=args.method_depth,
        dataset_size=args.dataset_size, fold=args.fold,
    )

    # drop ignored methods up front so they don't influence the cross-method
    # consistency check (which would otherwise pick a bad reference set).
    if args.ignore_methods:
        ignore = set(args.ignore_methods)
        before = len(metas)
        metas = [m for m in metas if m.method not in ignore]
        log.info(
            "[ignore] dropped %d/%d files matching --ignore-methods=%s",
            before - len(metas), before, sorted(ignore),
        )
        if not metas:
            raise RuntimeError(
                f"no files left after --ignore-methods filter ({sorted(ignore)})"
            )

    metas, method_unions = apply_consistency(metas, cols_by_path)

    # drop files whose canonical split isn't in --splits before loading any matrix
    if splits_filter is not None:
        before = len(metas)
        metas = [m for m in metas if m.split in splits_filter]
        log.info(
            "[stream] %d/%d files match --splits=%s; rest skipped before load",
            len(metas), before, list(splits_filter),
        )
        if not metas:
            raise RuntimeError(
                f"no files left after --splits filter ({list(splits_filter)})"
            )

    if args.replot:
        # --replot: skip loading + scoring entirely and rebuild the frames the scoring
        # loop would have produced from the tables a previous run wrote. Every figure is
        # a pure function of those tables, so layout / label fixes cost minutes instead
        # of a full re-score (2.5h for ctx7).
        (sweep_metrics, sweep_per_fold, sweep_perclass, sweep_perdataset,
         sweep_perdataset_macro, sweep_cc, sweep_group, rows_grouped, rows_weighted,
         train_stats_df, train_support, train_class_stats_df) = _load_plot_state(
            args.output_dir, thresholds)
        multi_fold = len({m.fold for m in metas if m.fold is not None}) > 1
    else:
        obs = None
        size_dir = None
        if args.splits_dir is not None:
            obs, size_dir, _ = load_obs(args.splits_dir, args.dataset_size, args.label_col)

        train_stats_df: pd.DataFrame | None = None
        train_support: pd.Series | None = None
        train_class_stats_df: pd.DataFrame | None = None
        if args.splits_dir is not None:
            train_stats_df = training_stats(
                args.splits_dir, dataset_size=args.dataset_size,
                fold=args.fold, label_col=args.label_col,
            )
            _write(train_stats_df, args.output_dir / "train_stats.csv")
            log.info("train stats:\n%s", train_stats_df.to_string(index=False))

            train_support = training_class_support(
                args.splits_dir, dataset_size=args.dataset_size,
                fold=args.fold, label_col=args.label_col,
            )
            _write(
                train_support.reset_index().rename(columns={"index": "class"}),
                args.output_dir / "train_class_support.csv",
            )

            # Per-class training-set features for the diagnostics plot
            # (n_train_cells / n_train_datasets / cells_per_dataset / dataset_entropy).
            train_class_stats_df = training_class_stats(
                args.splits_dir, dataset_size=args.dataset_size,
                fold=args.fold, label_col=args.label_col,
            )
            _write(
                train_class_stats_df, args.output_dir / "train_class_stats.csv",
            )

        # Phase 2: stream — for each file, load → score all thresholds → discard.
        multi_fold = len({m.fold for m in metas if m.fold is not None}) > 1

        rows_metrics:         list[pd.DataFrame] = []
        rows_per_fold:        list[pd.DataFrame] = []
        rows_grouped:         list[pd.DataFrame] = []
        rows_perclass:        list[pd.DataFrame] = []
        rows_perdataset:      list[pd.DataFrame] = []
        rows_perdataset_macro: list[pd.DataFrame] = []
        rows_cell_counts:     list[pd.DataFrame] = []
        rows_group:           list[pd.DataFrame] = []
        rows_weighted:        list[pd.DataFrame] = []

        import gc

        for i, meta in enumerate(metas, 1):
            log.info("[stream %d/%d] %s", i, len(metas), meta.path)
            union_sorted = method_unions[(meta.direction, meta.method)]
            fd_full = load_one_file(
                meta, obs=obs, size_dir=size_dir,
                label_col=args.label_col, obs_cols=obs_cols,
                union_sorted=union_sorted,
            )
            if fd_full is None:
                log.warning("[stream %d/%d] skipping %s (see warning above)",
                            i, len(metas), meta.path)
                continue

            # Every file is split into per-dataset views, each carrying
            # split=<original>:<dataset>, so all of id_test / id_val / ood_test get
            # symmetric per-dataset metrics. Single-dataset files yield a
            # one-element list so the loop stays uniform.
            for fd in _split_by_dataset(fd_full):
                split_tag = fd.meta.split

                for thresh in thresholds:
                    thresh_val = np.nan if thresh is None else float(thresh)
                    if thresh is None:
                        fd_t = fd
                    else:
                        try:
                            mask = _filedata_efficiency_mask(fd, args.efficiency_col, thresh)
                        except KeyError:
                            log.warning(
                                "[%s thresh=%g] %s: %s not in obs — skipping efficiency-filtered "
                                "scoring for this file (it'll only show up in the unfiltered run)",
                                split_tag, thresh, meta.path, args.efficiency_col,
                            )
                            continue
                        kept = int(mask.sum())
                        log.info("[%s thresh=%g] kept %d/%d rows (%.1f%%)",
                                 split_tag, thresh, kept, fd.n_rows,
                                 100 * kept / max(fd.n_rows, 1))
                        fd_t = fd.filter_mask(mask)
                        if fd_t.n_rows == 0:
                            continue

                    m = score_one(fd_t, ks=ks)
                    if args.over_random:
                        m = add_over_random(m, len(fd_t.classes), ks)
                    m["min_efficiency"] = thresh_val
                    rows_metrics.append(pd.DataFrame([m]))

                    if args.group_by:
                        g = score_grouped_one(fd_t, ks=ks, group_cols=list(args.group_by))
                        if g:
                            g["min_efficiency"] = thresh_val
                            rows_grouped.append(pd.DataFrame([g]))

                    if multi_fold:
                        # per-fold scoring is identical to per-file scoring under streaming
                        # (one fold per file). just re-use the row.
                        rows_per_fold.append(pd.DataFrame([m | {"min_efficiency": thresh_val}]))

                    pc = score_per_class_one(fd_t, ks=ks)
                    if not pc.empty:
                        pc["min_efficiency"] = thresh_val
                        rows_perclass.append(pc)

                    if obs is not None and f"{OBS_PREFIX}dataset" in fd_t.obs.columns:
                        pdd = score_per_dataset_class_one(fd_t)
                        if not pdd.empty:
                            pdd["min_efficiency"] = thresh_val
                            pdd["fold"] = meta.fold
                            rows_perdataset.append(pdd)
                        pdm = score_per_dataset_one(fd_t)
                        if not pdm.empty:
                            pdm["min_efficiency"] = thresh_val
                            pdm["fold"] = meta.fold
                            rows_perdataset_macro.append(pdm)

                    cc = cells_per_perturbation_one(fd_t)
                    cc["min_efficiency"] = thresh_val
                    cc["fold"] = meta.fold
                    rows_cell_counts.append(cc)

                    if obs is not None and f"{OBS_PREFIX}dataset" in fd_t.obs.columns:
                        grp = score_group_level_one(fd_t, ks=ks)
                        if grp is not None:
                            grp["min_efficiency"] = thresh_val
                            rows_group.append(pd.DataFrame([grp]))

                    if fd_t is not fd:
                        del fd_t

                # weighted metrics — once per (file × OOD-dataset) view, using the
                # full set of cells in that view weighted by efficiency.
                ws = score_weighted_one(fd, ks=ks, weight_col=args.efficiency_col)
                if ws is not None:
                    rows_weighted.append(pd.DataFrame([ws]))

            del fd_full
            gc.collect()

        # Phase 3: aggregate + plot per threshold + sweep plots.
        sweep_metrics    = pd.concat(rows_metrics,    ignore_index=True)
        sweep_per_fold   = pd.concat(rows_per_fold,   ignore_index=True) if rows_per_fold   else None
        sweep_perclass   = pd.concat(rows_perclass,   ignore_index=True)
        sweep_perdataset = pd.concat(rows_perdataset, ignore_index=True) if rows_perdataset else None
        sweep_perdataset_macro = (
            pd.concat(rows_perdataset_macro, ignore_index=True)
            if rows_perdataset_macro else None
        )
        sweep_cc         = pd.concat(rows_cell_counts, ignore_index=True)
        sweep_group      = pd.concat(rows_group,      ignore_index=True) if rows_group      else None

    n_classes_all = len(method_unions[next(iter(method_unions))]) if method_unions else 0
    for _fr in (sweep_metrics, sweep_per_fold, sweep_group):
        _derive_scaled_rank(_fr, n_classes_all)
    if rows_weighted:
        rows_weighted = [_derive_scaled_rank(pd.concat(rows_weighted, ignore_index=True),
                                             n_classes_all)]

    # palette ranks across all thresholds combined so colours are stable across plots
    # auto-detect baselines from method name (case-insensitive substring match)
    ranked = _rank_methods(sweep_metrics, RECALL_PREFIX)
    baselines = tuple(m for m in ranked if "baseline" in m.lower())
    if baselines:
        log.info("[baselines] auto-detected (greyscale palette): %s", list(baselines))
    palette = _method_palette(ranked, baselines=baselines)

    def _agg_folds(per_file: pd.DataFrame) -> pd.DataFrame:
        """Mean numeric metrics across folds, one row per (direction, method, split)."""
        gcols = [c for c in ("direction", "method", "split") if c in per_file.columns]
        num_cols = [c for c in per_file.columns
                    if c not in (*gcols, "fold", "min_efficiency")
                    and pd.api.types.is_numeric_dtype(per_file[c])]
        return per_file.groupby(gcols, sort=False, dropna=False)[num_cols].mean().reset_index()

    # --top-plots-only (with --replot): skip the per-threshold figures, which are most of
    # a replot's hour, and draw only the top-level plots/ comparisons.
    for thresh in ([] if args.top_plots_only else
                   tqdm(thresholds, unit='threshold', desc='Calculating cutoffs')):
        thresh_val = np.nan if thresh is None else float(thresh)
        sub = args.output_dir / _eff_subdir(thresh)
        plots_sub = sub / "plots"
        sub.mkdir(parents=True, exist_ok=True)
        plots_sub.mkdir(parents=True, exist_ok=True)

        def _slice(df: pd.DataFrame) -> pd.DataFrame:
            mask = (df["min_efficiency"].isna() if pd.isna(thresh_val)
                    else df["min_efficiency"] == thresh_val)
            return df[mask].drop(columns="min_efficiency")

        # Per-dataset views (split=<orig>:<dataset>) are kept on disk for the
        # full breakdown, but plots collapse the suffix so each dataset becomes
        # one of N data points within its parent split (id_test / ood_test /
        # ...). The `_canon` versions drive plotting; the un-suffixed ones get
        # written out.
        def _canon_split(df: pd.DataFrame) -> pd.DataFrame:
            if "split" not in df.columns or df.empty:
                return df
            out = df.copy()
            out["split"] = out["split"].astype(str).str.split(":", n=1).str[0]
            return out

        m_files = _slice(sweep_metrics)         # per-file rows (one per fold × dataset)
        m_merged = _agg_folds(m_files)
        _write(m_merged, sub / "all_metrics.csv")

        # Group-level table (--group-by): one row per (method, split, fold) scored on
        # (dataset, perturbation) GROUPS instead of cells. Written beside the per-cell
        # table rather than replacing it, so both readings are available.
        if rows_grouped:
            _g = _slice(pd.concat(rows_grouped, ignore_index=True))
            if not _g.empty:
                _write(_g, sub / "all_metrics_grouped.csv")
        log.info("[%s] merged metrics:\n%s",
                 _eff_subdir(thresh), m_merged.to_string(index=False))

        pc = _slice(sweep_perclass)
        _write(pc, sub / "all_perclass_metrics.parquet")

        if sweep_per_fold is not None:
            pf = _slice(sweep_per_fold)
            _write(pf, sub / "all_metrics_per_fold.csv")

        if sweep_perdataset is not None:
            pdd = _slice(sweep_perdataset)
            _write(pdd, sub / "all_perdataset_class_metrics.parquet")

        if sweep_perdataset_macro is not None:
            pdm_files = _slice(sweep_perdataset_macro)
            # Aggregate across folds (one row per direction/method/split/dataset).
            gcols = [c for c in ("direction", "method", "split", "dataset")
                     if c in pdm_files.columns]
            num_cols = [c for c in pdm_files.columns
                        if c not in (*gcols, "fold")
                        and pd.api.types.is_numeric_dtype(pdm_files[c])]
            pdm = (pdm_files.groupby(gcols, sort=False, dropna=False)[num_cols]
                            .mean().reset_index()) if num_cols else pdm_files
            _write(pdm, sub / "all_perdataset_metrics.csv")

        ranked = _rank_methods(m_merged, RECALL_PREFIX)
        log.info("[%s] method ranking (best→worst): %s", _eff_subdir(thresh), ranked)

        # plot inputs: per-fold for distribution plots, merged for class-level.
        # Collapse per-dataset split tags so the dataset axis becomes "more
        # points" on the same id_test/ood_test column instead of fanning out.
        plot_input = _canon_split(pf if sweep_per_fold is not None else m_merged)
        plot_pc = _canon_split(pc)
        # cell-counts: collapse first, then union class names across (canonical
        # split × dataset) so the facet title shows the parent split's pool.
        cc_canon = _canon_split(_slice(sweep_cc))
        counts_split = (
            cc_canon.groupby("split", sort=False)["label"].nunique().to_dict()
        )
        n_total = len(method_unions[next(iter(method_unions))])
        counts = {s: (q, n_total) for s, q in counts_split.items()}

        # topk_recall.png carries the REPORTED flavour (--recall-mode); the other one
        # is written beside it so both are always on disk to compare.
        other = "recall@" if RECALL_PREFIX != "recall@" else "balanced_recall@"
        plot_topk_recall(plot_input, plots_sub / f"topk_recall.{plot_ext}",
                         ks=ks, palette=palette, counts=counts, prefix=RECALL_PREFIX)
        plot_topk_recall(plot_input,
                         plots_sub / f"topk_recall_{other.rstrip('@')}.{plot_ext}",
                         ks=ks, palette=palette, counts=counts, prefix=other)
        plot_macro_f1(plot_input, plots_sub / f"macro_f1.{plot_ext}",
                      palette=palette, counts=counts)
        if "macro_f1_no_ds_bias" in plot_input.columns:
            plot_macro_f1(
                plot_input,
                plots_sub / f"macro_f1_no_ds_bias.{plot_ext}",
                palette=palette, counts=counts,
                metric="macro_f1_no_ds_bias",
                title_label="Macro F1 (no dataset bias)",
            )
        # Rank metrics (add_rank_metrics) as their own figures. auroc.png carries the
        # REPORTED flavour like topk_recall.png, and the other is written beside it.
        # Chance is drawn: 0.5 for AUROC at any class count, (C+1)/2 for median rank.
        _plot_rank_figures(plot_input, plots_sub, plot_ext, palette, counts, n_total,
                           RECALL_PREFIX)
        plot_perclass_f1_boxen(plot_pc, plots_sub / f"perclass_f1_boxen.{plot_ext}",
                               palette=palette, counts=counts)
        plot_perclass_topk_boxen(plot_pc, plots_sub / f"perclass_topk_boxen.{plot_ext}",
                                 ks=ks, palette=palette, counts=counts)
        if sweep_perdataset is not None and not pdd.empty:
            # perdataset_class plots are already organised by `dataset` column;
            # the split tag isn't read by the plotter, no canonicalisation needed.
            plot_perdataset_f1_boxen(
                pdd, plots_sub / f"perdataset_f1_boxen.{plot_ext}",
                palette=palette, train_stats=train_stats_df,
            )
        plot_support_vs_performance(
            plot_pc, plots_sub / f"support_vs_f1.{plot_ext}",
            palette=palette, metric="f1", train_support=train_support,
        )

        # Class-level diagnostic: do high-support / high-coverage classes do
        # better? One scatter per (training-feature × split), Spearman ρ
        # per method annotated. Helps separate "low recall = under-trained"
        # from "low recall = inherently hard regardless of training depth".
        if train_class_stats_df is not None and not plot_pc.empty:
            plot_class_training_diagnostics(
                plot_pc, train_class_stats_df,
                plots_sub / f"class_training_diagnostics.{plot_ext}",
                palette=palette, metric="f1",
            )

        # group-level (dataset × perturbation) — pool cells per group, take
        # mean log-prob per class, predict + score (a la fingerprinting).
        if sweep_group is not None:
            grp_files = _slice(sweep_group)
            if not grp_files.empty:
                grp_merged = _agg_folds(grp_files)
                _write(grp_merged, sub / "all_metrics_group.csv")
                log.info("[%s] group-level metrics:\n%s",
                         _eff_subdir(thresh), grp_merged.to_string(index=False))
                plot_input_g = _canon_split(grp_files if multi_fold else grp_merged)
                plot_topk_recall(plot_input_g, plots_sub / f"topk_recall_group.{plot_ext}",
                                 prefix=RECALL_PREFIX,
                                 ks=ks, palette=palette, counts=counts)
                plot_macro_f1(plot_input_g, plots_sub / f"macro_f1_group.{plot_ext}",
                              palette=palette, counts=counts)
                _plot_rank_figures(plot_input_g, plots_sub, plot_ext, palette, counts,
                                   n_total, RECALL_PREFIX, suffix="_group")
        log.info("[%s] wrote plots to %s", _eff_subdir(thresh), plots_sub)

    top_plots = args.output_dir / "plots"
    top_plots.mkdir(parents=True, exist_ok=True)

    # Same split canonicalisation as the per-threshold dispatch above —
    # collapse per-dataset tags so plots show one column per parent split
    # with N points (datasets × folds × thresholds) inside it.
    def _canon_split_top(df: pd.DataFrame) -> pd.DataFrame:
        if "split" not in df.columns or df.empty:
            return df
        out = df.copy()
        out["split"] = out["split"].astype(str).str.split(":", n=1).str[0]
        return out

    # F1 / recall@5 / recall@10 variant of the weighted + sweep panels. accuracy and
    # recall@1 are near the floor at 200 classes and compress the y-range; @5/@10 are
    # where the methods actually separate. Uses the reported recall flavour, so on a
    # balanced run these are the balanced columns.
    def _f1_r5_r10(prefix: str, f1: str) -> list[str]:
        return [f1, f"{prefix}5", f"{prefix}10"]

    if rows_weighted:
        weighted = pd.concat(rows_weighted, ignore_index=True)
        _write(weighted, args.output_dir / "weighted_metrics.csv")
        # 1) headline metrics (recall@10, MRR, scaled median rank) with chance lines
        hl_w, hl_w_refs = _headline_metrics(W_RECALL_PREFIX, n_classes_all, weighted=True)
        plot_weighted_metrics(
            _canon_split_top(weighted), top_plots / f"weighted_metrics.{plot_ext}",
            recall_prefix=W_RECALL_PREFIX, ks=ks, palette=palette,
            metrics=hl_w, refs=hl_w_refs,
        )
        # 1b) same panel restricted to F1 / recall@5 / recall@10
        plot_weighted_metrics(
            _canon_split_top(weighted), top_plots / f"weighted_metrics_f1_r5_r10.{plot_ext}",
            recall_prefix=W_RECALL_PREFIX, ks=ks, palette=palette,
            metrics=_f1_r5_r10(W_RECALL_PREFIX, "weighted_macro_f1"),
        )
        # 2+3) the same metric broken out by which distribution shift the fold
        # encodes -- one figure for in-distribution, one for held-out.
        shifted = canon_split_shift(weighted)
        for tag, prefix, group_order in (
            ("ID", "id_", ID_GROUP_ORDER), ("OOD", "ood_", OOD_GROUP_ORDER),
        ):
            sub = shifted[shifted["split"].astype(str).str.startswith(prefix)]
            if sub.empty:
                continue
            order = [g for g in group_order if g in set(sub["split"])]
            sub = sub.copy()
            sub["split"] = pd.Categorical(sub["split"], categories=order, ordered=True)
            sub = sub.sort_values("split")
            plot_weighted_metrics(
                sub, top_plots / f"{tag}_weighted_metrics.{plot_ext}",
                recall_prefix=W_RECALL_PREFIX, ks=ks, palette=palette,
                metrics=hl_w, refs=hl_w_refs,
            )
            plot_weighted_metrics(
                sub, top_plots / f"{tag}_weighted_metrics_f1_r5_r10.{plot_ext}",
                recall_prefix=W_RECALL_PREFIX, ks=ks, palette=palette,
                metrics=_f1_r5_r10(W_RECALL_PREFIX, "weighted_macro_f1"),
            )
            log.info("wrote %s weighted-metrics plots (%s)", tag, ", ".join(order))
        # Weighted rank metrics as individual benchmark figures, in the same three
        # views as weighted_metrics (overall, and by the shift the fold encodes).
        n_cls = len(method_unions[next(iter(method_unions))])
        w_auroc = ("balanced_weighted_mean_auroc" if W_RECALL_PREFIX.startswith("balanced")
                   else "weighted_mean_auroc")
        rank_chance = (n_cls + 1) / 2
        rank_figs = []
        if w_auroc in weighted.columns:
            rank_figs.append(("weighted_auroc", w_auroc, 0.5, _auroc_ylim(weighted[w_auroc]),
                              f"{_metric_label(w_auroc)}"))
        if "weighted_median_rank" in weighted.columns:
            rank_figs.append((
                "weighted_median_rank", "weighted_median_rank", rank_chance,
                _rank_ylim(weighted["weighted_median_rank"], rank_chance),
                f"Weighted median rank of the true class (of {n_cls:,}; lower = better)"))
        for fname, key, ref, yl, ttl in rank_figs:
            views = [("", _canon_split_top(weighted))]
            for tag, prefix, group_order in (
                ("ID_", "id_", ID_GROUP_ORDER), ("OOD_", "ood_", OOD_GROUP_ORDER),
            ):
                v = shifted[shifted["split"].astype(str).str.startswith(prefix)]
                if v.empty:
                    continue
                order = [g for g in group_order if g in set(v["split"])]
                v = v.copy()
                v["split"] = pd.Categorical(v["split"], categories=order, ordered=True)
                views.append((tag, v.sort_values("split")))
            for tag, view in views:
                plot_weighted_metrics(
                    view, top_plots / f"{tag}{fname}.{plot_ext}",
                    recall_prefix=W_RECALL_PREFIX, ks=ks, palette=palette,
                    metrics=[key], refs={key: ref}, ylims={key: yl}, title=ttl,
                )
        log.info("wrote weighted metrics + plot to %s",
                 top_plots / f"weighted_metrics.{plot_ext}")

    # Raw (unsliced, per-fold) frames that --replot needs and nothing else writes.
    if sweep_group is not None:
        _write(sweep_group, args.output_dir / "sweep_group.csv")
    _write(sweep_cc, args.output_dir / "sweep_cells_per_perturbation.parquet")
    if len(thresholds) == 1:
        _write(sweep_metrics, args.output_dir / "sweep_metrics.csv")
        if sweep_per_fold is not None:
            _write(sweep_per_fold, args.output_dir / "sweep_metrics_per_fold.csv")

    if len(thresholds) > 1:
        _write(sweep_metrics, args.output_dir / "sweep_metrics.csv")
        sweep_input = sweep_per_fold if sweep_per_fold is not None else sweep_metrics
        if sweep_per_fold is not None:
            _write(sweep_per_fold, args.output_dir / "sweep_metrics_per_fold.csv")
        hl_u, _ = _headline_metrics(RECALL_PREFIX, n_classes_all, weighted=False)
        plot_efficiency_sweep(_canon_split_top(sweep_input),
                              top_plots / f"efficiency_sweep.{plot_ext}",
                              ks=ks, palette=palette, metrics=hl_u)
        # same panel restricted to F1 / recall@5 / recall@10 (see _f1_r5_r10)
        plot_efficiency_sweep(_canon_split_top(sweep_input),
                              top_plots / f"efficiency_sweep_f1_r5_r10.{plot_ext}",
                              ks=ks, palette=palette,
                              metrics=_f1_r5_r10(RECALL_PREFIX, "macro_f1"))
        if sweep_per_fold is not None:
            plot_ood_matched_sweep(
                sweep_per_fold, top_plots / f"ood_sweep_matched.{plot_ext}",
                metrics=hl_u, palette=palette)
            plot_ood_matched_sweep(
                sweep_per_fold, top_plots / f"ood_sweep_matched_f1_r5_r10.{plot_ext}",
                metrics=_f1_r5_r10(RECALL_PREFIX, "macro_f1"), palette=palette)
            for _ext in ("svg", plot_ext):
                save_method_legend(palette, top_plots / f"legend.{_ext}")
        log.info("wrote sweep plots to %s", top_plots / f"efficiency_sweep.{plot_ext}")

        plot_cells_per_perturbation_sweep(
            _canon_split_top(sweep_cc),
            top_plots / f"cells_per_perturbation_sweep.{plot_ext}",
        )
        log.info("wrote class-count sweep plot to %s",
                 top_plots / f"cells_per_perturbation_sweep.{plot_ext}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
