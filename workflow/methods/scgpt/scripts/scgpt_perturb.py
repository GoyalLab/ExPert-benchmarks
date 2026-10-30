import argparse
import csv
import gc
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader, Dataset, Sampler
from scipy import sparse

warnings.filterwarnings("ignore", "Observation names are not unique", UserWarning)
logging.getLogger("scvi").setLevel(logging.WARNING)


# Resolve paths relative to this file's location in the repo
HERE = Path(__file__).resolve().parent.parent
SCGPT_SRC = HERE / "scGPT"
if not (SCGPT_SRC / "scgpt" / "__init__.py").exists():
    raise FileNotFoundError(f"Expected vendored scGPT package at {SCGPT_SRC / 'scgpt'}")
sys.path.insert(0, str(SCGPT_SRC))
if importlib.util.find_spec("IPython") is None:
    import types
    ipython_stub = types.ModuleType("IPython")
    ipython_stub.get_ipython = lambda: None
    sys.modules["IPython"] = ipython_stub
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id

PERTURBATION_KEY = "perturbation"
CONTROL_LABEL = "control"
POOL_PERTURBATION_KEYS = (
    "pool_perturbation",
    "perturbation",
    "gene",
    "target",
    "gene_target",
)
P_COLS = (
    "perturbation",
    "gene",
    "perturbed_gene_name",
    "perturbation_1",
    "target",
    "gene_target",
    "guides_collapsed_by_gene",
)
CTRL_KEYS = (
    "control",
    "non-targeting",
    "nt",
    "ntc",
)
SCGPT_LABEL_KEY = "__scgpt_label__"
SCGPT_SHARED_LABEL = "__shared__"
DEFAULT_EVAL_SPLITS = ("ood_test",)
GENERALIZATION_KEY_ALIASES = (
    "cell_type",
    "celltype",
    "CellType",
    "cell_type_lvl2",
    "cell_type_l2",
    "dataset",
)
SUMMARY_FIELDNAMES = (
    "cell_id",
    "split",
    "generalization_key",
    "generalization_group",
    "true_perturbation",
    "original_rank",
    "predicted_rank",
    "original_top_prediction",
    "predicted_top_prediction",
    "original_top_score",
    "predicted_top_score",
    "original_recall_at_10",
    "original_recall_at_100",
    "predicted_recall_at_10",
    "predicted_recall_at_100",
    "original_ranked_candidates",
    "predicted_ranked_candidates",
    "original_ranked_scores",
    "predicted_ranked_scores",
)
RANKING_FIELDNAMES = (
    "cell_id",
    "split",
    "generalization_group",
    "true_perturbation",
    "query_kind",
    "rank_position",
    "candidate_perturbation",
    "score",
)
CELL_FLUSH_BATCH = 8
INFERENCE_BATCH_SIZE = 4
DEFAULT_TRAIN_STEPS_PER_EPOCH = 500
SCGPT_PAD_TOKEN = "<pad>"
SCGPT_SPECIAL_TOKENS = (SCGPT_PAD_TOKEN, "<cls>", "<eoc>")
SCGPT_PAD_VALUE = 0
SCGPT_PERT_PAD_ID = 0
SCGPT_INCLUDE_ZERO_GENE = "all"

class DeadlineExceeded(RuntimeError):
    pass


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        slurm_job_gpus = os.environ.get("SLURM_JOB_GPUS", "<unset>")
        slurm_step_gpus = os.environ.get("SLURM_STEP_GPUS", "<unset>")
        if not torch.cuda.is_available():
            if torch.version.cuda is None:
                cuda_reason = (
                    "this is a CPU-only PyTorch build. Use a Python environment "
                    "with CUDA-enabled torch for multi-GPU scGPT training."
                )
            else:
                cuda_reason = (
                    "PyTorch was built with CUDA, but no CUDA device is visible "
                    "to this process."
                )
            raise RuntimeError(
                "Distributed scGPT training requires CUDA, but "
                f"{cuda_reason} "
                f"torch.__version__={torch.__version__!r}, "
                f"torch.version.cuda={torch.version.cuda!r}, "
                f"torch.cuda.device_count()={torch.cuda.device_count()}, "
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}, "
                f"SLURM_JOB_GPUS={slurm_job_gpus!r}, "
                f"SLURM_STEP_GPUS={slurm_step_gpus!r}, "
                f"Python executable: {sys.executable}"
            )
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                "Distributed scGPT local rank exceeds visible CUDA device count: "
                f"local_rank={local_rank}, device_count={device_count}, "
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}, "
                f"SLURM_JOB_GPUS={slurm_job_gpus!r}, "
                f"SLURM_STEP_GPUS={slurm_step_gpus!r}, "
                f"Python executable: {sys.executable}"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args: argparse.Namespace) -> bool:
    context = getattr(args, "distributed_context", None)
    return context is None or context.is_main


def main_print(args: argparse.Namespace, message: str) -> None:
    if is_main_process(args):
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ctrl-path", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--perturbation-pool-file", type=Path, default=None)
    parser.add_argument("--disable-perturbation-pool", action="store_true")
    parser.add_argument("--generalization-key", type=str, default=None)
    parser.add_argument("--holdout-value", type=str, default=None)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=list(DEFAULT_EVAL_SPLITS),
        choices=["id_val", "id_test", "ood_test"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--early-stopping-patience", type=int, required=True)
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Directory to cache the fine-tuned model (best_model.pt). If it already "
        "contains best_model.pt, training is skipped and the cached weights are loaded "
        "(the vocab still comes from --load-model).",
    )
    parser.add_argument(
        "--no-logits-csv",
        action="store_true",
        help="Skip writing the legacy per-cell logits.csv; logits.h5 is the scoring artifact.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train (and cache via --save-model) then exit without evaluating.",
    )
    parser.add_argument("--pert-chunk-idx", type=int, default=0,
                        help="With --n-pert-chunks>1, evaluate only this strided slice of "
                             "candidate perturbations (the cached model is loaded, no training).")
    parser.add_argument("--n-pert-chunks", type=int, default=1,
                        help="Split candidate-perturbation evaluation into this many chunks "
                             "(parallel jobs); each writes a partial logits.h5 to concat later.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--embsize", type=int, default=512)
    parser.add_argument("--d-hid", type=int, default=512)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n-layers-cls", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-fast-transformer", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-steps-per-epoch", type=int, default=DEFAULT_TRAIN_STEPS_PER_EPOCH)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--score-n-genes", type=int, default=None,
                        help="Restrict every profile comparison to the FIRST N gene columns. "
                             "Use it with a padded store: the synthetic target columns then "
                             "exist only so pert_flags[:, idx]=1 has a position to write to, "
                             "and never enter the metric. Under --classifier-readout the model "
                             "predicts from CONTROL cells, so the measured cell reaches the "
                             "score only through this cosine -- excluding the padded block here "
                             "removes the label leak outright, with no second store and no "
                             "train/eval mismatch. Controls carry no dips either way (verified: "
                             "padded columns in the control store are background Poisson).")
    parser.add_argument("--min-train-cells-per-perturbation", type=int, required=True)
    parser.add_argument("--max-control-cells-per-group", type=int, default=None)
    parser.add_argument(
        "--max-control-cells-per-context",
        dest="max_control_cells_per_group",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-eval-cells-per-split", type=int, default=None)
    # --- classifier readout -------------------------------------------------
    # DEFAULT (legacy) readout writes logits = cosine(model prediction of the cell's
    # OWN TRUE perturbation, candidate centroid). That prediction is conditioned on
    # the label, so the logits are derived from the answer -- it measures prediction
    # FIDELITY, not classification, and cannot be compared with the reverse methods.
    # --classifier-readout instead asks the model, per dataset: "what does candidate
    # p look like here?" for EVERY p (never using the cell's label), then scores each
    # MEASURED profile against those predictions. Cost drops from n_cells forward
    # passes per chunk to n_candidates x K, independent of test-set size.
    parser.add_argument("--classifier-readout", action="store_true",
                        help="Score measured profiles against per-candidate predictions "
                             "built from the group's control cells (no label leakage).")
    parser.add_argument("--pred-control-cells", type=int, default=32,
                        help="K control cells averaged into each candidate's predicted "
                             "profile under --classifier-readout.")
    parser.add_argument("--deadline-hours", type=float, default=None)
    parser.add_argument("--deadline-buffer-minutes", type=float, default=20.0)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--predicted-h5ad", type=Path, required=True)
    parser.add_argument("--original-h5ad", type=Path, required=True)
    parser.add_argument("--rankings-all-csv", type=Path, required=True)
    parser.add_argument("--rankings-top10-csv", type=Path, required=True)
    parser.add_argument("--rankings-top100-csv", type=Path, required=True)
    parser.add_argument("--no-save-h5ad", action="store_true")
    parser.add_argument("--no-rankings-all", action="store_true")
    # logits.h5 is the only artifact the benchmark consumes. The per-cell records
    # loop below (rank_single_row x2 + serialize_ranked_labels x4 over 200
    # candidates, for EVERY cell) is pure Python and dominates eval once the
    # forward passes are amortised by --classifier-readout -- --no-rankings-all
    # only changed WHICH csv got written, not whether the work ran.
    parser.add_argument("--logits-only", action="store_true",
                        help="Skip per-cell ranking/summary records and their CSVs; "
                             "write only logits.h5.")
    args = parser.parse_args()
    normalize_path_args(args)
    return args


def project_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return HERE / path


def normalize_path_args(args: argparse.Namespace) -> None:
    path_args = (
        "data_path",
        "ctrl_path",
        "split_dir",
        "perturbation_pool_file",
        "load_model",
        "output_csv",
        "summary_json",
        "predicted_h5ad",
        "original_h5ad",
        "rankings_all_csv",
        "rankings_top10_csv",
        "rankings_top100_csv",
    )
    for name in path_args:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, project_path(value))


def runtime_limit_seconds(args: argparse.Namespace) -> Optional[float]:
    if args.deadline_hours is None or args.deadline_hours <= 0:
        return None
    return max(0.0, float(args.deadline_hours) * 3600.0)


def deadline_buffer_seconds(args: argparse.Namespace) -> float:
    return max(0.0, float(args.deadline_buffer_minutes) * 60.0)


def elapsed_seconds(args: argparse.Namespace) -> float:
    return time.time() - float(getattr(args, "run_start_time", time.time()))


def remaining_runtime_seconds(args: argparse.Namespace) -> Optional[float]:
    limit = runtime_limit_seconds(args)
    if limit is None:
        return None
    return limit - elapsed_seconds(args)


def deadline_reached(args: argparse.Namespace) -> bool:
    remaining = remaining_runtime_seconds(args)
    return remaining is not None and remaining <= deadline_buffer_seconds(args)


def deadline_message(args: argparse.Namespace, phase: str) -> str:
    remaining = remaining_runtime_seconds(args)
    if remaining is None:
        remaining_text = "unlimited"
    else:
        remaining_text = f"{remaining / 60.0:.1f} min"
    return (
        f"Stopping before {phase}: remaining runtime is {remaining_text}, "
        f"buffer is {deadline_buffer_seconds(args) / 60.0:.1f} min."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _find_match_idx(columns: Sequence[str], candidates: Sequence[str]) -> int:
    lower_to_idx = {str(column).lower(): idx for idx, column in enumerate(columns)}
    for candidate in candidates:
        idx = lower_to_idx.get(str(candidate).lower())
        if idx is not None:
            return idx
    raise KeyError(f"Could not find any of {list(candidates)} in obs columns.")


def read_perturbation_pool(path: Path) -> pd.Series:
    if path is None:
        raise ValueError("A perturbation pool file is required when pool filtering is enabled.")
    if not path.exists():
        raise FileNotFoundError(f"Perturbation pool file does not exist: {path}")
    if path.suffix.lower() in {".txt", ".gz"} and path.name.endswith((".txt", ".txt.gz")):
        return pd.Series(np.atleast_1d(np.loadtxt(path, dtype=str)), dtype="object").dropna().astype(str)

    pool = pd.read_csv(path, index_col=0)
    for key in POOL_PERTURBATION_KEYS:
        if key in pool.columns:
            return pool[key].dropna().astype(str)
    if pool.shape[1] == 1:
        return pool.iloc[:, 0].dropna().astype(str)
    raise KeyError(
        f"Could not identify perturbation-pool column in {path}; "
        f"looked for {list(POOL_PERTURBATION_KEYS)}."
    )


def split_codes_for_original_rows(split_dir: Path, n_obs: int) -> np.ndarray:
    categories = ["train", "id_val", "id_test", "ood_test", "ignore"]
    codes = np.full(n_obs, categories.index("ignore"), dtype=np.int8)
    split_files = {
        "train": split_dir / "id_train.txt.gz",
        "id_val": split_dir / "id_val.txt.gz",
        "id_test": split_dir / "id_test.txt.gz",
        "ood_test": split_dir / "ood_test.txt.gz",
    }
    for split_name, path in split_files.items():
        idx = load_index_vector(path)
        codes[idx] = categories.index(split_name)
    return codes

def read_subset(
    adata_p: Path,
    p_col: str = PERTURBATION_KEY,
    ctrl_key: str = CONTROL_LABEL,
    perturbation_pool_file: Optional[Path] = None,
    use_perturbation_pool: bool = True,
    n_ctrl: Optional[int] = None,
    ensure_csr: bool = True,
    split_dir: Optional[Path] = None,
    seed: int = 42,
) -> ad.AnnData:
    categories = ["train","id_val","id_test","ood_test","ignore"]

    adata = ad.read_h5ad(adata_p, backed="r")
    p_key = adata.obs.columns[
        _find_match_idx(
            adata.obs.columns,
            P_COLS
        )
    ]

    logging.info(
        "Opened %d cells in backed mode",
        adata.n_obs
    )

    adata.obs[p_col] = (
        adata.obs[p_key]
        .astype(str)
        .values
    )

    subset_ctrl_mask=(
        adata.obs[p_col]
        .astype(str)
        .str.lower()
        .isin(CTRL_KEYS)
        .to_numpy()
    )
    if subset_ctrl_mask.any():
        ps=np.array(adata.obs[p_col], dtype=str)
        ps[subset_ctrl_mask]=ctrl_key
        adata.obs[p_col]=pd.Categorical(ps)

    if split_dir is not None:

        split_codes=split_codes_for_original_rows(
            split_dir,
            adata.n_obs
        )

        adata.obs["split"]=pd.Categorical.from_codes(
            split_codes,
            categories=categories
        )

    return adata


def load_index_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=int)
    values = np.atleast_1d(values)
    return values.astype(int)


def assign_splits(adata: ad.AnnData, split_dir: Path) -> ad.AnnData:
    if "split" in adata.obs.columns:
        return adata
    split = pd.Series("ignore", index=adata.obs_names, dtype="object")
    split_files = {
        "train": split_dir / "id_train.txt.gz",
        "id_val": split_dir / "id_val.txt.gz",
        "id_test": split_dir / "id_test.txt.gz",
        "ood_test": split_dir / "ood_test.txt.gz",
    }
    for split_name, path in split_files.items():
        idx = load_index_vector(path)
        split.iloc[idx] = split_name
    adata.obs["split"] = pd.Categorical(
        split,
        categories=["train", "id_val", "id_test", "ood_test", "ignore"],
    )
    return adata


def ensure_required_columns(adata: ad.AnnData, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in adata.obs.columns]
    if missing:
        raise KeyError(f"{label} is missing required obs columns: {missing}")


def resolve_generalization_key(
    perturbed: ad.AnnData,
    controls: ad.AnnData,
    requested_key: Optional[str],
) -> str:
    if requested_key is not None:
        ensure_required_columns(perturbed, [requested_key], "small.h5ad")
        ensure_required_columns(controls, [requested_key], "ctrl.h5ad")
        return requested_key

    for key in GENERALIZATION_KEY_ALIASES:
        if key in perturbed.obs.columns and key in controls.obs.columns:
            return key

    raise KeyError(
        "Could not infer a shared cell-type/generalization column. "
        "Pass --generalization-key explicitly."
    )


def infer_holdout_value(
    perturbed: ad.AnnData,
    generalization_key: str,
    requested_value: Optional[str],
) -> List[str]:
    if requested_value is not None:
        requested_values = [value.strip() for value in requested_value.split(",") if value.strip()]
        if not requested_values:
            raise ValueError("Received an empty --holdout-value.")
        return requested_values

    ood_values = (
        perturbed.obs.loc[perturbed.obs["split"] == "ood_test", generalization_key]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    if not ood_values:
        raise ValueError(
            "Could not infer any held-out groups from ood_test cells for "
            f"{generalization_key!r}. Pass --holdout-value explicitly."
        )
    return sorted(ood_values)


def cast_expression(adata: ad.AnnData) -> ad.AnnData:
    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


def cast_expression_for_storage(adata: ad.AnnData) -> ad.AnnData:
    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


def align_genes(perturbed: ad.AnnData, controls: ad.AnnData):
    if perturbed.var_names.equals(controls.var_names):
        return perturbed, controls
    shared = perturbed.var_names.intersection(controls.var_names)
    if len(shared) == 0:
        raise ValueError("No shared genes between perturbed and control h5ad files.")
    perturbed_aligned = perturbed if perturbed.var_names.equals(shared) else perturbed[:, shared]
    controls_aligned = controls if controls.var_names.equals(shared) else controls[:, shared]
    return perturbed_aligned, controls_aligned


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _dense_rows_from_x(X, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        n_vars = int(getattr(X, "shape", (0, 0))[1])
        return np.empty((0, n_vars), dtype=np.float32)

    unique_rows, inverse = np.unique(rows, return_inverse=True)
    try:
        block = X[unique_rows]
    except Exception:
        loaded = [_get_dense_row(X, int(row)) for row in unique_rows]
        block = np.vstack(loaded) if loaded else np.empty((0, X.shape[1]), dtype=np.float32)
    if sparse.issparse(block):
        block = block.toarray()
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 1:
        block = block.reshape(1, -1)
    return block[inverse]


class LazyTrainingX:
    def __init__(
        self,
        perturbed: ad.AnnData,
        controls: ad.AnnData,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ) -> None:
        self.perturbed = perturbed
        self.controls = controls
        self.perturbed_indices = np.asarray(perturbed_indices, dtype=np.int64)
        self.control_indices = np.asarray(control_indices, dtype=np.int64)
        self.n_perturbed = int(self.perturbed_indices.size)
        self.shape = (
            self.n_perturbed + int(self.control_indices.size),
            int(perturbed.n_vars),
        )

    def _source_row(self, row: int):
        row = int(row)
        if row < self.n_perturbed:
            return self.perturbed.X[int(self.perturbed_indices[row])]
        return self.controls.X[int(self.control_indices[row - self.n_perturbed])]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
        else:
            rows, cols = key, slice(None)

        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(self.shape[0]), dtype=np.int64)

        if np.isscalar(rows):
            row = self._source_row(int(rows))
            if sparse.issparse(row):
                return row[:, cols]
            return np.asarray(row).ravel()[cols]

        row_indices = np.asarray(rows, dtype=np.int64)
        result = np.empty((row_indices.size, self.shape[1]), dtype=np.float32)

        perturbed_mask = row_indices < self.n_perturbed
        if np.any(perturbed_mask):
            source_rows = self.perturbed_indices[row_indices[perturbed_mask]]
            result[perturbed_mask] = _dense_rows_from_x(self.perturbed.X, source_rows)
        if np.any(~perturbed_mask):
            source_rows = self.control_indices[row_indices[~perturbed_mask] - self.n_perturbed]
            result[~perturbed_mask] = _dense_rows_from_x(self.controls.X, source_rows)
        return result[:, cols].astype(np.float32, copy=False)


class LazyTrainingAnnData:
    def __init__(
        self,
        perturbed: ad.AnnData,
        controls: ad.AnnData,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
        obs: pd.DataFrame,
    ) -> None:
        self.obs = obs
        self.var = perturbed.var
        self.var_names = perturbed.var_names
        self.n_obs = int(obs.shape[0])
        self.n_vars = int(perturbed.n_vars)
        self.X = LazyTrainingX(perturbed, controls, perturbed_indices, control_indices)


class LazyIndexedX:
    def __init__(self, source: ad.AnnData, indices: np.ndarray) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (int(self.indices.size), int(source.n_vars))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
        else:
            rows, cols = key, slice(None)

        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(self.shape[0]), dtype=np.int64)

        if np.isscalar(rows):
            row = self.source.X[int(self.indices[int(rows)])]
            if sparse.issparse(row):
                return row[:, cols]
            return np.asarray(row).ravel()[cols]

        row_indices = np.asarray(rows, dtype=np.int64)
        source_rows = self.indices[row_indices]
        return _dense_rows_from_x(self.source.X, source_rows)[:, cols].astype(np.float32, copy=False)


class LazyIndexedAnnData:
    def __init__(self, source: ad.AnnData, indices: np.ndarray) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.obs = source.obs.iloc[self.indices].copy()
        self.var = source.var
        self.var_names = source.var_names
        self.n_obs = int(self.indices.size)
        self.n_vars = int(source.n_vars)
        self.X = LazyIndexedX(source, self.indices)

    @property
    def obs_names(self):
        return self.obs.index

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            if cols != slice(None):
                raise NotImplementedError("LazyIndexedAnnData only supports row subsetting.")
        else:
            rows = key

        if isinstance(rows, slice):
            local_indices = np.arange(*rows.indices(self.n_obs), dtype=np.int64)
        elif np.isscalar(rows):
            local_indices = np.array([int(rows)], dtype=np.int64)
        else:
            if isinstance(rows, (pd.Index, pd.Series)):
                rows = rows.to_numpy()
            rows = np.asarray(rows)
            if rows.dtype == bool:
                local_indices = np.flatnonzero(rows)
            elif np.issubdtype(rows.dtype, np.integer):
                local_indices = rows.astype(np.int64, copy=False)
            else:
                local_indices = self.obs.index.get_indexer(rows.astype(str))
                if np.any(local_indices < 0):
                    raise KeyError("One or more observation names were not found.")
        return self.take(local_indices)

    def take(self, local_indices: np.ndarray) -> "LazyIndexedAnnData":
        return LazyIndexedAnnData(self.source, self.indices[np.asarray(local_indices, dtype=np.int64)])


def build_training_anndata(
    perturbed: ad.AnnData,
    controls: ad.AnnData,
    generalization_key: str,
    holdout_values: Sequence[str],
):
    train_mask = perturbed.obs["split"].astype(str).to_numpy() == "train"
    train_perturbed_idx = np.flatnonzero(train_mask)
    holdout_set = {str(value) for value in holdout_values}
    control_mask = ~controls.obs[generalization_key].astype(str).isin(holdout_set).to_numpy()
    train_control_idx = np.flatnonzero(control_mask)

    train_perturbed_obs = perturbed.obs.iloc[train_perturbed_idx].copy()
    train_controls_obs = controls.obs.iloc[train_control_idx].copy()
    train_controls_obs[PERTURBATION_KEY] = CONTROL_LABEL
    train_perturbed_obs[SCGPT_LABEL_KEY] = train_perturbed_obs[generalization_key].astype(str)
    train_controls_obs[SCGPT_LABEL_KEY] = train_controls_obs[generalization_key].astype(str)
    obs = pd.concat([train_perturbed_obs, train_controls_obs], axis=0, copy=False)
    obs[PERTURBATION_KEY] = obs[PERTURBATION_KEY].astype("category")
    obs[SCGPT_LABEL_KEY] = obs[SCGPT_LABEL_KEY].astype("category")

    return LazyTrainingAnnData(
        perturbed=perturbed,
        controls=controls,
        perturbed_indices=train_perturbed_idx,
        control_indices=train_control_idx,
        obs=obs,
    )


def get_gene_names(adata: ad.AnnData) -> List[str]:
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def make_feature_matrix(x_values: np.ndarray, pert_gene_idx: int) -> torch.Tensor:
    gene_values = torch.as_tensor(x_values, dtype=torch.float32)
    pert_flags = torch.zeros_like(gene_values)
    pert_flags[pert_gene_idx] = 1.0
    return torch.stack((gene_values, pert_flags), dim=1)


class PerturbationPairDataset(Dataset):
    def __init__(
        self,
        train_data: ad.AnnData,
        generalization_key: str,
        gene_to_idx: Dict[str, int],
        seed: int,
    ) -> None:
        perturbations = train_data.obs[PERTURBATION_KEY].astype(str).to_numpy()
        valid_perturbations = np.isin(perturbations, list(gene_to_idx.keys()))
        self.perturbed_indices = np.flatnonzero(
            (perturbations != CONTROL_LABEL) & valid_perturbations
        )
        self.control_indices = np.flatnonzero(perturbations == CONTROL_LABEL)
        if self.perturbed_indices.size == 0 or self.control_indices.size == 0:
            raise ValueError("scGPT training requires perturbed and control cells.")

        self.train_data = train_data
        self.perturbed_labels = perturbations[self.perturbed_indices]
        self.perturbed_groups = (
            train_data.obs.iloc[self.perturbed_indices][generalization_key].astype(str).to_numpy()
        )
        self.control_by_group: Dict[str, np.ndarray] = {}
        control_obs = train_data.obs.iloc[self.control_indices]
        for group, obs_names in control_obs.groupby(generalization_key, observed=True).groups.items():
            local_indices = control_obs.index.get_indexer(obs_names)
            self.control_by_group[str(group)] = self.control_indices[local_indices]
        self.all_control_indices = self.control_indices
        self.generalization_key = generalization_key
        self.gene_to_idx = gene_to_idx
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.perturbed_indices.size)

    def __getitem__(self, idx: int) -> Data:
        return self._make_item(int(idx))

    def __getitems__(self, indices: Sequence[int]) -> List[Data]:
        indices = [int(idx) for idx in indices]
        if not indices:
            return []

        perturbed_indices = self.perturbed_indices[indices]
        pert_names = self.perturbed_labels[indices].astype(str)
        groups = self.perturbed_groups[indices].astype(str)
        control_indices = np.empty(len(indices), dtype=np.int64)
        pert_gene_indices = np.empty(len(indices), dtype=np.int64)

        for i, (pert_name, group) in enumerate(zip(pert_names, groups)):
            control_pool = self.control_by_group.get(group, self.all_control_indices)
            control_indices[i] = int(self.rng.choice(control_pool))
            pert_gene_indices[i] = self.gene_to_idx[str(pert_name)]

        x_values = self.train_data.X[control_indices]
        y_values = self.train_data.X[perturbed_indices]

        items: List[Data] = []
        for i, pert_name in enumerate(pert_names):
            feature_mat = make_feature_matrix(x_values[i], int(pert_gene_indices[i]))
            items.append(
                Data(
                    x=feature_mat,
                    y=torch.as_tensor(y_values[i].reshape(1, -1), dtype=torch.float32),
                    pert=[str(pert_name)],
                )
            )
        return items

    def _make_item(self, idx: int) -> Data:
        perturbed_idx = int(self.perturbed_indices[idx])
        pert_name = str(self.perturbed_labels[idx])
        group = str(self.perturbed_groups[idx])
        control_pool = self.control_by_group.get(group, self.all_control_indices)
        control_idx = int(self.rng.choice(control_pool))
        x_values = _get_dense_row(self.train_data.X, control_idx)
        y_values = _get_dense_row(self.train_data.X, perturbed_idx)
        feature_mat = make_feature_matrix(x_values, self.gene_to_idx[pert_name])
        return Data(
            x=feature_mat,
            y=torch.as_tensor(y_values.reshape(1, -1), dtype=torch.float32),
            pert=[pert_name],
        )


class RandomStepBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        steps_per_epoch: Optional[int],
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = steps_per_epoch
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        if self.dataset_size <= 0:
            raise ValueError("Cannot sample an empty training dataset.")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.steps_per_epoch is None or self.steps_per_epoch <= 0:
            indices = rng.permutation(self.dataset_size)
            usable = (indices.size // self.batch_size) * self.batch_size
            indices = indices[:usable]
        else:
            n_items = int(self.steps_per_epoch) * self.batch_size * self.world_size
            indices = rng.integers(0, self.dataset_size, size=n_items, dtype=np.int64)

        if indices.size == 0:
            return

        batches = indices.reshape(-1, self.batch_size).tolist()
        for batch_index in range(self.rank, len(batches), self.world_size):
            yield batches[batch_index]

    def __len__(self) -> int:
        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
            return int(self.steps_per_epoch)
        total = self.dataset_size // self.batch_size
        return (total + self.world_size - 1 - self.rank) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def collate_graphs(items: Sequence[Data]) -> Batch:
    return Batch.from_data_list(list(items))


class ScGPTForwardModel:
    def __init__(self, train_data: ad.AnnData, args: argparse.Namespace) -> None:
        self.args = args
        self.gene_names = get_gene_names(train_data)
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_names)}
        self.distributed_context: DistributedContext = getattr(
            args,
            "distributed_context",
            DistributedContext(False, 0, 0, 1),
        )
        if self.distributed_context.enabled:
            self.device = torch.device("cuda", self.distributed_context.local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(args.amp and self.device.type == "cuda")
        self.max_seq_len = int(args.max_seq_len)
        self.vocab = self._build_vocab(args.load_model)
        self.gene_ids = np.array(
            [
                self.vocab[gene] if gene in self.vocab else self.vocab[SCGPT_PAD_TOKEN]
                for gene in self.gene_names
            ],
            dtype=np.int64,
        )
        self.model = TransformerGenerator(
            len(self.vocab),
            args.embsize,
            args.nhead,
            args.d_hid,
            args.nlayers,
            nlayers_cls=args.n_layers_cls,
            n_cls=1,
            vocab=self.vocab,
            dropout=args.dropout,
            pad_token=SCGPT_PAD_TOKEN,
            pad_value=SCGPT_PAD_VALUE,
            pert_pad_id=SCGPT_PERT_PAD_ID,
            use_fast_transformer=args.use_fast_transformer,
        )
        self._load_pretrained(args.load_model)
        self.model.to(self.device)
        if self.distributed_context.enabled:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.distributed_context.local_rank],
                output_device=self.distributed_context.local_rank,
                find_unused_parameters=True,
            )

    def _core_model(self) -> nn.Module:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def _build_vocab(self, load_model: Optional[Path]) -> GeneVocab:
        if load_model is not None and (load_model / "vocab.json").exists():
            vocab = GeneVocab.from_file(load_model / "vocab.json")
        else:
            vocab = GeneVocab(self.gene_names)
        for token in SCGPT_SPECIAL_TOKENS:
            if token not in vocab:
                vocab.append_token(token)
        return vocab

    def _load_pretrained(self, load_model: Optional[Path]) -> None:
        if load_model is None:
            return
        model_file = load_model / "best_model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Could not find scGPT checkpoint at {model_file}")
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_file, map_location="cpu")
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict
            and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def train_forward(
        self,
        train_data: ad.AnnData,
        generalization_key: str,
        args: argparse.Namespace,
    ) -> None:
        dataset = PerturbationPairDataset(
            train_data=train_data,
            generalization_key=generalization_key,
            gene_to_idx=self.gene_to_idx,
            seed=args.seed,
        )
        train_sampler = RandomStepBatchSampler(
            dataset_size=len(dataset),
            batch_size=args.batch_size,
            steps_per_epoch=args.train_steps_per_epoch,
            seed=args.seed,
            rank=self.distributed_context.rank,
            world_size=self.distributed_context.world_size,
        )
        num_workers = max(0, int(args.train_num_workers))
        loader = DataLoader(
            dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_graphs,
            pin_memory=self.device.type == "cuda",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        best_loss = math.inf
        stale_epochs = 0
        for epoch in range(1, args.max_epochs + 1):
            stop_now = deadline_reached(args)
            if self.distributed_context.enabled:
                stop_tensor = torch.tensor(
                    [1 if stop_now else 0],
                    dtype=torch.int64,
                    device=self.device,
                )
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                stop_now = bool(stop_tensor.item())
            if stop_now:
                if self.distributed_context.is_main:
                    print(deadline_message(args, f"epoch {epoch}"), flush=True)
                break
            train_sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0
            total_cells = 0
            for batch_data in loader:
                stop_now = deadline_reached(args)
                if self.distributed_context.enabled:
                    stop_tensor = torch.tensor(
                        [1 if stop_now else 0],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                    stop_now = bool(stop_tensor.item())
                if stop_now:
                    if self.distributed_context.is_main:
                        print(deadline_message(args, f"epoch {epoch} batch"), flush=True)
                    return
                batch_data = batch_data.to(self.device)
                batch_size = len(batch_data.y)
                x = batch_data.x
                ori_gene_values = x[:, 0].view(batch_size, -1)
                pert_flags = x[:, 1].long().view(batch_size, -1)
                target_gene_values = batch_data.y.view(batch_size, -1)
                input_gene_ids = self._input_gene_ids(ori_gene_values)
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]
                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
                src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=self.device)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    output = self.model(
                        mapped_input_gene_ids,
                        input_values,
                        input_pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                    )["mlm_output"]
                    loss = masked_mse_loss(
                        output,
                        target_values,
                        torch.ones_like(input_values, dtype=torch.bool),
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item()) * batch_size
                total_cells += batch_size
            loss_and_count = torch.tensor(
                [total_loss, float(total_cells)],
                dtype=torch.float64,
                device=self.device,
            )
            if self.distributed_context.enabled:
                dist.all_reduce(loss_and_count, op=dist.ReduceOp.SUM)
            epoch_loss = float(loss_and_count[0].item() / max(loss_and_count[1].item(), 1.0))
            if self.distributed_context.is_main:
                print(f"scGPT epoch {epoch}: train_mse={epoch_loss:.6f}", flush=True)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= args.early_stopping_patience:
                    if self.distributed_context.is_main:
                        print(f"Early stopping scGPT at epoch {epoch}.", flush=True)
                    break

    def _input_gene_ids(self, values: torch.Tensor) -> torch.Tensor:
        if SCGPT_INCLUDE_ZERO_GENE == "all":
            input_gene_ids = torch.arange(values.size(1), device=self.device, dtype=torch.long)
        else:
            input_gene_ids = values.nonzero()[:, 1].flatten().unique().sort()[0]
        if len(input_gene_ids) > self.max_seq_len:
            input_gene_ids = input_gene_ids[
                torch.randperm(len(input_gene_ids), device=self.device)[: self.max_seq_len]
            ]
        return input_gene_ids

    def predict_adata(self, controls: ad.AnnData, perturbation: str, batch_size: int) -> np.ndarray:
        if perturbation not in self.gene_to_idx:
            print(
                f"Warning: perturbation {perturbation!r} is not present in the scGPT gene list; "
                "returning zero predictions.",
                flush=True,
            )
            return np.zeros((controls.n_obs, controls.n_vars), dtype=np.float32)
        model = self._core_model()
        model.eval()
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, controls.n_obs, batch_size):
                if deadline_reached(self.args):
                    raise DeadlineExceeded(deadline_message(self.args, "prediction batch"))
                stop = min(start + batch_size, controls.n_obs)
                pert_gene_idx = self.gene_to_idx[perturbation]
                x_values = np.vstack([
                    _get_dense_row(controls.X, i)
                    for i in range(start, stop)
                ]).astype(np.float32, copy=False)
                gene_values = torch.as_tensor(x_values, dtype=torch.float32)
                pert_flags = torch.zeros_like(gene_values)
                pert_flags[:, pert_gene_idx] = 1.0
                batch = Data(
                    x=torch.stack((gene_values, pert_flags), dim=2).reshape(-1, 2),
                    pert=[perturbation] * (stop - start),
                ).to(self.device)
                pred = model.pred_perturb(
                    batch,
                    include_zero_gene=SCGPT_INCLUDE_ZERO_GENE,
                    gene_ids=self.gene_ids,
                    amp=self.amp,
                )
                outputs.append(pred.detach().cpu().numpy().astype(np.float32))
                del x_values, gene_values, pert_flags, batch, pred
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
        return np.concatenate(outputs, axis=0)

    def _make_prediction_graph(self, x_values: np.ndarray, perturbation: str) -> Data:
        feature_mat = make_feature_matrix(x_values, self.gene_to_idx[perturbation])
        return Data(x=feature_mat, pert=[perturbation])


def fit_scgpt(
    train_data: ad.AnnData,
    args: argparse.Namespace,
    generalization_key: str,
) -> ScGPTForwardModel:
    model = ScGPTForwardModel(train_data, args)  # builds vocab + loads --load-model (scGPT_human)
    cache_dir = getattr(args, "save_model", None)
    cached_ckpt = (cache_dir / "best_model.pt") if cache_dir is not None else None
    if cached_ckpt is not None and cached_ckpt.exists():
        # Cache hit: load the fine-tuned weights over the pretrained init and skip training.
        print(f"Loading cached fine-tuned scGPT from {cached_ckpt}; skipping training.", flush=True)
        model._load_pretrained(cache_dir)
    else:
        model.train_forward(train_data, generalization_key, args)
        if cache_dir is not None and model.distributed_context.is_main:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model._core_model().state_dict(), cache_dir / "best_model.pt")
            print(f"Saved fine-tuned scGPT to {cache_dir / 'best_model.pt'}", flush=True)
    return model


def get_candidate_perturbations(train_data: ad.AnnData, min_cells: int) -> List[str]:
    counts = train_data.obs[PERTURBATION_KEY].value_counts()
    gene_set = set(get_gene_names(train_data))
    keep = counts[
        (counts.index != CONTROL_LABEL)
        & (counts >= min_cells)
        & counts.index.astype(str).isin(gene_set)
    ]
    if keep.empty:
        raise ValueError(
            "No perturbations satisfy the training cell threshold and scGPT gene-list match. "
            "Lower --min-train-cells-per-perturbation or check perturbation labels."
        )
    return sorted(map(str, keep.index.tolist()))


def get_group_controls(
    controls: ad.AnnData,
    generalization_key: str,
    group_value: str,
    max_cells: Optional[int],
    seed: int,
) -> ad.AnnData:
    mask = controls.obs[generalization_key].astype(str).to_numpy() == str(group_value)
    group_indices = np.flatnonzero(mask)
    if group_indices.size == 0:
        raise ValueError(
            f"No control cells found for {generalization_key}={group_value!r}."
        )
    if max_cells is not None and max_cells < group_indices.size:
        rng = np.random.default_rng(seed)
        group_indices = np.sort(rng.choice(group_indices, size=max_cells, replace=False))
    return LazyIndexedAnnData(controls, group_indices)


def sample_controls_for_prediction(
    control_subset: ad.AnnData,
    n_cells: int,
    seed: int,
) -> ad.AnnData:
    if n_cells <= 0:
        return control_subset[:0]
    rng = np.random.default_rng(seed)
    replace = control_subset.n_obs < n_cells
    sampled = rng.choice(control_subset.n_obs, size=n_cells, replace=replace)
    if hasattr(control_subset, "take"):
        return control_subset.take(sampled)
    return control_subset[np.asarray(sampled, dtype=np.int64)]


def get_inference_batch_size(args: argparse.Namespace) -> int:
    return max(1, min(INFERENCE_BATCH_SIZE, int(args.batch_size)))


def safe_mean(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0), dtype=np.float32).ravel()
    return np.asarray(X, dtype=np.float32).mean(axis=0)


def build_train_centroids(
    train_data,
    candidate_perts,
    chunk_size=512
):
    perturbations = (
        train_data.obs[PERTURBATION_KEY]
        .astype(str)
        .values
    )

    candidate_set = {str(pert) for pert in candidate_perts}
    running_sums = {
        str(pert): np.zeros(train_data.n_vars, dtype=np.float64)
        for pert in candidate_perts
    }
    counts = {str(pert): 0 for pert in candidate_perts}

    candidate_rows = np.flatnonzero(np.isin(perturbations, list(candidate_set)))
    for start in range(0, len(candidate_rows), chunk_size):
        rows = candidate_rows[start:start + chunk_size]
        values = train_data.X[rows]
        labels = perturbations[rows].astype(str)
        for pert in np.unique(labels):
            local_mask = labels == pert
            running_sums[str(pert)] += values[local_mask].sum(axis=0, dtype=np.float64)
            counts[str(pert)] += int(local_mask.sum())
        gc.collect()

    return {
        str(pert): (running_sums[str(pert)] / counts[str(pert)]).astype(np.float32)
        for pert in candidate_perts
    }


def cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query, axis=1, keepdims=True)
    c_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    q_norm[q_norm == 0.0] = 1.0
    c_norm[c_norm == 0.0] = 1.0
    return (query / q_norm) @ (candidates / c_norm).T


def euclidean_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q_sq = np.sum(query * query, axis=1, keepdims=True)
    c_sq = np.sum(candidates * candidates, axis=1)[None, :]
    dist_sq = np.maximum(q_sq + c_sq - 2.0 * (query @ candidates.T), 0.0)
    return -np.sqrt(dist_sq)


def rank_single_row(
    query_vec: np.ndarray,
    score_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a pre-computed score vector for one query cell, return sorted
    indices and scores.  score_vec is shape (n_candidates,)."""
    ranked_indices = np.argsort(-score_vec).astype(np.int32)
    ranked_scores = score_vec[ranked_indices]
    return ranked_indices, ranked_scores


def first_rank(ranked_indices: np.ndarray, truth_index: int) -> int:
    positions = np.flatnonzero(ranked_indices == truth_index)
    if positions.size == 0:
        return math.inf
    return int(positions[0]) + 1


def serialize_ranked_labels(candidate_labels: np.ndarray, ranked_indices: np.ndarray) -> str:
    return "|".join(map(str, candidate_labels[ranked_indices].tolist()))


def serialize_ranked_scores(ranked_scores: np.ndarray) -> str:
    return "|".join(f"{float(score):.8f}" for score in ranked_scores.tolist())


def build_ranking_rows(
    *,
    cell_id: str,
    split_name: str,
    group_value: str,
    true_perturbation: str,
    candidate_labels: np.ndarray,
    ranked_indices: np.ndarray,
    ranked_scores: np.ndarray,
    query_kind: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for position, (idx, score) in enumerate(zip(ranked_indices.tolist(), ranked_scores.tolist()), start=1):
        rows.append(
            {
                "cell_id": cell_id,
                "split": split_name,
                "generalization_group": group_value,
                "true_perturbation": true_perturbation,
                "query_kind": query_kind,
                "rank_position": position,
                "candidate_perturbation": str(candidate_labels[idx]),
                "score": float(score),
            }
        )
    return rows


def _get_dense_row(X, i: int) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X[i].toarray(), dtype=np.float32).ravel()
    row = X[i]
    if sparse.issparse(row):
        return np.asarray(row.toarray(), dtype=np.float32).ravel()
    return np.asarray(row, dtype=np.float32).ravel()


def _to_dense_mat(X) -> np.ndarray:
    """Convert an AnnData .X slice to float32 without sparse-wide temporaries."""
    if sparse.issparse(X):
        return np.vstack([_get_dense_row(X, i) for i in range(X.shape[0])]).astype(np.float32)
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(1, -1).astype(np.float32, copy=False)
    return np.vstack([np.asarray(arr[i], dtype=np.float32).ravel() for i in range(arr.shape[0])])

class LogitsH5Writer:
    """Incrementally write the per-cell x per-candidate score matrix to
    ``logits.h5`` in the benchmark's shared schema (matches cellot/scouter):

        /logits     (n_cells, n_perts) float32   — score[i, p]
        /idx        (n_cells,)         int64      — 0-based row in the SOURCE adata
        /obs_names  (n_cells,)         str        — test cell barcodes
        /perts      (n_perts,)         str        — candidate perturbation labels
        /columns    (n_perts,)         str        — alias of /perts

    `idx` and `columns` are what collect_results.py actually reads: it looks for
    class names under H5_COLUMN_KEYS (which does NOT include "perts") and for a
    row index under "idx". Without them collect falls back to POSITIONAL row
    order -- and this writer appends in dataset-group order, not split-file
    order, so every label would silently attach to the wrong cell.

    Rows are appended block-by-block (same order as logits.csv) so the full
    matrix never has to live in memory — important for the `full` split where
    n_cells reaches ~1e6. Consumed by src/collect_results.py for cross-method
    scoring (accuracy / recall@k / macro-F1 / MRR).
    """

    def __init__(self, path: Path, perts: Sequence[str],
                 source_obs_names: Optional[Sequence[str]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.n_perts = len(perts)
        str_dt = h5py.string_dtype()
        self._f = h5py.File(path, "w")
        cols = np.asarray([str(p) for p in perts], dtype=object)
        self._f.create_dataset("perts", data=cols, dtype=str_dt)
        self._f.create_dataset("columns", data=cols, dtype=str_dt)
        # barcode -> source row, so /idx can be written as blocks arrive
        self._pos = ({str(n): i for i, n in enumerate(source_obs_names)}
                     if source_obs_names is not None else None)
        self._idx = self._f.create_dataset(
            "idx", shape=(0,), maxshape=(None,), dtype="int64",
            compression="lzf", chunks=True,
        ) if self._pos is not None else None
        self._logits = self._f.create_dataset(
            "logits",
            shape=(0, self.n_perts),
            maxshape=(None, self.n_perts),
            dtype="float32",
            compression="lzf",
            chunks=True,
        )
        self._obs = self._f.create_dataset(
            "obs_names",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt,
            compression="lzf",
            chunks=True,
        )
        self._n = 0

    def append(self, obs_names: Sequence[str], scores: np.ndarray) -> None:
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] != self.n_perts:
            raise ValueError(
                f"scores block shape {scores.shape} incompatible with n_perts={self.n_perts}"
            )
        m = scores.shape[0]
        if m == 0:
            return
        new_n = self._n + m
        self._logits.resize(new_n, axis=0)
        self._logits[self._n:new_n, :] = scores
        self._obs.resize(new_n, axis=0)
        names = [str(x) for x in obs_names]
        self._obs[self._n:new_n] = np.asarray(names, dtype=object)
        if self._idx is not None:
            rows = np.fromiter((self._pos.get(x, -1) for x in names),
                               dtype=np.int64, count=m)
            if (rows < 0).any():
                raise KeyError(
                    f"{int((rows < 0).sum())} eval barcodes not found in the source "
                    "adata; /idx would be wrong -- refusing to write a misaligned file."
                )
            self._idx.resize(new_n, axis=0)
            self._idx[self._n:new_n] = rows
        self._n = new_n

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def evaluate_split_forward(
    model,
    perturbed,
    controls,
    split_name,
    generalization_key,
    candidate_perts,
    train_centroids,
    args,
    output_csv,
    original_split_h5ad,
    predicted_split_h5ad,
    rankings_all_csv,
    rankings_top10_csv,
    rankings_top100_csv,
    logits_csv,
    labels_csv,
):
    split_mask = perturbed.obs["split"].astype(str).to_numpy() == str(split_name)
    split_data = LazyIndexedAnnData(perturbed, np.flatnonzero(split_mask))

    if split_data.n_obs == 0:
        raise ValueError(f"No evaluation cells found for split {split_name!r}.")

    candidate_labels = np.array(candidate_perts)
    label_to_index = {str(label): idx for idx, label in enumerate(candidate_labels.tolist())}
    original_group_paths: List[Path] = []
    predicted_group_paths: List[Path] = []

    original_ranks = []
    predicted_ranks = []
    original_recall_at_10_sum = 0.0
    original_recall_at_100_sum = 0.0
    predicted_recall_at_10_sum = 0.0
    predicted_recall_at_100_sum = 0.0
    n_cells = 0

    logits_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_logits_csv:
        reset_output_file(logits_csv)
    reset_output_file(labels_csv)

    logits_header_written = False
    labels_header_written = False
    # label_rows = []

    # Benchmark scoring artifact: stream the per-cell x per-candidate score
    # matrix to logits.h5 (cellot/scouter schema) alongside the legacy CSV.
    logits_h5_writer = LogitsH5Writer(logits_csv.parent / "logits.h5", candidate_perts,
                                      source_obs_names=perturbed.obs_names)

    inference_batch_size = get_inference_batch_size(args)
    block_size = inference_batch_size

    for group_index, (group_value, obs_names) in enumerate(
        split_data.obs.groupby(generalization_key, observed=True).groups.items()
    ):
        if deadline_reached(args):
            logits_h5_writer.close()
            raise DeadlineExceeded(deadline_message(args, f"{split_name} group {group_index}"))

        group_value = str(group_value)
        subset = split_data[obs_names]
        truths = subset.obs[PERTURBATION_KEY].astype(str).to_numpy()
        group_obs_index = pd.Index(subset.obs_names)
        predicted_group_rows = [] if not args.no_save_h5ad else None

        control_subset = get_group_controls(
            controls=controls,
            generalization_key=generalization_key,
            group_value=group_value,
            max_cells=args.max_control_cells_per_group,
            seed=args.seed + group_index,
        )

        # --classifier-readout: one predicted profile per CANDIDATE for this dataset,
        # averaged over K controls. Built once per group; reused for every cell.
        pred_profiles = None
        if args.classifier_readout:
            k = min(int(args.pred_control_cells), int(control_subset.n_obs))
            ctrl_k = sample_controls_for_prediction(
                control_subset=control_subset, n_cells=k, seed=args.seed + group_index,
            )
            pred_profiles = np.empty((len(candidate_perts), subset.n_vars), dtype=np.float32)
            for j, pert in enumerate(candidate_perts):
                mp = model.predict_adata(ctrl_k, str(pert), batch_size=inference_batch_size)
                pred_profiles[j] = np.asarray(mp, dtype=np.float32).mean(axis=0)
                del mp
            del ctrl_k
            gc.collect()
            main_print(args, f"[{split_name}/{group_value}] built {len(candidate_perts)} "
                              f"candidate profiles from {k} control cells")

        for start_i in range(0, subset.n_obs, block_size):
            stop_i = min(start_i + block_size, subset.n_obs)
            block = subset[start_i:stop_i]
            orig_block = np.vstack([
                _get_dense_row(block.X, i)
                for i in range(block.n_obs)
            ]).astype(np.float32, copy=False)

            pred_block = np.empty((block.n_obs, block.n_vars), dtype=np.float32)
            block_obs_index = pd.Index(block.obs_names)

            for pert_offset, (pert_value, obs_names2) in enumerate(
                [] if args.classifier_readout else block.obs.groupby(
                    PERTURBATION_KEY,
                    observed=True
                ).groups.items()
            ):
                row_positions = block_obs_index.get_indexer(obs_names2)
                if np.any(row_positions < 0):
                    raise RuntimeError("Could not align prediction rows to evaluation block.")

                control_block = sample_controls_for_prediction(
                    control_subset=control_subset,
                    n_cells=len(row_positions),
                    seed=args.seed + group_index + start_i + pert_offset,
                )

                mini_pred = model.predict_adata(
                    control_block,
                    str(pert_value),
                    batch_size=inference_batch_size
                )

                pred_block[
                    row_positions,
                    :
                ] = mini_pred

                del control_block, mini_pred, row_positions
                gc.collect()

            orig_scores_block = np.empty((block.n_obs, len(candidate_perts)), dtype=np.float32)
            pred_scores_block = np.empty((block.n_obs, len(candidate_perts)), dtype=np.float32)

            # Columns that participate in the metric. Everything past --score-n-genes
            # (the synthetic padded block) is dropped from BOTH sides of every
            # comparison below.
            gs = slice(0, args.score_n_genes) if args.score_n_genes else slice(None)

            for j, pert in enumerate(candidate_perts):
                centroid = train_centroids[str(pert)].reshape(1, -1)[:, gs]
                # `orig_*` stays the model-free centroid match (kept as a diagnostic).
                # `pred_*` is what reaches logits.h5: legacy = the label-conditioned
                # prediction vs centroid; classifier = the MEASURED cell vs this
                # candidate's predicted profile.
                ref = pred_profiles[j].reshape(1, -1)[:, gs] if args.classifier_readout else None
                _obs = orig_block[:, gs]
                _alt = (orig_block if args.classifier_readout else pred_block)[:, gs]
                if args.metric == "cosine":
                    orig_scores_block[:, j] = cosine_scores(_obs, centroid).ravel()
                    pred_scores_block[:, j] = cosine_scores(
                        _alt, ref if args.classifier_readout else centroid).ravel()
                else:
                    orig_scores_block[:, j] = euclidean_scores(_obs, centroid).ravel()
                    pred_scores_block[:, j] = euclidean_scores(
                        _alt, ref if args.classifier_readout else centroid).ravel()

                del centroid

            group_positions = group_obs_index.get_indexer(block.obs_names)
            if np.any(group_positions < 0):
                raise RuntimeError("Could not align prediction block to group output rows.")
            if not args.no_save_h5ad:
                predicted_group_rows.append(sparse.csr_matrix(pred_block))

            # write logits incrementally (legacy CSV — optional; h5 is the artifact)
            if not args.no_logits_csv:
                logits_block = pd.DataFrame(
                    pred_scores_block,
                    columns=[str(x) for x in candidate_labels],
                )
                logits_block.insert(0, "cell_id", block.obs_names.astype(str))
                logits_block.to_csv(
                    logits_csv,
                    mode="a",
                    index=False,
                    header=not logits_header_written,
                )
                del logits_block
                logits_header_written = True
            gc.collect()

            # Mirror the same block into logits.h5 (benchmark scoring artifact).
            logits_h5_writer.append(block.obs_names.astype(str), pred_scores_block)
            if args.logits_only:
                # the per-cell records loop below is skipped, so count here instead --
                # otherwise the end-of-split `n_cells == 0` guard aborts a split whose
                # logits.h5 was written correctly, and every array task exits 1.
                n_cells += int(block.n_obs)

            summary_records = []
            ranking_records = []

            for local_i, obs_name in enumerate(
                [] if args.logits_only else block.obs_names
            ):
                global_i = start_i + local_i
                truth = str(truths[global_i])

                orig_ranked_indices, orig_ranked_scores = rank_single_row(
                    None,
                    orig_scores_block[local_i],
                )
                pred_ranked_indices, pred_ranked_scores = rank_single_row(
                    None,
                    pred_scores_block[local_i],
                )

                truth_index = label_to_index.get(truth, -1)
                original_rank = first_rank(orig_ranked_indices, truth_index)
                predicted_rank = first_rank(pred_ranked_indices, truth_index)

                original_ranks.append(original_rank)
                predicted_ranks.append(predicted_rank)

                original_recall_at_10_sum += float(original_rank <= 10)
                original_recall_at_100_sum += float(original_rank <= 100)
                predicted_recall_at_10_sum += float(predicted_rank <= 10)
                predicted_recall_at_100_sum += float(predicted_rank <= 100)
                n_cells += 1

                # label_rows.append({"label": truth})
                pd.DataFrame(
                    [{"label": truth}]
                ).to_csv(
                    labels_csv,
                    mode="a",
                    index=False,
                    header=not labels_header_written
                )

                labels_header_written = True

                summary_records.append({
                    "cell_id": obs_name,
                    "split": split_name,
                    "generalization_key": generalization_key,
                    "generalization_group": group_value,
                    "true_perturbation": truth,
                    "original_rank": original_rank,
                    "predicted_rank": predicted_rank,
                    "original_top_prediction": str(candidate_labels[orig_ranked_indices[0]]),
                    "predicted_top_prediction": str(candidate_labels[pred_ranked_indices[0]]),
                    "original_top_score": float(orig_ranked_scores[0]),
                    "predicted_top_score": float(pred_ranked_scores[0]),
                    "original_recall_at_10": float(original_rank <= 10),
                    "original_recall_at_100": float(original_rank <= 100),
                    "predicted_recall_at_10": float(predicted_rank <= 10),
                    "predicted_recall_at_100": float(predicted_rank <= 100),
                    "original_ranked_candidates": serialize_ranked_labels(candidate_labels, orig_ranked_indices),
                    "predicted_ranked_candidates": serialize_ranked_labels(candidate_labels, pred_ranked_indices),
                    "original_ranked_scores": serialize_ranked_scores(orig_ranked_scores),
                    "predicted_ranked_scores": serialize_ranked_scores(pred_ranked_scores),
                })

                ranking_records.extend(
                    build_ranking_rows(
                        cell_id=obs_name,
                        split_name=split_name,
                        group_value=group_value,
                        true_perturbation=truth,
                        candidate_labels=candidate_labels,
                        ranked_indices=orig_ranked_indices,
                        ranked_scores=orig_ranked_scores,
                        query_kind="original",
                    )
                )

                ranking_records.extend(
                    build_ranking_rows(
                        cell_id=obs_name,
                        split_name=split_name,
                        group_value=group_value,
                        true_perturbation=truth,
                        candidate_labels=candidate_labels,
                        ranked_indices=pred_ranked_indices,
                        ranked_scores=pred_ranked_scores,
                        query_kind="predicted",
                    )
                )

            write_records(output_csv, SUMMARY_FIELDNAMES, summary_records)
            if args.no_rankings_all:
                write_records(
                    rankings_top10_csv,
                    RANKING_FIELDNAMES,
                    [row for row in ranking_records if int(row["rank_position"]) <= 10],
                )
                write_records(
                    rankings_top100_csv,
                    RANKING_FIELDNAMES,
                    [row for row in ranking_records if int(row["rank_position"]) <= 100],
                )
            else:
                write_ranking_tables(
                    rankings=ranking_records,
                    rankings_all_csv=rankings_all_csv,
                    rankings_top10_csv=rankings_top10_csv,
                    rankings_top100_csv=rankings_top100_csv,
                )

            del block, orig_block, pred_block
            del orig_scores_block, pred_scores_block
            del summary_records, ranking_records
            gc.collect()

        original_group_h5ad = build_group_h5ad_path(original_split_h5ad, split_name, group_index)
        predicted_group_h5ad = build_group_h5ad_path(predicted_split_h5ad, split_name, group_index)
        if not args.no_save_h5ad:
            original_group_h5ad.parent.mkdir(parents=True, exist_ok=True)
            predicted_group_h5ad.parent.mkdir(parents=True, exist_ok=True)

            original_group = ad.AnnData(
                X=sparse.vstack([
                    sparse.csr_matrix(_get_dense_row(subset.X, i))
                    for i in range(subset.n_obs)
                ]),
                obs=subset.obs.copy(),
                var=subset.var.copy(),
            )
            predicted_group = ad.AnnData(
                X=sparse.vstack(predicted_group_rows),
                obs=subset.obs.copy(),
                var=subset.var.copy(),
            )
            predicted_group.obs["is_predicted"] = True

            original_group.write_h5ad(original_group_h5ad)
            predicted_group.write_h5ad(predicted_group_h5ad)

            original_group_paths.append(original_group_h5ad)
            predicted_group_paths.append(predicted_group_h5ad)

            del original_group, predicted_group
        del subset, control_subset
        gc.collect()

    # pd.DataFrame(label_rows).to_csv(labels_csv, index=False)

    logits_h5_writer.close()

    if n_cells == 0:
        raise ValueError(f"No cells evaluated for split {split_name!r}.")

    if args.logits_only:
        # summary stats come from the skipped per-cell loop; logits.h5 is the artifact.
        return {"split": split_name, "n_cells": n_cells, "logits_only": True}

    summary = {
        "split": split_name,
        "n_cells": n_cells,
        "original_recall_at_10": original_recall_at_10_sum / n_cells,
        "original_recall_at_100": original_recall_at_100_sum / n_cells,
        "predicted_recall_at_10": predicted_recall_at_10_sum / n_cells,
        "predicted_recall_at_100": predicted_recall_at_100_sum / n_cells,
        "original_median_rank": float(np.median(np.asarray(original_ranks, dtype=np.float32))),
        "predicted_median_rank": float(np.median(np.asarray(predicted_ranks, dtype=np.float32))),
    }

    if not args.no_save_h5ad:
        combine_h5ad_files(original_group_paths, original_split_h5ad)
        combine_h5ad_files(predicted_group_paths, predicted_split_h5ad)
        cleanup_files(original_group_paths)
        cleanup_files(predicted_group_paths)

    del split_data
    gc.collect()

    return summary

def summarize_results(rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
    summary = pd.DataFrame.from_records(rows)
    if summary.empty:
        raise ValueError("No summary rows were produced.")
    return summary.sort_values("split").reset_index(drop=True)


def write_records(
    path: Path,
    fieldnames: Sequence[str],
    records: Sequence[Dict[str, object]],
) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def write_ranking_tables(
    rankings: Sequence[Dict[str, object]],
    rankings_all_csv: Path,
    rankings_top10_csv: Path,
    rankings_top100_csv: Path,
) -> None:
    if not rankings:
        return
    write_records(rankings_all_csv, RANKING_FIELDNAMES, rankings)
    write_records(
        rankings_top10_csv,
        RANKING_FIELDNAMES,
        [row for row in rankings if int(row["rank_position"]) <= 10],
    )
    write_records(
        rankings_top100_csv,
        RANKING_FIELDNAMES,
        [row for row in rankings if int(row["rank_position"]) <= 100],
    )


def reset_output_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def build_split_h5ad_path(base_path: Path, split_name: str) -> Path:
    return base_path.with_name(f"{base_path.stem}.{split_name}{base_path.suffix}")


def build_group_h5ad_path(base_path: Path, split_name: str, group_index: int) -> Path:
    return base_path.with_name(
        f"{base_path.stem}.{split_name}.group_{group_index:04d}{base_path.suffix}"
    )


def build_group_csv_path(base_path: Path, split_name: str, group_index: int) -> Path:
    return base_path.with_name(
        f"{base_path.stem}.{split_name}.group_{group_index:04d}{base_path.suffix}"
    )


def write_logits_file(
    path: Path,
    candidate_labels: Sequence[str],
    obs_names: Sequence[str],
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logits = pd.DataFrame(scores, columns=[str(label) for label in candidate_labels])
    logits.insert(0, "cell_id", [str(obs_name) for obs_name in obs_names])
    logits.to_csv(path, index=False)


def combine_logits_files(
    input_paths: Sequence[Path],
    output_path: Path,
    obs_names: Sequence[str],
) -> List[str]:
    if not input_paths:
        raise ValueError("No logits files were produced.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_obs_names: List[str] = []
    wrote_header = False
    with output_path.open("w", newline="") as out_handle:
        writer = None
        for path in input_paths:
            with path.open("r", newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                if reader.fieldnames is None:
                    continue
                if not wrote_header:
                    writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    wrote_header = True
                if writer is None:
                    raise ValueError(f"Could not initialize logits writer for {output_path}")
                for row in reader:
                    written_obs_names.append(str(row["cell_id"]))
                    writer.writerow(row)
    return written_obs_names


def write_labels_file(path: Path, labels: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": labels.astype(str).to_numpy()}).to_csv(path, index=False)


def combine_h5ad_files(input_paths: Sequence[Path], output_path: Path) -> None:
    if not input_paths:
        raise ValueError("No AnnData files were produced.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        import shutil
        shutil.copy2(input_paths[0], output_path)
        return
    concat_on_disk = getattr(getattr(ad, "experimental", None), "concat_on_disk", None)
    if concat_on_disk is not None:
        concat_on_disk(
            [str(path) for path in input_paths],
            str(output_path),
            join="inner",
            merge="same",
        )
        return
    parts = [ad.read_h5ad(p) for p in input_paths]
    combined = ad.concat(parts, join="inner", merge="same")
    del parts
    gc.collect()
    combined.write_h5ad(output_path)
    del combined
    gc.collect()


def cleanup_files(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def get_eval_groups(
    perturbed: ad.AnnData,
    generalization_key: str,
    eval_splits: Sequence[str],
) -> List[str]:
    mask = perturbed.obs["split"].astype(str).isin(eval_splits)
    groups = perturbed.obs.loc[mask, generalization_key].astype(str).unique().tolist()
    return sorted(groups)


def print_summary(summary: pd.DataFrame, elapsed_seconds: float) -> None:
    print("scGPT forward cell-type generalization benchmark summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nElapsed seconds: {elapsed_seconds:.2f}")


def main() -> None:
    args = parse_args()
    args.distributed_context = setup_distributed()
    set_seed(args.seed)

    try:
        start = time.time()
        args.run_start_time = start

        # --- All ranks: load data and train ---
        if args.perturbation_pool_file is not None and not args.disable_perturbation_pool:
            perturbed_raw = read_subset(
                args.data_path,
                perturbation_pool_file=args.perturbation_pool_file,
                use_perturbation_pool=True,
                n_ctrl=None,
                split_dir=args.split_dir,
                seed=args.seed,
            )
        else:
            perturbed_raw = ad.read_h5ad(args.data_path, backed="r")
        controls = ad.read_h5ad(args.ctrl_path, backed="r")
        # Every cell in ctrl_h5ad IS a control; the harmonised DB keeps each study's own
        # control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET / control), so
        # comparing the label against one value misses 18/22 datasets.
        controls.obs["perturbation"] = "control"
        perturbed_raw.obs_names_make_unique()
        controls.obs_names_make_unique()

        ensure_required_columns(perturbed_raw, [PERTURBATION_KEY], "full.h5ad")
        perturbed_raw, controls = align_genes(perturbed_raw, controls)
        perturbed_raw = assign_splits(perturbed_raw, args.split_dir)

        generalization_key = resolve_generalization_key(perturbed_raw, controls,
                                                        args.generalization_key)
        holdout_values = infer_holdout_value(perturbed_raw, generalization_key,
                                             args.holdout_value)

        train_data = build_training_anndata(perturbed_raw, controls,
                                            generalization_key, holdout_values)
        candidate_perts = get_candidate_perturbations(train_data,
                                                      args.min_train_cells_per_perturbation)
        if args.n_pert_chunks > 1:
            # Evaluate only this chunk's strided slice of the (sorted, deterministic)
            # candidate perts; strided so expensive perts spread evenly across chunks.
            full_candidates = list(candidate_perts)
            candidate_perts = full_candidates[args.pert_chunk_idx::args.n_pert_chunks]
            main_print(args, f"Pert chunk {args.pert_chunk_idx}/{args.n_pert_chunks}: "
                             f"{len(candidate_perts)}/{len(full_candidates)} candidate perts.")
        train_centroids = build_train_centroids(train_data, candidate_perts)

        # Non-main ranks: keep only train_data in memory during training
        if not args.distributed_context.is_main:
            del perturbed_raw, controls
            gc.collect()

        main_print(args, f"Training scGPT on {train_data.n_obs} cells, "
                         f"{train_data.n_vars} genes, "
                         f"{len(candidate_perts)} candidate perturbations "
                         f"across {args.distributed_context.world_size} process(es).")
        main_print(args, f"Generalization key: {generalization_key}")
        if len(holdout_values) == 1:
            main_print(args, f"Held-out group for OOD evaluation: {holdout_values[0]}")
        else:
            main_print(args, "Held-out groups for OOD evaluation: " + ", ".join(map(str, holdout_values)))
        main_print(args, "Forward evaluation mode: predict transcriptomes first, then rank predicted vs original.")

        # Train once — model, candidate_perts, generalization_key, holdout_values
        # are all set here and reused by rank 0 for evaluation below.
        model = fit_scgpt(train_data, args, generalization_key)
        del train_data
        torch.cuda.empty_cache()
        gc.collect()

        if args.train_only:
            main_print(args, "Train-only mode: model cached; skipping evaluation.")
            return

        if args.distributed_context.enabled:
            dist.barrier()

        # Non-main ranks exit here — they never load the full eval data
        if not args.distributed_context.is_main:
            return

        # --- Rank 0 only: evaluation ---
        # Reuse perturbed_raw, controls, model, candidate_perts, generalization_key,
        # and holdout_values from the shared block above — no second training.
        perturbed = perturbed_raw

        # controls remains backed and is materialized per evaluation group.
        reset_output_file(args.output_csv)
        if not args.no_rankings_all:
            reset_output_file(args.rankings_all_csv)
        reset_output_file(args.rankings_top10_csv)
        reset_output_file(args.rankings_top100_csv)

        summary_rows = []
        original_split_paths: List[Path] = []
        predicted_split_paths: List[Path] = []
        stopped_for_deadline = False

        for split_name in args.eval_splits:
            if deadline_reached(args):
                main_print(args, deadline_message(args, f"split {split_name}"))
                stopped_for_deadline = True
                break
            original_split_h5ad = build_split_h5ad_path(args.original_h5ad, split_name)
            predicted_split_h5ad = build_split_h5ad_path(args.predicted_h5ad, split_name)
            split_output_dir = args.output_csv.parent / split_name
            logits_csv = split_output_dir / "logits.csv"
            labels_csv = split_output_dir / "labels.csv"
            if not args.no_save_h5ad:
                reset_output_file(original_split_h5ad)
                reset_output_file(predicted_split_h5ad)
            if not args.no_logits_csv:
                reset_output_file(logits_csv)
            reset_output_file(labels_csv)
            try:
                split_summary = evaluate_split_forward(
                    model=model,
                    perturbed=perturbed,
                    controls=controls,
                    split_name=split_name,
                    generalization_key=generalization_key,
                    candidate_perts=candidate_perts,
                    train_centroids=train_centroids,
                    args=args,
                    output_csv=args.output_csv,
                    original_split_h5ad=original_split_h5ad,
                    predicted_split_h5ad=predicted_split_h5ad,
                    rankings_all_csv=args.rankings_all_csv,
                    rankings_top10_csv=args.rankings_top10_csv,
                    rankings_top100_csv=args.rankings_top100_csv,
                    logits_csv=logits_csv,
                    labels_csv=labels_csv,
                )
            except DeadlineExceeded as exc:
                main_print(args, str(exc))
                stopped_for_deadline = True
                break
            summary_rows.append(split_summary)
            if not args.no_save_h5ad:
                original_split_paths.append(original_split_h5ad)
                predicted_split_paths.append(predicted_split_h5ad)
            gc.collect()

        elapsed = time.time() - start
        if not summary_rows:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "data_path": str(args.data_path),
                "ctrl_path": str(args.ctrl_path),
                "split_dir": str(args.split_dir),
                "generalization_key": generalization_key,
                "held_out_group": holdout_values[0] if len(holdout_values) == 1 else None,
                "held_out_groups": list(holdout_values),
                "metric": args.metric,
                "seed": args.seed,
                "max_epochs": args.max_epochs,
                "batch_size_per_process": args.batch_size,
                "world_size": args.distributed_context.world_size,
                "global_batch_size": args.batch_size * args.distributed_context.world_size,
                "candidate_perturbations": len(candidate_perts),
                "forward_problem": True,
                "runtime_stopped_early": stopped_for_deadline,
                "deadline_hours": args.deadline_hours,
                "deadline_buffer_minutes": args.deadline_buffer_minutes,
                "elapsed_seconds": elapsed,
                "summary": [],
            }
            args.summary_json.write_text(json.dumps(payload, indent=2))
            main_print(args, "No evaluation splits completed before the runtime deadline.")
            return

        summary = summarize_results(summary_rows)
        print_summary(summary, elapsed)

        if not args.no_save_h5ad:
            combine_h5ad_files(original_split_paths, args.original_h5ad)
            combine_h5ad_files(predicted_split_paths, args.predicted_h5ad)

        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_path": str(args.data_path),
            "ctrl_path": str(args.ctrl_path),
            "split_dir": str(args.split_dir),
            "generalization_key": generalization_key,
            "held_out_group": holdout_values[0] if len(holdout_values) == 1 else None,
            "held_out_groups": list(holdout_values),
            "metric": args.metric,
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "batch_size_per_process": args.batch_size,
            "world_size": args.distributed_context.world_size,
            "global_batch_size": args.batch_size * args.distributed_context.world_size,
            "candidate_perturbations": len(candidate_perts),
            "forward_problem": True,
            "runtime_stopped_early": stopped_for_deadline,
            "deadline_hours": args.deadline_hours,
            "deadline_buffer_minutes": args.deadline_buffer_minutes,
            "saved_outputs": {
                "per_cell_summary_csv": str(args.output_csv),
                "original_h5ad": str(args.original_h5ad),
                "predicted_h5ad": str(args.predicted_h5ad),
                "rankings_all_csv": str(args.rankings_all_csv),
                "rankings_top10_csv": str(args.rankings_top10_csv),
                "rankings_top100_csv": str(args.rankings_top100_csv),
                "logits_dir": str(args.output_csv.parent),
            },
            "elapsed_seconds": elapsed,
            "summary": summary.to_dict(orient="records"),
        }
        args.summary_json.write_text(json.dumps(payload, indent=2))
    finally:
        cleanup_distributed(args.distributed_context)


if __name__ == "__main__":
    main()
