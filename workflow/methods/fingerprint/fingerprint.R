# Script to run fingerprint benchmarks

# Import packages
library(argparse)
library(jsonlite)

# Prefix every message() with HH:MM:SS so long runs are easy to triage in the
# SLURM log. Both this script and utils.R (sourced into globalenv) resolve
# `message` lexically, so nothing else needs to change.
.base_message <- base::message
message <- function(..., domain = NULL, appendLF = TRUE) {
  .base_message("[", format(Sys.time(), "%H:%M:%S"), "] ", ...,
                domain = domain, appendLF = appendLF)
}

p <- ArgumentParser()
p$add_argument("--fold_dir", required = TRUE,
               help = "Fold directory containing {id_train,id_val,id_test,ood_test}.txt.gz. The parent directory must contain metadata.json whose `source_path` points to the adata the indices refer to.")
p$add_argument("--ctrl_p", required = TRUE,
               help = "Control adata (not part of fold indices)")
p$add_argument("--context_col", default = "context")
p$add_argument("--dataset_col", default = "dataset")
p$add_argument("--split_key", default = "dataset",
               help = "Meta column used for per-context SCTransform")
p$add_argument("--p_col", default = "perturbation")
p$add_argument("--ctrl_key", default = "control")
p$add_argument("--output_dir", default = "./output")
p$add_argument("--test_ms_score", type = "double", default = 0.0)
p$add_argument("--min_cells", type = "double", default = 0.0)
p$add_argument("--cores", type = "double", default = 1)
p$add_argument("--force", action = "store_true",
               help = "Force refitting the dictionary even if dict.rds is cached in the output directory.")
p$add_argument("--gene_panel", default = "all", choices = c("all", "dataset"),
               help = "Gene space for a --train_dataset model. 'all' (default) keeps the full var axis so every dataset's dictionary shares ONE coordinate system (panels are hard zero-padded); 'dataset' restricts to that dataset's own active panel.")
p$add_argument("--shared_genes", action = "store_true",
               help = "Restrict obj and ctrl to genes with at least one non-zero count in every dataset (per --dataset_col). Recommended when the merged adata is a union of per-dataset panels so structural zeros don't pollute SCTransform's NB fit. Output is written under a 'shared_genes' subdir so runs with and without the filter don't collide.")
p$add_argument("--max_cells_per_pert", type = "integer", default = 500,
               help = "Training-only cap on cells per (perturbation, dataset) cell. Bounds SCTransform + vars.to.regress memory without hurting the dictionary (ChooseFeatures already caps at 500/pert internally, fit() samples 1000 controls for the PCA baseline). Set to 0 to disable. Eval splits are never subsampled.")
p$add_argument("--max_ctrl_per_ds", type = "integer", default = 2000,
               help = "Training-only cap on control cells per dataset. Controls dominate the merged training object (~18k / dataset) but only ~1k are used for the PCA baseline. Set to 0 to disable.")
p$add_argument("--max_train_cells", type = "integer", default = 0,
               help = "Hard cap on total training cells after the per-(pert, dataset) cap. 0 (default) auto-bounds to the largest count whose realized matrix stays under dgCMatrix's 2^31 nnz cap, estimated from a 10k-cell sample. A positive value caps further (or matches the auto-bound if smaller).")
p$add_argument("--deg_batch_size", type = "integer", default = 0,
               help = "Per-batch perturbation count for the DEG step. >0 = process perts in batches of this size and persist each batch's features under <model_dir>/deg_cache/ for crash resumability. 0 (default) = single batch (no caching). Forks per batch is still controlled by --cores; lower it to bound peak fork-copy memory.")
p$add_argument("--mode", default = "full",
               choices = c("full", "prep", "deg_chunk", "concat_deg",
                           "fit_chunk", "concat_dict", "eval_chunk",
                           "concat_eval"),
               help = "Pipeline mode. full (default): single job runs prep + DEGs + dict fit + eval. prep: realize + SCT + (DEG unless --skip_deg) + persist train_obj.rds + features.rds, exit. deg_chunk: load cached train_obj, compute DEGs for this task's slice of perts (--chunk_idx of --n_chunks) serially, save under <model_dir>/deg_chunks/, exit. concat_deg: union the partial DEGs in deg_chunks/ into features.rds, exit. fit_chunk: load cached train_obj + features, fit this task's slice of perts, save a partial dict under <model_dir>/dict_chunks/, exit. concat_dict: cbind dict_chunks/ into dict.rds, exit (used before eval_chunk arrays). eval_chunk: load dict.rds + sce_lazy + ctrl, realize this task's slice of cells from --eval_split (id_val/id_test/ood_test), run SCT + query, save partial logits/labels/meta under <model_dir>/eval_chunks/<split>/, exit. concat_eval: build dict.rds (if missing), then per split either concat eval_chunks/ or run in-process eval; write final logits.h5 + report.csv. Per-chunk SCT is intentionally avoided so every Lambda lives in the same residual space.")
p$add_argument("--skip_deg", action = "store_true",
               help = "In --mode prep, save train_obj.rds and exit BEFORE the DEG step. Use when you plan to compute DEGs as a SLURM array via --mode deg_chunk + --mode concat_deg.")
p$add_argument("--eval_split", default = "",
               help = "Which eval split to process in --mode eval_chunk. One of id_val, id_test, ood_test (whichever the fold has).")
p$add_argument("--eval_splits", default = "",
               help = "Comma-separated subset of {id_val,id_test,ood_test} that --mode concat_eval (and the eval/full paths) should process. Empty (default) = every split present in the fold. Used to skip in-process eval of splits that aren't being scored (e.g. id_val).")
p$add_argument("--chunk_idx", type = "integer", default = 0,
               help = "0-based chunk index for --mode fit_chunk. Typically set to $SLURM_ARRAY_TASK_ID.")
p$add_argument("--n_chunks", type = "integer", default = 1,
               help = "Total number of chunks for --mode fit_chunk. Perts are split into ceil(n_perts / n_chunks) groups; this task fits group --chunk_idx.")
p$add_argument("--train_dataset", default = "",
               help = "Single-dataset training experiment. Restricts both train_idx and eval_idx to cells from this dataset (matching --dataset_col), uses that dataset's own gene panel (from adata.varm$feature_mask_per_dataset) as the gene set, and writes outputs under <model_dir>/per_dataset_panel/<train_dataset>/. Use to validate the method on a single-context regime closer to the paper's setup. Recommended alongside --max_cells_per_pert 0 (single dataset is naturally bounded). The --shared_genes flag is overridden — the dataset's own panel always wins when this is set.")

# Parse CL args
args <- p$parse_args()
print(args)

# Set I/O options
fold_dir      <- args$fold_dir
ctrl_p        <- args$ctrl_p
dataset_col   <- args$dataset_col
context_col   <- args$context_col
split_key     <- args$split_key
output_dir    <- args$output_dir
p_col         <- args$p_col
ctrl_key      <- args$ctrl_key
test_ms_score <- args$test_ms_score
min_cells     <- args$min_cells
shared_genes  <- args$shared_genes
cores         <- args$cores
max_cells_per_pert <- args$max_cells_per_pert
max_ctrl_per_ds    <- args$max_ctrl_per_ds
max_train_cells    <- args$max_train_cells
deg_batch_size     <- args$deg_batch_size
mode               <- args$mode
chunk_idx          <- args$chunk_idx
n_chunks           <- args$n_chunks
skip_deg           <- args$skip_deg
eval_split_arg     <- args$eval_split
train_dataset      <- args$train_dataset

# Load script functions
source("utils.R")

# Resolve source adata path from the splits metadata (one level up from fold_dir)
fold_dir_abs <- normalizePath(fold_dir, mustWork = TRUE)
meta_p       <- file.path(dirname(fold_dir_abs), "metadata.json")
stopifnot(file.exists(meta_p))
split_meta   <- jsonlite::fromJSON(meta_p)
data_p       <- split_meta$source_path
stopifnot(!is.null(data_p), nzchar(data_p), file.exists(data_p))
message("Source adata (from metadata.json): ", data_p)

# Output key: <data_bn>/<fold_name>/<sg_suffix>[/<train_dataset>]/. Each eval
# split writes into its own subdir. The --shared_genes / per-dataset paths
# live in separate branches so they never collide with an unfiltered run on
# the same fold.
# Derive the output key from the LOGICAL fold_dir (as passed), NOT the
# symlink-resolved fold_dir_abs. The splits root exposes `small` as a symlink to
# the real size dir (e.g. `200`); normalizePath() rewrites that segment, so the
# outputs would land under <output_dir>/200/... while Snakemake waits for
# <output_dir>/small/... -> MissingOutputException on every per-dataset job.
# fold_dir_abs is still used for READING (files resolve through the symlink fine).
fold_dir_logical <- sub("/+$", "", fold_dir)
data_bn   <- basename(dirname(fold_dir_logical))
fold_name <- basename(fold_dir_logical)
model_dir <- file.path(output_dir, data_bn, fold_name)
if (nzchar(train_dataset)) {
  sg_suffix <- "per_dataset_panel"
} else if (shared_genes) {
  sg_suffix <- "shared_genes"
} else {
  sg_suffix <- "all_genes"
}
model_dir <- file.path(model_dir, sg_suffix)
if (nzchar(train_dataset)) {
  model_dir <- file.path(model_dir, train_dataset)
}
dict_p      <- file.path(model_dir, "dict.rds")
features_p  <- file.path(model_dir, "features.rds")
train_obj_p <- file.path(model_dir, "train_obj.rds")  # legacy (joint-SCT path)
sct_model_p <- file.path(model_dir, "sct_model.rds")  # option-2: {model, offsets, features}
time_p      <- file.path(model_dir, "time.txt")

message("Saving output to: ", model_dir)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

# Check for cached dictionary — skip training entirely unless --force
has_cached_dict <- file.exists(dict_p) && !args$force
if (has_cached_dict) {
  message("Found cached dictionary, skipping training: ", dict_p)
} else if (file.exists(dict_p)) {
  message("--force set, overwriting cached dictionary: ", dict_p)
}

# Saving start timestamp
start_time <- timestamp()

#### ---- Load fold indices ----
load_idx <- function(path) {
  # Load a gzipped plain-text integer array; convert 0-based (Python) to 1-based (R)
  as.integer(scan(gzfile(path), what = integer(), quiet = TRUE)) + 1L
}

# Evaluation splits — each is a separate file in fold_dir and gets its own
# results subdirectory. Missing splits are skipped with a warning rather than
# treated as an error so partial folds still produce output.
eval_splits <- c("id_val", "id_test", "ood_test")
if (nzchar(args$eval_splits)) {
  requested <- trimws(strsplit(args$eval_splits, ",")[[1]])
  bad <- setdiff(requested, eval_splits)
  if (length(bad) > 0) stop("--eval_splits has unknown split(s): ", paste(bad, collapse = ", "))
  eval_splits <- intersect(eval_splits, requested)
  stopifnot(length(eval_splits) > 0)
}

message("Loading fold indices from ", fold_dir)
eval_idx <- list()
for (s in eval_splits) {
  p <- file.path(fold_dir, paste0(s, ".txt.gz"))
  if (!file.exists(p)) {
    message("  ", s, ": missing (", p, "), skipping")
    next
  }
  eval_idx[[s]] <- load_idx(p)
  message("  ", s, ": ", length(eval_idx[[s]]), " cells")
}
stopifnot(length(eval_idx) > 0)

# Train-side prep gates:
#   full         : run prep + fit + eval, all in one job
#   prep         : run prep, save train_obj.rds + features.rds, exit
#   fit_chunk    : skip prep (load train_obj.rds), fit one chunk, save partial, exit
#   concat_eval  : skip prep + fit, concat partial dicts, run eval
# need_realize        => realize the training Seurat from sce_lazy and run SCT
# need_lazy_sce       => sce_lazy + ctrl + shared_features (for realize OR eval)
# Skipped sub-stages: cached train_obj.rds skips realize+SCT; cached features
# skips DEG; cached dict.rds skips both.
has_cached_train_obj <- file.exists(train_obj_p) && !args$force
has_cached_model     <- file.exists(sct_model_p) && !args$force
has_cached_features  <- file.exists(features_p)  && !args$force
# Option 2: prep FITS one SCT model on a <=nnz-cap subsample (need_realize); the
# deg/fit chunk arrays REALIZE their own perturbations (uncapped) and apply that
# model (need_chunk). Both read sce_lazy; only prep caps its (fit) subsample.
# A cached dict.rds must NOT gate the chunk arrays or prep. Snakemake asks for a
# SPECIFIC output (deg_chunks/chunk_N.rds, sct_model.rds) and only does so when it
# has decided that output is stale -- typically because the upstream DB changed,
# which is exactly when the dict on disk is the thing being invalidated. Treating
# the stale dict as proof the work is done made the script exit 0 without writing,
# giving 390 MissingOutputExceptions. Idempotence is already handled per-output
# (the `file.exists(chunk_p) && !args$force` check below). `full` still short-
# circuits on a cached dict: it trains AND evals in one process, so a valid dict
# genuinely lets it skip the training realize.
need_realize  <- (mode == "full" && !has_cached_dict && !has_cached_model) ||
                 (mode == "prep" && !has_cached_model)
need_chunk    <- mode %in% c("deg_chunk", "fit_chunk")
# Need sce_lazy + ctrl + shared_features for any eval-side realize:
#   full / concat_eval — top-level eval loop (in-process or chunk-concat)
#   eval_chunk         — array task realizing one slice
need_eval     <- mode %in% c("full", "concat_eval", "eval_chunk")
need_lazy_sce <- need_realize || need_eval || need_chunk

# train_idx: prep caps it to the fit subsample (below); chunks use it UNCAPPED to
# locate their perturbations' cells. Load for both.
train_idx <- NULL
if (need_realize || need_chunk) {
  train_idx_p <- file.path(fold_dir, "id_train.txt.gz")
  stopifnot(file.exists(train_idx_p))
  train_idx <- load_idx(train_idx_p)
  message("  id_train: ", length(train_idx), " cells")
}

#### ---- Pre-processing ----
# Read source adata lazily (HDF5-backed). We never realize a single combined
# matrix — train and each eval split are realized into independent Seurat
# objects further down so no one realized matrix exceeds dgCMatrix's 2^31
# nnz cap. (For the full dataset, train + eval combined would otherwise
# realize ~10^10 nnz.)
# Skip when nothing downstream needs it: fit_chunk consumes only the cached
# train_obj.rds, and prep with both caches present has nothing left to do.
sce_lazy <- NULL
src_md   <- NULL
if (need_lazy_sce) {
  message("Reading source adata (lazy): ", data_p)
  sce_lazy <- readH5AD(data_p, reader = "R", use_hdf5 = TRUE)
  src_md   <- colData(sce_lazy)
}

# Apply the per-(pert, dataset) training cap upstream of any realization.
# Fewer training cells = smaller realized train Seurat. Eval splits are not
# subsampled per project policy.
if (need_realize && max_cells_per_pert > 0) {
  before <- length(train_idx)
  grp <- interaction(src_md[[p_col]][train_idx],
                     src_md[[dataset_col]][train_idx], drop = TRUE)
  set.seed(1)
  train_idx <- unlist(
    lapply(split(train_idx, grp), function(ix)
      if (length(ix) <= max_cells_per_pert) ix else sample(ix, max_cells_per_pert)),
    use.names = FALSE
  )
  message("Subsampled training indices: ", before, " -> ", length(train_idx),
          " (cap ", max_cells_per_pert, " per pert x dataset)")
}

# Load control adata + compute shared_features. Both are needed for prep
# (training-side merge_controls + gene trim) and for the eval loop. fit_chunk
# skips this since it consumes the cached train_obj which already has both
# baked in.
ctrl_obj_all    <- NULL
shared_features <- NULL
if (need_lazy_sce) {
  if (!is.null(ctrl_p) && nzchar(ctrl_p) && file.exists(ctrl_p)) {
    message("Reading control adata (cached for reuse): ", ctrl_p)
    ctrl_obj_all <- read_h5ad_to_seurat(ctrl_p)
    # Every cell in ctrl_h5ad IS a control, but the harmonised DB keeps each
    # study's own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET /
    # control). Downstream we identify controls by comparing this column against
    # ctrl_key, which would miss 18/22 datasets and leak controls into the logits.
    ctrl_obj_all@meta.data[[p_col]] <- ctrl_key
  }
  if (shared_genes) {
    shared_in_obj <- shared_genes_from_mask(sce_lazy)
    if (is.null(shared_in_obj)) {
      message("feature_mask_per_dataset not found; computing lazily across datasets")
      shared_in_obj <- shared_genes_lazy(sce_lazy, src_md, dataset_col)
    } else {
      message("Using precomputed feature_mask_per_dataset from adata.varm")
    }
    if (!is.null(ctrl_obj_all)) {
      shared_in_ctrl  <- shared_genes_across_datasets(ctrl_obj_all, dataset_col = dataset_col)
      shared_features <- intersect(shared_in_obj, shared_in_ctrl)
    } else {
      shared_features <- shared_in_obj
    }
    message("Shared-gene filter: ", length(shared_features), " / ", nrow(sce_lazy),
            " genes measured in every dataset")
    if (!is.null(ctrl_obj_all)) ctrl_obj_all <- ctrl_obj_all[shared_features, ]
    invisible(gc(verbose = FALSE))
  }
  # Single-dataset training: filter train_idx + eval_idx to the chosen
  # dataset's cells, and override shared_features with that dataset's own
  # gene panel (from feature_mask_per_dataset). Replaces the multi-dataset
  # --shared_genes filter when set. Done before max_cells_per_pert so the
  # cap operates on the already-filtered set.
  if (nzchar(train_dataset)) {
    src_ds <- as.character(src_md[[dataset_col]])
    ds_positions <- which(src_ds == train_dataset)
    if (length(ds_positions) == 0) {
      stop("--train_dataset ", shQuote(train_dataset),
           " not found in source adata's ", dataset_col, " column")
    }
    message("Single-dataset mode: '", train_dataset, "' (",
            length(ds_positions), " cells in source)")
    # Restrict for realize modes (full/prep) AND the chunk modes (deg_chunk/
    # fit_chunk): chunk_perts_and_cells() derives its perts + cells straight from
    # train_idx, so without this the chunked per-dataset path would fan DEGs/fits
    # over the WHOLE fold (all datasets' cells) instead of this dataset's — which
    # both defeats the point and OOMs. The chunked path is required per-dataset
    # too: some datasets carry ~9k perts at full.
    if ((need_realize || need_chunk) && !is.null(train_idx)) {
      before <- length(train_idx)
      train_idx <- intersect(train_idx, ds_positions)
      message("  train_idx: ", before, " -> ", length(train_idx))
    }
    drop_splits <- character(0)
    for (s in names(eval_idx)) {
      before <- length(eval_idx[[s]])
      kept <- intersect(eval_idx[[s]], ds_positions)
      message("  ", s, ": ", before, " -> ", length(kept))
      if (length(kept) == 0) drop_splits <- c(drop_splits, s)
      else eval_idx[[s]] <- kept
    }
    if (length(drop_splits) > 0) {
      message("  dropping empty splits: ", paste(drop_splits, collapse = ", "))
      eval_idx <- eval_idx[setdiff(names(eval_idx), drop_splits)]
    }

    # Gene space for this dataset's model. gene_panel="all" (default) keeps EVERY
    # gene so all per-dataset dictionaries live in one coordinate system (which is
    # what makes them comparable and mergeable); "dataset" narrows to D's own panel.
    if (identical(args$gene_panel, "all")) {
      shared_features <- rownames(sce_lazy)
      message("  gene panel: ALL ", length(shared_features), " genes (gene_panel=all)")
    } else {
    rd <- SummarizedExperiment::rowData(sce_lazy)
    if (!"feature_mask_per_dataset" %in% colnames(rd)) {
      stop("--train_dataset requires adata.varm$feature_mask_per_dataset")
    }
    mask <- rd$feature_mask_per_dataset
    # zellkonverter does NOT propagate varm column names, so colnames(mask) is
    # usually empty. The dataset -> column map lives in uns$dataset_to_mask_idx
    # (0-based python index). Try colnames first, then fall back to that.
    col_idx <- if (!is.null(colnames(mask)) && train_dataset %in% colnames(mask)) {
      match(train_dataset, colnames(mask))
    } else {
      d2m <- S4Vectors::metadata(sce_lazy)$dataset_to_mask_idx
      if (!is.null(d2m) && !is.null(d2m[[train_dataset]]))
        as.integer(d2m[[train_dataset]]) + 1L else NA_integer_
    }
    if (is.na(col_idx)) {
      d2m <- S4Vectors::metadata(sce_lazy)$dataset_to_mask_idx
      stop("Dataset ", shQuote(train_dataset),
           " not resolvable in feature_mask_per_dataset (no matching colname; ",
           "uns$dataset_to_mask_idx has: ",
           paste(names(d2m), collapse = ", "), ")")
    }
    shared_features <- rownames(sce_lazy)[mask[, col_idx] > 0]
    message("  per-dataset gene panel: ", length(shared_features),
            " / ", nrow(sce_lazy), " genes")
    }
    if (!is.null(ctrl_obj_all)) {
      ctrl_obj_all <- ctrl_obj_all[
        intersect(rownames(ctrl_obj_all), shared_features), ]
    }
  }
  # Chunk modes keep src_md — they use it to locate their perturbations' cells.
  if (!need_chunk) { rm(src_md); invisible(gc(verbose = FALSE)) }
}

# Global training-cell cap: keep realize under dgCMatrix's 2^31 nnz cap. We
# estimate nnz/cell from a 10k-cell sample (cheap), apply a safety factor,
# and uniformly random-subsample train_idx if it exceeds the resulting bound.
# An explicit --max_train_cells N caps further if N is smaller than the
# auto-bound. Eval splits are not subsampled per project policy.
if (need_realize) {
  est_cap  <- estimate_safe_cap(sce_lazy, train_idx, shared_features)
  hard_cap <- if (max_train_cells > 0) min(max_train_cells, est_cap) else est_cap
  if (length(train_idx) > hard_cap) {
    set.seed(3)
    before <- length(train_idx)
    train_idx <- sample(train_idx, hard_cap)
    message("Global training cap: ", before, " -> ", length(train_idx),
            " cells (random subsample to fit dgCMatrix nnz cap)")
  }
}

# Merge control cells into a split (restricted to shared dataset contexts),
# re-labelling cell ids to avoid collisions. Controls are required on both
# sides: training uses them to fit Λ; inference uses them as the reference
# distribution (`query()` subsets by dict classes + "control").
merge_controls <- function(target_obj, ctrl_all, dataset_col, target_tag) {
  if (is.null(ctrl_all)) return(target_obj)
  message("Adding control data (", target_tag, ")")
  shared_datasets <- intersect(target_obj@meta.data[[dataset_col]],
                               ctrl_all@meta.data[[dataset_col]])
  ctrl_sub   <- subset(ctrl_all, subset = !!sym(dataset_col) %in% shared_datasets)
  ctrl_sub   <- RenameCells(ctrl_sub,   add.cell.id = paste0("CTRL_", target_tag))
  target_obj <- RenameCells(target_obj, add.cell.id = target_tag)
  target_obj <- merge(target_obj, ctrl_sub)
  rm(ctrl_sub); invisible(gc(verbose = FALSE))
  target_obj
}

# ---- Option-2 chunk helpers (deg_chunk / fit_chunk realize their own perts) ----
# Split the training perturbations into n_chunks (same ordering as the legacy
# train_obj path) and return chunk `chunk_idx`'s perts + ALL their training cells
# (uncapped). src_md + train_idx come from the setup above.
chunk_perts_and_cells <- function(chunk_idx, n_chunks) {
  ptr <- as.character(src_md[[p_col]][train_idx])
  perts_all <- sort(unique(setdiff(ptr, ctrl_key)))
  bin   <- ceiling(length(perts_all) / max(1L, n_chunks))
  chunks <- unname(split(perts_all, ceiling(seq_along(perts_all) / bin)))
  if (chunk_idx + 1L > length(chunks)) return(NULL)
  cp <- chunks[[chunk_idx + 1L]]
  list(perts = cp, cells = train_idx[ptr %in% cp])
}

# Realize `cell_idx` (a chunk's perts) + the capped control set, then put them in
# the prep SCT model's residual space via apply_sct_model — so each task only
# realizes its own perturbations and no matrix nears the 2^31 nnz cap.
realize_chunk_in_model_space <- function(cell_idx, sct_fit) {
  obj <- realize_subset_seurat(sce_lazy, cell_idx, shared_features)
  ctrl_for_train <- ctrl_obj_all
  if (!is.null(ctrl_for_train) && max_ctrl_per_ds > 0) {
    md <- ctrl_for_train@meta.data; set.seed(2)
    keep <- unlist(lapply(split(seq_len(nrow(md)), md[[dataset_col]]), function(ix)
      if (length(ix) <= max_ctrl_per_ds) ix else sample(ix, max_ctrl_per_ds)),
      use.names = FALSE)
    ctrl_for_train <- ctrl_for_train[, keep]
  }
  obj <- merge_controls(obj, ctrl_for_train, dataset_col, "TRAIN")
  rm(ctrl_for_train); invisible(gc(verbose = FALSE))
  message("Applying cached SCT model to ", ncol(obj), " cells (chunk perts + controls)")
  apply_sct_model(obj, sct_fit)
}

#### ---- Training (or load cached / concat partial dicts) ----
dict_chunks_dir <- file.path(model_dir, "dict_chunks")
deg_chunks_dir  <- file.path(model_dir, "deg_chunks")

if (mode == "deg_chunk") {
  if (!file.exists(sct_model_p)) {
    stop("--mode deg_chunk requires cached sct_model.rds; ",
         "run --mode prep --skip_deg first. Looked at: ", sct_model_p)
  }
  dir.create(deg_chunks_dir, recursive = TRUE, showWarnings = FALSE)
  chunk_p <- file.path(deg_chunks_dir, sprintf("chunk_%04d.rds", chunk_idx))
  if (file.exists(chunk_p) && !args$force) {
    message("DEG chunk ", chunk_idx, " already at ", chunk_p, ", skipping")
  } else {
    cc <- chunk_perts_and_cells(chunk_idx, n_chunks)
    if (is.null(cc)) {
      message("chunk_idx ", chunk_idx, " exceeds the chunk count; nothing to do")
    } else {
      chunk_perts <- cc$perts
      message("Computing DEGs for chunk ", chunk_idx, " / ", n_chunks,
              ": ", length(chunk_perts), " perts, ", length(cc$cells),
              " cells (", chunk_perts[1], " ... ", chunk_perts[length(chunk_perts)], ")")
      # Realize this chunk's perts (uncapped) + controls and apply the cached SCT
      # model — no nnz cap since only one slice is realized. FindMarkers is serial.
      sct_fit <- readRDS(sct_model_p)
      sub <- realize_chunk_in_model_space(cc$cells, sct_fit)
      rm(sct_fit); invisible(gc(verbose = FALSE))

      features <- character(0)
      for (p in chunk_perts) {
        feats_p <- tryCatch({
          m <- Seurat::FindMarkers(sub, ident.1 = p, ident.2 = ctrl_key,
                                   group.by = p_col)
          n <- min(sum(m$p_val < 0.05), 500L)
          if (n == 0) character(0) else rownames(m)[seq_len(n)]
        }, error = function(e) {
          message("  FindMarkers failed for ", p, ": ", conditionMessage(e))
          character(0)
        })
        features <- unique(c(features, feats_p))
      }
      saveRDS(features, chunk_p)
      message("Saved ", length(features), " DEGs to ", chunk_p)
      rm(sub); invisible(gc(verbose = FALSE))
    }
    if (exists("train_obj")) rm(train_obj)
    invisible(gc(verbose = FALSE))
  }
  quit(save = "no")
}

if (mode == "concat_deg") {
  chunk_files <- sort(list.files(deg_chunks_dir,
                                  pattern = "^chunk_\\d+\\.rds$",
                                  full.names = TRUE))
  if (length(chunk_files) == 0) {
    stop("--mode concat_deg but no chunk files in ", deg_chunks_dir)
  }
  message("Concatenating ", length(chunk_files), " DEG chunk(s)")
  features <- character(0)
  for (cf in chunk_files) {
    f <- readRDS(cf)
    features <- unique(c(features, f))
  }
  saveRDS(features, features_p)
  message("Saved ", length(features), " union DEGs to ", features_p)
  quit(save = "no")
}

if (mode == "fit_chunk") {
  if (!file.exists(sct_model_p) || !file.exists(features_p)) {
    stop("--mode fit_chunk requires cached sct_model.rds and features.rds; ",
         "run --mode prep + concat_deg first. Looked at: ", sct_model_p, " / ", features_p)
  }
  dir.create(dict_chunks_dir, recursive = TRUE, showWarnings = FALSE)
  chunk_p <- file.path(dict_chunks_dir, sprintf("chunk_%04d.rds", chunk_idx))
  if (file.exists(chunk_p) && !args$force) {
    message("Chunk ", chunk_idx, " already at ", chunk_p, ", skipping")
  } else {
    features <- readRDS(features_p)
    cc <- chunk_perts_and_cells(chunk_idx, n_chunks)
    if (is.null(cc)) {
      message("chunk_idx ", chunk_idx, " exceeds the chunk count; nothing to fit")
    } else {
      chunk_perts <- cc$perts
      message("Fitting chunk ", chunk_idx, " / ", n_chunks,
              ": ", length(chunk_perts), " perts, ", length(cc$cells),
              " cells (", chunk_perts[1], " ... ", chunk_perts[length(chunk_perts)], ")")
      sct_fit <- readRDS(sct_model_p)
      sub <- realize_chunk_in_model_space(cc$cells, sct_fit)
      rm(sct_fit); invisible(gc(verbose = FALSE))
      chunk_dict <- LearnDictionary(
        sub,
        perturbation_meta = p_col,
        control_class     = ctrl_key,
        features          = features,
        override_filter   = TRUE
      )
      saveRDS(chunk_dict, chunk_p)
      message("Saved chunk dict to ", chunk_p)
      rm(sub, chunk_dict); invisible(gc(verbose = FALSE))
    }
    rm(features); invisible(gc(verbose = FALSE))
  }
  quit(save = "no")
}

if (mode == "concat_dict") {
  # Build dict.rds from dict_chunks/ and exit. Used as a barrier between the
  # fit_chunk array and eval_chunk arrays so the latter can read dict.rds.
  concat_dict_chunks(dict_chunks_dir, save_to = dict_p)
  quit(save = "no")
}

if (mode == "eval_chunk") {
  if (!nzchar(eval_split_arg) || !(eval_split_arg %in% names(eval_idx))) {
    stop("--mode eval_chunk requires --eval_split in {", paste(names(eval_idx), collapse = ", "), "}; got '", eval_split_arg, "'")
  }
  if (!file.exists(dict_p)) {
    stop("--mode eval_chunk requires dict.rds; run --mode concat_dict first.")
  }
  eval_chunk_root <- file.path(model_dir, "eval_chunks", eval_split_arg)
  dir.create(eval_chunk_root, recursive = TRUE, showWarnings = FALSE)
  out_p <- file.path(eval_chunk_root, sprintf("chunk_%04d.rds", chunk_idx))
  if (file.exists(out_p) && !args$force) {
    message("eval chunk ", chunk_idx, " already at ", out_p, ", skipping")
    quit(save = "no")
  }

  idx <- eval_idx[[eval_split_arg]]
  set.seed(11)
  perm <- sample(seq_along(idx))
  chunks <- unname(split(idx[perm], cut(seq_along(perm), n_chunks, labels = FALSE)))
  if (chunk_idx + 1L > length(chunks)) {
    message("chunk_idx ", chunk_idx, " exceeds ", length(chunks),
            " non-empty chunks; nothing to do")
    quit(save = "no")
  }
  ck <- chunks[[chunk_idx + 1L]]
  message("Eval chunk ", chunk_idx, " / ", n_chunks, " of ", eval_split_arg,
          ": ", length(ck), " cells")

  message("Loading dictionary: ", dict_p)
  dictionary <- readRDS(dict_p)

  test_obj <- realize_subset_seurat(sce_lazy, ck, shared_features)
  # Tag each eval cell with its 0-based source-adata index so it rides through
  # merge_controls / SCT / query and lands in prediction_obj@meta.data. Used
  # to write a /idx dataset to logits.h5 that maps each row back to a
  # row in the source adata's obs (and thus eval_idx).
  # source_idx now stamped inside realize_subset_seurat (survives dropped cells)
  test_obj <- merge_controls(test_obj, ctrl_obj_all, dataset_col,
                             paste0(eval_split_arg, "_c", chunk_idx))
  message("Pre-processing eval chunk")
  test_obj <- preprocess_data(test_obj, context_key = split_key)

  # subset = FALSE so cells whose perturbation isn't in dict still get
  # classified (they end up "unassigned" in practice). Keeps logits row
  # count aligned with eval_idx for downstream collation.
  prediction_obj <- query(test_obj, dict = dictionary, cores = cores,
                          subset = FALSE)
  rm(test_obj); invisible(gc(verbose = FALSE))

  logits <- fingerprint_logits(prediction_obj, dictionary,
                               condition_meta = p_col, control_class = ctrl_key)
  labels <- fingerprint_labels(prediction_obj,
                               condition_meta = p_col, control_class = ctrl_key)
  pert_mask <- as.character(prediction_obj@meta.data[[p_col]]) != ctrl_key
  meta <- prediction_obj@meta.data[pert_mask, , drop = FALSE]
  idx  <- meta$source_idx  # 0-based source-adata position per surviving cell
  rm(prediction_obj); invisible(gc(verbose = FALSE))

  saveRDS(list(logits = logits, labels = labels, meta = meta, idx = idx), out_p)
  message("Saved eval chunk to ", out_p)
  quit(save = "no")
}

if (has_cached_dict) {
  message("Loading cached dictionary: ", dict_p)
  dictionary <- readRDS(dict_p)
} else if (mode == "concat_eval") {
  # Build dict.rds from dict_chunks/ if not already there. (No-op if a
  # prior --mode concat_dict already wrote it; concat_dict_chunks always
  # rebuilds and overwrites — fine for this single-job stage.)
  dictionary <- concat_dict_chunks(dict_chunks_dir, save_to = dict_p)
} else {
  # Realize the (capped) training subset directly off the lazy SCE — no
  # combined obj is ever materialized. shared_features (if set) trims the
  # gene panel before realization. Skip realize+SCT entirely when
  # train_obj.rds is already on disk (most useful for --mode prep resumes
  # after a DEG-step OOM).
  if (mode == "prep" && skip_deg) {
    # ---- Option 2: fit ONE SCT model on the (capped) subsample and save it.
    # The deg/fit chunk arrays then realize their own perturbations (UNCAPPED)
    # and apply this model, so no single matrix nears the 2^31 nnz cap and the
    # dictionary is built from all training cells (not a 940k uniform subsample).
    if (has_cached_model) {
      message("Found cached sct_model, nothing to do: ", sct_model_p)
    } else {
      message("Realizing SCT-fit subsample: ", length(train_idx), " cells")
      fit_obj <- realize_subset_seurat(sce_lazy, train_idx, shared_features)
      ctrl_for_train <- ctrl_obj_all
      if (!is.null(ctrl_for_train) && max_ctrl_per_ds > 0) {
        md <- ctrl_for_train@meta.data; set.seed(2)
        keep <- unlist(lapply(split(seq_len(nrow(md)), md[[dataset_col]]), function(ix)
          if (length(ix) <= max_ctrl_per_ds) ix else sample(ix, max_ctrl_per_ds)),
          use.names = FALSE)
        ctrl_for_train <- ctrl_for_train[, keep]
      }
      fit_obj <- merge_controls(fit_obj, ctrl_for_train, dataset_col, "TRAIN")
      rm(ctrl_for_train); invisible(gc(verbose = FALSE))
      message("Fitting SCT model (+ per-(gene, ", split_key, ") offsets) on the subsample")
      sct_fit <- fit_sct_model(fit_obj, context_key = split_key)
      rm(fit_obj); invisible(gc(verbose = FALSE))
      saveRDS(sct_fit, file = sct_model_p)
      message("Saved SCT model to ", sct_model_p, " (", length(sct_fit$features),
              " genes; offsets ", if (is.null(sct_fit$offsets)) "none"
              else paste(dim(sct_fit$offsets), collapse = "x"), ")")
    }
    end_time <- timestamp()
    writeLines(c(start_time, end_time), con = time_p, sep = "\n")
    quit(save = "no")
  }

  # ---- Legacy joint-SCT path (full mode / prep without --skip_deg) ----
  if (has_cached_train_obj) {
    message("Found cached train_obj, loading: ", train_obj_p)
    train_obj <- readRDS(train_obj_p)
  } else {
    message("Realizing training subset: ", length(train_idx), " cells")
    train_obj <- realize_subset_seurat(sce_lazy, train_idx, shared_features)

    # Training-only: cap control cells per dataset. fit() only uses 1000 ctrls
    # for the PCA baseline, so keeping all ~18k/dataset is wasteful here. Do NOT
    # mutate ctrl_obj_all itself — it's reused untouched by every eval split.
    ctrl_for_train <- ctrl_obj_all
    if (!is.null(ctrl_for_train) && max_ctrl_per_ds > 0) {
      md <- ctrl_for_train@meta.data
      set.seed(2)
      keep <- unlist(
        lapply(split(seq_len(nrow(md)), md[[dataset_col]]), function(ix)
          if (length(ix) <= max_ctrl_per_ds) ix else sample(ix, max_ctrl_per_ds)),
        use.names = FALSE
      )
      message("Subsampling training controls: ", ncol(ctrl_for_train), " -> ",
              length(keep), " (cap ", max_ctrl_per_ds, " per dataset)")
      ctrl_for_train <- ctrl_for_train[, keep]
      invisible(gc(verbose = FALSE))
    }

    train_obj <- merge_controls(train_obj, ctrl_for_train, dataset_col, "TRAIN")
    rm(ctrl_for_train); invisible(gc(verbose = FALSE))

    message("Pre-processing training data")
    train_obj <- preprocess_data(train_obj, context_key = split_key)

    message("Caching train_obj to ", train_obj_p, " (a few minutes)")
    saveRDS(train_obj, file = train_obj_p)
  }

  # DEGs: cached on disk between runs. --cores 1 falls back to the package's
  # serial ChooseFeatures (no fork overhead); --cores >1 uses our parallel
  # reimpl with optional --deg_batch_size for resumability.
  if (has_cached_features) {
    message("Found cached features, loading: ", features_p)
    features <- readRDS(features_p)
  } else if (as.integer(cores) <= 1) {
    message("Calculating DEGs (serial, package ChooseFeatures)")
    features <- ChooseFeatures(
      train_obj, method = "de",
      perturbation_meta = p_col,
      control_class     = ctrl_key
    )
    saveRDS(features, file = features_p)
    message("Cached ", length(features), " DEGs to ", features_p)
  } else {
    message("Calculating DEGs (parallel, ", cores, " cores)")
    features <- choose_features_parallel(
      train_obj,
      perturbation_meta = p_col,
      control_class     = ctrl_key,
      cores             = as.integer(cores),
      batch_size        = deg_batch_size,
      cache_dir         = file.path(model_dir, "deg_cache")
    )
    saveRDS(features, file = features_p)
    message("Cached ", length(features), " DEGs to ", features_p)
  }

  if (mode == "prep") {
    message("Prep complete. Run --mode fit_chunk array, then --mode concat_eval.")
    rm(train_obj, features); invisible(gc(verbose = FALSE))
    end_time <- timestamp()
    writeLines(c(start_time, end_time), con = time_p, sep = "\n")
    quit(save = "no")
  }

  message("Learning dictionary")
  dictionary <- LearnDictionary(
    train_obj,
    perturbation_meta = p_col,
    control_class = ctrl_key,
    features = features,
    override_filter = TRUE
  )

  # Persist the dictionary first so the fit is not lost if any downstream
  # eval fails, then drop the training Seurat object before building any
  # test-side SCT matrices.
  saveRDS(dictionary, file = dict_p)
  rm(train_obj, features); invisible(gc(verbose = FALSE))
}

#### ---- Testing ----
# Evaluate each split sequentially. Each split's raw counts are realized
# independently from the lazy SCE. If the split is too big to realize as a
# single dgCMatrix (estimate_safe_cap), randomly partition it into chunks
# that each fit and run SCT + query() per chunk. Per-cell predictions are
# independent and the per-chunk control PCA baseline absorbs SCT differences
# across chunks (same justification as the per-split SCT we already do), so
# concatenating results is methodologically the same as a single realize.
# Eval indices themselves are NOT subsampled per project policy.
for (s in names(eval_idx)) {
  message("==== Evaluating split: ", s, " (", length(eval_idx[[s]]), " cells) ====")

  split_dir <- file.path(model_dir, s)
  dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)

  # Prefer pre-computed eval chunks from --mode eval_chunk array tasks.
  # Falls back to the in-process chunked-realize loop when none are present.
  eval_chunk_root <- file.path(model_dir, "eval_chunks", s)
  pre_chunks      <- if (dir.exists(eval_chunk_root)) {
    sort(list.files(eval_chunk_root, pattern    = "^chunk_\\d+\\.rds$",
                    full.names = TRUE))
  } else character(0)

  if (length(pre_chunks) > 0) {
    message("Concatenating ", length(pre_chunks), " pre-computed eval chunk(s) for ", s)
    all_logits <- vector("list", length(pre_chunks))
    all_labels <- vector("list", length(pre_chunks))
    all_meta   <- vector("list", length(pre_chunks))
    for (k in seq_along(pre_chunks)) {
      d <- readRDS(pre_chunks[[k]])
      all_logits[[k]] <- d$logits
      all_labels[[k]] <- d$labels
      all_meta[[k]]   <- d$meta
      rm(d)
    }
  } else {
    est_cap <- estimate_safe_cap(sce_lazy, eval_idx[[s]], shared_features)
    if (length(eval_idx[[s]]) <= est_cap) {
      chunks <- list(eval_idx[[s]])
    } else {
      n_chunks_inproc <- as.integer(ceiling(length(eval_idx[[s]]) / est_cap))
      set.seed(11)
      perm <- sample(seq_along(eval_idx[[s]]))  # shuffle so chunks aren't biased by row order
      chunks <- unname(split(eval_idx[[s]][perm],
                             cut(seq_along(perm), n_chunks_inproc, labels = FALSE)))
      message("Chunking ", s, ": ", n_chunks_inproc, " chunks of ~",
              ceiling(length(eval_idx[[s]]) / n_chunks_inproc), " cells each")
    }

    all_logits <- vector("list", length(chunks))
    all_labels <- vector("list", length(chunks))
    all_meta   <- vector("list", length(chunks))

    for (k in seq_along(chunks)) {
      ck <- chunks[[k]]
      chunk_tag <- if (length(chunks) > 1) paste0(s, "_c", k) else s
      if (length(chunks) > 1) {
        message("--- chunk ", k, "/", length(chunks), ": ", length(ck), " cells ---")
      }

      test_obj <- realize_subset_seurat(sce_lazy, ck, shared_features)  # stamps source_idx
      test_obj <- merge_controls(test_obj, ctrl_obj_all, dataset_col, chunk_tag)
      message("Pre-processing ", chunk_tag)
      test_obj <- preprocess_data(test_obj, context_key = split_key)

      # subset = FALSE so cells whose perturbation isn't in dict still get
      # classified (they end up "unassigned" in practice). Keeps logits row
      # count aligned with eval_idx for downstream collation.
      prediction_obj <- query(test_obj, dict = dictionary, cores = cores,
                              subset = FALSE)
      rm(test_obj); invisible(gc(verbose = FALSE))

      all_logits[[k]] <- fingerprint_logits(prediction_obj, dictionary,
                                            condition_meta = p_col,
                                            control_class  = ctrl_key)
      all_labels[[k]] <- fingerprint_labels(prediction_obj,
                                            condition_meta = p_col,
                                            control_class  = ctrl_key)
      pert_mask <- as.character(prediction_obj@meta.data[[p_col]]) != ctrl_key
      all_meta[[k]] <- prediction_obj@meta.data[pert_mask, , drop = FALSE]
      rm(prediction_obj); invisible(gc(verbose = FALSE))
    }
  }

  # Concatenate across chunks (single-chunk path is just the identity).
  logits <- do.call(rbind, all_logits)
  labels <- unlist(all_labels, use.names = FALSE)
  meta_combined <- do.call(rbind, all_meta)
  # idx: 0-based source-adata position for each surviving row. Pulled from
  # meta$source_idx if present (always populated for eval written by this
  # version), falls back to NA when reading older chunk rds files.
  idx <- if ("source_idx" %in% colnames(meta_combined)) {
    as.integer(meta_combined$source_idx)
  } else NULL
  stopifnot(nrow(logits) == length(labels),
            nrow(logits) == nrow(meta_combined))
  rm(all_logits, all_labels, all_meta); invisible(gc(verbose = FALSE))

  # Sanity: logits row count should equal eval_idx[[s]] (no eval cells dropped).
  # If it doesn't, the source adata likely contains cells with perturbation
  # == ctrl_key that get filtered by fingerprint_labels.
  if (nrow(logits) != length(eval_idx[[s]])) {
    warning("logits rows (", nrow(logits), ") != eval_idx[[", s, "]] (",
            length(eval_idx[[s]]), "); ",
            length(eval_idx[[s]]) - nrow(logits),
            " cells dropped — check for cells in source adata labeled as '",
            ctrl_key, "'.")
  }

  # Dense (n_perturbed_cells x n_classes) logits reconstructed from susie's
  # credible sets. Single HDF5 file mirroring cellot's predict.py output:
  # /logits float32, /labels strings, /perts strings.
  write_logits_h5(
    path   = file.path(split_dir, "logits.h5"),
    logits = logits,
    labels = labels,
    perts  = colnames(logits),
    idx    = idx
  )
  rm(logits, labels, idx); invisible(gc(verbose = FALSE))

  results <- test_results(
    meta_combined,
    dict = dictionary,
    control_key = ctrl_key,
    mixscale_score = test_ms_score,
    min_cells = min_cells
  )
  report <- get_report(results)
  message("  Macro F1 (", s, "): ", mean(report$f1))

  plot_f1_dist(results, report, file_p = file.path(split_dir, "f1_dist.svg"))
  #write.csv(results, file = file.path(split_dir, "results.csv"), quote = TRUE)
  write.csv(report, file = file.path(split_dir, "report.csv"), quote = FALSE)

  rm(meta_combined, results, report); invisible(gc(verbose = FALSE))
}
rm(sce_lazy, ctrl_obj_all); invisible(gc(verbose = FALSE))

# Save end time
end_time <- timestamp()
writeLines(c(start_time, end_time), con = time_p, sep = "\n")
