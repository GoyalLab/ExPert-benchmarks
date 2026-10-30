#!/usr/bin/env Rscript
# Per-dataset fingerprint ENSEMBLE inference.
#
# Each dataset was trained into its own dict.rds (per_dataset_panel/<dataset>/)
# on its own gene panel. This scores ONE chunk of a test split's cells against
# EVERY dataset's dict and combines them into a single logit vector over the
# global perturbation space:
#
#   for each dataset D:
#     realize the chunk cells in D's panel (= rownames of D's Lambdas), merge D's
#     own controls, SCTransform, query(dict_D) -> per-cell logits over D's perts;
#     softmax each cell's logits over D's perts -> probs.
#   combine: for each global pert p, average the prob over ONLY the datasets whose
#     class space contains p (covering models), then log.
#
# Uniform for all cells (id + ood): every cell is scored by every model. Chunking
# is by CELLS (the per-cell susie query is the wall-clock pole), like eval_chunk.
suppressMessages(library(argparse))
suppressMessages(library(jsonlite))
source("utils.R")   # RNAFingerprinting (query), Seurat, zellkonverter, helpers

p <- ArgumentParser()
p$add_argument("--fold_dir", required = TRUE)
p$add_argument("--ctrl_p", required = TRUE)
p$add_argument("--output_dir", required = TRUE, help = "RESULTS/fingerprint")
p$add_argument("--split", required = TRUE, help = "id_test | ood_test")
p$add_argument("--chunk_idx", type = "integer", required = TRUE)
p$add_argument("--n_chunks", type = "integer", required = TRUE)
p$add_argument("--dataset_col", default = "dataset")
p$add_argument("--p_col", default = "perturbation")
p$add_argument("--ctrl_key", default = "control")
p$add_argument("--split_key", default = "dataset")
p$add_argument("--min_overlap_genes", type = "integer", default = 20,
               help = "Skip a dataset's contribution to a cell-chunk when fewer than this many of its dictionary genes are measured in those cells.")
p$add_argument("--cores", type = "integer", default = 1)
p$add_argument("--mask_target", action = "store_true",
               help = "ABLATION: zero each perturbed cell's own target-gene SCT residual before the query")
p$add_argument("--max_ctrl_per_ds", type = "integer", default = 2000L,
               help = "Cap controls per dataset in the own-control merge (matches training). 0 disables.")
p$add_argument("--dataset", default = NULL,
               help = "FAN-OUT mode: score the chunk against ONLY this dataset's dict and write its per-cell probs to ensemble/<split>/ds_probs/<dataset>/chunk_<j>.rds (the concat combines all datasets). Omit for the legacy single-job loop over every dataset.")
args <- p$parse_args()

fold_dir    <- args$fold_dir
split       <- args$split
dataset_col <- args$dataset_col
p_col       <- args$p_col
ctrl_key    <- args$ctrl_key
split_key   <- args$split_key
chunk_idx   <- args$chunk_idx
n_chunks    <- args$n_chunks

fold      <- basename(fold_dir)
size      <- basename(dirname(fold_dir))
model_root <- file.path(args$output_dir, size, fold, "per_dataset_panel")

# ---- small helpers copied from fingerprint.R (kept decoupled) ----
load_idx <- function(path) {
  # +1L: the split files are 0-based (Python), and every consumer downstream --
  # realize_subset_seurat in particular -- treats these as 1-based R column
  # positions. Without the shift each evaluated cell was the one BEFORE the cell
  # the split names, so the whole eval ran on a set shifted by one: at dataset
  # boundaries that pulled in cells from datasets not in the split at all, and
  # those became 1-cell dataset groups that the UNWEIGHTED per-dataset average
  # counted equally with 40k-cell groups. fingerprint.R's load_idx has always had
  # the +1L; this copy and merge_eval.R's did not.
  con <- gzfile(path, "rt"); on.exit(close(con))
  as.integer(readLines(con)) + 1L
}
merge_controls <- function(target_obj, ctrl_all, dataset_col, target_tag,
                           max_ctrl_per_ds = 0L) {
  if (is.null(ctrl_all)) return(target_obj)
  shared_datasets <- intersect(target_obj@meta.data[[dataset_col]],
                               ctrl_all@meta.data[[dataset_col]])
  if (length(shared_datasets) == 0) return(target_obj)
  ctrl_sub   <- subset(ctrl_all, subset = !!rlang::sym(dataset_col) %in% shared_datasets)
  # Cap controls per dataset (matches training's --max_ctrl_per_ds, default 2000):
  # controls only set each cell's per-dataset baseline (~1k suffices), but the full
  # ctrl.h5ad is ~180k cells (~10k/dataset). Merging all of them into every
  # dataset's SCTransform is the OOM driver AND a train/eval inconsistency — the
  # dictionaries were fit against a 2k/dataset control baseline. Subsample to match.
  if (max_ctrl_per_ds > 0L) {
    dsv   <- as.character(ctrl_sub@meta.data[[dataset_col]])
    cells <- colnames(ctrl_sub)
    keep  <- unlist(lapply(split(cells, dsv), function(cc)
      if (length(cc) <= max_ctrl_per_ds) cc else sample(cc, max_ctrl_per_ds)),
      use.names = FALSE)
    if (length(keep) < length(cells)) {
      ctrl_sub <- subset(ctrl_sub, cells = keep)
      message("  capped controls to <=", max_ctrl_per_ds, "/dataset: ",
              length(keep), " of ", length(cells), " cells")
    }
  }
  ctrl_sub   <- RenameCells(ctrl_sub,   add.cell.id = paste0("CTRL_", target_tag))
  target_obj <- RenameCells(target_obj, add.cell.id = target_tag)
  target_obj <- merge(target_obj, ctrl_sub)
  rm(ctrl_sub); invisible(gc(verbose = FALSE))
  target_obj
}
softmax_rows <- function(M) {                       # per-row softmax (row max shift)
  mx <- apply(M, 1L, max)
  E  <- exp(M - mx)                                 # mx recycles down columns = per-row
  E / rowSums(E)
}

# ---- metadata + eval indices ----
meta <- fromJSON(file.path(dirname(fold_dir), "metadata.json"))
data_p <- meta$source_path
message("Source adata: ", data_p)

idx_all <- load_idx(file.path(fold_dir, paste0(split, ".txt.gz")))
set.seed(11)                                        # SAME chunking as eval_chunk
perm   <- sample(seq_along(idx_all))
chunks <- unname(split(idx_all[perm], cut(seq_along(perm), n_chunks, labels = FALSE)))
if (chunk_idx + 1L > length(chunks)) {
  message("chunk_idx ", chunk_idx, " exceeds ", length(chunks), " non-empty chunks; nothing to do")
  quit(save = "no")
}
ck <- chunks[[chunk_idx + 1L]]
message("Ensemble chunk ", chunk_idx, " / ", n_chunks, " of ", split, ": ", length(ck), " cells")

# Output path is mode-dependent; resolve it + do the early skip BEFORE the
# expensive source/ctrl load. Fan-out (--dataset) writes one dataset's probs to
# ensemble/<split>/ds_probs/<dataset>/; the legacy loop writes the combined chunk.
if (!is.null(args$dataset)) {
  out_p <- file.path(model_root, "..", "ensemble", split, "ds_probs",
                     args$dataset, sprintf("chunk_%04d.rds", chunk_idx))
} else {
  out_p <- file.path(model_root, "..", "ensemble", split, sprintf("chunk_%04d.rds", chunk_idx))
}
if (file.exists(out_p) && !isTRUE(args$force)) { message("exists, skipping: ", out_p); quit(save = "no") }
dir.create(dirname(out_p), recursive = TRUE, showWarnings = FALSE)

# ---- load source + controls ----
message("Reading source adata (lazy): ", data_p)
sce_lazy     <- readH5AD(data_p, reader = "R", use_hdf5 = TRUE)
ctrl_obj_all <- read_h5ad_to_seurat(args$ctrl_p)
# Every cell in ctrl_h5ad IS a control, but the harmonised DB keeps each
# study's own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET /
# control). Downstream we identify controls by comparing this column against
# ctrl_key, which would miss 18/22 datasets and leak controls into the logits.
ctrl_obj_all@meta.data[[p_col]] <- ctrl_key

# Score ONE dataset's dict against the chunk cells -> per-cell probs over its
# perts (rows align with source_idx of the perturbed cells). Shared by both modes.
score_one_dataset <- function(dp) {
  D       <- basename(dirname(dp))
  dict_D  <- readRDS(dp)
  panel_D <- rownames(dict_D$fingerprints$Lambdas)
  perts_D <- colnames(dict_D$fingerprints$Lambdas)
  message("  ", D, ": ", length(perts_D), " perts, ", length(panel_D), "-gene panel")
  obj <- realize_subset_seurat(sce_lazy, ck, panel_D)   # stamps meta.data$source_idx
  # Center eval cells on THEIR OWN dataset's controls (merge_controls keeps only
  # datasets present among the eval cells, capped per dataset), NOT model D's — a
  # cell's fingerprint is a control-relative delta w.r.t. its own batch; using D's
  # control would inject a cross-dataset baseline and penalize off-dataset models.
  obj <- merge_controls(obj, ctrl_obj_all, dataset_col,
                        paste0(D, "_", split, "_c", chunk_idx), args$max_ctrl_per_ds)
  obj <- preprocess_data(obj, context_key = split_key)
  if (isTRUE(args$mask_target)) obj <- mask_target_residuals(obj, p_col, ctrl_key)
  # A cell can only be scored against D's dictionary through genes D measured.
  # Each dataset carries its OWN 500-gene panel, hard zero-padded elsewhere, and
  # D's dictionary is DEG-selected so it lies entirely inside that panel (verified
  # 464/464 genes inside, 0 outside) REGARDLESS of gene_panel=all -- zero-padded
  # genes have no variance in D's cells and can never be picked as DEGs. Across
  # datasets the panels overlap by only 19-52 genes (median 52), so for some
  # (held-out dataset x dict) pairs almost nothing survives SCT and the package's
  # prcomp_irlba dies with "max(nu, nv) must be strictly less than min(nrow(A),
  # ncol(A))". That is a real limit of dataset-blind scoring on disjoint panels,
  # not a transient error: D cannot speak about these cells. Skip its contribution
  # -- the combiner already averages each pert over only the datasets that covered
  # it, so one fewer covering model is well defined.
  sct_feats <- tryCatch(rownames(obj[["SCT"]]), error = function(e) character(0))
  n_ov <- length(intersect(panel_D, sct_feats))
  if (n_ov < args$min_overlap_genes) {
    message("  SKIP ", D, ": only ", n_ov, " of ", length(panel_D),
            " dictionary genes measured in these cells (< ",
            args$min_overlap_genes, ")")
    return(list(perts_D = character(0), src_idx = integer(0),
                probs = matrix(numeric(0), nrow = 0, ncol = 0)))
  }
  pred <- query(obj, dict = dict_D, cores = args$cores, subset = FALSE)
  rm(obj); invisible(gc(verbose = FALSE))
  lg <- fingerprint_logits(pred, dict_D, condition_meta = p_col, control_class = ctrl_key)
  md <- pred@meta.data[as.character(pred@meta.data[[p_col]]) != ctrl_key, , drop = FALSE]
  rk <- as.integer(as.character(md$source_idx))       # 0-based src idx (perturbed cells)
  probs <- softmax_rows(lg)                           # per-row softmax; aligns with md
  rm(pred, dict_D); invisible(gc(verbose = FALSE))
  list(perts_D = perts_D, src_idx = rk, probs = probs)
}

# ==== FAN-OUT MODE: score ONLY --dataset, persist its probs; concat combines ====
if (!is.null(args$dataset)) {
  dp <- file.path(model_root, args$dataset, "dict.rds")
  if (!file.exists(dp)) stop("dict.rds not found for dataset ", args$dataset, ": ", dp)
  message("[fan-out] scoring dataset ", args$dataset)
  res    <- score_one_dataset(dp)
  labels <- as.character(colData(sce_lazy)[[p_col]][res$src_idx + 1L])
  saveRDS(list(perts_D = res$perts_D, src_idx = res$src_idx,
               probs = res$probs, labels = labels), out_p)
  message("Saved fan-out ds probs -> ", out_p, " (", length(res$src_idx),
          " cells x ", length(res$perts_D), " perts)")
  quit(save = "no")
}

# ==== LEGACY LOOP MODE: score every dataset in one job, combine + log here ====
dict_paths <- Sys.glob(file.path(model_root, "*", "dict.rds"))
if (length(dict_paths) == 0) stop("no per-dataset dict.rds under ", model_root)
datasets   <- basename(dirname(dict_paths))
message("Ensembling ", length(datasets), " dataset models: ", paste(datasets, collapse = ", "))
all_perts  <- sort(unique(unlist(lapply(dict_paths, function(dp)
  colnames(readRDS(dp)$fingerprints$Lambdas)))))
message("Global perturbation space: ", length(all_perts), " perts")

src_key   <- as.character(as.integer(ck) - 1L)       # 0-based source positions
prob_sum  <- matrix(0, nrow = length(ck), ncol = length(all_perts),
                    dimnames = list(src_key, all_perts))
cover_cnt <- setNames(numeric(length(all_perts)), all_perts)
for (di in seq_along(dict_paths)) {
  message("[", di, "/", length(dict_paths), "] ", datasets[di])
  res <- score_one_dataset(dict_paths[di])
  rk  <- as.character(res$src_idx)
  keep <- rk %in% src_key
  if (any(keep)) {
    prob_sum[rk[keep], res$perts_D] <- prob_sum[rk[keep], res$perts_D] +
      res$probs[keep, , drop = FALSE]
    cover_cnt[res$perts_D] <- cover_cnt[res$perts_D] + 1
  }
}

# combine: average prob over covering models, then log
covered <- cover_cnt > 0
combined <- prob_sum
combined[, covered] <- sweep(prob_sum[, covered, drop = FALSE], 2, cover_cnt[covered], "/")
logits <- log(combined + 1e-8)
labels <- as.character(colData(sce_lazy)[[p_col]][ck])
saveRDS(list(logits = logits, idx = as.integer(ck) - 1L, labels = labels), out_p)
message("Saved ensemble chunk -> ", out_p)
