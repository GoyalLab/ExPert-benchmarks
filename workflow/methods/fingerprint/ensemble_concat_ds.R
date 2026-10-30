#!/usr/bin/env Rscript
# Combine FAN-OUT per-dataset ensemble probs into one logits.h5.
#
# Each fp_ens_eval_ds_chunk[dataset, j] wrote ensemble/<split>/ds_probs/<dataset>/
# chunk_<j>.rds = {perts_D, src_idx (0-based perturbed cells), probs (softmax over
# perts_D), labels}. This does the dataset-blind combine the old single-job loop
# did inline: for each cell-chunk, average each pert's prob over ONLY the datasets
# whose class space covers it (covering models), log, then rbind all chunks and
# per-pert z-score (the dominant-fingerprint fix). No labels/dataset identity used
# in the combine — every cell is scored by every dataset, so cover[p] is just the
# number of datasets that trained on p.
suppressMessages(library(argparse))
source("utils.R")   # write_logits_h5

p <- ArgumentParser()
p$add_argument("--output_dir", required = TRUE, help = "RESULTS/fingerprint")
p$add_argument("--fold_dir", required = TRUE)
p$add_argument("--split", required = TRUE)
p$add_argument("--n_chunks", type = "integer", required = TRUE)
p$add_argument("--out_h5", required = TRUE)
args <- p$parse_args()

fold <- basename(args$fold_dir); size <- basename(dirname(args$fold_dir))
ds_root <- file.path(args$output_dir, size, fold, "ensemble", args$split, "ds_probs")
datasets <- list.dirs(ds_root, recursive = FALSE, full.names = FALSE)
if (length(datasets) == 0) stop("no ds_probs/<dataset>/ dirs under ", ds_root)
message("Combining ", length(datasets), " datasets x ", args$n_chunks, " chunks")

# global pert space = union of every dataset's classes (one chunk per dataset)
all_perts <- sort(unique(unlist(lapply(datasets, function(D) {
  f <- Sys.glob(file.path(ds_root, D, "chunk_*.rds"))
  if (length(f) == 0) NULL else readRDS(f[[1]])$perts_D
}))))
message("Global perturbation space: ", length(all_perts), " perts")

logits_list <- idx_list <- labels_list <- list()
for (j in seq_len(args$n_chunks) - 1L) {
  files <- file.path(ds_root, datasets, sprintf("chunk_%04d.rds", j))
  files <- files[file.exists(files)]
  if (length(files) == 0) next
  parts <- lapply(files, readRDS)

  # rows = union of scored (perturbed) cells across datasets (identical in practice)
  src_all <- sort(unique(unlist(lapply(parts, `[[`, "src_idx"))))
  srk     <- as.character(src_all)
  lab_map <- setNames(rep(NA_character_, length(src_all)), srk)
  for (pt in parts) lab_map[as.character(pt$src_idx)] <- pt$labels

  prob_sum <- matrix(0, nrow = length(src_all), ncol = length(all_perts),
                     dimnames = list(srk, all_perts))
  cover <- setNames(numeric(length(all_perts)), all_perts)
  for (pt in parts) {
    prob_sum[as.character(pt$src_idx), pt$perts_D] <-
      prob_sum[as.character(pt$src_idx), pt$perts_D] + pt$probs
    cover[pt$perts_D] <- cover[pt$perts_D] + 1
  }
  covered <- cover > 0
  prob_sum[, covered] <- sweep(prob_sum[, covered, drop = FALSE], 2, cover[covered], "/")

  logits_list[[length(logits_list) + 1L]] <- log(prob_sum + 1e-8)
  idx_list[[length(idx_list) + 1L]]       <- src_all
  labels_list[[length(labels_list) + 1L]] <- unname(lab_map[srk])
}
if (length(logits_list) == 0) stop("no chunks combined under ", ds_root)

logits <- do.call(rbind, logits_list)
idx    <- unlist(idx_list)
labels <- unlist(labels_list)
perts  <- colnames(logits)

# Per-pert z-score (same combiner fix as genepert): remove each perturbation's
# baseline so a dominant fingerprint that scores high for every cell doesn't
# collapse the argmax. Column mean/sd across all scored cells — exact, no labels.
mu  <- colMeans(logits)
sdv <- sqrt(pmax(colMeans(logits * logits) - mu * mu, 0))
active <- sdv > 1e-6
if (any(active)) {
  logits[, active] <- sweep(sweep(logits[, active, drop = FALSE], 2, mu[active], "-"),
                            2, sdv[active], "/")
}
message("per-pert z-score: ", sum(active), "/", length(active), " pert columns active")

dir.create(dirname(args$out_h5), recursive = TRUE, showWarnings = FALSE)
write_logits_h5(args$out_h5, logits, labels, perts, idx = idx)
message("Wrote ", args$out_h5, " (", nrow(logits), " cells x ", ncol(logits), " perts)")
