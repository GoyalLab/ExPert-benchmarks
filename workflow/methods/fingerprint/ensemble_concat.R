#!/usr/bin/env Rscript
# Stack per-cell ensemble chunks into one logits.h5 over the global pert space.
# Every chunk shares the same columns (the union of all dicts' classes), so this
# is a plain rbind of the logits + concat of idx/labels.
suppressMessages(library(argparse))
source("utils.R")   # write_logits_h5

p <- ArgumentParser()
p$add_argument("--output_dir", required = TRUE, help = "RESULTS/fingerprint")
p$add_argument("--fold_dir", required = TRUE)
p$add_argument("--split", required = TRUE)
p$add_argument("--n_chunks", type = "integer", required = TRUE)
p$add_argument("--out_h5", required = TRUE,
               help = "destination logits.h5 (a DISTINCT fingerprint_perdataset/ tree, so the collector doesn't merge it with shared fingerprint)")
args <- p$parse_args()

fold <- basename(args$fold_dir); size <- basename(dirname(args$fold_dir))
ens_dir <- file.path(args$output_dir, size, fold, "ensemble", args$split)
files <- file.path(ens_dir, sprintf("chunk_%04d.rds", seq_len(args$n_chunks) - 1L))
files <- files[file.exists(files)]
if (length(files) == 0) stop("no ensemble chunks under ", ens_dir)

parts  <- lapply(files, readRDS)
parts  <- Filter(function(x) !is.null(x$logits) && nrow(x$logits) > 0, parts)
logits <- do.call(rbind, lapply(parts, `[[`, "logits"))
idx    <- unlist(lapply(parts, `[[`, "idx"))
labels <- unlist(lapply(parts, `[[`, "labels"))
perts  <- colnames(logits)

# Per-pert z-score (same combiner fix as genepert): remove each perturbation's
# baseline (a dominant fingerprint scores high for EVERY cell, so softmax-avg
# makes the argmax collapse onto it). Column-wise mean/sd across the scored cells
# — we have the whole (cells x perts) matrix here, so it's exact and uses no
# labels/dataset identity. A per-cell log-softmax downstream preserves this
# ranking. Constant columns (globally uncovered perts at the log-eps floor) skip.
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
