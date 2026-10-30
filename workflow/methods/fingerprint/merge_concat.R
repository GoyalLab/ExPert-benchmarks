#!/usr/bin/env Rscript
# Concat merged per-cell-chunk logits into the final logits.h5. Every chunk was
# scored against the SAME merged dict, so all chunks share identical pert columns —
# concatenation is a straight rbind over cells (then reorder by source idx). This is
# what makes merged scalable at full: chunk the millions of cells, each chunk is a
# small single-node job, rbind here.
suppressPackageStartupMessages({library(argparse); library(rhdf5)})
source("utils.R")   # write_logits_h5

p <- ArgumentParser()
p$add_argument("--chunk_dir", required = TRUE, help = "dir with chunk_<j>.h5")
p$add_argument("--n_chunks", type = "integer", required = TRUE)
p$add_argument("--out_h5", required = TRUE)
args <- p$parse_args()

files <- file.path(args$chunk_dir, sprintf("chunk_%04d.h5", 0:(args$n_chunks - 1L)))
missing <- files[!file.exists(files)]
if (length(missing)) stop("missing ", length(missing), " chunk file(s), e.g. ", missing[[1]])

logits_list <- labels_list <- idx_list <- vector("list", length(files))
perts <- NULL
for (i in seq_along(files)) {
  f <- files[[i]]
  pc <- as.character(rhdf5::h5read(f, "perts"))
  if (is.null(perts)) perts <- pc else if (!identical(perts, pc)) stop("pert-column mismatch in ", f)
  logits_list[[i]] <- rhdf5::h5read(f, "logits")            # cells x perts
  labels_list[[i]] <- as.character(rhdf5::h5read(f, "labels"))
  idx_list[[i]]    <- as.numeric(rhdf5::h5read(f, "idx"))    # 0-based source idx
}
logits <- do.call(rbind, logits_list)
labels <- unlist(labels_list)
idx    <- unlist(idx_list)
o <- order(idx); logits <- logits[o, , drop = FALSE]; labels <- labels[o]; idx <- idx[o]

dir.create(dirname(args$out_h5), recursive = TRUE, showWarnings = FALSE)
write_logits_h5(args$out_h5, logits, labels, perts, idx = idx)
message("Wrote ", args$out_h5, " (", nrow(logits), " cells x ", ncol(logits),
        " perts) from ", length(files), " chunks")
