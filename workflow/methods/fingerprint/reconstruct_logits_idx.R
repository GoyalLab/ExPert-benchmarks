#!/usr/bin/env Rscript
# Recover /idx for an existing logits.h5 from the per-chunk rds metadata.
#
# Reads <model_dir>/eval_chunks/<split>/chunk_*.rds for each requested split,
# pulls each chunk's meta data.frame rownames (Seurat cell ids of the form
# "<split>_<orig>" or "<split>_c<N>_<orig>"), strips the chunk-tag prefix to
# recover the original cell id, looks each one up in the source adata's
# colnames (= obs_names) to get its 0-based position, and writes the result
# back into <model_dir>/<split>/logits.h5 as a new /idx dataset (int64,
# n_rows entries).
#
# Limitations:
#   - Requires eval_chunks/<split>/ to be on disk. In-process eval runs
#     (--n_eval_chunks 0) didn't persist per-chunk metadata; for those you
#     must re-run the eval stage.
#   - Does NOT recover cells that the old subset=TRUE query() filtered out;
#     those rows simply aren't in logits.h5 / chunk rds. /idx therefore
#     covers the rows that ARE in logits.h5, which is what's needed for
#     downstream collation by source-adata position.
#
# Usage:
#   Rscript reconstruct_logits_idx.R <model_dir> --source <source.h5ad> \
#       [--splits id_val id_test ood_test] [--force] [--dry_run]

suppressMessages({
  library(argparse)
  library(rhdf5)
})

p <- ArgumentParser()
p$add_argument("model_dir",
               help = "<output_dir>/<data_bn>/<fold>/<sg_suffix>")
p$add_argument("--source", required = TRUE,
               help = "Source adata (.h5ad) used during the run")
p$add_argument("--splits", nargs = "+",
               default = c("id_val", "id_test", "ood_test"))
p$add_argument("--force",   action = "store_true",
               help = "Overwrite /idx if it already exists in logits.h5")
p$add_argument("--dry_run", action = "store_true")
args <- p$parse_args()

model_dir <- normalizePath(args$model_dir, mustWork = TRUE)
source_p  <- normalizePath(args$source,    mustWork = TRUE)

# Source-adata cell ids. Read /obs/_index directly via rhdf5 (readH5AD
# eagerly parses all of obs which is minutes on 14M-cell atlases; we only
# need the cell-id column). Lookup uses match() — vectorized hash join,
# orders of magnitude faster than indexing a named vector at this scale.
message("Reading source cell ids from ", source_p)
t_read <- Sys.time()
src_names <- as.character(rhdf5::h5read(source_p, "/obs/_index"))
message("  ", length(src_names), " cells (",
        round(as.numeric(Sys.time() - t_read, units = "secs"), 1), "s)")

strip_chunk_tag <- function(cell_id, split) {
  # "<split>_<orig>"  or  "<split>_c<N>_<orig>"
  pat <- paste0("^", split, "(_c\\d+)?_")
  out <- sub(pat, "", cell_id)
  if (any(out == cell_id)) {
    bad <- cell_id[out == cell_id][1]
    stop("cell id ", shQuote(bad),
         " does not start with expected tag for split '", split, "'")
  }
  out
}

reconstruct_split <- function(split) {
  chunk_dir <- file.path(model_dir, "eval_chunks", split)
  h5_path   <- file.path(model_dir, split, "logits.h5")

  if (!file.exists(h5_path)) {
    message("[", split, "] logits.h5 missing at ", h5_path, ", skipping")
    return(invisible())
  }
  if (!dir.exists(chunk_dir)) {
    message("[", split, "] no eval_chunks at ", chunk_dir,
            "; in-process run? skipping (rerun the eval stage to recover)")
    return(invisible())
  }
  chunks <- sort(list.files(chunk_dir, pattern = "^chunk_\\d+\\.rds$",
                            full.names = TRUE))
  if (length(chunks) == 0) {
    message("[", split, "] eval_chunks dir empty at ", chunk_dir)
    return(invisible())
  }

  idx_per_chunk <- vector("list", length(chunks))
  for (i in seq_along(chunks)) {
    obj <- readRDS(chunks[[i]])
    meta_df <- obj$meta
    if (is.null(meta_df)) {
      stop(chunks[[i]], ": no $meta in saved rds")
    }
    cell_ids <- rownames(meta_df)
    stripped <- strip_chunk_tag(cell_ids, split)
    pos1     <- match(stripped, src_names)  # 1-based, NA if missing
    if (anyNA(pos1)) {
      bad <- stripped[which(is.na(pos1))[1]]
      stop(chunks[[i]], ": cell id ", shQuote(bad),
           " not found in source adata colnames")
    }
    idx_per_chunk[[i]] <- as.integer(pos1) - 1L  # 0-based
    rm(obj, meta_df, cell_ids, stripped, pos1)
  }
  idx <- unlist(idx_per_chunk, use.names = FALSE)

  # Sanity: matches logits row count.
  ds <- rhdf5::h5ls(h5_path)
  n_rows <- as.integer(strsplit(ds$dim[ds$name == "logits"], " x ")[[1]][1])
  if (length(idx) != n_rows) {
    stop("[", split, "] reconstructed idx length ", length(idx),
         " != logits rows ", n_rows,
         " — chunk rds files may be stale or mis-ordered")
  }

  has_idx <- "idx" %in% ds$name
  if (has_idx && !args$force) {
    message("[", split, "] /idx already in ", h5_path, " (pass --force to overwrite)")
    return(invisible())
  }

  if (args$dry_run) {
    message("[", split, "] would write /idx (", length(idx), " entries, ",
            "min=", min(idx), " max=", max(idx), ") to ", h5_path)
    return(invisible())
  }

  if (has_idx) {
    h5delete(h5_path, "idx")
  }
  rhdf5::h5createDataset(
    h5_path, "idx",
    dims         = length(idx),
    storage.mode = "double",
    H5type       = "H5T_NATIVE_INT64",
    level        = 4
  )
  rhdf5::h5write(as.numeric(idx), h5_path, "idx")
  rhdf5::H5close()
  message("[", split, "] wrote /idx (", length(idx), " entries) to ", h5_path)
}

for (s in args$splits) reconstruct_split(s)
