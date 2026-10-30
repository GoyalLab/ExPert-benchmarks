#!/usr/bin/env Rscript
# MERGED-WEIGHTS fingerprint eval — the SCALABLE (approximate) alternative to the
# per-dataset ensemble. The ensemble scores each cell against all D dicts (D
# classify passes/cell -> ~sum-of-perts correlations, the wall-clock pole). This
# merges the D per-dataset dicts into ONE (union genes zero-padded; a pert present
# in several datasets has its Lambda columns + dist params AVERAGED over them) and
# runs ONE classify pass/cell over the union pert space -> ~13-17x cheaper eval.
#
# CAVEAT (why it's the "not scalable" fallback, not the real method): the per-dataset
# dicts live in per-dataset SCT-residual spaces; the merge + a single SCT of the cell
# is an approximation. This script exists to MEASURE how much accuracy that costs vs
# the ensemble — validate on small before trusting it at full.
suppressMessages(library(argparse))
suppressMessages(library(jsonlite))
source("utils.R")

p <- ArgumentParser()
p$add_argument("--fold_dir", required = TRUE)
p$add_argument("--ctrl_p", required = TRUE)
p$add_argument("--output_dir", required = TRUE)
p$add_argument("--split", required = TRUE)
p$add_argument("--chunk_idx", type = "integer", default = 0)
p$add_argument("--n_chunks", type = "integer", default = 1)
p$add_argument("--out_h5", required = TRUE)
p$add_argument("--dataset_col", default = "dataset")
p$add_argument("--p_col", default = "perturbation")
p$add_argument("--ctrl_key", default = "control")
p$add_argument("--split_key", default = "dataset")
p$add_argument("--max_ctrl_per_ds", type = "integer", default = 2000L)
p$add_argument("--cores", type = "integer", default = 8)
p$add_argument("--mask_target", action = "store_true",
               help = "ABLATION: zero each perturbed cell's own target-gene SCT residual before the query")
p$add_argument("--merge_mode", default = "select", choices = c("select", "average"),
               help = "how to combine a pert's fingerprint across datasets: 'select' (#1: keep the single most-confident dataset's, no cross-SCT-space blending) or 'average' (legacy).")
args <- p$parse_args()

fold <- basename(args$fold_dir); size <- basename(dirname(args$fold_dir))
model_root <- file.path(args$output_dir, size, fold, "per_dataset_panel")

merge_controls <- function(target_obj, ctrl_all, dataset_col, target_tag, cap = 0L) {
  if (is.null(ctrl_all)) return(target_obj)
  shared <- intersect(target_obj@meta.data[[dataset_col]], ctrl_all@meta.data[[dataset_col]])
  if (length(shared) == 0) return(target_obj)
  cs <- subset(ctrl_all, subset = !!rlang::sym(dataset_col) %in% shared)
  if (cap > 0L) {
    dsv <- as.character(cs@meta.data[[dataset_col]]); cells <- colnames(cs)
    keep <- unlist(lapply(split(cells, dsv), function(c) if (length(c) <= cap) c else sample(c, cap)), use.names = FALSE)
    if (length(keep) < length(cells)) cs <- subset(cs, cells = keep)
  }
  cs <- RenameCells(cs, add.cell.id = paste0("CTRL_", target_tag))
  target_obj <- RenameCells(target_obj, add.cell.id = target_tag)
  merge(target_obj, cs)
}
softmax_rows <- function(M) { mx <- apply(M, 1L, max); E <- exp(M - mx); E / rowSums(E) }

# ---- Build the MERGED dict (union features x union perts; average shared perts) ----
dict_paths <- Sys.glob(file.path(model_root, "*", "dict.rds"))
if (length(dict_paths) == 0) stop("no dicts under ", model_root)
message("Merging ", length(dict_paths), " per-dataset dicts...")
dicts <- lapply(dict_paths, readRDS)
all_feats <- sort(unique(unlist(lapply(dicts, function(d) rownames(d$fingerprints$Lambdas)))))
all_perts <- sort(unique(unlist(lapply(dicts, function(d) colnames(d$fingerprints$Lambdas)))))
nf <- length(all_feats); np <- length(all_perts)
message("  union: ", nf, " features x ", np, " perts")

# #3 per-dataset scale = median per-pert L2 norm. classify()'s susie uses
# standardize=F on the raw Lambdas, so a larger-magnitude dataset would dominate;
# divide each dataset's L AND its null dists (same scale as L) by this so all
# datasets contribute comparably. The correlation pre-screen is already scale-free;
# scaling L+dists together keeps each pert's signal-to-null (LBF calibration) intact.
scales <- vapply(dicts, function(d) {
  L <- as.matrix(d$fingerprints$Lambdas); s <- stats::median(sqrt(colSums(L^2)))
  if (!is.finite(s) || s <= 0) 1 else s
}, numeric(1))

L_merged <- matrix(0, nf, np, dimnames = list(all_feats, all_perts))
dmean    <- matrix(0, nf, np, dimnames = list(all_feats, all_perts))
dsd      <- matrix(0, nf, np, dimnames = list(all_feats, all_perts))

if (args$merge_mode == "average") {
  # legacy: mean of covering datasets' (scaled) fingerprints — blends directions
  # from mismatched per-dataset SCT spaces.
  cover <- setNames(numeric(np), all_perts)
  for (di in seq_along(dicts)) {
    d <- dicts[[di]]; L <- as.matrix(d$fingerprints$Lambdas) / scales[di]
    fr <- rownames(L); pc <- colnames(L)
    L_merged[fr, pc] <- L_merged[fr, pc] + L
    cover[pc] <- cover[pc] + 1
    for (pt in pc) {
      dd <- d$fingerprints$dists[[pt]]
      dmean[fr, pt] <- dmean[fr, pt] + dd[, 1] / scales[di]
      dsd[fr, pt]   <- dsd[fr, pt] + dd[, 2] / scales[di]
    }
  }
  L_merged <- sweep(L_merged, 2, cover, "/")
  dmean    <- sweep(dmean, 2, cover, "/")
  dsd      <- sweep(dsd, 2, cover, "/")
} else {
  # #1 SELECT: per pert, keep the SINGLE most-confident dataset's fingerprint —
  # signal-to-null = ||L[,pt]|| / mean(null sd), scale-invariant — then take that
  # dataset's scaled L + dists verbatim. No cross-space blending. Same union dict
  # dims + same classify cost as average; only the build math differs.
  best_di  <- setNames(rep(NA_integer_, np), all_perts)
  best_snr <- setNames(rep(-Inf, np), all_perts)
  for (di in seq_along(dicts)) {
    L <- as.matrix(dicts[[di]]$fingerprints$Lambdas); pc <- colnames(L)
    for (pt in pc) {
      dd <- dicts[[di]]$fingerprints$dists[[pt]]
      snr <- sqrt(sum(L[, pt]^2)) / (mean(dd[, 2]) + 1e-8)
      if (is.finite(snr) && snr > best_snr[pt]) { best_snr[pt] <- snr; best_di[pt] <- di }
    }
  }
  for (di in seq_along(dicts)) {
    L <- as.matrix(dicts[[di]]$fingerprints$Lambdas); fr <- rownames(L); pc <- colnames(L)
    sel <- pc[!is.na(best_di[pc]) & best_di[pc] == di]
    if (length(sel)) {
      L_merged[fr, sel] <- L[, sel, drop = FALSE] / scales[di]
      for (pt in sel) {
        dd <- dicts[[di]]$fingerprints$dists[[pt]]
        dmean[fr, pt] <- dd[, 1] / scales[di]; dsd[fr, pt] <- dd[, 2] / scales[di]
      }
    }
  }
  message("  select: each pert -> its single most-confident dataset (signal-to-null)")
}
dsd[dsd < 1e-6] <- 1e-3   # zero-padded genes: tiny nonzero sd so resampling is stable
dists <- lapply(all_perts, function(pt) cbind(dmean[, pt], dsd[, pt]))
names(dists) <- all_perts
merged_dict <- list(fingerprints = list(Lambdas = L_merged, dists = dists))

# ---- Eval: ONE classify pass over the merged dict ----
meta <- fromJSON(file.path(dirname(args$fold_dir), "metadata.json"))
sce_lazy <- readH5AD(meta$source_path, reader = "R", use_hdf5 = TRUE)
ctrl_obj <- read_h5ad_to_seurat(args$ctrl_p)
# Every cell in ctrl_h5ad IS a control, but the harmonised DB keeps each
# study's own control nomenclature (NTC / Non-Targeting / SAFE / NO-TARGET /
# control). Downstream we identify controls by comparing this column against
# ctrl_key, which would miss 18/22 datasets and leak controls into the logits.
ctrl_obj@meta.data[[args$p_col]] <- args$ctrl_key

# +1L: split files are 0-based (Python); realize_subset_seurat indexes 1-based R
# columns. See the note in ensemble_eval.R's load_idx -- same bug, same fix.
idx_all <- as.integer(readLines(gzfile(file.path(args$fold_dir, paste0(args$split, ".txt.gz"))))) + 1L
set.seed(11); perm <- sample(seq_along(idx_all))
# n_chunks == 1 (the default, single-pass) must NOT hit cut(), whose single-break
# form errors ("invalid number of intervals"); one chunk = all cells.
grp <- rep(1L, length(perm))
if (args$n_chunks > 1L) grp <- cut(seq_along(perm), args$n_chunks, labels = FALSE)
chunks <- unname(split(idx_all[perm], grp))
ck <- chunks[[args$chunk_idx + 1L]]
message("Chunk ", args$chunk_idx, ": ", length(ck), " cells, ONE pass over ", np, " merged perts")

obj <- realize_subset_seurat(sce_lazy, ck, all_feats)       # union panel
obj <- merge_controls(obj, ctrl_obj, args$dataset_col,
                      paste0("merge_", args$split, "_c", args$chunk_idx), args$max_ctrl_per_ds)
obj <- preprocess_data(obj, context_key = args$split_key)
if (isTRUE(args$mask_target)) obj <- mask_target_residuals(obj, args$p_col, args$ctrl_key)
pred <- query(obj, dict = merged_dict, cores = args$cores, subset = FALSE)
lg <- fingerprint_logits(pred, merged_dict, condition_meta = args$p_col, control_class = args$ctrl_key)
md <- pred@meta.data[as.character(pred@meta.data[[args$p_col]]) != args$ctrl_key, , drop = FALSE]
src <- as.integer(as.character(md$source_idx))
probs <- softmax_rows(lg)
logits <- log(probs + 1e-8)
# per-pert z-score (same combiner)
mu <- colMeans(logits); sdv <- sqrt(pmax(colMeans(logits * logits) - mu * mu, 0)); act <- sdv > 1e-6
if (any(act)) logits[, act] <- sweep(sweep(logits[, act, drop = FALSE], 2, mu[act], "-"), 2, sdv[act], "/")
labels <- as.character(colData(sce_lazy)[[args$p_col]][src + 1L])

dir.create(dirname(args$out_h5), recursive = TRUE, showWarnings = FALSE)
write_logits_h5(args$out_h5, logits, labels, colnames(logits), idx = src)
message("Wrote ", args$out_h5, " (", nrow(logits), " cells x ", ncol(logits), " perts)")
