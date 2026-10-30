library(RNAFingerprinting)
library(RNAFingerprintingData)

options(Seurat.object.assay.version = 'v3') 
# Load libraries
library(Seurat)
# SeuratDisk intentionally not loaded: it is unused (all h5ad I/O goes through
# zellkonverter::readH5AD and conversions through base as.Seurat()), and its
# hdf5r dependency fails to compile against recent HDF5 (H5FD_family_init).
library(zellkonverter)
library(SingleCellExperiment)
library(dplyr)

library(ggplot2)

# Data functions
read_h5ad_to_seurat <- function(adata_p, layer = "X", keep_idx = NULL) {
  # Use zellkonverter to read sce. When `keep_idx` is supplied we go through
  # an HDF5-backed read + on-disk subset before realizing as a dgCMatrix; this
  # is required for the full dataset where the unsubsetted X exceeds the
  # 2^31 nnz cap of dgCMatrix and as.Seurat() fails with "SVT_SparseMatrix
  # contains too many nonzero values to be turned into a dgCMatrix".
  message("Reading h5ad adata: ", adata_p)
  use_hdf5 <- !is.null(keep_idx)
  sce <- readH5AD(adata_p, reader = "R", use_hdf5 = use_hdf5)
  if (use_hdf5) {
    message("On-disk subset: ", ncol(sce), " -> ", length(keep_idx), " cells")
    sce <- sce[, keep_idx]
    # Realize the (much smaller) subset to an in-memory CsparseMatrix and
    # rebuild a plain SCE so as.Seurat() doesn't see a DelayedMatrix.
    X  <- as(SummarizedExperiment::assay(sce, "X"), "CsparseMatrix")
    cd <- SummarizedExperiment::colData(sce)
    rd <- SummarizedExperiment::rowData(sce)
    sce <- SingleCellExperiment(
      assays = list(X = X), colData = cd, rowData = rd
    )
    rm(X, cd, rd); invisible(gc(verbose = FALSE))
  }
  # Drop rowData so the SCE→Seurat conversion doesn't try to stuff 11k+
  # gene-metadata columns into the assay's single meta.features slot, which
  # both throws a warning and allocates a useless data.frame copy.
  SummarizedExperiment::rowData(sce) <- NULL
  message("Converting to Seurat object")
  seurat_obj <- as.Seurat(sce, counts = "X", data = NULL)
  rm(sce); gc()
  seurat_obj
}

convert_ens_to_names <- function(seurat_obj, 
                                 col = "gene_symbol", 
                                 assay = NULL,
                                 src_assay = NULL,
                                 verbose = TRUE) {
  
  if (is.null(assay)) {
    assay <- DefaultAssay(seurat_obj)
  }
  if (is.null(src_assay)) {
    src_assay <- DefaultAssay(seurat_obj)
  }
  
  if (verbose) message("Using assay: ", assay)
  
  assay_obj <- seurat_obj[[assay]]
  meta_features <- seurat_obj[[src_assay]]@meta.features
  
  if (!col %in% colnames(meta_features)) {
    stop(paste0("Column '", col, "' not found in meta.features"))
  }
  
  current_features <- rownames(assay_obj)
  symbols <- meta_features[current_features, col]
  
  symbols[is.na(symbols) | symbols == ""] <- current_features[is.na(symbols) | symbols == ""]
  
  counts <- GetAssayData(seurat_obj, assay = assay, slot = "counts")
  
  # Resolve duplicates
  dup_symbols <- unique(symbols[duplicated(symbols)])
  features_to_keep <- current_features
  
  if (length(dup_symbols) > 0) {
    if (verbose) message("Removing duplicated ENS IDs...")
    
    for (sym in dup_symbols) {
      idx <- which(symbols == sym)
      totals <- Matrix::rowSums(counts[idx, , drop = FALSE])
      keep_idx <- idx[which.max(totals)]
      remove_idx <- setdiff(idx, keep_idx)
      features_to_keep <- setdiff(features_to_keep, current_features[remove_idx])
    }
  }
  
  # Subset data
  counts_new <- counts[features_to_keep, , drop = FALSE]
  
  # Create new feature names
  new_symbols <- meta_features[features_to_keep, col]
  new_symbols[is.na(new_symbols) | new_symbols == ""] <- features_to_keep[is.na(new_symbols) | new_symbols == ""]
  
  # Assign new rownames BEFORE creating assay
  rownames(counts_new) <- new_symbols
  
  # Rebuild assay
  new_assay <- CreateAssayObject(counts = counts_new)
  
  # Replace assay
  seurat_obj[[assay]] <- new_assay
  
  if (verbose) message("Rebuilt assay with gene symbols.")
  
  return(seurat_obj)
}

# Fast path: read shared genes from the precomputed feature_mask_per_dataset
# (n_genes x n_datasets 0/1 matrix in adata.varm, surfaced by zellkonverter
# as rowData(sce)$feature_mask_per_dataset). Returns NULL if the mask isn't
# present so the caller can fall back to shared_genes_lazy.
shared_genes_from_mask <- function(sce_lazy) {
  rd <- SummarizedExperiment::rowData(sce_lazy)
  if (!"feature_mask_per_dataset" %in% colnames(rd)) return(NULL)
  m <- rd$feature_mask_per_dataset
  if (is.null(dim(m)) || ncol(m) == 0) return(NULL)
  rownames(sce_lazy)[rowSums(m > 0) == ncol(m)]
}

# Same as shared_genes_across_datasets, but operates directly on a lazy
# (HDF5-backed) SingleCellExperiment so we never have to realize the full X.
# Required for the full dataset, where the unsubsetted X exceeds dgCMatrix's
# 2^31 nnz cap. Realizes one per-dataset chunk at a time (each is bounded by
# that dataset's cell count and fits comfortably). Used as a fallback when
# adata.varm$feature_mask_per_dataset is absent.
shared_genes_lazy <- function(sce_lazy, md, dataset_col) {
  feature_names <- rownames(sce_lazy)
  datasets   <- unique(as.character(md[[dataset_col]]))
  n_measured <- integer(length(feature_names))
  for (d in datasets) {
    cell_idx <- which(md[[dataset_col]] == d)
    if (length(cell_idx) == 0) next
    chunk <- as(SummarizedExperiment::assay(sce_lazy[, cell_idx], "X"),
                "CsparseMatrix")
    nnz <- if (inherits(chunk, "dgCMatrix")) {
      tabulate(chunk@i + 1L, nbins = nrow(chunk))
    } else {
      Matrix::rowSums(chunk != 0)
    }
    n_measured <- n_measured + as.integer(nnz > 0)
    message("  shared_genes_lazy: ", d, " (", length(cell_idx), " cells, ",
            sum(nnz > 0), " genes measured)")
    rm(chunk); invisible(gc(verbose = FALSE))
  }
  feature_names[n_measured == length(datasets)]
}

# Estimate the largest cell count for which `realize_subset_seurat(sce_lazy,
# idx, shared_features)` would stay under dgCMatrix's 2^31 nnz cap. Realizes
# a small random sample (default 10k cells) to measure nnz/cell, then
# extrapolates with a safety factor. Cheap (~seconds) compared to a failed
# full realize.
estimate_safe_cap <- function(sce_lazy, idx, shared_features = NULL,
                              sample_n = 10000, safety = 0.85) {
  cap <- .Machine$integer.max  # 2^31 - 1
  if (length(idx) <= sample_n) return(length(idx))
  set.seed(7)
  samp  <- sample(idx, sample_n)
  sce_s <- sce_lazy[, samp]
  if (!is.null(shared_features)) sce_s <- sce_s[shared_features, ]
  X <- as(SummarizedExperiment::assay(sce_s, "X"), "CsparseMatrix")
  nnz_per_cell <- length(X@x) / ncol(X)
  rm(X); invisible(gc(verbose = FALSE))
  max_cells <- as.integer(floor(cap * safety / nnz_per_cell))
  message("Estimated nnz/cell: ", round(nnz_per_cell, 1),
          " | safe cell cap (safety=", safety, "): ", max_cells)
  max_cells
}

# Realize a (cells = idx) slice of the lazy SCE into a fresh in-memory
# Seurat object. Optionally pre-trims to `shared_features` so we never
# allocate the full gene panel in the realized matrix. This is the entry
# point for both training and per-eval-split reads — each call gives an
# independent Seurat whose realized counts matrix is bounded by
# length(idx) * length(shared_features) (or full feature count) * density.
realize_subset_seurat <- function(sce_lazy, idx, shared_features = NULL) {
  # Drop indices outside R's valid 1..ncol range. The split files are 0-based, and
  # this pipeline reads them as R (1-based) column positions, so a chunk can contain
  # 0 (the 0-based first cell); R's `[` SILENTLY drops index 0, leaving ncol <
  # length(idx) and crashing the source_idx stamp below (`12501 to replace 12500`).
  # fold_0 never happened to hit index 0. Filtering keeps ncol == length(idx).
  idx <- as.integer(idx)
  idx <- idx[idx >= 1L & idx <= ncol(sce_lazy)]
  sce_sub <- sce_lazy[, idx]
  if (!is.null(shared_features)) {
    sce_sub <- sce_sub[shared_features, ]
  }
  X  <- as(SummarizedExperiment::assay(sce_sub, "X"), "CsparseMatrix")
  cd <- SummarizedExperiment::colData(sce_sub)
  # Stamp the 0-based source position PER CELL here, before the Seurat conversion,
  # so it travels with the cells: as.Seurat can drop cells (e.g. all-zero in a
  # dataset's panel), and assigning `as.integer(idx)-1L` to obj@meta.data AFTER
  # would then be a length mismatch (the `$<-.data.frame` error seen on some
  # fold/dataset chunks). Carried in colData, source_idx drops with any dropped
  # cell and stays aligned.
  cd$source_idx <- as.integer(idx) - 1L
  sce_sub <- SingleCellExperiment(assays = list(X = X), colData = cd)
  SummarizedExperiment::rowData(sce_sub) <- NULL
  obj <- as.Seurat(sce_sub, counts = "X", data = NULL)
  rm(sce_sub, X, cd); invisible(gc(verbose = FALSE))
  obj
}

# Return gene names with at least one non-zero count in every level of
# `dataset_col`. The merged adata has a 12k-gene universe built as the union
# across per-dataset panels, so genes absent from a dataset's panel sit at
# structural zero for every cell of that dataset. SCTransform can't tell
# those apart from biological zeros — it fits one NB per gene across all
# cells, and structural zeros contaminate both the NB parameters and the
# Pearson residuals, which in turn poisons the fingerprint Λ.
shared_genes_across_datasets <- function(seurat_obj, dataset_col, assay = NULL) {
  if (is.null(assay)) assay <- DefaultAssay(seurat_obj)
  counts   <- GetAssayData(seurat_obj, layer = "counts", assay = assay)
  datasets <- unique(as.character(seurat_obj@meta.data[[dataset_col]]))
  n_measured <- integer(nrow(counts))
  for (d in datasets) {
    cells <- which(seurat_obj@meta.data[[dataset_col]] == d)
    sub   <- counts[, cells, drop = FALSE]
    # tabulate on the row indices of a dgCMatrix gives per-row nnz without
    # densifying `sub != 0`
    nnz   <- if (inherits(sub, "dgCMatrix")) {
      tabulate(sub@i + 1L, nbins = nrow(sub))
    } else {
      Matrix::rowSums(sub != 0)
    }
    n_measured <- n_measured + as.integer(nnz > 0)
  }
  rownames(counts)[n_measured == length(datasets)]
}

.promote_counts_to_rna <- function(seurat_obj) {
  # Move counts from 'originalexp' (created by as.Seurat) into 'RNA' and drop
  # the original assay so we don't keep two copies of the sparse count matrix.
  counts <- GetAssayData(seurat_obj, layer = "counts", assay = "originalexp")
  seurat_obj[["RNA"]] <- CreateAssayObject(counts = counts)
  # Must switch default before dropping originalexp — Seurat refuses to
  # delete the assay currently set as default.
  DefaultAssay(seurat_obj) <- "RNA"
  seurat_obj[["originalexp"]] <- NULL
  rm(counts)
  invisible(gc(verbose = FALSE))
  seurat_obj
}

# ---- Option-2 SCT: fit once on a subsample, apply to all cells in shards ----
# preprocess_data() runs ONE joint SCTransform, so the realized matrix is bounded
# by dgCMatrix's 2^31 nnz cap (~940k cells at full) and training gets uniformly
# downsampled. These two helpers split fit from apply: fit_sct_model() fits the
# NB model (+ a per-(gene,context) offset table that replicates vars.to.regress)
# on a <=cap subsample; apply_sct_model() reproduces preprocess_data()'s SCT
# scale.data for ANY cells using that saved model — so each fit/deg chunk only
# realizes its own perturbations and no single matrix nears the cap. Validated on
# small/fold_0: residual cor 0.9992 (per-cell 0.9999) vs joint SCT+vars.to.regress.
fit_sct_model <- function(subsample_obj, context_key = "context") {
  subsample_obj <- .promote_counts_to_rna(subsample_obj)
  options(future.globals.maxSize = 100 * 1024^3)
  # Fit WITHOUT vars.to.regress; the context regression is replicated separately
  # as centering so it can be re-applied to new cells (reference.SCT.model carries
  # only the log_umi NB model, not vars.to.regress).
  # residual.features (NOT conserve.memory): Seurat 5.1.0 SILENTLY forces
  # `return.only.var.genes <- TRUE` inside the conserve.memory branch, overriding
  # the FALSE we pass. The cached model then covered only the top-variance genes
  # -- measured 482 of 3096, and because vst ranks by residual variance while the
  # DB's `gene_selection: fstat` ranked by perturbation F-stat, those 482 landed
  # ENTIRELY inside each dataset's 500-gene fstat panel (0 outside). Every stage
  # downstream inherits that model as reference.SCT.model, so FindMarkers and
  # LearnDictionary never saw the other ~2,600 measured genes -- silently undoing
  # gene_panel=all one layer below where it is applied.
  # Seurat dispatches on residual.features BEFORE conserve.memory and still
  # computes in chunks, so this keeps the memory behaviour while pinning the gene
  # set explicitly to everything the object carries.
  obj <- SCTransform(subsample_obj,
                     residual.features    = rownames(subsample_obj),
                     return.only.var.genes = FALSE, vars.to.regress = NULL,
                     verbose = FALSE)
  sd0 <- GetAssayData(obj, assay = "SCT", layer = "scale.data")
  offsets <- NULL
  if (!is.null(context_key) && context_key %in% colnames(obj@meta.data) &&
      length(unique(obj@meta.data[[context_key]])) > 1) {
    ctx  <- as.character(obj@meta.data[[context_key]])
    levs <- sort(unique(ctx))
    offsets <- vapply(levs, function(l) rowMeans(sd0[, ctx == l, drop = FALSE]),
                      numeric(nrow(sd0)))
    rownames(offsets) <- rownames(sd0); colnames(offsets) <- levs
  }
  list(model = slot(obj[["SCT"]], "SCTModel.list")[[1]], offsets = offsets,
       features = rownames(sd0), context_key = context_key)
}

# Apply a fit_sct_model() result to a fresh Seurat object: residuals from the
# saved NB model (reference.SCT.model) minus the per-(gene,context) offsets.
# Cells whose context level wasn't in the fit subsample get a zero offset.
apply_sct_model <- function(seurat_obj, sct_fit) {
  seurat_obj <- .promote_counts_to_rna(seurat_obj)
  options(future.globals.maxSize = 100 * 1024^3)
  seurat_obj <- SCTransform(seurat_obj, reference.SCT.model = sct_fit$model,
                            return.only.var.genes = FALSE, verbose = FALSE)
  if (!is.null(sct_fit$offsets)) {
    sd  <- GetAssayData(seurat_obj, assay = "SCT", layer = "scale.data")
    g   <- intersect(rownames(sd), rownames(sct_fit$offsets))
    ctx <- as.character(seurat_obj@meta.data[[sct_fit$context_key]])
    known <- ctx %in% colnames(sct_fit$offsets)
    if (any(known)) sd[g, known] <- sd[g, known] - sct_fit$offsets[g, ctx[known]]
    seurat_obj <- SetAssayData(seurat_obj, assay = "SCT", layer = "scale.data",
                               new.data = sd)
  }
  seurat_obj[["RNA"]] <- NULL
  invisible(gc(verbose = FALSE))
  seurat_obj
}


preprocess_data <- function(seurat_obj, context_key = "context") {
  # Single SCTransform pass over the whole object. When more than one context
  # level is present, pass context_key as vars.to.regress so dataset scale is
  # regressed out of the Pearson residuals in one fit — dramatically cheaper
  # than SplitObject → per-dataset SCTransform → merge
  seurat_obj <- .promote_counts_to_rna(seurat_obj)
  # 100 GiB (was 10): Seurat 5.1.0's SCTransform builds a larger vars.to.regress
  # closure than the working env's 5.3.1 (~18.6 GiB on small/fold_0); full is
  # bigger. Sequential future plan here, so no per-worker copy; prep has 200 GB.
  options(future.globals.maxSize = 100 * 1024^3)

  vars_to_regress <- NULL
  if (!is.null(context_key) &&
      context_key %in% colnames(seurat_obj@meta.data) &&
      length(unique(seurat_obj@meta.data[[context_key]])) > 1) {
    message("Regressing '", context_key, "' out of SCTransform residuals")
    seurat_obj@meta.data[[context_key]] <-
      as.factor(seurat_obj@meta.data[[context_key]])
    vars_to_regress <- context_key
  }

  seurat_obj <- SCTransform(
    seurat_obj,
    conserve.memory       = TRUE,
    return.only.var.genes = FALSE,
    vars.to.regress       = vars_to_regress,
    verbose               = FALSE
  )
  # SCT assay carries its own corrected counts; drop RNA to free the raw copy
  seurat_obj[["RNA"]] <- NULL
  invisible(gc(verbose = FALSE))
  seurat_obj
}


# Dense logits matrix reconstructed from susie's credible-set output on
# `prediction_obj` (produced by FingerprintCells). For each perturbed cell,
# every class in one of its credible sets gets `sign * lbf` of that CS; all
# other classes get 0. Argmax of this matrix reproduces top_credible_set, so
# the saved logits and the reported predictions use identical scoring —
# apples-to-apples with other methods' logits.csv at the top-k level.
fingerprint_logits <- function(prediction_obj, dict,
                               condition_meta = "perturbation",
                               control_class  = "control") {
  md      <- prediction_obj@meta.data
  classes <- colnames(dict$fingerprints$Lambdas)
  required <- c("all_credible_sets", "all_lbfs", "all_signs")
  missing  <- setdiff(required, colnames(md))
  if (length(missing) > 0) {
    stop("fingerprint_logits: prediction_obj missing susie columns: ",
         paste(missing, collapse = ", "),
         ". Run FingerprintCells with group_level = FALSE first.")
  }

  perturbed <- as.character(md[[condition_meta]]) != control_class
  md <- md[perturbed, , drop = FALSE]

  logits <- matrix(0, nrow = nrow(md), ncol = length(classes),
                   dimnames = list(rownames(md), classes))

  cs_vec    <- as.character(md$all_credible_sets)
  lbf_vec   <- as.character(md$all_lbfs)
  signs_vec <- as.character(md$all_signs)

  for (i in seq_len(nrow(md))) {
    cs_i <- cs_vec[i]
    if (is.na(cs_i) || cs_i %in% c("unassigned", "NA", "")) next
    cs_sets <- strsplit(cs_i, ",", fixed = TRUE)[[1]]
    lbfs    <- suppressWarnings(as.numeric(strsplit(lbf_vec[i], ",", fixed = TRUE)[[1]]))
    signs   <- strsplit(signs_vec[i], ",", fixed = TRUE)[[1]]
    sgn     <- ifelse(signs == "-", -1, 1)
    for (j in seq_along(cs_sets)) {
      members <- strsplit(trimws(cs_sets[j]), " ", fixed = TRUE)[[1]]
      members <- intersect(members, classes)
      if (length(members) > 0 && !is.na(lbfs[j])) {
        logits[i, members] <- sgn[j] * lbfs[j]
      }
    }
  }
  logits
}

# Read every chunk_NNNN.rds in `dict_chunks_dir`, cbind their Lambdas, and
# concatenate the per-pert dists list. Returns a "dictionary"-classed object
# matching what LearnDictionary builds. Optional save_to writes the dict to
# disk after assembly. Used by --mode concat_dict and --mode concat_eval.
concat_dict_chunks <- function(dict_chunks_dir, save_to = NULL) {
  chunk_files <- sort(list.files(dict_chunks_dir,
                                 pattern    = "^chunk_\\d+\\.rds$",
                                 full.names = TRUE))
  if (length(chunk_files) == 0) {
    stop("No chunk files in ", dict_chunks_dir)
  }
  message("Concatenating ", length(chunk_files), " dict chunk(s)")
  Lambdas_list <- list()
  dists_list   <- list()
  for (cf in chunk_files) {
    d <- readRDS(cf)
    if (!is.null(d$fingerprints) && !is.null(d$fingerprints$Lambdas)) {
      Lambdas_list[[length(Lambdas_list) + 1L]] <- d$fingerprints$Lambdas
      dists_list <- c(dists_list, d$fingerprints$dists)
    } else {
      message("  ", cf, ": empty (no perts passed signal filter)")
    }
    rm(d)
  }
  if (length(Lambdas_list) == 0) {
    stop("All chunk files were empty.")
  }
  Lambdas <- do.call(cbind, Lambdas_list)
  dictionary <- structure(
    list(fingerprints = list(Lambdas = Lambdas, dists = dists_list),
         order = NULL, data = NULL),
    class = "dictionary"
  )
  if (!is.null(save_to)) {
    saveRDS(dictionary, save_to)
    message("Concatenated dict (", ncol(Lambdas), " perts) saved to ", save_to)
  }
  dictionary
}

# Labels aligned to fingerprint_logits() rows: the perturbed (non-control)
# cells of `prediction_obj`, in meta.data order.
fingerprint_labels <- function(prediction_obj,
                               condition_meta = "perturbation",
                               control_class  = "control") {
  cm <- as.character(prediction_obj@meta.data[[condition_meta]])
  cm[cm != control_class]
}

# Write logits + labels + column names to a single HDF5 file matching the
# format cellot's predict.py emits: /logits float32 (n_cells x n_classes),
# /labels variable-strings (n_cells), /perts variable-strings (n_classes).
# rhdf5 doesn't support h5py's lzf, so we use gzip; h5py reads gzip
# transparently, so any downstream cellot code reads this file unchanged.
write_logits_h5 <- function(path, logits, labels, perts, idx = NULL, level = 4) {
  if (file.exists(path)) file.remove(path)
  rhdf5::h5createFile(path)
  rhdf5::h5createDataset(
    path, "logits",
    dims         = dim(logits),
    storage.mode = "double",
    H5type       = "H5T_NATIVE_FLOAT",
    chunk        = c(min(nrow(logits), 4096L), ncol(logits)),
    level        = level
  )
  rhdf5::h5write(logits, path, "logits")
  rhdf5::h5createDataset(
    path, "labels",
    dims         = length(labels),
    storage.mode = "character",
    size         = max(nchar(as.character(labels))) + 1L
  )
  rhdf5::h5write(as.character(labels), path, "labels")
  rhdf5::h5createDataset(
    path, "perts",
    dims         = length(perts),
    storage.mode = "character",
    size         = max(nchar(as.character(perts))) + 1L
  )
  rhdf5::h5write(as.character(perts), path, "perts")
  if (!is.null(idx)) {
    # Write idx as int64 so 0-based positions into the source adata roundtrip
    # cleanly (R 32-bit int could overflow on full atlases).
    rhdf5::h5createDataset(
      path, "idx",
      dims         = length(idx),
      storage.mode = "double",
      H5type       = "H5T_NATIVE_INT64",
      level        = level
    )
    rhdf5::h5write(as.numeric(idx), path, "idx")
  }
  rhdf5::H5close()
}

# Parallel reimplementation of ChooseFeatures(method = "de")
choose_features_parallel <- function(seurat_obj, perturbation_meta, control_class,
                                     max_per_perturbation = 500, cores = 1,
                                     batch_size = 0, cache_dir = NULL) {
  perts <- unique(as.character(seurat_obj@meta.data[[perturbation_meta]]))
  perts <- setdiff(perts, control_class)
  if (batch_size <= 0 || batch_size >= length(perts)) {
    batches <- list(perts)
  } else {
    batches <- unname(split(perts, ceiling(seq_along(perts) / batch_size)))
  }
  if (!is.null(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  }
  message("DEG computation: ", length(perts), " perts in ", length(batches),
          " batch(es), ", cores, " forks per batch")

  worker <- function(p) {
    tryCatch({
      m <- Seurat::FindMarkers(
        seurat_obj, ident.1 = p, ident.2 = control_class,
        group.by = perturbation_meta
      )
      n <- min(sum(m$p_val < 0.05), max_per_perturbation)
      if (n == 0) character(0) else rownames(m)[seq_len(n)]
    }, error = function(e) {
      message("  FindMarkers failed for ", p, ": ", conditionMessage(e))
      character(0)
    })
  }

  all_features <- character(0)
  for (i in seq_along(batches)) {
    batch   <- batches[[i]]
    cache_p <- if (!is.null(cache_dir)) {
      file.path(cache_dir, sprintf("batch_%04d.rds", i))
    } else NULL
    if (!is.null(cache_p) && file.exists(cache_p)) {
      message("  batch ", i, "/", length(batches), ": loading cached")
      feats <- readRDS(cache_p)
    } else {
      message("  batch ", i, "/", length(batches), ": ", length(batch), " perts")
      lst <- if (cores > 1) {
        parallel::mclapply(batch, worker, mc.cores = cores)
      } else {
        lapply(batch, worker)
      }
      feats <- unique(unlist(lst, use.names = FALSE))
      if (!is.null(cache_p)) saveRDS(feats, cache_p)
      rm(lst); invisible(gc(verbose = FALSE))
    }
    all_features <- unique(c(all_features, feats))
    rm(feats); invisible(gc(verbose = FALSE))
  }
  all_features
}

# Fingerprinting
query <- function(query_data, dict, condition_meta = "perturbation", control_class = "control", cores = 10, subset = T) {
  if (subset) {
    avail_classes <- c(colnames(dict$fingerprints$Lambdas), "control")
    tmp_test_obj <- subset(query_data, subset = !!sym(condition_meta) %in% avail_classes)
  } else {
    tmp_test_obj <- query_data
  }
  # Query model
  seurat_obj <- FingerprintCells(
    tmp_test_obj, 
    condition_meta = condition_meta, 
    control_class = control_class, 
    dictionary = dict, 
    group_level = FALSE, 
    mc.cores = cores
  )
  return(seurat_obj)
}

# Results and plotting
# Accepts either a Seurat object or a meta.data data.frame (the latter so
# chunked-eval callers can concatenate per-chunk metas before reporting).
test_results <- function(
    seurat_obj,
    dict,
    control_key = 'control',
    eff_col = 'efficiency',
    mixscale_score = 0.0,
    min_cells = 0
) {
  df <- if (inherits(seurat_obj, "Seurat")) seurat_obj@meta.data else seurat_obj
  df$top_prediction <- sapply(strsplit(as.character(df$top_credible_set), ","), function(x) ifelse(length(x) > 0, x[1], NA))
  # Set columns to factors
  df$perturbation <- as.factor(df$perturbation)
  df$top_prediction <- as.factor(df$top_prediction)
  # Predict for no control only
  df <- df[df$perturbation != control_key,]
  # Filter for minimum number of cells per class
  pert_counts <- table(df$perturbation)
  valid_pert <- names(pert_counts[pert_counts >= min_cells])
  df <- df[df$perturbation %in% valid_pert, ]
  # Filter for mixscale score
  if (mixscale_score > 0) {
    df <- df[df[[eff_col]] >= mixscale_score,]
  }
  return(df)
}

get_report <- function(df) {
  # All query classes to test for
  classes <- as.vector(sort(unique(df$perturbation)))
  
  # Function to compute F1 for a single class
  f1_for_class <- function(cls, truth, preds) {
    tp <- sum(preds == cls & truth == cls)
    fp <- sum(preds == cls & truth != cls)
    fn <- sum(preds != cls & truth == cls)
    tp <- ifelse(is.na(tp), 0, tp)
    fp <- ifelse(is.na(fp), 0, fp)
    fn <- ifelse(is.na(fn), 0, fn)
    
    prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    rec  <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
  }
  
  # Compute per-class F1
  f1_scores <- sapply(classes, f1_for_class, truth = df$perturbation, preds = df$top_prediction)
  # Convert to data.frame for ggplot2
  df_f1 <- data.frame(
    class = factor(names(f1_scores), levels = names(f1_scores)),
    f1 = f1_scores
  )
  return(df_f1)
}

plot_f1_dist <- function(results, report, file_p=NULL) {
  # Macro-average
  macro_f1 <- mean(report$f1)
  
  n_classes <- length(unique(results$perturbation))
  
  # Plot distribution
  p <- ggplot(report, aes(x = f1)) +
    geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
    geom_vline(xintercept = macro_f1, color = "red", linetype = "dashed", linewidth = 1) +
    labs(
      title = paste0("Distribution of Per-Class F1 Scores (N=", n_classes, ")"),
      x = "F1 Score",
      y = "Number of Classes"
    ) +
    annotate(
      "text",
      x = macro_f1 + 0.05,
      y = Inf,
      label = paste("Macro F1 =", round(macro_f1, 2)),
      vjust = 2,
      color = "red",
      fontface = "bold"
    ) +
    theme_minimal(base_size = 13)
  if(!is.null(file_p)) {
    message("Saving plot")
    ggsave(filename = file_p, plot = p, dpi = 300, width = 10, height = 6)
  } else {
    p
  }
}


# ABLATION (fingerprinting paper's protocol): for every perturbed cell, set the SCTransform residual
# (SCT scale.data -- what FingerprintCells reads) of its OWN target gene to 0. Fingerprints/dicts are
# untouched. Cells whose target is not an SCT feature, and controls, are left as is. Returns the object
# and logs how many cells were masked.
mask_target_residuals <- function(obj, p_col, ctrl_key) {
  sd <- GetAssayData(obj, assay = "SCT", layer = "scale.data")
  lab <- as.character(obj@meta.data[colnames(sd), p_col])
  hit <- which(lab != ctrl_key & lab %in% rownames(sd))
  if (length(hit) > 0) {
    sd[cbind(match(lab[hit], rownames(sd)), hit)] <- 0
    obj <- SetAssayData(obj, assay = "SCT", layer = "scale.data", new.data = sd)
  }
  message(sprintf("[mask-target] target residual zeroed in %d of %d perturbed cells",
                  length(hit), sum(lab != ctrl_key)))
  obj
}
