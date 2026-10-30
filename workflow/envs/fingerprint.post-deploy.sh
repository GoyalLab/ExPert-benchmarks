#!/bin/bash
# Snakemake conda post-deploy hook for workflow/envs/fingerprint.yaml. Snakemake
# runs this ONCE, right after building the env, with the env already activated
# (CONDA_PREFIX points at it). It installs the R packages not available via
# conda — the same steps as the legacy src/context/fingerprint/setup.sh — into
# the env's R library. Needs internet + a C/C++/Fortran toolchain (present on the
# CPU build node).
#
# upgrade="never" stops `remotes` from upgrading the conda-managed R packages
# (Seurat, Matrix, …) out from under the solved env.
set -eo pipefail
export R_LIBS_USER="${CONDA_PREFIX}/lib/R/library"

# NB: SeuratDisk is intentionally NOT installed — fingerprint doesn't use it (all
# h5ad I/O is via zellkonverter), and its hdf5r dep won't compile against recent
# HDF5. rhdf5 (utils.R writes logits.h5 via rhdf5::); AnnotationDbi + org.Hs.eg.db
# are presto/fingerprinting prerequisites.
Rscript -e 'BiocManager::install(c("rhdf5","AnnotationDbi","org.Hs.eg.db"), update=FALSE, ask=FALSE)'
Rscript -e 'remotes::install_github("immunogenomics/presto", INSTALL_opts="--no-lock", upgrade="never")'
Rscript -e 'remotes::install_github("satijalab/rna-fingerprinting", INSTALL_opts="--no-lock", upgrade="never")'
Rscript -e 'remotes::install_url("https://github.com/satijalab/rna-fingerprinting-data/raw/refs/heads/main/RNAFingerprintingData_0.1.0.tar.gz")'

# The installs above bump SeuratObject to 5.4.0, where PackageCheck() is defunct
# and as.Seurat.SingleCellExperiment() breaks at runtime. conda's intended pair
# is Seurat 5.1.0 + SeuratObject 5.2.0 (both pinned in the yaml, and the version
# the proven working env uses, where PackageCheck() is merely deprecated). Force
# SeuratObject back to 5.2.0 as the LAST step so nothing re-bumps it.
Rscript -e 'remotes::install_version("SeuratObject", version="5.2.0", upgrade="never", repos="https://cloud.r-project.org", INSTALL_opts="--no-lock")'

# Fail loudly if the key packages didn't land / SeuratObject drifted.
Rscript -e 'stopifnot(requireNamespace("RNAFingerprinting", quietly=TRUE), requireNamespace("RNAFingerprintingData", quietly=TRUE), requireNamespace("rhdf5", quietly=TRUE), as.character(packageVersion("SeuratObject")) == "5.2.0"); cat("fingerprint post-deploy OK; SeuratObject", as.character(packageVersion("SeuratObject")), "\n")'
