#!/bin/bash
# Stage the pretrained scGPT_human checkpoint into the path referenced by
# config.yaml (scgpt_pretrained_model). The conda env is built by the repo-level
# setup.sh via Snakemake --use-conda — NOT here.
#
#   bash workflow/methods/scgpt/.setup.sh
set -eo pipefail

# Destination = config scgpt_pretrained_model (repo default data/resources/scGPT_human).
# Source = any directory holding the released scGPT_human checkpoint (best_model.pt,
# vocab.json, args.json), e.g. downloaded from the scGPT README's model zoo.
CKPT_DST="${SCGPT_CKPT_DST:-data/resources/scGPT_human}"
CKPT_SRC="${SCGPT_CKPT_SRC:-}"

if [ -f "${CKPT_DST}/best_model.pt" ]; then
    echo "Checkpoint already staged at ${CKPT_DST}"
elif [ -n "${CKPT_SRC}" ] && [ -f "${CKPT_SRC}/best_model.pt" ]; then
    echo "Staging checkpoint: ${CKPT_SRC} -> ${CKPT_DST}"
    mkdir -p "${CKPT_DST}"
    cp -n "${CKPT_SRC}/best_model.pt" "${CKPT_SRC}/vocab.json" "${CKPT_SRC}/args.json" "${CKPT_DST}/"
    echo "  staged: $(ls -1 "${CKPT_DST}")"
else
    echo "WARN: no checkpoint at ${CKPT_DST} or ${CKPT_SRC}." >&2
    echo "      scGPT will train from random init. Set SCGPT_CKPT_SRC (and SCGPT_CKPT_DST" >&2
    echo "      if config scgpt_pretrained_model is not the default) and re-run." >&2
fi
