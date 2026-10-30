#!/bin/bash
# Snakemake-invoked scGPT runner. Reads source/context from the size's
# metadata.json and the knobs from the task yaml, then runs scgpt_perturb.py.
# No #SBATCH header — Snakemake's slurm profile provides the allocation; no conda
# activation — --use-conda already activated workflow/envs/scgpt.yaml.
#
#   run_scgpt.sh <splits_dir> <config_path> <output_root> <model_dir> <stage> \
#                [<pert_chunk_idx> <n_pert_chunks> <split>]
#
# stage:
#   full   train + eval all perts/both splits in one job (output_root/<split>/logits.h5)
#   train  train + cache the fine-tuned model into <model_dir>, then exit (no eval)
#   eval   load the cached model from <model_dir> and evaluate ONLY pert-chunk
#          <idx>/<n> of <split> → output_root/<split>/logits.h5  (a partial; concat later)
set -eo pipefail
SPLITS_PATH="$(realpath "$1")"
CONFIG_PATH="$(realpath "$2")"
RESULTS_ROOT="$3"
MODEL_DIR="${4:-${RESULTS_ROOT}/model}"
STAGE="${5:-full}"
PERT_CHUNK_IDX="${6:-0}"
N_PERT_CHUNKS="${7:-1}"
EVAL_SPLIT="${8:-}"

# workflow/methods/scgpt  (this script lives in .../scgpt/scripts/)
SCGPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
META_P="$(dirname "${SPLITS_PATH}")/metadata.json"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
[ -f "${META_P}" ] || { echo "ERROR: metadata.json not found: ${META_P}" >&2; exit 1; }
DATA_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_path"])' "${META_P}")"
GEN_KEY="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("context_col","") or "")' "${META_P}")"

# Pull config knobs as shell assignments (same extraction the legacy .train.sh used).
eval "$(python - "${CONFIG_PATH}" <<'PY'
import sys, yaml, shlex
c = yaml.safe_load(open(sys.argv[1])) or {}
d = c.get("data") or {}; m = c.get("model") or {}
t = c.get("training") or {}; p = c.get("prediction") or {}
def q(v):  return shlex.quote("" if v is None else str(v))
def flag(v): return "1" if v in (True, "true", "True", 1, "1") else "0"
print(f"CTRL_H5AD={q(d.get('ctrl_h5ad'))}")
print(f"PRETRAINED={q(d.get('pretrained_model'))}")
print(f"MAX_SEQ_LEN={q(m.get('max_seq_len', 1536))}")
print(f"EMBSIZE={q(m.get('embsize', 512))}")
print(f"D_HID={q(m.get('d_hid', 512))}")
print(f"NLAYERS={q(m.get('nlayers', 12))}")
print(f"NHEAD={q(m.get('nhead', 8))}")
print(f"N_LAYERS_CLS={q(m.get('n_layers_cls', 3))}")
print(f"DROPOUT={q(m.get('dropout', 0.0))}")
print(f"USE_FAST={flag(m.get('use_fast_transformer', False))}")
print(f"MAX_EPOCHS={q(t.get('max_epochs', 15))}")
print(f"BATCH_SIZE={q(t.get('batch_size', 64))}")
print(f"PATIENCE={q(t.get('early_stopping_patience', 10))}")
print(f"LR={q(t.get('learning_rate', 1e-4))}")
print(f"STEPS_PER_EPOCH={q(t.get('train_steps_per_epoch', 500))}")
print(f"MIN_CELLS={q(t.get('min_train_cells_per_perturbation', 25))}")
print(f"AMP={flag(t.get('amp', False))}")
print(f"SEED={q(t.get('seed', 42))}")
print(f"METRIC={q(p.get('metric', 'cosine'))}")
print(f"EVAL_SPLITS={q(' '.join(p.get('eval_splits') or ['id_test','ood_test']))}")
print(f"MAX_EVAL_CELLS={q(p.get('max_eval_cells_per_split', 0) or 0)}")
print(f"MAX_CTRL_CELLS={q(p.get('max_control_cells_per_group', 0) or 0)}")
print(f"CLS_READOUT={flag(p.get('classifier_readout', False))}")
print(f"PRED_CTRL_CELLS={q(p.get('pred_control_cells', 32))}")
print(f"SCORE_N_GENES={q(p.get('score_n_genes', '') or '')}")
s = c.get("slurm") or {}
print(f"DEADLINE_HOURS={q(s.get('deadline_hours', ''))}")
print(f"DEADLINE_BUFFER={q(s.get('deadline_buffer_minutes', 20))}")
PY
)"

# Data/model paths come from the pipeline config (exported by the rule), not the task
# yaml; the yaml values remain only as a fallback for manual runs.
#   control store: a padded source (scgpt_pad_targets.py) has its padded control copy
#   beside it, named <ctrl basename>.padded.h5ad -- use that, else the canonical one.
CTRL_H5AD="${SCGPT_CTRL_H5AD:-${CTRL_H5AD}}"
PRETRAINED="${SCGPT_PRETRAINED:-${PRETRAINED}}"
if [[ "${DATA_PATH}" == *.padded.h5ad && -n "${CTRL_H5AD}" ]]; then
    _pc="$(dirname "${DATA_PATH}")/$(basename "${CTRL_H5AD%.h5ad}").padded.h5ad"
    [ -f "${_pc}" ] && CTRL_H5AD="${_pc}"
fi
[ -n "${CTRL_H5AD}" ] || { echo "ERROR: no control h5ad (SCGPT_CTRL_H5AD or task yaml data.ctrl_h5ad)" >&2; exit 1; }

# eval stage evaluates a single split (passed in); train/full use the config list.
if [ "${STAGE}" = "eval" ]; then
    [ -n "${EVAL_SPLIT}" ] || { echo "ERROR: eval stage needs a <split> arg" >&2; exit 1; }
    EVAL_SPLITS_ARR=("${EVAL_SPLIT}")
else
    read -ra EVAL_SPLITS_ARR <<< "${EVAL_SPLITS}"
fi
mkdir -p "${RESULTS_ROOT}"
echo "scGPT stage=${STAGE} source=${DATA_PATH} ctrl=${CTRL_H5AD} gen_key=${GEN_KEY} metric=${METRIC}"
echo "  eval_splits=${EVAL_SPLITS_ARR[*]} model_dir=${MODEL_DIR} results=${RESULTS_ROOT}"

ARGS=(
    --data-path "${DATA_PATH}"
    --ctrl-path "${CTRL_H5AD}"
    --split-dir "${SPLITS_PATH}"
    --eval-splits "${EVAL_SPLITS_ARR[@]}"
    --metric "${METRIC}"
    --seed "${SEED}"
    --max-epochs "${MAX_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --early-stopping-patience "${PATIENCE}"
    --learning-rate "${LR}"
    --train-steps-per-epoch "${STEPS_PER_EPOCH}"
    --min-train-cells-per-perturbation "${MIN_CELLS}"
    --max-seq-len "${MAX_SEQ_LEN}"
    --embsize "${EMBSIZE}"
    --d-hid "${D_HID}"
    --nlayers "${NLAYERS}"
    --nhead "${NHEAD}"
    --n-layers-cls "${N_LAYERS_CLS}"
    --dropout "${DROPOUT}"
    --output-csv "${RESULTS_ROOT}/scGPT_rankings.csv"
    --summary-json "${RESULTS_ROOT}/scGPT_summary.json"
    --predicted-h5ad "${RESULTS_ROOT}/predicted.h5ad"
    --original-h5ad "${RESULTS_ROOT}/original.h5ad"
    --rankings-all-csv "${RESULTS_ROOT}/rankings_all.csv"
    --rankings-top10-csv "${RESULTS_ROOT}/rankings_top10.csv"
    --rankings-top100-csv "${RESULTS_ROOT}/rankings_top100.csv"
    --no-save-h5ad
    --no-rankings-all
    --no-logits-csv
    --logits-only
    --save-model "${MODEL_DIR}"
)
[ -n "${GEN_KEY}" ]               && ARGS+=(--generalization-key "${GEN_KEY}")
[ -n "${PRETRAINED}" ]            && ARGS+=(--load-model "${PRETRAINED}")
[ "${USE_FAST}" = "1" ]          && ARGS+=(--use-fast-transformer)
[ "${AMP}" = "1" ]               && ARGS+=(--amp)
[ -n "${DEADLINE_HOURS}" ]       && ARGS+=(--deadline-hours "${DEADLINE_HOURS}")
[ -n "${DEADLINE_BUFFER}" ]      && ARGS+=(--deadline-buffer-minutes "${DEADLINE_BUFFER}")
[ "${MAX_EVAL_CELLS:-0}" != "0" ] && ARGS+=(--max-eval-cells-per-split "${MAX_EVAL_CELLS}")
[ "${MAX_CTRL_CELLS:-0}" != "0" ] && ARGS+=(--max-control-cells-per-group "${MAX_CTRL_CELLS}")
# classifier readout: score MEASURED profiles against per-candidate predictions
# instead of the label-conditioned prediction vs centroids (see scgpt_perturb.py).
[ "${CLS_READOUT}" = "1" ]       && ARGS+=(--classifier-readout --pred-control-cells "${PRED_CTRL_CELLS}")
# prediction.score_n_genes: drop the synthetic padded columns from the metric. The
# padded store keeps them so each perturbation has a token position to flag; they
# must not reach the cosine, or the label is readable from the measured cell.
[ -n "${SCORE_N_GENES:-}" ]      && ARGS+=(--score-n-genes "${SCORE_N_GENES}")

case "${STAGE}" in
    train) ARGS+=(--train-only) ;;
    eval)  ARGS+=(--pert-chunk-idx "${PERT_CHUNK_IDX}" --n-pert-chunks "${N_PERT_CHUNKS}") ;;
    full)  : ;;
    *) echo "ERROR: unknown stage '${STAGE}' (train|eval|full)" >&2; exit 1 ;;
esac

python -u "${SCGPT_DIR}/scripts/scgpt_perturb.py" "${ARGS[@]}"
echo "scGPT stage=${STAGE} complete: ${RESULTS_ROOT}"
