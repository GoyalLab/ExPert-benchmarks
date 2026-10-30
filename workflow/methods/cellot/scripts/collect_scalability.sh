#!/bin/bash
# Collect slurm resource usage for every task in an array-job log directory.
#
# Infers the parent JobID from the directory name (expects .../run_<JobID>)
# and runs `sacct` over every array task, dumping a TSV.
#
# Usage:
#   ./scripts/collect_scalability.sh <log_dir> [out_file]
#
# Examples:
#   ./scripts/collect_scalability.sh .logs/fold_0/run_6025829
#   ./scripts/collect_scalability.sh .logs/fold_0/run_6025829 scalability.tsv
#
# Output columns: JobID JobName State Elapsed CPUTime MaxRSS MaxVMSize ReqMem AllocCPUS NodeList ExitCode
set -e

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <log_dir> [out_file]" >&2
    exit 1
fi

LOG_DIR="$(realpath "$1")"
OUT_FILE="${2:-/dev/stdout}"

DIR_NAME="$(basename "${LOG_DIR}")"
if [[ "${DIR_NAME}" != run_* ]]; then
    echo "ERROR: expected dir name like 'run_<JobID>', got '${DIR_NAME}'" >&2
    exit 1
fi
JOBID="${DIR_NAME#run_}"

FORMAT="JobID,JobName,State,Elapsed,CPUTime,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,NodeList,ExitCode"

# --noheader + -P (|-delimited) → post-process: pick only *.batch rows, which
# carry the MaxRSS etc. that the parent JobID row lacks.
sacct -j "${JOBID}" -P --noheader --format="${FORMAT}" \
    | awk -F'|' -v hdr="${FORMAT}" '
        BEGIN {
            gsub(",", "\t", hdr);
            print hdr;
        }
        # sacct emits one row per step. The .batch row has the useful metrics;
        # the bare ID row has State/Elapsed but no MaxRSS. Keep both and let
        # the consumer merge by array-task id.
        /\.batch\|/ || NR==1 || $1 ~ /^[0-9]+(_[0-9]+)?$/ {
            OFS="\t"; $1=$1; print
        }
    ' > "${OUT_FILE}"

if [ "${OUT_FILE}" != "/dev/stdout" ]; then
    ROWS=$(($(wc -l < "${OUT_FILE}") - 1))
    echo "Wrote ${ROWS} rows → ${OUT_FILE}"
fi
