#!/bin/bash
# ==============================================================================
# Aggregate NMF Leiden Results
# ==============================================================================
# Run this after all array tasks complete to create a summary of all configs.
#
# Usage:
#   bash bash_scripts/aggregate_job_results.sh <job_id>
#
# Example:
#   bash bash_scripts/aggregate_job_results.sh 12345678
# ==============================================================================

set -euo pipefail

ROOT="/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering"
PYTHON="${ROOT}/.venv/bin/python"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <job_id>"
    echo ""
    echo "Available job directories:"
    ls -1 "${ROOT}/results/nmf_leiden/" 2>/dev/null | head -20
    exit 1
fi

JOB_ID="$1"
JOB_DIR="${ROOT}/results/nmf_leiden/${JOB_ID}"

if [[ ! -d "$JOB_DIR" ]]; then
    echo "ERROR: Job directory not found: $JOB_DIR"
    exit 1
fi

echo "Aggregating results for job: $JOB_ID"
echo "Directory: $JOB_DIR"
echo ""

"$PYTHON" "${ROOT}/aggregate_results.py" "$JOB_DIR"

echo ""
echo "Results saved to:"
echo "  ${JOB_DIR}/aggregated_summary.csv"
echo "  ${JOB_DIR}/aggregated_summary.json"
