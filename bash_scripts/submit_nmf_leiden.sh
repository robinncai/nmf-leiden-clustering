#!/bin/bash
# ==============================================================================
# Submit NMF Leiden Clustering Array Job
# ==============================================================================
# This script submits the nmf_leiden_expand.sh as a SLURM array job,
# automatically determining the number of configurations from the config file.
#
# Usage:
#   bash bash_scripts/submit_nmf_leiden.sh [config_file] [--no-aggregate]
#
# Arguments:
#   config_file:    Optional path to config file (default: config/nmf_leiden_configs.txt)
#   --no-aggregate: Skip submitting the aggregation job
#
# Example:
#   bash bash_scripts/submit_nmf_leiden.sh
#   bash bash_scripts/submit_nmf_leiden.sh config/my_custom_configs.txt
#   bash bash_scripts/submit_nmf_leiden.sh --no-aggregate
# ==============================================================================

set -euo pipefail

ROOT="/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering"
SCRIPT="${ROOT}/bash_scripts/nmf_leiden_expand.sh"
AGGREGATE_SCRIPT="${ROOT}/bash_scripts/aggregate_nmf_leiden.sh"

# Parse arguments
CONFIG="${ROOT}/config/nmf_leiden_configs.txt"
DO_AGGREGATE=true

for arg in "$@"; do
    case $arg in
        --no-aggregate)
            DO_AGGREGATE=false
            ;;
        *)
            if [[ -f "$arg" ]]; then
                CONFIG="$arg"
            fi
            ;;
    esac
done

# Count non-comment, non-empty lines in config
N_CONFIGS=$(grep -v '^#' "$CONFIG" | grep -v '^$' | wc -l)

if [[ "$N_CONFIGS" -eq 0 ]]; then
    echo "ERROR: No configurations found in $CONFIG"
    exit 1
fi

echo "============================================================"
echo "NMF Leiden Clustering - Array Job Submission"
echo "============================================================"
echo "Config file: $CONFIG"
echo "Number of configurations: $N_CONFIGS"
echo ""
echo "Configurations to run:"
grep -v '^#' "$CONFIG" | grep -v '^$' | nl -w3 -s') '
echo ""
echo "============================================================"

# Create log directory if needed
mkdir -p "${ROOT}/log"

# Submit array job
echo "Submitting array job..."
JOB_ID=$(sbatch --array=1-${N_CONFIGS} --parsable "$SCRIPT")

echo ""
echo "Submitted job array: ${JOB_ID}"
echo "Array tasks: 1-${N_CONFIGS}"

# Submit aggregation job as dependency
if [[ "$DO_AGGREGATE" == true ]]; then
    echo ""
    echo "Submitting aggregation job (depends on array completion)..."

    # Create a simple aggregation job script
    AGG_JOB=$(sbatch --parsable \
        --dependency=afterok:${JOB_ID} \
        --job-name=nmf_aggregate \
        --output="${ROOT}/log/nmf_aggregate_${JOB_ID}.log" \
        --error="${ROOT}/log/nmf_aggregate_${JOB_ID}.err" \
        --time=00:30:00 \
        --cpus-per-task=1 \
        --mem=8G \
        --wrap="${ROOT}/.venv/bin/python ${ROOT}/aggregate_results.py ${ROOT}/results/nmf_leiden/${JOB_ID}")

    echo "Aggregation job: ${AGG_JOB} (will run after array completes)"
fi

echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${ROOT}/log/nmf_leiden_${JOB_ID}_*.log"
echo ""
echo "Results will be saved to:"
echo "  ${ROOT}/results/nmf_leiden/${JOB_ID}/"
echo ""
echo "After completion, aggregate with:"
echo "  bash bash_scripts/aggregate_job_results.sh ${JOB_ID}"
echo "============================================================"
