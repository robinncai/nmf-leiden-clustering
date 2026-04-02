#!/bin/bash
#SBATCH --job-name=nmf_leiden
#SBATCH --partition=bigmem
#SBATCH --output=log/nmf_leiden_%A_%a.log
#SBATCH --error=log/nmf_leiden_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=500G

set -euo pipefail

# ==============================================================================
# NMF Leiden Clustering - Multi-Configuration Runner
# ==============================================================================
# This script runs NMF + Leiden clustering with multiple (n, k, r) configurations.
#
# Usage:
#   1. Edit config file: config/nmf_leiden_configs.txt
#   2. Submit as array job:
#      sbatch --array=1-$(grep -v '^#' config/nmf_leiden_configs.txt | grep -v '^$' | wc -l) bash_scripts/nmf_leiden_expand.sh
#
#   Or use the helper script:
#      bash bash_scripts/submit_nmf_leiden.sh
#
# Output structure:
#   results/nmf_leiden/<JOB_ID>/
#     ├── n{n}_k{k}_r{r}/
#     │   ├── *_nmf_leiden_clusters.csv
#     │   ├── *_clustering_metrics.json
#     │   ├── *_cli_args.json
#     │   └── plots/
#     └── job_summary.json
# ==============================================================================

# --- Paths ---
ROOT="/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering"
PYTHON="${ROOT}/.venv/bin/python"
NMF="${ROOT}/nmf_full_leiden_clustering.py"
VIS="${ROOT}/visualize.py"
CONFIG="${ROOT}/config/nmf_leiden_configs.txt"

# --- Input data ---
IN_CSV="/scratch/groups/sartandi/rcai2/projects/KMEANS/results/all_15type_full/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_meta_cluster_radius200.csv"
META="/scratch/groups/sartandi/rcai2/projects/KMEANS/data/harmonized_level15_full.csv"

# --- Parse configuration from config file ---
# Get the line number corresponding to this array task
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK_ID="${SLURM_ARRAY_TASK_ID}"
else
    # If not running as array job, default to first config or use provided arg
    TASK_ID="${1:-1}"
fi

# Read config line (skip comments and empty lines)
CONFIG_LINE=$(grep -v '^#' "$CONFIG" | grep -v '^$' | sed -n "${TASK_ID}p")

if [[ -z "$CONFIG_LINE" ]]; then
    echo "ERROR: No configuration found for task ID ${TASK_ID}"
    echo "Check config file: $CONFIG"
    exit 1
fi

# Parse n, k, r from config line
read -r N_COMP N_NEIGH RESOLUTION <<< "$CONFIG_LINE"

echo "============================================================"
echo "Configuration ${TASK_ID}: n=${N_COMP}, k=${N_NEIGH}, r=${RESOLUTION}"
echo "============================================================"

# --- Output directories ---
# Use array job ID if available, otherwise use regular job ID
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
CONFIG_NAME="n${N_COMP}_k${N_NEIGH}_r${RESOLUTION}"

OUTDIR="${ROOT}/results/nmf_leiden/${JOB_ID}/${CONFIG_NAME}"
PLOTS="${OUTDIR}/plots"

mkdir -p "$OUTDIR" "$PLOTS"
cd "${SLURM_SUBMIT_DIR:-$ROOT}"

# --- Log job info ---
echo "Job ${JOB_ID} (task ${TASK_ID}) on $(hostname) @ $(date)"
echo "Python: $PYTHON"
echo "Config: n=${N_COMP}, k=${N_NEIGH}, r=${RESOLUTION}"
echo "Output: $OUTDIR"
echo "------------------------------------------------------------"

# --- Run NMF + Leiden clustering ---
echo "Running NMF + Leiden clustering..."
CMD="$PYTHON $NMF $IN_CSV -o $OUTDIR --normalize-by-fov -n $N_COMP -k $N_NEIGH -r $RESOLUTION"
echo "CMD: $CMD"
$CMD

# --- Verify output ---
BASENAME=$(basename "$IN_CSV" .csv)
OUT_CSV="${OUTDIR}/${BASENAME}_nmf_leiden_clusters.csv"
METRICS_JSON="${OUTDIR}/${BASENAME}_clustering_metrics.json"

if [[ ! -f "$OUT_CSV" ]]; then
    echo "ERROR: Missing output file: $OUT_CSV"
    exit 1
fi

# --- Create run summary JSON ---
SUMMARY_JSON="${OUTDIR}/run_summary.json"
cat > "$SUMMARY_JSON" << EOF
{
    "job_id": "${JOB_ID}",
    "task_id": "${TASK_ID}",
    "config": {
        "n_components": ${N_COMP},
        "n_neighbors": ${N_NEIGH},
        "resolution": ${RESOLUTION}
    },
    "paths": {
        "input": "${IN_CSV}",
        "output_dir": "${OUTDIR}",
        "clusters_csv": "${OUT_CSV}",
        "metrics_json": "${METRICS_JSON}"
    },
    "hostname": "$(hostname)",
    "started": "$(date -Iseconds)",
    "status": "running"
}
EOF

# --- Run visualization ---
echo "Running visualization..."
"$PYTHON" "$VIS" "$OUT_CSV" -o "$PLOTS" -m "$META" \
    --freq-pca --freq-csv "$IN_CSV" \
    --umap --subsample 200000

# --- Run Leiden spatial scatter plots ---
echo "Running spatial scatter plots..."
"$PYTHON" "$VIS" "$OUT_CSV" \
    -o "${PLOTS}/spatial_scatter" \
    --leiden-spatial-scatter \
    --leiden-scatter-meta "$META" \
    --n-fovs-per-cluster 3

# --- Update summary with completion status ---
COMPLETED_TIME=$(date -Iseconds)
if [[ -f "$METRICS_JSON" ]]; then
    # Read key metrics from the clustering metrics file
    METRICS=$(cat "$METRICS_JSON")
    cat > "$SUMMARY_JSON" << EOF
{
    "job_id": "${JOB_ID}",
    "task_id": "${TASK_ID}",
    "config": {
        "n_components": ${N_COMP},
        "n_neighbors": ${N_NEIGH},
        "resolution": ${RESOLUTION}
    },
    "metrics": ${METRICS},
    "paths": {
        "input": "${IN_CSV}",
        "output_dir": "${OUTDIR}",
        "clusters_csv": "${OUT_CSV}",
        "metrics_json": "${METRICS_JSON}",
        "plots_dir": "${PLOTS}"
    },
    "hostname": "$(hostname)",
    "started": "$(date -Iseconds)",
    "completed": "${COMPLETED_TIME}",
    "status": "completed"
}
EOF
fi

echo "============================================================"
echo "Done @ $(date)"
echo "Results: $OUTDIR"
echo "Metrics: $METRICS_JSON"
echo "Summary: $SUMMARY_JSON"
echo "============================================================"
