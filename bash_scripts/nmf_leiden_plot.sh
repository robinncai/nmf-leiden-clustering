#!/bin/bash
#SBATCH --job-name=nmf_plot
#SBATCH --partition=bigmem
#SBATCH --output=log/nmf_plot_%j.log
#SBATCH --error=log/nmf_plot_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=258G

set -euo pipefail

# --- paths ---
ROOT="/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering"
PYTHON="${ROOT}/.venv/bin/python"
VIS="${ROOT}/visualize.py"

# Input files
FREQ_CSV="/scratch/groups/sartandi/rcai2/projects/KMEANS/results/all_12type_2.8/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_meta_cluster_radius200.csv"
META="/scratch/groups/sartandi/rcai2/projects/KMEANS/data/harmonized_level12.csv"

# Output directory - specify the job results folder to plot
# Change this to the specific job folder you want to visualize
RESULTS_DIR="${ROOT}/results/nmf_leiden/nmf_leiden_14933886"
OUT_CSV="${RESULTS_DIR}/neighborhood_freqs-cell_meta_cluster_radius200_nmf_leiden_clusters.csv"
PLOTS="${RESULTS_DIR}/plots"

mkdir -p "$PLOTS"
cd "$ROOT"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) @ $(date)"
echo "Python: $PYTHON"
echo "Results dir: $RESULTS_DIR"
echo "Output plots: $PLOTS"
echo "---- script ----"; cat "$0"; echo "---------------"

# Check that cluster results exist
test -f "$OUT_CSV" || { echo "ERROR: missing $OUT_CSV"; exit 1; }

# --- visualization (no subsampling for full data) ---
echo "Running visualization pipeline..."
echo "RUN: $PYTHON $VIS $OUT_CSV -o $PLOTS -m $META --freq-csv $FREQ_CSV --freq-pca --no-pca"

"$PYTHON" "$VIS" "$OUT_CSV" \
    -o "$PLOTS" \
    -m "$META" \
    --freq-csv "$FREQ_CSV" \
    --freq-pca \
    --no-pca \
    --point-size 0.5 \
    --kde

echo "Done @ $(date)"
echo "Plots saved to: $PLOTS"