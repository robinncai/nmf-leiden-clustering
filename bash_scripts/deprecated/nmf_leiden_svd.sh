#!/bin/bash
#SBATCH --job-name=nmf_leiden
#SBATCH --output=log/nmf_leiden_%j.log
#SBATCH --error=log/nmf_leiden_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G

set -euo pipefail

# --- paths ---
ROOT="/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering"
PYTHON="${ROOT}/.venv/bin/python"
NMF="${ROOT}/nmf_full_leiden_clustering.py"
VIS="${ROOT}/visualize.py"

IN_CSV="/scratch/groups/sartandi/rcai2/projects/pan_cancer_subtype/KMEANS/results/all_12type/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_meta_cluster_radius200.csv"
META="/scratch/groups/sartandi/rcai2/projects/pan_cancer_subtype/KMEANS/data/harmonized_level12.csv"

# job-scoped outputs
OUTDIR="${ROOT}/results/nmf_leiden/nmf_leiden_${SLURM_JOB_ID}"
PLOTS="${OUTDIR}/plots"

mkdir -p "$PLOTS"
cd "$SLURM_SUBMIT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname) @ $(date)"
echo "Python: $PYTHON"
echo "OUTDIR: $OUTDIR"
echo "---- script ----"; cat "$0"; echo "---------------"

# --- run NMF ---
echo "RUN: $PYTHON $NMF $IN_CSV --svd -o $OUTDIR" 
"$PYTHON" "$NMF" "$IN_CSV" --svd --subsample "$META" --normalize-by-fov 

echo "Done @ $(date)"