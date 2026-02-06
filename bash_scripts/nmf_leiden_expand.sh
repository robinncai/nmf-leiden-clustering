#!/bin/bash
#SBATCH --job-name=nmf_leiden
#SBATCH --partition=bigmem
#SBATCH --output=log/nmf_leiden_%j.log
#SBATCH --error=log/nmf_leiden_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000G

set -euo pipefail

# --- paths ---
ROOT="/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering"
PYTHON="${ROOT}/.venv/bin/python"
NMF="${ROOT}/nmf_full_leiden_clustering.py"
VIS="${ROOT}/visualize.py"
VIS_EXPANDED="${ROOT}/visualize_expanded.py"

IN_CSV="/scratch/groups/sartandi/rcai2/projects/KMEANS/results/all_12type/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_meta_cluster_radius200.csv"
META="/scratch/groups/sartandi/rcai2/projects/KMEANS/data/harmonized_level12.csv"

# Neighborhood analysis CSV for spatial scatter visualization
SPATIAL_CSV="/oak/stanford/groups/sartandi/rcai2/pan_cancer_subtype/all_12type/spatial_analysis/neighborhood_analysis/cell_meta_cluster_radius200_counts_k13/job_11269003/harmonized_level12_kmeans_nh.csv"

# job-scoped outputs
OUTDIR="/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering/results/nmf_leiden/nmf_leiden_${SLURM_JOB_ID}"
PLOTS="${OUTDIR}/plots"
SPATIAL_PLOTS="${OUTDIR}/spatial_scatter_plots"

mkdir -p "$PLOTS" "$SPATIAL_PLOTS"
cd "$SLURM_SUBMIT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname) @ $(date)"
echo "Python: $PYTHON"
echo "OUTDIR: $OUTDIR"
echo "---- script ----"; cat "$0"; echo "---------------"

# --- run NMF ---
echo "RUN: $PYTHON $NMF $IN_CSV -o $OUTDIR --normalize-by-fov -k 130 -r 0.01 -n 5"
"$PYTHON" "$NMF" "$IN_CSV" -o "$OUTDIR" --normalize-by-fov -k 130 -r 0.01 -n 5

OUT_CSV="${OUTDIR}/neighborhood_freqs-cell_meta_cluster_radius200_nmf_leiden_clusters.csv"
test -f "$OUT_CSV" || { echo "ERROR: missing $OUT_CSV"; exit 1; }

# --- visualization ---
echo "RUN: $PYTHON $VIS $OUT_CSV -o $PLOTS -m $META"
"$PYTHON" "$VIS" "$OUT_CSV" -o "$PLOTS" -m "$META"

# --- spatial scatter visualization ---
# Generates scatter plots with cell positions (centroid-0, centroid-1),
# circle sizes based on cell_size, colored by cell_meta_cluster and kmeans_neighborhood
# Subsamples 3 FOVs per kmeans_neighborhood per batch
echo "RUN: $PYTHON $VIS_EXPANDED --spatial-scatter --spatial-scatter-csv $SPATIAL_CSV -o $SPATIAL_PLOTS --n-fovs-per-nh 3"
"$PYTHON" "$VIS_EXPANDED" --spatial-scatter --spatial-scatter-csv "$SPATIAL_CSV" -o "$SPATIAL_PLOTS" --n-fovs-per-nh 3

echo "Done @ $(date)"