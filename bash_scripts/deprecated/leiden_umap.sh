#!/bin/bash
#SBATCH --job-name=leiden_umap
#SBATCH --partition=bigmem
#SBATCH --output=log/leiden_umap_%j.log
#SBATCH --error=log/leiden_umap_%j.err
#SBATCH --time=24:00:00            # walltime (2 hours)
#SBATCH --cpus-per-task=4          # number of CPU cores
#SBATCH --mem=4000G                  # total memory

# activate .venv
source /scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering/.venv/bin/activate

# --- Move to the working directory (optional) ---
cd $SLURM_SUBMIT_DIR

# --- Print diagnostic info ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"

which python

# --- Run the pipeline ---
/scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering/.venv/bin/python /scratch/groups/sartandi/rcai2/projects/nmf-leiden-clustering/visualize.py \
    /scratch/groups/sartandi/rcai2/projects/results/neighborhood_freqs-cell_meta_cluster_radius200_nmf_leiden_clusters.csv -o test_plots -m /scratch/groups/sartandi/rcai2/projects/pan_cancer_subtype/KMEANS/data/harmonized_level12.csv

# --- Wrap up ---
echo "Job finished at: $(date)"