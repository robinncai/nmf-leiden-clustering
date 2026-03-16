#!/usr/bin/env python3
"""Generate PCA plots with three different preprocessing approaches."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

# Paths
CLUSTER_CSV = "results/nmf_leiden/nmf_leiden_14933886/neighborhood_freqs-cell_meta_cluster_radius200_nmf_leiden_clusters.csv"
ORIG_CSV = "/scratch/groups/sartandi/rcai2/projects/KMEANS/results/all_12type_2.8/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_meta_cluster_radius200.csv"
BASE_OUTPUT = "results/nmf_leiden/nmf_leiden_14933886/plots"

SUBSAMPLE = 100000
SEED = 42


def get_cluster_colors(n_clusters):
    """Generate distinct colors for clusters."""
    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    import matplotlib.colors as mcolors
    return [mcolors.rgb2hex(cmap(i / max(n_clusters, 1))) for i in range(n_clusters)]


def compute_pca(matrix, n_comps=2, scale=False):
    """Compute PCA embedding."""
    adata = ad.AnnData(matrix)
    if scale:
        sc.pp.scale(adata, zero_center=True, max_value=10)
    sc.tl.pca(adata, n_comps=min(n_comps, matrix.shape[1] - 1), svd_solver='arpack', random_state=SEED)
    return adata.obsm['X_pca'], adata.uns['pca']['variance_ratio']


def plot_pca(pca_coords, cluster_labels, explained_var, output_path, title):
    """Plot PCA colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    colors = get_cluster_colors(n_clusters)

    for i, cluster in enumerate(sorted(unique_clusters)):
        mask = cluster_labels == cluster
        ax.scatter(
            pca_coords[mask, 0], pca_coords[mask, 1],
            c=colors[i], label=f"Cluster {cluster}",
            s=1.0, alpha=0.5, rasterized=True
        )

    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
    ax.set_title(title)

    if n_clusters <= 20:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), markerscale=5, frameon=True)
    else:
        ax.text(1.02, 0.5, f"{n_clusters} clusters", transform=ax.transAxes, verticalalignment='center')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("Loading cluster results...")
    df_clusters = pd.read_csv(CLUSTER_CSV)
    print(f"Loaded {len(df_clusters):,} cells")

    # Load original data and merge FIRST (before subsampling)
    print("Loading original neighborhood frequencies...")
    df_orig = pd.read_csv(ORIG_CSV)

    # Merge to get both NMF factors and original frequencies
    df_merged = df_orig.merge(
        df_clusters[['fov', 'label', 'leiden_cluster'] + [c for c in df_clusters.columns if c.startswith('NMF_factor_')]],
        on=['fov', 'label'],
        how='inner'
    )
    print(f"Merged {len(df_merged):,} cells")

    # Subsample ONCE from merged data - same cells for all analyses
    print(f"Subsampling to {SUBSAMPLE:,} cells...")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df_merged), size=min(SUBSAMPLE, len(df_merged)), replace=False)
    df_sub = df_merged.iloc[idx].reset_index(drop=True)

    # Get NMF factors and cluster labels
    factor_cols = [c for c in df_sub.columns if c.startswith('NMF_factor_')]
    nmf_matrix = df_sub[factor_cols].values.astype(np.float32)
    cluster_labels = df_sub['leiden_cluster'].values

    # Get original frequency columns
    freq_cols = [c for c in df_sub.columns if c not in ['fov', 'label', 'cell_meta_cluster', 'leiden_cluster'] + factor_cols]
    print(f"NMF factors: {factor_cols}")
    print(f"Frequency columns: {freq_cols}")
    print(f"Number of clusters in subsample: {len(np.unique(cluster_labels))}")

    # =========================================================================
    # Option 1: Z-scored NMF factors
    # =========================================================================
    print("\n" + "="*60)
    print("Option 1: Z-scored NMF factors")
    print("="*60)

    output_dir1 = os.path.join(BASE_OUTPUT, "pca_zscored_nmf")
    os.makedirs(output_dir1, exist_ok=True)

    pca_coords1, exp_var1 = compute_pca(nmf_matrix, n_comps=10, scale=True)
    print(f"PC1: {exp_var1[0]*100:.2f}%, PC2: {exp_var1[1]*100:.2f}%")

    plot_pca(
        pca_coords1[:, :2], cluster_labels, exp_var1,
        os.path.join(output_dir1, "pca_zscored_nmf_clusters.png"),
        "PCA on Z-scored NMF factors"
    )

    # =========================================================================
    # Option 2: Log-transformed NMF factors
    # =========================================================================
    print("\n" + "="*60)
    print("Option 2: Log-transformed NMF factors")
    print("="*60)

    output_dir2 = os.path.join(BASE_OUTPUT, "pca_log_nmf")
    os.makedirs(output_dir2, exist_ok=True)

    # Log transform using log1p (log(1+x)) to handle zeros
    nmf_log = np.log1p(nmf_matrix * 1000)  # Scale up first since values are small
    # Replace any remaining inf/nan with 0
    nmf_log = np.nan_to_num(nmf_log, nan=0.0, posinf=0.0, neginf=0.0)

    pca_coords2, exp_var2 = compute_pca(nmf_log, n_comps=10, scale=False)
    print(f"PC1: {exp_var2[0]*100:.2f}%, PC2: {exp_var2[1]*100:.2f}%")

    plot_pca(
        pca_coords2[:, :2], cluster_labels, exp_var2,
        os.path.join(output_dir2, "pca_log_nmf_clusters.png"),
        "PCA on Log-transformed NMF factors"
    )

    # =========================================================================
    # Option 3: Original neighborhood frequencies
    # =========================================================================
    print("\n" + "="*60)
    print("Option 3: Original neighborhood frequencies")
    print("="*60)

    output_dir3 = os.path.join(BASE_OUTPUT, "pca_original_freqs")
    os.makedirs(output_dir3, exist_ok=True)

    # Use same subsampled data - freq_cols already defined above
    freq_matrix = df_sub[freq_cols].values.astype(np.float32)

    pca_coords3, exp_var3 = compute_pca(freq_matrix, n_comps=10, scale=True)
    print(f"PC1: {exp_var3[0]*100:.2f}%, PC2: {exp_var3[1]*100:.2f}%")

    plot_pca(
        pca_coords3[:, :2], cluster_labels, exp_var3,
        os.path.join(output_dir3, "pca_original_freqs_clusters.png"),
        "PCA on Original Neighborhood Frequencies (Z-scored)"
    )

    print("\n" + "="*60)
    print("All PCA plots generated!")
    print("="*60)
    print(f"Option 1 (Z-scored NMF):     {output_dir1}/")
    print(f"Option 2 (Log NMF):          {output_dir2}/")
    print(f"Option 3 (Original freqs):   {output_dir3}/")


if __name__ == '__main__':
    main()
