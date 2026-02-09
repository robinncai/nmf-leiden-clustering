#!/usr/bin/env python3
"""
Visualization module for NMF + Leiden clustering results.

Generates PCA/UMAP embeddings and diagnostic plots for:
- Cluster visualization
- Batch effect detection
- Per-cluster small multiples
- Spatial scatter plots from neighborhood analysis CSV (centroid-based visualization)

Optional:
- PCA->UMAP on normalized feature columns in the same CSV
  (select features by prefix or explicit column list)
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scanpy as sc
import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')


def compute_umap(
    factor_matrix: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute UMAP embedding from NMF factor loadings.

    Args:
        factor_matrix: NMF W matrix (cells x factors)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        random_state: Random seed
        metric: Distance metric for kNN graph ('cosine' or 'euclidean').
                Default 'cosine' to match clustering pipeline.

    Returns:
        umap_coords: 2D UMAP coordinates (cells x 2)
    """
    logger.info(f"Computing UMAP embedding (NMF space) for {factor_matrix.shape[0]:,} cells...")
    logger.info(f"  Using {metric} distance for kNN graph")

    # Row-normalize if using cosine to match clustering pipeline geometry
    if metric == 'cosine':
        row_sums = factor_matrix.sum(axis=1, keepdims=True)
        W_normalized = factor_matrix / (row_sums + 1e-12)
    else:
        W_normalized = factor_matrix

    adata = ad.AnnData(W_normalized)
    adata.obsm['X_nmf'] = W_normalized

    # Compute neighbors on NMF representation
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep='X_nmf',
        metric=metric,
        random_state=random_state
    )

    # Compute UMAP
    sc.tl.umap(
        adata,
        min_dist=min_dist,
        random_state=random_state
    )

    logger.info(f"UMAP embedding (NMF space, {metric}) complete")
    return adata.obsm['X_umap']


def compute_umap_from_pca(
    feature_matrix: np.ndarray,
    n_comps: int = 50,
    scale: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute UMAP embedding from PCA of (normalized) features.

    This expects a (cells x features) matrix. If `scale=True`, features are z-scored
    before PCA (common for many numeric feature sets).

    Args:
        feature_matrix: (cells x features) matrix (ideally normalized)
        n_comps: number of PCA components
        scale: whether to z-score features before PCA
        n_neighbors: UMAP neighbors
        min_dist: UMAP min_dist
        random_state: seed

    Returns:
        umap_coords: (cells x 2) UMAP coordinates
    """
    logger.info(
        f"Computing PCA->UMAP for {feature_matrix.shape[0]:,} cells "
        f"with {feature_matrix.shape[1]:,} features (n_comps={n_comps}, scale={scale})..."
    )

    adata = ad.AnnData(feature_matrix)

    if scale:
        # z-score per feature; clamp extreme values a bit
        sc.pp.scale(adata, zero_center=True, max_value=10)

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        svd_solver='arpack',
        random_state=random_state
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep='X_pca',
        random_state=random_state
    )

    sc.tl.umap(
        adata,
        min_dist=min_dist,
        random_state=random_state
    )

    logger.info("PCA->UMAP embedding complete")
    return adata.obsm['X_umap']


def compute_pca(
    factor_matrix: np.ndarray,
    n_comps: int = 2,
    scale: bool = False,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA embedding from NMF factor loadings.

    Args:
        factor_matrix: NMF W matrix (cells x factors)
        n_comps: Number of PCA components to compute
        scale: Whether to z-score before PCA
        random_state: Random seed

    Returns:
        pca_coords: PCA coordinates (cells x n_comps)
        explained_variance_ratio: Variance explained by each component
        loadings: PCA loadings (components x features)
    """
    logger.info(f"Computing PCA embedding for {factor_matrix.shape[0]:,} cells...")

    adata = ad.AnnData(factor_matrix)

    if scale:
        sc.pp.scale(adata, zero_center=True, max_value=10)

    sc.tl.pca(
        adata,
        n_comps=min(n_comps, factor_matrix.shape[1] - 1),
        svd_solver='arpack',
        random_state=random_state
    )

    pca_coords = adata.obsm['X_pca']
    explained_variance_ratio = adata.uns['pca']['variance_ratio']
    loadings = adata.varm['PCs']

    logger.info(f"PCA complete: {len(explained_variance_ratio)} components")
    for i in range(min(5, len(explained_variance_ratio))):
        logger.info(f"  PC{i+1}: {explained_variance_ratio[i]*100:.2f}% variance")

    return pca_coords, explained_variance_ratio, loadings


def get_cluster_colors(n_clusters: int) -> List[str]:
    """Generate distinct colors for clusters."""
    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    return [mcolors.rgb2hex(cmap(i / max(n_clusters, 1))) for i in range(n_clusters)]


def plot_umap_clusters(
    umap_coords: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
    title: str = "UMAP colored by Leiden cluster",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 1.0,
    alpha: float = 0.5
) -> None:
    """
    Plot UMAP embedding colored by cluster labels.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    colors = get_cluster_colors(n_clusters)

    for i, cluster in enumerate(sorted(unique_clusters)):
        mask = cluster_labels == cluster
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=colors[i],
            label=f"Cluster {cluster}",
            s=point_size,
            alpha=alpha,
            rasterized=True
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    # Legend with smaller markers
    if n_clusters <= 20:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            markerscale=5,
            frameon=True
        )
    else:
        ax.text(
            1.02, 0.5, f"{n_clusters} clusters",
            transform=ax.transAxes,
            verticalalignment='center'
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_pca_clusters(
    pca_coords: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
    explained_variance_ratio: np.ndarray = None,
    title: str = "PCA colored by Leiden cluster",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 1.0,
    alpha: float = 0.5
) -> None:
    """
    Plot PCA embedding colored by cluster labels.

    Args:
        pca_coords: PCA coordinates (cells x 2)
        cluster_labels: Cluster assignments
        output_path: Path to save figure
        explained_variance_ratio: Variance explained by each PC (for axis labels)
        title: Plot title
        figsize: Figure size
        point_size: Scatter point size
        alpha: Point transparency
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    colors = get_cluster_colors(n_clusters)

    for i, cluster in enumerate(sorted(unique_clusters)):
        mask = cluster_labels == cluster
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=colors[i],
            label=f"Cluster {cluster}",
            s=point_size,
            alpha=alpha,
            rasterized=True
        )

    # Add variance explained to axis labels if provided
    if explained_variance_ratio is not None and len(explained_variance_ratio) >= 2:
        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}% variance)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    ax.set_title(title)

    # Legend with smaller markers
    if n_clusters <= 20:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            markerscale=5,
            frameon=True
        )
    else:
        ax.text(
            1.02, 0.5, f"{n_clusters} clusters",
            transform=ax.transAxes,
            verticalalignment='center'
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_pca_metadata(
    pca_coords: np.ndarray,
    metadata_values: np.ndarray,
    output_path: str,
    explained_variance_ratio: np.ndarray = None,
    title: str = "PCA colored by metadata",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 1.0,
    alpha: float = 0.5,
    categorical: bool = True
) -> None:
    """
    Plot PCA embedding colored by metadata (batch, subtype, etc.).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if categorical:
        unique_values = np.unique(metadata_values)
        n_values = len(unique_values)
        colors = get_cluster_colors(n_values)

        for i, value in enumerate(sorted(unique_values)):
            mask = metadata_values == value
            ax.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=colors[i],
                label=str(value),
                s=point_size,
                alpha=alpha,
                rasterized=True
            )

        if n_values <= 20:
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                markerscale=5,
                frameon=True
            )
        else:
            ax.text(
                1.02, 0.5, f"{n_values} categories",
                transform=ax.transAxes,
                verticalalignment='center'
            )
    else:
        scatter = ax.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            c=metadata_values,
            cmap='viridis',
            s=point_size,
            alpha=alpha,
            rasterized=True
        )
        plt.colorbar(scatter, ax=ax)

    # Add variance explained to axis labels if provided
    if explained_variance_ratio is not None and len(explained_variance_ratio) >= 2:
        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}% variance)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_umap_metadata(
    umap_coords: np.ndarray,
    metadata_values: np.ndarray,
    output_path: str,
    title: str = "UMAP colored by metadata",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 1.0,
    alpha: float = 0.5,
    categorical: bool = True
) -> None:
    """
    Plot UMAP embedding colored by metadata (batch, subtype, etc.).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if categorical:
        unique_values = np.unique(metadata_values)
        n_values = len(unique_values)
        colors = get_cluster_colors(n_values)

        for i, value in enumerate(sorted(unique_values)):
            mask = metadata_values == value
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=colors[i],
                label=str(value),
                s=point_size,
                alpha=alpha,
                rasterized=True
            )

        if n_values <= 20:
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                markerscale=5,
                frameon=True
            )
        else:
            ax.text(
                1.02, 0.5, f"{n_values} categories",
                transform=ax.transAxes,
                verticalalignment='center'
            )
    else:
        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=metadata_values,
            cmap='viridis',
            s=point_size,
            alpha=alpha,
            rasterized=True
        )
        plt.colorbar(scatter, ax=ax)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_small_multiples(
    umap_coords: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
    max_clusters: int = 30,
    cols: int = 5,
    figsize_per_panel: Tuple[float, float] = (3, 3),
    point_size: float = 0.5,
    highlight_alpha: float = 0.8,
    background_alpha: float = 0.05
) -> None:
    """
    Plot small multiples with one panel per cluster.
    Each panel highlights cells from one cluster against a gray background.
    """
    unique_clusters = sorted(np.unique(cluster_labels))
    n_clusters = min(len(unique_clusters), max_clusters)

    if n_clusters < len(unique_clusters):
        logger.warning(f"Showing only first {max_clusters} of {len(unique_clusters)} clusters")

    rows = (n_clusters + cols - 1) // cols
    fig_width = cols * figsize_per_panel[0]
    fig_height = rows * figsize_per_panel[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes)

    colors = get_cluster_colors(n_clusters)

    for idx, cluster in enumerate(unique_clusters[:n_clusters]):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        mask = cluster_labels == cluster
        n_cells = int(mask.sum())

        # Background
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c='lightgray',
            s=point_size,
            alpha=background_alpha,
            rasterized=True
        )

        # Highlight cluster
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=colors[idx],
            s=point_size * 2,
            alpha=highlight_alpha,
            rasterized=True
        )

        ax.set_title(f"Cluster {cluster}\n(n={n_cells:,})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    # Hide empty panels
    for idx in range(n_clusters, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle("Small Multiples: One Panel per Cluster", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_cluster_celltype_heatmap(
    cluster_df: pd.DataFrame,
    output_path: str,
    cluster_col: str = 'leiden_cluster',
    celltype_col: str = 'cell_meta_cluster',
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis',
    annot: bool = True,
    fmt: str = '.2f'
) -> None:
    """
    Create a heatmap showing average cell type frequency for each cluster.

    Args:
        cluster_df: DataFrame with cluster and cell type columns
        output_path: Path to save the heatmap
        cluster_col: Column name for cluster labels
        celltype_col: Column name for cell type labels
        figsize: Figure size (width, height)
        cmap: Colormap for heatmap
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
    """
    logger.info(f"Creating cluster-celltype composition heatmap...")

    # Calculate cell type frequencies for each cluster
    # Create a crosstab showing counts
    composition = pd.crosstab(
        cluster_df[cluster_col],
        cluster_df[celltype_col],
        normalize='index'  # Normalize by row (cluster) to get frequencies
    )

    # Sort clusters and cell types for better visualization
    composition = composition.sort_index(axis=0)  # Sort clusters
    composition = composition.sort_index(axis=1)  # Sort cell types

    logger.info(f"Composition matrix shape: {composition.shape[0]} clusters x {composition.shape[1]} cell types")

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Determine annotation based on matrix size
    if composition.shape[0] * composition.shape[1] > 500:
        annot_flag = False
        logger.info("Large matrix detected, disabling cell annotations")
    else:
        annot_flag = annot

    sns.heatmap(
        composition,
        cmap=cmap,
        annot=annot_flag,
        fmt=fmt,
        cbar_kws={'label': 'Frequency'},
        ax=ax,
        linewidths=0.5 if composition.shape[0] < 30 else 0,
        linecolor='white' if composition.shape[0] < 30 else None
    )

    ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Cell Type Composition by Cluster', fontsize=14, fontweight='bold', pad=20)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved cluster-celltype heatmap: {output_path}")

    # Also save the composition matrix as CSV
    csv_path = output_path.replace('.png', '_composition.csv')
    composition.to_csv(csv_path)
    logger.info(f"Saved composition matrix: {csv_path}")


# ============================================================================
# SPATIAL SCATTER VISUALIZATION (Centroid-based)
# ============================================================================

def get_category_colors_dict(categories) -> dict:
    """Generate maximally distinct colors for categories as a dictionary."""
    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)

    # Curated list of 30 distinct, bold colors (no pastels/light colors)
    distinct_colors = [
        '#e41a1c',  # red
        '#377eb8',  # blue
        '#4daf4a',  # green
        '#984ea3',  # purple
        '#ff7f00',  # orange
        '#a65628',  # brown
        '#f781bf',  # pink
        '#666666',  # grey
        '#ffff33',  # yellow (kept bright for visibility)
        '#00ced1',  # dark turquoise
        '#dc143c',  # crimson
        '#00008b',  # dark blue
        '#006400',  # dark green
        '#8b008b',  # dark magenta
        '#ff4500',  # orange red
        '#2f4f4f',  # dark slate grey
        '#8b4513',  # saddle brown
        '#483d8b',  # dark slate blue
        '#b8860b',  # dark goldenrod
        '#008080',  # teal
        '#9400d3',  # dark violet
        '#ff1493',  # deep pink
        '#00fa9a',  # medium spring green
        '#ffd700',  # gold
        '#1e90ff',  # dodger blue
        '#fa8072',  # salmon
        '#7b68ee',  # medium slate blue
        '#32cd32',  # lime green
        '#ff6347',  # tomato
        '#4682b4',  # steel blue
    ]

    if n_cats <= len(distinct_colors):
        colors = distinct_colors[:n_cats]
    else:
        # If we need more colors, cycle through and add variations
        colors = []
        for i in range(n_cats):
            base_color = distinct_colors[i % len(distinct_colors)]
            if i >= len(distinct_colors):
                # Darken the color for additional cycles
                rgb = mcolors.hex2color(base_color)
                factor = 0.7 - 0.1 * (i // len(distinct_colors))
                factor = max(factor, 0.4)
                rgb = tuple(c * factor for c in rgb)
                colors.append(mcolors.rgb2hex(rgb))
            else:
                colors.append(base_color)

    return {cat: colors[i] for i, cat in enumerate(unique_cats)}


def subsample_fovs_per_neighborhood(
    df: pd.DataFrame,
    n_fovs_per_nh: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Subsample FOVs: select n_fovs_per_nh FOVs per kmeans_neighborhood within each batch.

    Args:
        df: DataFrame with fov, batch, kmeans_neighborhood columns
        n_fovs_per_nh: Number of FOVs to sample per neighborhood per batch
        random_state: Random seed

    Returns:
        Subsampled DataFrame
    """
    rng = np.random.default_rng(random_state)

    selected_fovs = set()

    for batch in df['batch'].unique():
        batch_df = df[df['batch'] == batch]

        for nh in batch_df['kmeans_neighborhood'].unique():
            nh_df = batch_df[batch_df['kmeans_neighborhood'] == nh]
            unique_fovs = nh_df['fov'].unique()

            n_to_sample = min(n_fovs_per_nh, len(unique_fovs))
            sampled = rng.choice(unique_fovs, size=n_to_sample, replace=False)
            selected_fovs.update(sampled)

    subsampled = df[df['fov'].isin(selected_fovs)].copy()
    logger.info(f"Subsampled {len(selected_fovs)} FOVs from {df['fov'].nunique()} total FOVs")

    return subsampled


def plot_spatial_fov(
    fov_df: pd.DataFrame,
    color_by: str,
    color_map: dict,
    output_path: str,
    fov_name: str,
    figsize: Tuple[int, int] = (10, 10),
    size_scale: float = 0.05,
    alpha: float = 0.7
) -> None:
    """
    Plot spatial scatter for a single FOV.

    Args:
        fov_df: DataFrame for single FOV
        color_by: Column name to color by
        color_map: Dict mapping categories to colors
        output_path: Path to save figure
        fov_name: FOV identifier for title
        figsize: Figure size
        size_scale: Scale factor for cell sizes
        alpha: Point transparency
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get coordinates and sizes
    x = fov_df['centroid-0'].values
    y = fov_df['centroid-1'].values
    sizes = fov_df['cell_size'].values * size_scale

    # Plot each category
    categories = fov_df[color_by].unique()
    for cat in sorted(categories):
        mask = fov_df[color_by] == cat
        color = color_map.get(cat, 'gray')
        ax.scatter(
            x[mask],
            y[mask],
            s=sizes[mask],
            c=color,
            label=str(cat),
            alpha=alpha,
            edgecolors='none',
            rasterized=True
        )

    ax.set_xlabel("Centroid X")
    ax.set_ylabel("Centroid Y")
    ax.set_title(f"FOV: {fov_name}\nColored by {color_by}")
    ax.set_aspect('equal')

    # Legend
    n_cats = len(categories)
    if n_cats <= 20:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            markerscale=2,
            frameon=True,
            fontsize=8
        )
    else:
        ax.text(
            1.02, 0.5, f"{n_cats} categories",
            transform=ax.transAxes,
            verticalalignment='center'
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_spatial_fov_comparison(
    fov_df: pd.DataFrame,
    cell_type_colors: dict,
    nh_colors: dict,
    output_path: str,
    fov_name: str,
    figsize: Tuple[float, float] = (20, 10),
    size_scale: float = 0.05,
    alpha: float = 0.7
) -> None:
    """
    Plot side-by-side spatial scatter for a single FOV.
    Left: colored by cell_meta_cluster, Right: colored by kmeans_neighborhood.

    Args:
        fov_df: DataFrame for single FOV
        cell_type_colors: Dict mapping cell_meta_cluster to colors
        nh_colors: Dict mapping kmeans_neighborhood to colors
        output_path: Path to save figure
        fov_name: FOV identifier for title
        figsize: Figure size (width, height)
        size_scale: Scale factor for cell sizes
        alpha: Point transparency
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x = fov_df['centroid-0'].values
    y = fov_df['centroid-1'].values
    sizes = fov_df['cell_size'].values * size_scale

    # Left panel: cell_meta_cluster
    ax_left = axes[0]
    for cat in sorted(fov_df['cell_meta_cluster'].unique()):
        mask = fov_df['cell_meta_cluster'] == cat
        color = cell_type_colors.get(cat, 'gray')
        ax_left.scatter(
            x[mask], y[mask], s=sizes[mask], c=color,
            label=str(cat), alpha=alpha, edgecolors='none', rasterized=True
        )
    ax_left.set_xlabel("Centroid X")
    ax_left.set_ylabel("Centroid Y")
    ax_left.set_title(f"cell_meta_cluster")
    ax_left.set_aspect('equal')

    n_cell_cats = fov_df['cell_meta_cluster'].nunique()
    if n_cell_cats <= 20:
        ax_left.legend(loc='upper right', fontsize=6, markerscale=1.5)

    # Right panel: kmeans_neighborhood
    ax_right = axes[1]
    for cat in sorted(fov_df['kmeans_neighborhood'].unique()):
        mask = fov_df['kmeans_neighborhood'] == cat
        color = nh_colors.get(cat, 'gray')
        ax_right.scatter(
            x[mask], y[mask], s=sizes[mask], c=color,
            label=str(cat), alpha=alpha, edgecolors='none', rasterized=True
        )
    ax_right.set_xlabel("Centroid X")
    ax_right.set_ylabel("Centroid Y")
    ax_right.set_title(f"kmeans_neighborhood")
    ax_right.set_aspect('equal')

    n_nh_cats = fov_df['kmeans_neighborhood'].nunique()
    if n_nh_cats <= 20:
        ax_right.legend(loc='upper right', fontsize=6, markerscale=1.5)

    plt.suptitle(f"FOV: {fov_name} (n={len(fov_df):,} cells)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_spatial_grid(
    df: pd.DataFrame,
    color_by: str,
    color_map: dict,
    output_path: str,
    fovs: List[str],
    cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5, 5),
    size_scale: float = 0.03,
    alpha: float = 0.7
) -> None:
    """
    Plot spatial scatter grid for multiple FOVs.

    Args:
        df: DataFrame with all data
        color_by: Column name to color by
        color_map: Dict mapping categories to colors
        output_path: Path to save figure
        fovs: List of FOV names to plot
        cols: Number of columns in grid
        figsize_per_panel: Size per panel
        size_scale: Scale factor for cell sizes
        alpha: Point transparency
    """
    n_fovs = len(fovs)
    rows = (n_fovs + cols - 1) // cols

    fig_width = cols * figsize_per_panel[0]
    fig_height = rows * figsize_per_panel[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, fov in enumerate(fovs):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        fov_df = df[df['fov'] == fov]

        x = fov_df['centroid-0'].values
        y = fov_df['centroid-1'].values
        sizes = fov_df['cell_size'].values * size_scale

        for cat in sorted(df[color_by].unique()):
            mask = fov_df[color_by] == cat
            if mask.sum() == 0:
                continue
            color = color_map.get(cat, 'gray')
            ax.scatter(
                x[mask],
                y[mask],
                s=sizes[mask],
                c=color,
                alpha=alpha,
                edgecolors='none',
                rasterized=True
            )

        ax.set_title(f"{fov}\n(n={len(fov_df):,})", fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty panels
    for idx in range(n_fovs, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    # Add legend
    handles = []
    labels = []
    for cat in sorted(df[color_by].unique()):
        color = color_map.get(cat, 'gray')
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=color, markersize=8))
        labels.append(str(cat))

    n_cats = len(labels)
    if n_cats <= 20:
        fig.legend(handles, labels, loc='center right',
                   bbox_to_anchor=(1.15, 0.5), fontsize=8)

    plt.suptitle(f"Spatial plots colored by {color_by}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_spatial_grid_comparison(
    df: pd.DataFrame,
    cell_type_colors: dict,
    nh_colors: dict,
    output_path: str,
    fovs: List[str],
    figsize_per_panel: Tuple[float, float] = (4, 4),
    size_scale: float = 0.03,
    alpha: float = 0.7
) -> None:
    """
    Plot side-by-side spatial scatter grid for multiple FOVs.
    Each row shows one FOV with cell_meta_cluster (left) and kmeans_neighborhood (right).

    Args:
        df: DataFrame with all data
        cell_type_colors: Dict mapping cell_meta_cluster to colors
        nh_colors: Dict mapping kmeans_neighborhood to colors
        output_path: Path to save figure
        fovs: List of FOV names to plot
        figsize_per_panel: Size per panel
        size_scale: Scale factor for cell sizes
        alpha: Point transparency
    """
    n_fovs = len(fovs)
    rows = n_fovs
    cols = 2  # Two columns: cell_meta_cluster and kmeans_neighborhood

    fig_width = cols * figsize_per_panel[0] + 3  # Extra space for legends
    fig_height = rows * figsize_per_panel[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, fov in enumerate(fovs):
        fov_df = df[df['fov'] == fov]

        x = fov_df['centroid-0'].values
        y = fov_df['centroid-1'].values
        sizes = fov_df['cell_size'].values * size_scale

        # Left column: cell_meta_cluster
        ax_left = axes[idx, 0]
        for cat in sorted(df['cell_meta_cluster'].unique()):
            mask = fov_df['cell_meta_cluster'] == cat
            if mask.sum() == 0:
                continue
            color = cell_type_colors.get(cat, 'gray')
            ax_left.scatter(
                x[mask], y[mask], s=sizes[mask], c=color,
                alpha=alpha, edgecolors='none', rasterized=True
            )
        ax_left.set_aspect('equal')
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_left.set_ylabel(f"{fov}\n(n={len(fov_df):,})", fontsize=8)

        # Right column: kmeans_neighborhood
        ax_right = axes[idx, 1]
        for cat in sorted(df['kmeans_neighborhood'].unique()):
            mask = fov_df['kmeans_neighborhood'] == cat
            if mask.sum() == 0:
                continue
            color = nh_colors.get(cat, 'gray')
            ax_right.scatter(
                x[mask], y[mask], s=sizes[mask], c=color,
                alpha=alpha, edgecolors='none', rasterized=True
            )
        ax_right.set_aspect('equal')
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    # Column titles
    axes[0, 0].set_title("cell_meta_cluster", fontsize=10, fontweight='bold')
    axes[0, 1].set_title("kmeans_neighborhood", fontsize=10, fontweight='bold')

    # Add legends on the right side
    # Cell type legend
    handles_ct = []
    labels_ct = []
    for cat in sorted(df['cell_meta_cluster'].unique()):
        color = cell_type_colors.get(cat, 'gray')
        handles_ct.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, markersize=6))
        labels_ct.append(str(cat))

    # Neighborhood legend
    handles_nh = []
    labels_nh = []
    for cat in sorted(df['kmeans_neighborhood'].unique()):
        color = nh_colors.get(cat, 'gray')
        handles_nh.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, markersize=6))
        labels_nh.append(str(cat))

    # Add legends if not too many categories
    if len(labels_ct) <= 20:
        fig.legend(handles_ct, labels_ct, loc='upper right',
                   bbox_to_anchor=(1.0, 0.95), fontsize=6,
                   title='cell_meta_cluster', title_fontsize=7)
    if len(labels_nh) <= 20:
        fig.legend(handles_nh, labels_nh, loc='lower right',
                   bbox_to_anchor=(1.0, 0.05), fontsize=6,
                   title='kmeans_neighborhood', title_fontsize=7)

    plt.suptitle("Spatial Comparison: cell_meta_cluster vs kmeans_neighborhood",
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def run_spatial_scatter_visualization(
    csv_path: str,
    output_dir: str = 'spatial_plots',
    n_fovs_per_nh: int = 3,
    size_scale: float = 0.05,
    random_state: int = 42
) -> None:
    """
    Run spatial scatter visualization pipeline.

    Reads a CSV with cell neighborhood data and generates scatter plots
    using centroid coordinates, with circle sizes proportional to cell_size.

    Args:
        csv_path: Path to harmonized_level12_kmeans_nh.csv
        output_dir: Directory for output plots
        n_fovs_per_nh: Number of FOVs to sample per neighborhood per batch
        size_scale: Scale factor for cell sizes
        random_state: Random seed
    """
    logger.info("=" * 60)
    logger.info("Spatial Scatter Visualization Pipeline")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f"\n[Step 1] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} cells from {df['fov'].nunique()} FOVs")

    # Validate required columns
    required_cols = ['cell_size', 'centroid-0', 'centroid-1', 'fov',
                     'cell_meta_cluster', 'batch', 'kmeans_neighborhood']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Subsample FOVs
    logger.info(f"\n[Step 2] Subsampling {n_fovs_per_nh} FOVs per neighborhood per batch...")
    df_sub = subsample_fovs_per_neighborhood(df, n_fovs_per_nh, random_state)

    # Generate color maps
    logger.info("\n[Step 3] Generating color maps...")
    cell_type_colors = get_category_colors_dict(df['cell_meta_cluster'])
    nh_colors = get_category_colors_dict(df['kmeans_neighborhood'])

    # Get list of FOVs to plot
    fovs = sorted(df_sub['fov'].unique())
    logger.info(f"Will generate plots for {len(fovs)} FOVs")

    # Plot 1: Side-by-side comparison grid
    logger.info("\n[Step 4] Generating side-by-side comparison grid plot...")
    plot_spatial_grid_comparison(
        df_sub,
        cell_type_colors=cell_type_colors,
        nh_colors=nh_colors,
        output_path=os.path.join(output_dir, 'spatial_grid_comparison.png'),
        fovs=fovs,
        size_scale=size_scale
    )

    # Individual FOV comparison plots (side-by-side)
    logger.info("\n[Step 5] Generating individual FOV comparison plots...")
    fov_dir_comparison = os.path.join(output_dir, 'comparison_plots')
    os.makedirs(fov_dir_comparison, exist_ok=True)

    for fov in fovs:
        fov_df = df_sub[df_sub['fov'] == fov]
        safe_fov = fov.replace('/', '_').replace(' ', '_')

        plot_spatial_fov_comparison(
            fov_df,
            cell_type_colors=cell_type_colors,
            nh_colors=nh_colors,
            output_path=os.path.join(fov_dir_comparison, f'{safe_fov}_comparison.png'),
            fov_name=fov,
            size_scale=size_scale
        )

    logger.info(f"\nGenerated {len(fovs)} individual FOV comparison plots")

    # Save subsampled data
    sub_path = os.path.join(output_dir, 'subsampled_data.csv')
    df_sub.to_csv(sub_path, index=False)
    logger.info(f"Saved subsampled data to {sub_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Spatial scatter visualization complete!")
    logger.info(f"Plots saved to: {output_dir}/")
    logger.info("  - spatial_grid_comparison.png (side-by-side grid)")
    logger.info(f"  - comparison_plots/ ({len(fovs)} individual FOV comparisons)")
    logger.info("=" * 60)


def run_visualization(
    cluster_results_path: str,
    metadata_path: Optional[str] = None,
    output_dir: str = 'plots',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    point_size: float = 1.0,
    subsample: Optional[int] = None,
    random_state: int = 42,
    do_pca_umap: bool = False,
    pca_feature_prefix: Optional[str] = None,
    pca_feature_cols: Optional[List[str]] = None,
    pca_n_comps: int = 50,
    pca_scale: bool = True,
    # Distance metric for UMAP
    umap_metric: str = 'cosine',
    # UMAP visualization (optional, disabled by default)
    do_umap: bool = False,
    # PCA visualization (direct PCA, not PCA->UMAP)
    do_pca: bool = True,
    cell_type_col: str = 'cell_meta_cluster'
) -> None:
    """
    Run full visualization pipeline.

    Parameters:
        umap_metric: Distance metric for UMAP kNN graph ('cosine' or 'euclidean').
                     Default 'cosine' to match the clustering pipeline.
        do_umap: Generate UMAP visualization (default False, use --umap to enable)
        do_pca: Generate direct PCA visualization (2D PCA scatter plots)
        cell_type_col: Column name for cell type annotations (default: cell_meta_cluster)
    """
    logger.info("=" * 60)
    logger.info("Visualization Pipeline")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    basename = Path(cluster_results_path).stem.replace('_nmf_leiden_clusters', '')

    # Load data
    logger.info("\n[Step 1] Loading data...")
    df = pd.read_csv(cluster_results_path)

    # Merge metadata if provided
    if metadata_path:
        logger.info("\n[Step 2] Merging metadata...")
        metadata_df = pd.read_csv(metadata_path)
        df = df.merge(
            metadata_df[['fov', 'label', 'batch', 'Subtype']],
            on=['fov', 'label'],
            how='left'
        )
        has_metadata = True
        n_matched = df['batch'].notna().sum()
        logger.info(f"Matched {n_matched:,} / {len(df):,} cells")
    else:
        has_metadata = False

    # Extract NMF factor columns
    factor_cols = [c for c in df.columns if c.startswith('NMF_factor_')]
    if len(factor_cols) == 0:
        raise ValueError("No NMF factor columns found (expected columns starting with 'NMF_factor_').")

    factor_matrix_all = df[factor_cols].values.astype(np.float32)
    cluster_labels_all = df['leiden_cluster'].values

    logger.info(f"Loaded {len(df):,} cells with {len(factor_cols)} NMF factors")

    # Subsample if requested (applies to BOTH NMF and PCA UMAP)
    if subsample and subsample < len(df):
        logger.info(f"\n[Step 3] Subsampling to {subsample:,} cells...")
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(df), size=subsample, replace=False)

        df = df.iloc[idx].reset_index(drop=True)
        factor_matrix = factor_matrix_all[idx]
        cluster_labels = cluster_labels_all[idx]
    else:
        logger.info("\n[Step 3] Using all cells (no subsampling)")
        factor_matrix = factor_matrix_all
        cluster_labels = cluster_labels_all

    # ---- NMF UMAP (optional) ----
    umap_coords = None
    if do_umap:
        logger.info("\n[Step 4] Computing UMAP embedding (NMF space)...")
        umap_coords = compute_umap(
            factor_matrix,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric=umap_metric
        )

        # Save NMF UMAP coordinates (full df with added columns)
        df['UMAP_1'] = umap_coords[:, 0]
        df['UMAP_2'] = umap_coords[:, 1]
        umap_path = os.path.join(output_dir, f'{basename}_umap_coords.csv')
        df.to_csv(umap_path, index=False)
        logger.info(f"Saved NMF-space UMAP coordinates to {umap_path}")

        logger.info("\n[Step 5] Generating NMF-space plots...")
        plot_umap_clusters(
            umap_coords,
            cluster_labels,
            os.path.join(output_dir, f'{basename}_umap_clusters.png'),
            title=f"UMAP (NMF space, {umap_metric}) colored by Leiden cluster",
            point_size=point_size
        )

        # Check number of clusters before generating small multiples overlay
        n_clusters = len(np.unique(cluster_labels))
        if n_clusters > 50:
            logger.info(f"Skipping small multiples overlay: {n_clusters} clusters exceeds maximum of 50")
        else:
            plot_small_multiples(
                umap_coords,
                cluster_labels,
                os.path.join(output_dir, f'{basename}_small_multiples.png'),
                point_size=point_size * 0.5
            )
    else:
        logger.info("\n[Step 4] Skipping UMAP (--no-umap specified)")

    # ---- Direct PCA visualization ----
    if do_pca:
        logger.info("\n[Step 5.1] Computing and plotting PCA (NMF space)...")
        pca_coords, explained_var, pca_loadings = compute_pca(
            factor_matrix,
            n_comps=min(10, factor_matrix.shape[1] - 1),
            scale=False,
            random_state=random_state
        )

        # Save PCA coordinates
        df['PCA_1'] = pca_coords[:, 0]
        df['PCA_2'] = pca_coords[:, 1]
        pca_path = os.path.join(output_dir, f'{basename}_pca_coords.csv')
        df.to_csv(pca_path, index=False)
        logger.info(f"Saved PCA coordinates to {pca_path}")

        # Plot PCA colored by cluster
        plot_pca_clusters(
            pca_coords[:, :2],
            cluster_labels,
            os.path.join(output_dir, f'{basename}_pca_clusters.png'),
            explained_variance_ratio=explained_var,
            title="PCA (NMF space) colored by Leiden cluster",
            point_size=point_size
        )

        # Plot PCA colored by metadata if available
        if has_metadata:
            if 'batch' in df.columns and df['batch'].notna().any():
                plot_pca_metadata(
                    pca_coords[:, :2],
                    df['batch'].fillna('Unknown').values,
                    os.path.join(output_dir, f'{basename}_pca_batch.png'),
                    explained_variance_ratio=explained_var,
                    title="PCA (NMF space) colored by Batch",
                    point_size=point_size
                )

            if 'Subtype' in df.columns and df['Subtype'].notna().any():
                plot_pca_metadata(
                    pca_coords[:, :2],
                    df['Subtype'].fillna('Unknown').values,
                    os.path.join(output_dir, f'{basename}_pca_subtype.png'),
                    explained_variance_ratio=explained_var,
                    title="PCA (NMF space) colored by Subtype",
                    point_size=point_size
                )

    if has_metadata and do_umap and umap_coords is not None:
        if 'batch' in df.columns and df['batch'].notna().any():
            plot_umap_metadata(
                umap_coords,
                df['batch'].fillna('Unknown').values,
                os.path.join(output_dir, f'{basename}_umap_batch.png'),
                title="UMAP (NMF space) colored by Batch",
                point_size=point_size
            )

        if 'Subtype' in df.columns and df['Subtype'].notna().any():
            plot_umap_metadata(
                umap_coords,
                df['Subtype'].fillna('Unknown').values,
                os.path.join(output_dir, f'{basename}_umap_subtype.png'),
                title="UMAP (NMF space) colored by Subtype",
                point_size=point_size
            )

    # ---- Cluster-Celltype Composition Heatmap ----
    logger.info("\n[Step 5.5] Generating cluster-celltype composition heatmap...")
    if cell_type_col in df.columns:
        # Calculate appropriate figure size based on number of clusters and cell types
        n_clusters = df['leiden_cluster'].nunique()
        n_celltypes = df[cell_type_col].nunique()
        figsize_width = max(12, min(20, n_celltypes * 0.6))
        figsize_height = max(8, min(16, n_clusters * 0.4))

        plot_cluster_celltype_heatmap(
            cluster_df=df,
            output_path=os.path.join(output_dir, f'{basename}_cluster_celltype_heatmap.png'),
            cluster_col='leiden_cluster',
            celltype_col=cell_type_col,
            figsize=(figsize_width, figsize_height),
            cmap='viridis',
            annot=True,
            fmt='.2f'
        )
    else:
        logger.warning(f"Cell type column '{cell_type_col}' not found in data. Skipping heatmap.")

    # ---- PCA UMAP (optional) ----
    if do_pca_umap:
        logger.info("\n[Step 6] Computing PCA-based UMAP embedding...")

        # Determine feature columns for PCA
        if pca_feature_cols is not None and len(pca_feature_cols) > 0:
            feature_cols = pca_feature_cols
        elif pca_feature_prefix is not None:
            feature_cols = [c for c in df.columns if c.startswith(pca_feature_prefix)]
        else:
            raise ValueError(
                "PCA UMAP requested, but no feature columns specified. "
                "Use --pca-feature-prefix or --pca-feature-cols."
            )

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            preview = missing[:10]
            raise ValueError(
                f"Missing PCA feature columns in CSV (showing up to 10): {preview}"
                + (" ..." if len(missing) > 10 else "")
            )

        if len(feature_cols) < 2:
            raise ValueError(
                f"Need at least 2 feature columns for PCA UMAP, found {len(feature_cols)}."
            )

        logger.info(f"Using {len(feature_cols):,} feature columns for PCA UMAP")

        feature_matrix = df[feature_cols].values.astype(np.float32)

        umap_pca_coords = compute_umap_from_pca(
            feature_matrix,
            n_comps=pca_n_comps,
            scale=pca_scale,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )

        df['UMAP_PCA_1'] = umap_pca_coords[:, 0]
        df['UMAP_PCA_2'] = umap_pca_coords[:, 1]

        umap_pca_path = os.path.join(output_dir, f'{basename}_umap_pca_coords.csv')
        df[['fov', 'label', 'leiden_cluster', 'UMAP_PCA_1', 'UMAP_PCA_2']].to_csv(umap_pca_path, index=False)
        logger.info(f"Saved PCA-space UMAP coordinates to {umap_pca_path}")

        logger.info("\n[Step 7] Generating PCA-space plots...")
        plot_umap_clusters(
            umap_pca_coords,
            cluster_labels,
            os.path.join(output_dir, f'{basename}_umap_pca_clusters.png'),
            title="UMAP (PCA space) colored by Leiden cluster",
            point_size=point_size
        )

        if has_metadata:
            if 'batch' in df.columns and df['batch'].notna().any():
                plot_umap_metadata(
                    umap_pca_coords,
                    df['batch'].fillna('Unknown').values,
                    os.path.join(output_dir, f'{basename}_umap_pca_batch.png'),
                    title="UMAP (PCA space) colored by Batch",
                    point_size=point_size
                )

            if 'Subtype' in df.columns and df['Subtype'].notna().any():
                plot_umap_metadata(
                    umap_pca_coords,
                    df['Subtype'].fillna('Unknown').values,
                    os.path.join(output_dir, f'{basename}_umap_pca_subtype.png'),
                    title="UMAP (PCA space) colored by Subtype",
                    point_size=point_size
                )

    logger.info("\n" + "=" * 60)
    logger.info("Visualization complete!")
    logger.info(f"Plots saved to: {output_dir}/")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate UMAP visualizations for NMF + Leiden clustering results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'cluster_results',
        nargs='?',
        default=None,
        help='Path to *_nmf_leiden_clusters.csv file (not required if using --spatial-scatter)'
    )

    parser.add_argument(
        '-m', '--metadata',
        help='Path to metadata CSV (with batch, Subtype columns)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        default='plots',
        help='Output directory for plots'
    )

    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter'
    )

    parser.add_argument(
        '--min-dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter'
    )

    parser.add_argument(
        '--point-size',
        type=float,
        default=1.0,
        help='Size of scatter points'
    )

    parser.add_argument(
        '--subsample',
        type=int,
        default=None,
        help='Subsample to N cells for faster plotting'
    )

    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--umap-metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean'],
        help='Distance metric for UMAP kNN graph. Default: cosine (matches clustering pipeline)'
    )

    # ---- UMAP visualization (optional) ----
    parser.add_argument(
        '--umap',
        action='store_true',
        help='Generate UMAP visualization (disabled by default, PCA is used instead)'
    )

    # ---- PCA visualization ----
    parser.add_argument(
        '--pca',
        action='store_true',
        default=True,
        help='Generate direct PCA visualization (2D scatter plots of PC1 vs PC2). Enabled by default.'
    )
    parser.add_argument(
        '--no-pca',
        action='store_true',
        help='Disable PCA visualization'
    )

    # ---- PCA-UMAP options ----
    parser.add_argument(
        '--pca-umap',
        action='store_true',
        help='Also compute an additional UMAP from PCA of normalized feature columns'
    )

    parser.add_argument(
        '--pca-feature-prefix',
        default=None,
        help='Prefix to select feature columns for PCA (e.g., "norm_" selects all columns starting with norm_)'
    )

    parser.add_argument(
        '--pca-feature-cols',
        default=None,
        help='Comma-separated list of feature columns for PCA (alternative to --pca-feature-prefix)'
    )

    parser.add_argument(
        '--pca-n-comps',
        type=int,
        default=50,
        help='Number of PCA components for PCA-based UMAP'
    )

    parser.add_argument(
        '--pca-scale',
        dest='pca_scale',
        action='store_true',
        help='Z-score features before PCA (default)'
    )
    parser.add_argument(
        '--no-pca-scale',
        dest='pca_scale',
        action='store_false',
        help='Do not z-score features before PCA'
    )
    parser.set_defaults(pca_scale=True)

    parser.add_argument(
        '--cell-type-col',
        default='cell_meta_cluster',
        help='Column name for cell type annotations (default: cell_meta_cluster)'
    )

    # ---- Spatial scatter visualization options ----
    parser.add_argument(
        '--spatial-scatter',
        action='store_true',
        help='Run spatial scatter visualization from neighborhood CSV (standalone mode)'
    )

    parser.add_argument(
        '--spatial-scatter-csv',
        default=None,
        help='Path to neighborhood CSV (e.g., harmonized_level12_kmeans_nh.csv) for spatial scatter plots'
    )

    parser.add_argument(
        '--n-fovs-per-nh',
        type=int,
        default=3,
        help='Number of FOVs to sample per kmeans_neighborhood per batch'
    )

    parser.add_argument(
        '--size-scale',
        type=float,
        default=0.05,
        help='Scale factor for cell sizes in spatial scatter plots'
    )

    args = parser.parse_args()

    # Handle spatial scatter mode (standalone)
    if args.spatial_scatter:
        if args.spatial_scatter_csv is None:
            logger.error("--spatial-scatter-csv is required when using --spatial-scatter")
            return
        if not os.path.exists(args.spatial_scatter_csv):
            logger.error(f"Spatial scatter CSV not found: {args.spatial_scatter_csv}")
            return

        run_spatial_scatter_visualization(
            csv_path=args.spatial_scatter_csv,
            output_dir=args.output_dir,
            n_fovs_per_nh=args.n_fovs_per_nh,
            size_scale=args.size_scale,
            random_state=args.seed
        )
        return

    # For non-spatial-scatter mode, cluster_results is required
    if args.cluster_results is None:
        logger.error("cluster_results is required unless using --spatial-scatter mode")
        return

    if not os.path.exists(args.cluster_results):
        logger.error(f"Cluster results file not found: {args.cluster_results}")
        return

    pca_cols: Optional[List[str]] = None
    if args.pca_feature_cols:
        pca_cols = [c.strip() for c in args.pca_feature_cols.split(',') if c.strip()]

    run_visualization(
        cluster_results_path=args.cluster_results,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        point_size=args.point_size,
        subsample=args.subsample,
        random_state=args.seed,
        do_pca_umap=args.pca_umap,
        pca_feature_prefix=args.pca_feature_prefix,
        pca_feature_cols=pca_cols,
        pca_n_comps=args.pca_n_comps,
        pca_scale=args.pca_scale,
        umap_metric=args.umap_metric,
        do_umap=args.umap,
        do_pca=(args.pca and not args.no_pca),
        cell_type_col=args.cell_type_col
    )


if __name__ == '__main__':
    main()