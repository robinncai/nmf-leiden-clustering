#!/usr/bin/env python3
"""
Visualization module for NMF + Leiden clustering results.

Generates UMAP embeddings and diagnostic plots for:
- Cluster visualization
- Batch effect detection
- Per-cluster small multiples

NEW:
- Optional PCA->UMAP on normalized feature columns in the same CSV
  (select features by prefix or explicit column list)
- Optional spatial overlay visualizations (cell type vs cluster assignments)
- Spatial scatter plots from neighborhood analysis CSV (centroid-based visualization)

Dependencies:
- ark-analysis v0.7.2 (for spatial overlays)
  Install: git clone -b v0.7.2 https://github.com/angelolab/ark-analysis.git
           cd ark-analysis && conda env create -f environment.yml
"""

import argparse
import logging
import os
import sys
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

# Optional imports for spatial overlay functionality
# Add parent directories to path to find phenotype_mask_utils
SPATIAL_OVERLAYS_AVAILABLE = False
try:
    # Add KMEANS directory to path to import phenotype_mask_utils
    kmeans_dir = Path(__file__).resolve().parent.parent / 'pan_cancer_subtype' / 'KMEANS'
    if kmeans_dir.exists():
        sys.path.insert(0, str(kmeans_dir))

    from phenotype_mask_utils import generate_and_save_phenotype_masks
    from ark.utils import data_utils, load_utils
    SPATIAL_OVERLAYS_AVAILABLE = True
    logger.info("Spatial overlay functionality available (ark-analysis detected)")
except ImportError as e:
    logger.debug(f"Spatial overlay functionality not available: {e}")
    logger.debug("To enable spatial overlays, install ark-analysis v0.7.2")
    SPATIAL_OVERLAYS_AVAILABLE = False


def compute_umap(
    factor_matrix: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute UMAP embedding from NMF factor loadings.

    Args:
        factor_matrix: NMF W matrix (cells x factors)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        random_state: Random seed

    Returns:
        umap_coords: 2D UMAP coordinates (cells x 2)
    """
    logger.info(f"Computing UMAP embedding (NMF space) for {factor_matrix.shape[0]:,} cells...")

    adata = ad.AnnData(factor_matrix)
    adata.obsm['X_nmf'] = factor_matrix

    # Compute neighbors on NMF representation
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep='X_nmf',
        random_state=random_state
    )

    # Compute UMAP
    sc.tl.umap(
        adata,
        min_dist=min_dist,
        random_state=random_state
    )

    logger.info("UMAP embedding (NMF space) complete")
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
# SPATIAL OVERLAY VISUALIZATION
# ============================================================================

def plot_spatial_overlay_comparison(
    phenotype_img,
    cluster_img,
    fov: str,
    save_dir: str,
    cell_table: pd.DataFrame,
    phenotype_cmap: str = 'tab20',
    cluster_cmap: str = 'tab20'
) -> None:
    """
    Create side-by-side comparison of cell type vs cluster spatial distribution.

    Args:
        phenotype_img: xarray image with phenotype mask
        cluster_img: xarray image with cluster mask
        fov: FOV identifier
        save_dir: Directory to save plot
        cell_table: Cell table with metadata for cluster counts
        phenotype_cmap: Colormap for phenotype (default: tab20)
        cluster_cmap: Colormap for clusters (default: tab20)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Extract arrays from xarray
    phenotype_data = phenotype_img.sel(fovs=fov).values.squeeze()
    cluster_data = cluster_img.sel(fovs=fov).values.squeeze()

    # Get number of unique values for colormaps
    n_phenotypes = len(np.unique(phenotype_data[phenotype_data > 0]))
    n_clusters = len(np.unique(cluster_data[cluster_data > 0]))

    # Plot phenotype
    im1 = axes[0].imshow(phenotype_data, cmap=phenotype_cmap, interpolation='nearest')
    axes[0].set_title(f'{fov}\nCell Type (Phenotype)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Cell Type', rotation=270, labelpad=20)

    # Plot leiden cluster
    im2 = axes[1].imshow(cluster_data, cmap=cluster_cmap, interpolation='nearest')

    # Get cluster statistics for this FOV
    fov_cells = cell_table[cell_table['fov'] == fov]
    n_cells = len(fov_cells)
    cluster_counts = fov_cells['leiden_cluster'].value_counts().to_dict()
    cluster_summary = f"n={n_cells:,} cells, {n_clusters} clusters"

    axes[1].set_title(f'{fov}\nLeiden Cluster\n({cluster_summary})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Cluster ID', rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = os.path.join(save_dir, f'{fov}_phenotype_vs_cluster.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved spatial overlay: {output_path}")


def generate_spatial_overlays(
    cluster_df: pd.DataFrame,
    seg_dir: str,
    output_dir: str,
    specified_fovs: List[str],
    cell_type_col: str = 'cell_meta_cluster'
) -> None:
    """
    Generate spatial overlay visualizations for specified FOVs.

    Args:
        cluster_df: DataFrame with fov, label, leiden_cluster, cell_meta_cluster columns
        seg_dir: Path to segmentation masks directory
        output_dir: Directory to save overlays
        specified_fovs: List of FOV identifiers to visualize
        cell_type_col: Column name for cell type labels (default: cell_meta_cluster)
    """
    if not SPATIAL_OVERLAYS_AVAILABLE:
        logger.error("Spatial overlay functionality not available. Install ark-analysis v0.7.2")
        logger.error("Installation: git clone -b v0.7.2 https://github.com/angelolab/ark-analysis.git")
        logger.error("              cd ark-analysis && conda env create -f environment.yml")
        return

    logger.info(f"Generating spatial overlays for {len(specified_fovs)} FOVs...")

    # Create output directory
    overlay_dir = os.path.join(output_dir, 'spatial_overlays')
    os.makedirs(overlay_dir, exist_ok=True)

    # Filter to specified FOVs
    fov_data = cluster_df[cluster_df['fov'].isin(specified_fovs)].copy()

    if len(fov_data) == 0:
        logger.warning(f"No cells found for specified FOVs: {specified_fovs}")
        return

    logger.info(f"Found {len(fov_data):,} cells across {len(specified_fovs)} FOVs")

    # Generate phenotype masks
    logger.info("Generating phenotype masks...")
    generate_and_save_phenotype_masks(
        fovs=specified_fovs,
        save_dir=overlay_dir,
        seg_dir=seg_dir,
        cell_table=fov_data,
        cell_type_col=cell_type_col,
        name_suffix='_phenotype_mask'
    )

    # Generate Leiden cluster masks
    logger.info("Generating Leiden cluster masks...")
    # Note: Assumes neighborhood_data has 'kmeans_cluster' column - we need to rename
    fov_data_for_mask = fov_data.copy()
    fov_data_for_mask['kmeans_cluster'] = fov_data_for_mask['leiden_cluster']

    data_utils.generate_and_save_neighborhood_cluster_masks(
        fovs=specified_fovs,
        save_dir=overlay_dir,
        seg_dir=seg_dir,
        neighborhood_data=fov_data_for_mask,
        name_suffix='_leiden_cluster_mask'
    )

    # Generate comparison plots
    logger.info("Generating side-by-side comparison plots...")
    for fov in specified_fovs:
        try:
            # Load phenotype mask
            phenotype_mask = load_utils.load_imgs_from_dir(
                data_dir=overlay_dir,
                files=[f'{fov}_phenotype_mask.tiff'],
                trim_suffix='_phenotype_mask',
                match_substring='_phenotype_mask',
                xr_dim_name='phenotype_mask',
                xr_channel_names=None
            )

            # Load cluster mask
            cluster_mask = load_utils.load_imgs_from_dir(
                data_dir=overlay_dir,
                files=[f'{fov}_leiden_cluster_mask.tiff'],
                trim_suffix='_leiden_cluster_mask',
                match_substring='_leiden_cluster_mask',
                xr_dim_name='leiden_cluster_mask',
                xr_channel_names=None
            )

            # Create comparison plot
            plot_spatial_overlay_comparison(
                phenotype_img=phenotype_mask,
                cluster_img=cluster_mask,
                fov=fov,
                save_dir=overlay_dir,
                cell_table=fov_data,
                phenotype_cmap='tab20',
                cluster_cmap='tab20'
            )

        except Exception as e:
            logger.error(f"Error generating overlay for FOV {fov}: {e}")
            continue

    # Save FOV list
    fov_list_path = os.path.join(overlay_dir, 'visualized_fovs.txt')
    with open(fov_list_path, 'w') as f:
        f.write('\n'.join(specified_fovs))
    logger.info(f"Saved FOV list to {fov_list_path}")

    logger.info(f"Spatial overlays complete! Results saved to {overlay_dir}")


# ============================================================================
# SPATIAL SCATTER VISUALIZATION (Centroid-based)
# ============================================================================

def get_category_colors_dict(categories) -> dict:
    """Generate distinct colors for categories as a dictionary."""
    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)

    if n_cats <= 10:
        cmap = plt.cm.tab10
    elif n_cats <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    return {cat: mcolors.rgb2hex(cmap(i / max(n_cats, 1))) for i, cat in enumerate(unique_cats)}


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

    # Plot 1: Grid colored by cell_meta_cluster
    logger.info("\n[Step 4] Generating grid plot colored by cell_meta_cluster...")
    plot_spatial_grid(
        df_sub,
        color_by='cell_meta_cluster',
        color_map=cell_type_colors,
        output_path=os.path.join(output_dir, 'spatial_grid_cell_meta_cluster.png'),
        fovs=fovs,
        size_scale=size_scale
    )

    # Plot 2: Grid colored by kmeans_neighborhood
    logger.info("\n[Step 5] Generating grid plot colored by kmeans_neighborhood...")
    plot_spatial_grid(
        df_sub,
        color_by='kmeans_neighborhood',
        color_map=nh_colors,
        output_path=os.path.join(output_dir, 'spatial_grid_kmeans_neighborhood.png'),
        fovs=fovs,
        size_scale=size_scale
    )

    # Individual FOV plots
    logger.info("\n[Step 6] Generating individual FOV plots...")
    fov_dir_cluster = os.path.join(output_dir, 'by_cell_meta_cluster')
    fov_dir_nh = os.path.join(output_dir, 'by_kmeans_neighborhood')
    os.makedirs(fov_dir_cluster, exist_ok=True)
    os.makedirs(fov_dir_nh, exist_ok=True)

    for fov in fovs:
        fov_df = df_sub[df_sub['fov'] == fov]
        safe_fov = fov.replace('/', '_').replace(' ', '_')

        # By cell_meta_cluster
        plot_spatial_fov(
            fov_df,
            color_by='cell_meta_cluster',
            color_map=cell_type_colors,
            output_path=os.path.join(fov_dir_cluster, f'{safe_fov}_cell_meta_cluster.png'),
            fov_name=fov,
            size_scale=size_scale
        )

        # By kmeans_neighborhood
        plot_spatial_fov(
            fov_df,
            color_by='kmeans_neighborhood',
            color_map=nh_colors,
            output_path=os.path.join(fov_dir_nh, f'{safe_fov}_kmeans_neighborhood.png'),
            fov_name=fov,
            size_scale=size_scale
        )

    logger.info(f"\nGenerated {len(fovs) * 2} individual FOV plots")

    # Save subsampled data
    sub_path = os.path.join(output_dir, 'subsampled_data.csv')
    df_sub.to_csv(sub_path, index=False)
    logger.info(f"Saved subsampled data to {sub_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Spatial scatter visualization complete!")
    logger.info(f"Plots saved to: {output_dir}/")
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
    # Spatial overlay parameters
    generate_spatial_overlays: bool = False,
    seg_dir: Optional[str] = None,
    spatial_fovs: Optional[List[str]] = None,
    cell_type_col: str = 'cell_meta_cluster'
) -> None:
    """
    Run full visualization pipeline.

    New parameters:
        generate_spatial_overlays: Enable spatial overlay visualization
        seg_dir: Path to segmentation masks (required if generate_spatial_overlays=True)
        spatial_fovs: List of FOV IDs to visualize (required if generate_spatial_overlays=True)
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

    # ---- NMF UMAP ----
    logger.info("\n[Step 4] Computing UMAP embedding (NMF space)...")
    umap_coords = compute_umap(
        factor_matrix,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
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
        title="UMAP (NMF space) colored by Leiden cluster",
        point_size=point_size
    )

    plot_small_multiples(
        umap_coords,
        cluster_labels,
        os.path.join(output_dir, f'{basename}_small_multiples.png'),
        point_size=point_size * 0.5
    )

    if has_metadata:
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

    # ---- Spatial Overlays (optional) ----
    if generate_spatial_overlays:
        if not SPATIAL_OVERLAYS_AVAILABLE:
            logger.warning("Spatial overlay functionality not available")
            logger.warning("Install ark-analysis v0.7.2 to enable spatial overlays")
        elif seg_dir is None:
            logger.warning("Segmentation directory (--seg-dir) required for spatial overlays")
        elif not os.path.exists(seg_dir):
            logger.warning(f"Segmentation directory not found: {seg_dir}")
        elif spatial_fovs is None or len(spatial_fovs) == 0:
            logger.warning("No FOVs specified for spatial overlays (use --spatial-fovs)")
        else:
            logger.info("\n[Step 6] Generating spatial overlay visualizations...")
            generate_spatial_overlays(
                cluster_df=df,
                seg_dir=seg_dir,
                output_dir=output_dir,
                specified_fovs=spatial_fovs,
                cell_type_col=cell_type_col
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

    # ---- Spatial overlay options ----
    parser.add_argument(
        '--spatial-overlays',
        action='store_true',
        help='Generate spatial overlay visualizations (requires --seg-dir and --spatial-fovs)'
    )

    parser.add_argument(
        '--seg-dir',
        default=None,
        help='Path to segmentation masks directory (required for spatial overlays)'
    )

    parser.add_argument(
        '--spatial-fovs',
        nargs='+',
        default=None,
        help='FOV IDs to visualize as spatial overlays (e.g., TONIC_TMA10_R3C3 TONIC_TMA10_R3C4)'
    )

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
        generate_spatial_overlays=args.spatial_overlays,
        seg_dir=args.seg_dir,
        spatial_fovs=args.spatial_fovs,
        cell_type_col=args.cell_type_col
    )


if __name__ == '__main__':
    main()