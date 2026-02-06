#!/usr/bin/env python3
"""
Visualization module for NMF + Leiden clustering results.

Generates UMAP embeddings and diagnostic plots for:
- Cluster visualization
- Batch effect detection
- Per-cluster small multiples
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
from matplotlib.gridspec import GridSpec
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
    logger.info(f"Computing UMAP embedding for {factor_matrix.shape[0]:,} cells...")

    adata = ad.AnnData(factor_matrix)
    adata.obsm['X_nmf'] = factor_matrix

    # Compute neighbors
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

    logger.info("UMAP embedding complete")
    return adata.obsm['X_umap']


def get_cluster_colors(n_clusters: int) -> List[str]:
    """Generate distinct colors for clusters."""
    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    return [mcolors.rgb2hex(cmap(i / n_clusters)) for i in range(n_clusters)]


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

    Args:
        umap_coords: 2D UMAP coordinates
        cluster_labels: Cluster assignments
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Size of scatter points
        alpha: Point transparency
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
        legend = ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            markerscale=5,
            frameon=True
        )
    else:
        # Too many clusters for legend
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

    Args:
        umap_coords: 2D UMAP coordinates
        metadata_values: Metadata values for coloring
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Size of scatter points
        alpha: Point transparency
        categorical: Whether metadata is categorical
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
        # Continuous colormap
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

    Args:
        umap_coords: 2D UMAP coordinates
        cluster_labels: Cluster assignments
        output_path: Path to save figure
        max_clusters: Maximum number of clusters to show
        cols: Number of columns in grid
        figsize_per_panel: Size of each panel
        point_size: Size of scatter points
        highlight_alpha: Alpha for highlighted cluster
        background_alpha: Alpha for background cells
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
        n_cells = mask.sum()

        # Plot background (all cells in gray)
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


def load_and_merge_metadata(
    cluster_results_path: str,
    metadata_path: str
) -> pd.DataFrame:
    """
    Load cluster results and merge with external metadata.

    Args:
        cluster_results_path: Path to *_nmf_leiden_clusters.csv
        metadata_path: Path to metadata CSV (e.g., harmonized_level12.csv)

    Returns:
        Merged DataFrame with cluster results and metadata
    """
    logger.info(f"Loading cluster results from {cluster_results_path}")
    clusters_df = pd.read_csv(cluster_results_path)

    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)

    # Merge on fov and label
    logger.info("Merging on fov + label...")
    merged = clusters_df.merge(
        metadata_df[['fov', 'label', 'batch', 'Subtype']],
        on=['fov', 'label'],
        how='left'
    )

    n_matched = merged['batch'].notna().sum()
    logger.info(f"Matched {n_matched:,} / {len(merged):,} cells with metadata")

    return merged


def run_visualization(
    cluster_results_path: str,
    metadata_path: Optional[str] = None,
    output_dir: str = 'plots',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    point_size: float = 1.0,
    subsample: Optional[int] = None,
    random_state: int = 42
) -> None:
    """
    Run full visualization pipeline.

    Args:
        cluster_results_path: Path to *_nmf_leiden_clusters.csv
        metadata_path: Optional path to metadata CSV
        output_dir: Directory for output plots
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        point_size: Size of scatter points
        subsample: Optionally subsample cells for faster plotting
        random_state: Random seed
    """
    logger.info("=" * 60)
    logger.info("Visualization Pipeline")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    basename = Path(cluster_results_path).stem.replace('_nmf_leiden_clusters', '')

    # Load data
    logger.info("\n[Step 1] Loading data...")
    df = pd.read_csv(cluster_results_path)

    # Extract NMF factor columns
    factor_cols = [c for c in df.columns if c.startswith('NMF_factor_')]
    factor_matrix = df[factor_cols].values.astype(np.float32)
    cluster_labels = df['leiden_cluster'].values

    logger.info(f"Loaded {len(df):,} cells with {len(factor_cols)} NMF factors")

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

    # Subsample if requested
    if subsample and subsample < len(df):
        logger.info(f"\n[Step 3] Subsampling to {subsample:,} cells...")
        np.random.seed(random_state)
        idx = np.random.choice(len(df), size=subsample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        factor_matrix = factor_matrix[idx]
        cluster_labels = cluster_labels[idx]
    else:
        logger.info("\n[Step 3] Using all cells (no subsampling)")

    # Compute UMAP
    logger.info("\n[Step 4] Computing UMAP embedding...")
    umap_coords = compute_umap(
        factor_matrix,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )

    # Save UMAP coordinates
    df['UMAP_1'] = umap_coords[:, 0]
    df['UMAP_2'] = umap_coords[:, 1]
    umap_path = os.path.join(output_dir, f'{basename}_umap_coords.csv')
    df.to_csv(umap_path, index=False)
    logger.info(f"Saved UMAP coordinates to {umap_path}")

    # Plot 1: UMAP colored by cluster
    logger.info("\n[Step 5] Generating plots...")
    plot_umap_clusters(
        umap_coords,
        cluster_labels,
        os.path.join(output_dir, f'{basename}_umap_clusters.png'),
        title="UMAP colored by Leiden cluster",
        point_size=point_size
    )

    # Plot 2: Small multiples
    plot_small_multiples(
        umap_coords,
        cluster_labels,
        os.path.join(output_dir, f'{basename}_small_multiples.png'),
        point_size=point_size * 0.5
    )

    # Plot 3 & 4: Metadata plots
    if has_metadata:
        # Batch
        if 'batch' in df.columns and df['batch'].notna().any():
            plot_umap_metadata(
                umap_coords,
                df['batch'].fillna('Unknown').values,
                os.path.join(output_dir, f'{basename}_umap_batch.png'),
                title="UMAP colored by Batch",
                point_size=point_size
            )

        # Subtype
        if 'Subtype' in df.columns and df['Subtype'].notna().any():
            plot_umap_metadata(
                umap_coords,
                df['Subtype'].fillna('Unknown').values,
                os.path.join(output_dir, f'{basename}_umap_subtype.png'),
                title="UMAP colored by Subtype",
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
        help='Path to *_nmf_leiden_clusters.csv file'
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

    args = parser.parse_args()

    if not os.path.exists(args.cluster_results):
        logger.error(f"Cluster results file not found: {args.cluster_results}")
        return

    run_visualization(
        cluster_results_path=args.cluster_results,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        point_size=args.point_size,
        subsample=args.subsample,
        random_state=args.seed
    )


if __name__ == '__main__':
    main()
