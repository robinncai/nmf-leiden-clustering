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
    pca_scale: bool = True
) -> None:
    """
    Run full visualization pipeline.
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

    args = parser.parse_args()

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
        pca_scale=args.pca_scale
    )


if __name__ == '__main__':
    main()