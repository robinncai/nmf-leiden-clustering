#!/usr/bin/env python3
"""
Memory-efficient NMF decomposition and Leiden clustering for large cell datasets.

This script processes neighborhood matrices with millions of cells using:
- Chunked CSV reading to limit memory usage during loading
- MiniBatchNMF for memory-efficient matrix factorization
- Approximate nearest neighbors for scalable graph construction
- Leiden algorithm for community detection

Usage:
    python nmf_leiden_clustering.py input_file.csv --n_components 10 --resolution 0.1
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchNMF
import scanpy as sc
import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_chunked(
    filepath: str,
    chunksize: int = 100_000,
    metadata_cols: list = None
) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Load large CSV file in chunks to reduce peak memory usage.

    Args:
        filepath: Path to the CSV file
        chunksize: Number of rows per chunk
        metadata_cols: List of metadata column names (default: first 3 columns)

    Returns:
        metadata_df: DataFrame with metadata columns
        data_matrix: Numpy array of numeric features
        feature_names: List of feature column names
    """
    logger.info(f"Loading data from {filepath} in chunks of {chunksize:,}")

    # First, read just the header to identify columns
    header = pd.read_csv(filepath, nrows=0)
    all_cols = header.columns.tolist()

    if metadata_cols is None:
        metadata_cols = all_cols[:3]  # Default: fov, label, cell_meta_cluster

    feature_cols = [c for c in all_cols if c not in metadata_cols]
    logger.info(f"Metadata columns: {metadata_cols}")
    logger.info(f"Feature columns: {feature_cols}")

    # Load data in chunks
    metadata_chunks = []
    data_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
        metadata_chunks.append(chunk[metadata_cols].copy())
        # Convert to float32 to save memory
        data_chunks.append(chunk[feature_cols].values.astype(np.float32))
        total_rows += len(chunk)
        logger.info(f"  Loaded {total_rows:,} rows...")

    logger.info(f"Concatenating {len(data_chunks)} chunks...")
    metadata_df = pd.concat(metadata_chunks, ignore_index=True)
    data_matrix = np.vstack(data_chunks)

    # Free memory from chunks
    del metadata_chunks, data_chunks

    logger.info(f"Loaded {data_matrix.shape[0]:,} cells x {data_matrix.shape[1]} features")
    return metadata_df, data_matrix, feature_cols


def run_minibatch_nmf(
    X: np.ndarray,
    n_components: int = 10,
    batch_size: int = 1024,
    max_iter: int = 200,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MiniBatchNMF for memory-efficient decomposition.

    Args:
        X: Input matrix (cells x features)
        n_components: Number of NMF components/factors
        batch_size: Size of mini-batches
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility

    Returns:
        W: Basis matrix (cells x components) - cell factor loadings
        H: Components matrix (components x features) - feature weights per factor
    """
    logger.info(f"Running MiniBatchNMF with {n_components} components...")
    logger.info(f"  Batch size: {batch_size:,}, Max iterations: {max_iter}")

    # Ensure non-negative values
    X_nn = np.clip(X, 0, None)

    model = MiniBatchNMF(
        n_components=n_components,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
        init='nndsvda',  # Better initialization for sparse data
        verbose=1
    )

    # W is cells x components (factor loadings per cell)
    W = model.fit_transform(X_nn)
    # H is components x features (feature weights per factor)
    H = model.components_

    logger.info(f"NMF complete. W shape: {W.shape}, H shape: {H.shape}")
    logger.info(f"Reconstruction error: {model.reconstruction_err_:.4f}")

    return W.astype(np.float32), H.astype(np.float32)


def run_leiden_clustering(
    factor_matrix: np.ndarray,
    resolution: float = 0.1,
    n_neighbors: int = 15,
    random_state: int = 42
) -> np.ndarray:
    """
    Run Leiden clustering on factor loadings using scanpy.

    Args:
        factor_matrix: Cell factor loadings (cells x components)
        resolution: Leiden resolution parameter (higher = more clusters)
        n_neighbors: Number of neighbors for kNN graph
        random_state: Random seed

    Returns:
        cluster_labels: Cluster assignment for each cell
    """
    logger.info(f"Building AnnData object with {factor_matrix.shape[0]:,} cells...")

    # Create AnnData with factor loadings as the main representation
    adata = ad.AnnData(factor_matrix)

    # Store factor loadings in obsm for use with neighbors
    adata.obsm['X_nmf'] = factor_matrix

    logger.info(f"Computing {n_neighbors} nearest neighbors...")

    # Use approximate neighbors for large datasets
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep='X_nmf',
        method='umap',  # Uses pynndescent for approximate NN
        random_state=random_state
    )

    logger.info(f"Running Leiden clustering with resolution={resolution}...")

    # Run Leiden clustering
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor='igraph',  # More memory efficient
        n_iterations=2
    )

    cluster_labels = adata.obs['leiden'].values.astype(int)
    n_clusters = len(np.unique(cluster_labels))
    logger.info(f"Found {n_clusters} clusters")

    return cluster_labels


def save_results(
    output_dir: str,
    basename: str,
    metadata_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    factor_loadings: np.ndarray,
    basis_matrix: np.ndarray,
    feature_names: list
) -> None:
    """
    Save clustering results to CSV files.

    Args:
        output_dir: Output directory path
        basename: Base name for output files
        metadata_df: Cell metadata
        cluster_labels: Cluster assignments
        factor_loadings: NMF factor loadings (W matrix)
        basis_matrix: NMF basis (H matrix, transposed for output)
        feature_names: Names of features/cell types
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create output DataFrame with metadata, clusters, and factor loadings
    n_factors = factor_loadings.shape[1]
    factor_cols = [f'NMF_factor_{i+1}' for i in range(n_factors)]

    result_df = metadata_df.copy()
    result_df['leiden_cluster'] = cluster_labels

    for i, col in enumerate(factor_cols):
        result_df[col] = factor_loadings[:, i]

    # Save main results
    output_path = os.path.join(output_dir, f'{basename}_nmf_leiden_clusters.csv')
    logger.info(f"Saving cluster results to {output_path}")
    result_df.to_csv(output_path, index=False)

    # Save basis matrix (H transposed: features x factors)
    basis_df = pd.DataFrame(
        basis_matrix.T,
        columns=factor_cols,
        index=feature_names
    )
    basis_df.index.name = 'cell_type'
    basis_path = os.path.join(output_dir, f'{basename}_nmf_basis_H.csv')
    logger.info(f"Saving NMF basis matrix to {basis_path}")
    basis_df.to_csv(basis_path)

    logger.info("Results saved successfully!")


def run_pipeline(
    input_file: str,
    output_dir: str = 'results',
    n_components: int = 10,
    resolution: float = 0.1,
    batch_size: int = 1024,
    n_neighbors: int = 15,
    chunksize: int = 100_000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run the complete NMF + Leiden clustering pipeline.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory for output files
        n_components: Number of NMF factors
        resolution: Leiden resolution parameter
        batch_size: NMF mini-batch size
        n_neighbors: Number of neighbors for clustering
        chunksize: CSV loading chunk size
        random_state: Random seed

    Returns:
        DataFrame with clustering results
    """
    logger.info("=" * 60)
    logger.info("NMF + Leiden Clustering Pipeline")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\n[Step 1/4] Loading data...")
    metadata_df, data_matrix, feature_names = load_data_chunked(
        input_file,
        chunksize=chunksize
    )

    # Step 2: Run NMF
    logger.info("\n[Step 2/4] Running NMF decomposition...")
    W, H = run_minibatch_nmf(
        data_matrix,
        n_components=n_components,
        batch_size=batch_size,
        random_state=random_state
    )

    # Free memory from original data matrix
    del data_matrix

    # Step 3: Run Leiden clustering
    logger.info("\n[Step 3/4] Running Leiden clustering...")
    cluster_labels = run_leiden_clustering(
        W,
        resolution=resolution,
        n_neighbors=n_neighbors,
        random_state=random_state
    )

    # Step 4: Save results
    logger.info("\n[Step 4/4] Saving results...")
    basename = Path(input_file).stem
    save_results(
        output_dir=output_dir,
        basename=basename,
        metadata_df=metadata_df,
        cluster_labels=cluster_labels,
        factor_loadings=W,
        basis_matrix=H,
        feature_names=feature_names
    )

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)

    # Return results for programmatic use
    result_df = metadata_df.copy()
    result_df['leiden_cluster'] = cluster_labels
    for i in range(W.shape[1]):
        result_df[f'NMF_factor_{i+1}'] = W[:, i]

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Memory-efficient NMF + Leiden clustering for large cell datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_file',
        help='Path to input CSV file (neighborhood matrix)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        default='results',
        help='Output directory for results'
    )

    parser.add_argument(
        '-n', '--n-components',
        type=int,
        default=10,
        help='Number of NMF components/factors'
    )

    parser.add_argument(
        '-r', '--resolution',
        type=float,
        default=0.1,
        help='Leiden clustering resolution (higher = more clusters)'
    )

    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=1024,
        help='Mini-batch size for NMF'
    )

    parser.add_argument(
        '-k', '--n-neighbors',
        type=int,
        default=15,
        help='Number of neighbors for kNN graph'
    )

    parser.add_argument(
        '-c', '--chunksize',
        type=int,
        default=100_000,
        help='Chunk size for reading CSV'
    )

    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        n_components=args.n_components,
        resolution=args.resolution,
        batch_size=args.batch_size,
        n_neighbors=args.n_neighbors,
        chunksize=args.chunksize,
        random_state=args.seed
    )


if __name__ == '__main__':
    main()
