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
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchNMF
from sklearn.metrics import adjusted_rand_score
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


def subsample_stratified_by_fov(
    metadata_df: pd.DataFrame,
    data_matrix: np.ndarray,
    n_samples: int = 200_000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Subsample cells stratified by FOV to maintain representation.

    Args:
        metadata_df: DataFrame with metadata (must have 'fov' column)
        data_matrix: Feature matrix (cells x features)
        n_samples: Target number of cells to sample
        random_state: Random seed

    Returns:
        Subsampled metadata_df and data_matrix
    """
    np.random.seed(random_state)

    n_total = len(metadata_df)
    if n_samples >= n_total:
        logger.info(f"Requested {n_samples:,} samples but only {n_total:,} available. Using all.")
        return metadata_df, data_matrix

    # Calculate samples per FOV proportionally
    fov_counts = metadata_df['fov'].value_counts()
    fov_fractions = fov_counts / n_total
    samples_per_fov = (fov_fractions * n_samples).round().astype(int)

    # Adjust to hit exact target
    diff = n_samples - samples_per_fov.sum()
    if diff != 0:
        # Add/remove from largest FOVs
        largest_fovs = samples_per_fov.nlargest(abs(diff)).index
        for fov in largest_fovs:
            samples_per_fov[fov] += 1 if diff > 0 else -1

    # Sample from each FOV
    sampled_indices = []
    for fov, n_fov_samples in samples_per_fov.items():
        fov_indices = metadata_df[metadata_df['fov'] == fov].index.tolist()
        if n_fov_samples >= len(fov_indices):
            sampled_indices.extend(fov_indices)
        else:
            sampled_indices.extend(
                np.random.choice(fov_indices, size=n_fov_samples, replace=False).tolist()
            )

    sampled_indices = sorted(sampled_indices)

    logger.info(f"Subsampled {len(sampled_indices):,} cells from {len(fov_counts)} FOVs")

    return (
        metadata_df.iloc[sampled_indices].reset_index(drop=True),
        data_matrix[sampled_indices]
    )


def run_nmf_with_reconstruction_error(
    X: np.ndarray,
    n_components: int,
    batch_size: int = 1024,
    max_iter: int = 200,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Run NMF and return factor loadings with reconstruction error.
    """
    X_nn = np.clip(X, 0, None)

    # Use nndsvda when possible, fall back to random for high n_components
    n_features = X_nn.shape[1]
    init = 'nndsvda' if n_components <= n_features else 'random'

    model = MiniBatchNMF(
        n_components=n_components,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
        init=init,
        verbose=0
    )

    W = model.fit_transform(X_nn)
    return W.astype(np.float32), model.reconstruction_err_


def run_leiden_with_labels(
    factor_matrix: np.ndarray,
    resolution: float,
    n_neighbors: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Run Leiden clustering and return labels (quiet version for tuning).
    """
    adata = ad.AnnData(factor_matrix)
    adata.obsm['X_nmf'] = factor_matrix

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep='X_nmf',
        method='umap',
        random_state=random_state
    )

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor='igraph',
        n_iterations=2
    )

    return adata.obs['leiden'].values.astype(int)


def compute_cluster_stats(labels: np.ndarray) -> Dict:
    """Compute cluster statistics."""
    unique, counts = np.unique(labels, return_counts=True)
    return {
        'n_clusters': len(unique),
        'min_cluster_size': int(counts.min()),
        'max_cluster_size': int(counts.max()),
        'median_cluster_size': int(np.median(counts)),
        'cluster_sizes': {int(k): int(v) for k, v in zip(unique, counts)}
    }


def compute_stability_ari(
    factor_matrix: np.ndarray,
    resolution: float,
    n_neighbors: int,
    seeds: List[int]
) -> Tuple[float, float]:
    """
    Compute clustering stability via ARI across different random seeds.

    Returns:
        mean_ari: Mean pairwise ARI across seeds
        std_ari: Standard deviation of pairwise ARI
    """
    all_labels = []
    for seed in seeds:
        labels = run_leiden_with_labels(factor_matrix, resolution, n_neighbors, seed)
        all_labels.append(labels)

    # Compute pairwise ARI
    ari_scores = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)

    return float(np.mean(ari_scores)), float(np.std(ari_scores))


def run_tuning(
    input_file: str,
    output_dir: str = 'tuning_results',
    n_subsample: int = 200_000,
    n_components_list: List[int] = None,
    n_neighbors_list: List[int] = None,
    resolution_list: List[float] = None,
    stability_seeds: List[int] = None,
    batch_size: int = 1024,
    chunksize: int = 100_000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run hyperparameter tuning on a subsample of the data.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory for tuning results
        n_subsample: Number of cells to subsample
        n_components_list: List of n_components values to try
        n_neighbors_list: List of n_neighbors values to try
        resolution_list: List of resolution values to try
        stability_seeds: Seeds for stability analysis
        batch_size: NMF mini-batch size
        chunksize: CSV loading chunk size
        random_state: Base random seed

    Returns:
        DataFrame with tuning results
    """
    # Default parameter grids
    if n_components_list is None:
        n_components_list = [5, 8, 10, 12, 15]
    if n_neighbors_list is None:
        n_neighbors_list = [10, 15, 20, 30]
    if resolution_list is None:
        resolution_list = [0.1, 0.3, 0.5, 0.8, 1.0]
    if stability_seeds is None:
        stability_seeds = [42, 123, 456, 789, 1011]

    logger.info("=" * 60)
    logger.info("NMF + Leiden Tuning Mode")
    logger.info("=" * 60)
    logger.info(f"Parameters to evaluate:")
    logger.info(f"  n_components: {n_components_list}")
    logger.info(f"  n_neighbors: {n_neighbors_list}")
    logger.info(f"  resolution: {resolution_list}")
    logger.info(f"  stability seeds: {stability_seeds}")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and subsample data
    logger.info(f"\n[Step 1] Loading and subsampling to {n_subsample:,} cells...")
    metadata_df, data_matrix, feature_names = load_data_chunked(input_file, chunksize)
    metadata_df, data_matrix = subsample_stratified_by_fov(
        metadata_df, data_matrix, n_subsample, random_state
    )

    # Step 2: Evaluate n_components (reconstruction error)
    logger.info("\n[Step 2] Evaluating n_components (reconstruction error)...")
    nmf_results = []
    W_cache = {}

    for n in n_components_list:
        logger.info(f"  n_components={n}...")
        W, recon_err = run_nmf_with_reconstruction_error(
            data_matrix, n, batch_size=batch_size, random_state=random_state
        )
        W_cache[n] = W
        nmf_results.append({
            'n_components': n,
            'reconstruction_error': recon_err
        })
        logger.info(f"    reconstruction_error={recon_err:.6f}")

    nmf_df = pd.DataFrame(nmf_results)

    # Step 3: Grid search over n/k/r
    logger.info("\n[Step 3] Grid search over n_components/n_neighbors/resolution...")
    grid_results = []
    total_combos = len(n_components_list) * len(n_neighbors_list) * len(resolution_list)
    combo_idx = 0

    for n in n_components_list:
        W = W_cache[n]
        for k in n_neighbors_list:
            for r in resolution_list:
                combo_idx += 1
                logger.info(f"  [{combo_idx}/{total_combos}] n={n}, k={k}, r={r}")

                # Run clustering
                labels = run_leiden_with_labels(W, r, k, random_state)
                stats = compute_cluster_stats(labels)

                grid_results.append({
                    'n_components': n,
                    'n_neighbors': k,
                    'resolution': r,
                    'n_clusters': stats['n_clusters'],
                    'min_cluster_size': stats['min_cluster_size'],
                    'max_cluster_size': stats['max_cluster_size'],
                    'median_cluster_size': stats['median_cluster_size']
                })

    grid_df = pd.DataFrame(grid_results)

    # Step 4: Stability analysis (ARI across seeds) for each resolution
    logger.info("\n[Step 4] Stability analysis (ARI across seeds)...")
    stability_results = []

    # Use middle n_components and n_neighbors for stability
    mid_n = n_components_list[len(n_components_list) // 2]
    mid_k = n_neighbors_list[len(n_neighbors_list) // 2]
    W_mid = W_cache[mid_n]

    for r in resolution_list:
        logger.info(f"  resolution={r} (n={mid_n}, k={mid_k})...")
        mean_ari, std_ari = compute_stability_ari(W_mid, r, mid_k, stability_seeds)
        stability_results.append({
            'resolution': r,
            'n_components': mid_n,
            'n_neighbors': mid_k,
            'mean_ari': mean_ari,
            'std_ari': std_ari
        })
        logger.info(f"    ARI={mean_ari:.4f} ± {std_ari:.4f}")

    stability_df = pd.DataFrame(stability_results)

    # Save results
    logger.info("\n[Step 5] Saving tuning results...")
    basename = Path(input_file).stem

    nmf_df.to_csv(os.path.join(output_dir, f'{basename}_nmf_reconstruction.csv'), index=False)
    grid_df.to_csv(os.path.join(output_dir, f'{basename}_grid_search.csv'), index=False)
    stability_df.to_csv(os.path.join(output_dir, f'{basename}_stability.csv'), index=False)

    # Generate summary report
    report = generate_tuning_report(nmf_df, grid_df, stability_df, n_subsample)
    report_path = os.path.join(output_dir, f'{basename}_tuning_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(f"  - {basename}_nmf_reconstruction.csv")
    logger.info(f"  - {basename}_grid_search.csv")
    logger.info(f"  - {basename}_stability.csv")
    logger.info(f"  - {basename}_tuning_report.txt")

    print("\n" + report)

    return grid_df


def generate_tuning_report(
    nmf_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    n_subsample: int
) -> str:
    """Generate a human-readable tuning report."""
    lines = [
        "=" * 60,
        "NMF + LEIDEN CLUSTERING TUNING REPORT",
        "=" * 60,
        f"\nSubsample size: {n_subsample:,} cells",
        "\n" + "-" * 40,
        "1. RECONSTRUCTION ERROR vs n_components",
        "-" * 40,
    ]

    for _, row in nmf_df.iterrows():
        lines.append(f"  n={int(row['n_components']):2d}  error={row['reconstruction_error']:.6f}")

    # Find elbow (largest drop in error)
    if len(nmf_df) > 1:
        errors = nmf_df['reconstruction_error'].values
        deltas = np.diff(errors)
        best_idx = np.argmin(deltas)  # Largest negative change
        suggested_n = int(nmf_df.iloc[best_idx + 1]['n_components'])
        lines.append(f"\n  Suggested n_components: {suggested_n} (elbow method)")

    lines.extend([
        "\n" + "-" * 40,
        "2. STABILITY (ARI) vs resolution",
        "-" * 40,
    ])

    for _, row in stability_df.iterrows():
        lines.append(
            f"  r={row['resolution']:.1f}  ARI={row['mean_ari']:.4f} ± {row['std_ari']:.4f}"
        )

    # Suggest resolution with highest stability
    best_stability = stability_df.loc[stability_df['mean_ari'].idxmax()]
    lines.append(f"\n  Most stable resolution: {best_stability['resolution']:.1f} (ARI={best_stability['mean_ari']:.4f})")

    lines.extend([
        "\n" + "-" * 40,
        "3. CLUSTER COUNTS vs parameters",
        "-" * 40,
    ])

    # Group by resolution to show range
    for r in sorted(grid_df['resolution'].unique()):
        subset = grid_df[grid_df['resolution'] == r]
        n_clusters_range = f"{subset['n_clusters'].min()}-{subset['n_clusters'].max()}"
        min_size_range = f"{subset['min_cluster_size'].min()}-{subset['min_cluster_size'].max()}"
        lines.append(f"  r={r:.1f}  n_clusters={n_clusters_range:>8}  min_size={min_size_range}")

    lines.extend([
        "\n" + "-" * 40,
        "4. RECOMMENDATIONS",
        "-" * 40,
    ])

    # Filter for reasonable cluster sizes (min >= 50)
    good_configs = grid_df[grid_df['min_cluster_size'] >= 50]
    if len(good_configs) > 0:
        # Prefer higher stability resolutions
        stable_r = best_stability['resolution']
        matching = good_configs[good_configs['resolution'] == stable_r]
        if len(matching) > 0:
            rec = matching.iloc[0]
        else:
            # Fall back to first good config
            rec = good_configs.iloc[0]
        lines.append(f"  Recommended: n={int(rec['n_components'])}, k={int(rec['n_neighbors'])}, r={rec['resolution']:.1f}")
        lines.append(f"    -> {int(rec['n_clusters'])} clusters, min size={int(rec['min_cluster_size'])}")
    else:
        lines.append("  No configurations with min_cluster_size >= 50 found.")
        lines.append("  Consider using lower resolution values.")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


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


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(',')]


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated floats."""
    return [float(x.strip()) for x in s.split(',')]


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

    # Tuning mode arguments
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run in tuning mode: subsample data and grid search over parameters'
    )

    parser.add_argument(
        '--tune-subsample',
        type=int,
        default=200_000,
        help='Number of cells to subsample for tuning (default: 200000)'
    )

    parser.add_argument(
        '--tune-n',
        type=str,
        default='5,8,10,12,15',
        help='Comma-separated n_components values to try (default: 5,8,10,12,15)'
    )

    parser.add_argument(
        '--tune-k',
        type=str,
        default='10,15,20,30',
        help='Comma-separated n_neighbors values to try (default: 10,15,20,30)'
    )

    parser.add_argument(
        '--tune-r',
        type=str,
        default='0.1,0.3,0.5,0.8,1.0',
        help='Comma-separated resolution values to try (default: 0.1,0.3,0.5,0.8,1.0)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    if args.tune:
        # Tuning mode
        run_tuning(
            input_file=args.input_file,
            output_dir=args.output_dir,
            n_subsample=args.tune_subsample,
            n_components_list=parse_int_list(args.tune_n),
            n_neighbors_list=parse_int_list(args.tune_k),
            resolution_list=parse_float_list(args.tune_r),
            batch_size=args.batch_size,
            chunksize=args.chunksize,
            random_state=args.seed
        )
    else:
        # Normal pipeline mode
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
