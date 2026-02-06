#!/usr/bin/env python3
"""
Memory-efficient NMF decomposition and Leiden clustering for large cell datasets.

UPDATED: Uses regular (full-batch) sklearn.decomposition.NMF instead of MiniBatchNMF.

Notes:
- Regular NMF can be much more memory/compute intensive than MiniBatchNMF on millions of cells.
- The CLI flag --batch-size is kept for backward compatibility but is NOT used by regular NMF.

Usage:
    python nmf_leiden_clustering.py input_file.csv --n_components 10 --resolution 0.1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
import scanpy as sc
import anndata as ad
import igraph as ig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data_chunked(
    filepath: str,
    chunksize: int = 100_000,
    metadata_cols: list = None,
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

    header = pd.read_csv(filepath, nrows=0)
    all_cols = header.columns.tolist()

    if metadata_cols is None:
        metadata_cols = all_cols[:3]  # Default: fov, label, cell_meta_cluster

    feature_cols = [c for c in all_cols if c not in metadata_cols]
    logger.info(f"Metadata columns: {metadata_cols}")
    logger.info(f"Feature columns: {feature_cols}")

    metadata_chunks = []
    data_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
        metadata_chunks.append(chunk[metadata_cols].copy())
        data_chunks.append(chunk[feature_cols].values.astype(np.float32))
        total_rows += len(chunk)
        logger.info(f"  Loaded {total_rows:,} rows...")

    logger.info(f"Concatenating {len(data_chunks)} chunks...")
    metadata_df = pd.concat(metadata_chunks, ignore_index=True)
    data_matrix = np.vstack(data_chunks)

    del metadata_chunks, data_chunks

    logger.info(f"Loaded {data_matrix.shape[0]:,} cells x {data_matrix.shape[1]} features")
    return metadata_df, data_matrix, feature_cols


def normalize_by_fov(
    metadata_df: pd.DataFrame,
    data_matrix: np.ndarray,
    fov_col: str = "fov",
) -> np.ndarray:
    """
    Normalize cell type composition by FOV-level (sample-level) composition.
    """
    logger.info("Applying sample-level (FOV) normalization...")

    normalized = data_matrix.copy()
    fov_values = metadata_df[fov_col].values

    for fov in np.unique(fov_values):
        mask = fov_values == fov
        fov_mean = data_matrix[mask].mean(axis=0)
        fov_mean = np.where(fov_mean == 0, 1e-10, fov_mean)
        normalized[mask] = data_matrix[mask] / fov_mean

    logger.info(f"Normalized {len(np.unique(fov_values))} FOVs")
    return normalized.astype(np.float32)


# ---------------------------
# SVD DIMENSIONALITY ANALYSIS
# ---------------------------

def run_svd_analysis(
    X: np.ndarray,
    max_components: int = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run truncated SVD to analyze the effective dimensionality of the data.

    SVD reveals the low-rank structure of the data and helps choose n_components
    for NMF. The singular values and explained variance ratios indicate how many
    components capture meaningful signal vs noise.

    Args:
        X: Input matrix (cells x features)
        max_components: Maximum components to compute (default: min(n_samples, n_features) - 1)
        random_state: Random seed for reproducibility

    Returns:
        singular_values: Array of singular values in descending order
        explained_variance_ratio: Proportion of variance explained by each component (float64)
        cumulative_variance: Cumulative variance explained (float64)
    """
    n_samples, n_features = X.shape
    if max_components is None:
        max_components = min(n_samples, n_features) - 1

    # Cap at reasonable number for large datasets
    max_components = min(max_components, 50)

    logger.info(f"Running SVD analysis with up to {max_components} components...")

    # Use float64 for accurate variance computation
    X_64 = X.astype(np.float64)
    X_centered = X_64 - X_64.mean(axis=0)

    svd = TruncatedSVD(
        n_components=max_components,
        random_state=random_state,
        algorithm="randomized",
    )
    svd.fit(X_centered)

    singular_values = svd.singular_values_
    explained_variance_ratio = svd.explained_variance_ratio_.astype(np.float64)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    logger.info("SVD Analysis Results:")
    logger.info("  Component | Singular Value | Var Explained | Cumulative Var")
    logger.info("  " + "-" * 60)
    for i in range(min(15, len(singular_values))):
        logger.info(
            f"  {i+1:9d} | {singular_values[i]:14.4f} | {explained_variance_ratio[i]:13.6f} | {cumulative_variance[i]:14.6f}"
        )
    if len(singular_values) > 15:
        logger.info(f"  ... ({len(singular_values) - 15} more components)")

    # Suggest n_components based on elbow/variance threshold
    var_90 = np.searchsorted(cumulative_variance, 0.90) + 1
    var_95 = np.searchsorted(cumulative_variance, 0.95) + 1
    var_99 = np.searchsorted(cumulative_variance, 0.99) + 1

    logger.info("")
    logger.info("  Suggested n_components:")
    logger.info(f"    90% variance: {var_90} components")
    logger.info(f"    95% variance: {var_95} components")
    logger.info(f"    99% variance: {var_99} components")

    return singular_values, explained_variance_ratio, cumulative_variance


# ---------------------------
# REGULAR (FULL-BATCH) NMF
# ---------------------------

def run_nmf(
    X: np.ndarray,
    n_components: int = 10,
    max_iter: int = 400,
    random_state: int = 42,
    solver: str = "cd",       # "cd" typically fastest; try "mu" if needed
    init: str = "nndsvda",    # good default; try "nndsvdar" if you hit issues with zeros
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run regular (full-batch) sklearn NMF decomposition.

    Args:
        X: Input matrix (cells x features)
        n_components: Number of NMF components/factors
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        solver: "cd" or "mu"
        init: Initialization strategy

    Returns:
        W: Basis matrix (cells x components) - cell factor loadings
        H: Components matrix (components x features) - feature weights per factor
    """
    logger.info(f"Running NMF with {n_components} components...")
    logger.info(f"  Solver: {solver}, Init: {init}, Max iterations: {max_iter}")

    X_nn = np.clip(X, 0, None)

    model = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )

    W = model.fit_transform(X_nn)
    H = model.components_

    logger.info(f"NMF complete. W shape: {W.shape}, H shape: {H.shape}")
    logger.info(f"Reconstruction error: {model.reconstruction_err_:.4f}")

    return W.astype(np.float32), H.astype(np.float32)


def run_nmf_with_reconstruction_error(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 400,
    random_state: int = 42,
    solver: str = "cd",
) -> Tuple[np.ndarray, float, float]:
    """
    Run regular NMF and return factor loadings with reconstruction error and explained variance.

    Note: Explained variance is computed in float64 for numerical accuracy.

    Returns:
        W: Factor loadings matrix (cells x components)
        reconstruction_err: Frobenius norm of reconstruction error
        explained_variance_ratio: Proportion of variance explained (0-1), computed in float64
    """
    X_nn = np.clip(X, 0, None)

    n_features = X_nn.shape[1]
    init = "nndsvda" if n_components <= n_features else "random"

    model = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )

    W = model.fit_transform(X_nn)
    H = model.components_

    # Compute explained variance in float64 for numerical accuracy
    X_64 = X_nn.astype(np.float64)
    W_64 = W.astype(np.float64)
    H_64 = H.astype(np.float64)

    reconstruction = W_64 @ H_64
    total_variance = np.sum(X_64 ** 2)
    residual_variance = np.sum((X_64 - reconstruction) ** 2)
    explained_variance_ratio = 1.0 - (residual_variance / total_variance)

    return W.astype(np.float32), float(model.reconstruction_err_), float(explained_variance_ratio)


# ---------------------------
# LEIDEN / GRAPH CLUSTERING
# ---------------------------

def run_leiden_clustering(
    factor_matrix: np.ndarray,
    resolution: float = 0.1,
    n_neighbors: int = 15,
    random_state: int = 42,
    return_graph: bool = False,
    use_cosine: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, scipy.sparse.csr_matrix]]:
    """
    Run Leiden clustering on factor loadings using scanpy.

    Args:
        factor_matrix: NMF factor loadings (cells x components)
        resolution: Leiden resolution parameter
        n_neighbors: Number of neighbors for kNN graph
        random_state: Random seed
        return_graph: If True, return (labels, connectivity_matrix)
        use_cosine: If True, row-normalize W and use cosine distance for kNN graph.
                    This matches the geometry used for silhouette/DB scoring.

    Returns:
        labels: Cluster assignments (if return_graph=False)
        (labels, connectivity_matrix): If return_graph=True
    """
    logger.info(f"Building AnnData object with {factor_matrix.shape[0]:,} cells...")

    # Row-normalize if using cosine distance to match scoring geometry
    if use_cosine:
        row_sums = factor_matrix.sum(axis=1, keepdims=True)
        W_normalized = factor_matrix / (row_sums + 1e-12)
        logger.info("Using row-normalized W with cosine distance for kNN graph")
    else:
        W_normalized = factor_matrix
        logger.info("Using raw W with Euclidean distance for kNN graph")

    adata = ad.AnnData(W_normalized)
    adata.obsm["X_nmf"] = W_normalized

    logger.info(f"Computing {n_neighbors} nearest neighbors...")

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_nmf",
        method="umap",  # approximate neighbors via pynndescent
        metric="cosine" if use_cosine else "euclidean",
        random_state=random_state,
    )

    logger.info(f"Running Leiden clustering with resolution={resolution}...")

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        n_iterations=2,
    )

    cluster_labels = adata.obs["leiden"].values.astype(int)
    n_clusters = len(np.unique(cluster_labels))
    logger.info(f"Found {n_clusters} clusters")

    # NEW: Optionally return connectivity matrix
    if return_graph:
        connectivity_matrix = adata.obsp["connectivities"]
        return cluster_labels, connectivity_matrix
    else:
        return cluster_labels


def subsample_stratified_by_fov(
    metadata_df: pd.DataFrame,
    data_matrix: np.ndarray,
    n_samples: int = 200_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Subsample cells stratified by FOV to maintain representation.
    """
    np.random.seed(random_state)

    n_total = len(metadata_df)
    if n_samples >= n_total:
        logger.info(f"Requested {n_samples:,} samples but only {n_total:,} available. Using all.")
        return metadata_df, data_matrix

    fov_counts = metadata_df["fov"].value_counts()
    fov_fractions = fov_counts / n_total
    samples_per_fov = (fov_fractions * n_samples).round().astype(int)

    diff = n_samples - samples_per_fov.sum()
    if diff != 0:
        largest_fovs = samples_per_fov.nlargest(abs(diff)).index
        for fov in largest_fovs:
            samples_per_fov[fov] += 1 if diff > 0 else -1

    sampled_indices = []
    for fov, n_fov_samples in samples_per_fov.items():
        fov_indices = metadata_df[metadata_df["fov"] == fov].index.tolist()
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
        data_matrix[sampled_indices],
    )


def run_leiden_with_labels(
    factor_matrix: np.ndarray,
    resolution: float,
    n_neighbors: int,
    random_state: int = 42,
    return_graph: bool = False,
    use_cosine: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, scipy.sparse.csr_matrix]]:
    """
    Run Leiden clustering and return labels (quiet version for tuning).

    Args:
        factor_matrix: NMF factor loadings (cells x components)
        resolution: Leiden resolution parameter
        n_neighbors: Number of neighbors for kNN graph
        random_state: Random seed
        return_graph: If True, return (labels, connectivity_matrix)
        use_cosine: If True, row-normalize W and use cosine distance for kNN graph.

    Returns:
        labels: Cluster assignments (if return_graph=False)
        (labels, connectivity_matrix): If return_graph=True
    """
    # Row-normalize if using cosine distance to match scoring geometry
    if use_cosine:
        row_sums = factor_matrix.sum(axis=1, keepdims=True)
        W_normalized = factor_matrix / (row_sums + 1e-12)
    else:
        W_normalized = factor_matrix

    adata = ad.AnnData(W_normalized)
    adata.obsm["X_nmf"] = W_normalized

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_nmf",
        method="umap",
        metric="cosine" if use_cosine else "euclidean",
        random_state=random_state,
    )

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        n_iterations=2,
    )

    cluster_labels = adata.obs["leiden"].values.astype(int)

    # NEW: Optionally return connectivity matrix
    if return_graph:
        connectivity_matrix = adata.obsp["connectivities"]
        return cluster_labels, connectivity_matrix
    else:
        return cluster_labels


def compute_cluster_stats(
    labels: np.ndarray,
    factor_matrix: np.ndarray = None,
    compute_silhouette: bool = True,
    silhouette_n: int = 0,
    silhouette_seed: int = 42,
    connectivity_matrix: scipy.sparse.csr_matrix = None,
) -> Dict:
    """
    Compute cluster statistics and quality metrics.

    Args:
        labels: Cluster assignments
        factor_matrix: NMF factor loadings (cells x components)
        compute_silhouette: Whether to compute silhouette score
        silhouette_n: Subsample size for silhouette (0 = all cells)
        silhouette_seed: Random seed for silhouette subsampling
        connectivity_matrix: Sparse connectivity matrix for graph-based metrics

    Returns:
        Dictionary with cluster statistics and quality metrics

    Note:
        factor_matrix is normalized row-wise before computing silhouette and DB scores
        to convert raw factor loadings to proportions (as recommended for NMF).
    """
    unique, counts = np.unique(labels, return_counts=True)

    stats = {
        "n_clusters": len(unique),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "median_cluster_size": int(np.median(counts)),
    }

    # Default metrics
    stats["silhouette_score"] = None
    stats["davies_bouldin_score"] = None
    stats["silhouette_n"] = None
    stats["modularity"] = None
    stats["mean_conductance"] = None
    stats["mean_cut_ratio"] = None
    stats["n_edges"] = None

    if factor_matrix is None or len(unique) <= 1:
        # Still compute graph metrics if available
        if connectivity_matrix is not None and len(unique) > 1:
            graph_metrics = compute_graph_metrics(labels, connectivity_matrix)
            stats.update(graph_metrics)
        return stats

    # NORMALIZE W BEFORE COMPUTING METRICS
    # Convert raw factor loadings to proportions (row-wise normalization)
    row_sums = factor_matrix.sum(axis=1, keepdims=True)
    W_normalized = factor_matrix / (row_sums + 1e-12)  # Add epsilon to prevent division by zero

    # Davies-Bouldin is relatively cheap
    try:
        stats["davies_bouldin_score"] = float(davies_bouldin_score(W_normalized, labels))
    except Exception:
        stats["davies_bouldin_score"] = None

    # Silhouette can be very expensive; optionally subsample
    if not compute_silhouette:
        # Compute graph metrics before returning
        if connectivity_matrix is not None:
            graph_metrics = compute_graph_metrics(labels, connectivity_matrix)
            stats.update(graph_metrics)
        return stats

    try:
        n = W_normalized.shape[0]
        if silhouette_n and silhouette_n < n:
            rng = np.random.default_rng(silhouette_seed)
            idx = rng.choice(n, size=silhouette_n, replace=False)
            stats["silhouette_score"] = float(silhouette_score(W_normalized[idx], labels[idx], metric="cosine"))
            stats["silhouette_n"] = int(silhouette_n)
        else:
            stats["silhouette_score"] = float(silhouette_score(W_normalized, labels, metric="cosine"))
            stats["silhouette_n"] = int(n)
    except Exception:
        stats["silhouette_score"] = None
        stats["silhouette_n"] = None

    # Compute graph-based metrics if connectivity matrix provided
    if connectivity_matrix is not None:
        graph_metrics = compute_graph_metrics(labels, connectivity_matrix)
        stats.update(graph_metrics)

    return stats


def compute_graph_metrics(
    labels: np.ndarray,
    connectivity_matrix: scipy.sparse.csr_matrix,
) -> Dict:
    """
    Compute graph-based clustering quality metrics.

    Since Leiden clustering is graph-based, these metrics evaluate cluster quality
    in the context of the k-nearest neighbor graph structure rather than the
    embedding space.

    Args:
        labels: Cluster assignments for each node
        connectivity_matrix: Sparse adjacency/connectivity matrix from kNN graph

    Returns:
        Dictionary with:
        - modularity: Graph modularity score (range [-0.5, 1], higher is better)
        - mean_conductance: Mean conductance across clusters (range [0, 1], lower is better)
        - mean_cut_ratio: Mean normalized cut ratio (range [0, 1], lower is better)
        - n_edges: Total number of edges in graph

    Metric Interpretations:
        Modularity: Fraction of edges within communities minus expected value
            >0.7: Strong community structure
            0.4-0.7: Moderate communities
            <0.4: Weak communities

        Conductance: Fraction of edges leaving each cluster
            <0.2: Well-separated clusters
            0.2-0.5: Moderate separation
            >0.5: Poor separation

        Cut Ratio: Edges crossing cluster boundary / max possible cuts
            <0.01: Very sparse cuts
            0.01-0.1: Moderate cuts
            >0.1: Dense cuts
    """
    try:
        # Convert sparse matrix to edge list for igraph
        if not scipy.sparse.issparse(connectivity_matrix):
            connectivity_matrix = scipy.sparse.csr_matrix(connectivity_matrix)

        # Get edges from sparse matrix (upper triangle only for undirected graph)
        coo = connectivity_matrix.tocoo()
        edge_list = []
        weights = []

        for i in range(len(coo.row)):
            if coo.row[i] < coo.col[i]:  # Upper triangle only
                edge_list.append((coo.row[i], coo.col[i]))
                weights.append(coo.data[i])

        n_nodes = connectivity_matrix.shape[0]

        # Build igraph Graph
        graph = ig.Graph(n=n_nodes, edges=edge_list, directed=False)
        if weights:
            graph.es['weight'] = weights

        # 1. MODULARITY
        membership = labels.astype(int).tolist()
        modularity = graph.modularity(membership, weights='weight' if weights else None)

        # 2. CONDUCTANCE (per cluster, then average)
        conductances = []
        unique_labels = np.unique(labels)

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            internal_weight = 0.0
            boundary_weight = 0.0

            # Iterate through cluster nodes
            for node in cluster_indices:
                neighbors = connectivity_matrix[node].nonzero()[1]
                for neighbor in neighbors:
                    weight = connectivity_matrix[node, neighbor]
                    if labels[neighbor] == cluster_id:
                        internal_weight += weight
                    else:
                        boundary_weight += weight

            # Conductance = boundary / (internal + boundary)
            total_weight = internal_weight + boundary_weight
            if total_weight > 0:
                conductance = boundary_weight / total_weight
                conductances.append(conductance)

        mean_conductance = float(np.mean(conductances)) if conductances else None

        # 3. CUT RATIO (per cluster, then average)
        cut_ratios = []

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum()
            outside_size = len(labels) - cluster_size

            if cluster_size == 0 or outside_size == 0:
                continue

            # Count edges between cluster and outside
            cut_weight = 0.0
            cluster_indices = np.where(cluster_mask)[0]

            for node in cluster_indices:
                neighbors = connectivity_matrix[node].nonzero()[1]
                for neighbor in neighbors:
                    if not cluster_mask[neighbor]:
                        cut_weight += connectivity_matrix[node, neighbor]

            # Normalized cut ratio
            max_possible_cuts = cluster_size * outside_size
            if max_possible_cuts > 0:
                cut_ratio = cut_weight / max_possible_cuts
                cut_ratios.append(cut_ratio)

        mean_cut_ratio = float(np.mean(cut_ratios)) if cut_ratios else None

        return {
            "modularity": float(modularity),
            "mean_conductance": mean_conductance,
            "mean_cut_ratio": mean_cut_ratio,
            "n_edges": len(edge_list),
        }

    except Exception as e:
        logger.warning(f"Failed to compute graph metrics: {e}")
        return {
            "modularity": None,
            "mean_conductance": None,
            "mean_cut_ratio": None,
            "n_edges": None,
        }


def compute_stability_ari(
    factor_matrix: np.ndarray,
    resolution: float,
    n_neighbors: int,
    seeds: List[int],
    neighbor_seed: int = 42,
    use_cosine: bool = True,
) -> Tuple[float, float]:
    """
    Compute clustering stability via ARI across different Leiden random seeds.

    Builds the neighbor graph ONCE with a fixed seed, then runs Leiden clustering
    multiple times with different seeds to test only the stochasticity of the
    Leiden algorithm itself (not the neighbor graph construction).

    Args:
        factor_matrix: NMF factor loadings (cells x components)
        resolution: Leiden resolution parameter
        n_neighbors: Number of neighbors for kNN graph
        seeds: List of random seeds to test Leiden stability
        neighbor_seed: Fixed seed for neighbor graph construction (default: 42)
        use_cosine: If True, row-normalize W and use cosine distance for kNN graph.

    Returns:
        mean_ari: Mean ARI across all pairwise comparisons
        std_ari: Standard deviation of ARI scores
    """
    # Row-normalize if using cosine distance to match scoring geometry
    if use_cosine:
        row_sums = factor_matrix.sum(axis=1, keepdims=True)
        W_normalized = factor_matrix / (row_sums + 1e-12)
    else:
        W_normalized = factor_matrix

    # Build neighbor graph ONCE with fixed seed
    adata = ad.AnnData(W_normalized)
    adata.obsm["X_nmf"] = W_normalized

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_nmf",
        method="umap",
        metric="cosine" if use_cosine else "euclidean",
        random_state=neighbor_seed,  # Fixed seed for reproducible neighbor graph
    )

    # Run Leiden multiple times with different seeds on the SAME graph
    all_labels = []
    for seed in seeds:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=seed,  # Only Leiden algorithm varies
            flavor="igraph",
            n_iterations=2,
            key_added=f"leiden_{seed}",  # Store each result separately
        )
        labels = adata.obs[f"leiden_{seed}"].values.astype(int)
        all_labels.append(labels)

    # Compute pairwise ARI scores
    ari_scores = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)

    return float(np.mean(ari_scores)), float(np.std(ari_scores))


def run_tuning(
    input_file: str,
    output_dir: str = "tuning_results",
    n_subsample: int = 200_000,
    n_components_list: List[int] = None,
    n_neighbors_list: List[int] = None,
    resolution_list: List[float] = None,
    stability_seeds: List[int] = None,
    chunksize: int = 100_000,
    random_state: int = 42,
    compute_silhouette: bool = True,
    silhouette_n: int = 20000,
    silhouette_seed: int = 42,
    normalize_by_fov_flag: bool = True,
    use_cosine: bool = True,
    args: Optional[argparse.Namespace] = None,
    ground_truth_col: str = "cell_meta_cluster",
) -> pd.DataFrame:
    """
    Run hyperparameter tuning on a subsample of the data.
    """
    if n_components_list is None:
        n_components_list = [5, 6, 7, 8]
    if n_neighbors_list is None:
        n_neighbors_list = [70, 90, 110]
    if resolution_list is None:
        resolution_list = [0.02, 0.04, 0.06, 0.1, 0.2]
    if stability_seeds is None:
        stability_seeds = [42, 123, 456, 789, 1011]

    logger.info("=" * 60)
    logger.info("NMF + Leiden Tuning Mode")
    logger.info("=" * 60)
    logger.info("Parameters to evaluate:")
    logger.info(f"  n_components: {n_components_list}")
    logger.info(f"  n_neighbors: {n_neighbors_list}")
    logger.info(f"  resolution: {resolution_list}")
    logger.info(f"  stability seeds: {stability_seeds}")

    os.makedirs(output_dir, exist_ok=True)

    # Save CLI arguments for reproducibility
    if args is not None:
        basename = Path(input_file).stem
        save_cli_args(output_dir, basename, args, mode="tuning")

    logger.info(f"\n[Step 1] Loading and subsampling to {n_subsample:,} cells...")
    metadata_df, data_matrix, feature_names = load_data_chunked(input_file, chunksize)
    metadata_df, data_matrix = subsample_stratified_by_fov(
        metadata_df, data_matrix, n_subsample, random_state
    )

    # after subsample_stratified_by_fov(...)
    if normalize_by_fov_flag:
        data_matrix = normalize_by_fov(metadata_df, data_matrix)

    logger.info("\n[Step 2] Evaluating n_components (reconstruction error + explained variance)...")
    nmf_results = []
    W_cache = {}

    for n in n_components_list:
        logger.info(f"  n_components={n}...")
        W, recon_err, explained_var = run_nmf_with_reconstruction_error(
            data_matrix, n, random_state=random_state
        )
        W_cache[n] = W
        nmf_results.append(
            {
                "n_components": n,
                "reconstruction_error": recon_err,
                "explained_variance": explained_var,
            }
        )
        logger.info(
            f"    reconstruction_error={recon_err:.6f}, explained_variance={explained_var:.4f}"
        )

    nmf_df = pd.DataFrame(nmf_results)

    logger.info("\n[Step 3] Grid search over n_components/n_neighbors/resolution...")
    logger.info("  (Computing stability ARI for each parameter combination)")
    grid_results = []
    total_combos = len(n_components_list) * len(n_neighbors_list) * len(resolution_list)
    combo_idx = 0

    for n in n_components_list:
        W = W_cache[n]
        for k in n_neighbors_list:
            for r in resolution_list:
                combo_idx += 1
                logger.info(f"  [{combo_idx}/{total_combos}] n={n}, k={k}, r={r}")

                labels, connectivity_matrix = run_leiden_with_labels(W, r, k, random_state, return_graph=True, use_cosine=use_cosine)
                stats = compute_cluster_stats(
                    labels,
                    W,
                    compute_silhouette=compute_silhouette,
                    silhouette_n=silhouette_n,
                    silhouette_seed=silhouette_seed,
                    connectivity_matrix=connectivity_matrix,
                )

                # Compute stability ARI for this parameter combination
                mean_ari, std_ari = compute_stability_ari(
                    W, r, k, stability_seeds, neighbor_seed=random_state, use_cosine=use_cosine
                )

                result = {
                    "n_components": n,
                    "n_neighbors": k,
                    "resolution": r,
                    "n_clusters": stats["n_clusters"],
                    "min_cluster_size": stats["min_cluster_size"],
                    "max_cluster_size": stats["max_cluster_size"],
                    "median_cluster_size": stats["median_cluster_size"],
                    "silhouette_score": stats["silhouette_score"],
                    "silhouette_n": stats["silhouette_n"],
                    "davies_bouldin_score": stats["davies_bouldin_score"],
                    "modularity": stats["modularity"],
                    "mean_conductance": stats["mean_conductance"],
                    "mean_cut_ratio": stats["mean_cut_ratio"],
                    "n_edges": stats["n_edges"],
                    "mean_ari": mean_ari,
                    "std_ari": std_ari,
                }
                grid_results.append(result)

                sil = stats["silhouette_score"]
                db = stats["davies_bouldin_score"]
                mod = stats["modularity"]
                cond = stats["mean_conductance"]
                cut = stats["mean_cut_ratio"]
                n_edges = stats["n_edges"]

                logger.info(
                    "    clusters=%s, min/med/max=%s/%s/%s, ARI=%.4f±%.4f, "
                    "sil=%s (n=%s), DB=%s, "
                    "mod=%s, cond=%s, cut=%s, edges=%s",
                    stats["n_clusters"],
                    stats["min_cluster_size"],
                    stats["median_cluster_size"],
                    stats["max_cluster_size"],
                    mean_ari,
                    std_ari,
                    f"{sil:.4f}" if sil is not None else "NA",
                    stats["silhouette_n"] if stats["silhouette_n"] is not None else "NA",
                    f"{db:.4f}" if db is not None else "NA",
                    f"{mod:.4f}" if mod is not None else "NA",
                    f"{cond:.4f}" if cond is not None else "NA",
                    f"{cut:.6f}" if cut is not None else "NA",
                    n_edges if n_edges is not None else "NA",
                )

    grid_df = pd.DataFrame(grid_results)

    logger.info("\n[Step 4] Stability analysis (ARI across seeds)...")
    logger.info("  Testing Leiden algorithm stability with different random seeds")
    logger.info("  (neighbor graph is built once with fixed seed, only Leiden varies)")
    stability_results = []

    mid_n = n_components_list[len(n_components_list) // 2]
    mid_k = n_neighbors_list[len(n_neighbors_list) // 2]
    W_mid = W_cache[mid_n]

    for r in resolution_list:
        logger.info(f"  resolution={r} (n={mid_n}, k={mid_k})...")
        mean_ari, std_ari = compute_stability_ari(
            W_mid, r, mid_k, stability_seeds, neighbor_seed=random_state, use_cosine=use_cosine
        )
        stability_results.append(
            {
                "resolution": r,
                "n_components": mid_n,
                "n_neighbors": mid_k,
                "mean_ari": mean_ari,
                "std_ari": std_ari,
            }
        )
        logger.info(f"    ARI={mean_ari:.4f} ± {std_ari:.4f}")

    stability_df = pd.DataFrame(stability_results)

    logger.info("\n[Step 5] Saving tuning results...")
    basename = Path(input_file).stem

    nmf_df.to_csv(os.path.join(output_dir, f"{basename}_nmf_reconstruction.csv"), index=False)
    grid_df.to_csv(os.path.join(output_dir, f"{basename}_grid_search.csv"), index=False)
    stability_df.to_csv(os.path.join(output_dir, f"{basename}_stability.csv"), index=False)

    report = generate_tuning_report(nmf_df, grid_df, stability_df, n_subsample)
    report_path = os.path.join(output_dir, f"{basename}_tuning_report.txt")
    with open(report_path, "w") as f:
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
    n_subsample: int,
) -> str:
    """Generate a human-readable tuning report with comprehensive metrics."""
    lines = [
        "=" * 70,
        "NMF + LEIDEN CLUSTERING TUNING REPORT",
        "=" * 70,
        f"\nSubsample size: {n_subsample:,} cells",
        "\n" + "-" * 50,
        "1. NMF: RECONSTRUCTION ERROR & EXPLAINED VARIANCE",
        "-" * 50,
    ]

    has_explained_var = "explained_variance" in nmf_df.columns

    for _, row in nmf_df.iterrows():
        if has_explained_var:
            lines.append(
                f"  n={int(row['n_components']):2d}  error={row['reconstruction_error']:.6f}  "
                f"explained_var={row['explained_variance']:.4f}"
            )
        else:
            lines.append(f"  n={int(row['n_components']):2d}  error={row['reconstruction_error']:.6f}")

    if len(nmf_df) > 1:
        errors = nmf_df["reconstruction_error"].values
        deltas = np.diff(errors)
        best_idx = np.argmin(deltas)
        suggested_n = int(nmf_df.iloc[best_idx + 1]["n_components"])
        lines.append(f"\n  Suggested n_components: {suggested_n} (elbow method)")

        if has_explained_var:
            above_80 = nmf_df[nmf_df["explained_variance"] >= 0.80]
            if len(above_80) > 0:
                min_n_80 = int(above_80["n_components"].min())
                lines.append(f"  First n with ≥80% variance: {min_n_80}")

    lines.extend(
        [
            "\n" + "-" * 50,
            "2. LEIDEN: STABILITY (ARI across seeds)",
            "-" * 50,
            "  Note: Neighbor graph built once, only Leiden algorithm randomness tested",
        ]
    )

    for _, row in stability_df.iterrows():
        ari_str = f"ARI={row['mean_ari']:.4f} ± {row['std_ari']:.4f}"
        stable_marker = " ✓" if row["mean_ari"] >= 0.9 else ""
        lines.append(f"  r={row['resolution']:.2f}  {ari_str}{stable_marker}")

    best_stability = stability_df.loc[stability_df["mean_ari"].idxmax()]
    lines.append(
        f"\n  Most stable: r={best_stability['resolution']:.2f} (ARI={best_stability['mean_ari']:.4f})"
    )

    lines.extend(
        [
            "\n" + "-" * 50,
            "3. CLUSTER QUALITY METRICS (Silhouette & Davies-Bouldin)",
            "-" * 50,
            "  (Higher Silhouette = better; Lower Davies-Bouldin = better)",
        ]
    )

    has_silhouette = "silhouette_score" in grid_df.columns

    if has_silhouette:
        valid_sil = grid_df[grid_df["silhouette_score"].notna()].copy()
        if len(valid_sil) > 0:
            top_5 = valid_sil.nlargest(5, "silhouette_score")
            lines.append("\n  Top 5 configurations by Silhouette Score:")
            for _, row in top_5.iterrows():
                ari_str = f", ARI={row['mean_ari']:.4f}±{row['std_ari']:.4f}" if "mean_ari" in row else ""
                lines.append(
                    f"    n={int(row['n_components'])}, k={int(row['n_neighbors'])}, "
                    f"r={row['resolution']:.2f} -> sil={row['silhouette_score']:.4f}, "
                    f"DB={row['davies_bouldin_score']:.4f}{ari_str}, clusters={int(row['n_clusters'])}"
                )

    # Add section for stability ARI
    has_ari = "mean_ari" in grid_df.columns
    if has_ari:
        lines.extend(
            [
                "\n" + "-" * 50,
                "3.5. STABILITY (ARI across seeds per configuration)",
                "-" * 50,
                "  (Higher ARI = more stable clustering; ≥0.9 is excellent)",
            ]
        )
        valid_ari = grid_df[grid_df["mean_ari"].notna()].copy()
        if len(valid_ari) > 0:
            top_5_ari = valid_ari.nlargest(5, "mean_ari")
            lines.append("\n  Top 5 configurations by Stability (ARI):")
            for _, row in top_5_ari.iterrows():
                lines.append(
                    f"    n={int(row['n_components'])}, k={int(row['n_neighbors'])}, "
                    f"r={row['resolution']:.2f} -> ARI={row['mean_ari']:.4f}±{row['std_ari']:.4f}, "
                    f"clusters={int(row['n_clusters'])}"
                )

    lines.extend(
        [
            "\n" + "-" * 50,
            "3.6. GRAPH-BASED METRICS",
            "-" * 50,
            "  Modularity: Higher is better (>0.7 = strong communities)",
            "  Conductance: Lower is better (<0.2 = well-separated)",
            "  Cut Ratio: Lower is better (<0.01 = sparse cuts)",
        ]
    )

    has_modularity = "modularity" in grid_df.columns
    if has_modularity:
        valid_mod = grid_df[grid_df["modularity"].notna()].copy()
        if len(valid_mod) > 0:
            top_5_mod = valid_mod.nlargest(5, "modularity")
            lines.append("\n  Top 5 configurations by Modularity:")
            for _, row in top_5_mod.iterrows():
                lines.append(
                    f"    n={int(row['n_components'])}, k={int(row['n_neighbors'])}, "
                    f"r={row['resolution']:.2f} -> mod={row['modularity']:.4f}, "
                    f"cond={row['mean_conductance']:.4f}, cut={row['mean_cut_ratio']:.6f}, "
                    f"clusters={int(row['n_clusters'])}"
                )

    lines.extend(
        [
            "\n" + "-" * 50,
            "4. CLUSTER SIZE DISTRIBUTION",
            "-" * 50,
        ]
    )

    for r in sorted(grid_df["resolution"].unique()):
        subset = grid_df[grid_df["resolution"] == r]
        n_clusters_range = f"{subset['n_clusters'].min()}-{subset['n_clusters'].max()}"
        min_size_range = f"{subset['min_cluster_size'].min()}-{subset['min_cluster_size'].max()}"
        lines.append(f"  r={r:.2f}  n_clusters={n_clusters_range:>8}  min_size={min_size_range}")

    lines.extend(
        [
            "\n" + "-" * 50,
            "5. FINAL RECOMMENDATIONS",
            "-" * 50,
        ]
    )

    good_configs = grid_df[grid_df["min_cluster_size"] >= 50].copy()

    if len(good_configs) > 0 and has_silhouette:
        # Filter by stability ARI from grid search results if available
        if has_ari:
            stable_configs = good_configs[good_configs["mean_ari"] >= 0.9]
            if len(stable_configs) > 0:
                good_configs = stable_configs
        else:
            # Fallback to stability_df for backward compatibility
            stable_resolutions = stability_df[stability_df["mean_ari"] >= 0.9]["resolution"].tolist()
            if stable_resolutions:
                stable_configs = good_configs[good_configs["resolution"].isin(stable_resolutions)]
                if len(stable_configs) > 0:
                    good_configs = stable_configs

        good_configs = good_configs[good_configs["silhouette_score"].notna()]

        if len(good_configs) > 0:
            best = good_configs.loc[good_configs["silhouette_score"].idxmax()]

            lines.append("  RECOMMENDED PARAMETERS:")
            lines.append(f"    n_components: {int(best['n_components'])}")
            lines.append(f"    n_neighbors:  {int(best['n_neighbors'])}")
            lines.append(f"    resolution:   {best['resolution']:.2f}")
            lines.append("")
            lines.append("  Expected results:")
            lines.append(f"    Clusters:         {int(best['n_clusters'])}")
            lines.append(f"    Min cluster size: {int(best['min_cluster_size'])}")
            lines.append(f"    Silhouette score: {best['silhouette_score']:.4f}")
            lines.append(f"    Davies-Bouldin:   {best['davies_bouldin_score']:.4f}")
            if "mean_ari" in best.index and best['mean_ari'] is not None:
                lines.append(f"    Stability (ARI):  {best['mean_ari']:.4f} ± {best['std_ari']:.4f}")
            if "modularity" in best.index and best['modularity'] is not None:
                lines.append(f"    Modularity:       {best['modularity']:.4f}")
                lines.append(f"    Mean Conductance: {best['mean_conductance']:.4f}")
                lines.append(f"    Mean Cut Ratio:   {best['mean_cut_ratio']:.6f}")

            lines.append("")
            lines.append("  Run with:")
            lines.append(
                f"    python nmf_full_leiden_clustering.py <input.csv> "
                f"-n {int(best['n_components'])} -k {int(best['n_neighbors'])} "
                f"-r {best['resolution']:.2f}"
            )
        else:
            lines.append("  No configurations meet all criteria.")
            lines.append("  Consider relaxing constraints or adjusting parameter ranges.")
    elif len(good_configs) > 0:
        rec = good_configs.iloc[0]
        lines.append(
            f"  Recommended: n={int(rec['n_components'])}, k={int(rec['n_neighbors'])}, r={rec['resolution']:.2f}"
        )
        lines.append(f"    -> {int(rec['n_clusters'])} clusters, min size={int(rec['min_cluster_size'])}")
    else:
        lines.append("  No configurations with min_cluster_size >= 50 found.")
        lines.append("  Consider using lower resolution values.")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def save_cli_args(
    output_dir: str,
    basename: str,
    args: argparse.Namespace,
    mode: str = "pipeline",
) -> None:
    """
    Save CLI arguments to JSON and TXT files for reproducibility.

    Args:
        output_dir: Output directory
        basename: Base name for output files
        args: Parsed command-line arguments
        mode: Either "pipeline" or "tuning" to indicate which mode was run
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert args namespace to dictionary
    args_dict = vars(args).copy()

    # Add metadata
    args_dict["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "script": "nmf_full_leiden_clustering.py",
    }

    # Save as JSON
    json_path = os.path.join(output_dir, f"{basename}_cli_args.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=2, default=str)
    logger.info(f"Saved CLI arguments to {json_path}")

    # Save as human-readable TXT
    txt_path = os.path.join(output_dir, f"{basename}_cli_args.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("NMF + Leiden Clustering - CLI Arguments\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {args_dict['_metadata']['timestamp']}\n")
        f.write(f"Mode: {mode}\n")
        f.write("-" * 60 + "\n\n")

        f.write("INPUT/OUTPUT:\n")
        f.write(f"  input_file:        {args.input_file}\n")
        f.write(f"  output_dir:        {args.output_dir}\n")
        f.write("\n")

        if mode == "pipeline":
            f.write("NMF PARAMETERS:\n")
            f.write(f"  n_components:      {args.n_components}\n")
            f.write(f"  batch_size:        {args.batch_size} (ignored for full NMF)\n")
            f.write("\n")

            f.write("LEIDEN PARAMETERS:\n")
            f.write(f"  resolution:        {args.resolution}\n")
            f.write(f"  n_neighbors:       {args.n_neighbors}\n")
            f.write(f"  kNN metric:        {'euclidean (raw W)' if args.euclidean else 'cosine (row-normalized W)'}\n")
            f.write("\n")

            f.write("PREPROCESSING:\n")
            f.write(f"  normalize_by_fov:  {args.normalize_by_fov}\n")
            f.write(f"  svd:               {args.svd}\n")
            f.write("\n")

            f.write("SUBSAMPLING:\n")
            f.write(f"  subsample:         {args.subsample}\n")
            f.write(f"  subsample_fraction:{args.subsample_fraction}\n")
            f.write("\n")

        elif mode == "tuning":
            f.write("TUNING PARAMETERS:\n")
            f.write(f"  tune_subsample:    {args.tune_subsample}\n")
            f.write(f"  tune_n:            {args.tune_n}\n")
            f.write(f"  tune_k:            {args.tune_k}\n")
            f.write(f"  tune_r:            {args.tune_r}\n")
            f.write(f"  no_silhouette:     {args.no_silhouette}\n")
            f.write(f"  silhouette_n:      {args.silhouette_n}\n")
            f.write(f"  normalize_by_fov:  {args.normalize_by_fov}\n")
            f.write(f"  kNN metric:        {'euclidean (raw W)' if args.euclidean else 'cosine (row-normalized W)'}\n")
            f.write("\n")

        f.write("OTHER:\n")
        f.write(f"  chunksize:         {args.chunksize}\n")
        f.write(f"  seed:              {args.seed}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("COMMAND LINE:\n")
        f.write(f"  python nmf_full_leiden_clustering.py {' '.join(sys.argv[1:])}\n")
        f.write("=" * 60 + "\n")

    logger.info(f"Saved CLI arguments to {txt_path}")


def save_results(
    output_dir: str,
    basename: str,
    metadata_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    factor_loadings: np.ndarray,
    basis_matrix: np.ndarray,
    feature_names: list,
) -> None:
    """Save clustering results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    n_factors = factor_loadings.shape[1]
    factor_cols = [f"NMF_factor_{i+1}" for i in range(n_factors)]

    result_df = metadata_df.copy()
    result_df["leiden_cluster"] = cluster_labels

    for i, col in enumerate(factor_cols):
        result_df[col] = factor_loadings[:, i]

    output_path = os.path.join(output_dir, f"{basename}_nmf_leiden_clusters.csv")
    logger.info(f"Saving cluster results to {output_path}")
    result_df.to_csv(output_path, index=False)

    basis_df = pd.DataFrame(
        basis_matrix.T,
        columns=factor_cols,
        index=feature_names,
    )
    basis_df.index.name = "cell_type"
    basis_path = os.path.join(output_dir, f"{basename}_nmf_basis_H.csv")
    logger.info(f"Saving NMF basis matrix to {basis_path}")
    basis_df.to_csv(basis_path)

    logger.info("Results saved successfully!")


def run_pipeline(
    input_file: str,
    output_dir: str = "results",
    n_components: int = 10,
    resolution: float = 0.1,
    batch_size: int = 1024,  # kept for backward compat; not used by regular NMF
    n_neighbors: int = 15,
    chunksize: int = 100_000,
    random_state: int = 42,
    normalize_by_fov_flag: bool = False,
    subsample_metadata_path: Optional[str] = None,
    subsample_fraction: float = 0.1,
    run_svd: bool = False,
    use_cosine: bool = True,
    args: Optional[argparse.Namespace] = None,
) -> pd.DataFrame:
    """
    Run the complete NMF + Leiden clustering pipeline.
    """
    logger.info("=" * 60)
    logger.info("NMF + Leiden Clustering Pipeline")
    logger.info("=" * 60)

    # Save CLI arguments for reproducibility
    if args is not None:
        basename = Path(input_file).stem
        save_cli_args(output_dir, basename, args, mode="pipeline")

    # Step 1: Load data (with optional batch subsampling)
    logger.info("\n[Step 1/6] Loading data...")
    if subsample_metadata_path:
        from input_data_sample import load_data_with_batch_subsample
        logger.info(f"Batch-stratified subsampling enabled (fraction={subsample_fraction})")
        metadata_df, data_matrix, feature_names = load_data_with_batch_subsample(
            input_file,
            subsample_metadata_path,
            fraction=subsample_fraction,
            chunksize=chunksize,
            random_state=random_state,
        )
    else:
        metadata_df, data_matrix, feature_names = load_data_chunked(
            input_file,
            chunksize=chunksize,
        )

    if normalize_by_fov_flag:
        logger.info("\n[Step 2/6] Applying sample-level normalization...")
        data_matrix = normalize_by_fov(metadata_df, data_matrix)
    else:
        logger.info("\n[Step 2/6] Skipping sample-level normalization")

    # Optional SVD analysis to determine effective dimensionality
    if run_svd:
        logger.info("\n[Step 3/6] Running SVD analysis...")
        singular_values, explained_var, cumulative_var = run_svd_analysis(
            data_matrix, random_state=random_state
        )
        # Save SVD results
        os.makedirs(output_dir, exist_ok=True)
        svd_df = pd.DataFrame({
            "component": range(1, len(singular_values) + 1),
            "singular_value": singular_values,
            "explained_variance": explained_var,
            "cumulative_variance": cumulative_var,
        })
        svd_path = os.path.join(output_dir, f"{Path(input_file).stem}_svd_analysis.csv")
        svd_df.to_csv(svd_path, index=False)
        logger.info(f"SVD results saved to {svd_path}")
    else:
        logger.info("\n[Step 3/6] Skipping SVD analysis (use --svd to enable)")

    logger.info(
        "\n[Step 4/6] Running NMF decomposition (regular full-batch NMF)..."
    )
    if batch_size is not None:
        logger.info("  Note: --batch-size is ignored for regular NMF.")

    W, H = run_nmf(
        data_matrix,
        n_components=n_components,
        max_iter=400,
        random_state=random_state,
        solver="cd",
        init="nndsvda",
    )

    del data_matrix

    logger.info("\n[Step 5/6] Running Leiden clustering...")
    cluster_labels, connectivity_matrix = run_leiden_clustering(
        W,
        resolution=resolution,
        n_neighbors=n_neighbors,
        random_state=random_state,
        use_cosine=use_cosine,
        return_graph=True,
    )

    # Compute and report clustering quality metrics (excluding ARI which requires stability testing)
    logger.info("\n[Step 5.5/6] Computing clustering quality metrics...")
    stats = compute_cluster_stats(
        cluster_labels,
        W,
        compute_silhouette=True,
        silhouette_n=20000,
        silhouette_seed=random_state,
        connectivity_matrix=connectivity_matrix,
    )

    logger.info("Clustering Quality Metrics:")
    logger.info(f"  Number of clusters:    {stats['n_clusters']}")
    logger.info(f"  Cluster sizes:         min={stats['min_cluster_size']}, median={stats['median_cluster_size']}, max={stats['max_cluster_size']}")
    if stats['silhouette_score'] is not None:
        logger.info(f"  Silhouette score:      {stats['silhouette_score']:.4f} (n={stats['silhouette_n']})")
    if stats['davies_bouldin_score'] is not None:
        logger.info(f"  Davies-Bouldin score:  {stats['davies_bouldin_score']:.4f}")
    if stats['modularity'] is not None:
        logger.info(f"  Modularity:            {stats['modularity']:.4f}")
    if stats['mean_conductance'] is not None:
        logger.info(f"  Mean conductance:      {stats['mean_conductance']:.4f}")
    if stats['mean_cut_ratio'] is not None:
        logger.info(f"  Mean cut ratio:        {stats['mean_cut_ratio']:.6f}")
    if stats['n_edges'] is not None:
        logger.info(f"  Graph edges:           {stats['n_edges']}")

    logger.info("\n[Step 6/6] Saving results...")
    basename = Path(input_file).stem
    save_results(
        output_dir=output_dir,
        basename=basename,
        metadata_df=metadata_df,
        cluster_labels=cluster_labels,
        factor_loadings=W,
        basis_matrix=H,
        feature_names=feature_names,
    )

    # Save clustering metrics to JSON
    metrics_path = os.path.join(output_dir, f"{basename}_clustering_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved clustering metrics to {metrics_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)

    result_df = metadata_df.copy()
    result_df["leiden_cluster"] = cluster_labels
    for i in range(W.shape[1]):
        result_df[f"NMF_factor_{i+1}"] = W[:, i]

    return result_df


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",")]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient NMF + Leiden clustering for large cell datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_file", help="Path to input CSV file (neighborhood matrix)")

    parser.add_argument("-o", "--output-dir", default="results", help="Output directory for results")

    parser.add_argument(
        "-n", "--n-components", type=int, default=5, help="Number of NMF components/factors"
    )

    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=0.1,
        help="Leiden clustering resolution (higher = more clusters)",
    )

    # Kept for compatibility, but ignored by regular NMF
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1024,
        help="(Ignored) Mini-batch size (only relevant for MiniBatchNMF)",
    )

    parser.add_argument(
        "-k", "--n-neighbors", type=int, default=100, help="Number of neighbors for kNN graph"
    )

    parser.add_argument(
        "-c", "--chunksize", type=int, default=100_000, help="Chunk size for reading CSV"
    )

    parser.add_argument(
        "--subsample",
        type=str,
        default=None,
        metavar="METADATA_CSV",
        help="Enable batch-stratified subsampling. Path to metadata CSV with fov, label, batch columns.",
    )

    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.1,
        help="Fraction of data to sample from each batch (default: 0.1 = 10%%). Only used with --subsample.",
    )

    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument(
        "--normalize-by-fov",
        action="store_true",
        default=False,
        help="Normalize cell composition by FOV-level mean to reduce sample-level bias",
    )

    # SVD analysis
    parser.add_argument(
        "--svd",
        action="store_true",
        help="Run SVD analysis to determine effective dimensionality before NMF. Helps choose n_components.",
    )

    # Tuning mode
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run in tuning mode: subsample data and grid search over parameters",
    )

    parser.add_argument(
        "--tune-subsample",
        type=int,
        default=200_000,
        help="Number of cells to subsample for tuning (default: 200000)",
    )

    parser.add_argument(
        "--tune-n",
        type=str,
        default=None,
        help="Comma-separated n_components values to try (default: 3,5,8,10)",
    )

    parser.add_argument(
        "--tune-k",
        type=str,
        default=None,
        help="Comma-separated n_neighbors values to try (default: 10,15,20,30)",
    )

    parser.add_argument(
        "--tune-r",
        type=str,
        default=None,
        help="Comma-separated resolution values to try (default: 0.1,0.3,0.5,0.8,1.0)",
    )

    parser.add_argument(
        "--no-silhouette",
        action="store_true",
        help="Disable silhouette score computation (saves time).",
    )

    parser.add_argument(
        "--silhouette-n",
        type=int,
        default=20000,
        help="Compute silhouette on a random subset of N cells (0 = all cells).",
    )

    parser.add_argument(
        "--euclidean",
        action="store_true",
        help="Use Euclidean distance on raw W for kNN graph instead of cosine distance on row-normalized W. "
             "Default uses cosine to match the geometry of silhouette/DB scoring.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    if args.subsample and not os.path.exists(args.subsample):
        logger.error(f"Metadata file not found: {args.subsample}")
        sys.exit(1)

    if args.subsample_fraction <= 0 or args.subsample_fraction > 1:
        logger.error("--subsample-fraction must be between 0 and 1")
        sys.exit(1)

    use_cosine = not args.euclidean

    if args.tune:
        run_tuning(
            input_file=args.input_file,
            output_dir=args.output_dir,
            n_subsample=args.tune_subsample,
            n_components_list=parse_int_list(args.tune_n) if args.tune_n else None,
            n_neighbors_list=parse_int_list(args.tune_k) if args.tune_k else None,
            resolution_list=parse_float_list(args.tune_r) if args.tune_r else None,
            chunksize=args.chunksize,
            random_state=args.seed,
            compute_silhouette=(not args.no_silhouette),
            silhouette_n=args.silhouette_n,
            silhouette_seed=args.seed,
            normalize_by_fov_flag=args.normalize_by_fov,
            use_cosine=use_cosine,
            args=args,
        )
    else:
        run_pipeline(
            input_file=args.input_file,
            output_dir=args.output_dir,
            n_components=args.n_components,
            resolution=args.resolution,
            batch_size=args.batch_size,  # ignored
            n_neighbors=args.n_neighbors,
            chunksize=args.chunksize,
            random_state=args.seed,
            normalize_by_fov_flag=args.normalize_by_fov,
            subsample_metadata_path=args.subsample,
            subsample_fraction=args.subsample_fraction,
            run_svd=args.svd,
            use_cosine=use_cosine,
            args=args,
        )


if __name__ == "__main__":
    main()