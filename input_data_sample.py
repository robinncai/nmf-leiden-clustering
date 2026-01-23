#!/usr/bin/env python3
"""
Batch-stratified subsampling for NMF clustering pipeline.

This module provides functions to subsample data proportionally from each batch,
enabling faster experimentation while maintaining batch representation.

Usage:
    from input_data_sample import load_data_with_batch_subsample

    metadata_df, data_matrix, features = load_data_with_batch_subsample(
        'input.csv', 'metadata.csv', fraction=0.1
    )
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def subsample_stratified_by_batch(
    metadata_df: pd.DataFrame,
    data_matrix: np.ndarray,
    batch_series: pd.Series,
    fraction: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Subsample data stratified by batch to maintain batch representation.

    Args:
        metadata_df: DataFrame with metadata columns
        data_matrix: Feature matrix (cells x features)
        batch_series: Series containing batch values for each row
        fraction: Fraction of data to sample from each batch (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        subsampled_metadata_df: Subsampled metadata DataFrame
        subsampled_data_matrix: Subsampled feature matrix

    Raises:
        ValueError: If fraction is not between 0 and 1
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

    np.random.seed(random_state)

    n_total = len(metadata_df)
    if fraction >= 1.0:
        logger.info(f"Fraction is 1.0, returning all {n_total:,} rows")
        return metadata_df, data_matrix

    # Get unique batches and their counts
    batch_counts = batch_series.value_counts()
    logger.info(f"Found {len(batch_counts)} unique batches")

    # Sample from each batch
    sampled_indices = []
    for batch, count in batch_counts.items():
        batch_indices = batch_series[batch_series == batch].index.tolist()
        n_samples = max(1, int(count * fraction))  # At least 1 sample per batch

        if n_samples >= len(batch_indices):
            # Take all if sample size exceeds batch size
            sampled = batch_indices
        else:
            sampled = np.random.choice(
                batch_indices, size=n_samples, replace=False
            ).tolist()

        sampled_indices.extend(sampled)
        logger.info(
            f"  Batch '{batch}': sampled {len(sampled):,} / {count:,} cells "
            f"({len(sampled)/count*100:.1f}%)"
        )

    # Sort indices to maintain original order
    sampled_indices = sorted(sampled_indices)

    total_sampled = len(sampled_indices)
    logger.info(
        f"Total subsampled: {total_sampled:,} / {n_total:,} cells "
        f"({total_sampled/n_total*100:.1f}%)"
    )

    return (
        metadata_df.iloc[sampled_indices].reset_index(drop=True),
        data_matrix[sampled_indices]
    )


def load_data_with_batch_subsample(
    input_filepath: str,
    metadata_filepath: str,
    fraction: float = 0.1,
    chunksize: int = 100_000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Load data and subsample by batch in one step.

    This is the main entry point for the --subsample feature. It loads the input
    CSV and metadata CSV, merges them on (fov, label), and samples a fraction
    of rows from each unique batch value.

    Args:
        input_filepath: Path to input CSV (neighborhood frequencies)
        metadata_filepath: Path to metadata CSV (must have fov, label, batch columns)
        fraction: Fraction to sample from each batch (default: 0.1 = 10%)
        chunksize: Chunk size for reading large CSV files
        random_state: Random seed for reproducibility

    Returns:
        metadata_df: Subsampled metadata (fov, label, cell_meta_cluster columns)
        data_matrix: Subsampled feature matrix (cells x features)
        feature_names: List of feature column names

    Raises:
        ValueError: If metadata file is missing required columns
        ValueError: If no rows match between input and metadata
        FileNotFoundError: If input or metadata file does not exist

    Example:
        >>> metadata_df, data_matrix, features = load_data_with_batch_subsample(
        ...     'input.csv', 'metadata.csv', fraction=0.1
        ... )
    """
    logger.info(f"Loading data with batch-stratified subsampling (fraction={fraction})")
    logger.info(f"  Input file: {input_filepath}")
    logger.info(f"  Metadata file: {metadata_filepath}")

    # Step 1: Load metadata CSV
    logger.info("Loading metadata...")
    metadata_external = pd.read_csv(metadata_filepath)

    # Validate metadata has required columns
    required_cols = {'fov', 'label', 'batch'}
    missing_cols = required_cols - set(metadata_external.columns)
    if missing_cols:
        raise ValueError(
            f"Metadata file missing required columns: {missing_cols}. "
            f"Found columns: {list(metadata_external.columns)}"
        )

    logger.info(f"  Loaded {len(metadata_external):,} metadata rows")

    # Step 2: Load input CSV header to identify columns
    header = pd.read_csv(input_filepath, nrows=0)
    all_cols = header.columns.tolist()

    # Default metadata columns in input file
    input_metadata_cols = all_cols[:3]  # fov, label, cell_meta_cluster
    feature_cols = [c for c in all_cols if c not in input_metadata_cols]
    logger.info(f"  Input metadata columns: {input_metadata_cols}")
    logger.info(f"  Feature columns: {feature_cols}")

    # Step 3: Load input data in chunks
    logger.info(f"Loading input data in chunks of {chunksize:,}...")
    input_metadata_chunks = []
    data_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(input_filepath, chunksize=chunksize, low_memory=False):
        input_metadata_chunks.append(chunk[input_metadata_cols].copy())
        data_chunks.append(chunk[feature_cols].values.astype(np.float32))
        total_rows += len(chunk)
        logger.info(f"  Loaded {total_rows:,} rows...")

    input_metadata_df = pd.concat(input_metadata_chunks, ignore_index=True)
    data_matrix = np.vstack(data_chunks)

    del input_metadata_chunks, data_chunks

    logger.info(f"Loaded {data_matrix.shape[0]:,} cells x {data_matrix.shape[1]} features")

    # Step 4: Merge with external metadata to get batch info
    logger.info("Merging with metadata to get batch information...")

    # Create a batch lookup from external metadata
    batch_lookup = metadata_external[['fov', 'label', 'batch']].copy()

    # Merge on (fov, label) - left join to keep all input rows
    merged = input_metadata_df.merge(
        batch_lookup,
        on=['fov', 'label'],
        how='left'
    )

    # Check for unmatched rows (will have NaN batch)
    unknown_count = merged['batch'].isna().sum()
    if unknown_count > 0:
        logger.info(
            f"  {unknown_count:,} rows have no matching metadata - grouped as 'Unknown' batch"
        )
        merged['batch'] = merged['batch'].fillna('Unknown')

    if merged['batch'].isna().all():
        raise ValueError(
            "No rows matched between input and metadata files. "
            "Check that (fov, label) values are consistent."
        )

    batch_series = merged['batch']

    # Step 5: Subsample stratified by batch
    logger.info(f"Subsampling {fraction*100:.1f}% from each batch...")
    subsampled_metadata, subsampled_data = subsample_stratified_by_batch(
        input_metadata_df,
        data_matrix,
        batch_series,
        fraction=fraction,
        random_state=random_state
    )

    return subsampled_metadata, subsampled_data, feature_cols


if __name__ == '__main__':
    # Simple test/demo
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 3:
        print("Usage: python input_data_sample.py <input.csv> <metadata.csv> [fraction]")
        print("Example: python input_data_sample.py data.csv metadata.csv 0.1")
        sys.exit(1)

    input_file = sys.argv[1]
    metadata_file = sys.argv[2]
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    metadata_df, data_matrix, features = load_data_with_batch_subsample(
        input_file, metadata_file, fraction=frac
    )

    print(f"\nResult: {len(metadata_df):,} rows, {len(features)} features")
    print(f"Metadata columns: {list(metadata_df.columns)}")
    print(f"Data matrix shape: {data_matrix.shape}")
