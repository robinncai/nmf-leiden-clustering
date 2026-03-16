#!/usr/bin/env python3
"""
Aggregate NMF Leiden clustering results from multiple configurations.

This script scans a job directory containing multiple n_k_r subdirectories
and aggregates the metrics into a single summary CSV and JSON file.

Usage:
    python aggregate_results.py <job_dir>
    python aggregate_results.py results/nmf_leiden/12345678

Output:
    <job_dir>/aggregated_summary.csv
    <job_dir>/aggregated_summary.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def parse_config_dir(dirname: str) -> Optional[Dict[str, float]]:
    """
    Parse configuration from directory name like 'n8_k60_r0.2'.

    Returns:
        Dictionary with n_components, n_neighbors, resolution or None if parsing fails.
    """
    pattern = r"n(\d+)_k(\d+)_r([\d.]+)"
    match = re.match(pattern, dirname)
    if match:
        return {
            "n_components": int(match.group(1)),
            "n_neighbors": int(match.group(2)),
            "resolution": float(match.group(3)),
        }
    return None


def load_metrics(config_dir: Path) -> Optional[Dict]:
    """
    Load clustering metrics from a configuration directory.

    Looks for *_clustering_metrics.json in the directory.
    """
    metrics_files = list(config_dir.glob("*_clustering_metrics.json"))
    if not metrics_files:
        return None

    with open(metrics_files[0], "r") as f:
        return json.load(f)


def load_run_summary(config_dir: Path) -> Optional[Dict]:
    """Load run summary JSON if it exists."""
    summary_file = config_dir / "run_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            return json.load(f)
    return None


def aggregate_job_results(job_dir: Path) -> pd.DataFrame:
    """
    Aggregate results from all configuration subdirectories.

    Args:
        job_dir: Path to job directory containing n_k_r subdirectories

    Returns:
        DataFrame with aggregated metrics for each configuration
    """
    results = []

    # Find all config subdirectories
    for subdir in sorted(job_dir.iterdir()):
        if not subdir.is_dir():
            continue

        config = parse_config_dir(subdir.name)
        if config is None:
            continue

        # Load metrics
        metrics = load_metrics(subdir)
        run_summary = load_run_summary(subdir)

        if metrics is None:
            print(f"Warning: No metrics found in {subdir}")
            continue

        # Combine config and metrics
        row = {
            "config_dir": subdir.name,
            **config,
            **metrics,
        }

        # Add run info if available
        if run_summary:
            row["status"] = run_summary.get("status", "unknown")
            row["task_id"] = run_summary.get("task_id")

        results.append(row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate NMF Leiden clustering results from multiple configurations"
    )
    parser.add_argument(
        "job_dir",
        help="Path to job directory containing n_k_r subdirectories",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output prefix (default: <job_dir>/aggregated_summary)",
    )
    parser.add_argument(
        "--sort-by",
        default="silhouette_score",
        help="Column to sort results by (default: silhouette_score)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending)",
    )

    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    if not job_dir.exists():
        print(f"ERROR: Directory not found: {job_dir}")
        sys.exit(1)

    print(f"Aggregating results from: {job_dir}")

    # Aggregate results
    df = aggregate_job_results(job_dir)

    if len(df) == 0:
        print("ERROR: No results found")
        sys.exit(1)

    print(f"Found {len(df)} configurations")

    # Sort by specified column
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=args.ascending, na_position="last")

    # Output paths
    output_prefix = args.output or str(job_dir / "aggregated_summary")
    csv_path = f"{output_prefix}.csv"
    json_path = f"{output_prefix}.json"

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save JSON
    results_json = {
        "job_dir": str(job_dir),
        "n_configs": len(df),
        "sorted_by": args.sort_by,
        "results": df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS SUMMARY")
    print("=" * 80)

    # Select key columns for display
    display_cols = [
        "n_components", "n_neighbors", "resolution", "n_clusters",
        "min_cluster_size", "silhouette_score", "davies_bouldin_score",
        "modularity", "status",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    print(df[display_cols].to_string(index=False))

    # Print best configuration
    if "silhouette_score" in df.columns:
        best_idx = df["silhouette_score"].idxmax()
        best = df.loc[best_idx]
        print("\n" + "-" * 80)
        print("BEST CONFIGURATION (by silhouette score):")
        print(f"  n_components: {int(best['n_components'])}")
        print(f"  n_neighbors:  {int(best['n_neighbors'])}")
        print(f"  resolution:   {best['resolution']}")
        print(f"  n_clusters:   {int(best['n_clusters'])}")
        print(f"  silhouette:   {best['silhouette_score']:.4f}")
        if "modularity" in best and best["modularity"] is not None:
            print(f"  modularity:   {best['modularity']:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
