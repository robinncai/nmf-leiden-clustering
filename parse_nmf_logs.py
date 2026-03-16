#!/usr/bin/env python3
"""
Parse NMF-Leiden clustering log files and generate a summary table.

Usage:
    python parse_nmf_logs.py [LOG_DIR] [--output OUTPUT_CSV]

Example:
    python parse_nmf_logs.py log/ --output results/job_summary.csv
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def parse_log_file(log_path: str) -> Dict[str, Any]:
    """
    Parse a single .log file to extract parameters from the command line.

    Returns dict with: job_id, n_components, k_neighbors, resolution
    """
    result = {
        'job_id': None,
        'n_components': None,
        'k_neighbors': None,
        'resolution': None,
    }

    # Extract job ID from filename
    match = re.search(r'nmf_leiden_(\d+)\.log', os.path.basename(log_path))
    if match:
        result['job_id'] = int(match.group(1))

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Look for the RUN command line to extract parameters
        # Pattern: -k <kNN> -r <resolution> -n <n_components>
        run_match = re.search(r'RUN:.*?-k\s+(\d+)\s+-r\s+([\d.]+)\s+-n\s+(\d+)', content)
        if run_match:
            result['k_neighbors'] = int(run_match.group(1))
            result['resolution'] = float(run_match.group(2))
            result['n_components'] = int(run_match.group(3))
        else:
            # Try alternative order: -n ... -k ... -r ...
            run_match = re.search(r'RUN:.*?-n\s+(\d+).*?-k\s+(\d+).*?-r\s+([\d.]+)', content)
            if run_match:
                result['n_components'] = int(run_match.group(1))
                result['k_neighbors'] = int(run_match.group(2))
                result['resolution'] = float(run_match.group(3))
    except Exception as e:
        print(f"Warning: Could not parse {log_path}: {e}")

    return result


def parse_err_file(err_path: str) -> Dict[str, Any]:
    """
    Parse a single .err file to extract results and status.

    Returns dict with: n_cells, n_clusters, status
    """
    result = {
        'n_cells': None,
        'n_clusters': None,
        'status': 'Unknown',
    }

    try:
        with open(err_path, 'r') as f:
            content = f.read()

        # Check for number of cells loaded
        cells_match = re.search(r'Loaded\s+([\d,]+)\s+cells\s+x\s+\d+\s+features', content)
        if cells_match:
            result['n_cells'] = int(cells_match.group(1).replace(',', ''))

        # Check for number of clusters found
        clusters_match = re.search(r'Found\s+(\d+)\s+clusters', content)
        if clusters_match:
            result['n_clusters'] = int(clusters_match.group(1))

        # Determine status
        if 'Pipeline complete!' in content:
            if 'Visualization complete!' in content:
                result['status'] = 'Complete'
            else:
                result['status'] = 'Complete (no viz)'
        elif 'Input file not found' in content or 'FileNotFoundError' in content:
            result['status'] = 'Failed (file not found)'
        elif 'DUE TO TIME LIMIT' in content or 'CANCELLED' in content:
            result['status'] = 'Failed (time limit)'
        elif 'Traceback' in content:
            result['status'] = 'Failed (error)'
        elif result['n_clusters'] is not None:
            # Found clusters but no "Pipeline complete!" - likely timed out during metrics/viz
            result['status'] = 'Failed (time limit)'
        elif 'Computing' in content and 'nearest neighbors' in content:
            # Started kNN computation but didn't finish
            result['status'] = 'Failed (time limit - kNN)'
        elif result['n_cells'] is not None and result['n_clusters'] is None:
            # Loaded data but no clusters - either failed or still running
            result['status'] = 'Failed (time limit)'
        else:
            result['status'] = 'Incomplete'

    except Exception as e:
        print(f"Warning: Could not parse {err_path}: {e}")

    return result


def parse_all_logs(log_dir: str, exclude_subfolders: bool = True) -> pd.DataFrame:
    """
    Parse all nmf_leiden log files in a directory.

    Args:
        log_dir: Directory containing .log and .err files
        exclude_subfolders: If True, only parse files in the top-level directory

    Returns:
        DataFrame with columns: job_id, n_components, k_neighbors, resolution, n_cells, n_clusters, status
    """
    log_dir = Path(log_dir)

    # Find all .log files (excluding nmf_plot files)
    if exclude_subfolders:
        log_files = list(log_dir.glob('nmf_leiden_*.log'))
    else:
        log_files = list(log_dir.rglob('nmf_leiden_*.log'))

    records = []

    for log_path in sorted(log_files):
        job_id_match = re.search(r'nmf_leiden_(\d+)\.log', log_path.name)
        if not job_id_match:
            continue

        job_id = int(job_id_match.group(1))
        err_path = log_path.with_suffix('.err')

        # Parse .log file for parameters
        log_data = parse_log_file(str(log_path))

        # Parse .err file for results (if exists)
        if err_path.exists():
            err_data = parse_err_file(str(err_path))
        else:
            err_data = {
                'n_cells': None,
                'n_clusters': None,
                'status': 'No .err file',
            }

        # Combine data
        record = {
            'job_id': job_id,
            'n_components': log_data['n_components'],
            'k_neighbors': log_data['k_neighbors'],
            'resolution': log_data['resolution'],
            'n_cells': err_data['n_cells'],
            'n_clusters': err_data['n_clusters'],
            'status': err_data['status'],
        }
        records.append(record)

    # Also check for .err files without corresponding .log files
    if exclude_subfolders:
        err_files = list(log_dir.glob('nmf_leiden_*.err'))
    else:
        err_files = list(log_dir.rglob('nmf_leiden_*.err'))

    existing_job_ids = {r['job_id'] for r in records}

    for err_path in sorted(err_files):
        job_id_match = re.search(r'nmf_leiden_(\d+)\.err', err_path.name)
        if not job_id_match:
            continue

        job_id = int(job_id_match.group(1))
        if job_id in existing_job_ids:
            continue

        # Parse .err file only
        err_data = parse_err_file(str(err_path))

        record = {
            'job_id': job_id,
            'n_components': None,
            'k_neighbors': None,
            'resolution': None,
            'n_cells': err_data['n_cells'],
            'n_clusters': err_data['n_clusters'],
            'status': err_data['status'] + ' (no .log)',
        }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('job_id').reset_index(drop=True)

    return df


def generate_summary_table(df: pd.DataFrame) -> str:
    """Generate a markdown summary table from the DataFrame."""

    # Header
    lines = [
        "## NMF-Leiden Pipeline Job Summary",
        "",
        "**Parameters:** `-n` = NMF components, `-k` = kNN neighbors, `-r` = Leiden resolution",
        "",
        "| Job ID | n (NMF) | k (kNN) | r (resolution) | n_cells | clusters | Status |",
        "|--------|---------|---------|----------------|---------|----------|--------|",
    ]

    # Data rows
    for _, row in df.iterrows():
        n = row['n_components'] if pd.notna(row['n_components']) else '?'
        k = row['k_neighbors'] if pd.notna(row['k_neighbors']) else '?'
        r = row['resolution'] if pd.notna(row['resolution']) else '?'
        cells = f"{int(row['n_cells']):,}" if pd.notna(row['n_cells']) else '-'
        clusters = int(row['n_clusters']) if pd.notna(row['n_clusters']) else '-'

        lines.append(f"| {row['job_id']} | {n} | {k} | {r} | {cells} | {clusters} | {row['status']} |")

    # Summary statistics
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")

    status_counts = df['status'].value_counts()
    total = len(df)
    complete = status_counts.get('Complete', 0) + status_counts.get('Complete (no viz)', 0)

    lines.append(f"- **Total jobs**: {total}")
    lines.append(f"- **Complete**: {complete} ({100*complete/total:.0f}%)")
    lines.append(f"- **Failed/Incomplete**: {total - complete} ({100*(total-complete)/total:.0f}%)")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Parse NMF-Leiden log files and generate summary table'
    )
    parser.add_argument(
        'log_dir',
        nargs='?',
        default='log',
        help='Directory containing log files (default: log/)'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output CSV file path (default: print to stdout)'
    )
    parser.add_argument(
        '--markdown',
        action='store_true',
        help='Also output markdown summary'
    )
    parser.add_argument(
        '--include-subfolders',
        action='store_true',
        help='Include log files in subfolders'
    )

    args = parser.parse_args()

    # Parse logs
    df = parse_all_logs(args.log_dir, exclude_subfolders=not args.include_subfolders)

    if len(df) == 0:
        print(f"No nmf_leiden log files found in {args.log_dir}")
        return

    # Output
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved summary to {args.output}")

        # Also save markdown if requested
        if args.markdown:
            md_path = args.output.replace('.csv', '.md')
            with open(md_path, 'w') as f:
                f.write(generate_summary_table(df))
            print(f"Saved markdown summary to {md_path}")
    else:
        # Print to stdout
        print(generate_summary_table(df))
        print()
        print("CSV format:")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
