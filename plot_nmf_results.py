#!/usr/bin/env python3
"""Parse NMF results from log file and plot reconstruction error and explained variance."""

import re
import numpy as np
import matplotlib.pyplot as plt

def parse_results(filepath):
    """Parse the results.txt file to extract NMF metrics."""
    n_components_list = []
    reconstruction_errors = []
    explained_variances = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match lines like: n_components=2...
    # followed by: reconstruction_error=115.919304, explained_variance=0.8730
    n_comp_pattern = r'n_components=(\d+)\.\.\.'
    metrics_pattern = r'reconstruction_error=([\d.]+), explained_variance=([\d.]+)'

    lines = content.split('\n')
    current_n_comp = None

    for line in lines:
        # Check for n_components line
        n_comp_match = re.search(n_comp_pattern, line)
        if n_comp_match:
            current_n_comp = int(n_comp_match.group(1))

        # Check for metrics line
        metrics_match = re.search(metrics_pattern, line)
        if metrics_match and current_n_comp is not None:
            n_components_list.append(current_n_comp)
            reconstruction_errors.append(float(metrics_match.group(1)))
            explained_variances.append(float(metrics_match.group(2)))
            current_n_comp = None

    return n_components_list, reconstruction_errors, explained_variances


def parse_grid_search_results(filepath):
    """Parse the log file to extract grid search metrics (mod, cond, sil).

    Returns a dict organized by n_components with k, r, and metric values.
    """
    results = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match parameter lines: [1/125] n=5, k=50, r=0.01
    param_pattern = r'\[(\d+)/\d+\]\s+n=(\d+),\s+k=(\d+),\s+r=([\d.]+)'
    # Pattern to match metrics: sil=-0.5349 ... mod=0.2284, cond=0.0013
    metrics_pattern = r'sil=([-\d.]+).*?mod=([\d.]+),\s*cond=([\d.]+)'

    lines = content.split('\n')
    current_params = None

    for line in lines:
        # Check for parameter line
        param_match = re.search(param_pattern, line)
        if param_match:
            current_params = {
                'n': int(param_match.group(2)),
                'k': int(param_match.group(3)),
                'r': float(param_match.group(4))
            }

        # Check for metrics line
        metrics_match = re.search(metrics_pattern, line)
        if metrics_match and current_params is not None:
            n = current_params['n']
            if n not in results:
                results[n] = {'k': [], 'r': [], 'mod': [], 'cond': [], 'sil': []}

            results[n]['k'].append(current_params['k'])
            results[n]['r'].append(current_params['r'])
            results[n]['sil'].append(float(metrics_match.group(1)))
            results[n]['mod'].append(float(metrics_match.group(2)))
            results[n]['cond'].append(float(metrics_match.group(3)))
            current_params = None

    return results


def plot_metric_heatmaps(data, output_dir):
    """Create heatmaps for modularity, conductance, and silhouette for each n_components.

    Each n_components gets one figure with 3 subplots (one per metric).
    """
    metrics = [('mod', 'Modularity'), ('cond', 'Conductance'), ('sil', 'Silhouette')]

    for n in sorted(data.keys()):
        n_data = data[n]

        # Get unique k and r values
        k_vals = sorted(set(n_data['k']))
        r_vals = sorted(set(n_data['r']))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Grid Search Metrics for n_components={n}', fontsize=14, fontweight='bold')

        for ax, (metric_key, metric_name) in zip(axes, metrics):
            # Build the heatmap matrix
            heatmap = np.full((len(k_vals), len(r_vals)), np.nan)

            for i, (k, r, val) in enumerate(zip(n_data['k'], n_data['r'], n_data[metric_key])):
                k_idx = k_vals.index(k)
                r_idx = r_vals.index(r)
                heatmap[k_idx, r_idx] = val

            # Plot heatmap
            im = ax.imshow(heatmap, aspect='auto', cmap='viridis')

            # Set ticks and labels
            ax.set_xticks(range(len(r_vals)))
            ax.set_xticklabels([str(r) for r in r_vals])
            ax.set_yticks(range(len(k_vals)))
            ax.set_yticklabels([str(k) for k in k_vals])

            ax.set_xlabel('Resolution (r)')
            ax.set_ylabel('n_neighbors (k)')
            ax.set_title(metric_name)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add text annotations
            for i in range(len(k_vals)):
                for j in range(len(r_vals)):
                    if not np.isnan(heatmap[i, j]):
                        text_color = 'white' if heatmap[i, j] < (heatmap[~np.isnan(heatmap)].max() + heatmap[~np.isnan(heatmap)].min()) / 2 else 'black'
                        ax.text(j, i, f'{heatmap[i, j]:.3f}', ha='center', va='center',
                               color=text_color, fontsize=8)

        plt.tight_layout()
        output_path = f'{output_dir}/heatmap_n{n}_metrics.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_metrics(n_components, reconstruction_errors, explained_variances, output_path):
    """Create a dual y-axis plot for the NMF metrics."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot reconstruction error on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Components', fontsize=12)
    ax1.set_ylabel('Reconstruction Error', color=color1, fontsize=12)
    line1 = ax1.plot(n_components, reconstruction_errors, 'o-', color=color1,
                     linewidth=2, markersize=8, label='Reconstruction Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(n_components)

    # Create second y-axis for explained variance
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Explained Variance', color=color2, fontsize=12)
    line2 = ax2.plot(n_components, explained_variances, 's-', color=color2,
                     linewidth=2, markersize=8, label='Explained Variance')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)

    plt.title('NMF Reconstruction Error and Explained Variance vs Number of Components',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Plot NMF tuning results from log file')
    parser.add_argument('--log-file', '-l', default='log/nmf_leiden_14834942.err',
                        help='Path to the log file')
    parser.add_argument('--output-dir', '-o', default='results/nmf_leiden/nmf_leiden_14834942',
                        help='Output directory for plots')
    parser.add_argument('--results-file', '-r', default=None,
                        help='Path to results.txt file (optional)')
    args = parser.parse_args()

    log_file = args.log_file
    output_dir = args.output_dir
    results_file = args.results_file if args.results_file else os.path.join(output_dir, 'results.txt')
    output_plot = os.path.join(output_dir, 'nmf_metrics_plot.png')

    os.makedirs(output_dir, exist_ok=True)

    # Plot reconstruction error / explained variance if results file exists
    if os.path.exists(results_file):
        print("Parsing results file...")
        n_components, reconstruction_errors, explained_variances = parse_results(results_file)

        print(f"Found {len(n_components)} data points:")
        for i, (n, r, e) in enumerate(zip(n_components, reconstruction_errors, explained_variances)):
            print(f"  n_components={n}: reconstruction_error={r:.6f}, explained_variance={e:.4f}")

        print("\nCreating NMF metrics plot...")
        plot_metrics(n_components, reconstruction_errors, explained_variances, output_plot)
    else:
        print(f"Results file not found: {results_file}, skipping NMF metrics plot.")

    print("\nParsing grid search results from log file...")
    grid_data = parse_grid_search_results(log_file)

    if grid_data:
        print(f"Found grid search data for n_components: {sorted(grid_data.keys())}")
        for n in sorted(grid_data.keys()):
            print(f"  n={n}: {len(grid_data[n]['k'])} parameter combinations")

        print("\nCreating heatmap plots...")
        plot_metric_heatmaps(grid_data, output_dir)
    else:
        print("No grid search data found in log file.")


if __name__ == '__main__':
    main()
