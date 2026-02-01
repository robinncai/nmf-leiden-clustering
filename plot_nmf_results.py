#!/usr/bin/env python3
"""Parse NMF results from log file and plot reconstruction error and explained variance."""

import re
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
    results_file = 'results/results.txt'
    output_plot = 'results/nmf_metrics_plot.png'

    print("Parsing results file...")
    n_components, reconstruction_errors, explained_variances = parse_results(results_file)

    print(f"Found {len(n_components)} data points:")
    for i, (n, r, e) in enumerate(zip(n_components, reconstruction_errors, explained_variances)):
        print(f"  n_components={n}: reconstruction_error={r:.6f}, explained_variance={e:.4f}")

    print("\nCreating plot...")
    plot_metrics(n_components, reconstruction_errors, explained_variances, output_plot)


if __name__ == '__main__':
    main()
