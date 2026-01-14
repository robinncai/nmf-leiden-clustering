#!/usr/bin/env python3
"""
Generate synthetic test data matching the neighborhood matrix format.
"""

import numpy as np
import pandas as pd
import os

def create_test_data(
    n_cells: int = 10_000,
    n_fovs: int = 10,
    output_dir: str = "test_data",
    seed: int = 42
) -> str:
    """
    Generate synthetic neighborhood frequency data.

    Creates data with structure matching ../clean_data/neighborhood_mats/ files:
    - fov: Field of view identifier
    - label: Cell ID within FOV
    - cell_meta_cluster: Cell type
    - Cancer cell, Myeloid cell, Lymphocyte, Endothelial cell, Fibroblast, Other: frequencies
    """
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    cell_types = ['Cancer cell', 'Myeloid cell', 'Lymphocyte',
                  'Endothelial cell', 'Fibroblast', 'Other']

    # Create cell metadata
    fovs = [f"TEST_FOV_{i+1}" for i in range(n_fovs)]
    cells_per_fov = n_cells // n_fovs

    data = []

    for fov_idx, fov in enumerate(fovs):
        for cell_idx in range(cells_per_fov):
            # Assign cell type (weighted distribution)
            cell_type = np.random.choice(
                cell_types,
                p=[0.3, 0.2, 0.2, 0.1, 0.15, 0.05]
            )

            # Generate neighborhood frequencies based on cell type
            # Each cell type has a characteristic neighborhood profile
            if cell_type == 'Cancer cell':
                # Cancer cells tend to cluster with other cancer cells
                base_freqs = np.array([0.5, 0.1, 0.1, 0.1, 0.15, 0.05])
            elif cell_type == 'Myeloid cell':
                # Myeloid cells near lymphocytes and cancer
                base_freqs = np.array([0.25, 0.3, 0.25, 0.05, 0.1, 0.05])
            elif cell_type == 'Lymphocyte':
                # Lymphocytes cluster together
                base_freqs = np.array([0.15, 0.2, 0.4, 0.05, 0.15, 0.05])
            elif cell_type == 'Endothelial cell':
                # Endothelial cells near fibroblasts
                base_freqs = np.array([0.1, 0.1, 0.1, 0.3, 0.35, 0.05])
            elif cell_type == 'Fibroblast':
                # Fibroblasts in stroma
                base_freqs = np.array([0.15, 0.1, 0.15, 0.25, 0.3, 0.05])
            else:
                # Other
                base_freqs = np.array([0.2, 0.15, 0.15, 0.15, 0.2, 0.15])

            # Add noise and normalize
            noise = np.random.dirichlet(np.ones(6) * 5)
            freqs = 0.7 * base_freqs + 0.3 * noise
            freqs = freqs / freqs.sum()  # Ensure sums to 1

            row = {
                'fov': fov,
                'label': cell_idx + 1,
                'cell_meta_cluster': cell_type,
            }
            for i, ct in enumerate(cell_types):
                row[ct] = freqs[i]

            data.append(row)

    df = pd.DataFrame(data)

    output_path = os.path.join(output_dir, f"test_neighborhood_freqs_{n_cells}.csv")
    df.to_csv(output_path, index=False)

    print(f"Created test data: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Cell types distribution:")
    print(df['cell_meta_cluster'].value_counts())

    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate test data')
    parser.add_argument('-n', '--n-cells', type=int, default=10_000,
                        help='Number of cells to generate')
    parser.add_argument('-f', '--n-fovs', type=int, default=10,
                        help='Number of FOVs')
    parser.add_argument('-o', '--output-dir', default='test_data',
                        help='Output directory')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    create_test_data(
        n_cells=args.n_cells,
        n_fovs=args.n_fovs,
        output_dir=args.output_dir,
        seed=args.seed
    )
