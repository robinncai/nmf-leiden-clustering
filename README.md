# NMF-Leiden Clustering

Memory-efficient NMF decomposition and Leiden clustering for large cell datasets (1M+ cells).

## Features

- **Chunked CSV loading** - Reads large files in batches to limit peak memory
- **MiniBatchNMF** - Processes NMF in mini-batches instead of loading full matrix
- **Approximate nearest neighbors** - Scales to millions of cells using pynndescent
- **float32 precision** - Halves memory usage compared to float64

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/robinncai/nmf-leiden-clustering.git
cd nmf-leiden-clustering

# Create virtual environment and install dependencies
uv venv --python 3.9
uv sync

# Activate the environment
source .venv/bin/activate
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/robinncai/nmf-leiden-clustering.git
cd nmf-leiden-clustering

# Create virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install .
```

## Usage

### Basic usage

```bash
python nmf_leiden_clustering.py input_file.csv
```

### With custom parameters

```bash
python nmf_leiden_clustering.py input_file.csv \
    --n-components 10 \
    --resolution 0.5 \
    --output-dir results
```

### Full options

```
usage: nmf_leiden_clustering.py [-h] [-o OUTPUT_DIR] [-n N_COMPONENTS]
                                 [-r RESOLUTION] [-b BATCH_SIZE]
                                 [-k N_NEIGHBORS] [-c CHUNKSIZE] [-s SEED]
                                 input_file

positional arguments:
  input_file            Path to input CSV file (neighborhood matrix)

options:
  -h, --help            show this help message and exit
  -o, --output-dir      Output directory for results (default: results)
  -n, --n-components    Number of NMF components/factors (default: 10)
  -r, --resolution      Leiden clustering resolution - higher values
                        produce more clusters (default: 0.1)
  -b, --batch-size      Mini-batch size for NMF (default: 1024)
  -k, --n-neighbors     Number of neighbors for kNN graph (default: 15)
  -c, --chunksize       Chunk size for reading CSV (default: 100000)
  -s, --seed            Random seed for reproducibility (default: 42)
```

## Input Format

The input CSV file should have the following structure:

| Column | Description |
|--------|-------------|
| `fov` | Field of view identifier |
| `label` | Cell ID within the FOV |
| `cell_meta_cluster` | Cell type classification |
| `<cell_type_1>` | Neighborhood frequency/count for cell type 1 |
| `<cell_type_2>` | Neighborhood frequency/count for cell type 2 |
| ... | Additional cell type columns |

Example:
```csv
fov,label,cell_meta_cluster,Cancer cell,Myeloid cell,Lymphocyte,Endothelial cell,Fibroblast,Other
FOV_1,1,Cancer cell,0.45,0.15,0.10,0.10,0.15,0.05
FOV_1,2,Myeloid cell,0.20,0.35,0.25,0.05,0.10,0.05
```

## Output Files

The pipeline generates two output files in the specified output directory:

### 1. `<input_name>_nmf_leiden_clusters.csv`

Cell-level results with cluster assignments and NMF factor loadings:

| Column | Description |
|--------|-------------|
| `fov` | Original FOV identifier |
| `label` | Original cell label |
| `cell_meta_cluster` | Original cell type |
| `leiden_cluster` | Assigned cluster ID |
| `NMF_factor_1` | Loading for NMF factor 1 |
| `NMF_factor_2` | Loading for NMF factor 2 |
| ... | Additional NMF factors |

### 2. `<input_name>_nmf_basis_H.csv`

NMF basis matrix showing how each cell type contributes to each factor:

| Column | Description |
|--------|-------------|
| `cell_type` | Cell type name (index) |
| `NMF_factor_1` | Weight for factor 1 |
| `NMF_factor_2` | Weight for factor 2 |
| ... | Additional factors |

## Tuning Mode

Before running on your full dataset, use tuning mode to find optimal hyperparameters:

```bash
python nmf_leiden_clustering.py input_file.csv --tune -o tuning_results
```

### What tuning mode does

1. **Subsamples data** - Stratified by FOV to maintain representation (default: 200k cells)
2. **Evaluates n_components** - Reports reconstruction error for each value
3. **Grid search** - Tests all combinations of n/k/r parameters
4. **Stability analysis** - Measures clustering consistency (ARI) across random seeds
5. **Generates report** - Provides recommendations based on results

### Tuning options

```bash
python nmf_leiden_clustering.py input_file.csv --tune \
    --tune-subsample 200000 \
    --tune-n "5,8,10,12,15" \
    --tune-k "10,15,20,30" \
    --tune-r "0.1,0.3,0.5,0.8,1.0" \
    -o tuning_results
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tune-subsample` | 200000 | Number of cells to subsample |
| `--tune-n` | 5,8,10,12,15 | n_components values to try |
| `--tune-k` | 10,15,20,30 | n_neighbors values to try |
| `--tune-r` | 0.1,0.3,0.5,0.8,1.0 | resolution values to try |

### Tuning output files

- `*_nmf_reconstruction.csv` - Reconstruction error vs n_components
- `*_grid_search.csv` - Cluster counts/sizes for all parameter combinations
- `*_stability.csv` - ARI stability metrics vs resolution
- `*_tuning_report.txt` - Human-readable summary with recommendations

### Example tuning report

```
============================================================
NMF + LEIDEN CLUSTERING TUNING REPORT
============================================================

Subsample size: 200,000 cells

----------------------------------------
1. RECONSTRUCTION ERROR vs n_components
----------------------------------------
  n= 5  error=1.661571
  n= 8  error=0.892341
  n=10  error=0.654123

  Suggested n_components: 8 (elbow method)

----------------------------------------
2. STABILITY (ARI) vs resolution
----------------------------------------
  r=0.1  ARI=0.9234 ± 0.0312
  r=0.3  ARI=0.8856 ± 0.0445
  r=0.5  ARI=0.8123 ± 0.0567

  Most stable resolution: 0.1 (ARI=0.9234)

----------------------------------------
3. CLUSTER COUNTS vs parameters
----------------------------------------
  r=0.1  n_clusters=    5-12  min_size=1200-5000
  r=0.3  n_clusters=   10-25  min_size=400-2000
  r=0.5  n_clusters=   15-40  min_size=150-800

----------------------------------------
4. RECOMMENDATIONS
----------------------------------------
  Recommended: n=8, k=15, r=0.3
    -> 18 clusters, min size=850
============================================================
```

## Example: Generate Test Data

```bash
# Generate 10,000 synthetic cells for testing
python create_test_data.py -n 10000 -o test_data

# Run tuning on test data
python nmf_leiden_clustering.py test_data/test_neighborhood_freqs_10000.csv \
    --tune --tune-subsample 5000 -o tuning_test

# Run the pipeline with recommended parameters
python nmf_leiden_clustering.py test_data/test_neighborhood_freqs_10000.csv \
    -n 5 -r 0.5 -o test_results

# Generate visualizations
python visualize.py test_results/test_neighborhood_freqs_10000_nmf_leiden_clusters.csv \
    -m metadata.csv -o plots
```

## Visualization

After running the clustering pipeline, use `visualize.py` to generate UMAP plots for quality control and batch effect detection.

### Basic usage

```bash
python visualize.py results/your_file_nmf_leiden_clusters.csv -o plots
```

### With metadata (batch/subtype coloring)

```bash
python visualize.py results/your_file_nmf_leiden_clusters.csv \
    -m /path/to/harmonized_level12.csv \
    -o plots
```

The metadata file should have columns: `fov`, `label`, `batch`, `Subtype`

### Visualization options

```bash
python visualize.py cluster_results.csv \
    -m metadata.csv \
    -o plots \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --point-size 1.0 \
    --subsample 100000
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--metadata` | None | Path to metadata CSV with batch/Subtype |
| `-o`, `--output-dir` | plots | Output directory |
| `--n-neighbors` | 15 | UMAP n_neighbors parameter |
| `--min-dist` | 0.1 | UMAP min_dist parameter |
| `--point-size` | 1.0 | Size of scatter points |
| `--subsample` | None | Subsample to N cells for faster plotting |

### Output plots

| File | Description |
|------|-------------|
| `*_umap_clusters.png` | UMAP colored by Leiden cluster |
| `*_umap_batch.png` | UMAP colored by batch (detects batch effects) |
| `*_umap_subtype.png` | UMAP colored by subtype |
| `*_small_multiples.png` | One panel per cluster (reveals shape, fragmentation) |
| `*_umap_coords.csv` | UMAP coordinates for custom plotting |

### Interpreting the plots

**UMAP by cluster**: Check for clear separation between clusters. Overlapping clusters may indicate over-clustering (try lower resolution).

**UMAP by batch**: If clusters separate by batch rather than biology, you have batch effects. Consider batch correction or stratified analysis.

**UMAP by subtype**: Verify biological signal. Subtypes should show meaningful structure (e.g., separation or gradients).

**Small multiples**: Each panel highlights one cluster. Look for:
- Compact, well-defined shapes (good)
- Fragmented clusters split across UMAP (may need different parameters)
- Bridges between clusters (may be over-split)

## Memory Considerations

For a dataset with 1.2M cells and 6 features:
- Expected memory usage: 2-4 GB
- Adjust `--chunksize` if memory is limited during loading
- Adjust `--batch-size` if memory is limited during NMF

## License

MIT
