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

## Example: Generate Test Data

```bash
# Generate 10,000 synthetic cells for testing
python create_test_data.py -n 10000 -o test_data

# Run the pipeline on test data
python nmf_leiden_clustering.py test_data/test_neighborhood_freqs_10000.csv \
    -n 5 \
    -r 0.5 \
    -o test_results
```

## Memory Considerations

For a dataset with 1.2M cells and 6 features:
- Expected memory usage: 2-4 GB
- Adjust `--chunksize` if memory is limited during loading
- Adjust `--batch-size` if memory is limited during NMF

## License

MIT
