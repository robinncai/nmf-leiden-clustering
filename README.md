# NMF-Leiden Clustering

Memory-efficient NMF decomposition and Leiden clustering for large cell datasets (1M+ cells).

## Features

- **Chunked CSV loading** - Reads large files in batches to limit peak memory
- **MiniBatchNMF** - Processes NMF in mini-batches instead of loading full matrix
- **Approximate nearest neighbors** - Scales to millions of cells using pynndescent
- **float32 precision** - Halves memory usage compared to float64
- **SVD analysis** - Determine effective dimensionality before NMF
- **Batch-stratified subsampling** - Quick testing on representative data subsets

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
                                 [-k N_NEIGHBORS] [-c CHUNKSIZE]
                                 [--subsample METADATA_CSV]
                                 [--subsample-fraction FRACTION]
                                 [-s SEED] [--svd]
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
  --subsample           Path to metadata CSV for batch-stratified subsampling
  --subsample-fraction  Fraction to sample from each batch (default: 0.1)
  -s, --seed            Random seed for reproducibility (default: 42)
  --svd                 Run SVD analysis to determine effective dimensionality
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

## Batch-Stratified Subsampling

For quick testing or experimentation, use `--subsample` to sample a fraction of data from each batch:

```bash
python nmf_leiden_clustering.py input_file.csv \
    --subsample metadata.csv \
    --subsample-fraction 0.1
```

This requires a metadata CSV with columns: `fov`, `label`, `batch`. The script will:
1. Merge input data with metadata on (fov, label)
2. Sample 10% (or specified fraction) from each unique batch value
3. Run the pipeline on the subsampled data

Rows without matching metadata are grouped as 'Unknown' batch.

## SVD Analysis

Use `--svd` to run SVD analysis before NMF. This helps determine the effective dimensionality of your data:

```bash
python nmf_full_leiden_clustering.py input_file.csv --svd -o results
```

The SVD analysis:
- Computes singular values and explained variance ratios (in float64 for accuracy)
- Suggests n_components based on 90%, 95%, 99% variance thresholds
- Saves results to `*_svd_analysis.csv`

This helps you choose an appropriate `--n-components` value rather than guessing.

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
    --tune-n "3,5,8,10" \
    --tune-k "10,15,20,30" \
    --tune-r "0.1,0.3,0.5,0.8,1.0" \
    -o tuning_results
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tune-subsample` | 200000 | Number of cells to subsample |
| `--tune-n` | 3,5,8,10 | n_components values to try |
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

# Generate visualizations (UMAP + PCA enabled by default)
python visualize_expanded.py test_results/test_neighborhood_freqs_10000_nmf_leiden_clusters.csv \
    -m metadata.csv -o plots
```

## Visualization

After running the clustering pipeline, use `visualize_expanded.py` to generate UMAP and PCA plots for quality control and batch effect detection.

### Basic usage

```bash
python visualize_expanded.py results/your_file_nmf_leiden_clusters.csv -o plots
```

### With metadata (batch/subtype coloring)

```bash
python visualize_expanded.py results/your_file_nmf_leiden_clusters.csv \
    -m /path/to/harmonized_level12.csv \
    -o plots
```

The metadata file should have columns: `fov`, `label`, `batch`, `Subtype`

### Disabling PCA visualization

PCA visualization is enabled by default. To disable it:

```bash
python visualize_expanded.py results/your_file_nmf_leiden_clusters.csv \
    -m metadata.csv \
    -o plots \
    --no-pca
```

### Distance Metric for UMAP

By default, the UMAP kNN graph uses **cosine distance** to match the clustering pipeline. This ensures the UMAP embedding reflects the same geometry used during Leiden clustering.

```bash
# Default: cosine distance (matches clustering pipeline)
python visualize_expanded.py results.csv -o plots

# Use Euclidean distance instead
python visualize_expanded.py results.csv -o plots --umap-metric euclidean
```

### Visualization options

```bash
python visualize_expanded.py cluster_results.csv \
    -m metadata.csv \
    -o plots \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --point-size 1.0 \
    --subsample 100000 \
    --umap-metric cosine
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--metadata` | None | Path to metadata CSV with batch/Subtype |
| `-o`, `--output-dir` | plots | Output directory |
| `--n-neighbors` | 15 | UMAP n_neighbors parameter |
| `--min-dist` | 0.1 | UMAP min_dist parameter |
| `--point-size` | 1.0 | Size of scatter points |
| `--subsample` | None | Subsample to N cells for faster plotting |
| `--umap-metric` | cosine | Distance metric for UMAP kNN graph (`cosine` or `euclidean`) |
| `--no-pca` | False | Disable PCA visualization (PCA is enabled by default) |

### Output plots

| File | Description |
|------|-------------|
| `*_umap_clusters.png` | UMAP colored by Leiden cluster |
| `*_umap_batch.png` | UMAP colored by batch (detects batch effects) |
| `*_umap_subtype.png` | UMAP colored by subtype |
| `*_small_multiples.png` | One panel per cluster (reveals shape, fragmentation) |
| `*_umap_coords.csv` | UMAP coordinates for custom plotting |
| `*_pca_clusters.png` | PCA colored by Leiden cluster |
| `*_pca_batch.png` | PCA colored by batch |
| `*_pca_subtype.png` | PCA colored by subtype |
| `*_pca_coords.csv` | PCA coordinates |

### Interpreting the plots

**UMAP by cluster**: Check for clear separation between clusters. Overlapping clusters may indicate over-clustering (try lower resolution).

**UMAP by batch**: If clusters separate by batch rather than biology, you have batch effects. Consider batch correction or stratified analysis.

**UMAP by subtype**: Verify biological signal. Subtypes should show meaningful structure (e.g., separation or gradients).

**Small multiples**: Each panel highlights one cluster. Look for:
- Compact, well-defined shapes (good)
- Fragmented clusters split across UMAP (may need different parameters)
- Bridges between clusters (may be over-split)

**PCA plots** (with `--pca`): PCA provides a linear projection complementary to UMAP:
- Axis labels show variance explained by each PC (e.g., "PC1 (45.2% variance)")
- Compare PCA and UMAP: if clusters separate in PCA but not UMAP (or vice versa), this reveals different aspects of the data structure
- PCA is deterministic and faster than UMAP, useful for quick QC

## Spatial Overlay Visualization

The `visualize_expanded.py` script supports spatial overlay visualization, allowing you to see how Leiden clusters correspond to spatial tissue organization and cell type distributions.

### Prerequisites for Spatial Overlays

#### Install ark-analysis v0.7.2

```bash
cd /scratch/groups/sartandi/rcai2/projects/
git clone -b v0.7.2 https://github.com/angelolab/ark-analysis.git
cd ark-analysis
conda env create -f environment.yml
conda activate ark-analysis
```

#### Ensure phenotype_mask_utils.py is available

The script automatically looks for `phenotype_mask_utils.py` in the adjacent `pan_cancer_subtype/KMEANS/` directory.

### Usage

#### Basic Example: Generate UMAP plots only

```bash
python visualize_expanded.py \
    results/neighborhood_freqs-cell_meta_cluster_radius200_nmf_leiden_clusters.csv \
    --metadata ../pan_cancer_subtype/KMEANS/data/harmonized_full_with_metadata.csv \
    --output-dir plots/
```

#### Add spatial overlays for specific FOVs

```bash
python visualize_expanded.py \
    results/neighborhood_freqs-cell_meta_cluster_radius200_nmf_leiden_clusters.csv \
    --metadata ../pan_cancer_subtype/KMEANS/data/harmonized_full_with_metadata.csv \
    --output-dir plots/ \
    --spatial-overlays \
    --seg-dir /oak/stanford/groups/ccurtis2/users/syparkmd/Projects/TONIC/Data_from_Noah/deepcell_output \
    --spatial-fovs TONIC_TMA10_R3C3 TONIC_TMA10_R3C4 TONIC_TMA10_R7C1
```

### Spatial Overlay Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--spatial-overlays` | Optional | Enable spatial overlay visualization |
| `--seg-dir` | Required* | Path to segmentation masks directory |
| `--spatial-fovs` | Required* | Space-separated list of FOV IDs to visualize |
| `--cell-type-col` | Optional | Column name for cell type annotations (default: `cell_meta_cluster`) |

*Required only if `--spatial-overlays` is enabled

### Spatial Overlay Output Files

When spatial overlays are generated, the following files are created in `<output_dir>/spatial_overlays/`:

**For each FOV:**
- `<FOV>_phenotype_vs_cluster.png` - Side-by-side comparison showing:
  - **Left panel**: Cell types (phenotype) in spatial context
  - **Right panel**: Leiden cluster assignments in spatial context
- `<FOV>_phenotype_mask.tiff` - Spatial mask colored by cell type
- `<FOV>_leiden_cluster_mask.tiff` - Spatial mask colored by cluster ID

**Summary file:**
- `visualized_fovs.txt` - List of FOVs that were visualized

### Example Workflow with Spatial Overlays

#### 1. Run NMF-Leiden clustering
```bash
python nmf_full_leiden_clustering.py \
    input_data.csv \
    -n 7 -k 90 -r 0.5 \
    --output-dir results/
```

#### 2. Identify FOVs of interest

```bash
# List all TONIC FOVs in your data
grep "^TONIC" ../pan_cancer_subtype/KMEANS/data/harmonized_full_with_metadata.csv | \
    cut -d',' -f5 | sort | uniq | head -20
```

#### 3. Generate visualizations with spatial overlays

```bash
python visualize_expanded.py \
    results/*_clusters.csv \
    --metadata ../pan_cancer_subtype/KMEANS/data/harmonized_full_with_metadata.csv \
    --output-dir plots/ \
    --spatial-overlays \
    --seg-dir /oak/stanford/groups/ccurtis2/users/syparkmd/Projects/TONIC/Data_from_Noah/deepcell_output \
    --spatial-fovs TONIC_TMA10_R3C3 TONIC_TMA10_R3C4
```

### Interpreting Spatial Overlay Results

The `<FOV>_phenotype_vs_cluster.png` files show:
- **Left (Phenotype)**: Spatial distribution of cell types
- **Right (Cluster)**: Spatial distribution of Leiden clusters

**What to Look For:**

1. **Spatial Organization**:
   - Do clusters capture spatial regions (tumor core, periphery, stroma)?
   - Or do clusters mix cells from different spatial locations?

2. **Cell Type Composition**:
   - Are clusters homogeneous (single cell type) or mixed?
   - Do clusters group spatially adjacent cells of different types?

3. **Biological Interpretation**:
   - **Homogeneous spatial clusters**: Suggest clusters represent distinct tissue compartments
   - **Mixed cell type, spatially coherent**: Suggest clusters represent microenvironments or niches
   - **Scattered, mixed patterns**: May indicate clusters driven by composition rather than spatial organization

### Troubleshooting Spatial Overlays

**Error: "Spatial overlay functionality not available"**
- Install ark-analysis v0.7.2 (see Prerequisites)
- Ensure you're in the correct conda environment

**Error: "Segmentation directory not found"**
- Verify the path to segmentation masks is correct
- Check file permissions

**Error: "No cells found for specified FOVs"**
- Verify FOV names match exactly (case-sensitive)
- Check that FOVs exist in your cluster results CSV

## Memory Considerations

For a dataset with 1.2M cells and 6 features:
- Expected memory usage: 2-4 GB
- Adjust `--chunksize` if memory is limited during loading
- Adjust `--batch-size` if memory is limited during NMF

## License

MIT
