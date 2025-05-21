# EMDataBridge v0.0.1

A bridge between electron microscopy data formats with standardization, translation, and discovery capabilities:

1. **Format Standardizer**: Standardize various electron microscopy data formats to a common schema
2. **Format Translator**: Convert between different electron microscopy data formats
3. **Metadata Discovery**: Automatically discover and extract self-describing data organization from electron microscopy files

## Installation

```bash
# From PyPI (coming soon)
pip install EMDataBridge

# From source
pip install .
```

## Usage

See the documentation for each tool in their respective modules:

- `standardizer.py` - Standardize data formats
- `translator.py` - Translate between formats
- `discovery.py` - Discover self-describing data organization

## Supported Formats

- MRC (Medical Research Council)
- EM (Electron Microscopy)
- DM3/DM4 (Digital Micrograph)
- TIF/TIFF with metadata
- HDF5/Zarr-based formats
- MSA (EMSA format)
- EMD (Thermo Fisher Scientific's electron microscopy data format, formerly FEI)
- SER (Thermo Fisher Scientific's TIA series data format, formerly FEI)
- SerialEM formats
- Oxford Instruments formats (AZtec, INCA)
- Hitachi formats (.hds, .tif with metadata)
- Thermo Fisher Scientific formats (formerly FEI)

## Requirements

See `requirements.txt` for dependencies.

## Benchmarking

The package includes benchmarking tools to measure performance of format conversions and metadata extraction.

### Running Benchmarks

```bash
# Run the sample benchmark suite
python run_benchmarks.py

# Run specific benchmarks using the benchmark module
python benchmark.py standardize --files sample_data/*.tif --output-dir benchmark_results
python benchmark.py translate --files sample_data/*.dm4 --formats tiff hdf5 mrc --output-dir benchmark_results
python benchmark.py batch --input-dir sample_data --formats tiff emd ser --output-dir benchmark_results

# Run a full benchmark suite
python benchmark.py full --files sample_data/*.* --formats tiff hdf5 emd ser --input-dir sample_data --output-dir benchmark_results
```

### Benchmark Reports

Benchmark results are saved as JSON files in the output directory, along with visualizations showing performance metrics for different formats and operations.
