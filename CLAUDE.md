# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
使用中文回复
在项目中的代码注释使用英文并使用Google Style
## Overview

hydrodataset is a Python package for accessing hydrological datasets with a unified API, serving as a data-adapting layer on top of AquaFetch. It standardizes diverse hydrological datasets into a consistent NetCDF format optimized for deep learning models (especially torchhydro).

**Core Workflow:**
1. Fetch raw data via AquaFetch backend
2. Standardize into xarray.Dataset format with unified variable names
3. Cache as `.nc` files (separate for timeseries and attributes) in location specified by `hydro_setting.yml`
4. All subsequent reads use fast `.nc` cache

## Environment Setup

### Using uv (Recommended for Development)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable package and environment management.

```bash
# Install dependencies with all extras (dev, docs, lint)
uv sync --all-extras

# Install only base dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Run commands in the virtual environment
uv run pytest
uv run python examples/read_dataset.py camels_us
```

### Using pip (For End Users)

```bash
pip install hydrodataset
```

**Configuration Required:** Create `~/hydro_setting.yml` in your home directory:
```yaml
local_data_path:
  root: 'D:\data\waterism'  # Update with your root data directory
  datasets-origin: 'D:\data\waterism\datasets-origin'  # Raw data from AquaFetch
  cache: 'D:\data\waterism\cache'  # Processed NetCDF cache files
```

The `cache` directory is where standardized `.nc` files are stored for fast subsequent reads.

## Common Commands

### Testing
```bash
# Run all tests
pytest

# Run all tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_camels_series.py

# Run specific test function
pytest tests/test_camels_series.py::test_read_attr_xrdataset

# Run tests matching a pattern
pytest -k "camels_aus"

# Run tests with coverage report
pytest --cov=hydrodataset --cov-report=html

# Run tests and stop at first failure
pytest -x
```

### Linting and Formatting
```bash
# Format code with black
black hydrodataset tests

# Check with flake8
flake8 hydrodataset tests
```

### Building Documentation
```bash
mkdocs serve  # Local preview
mkdocs build  # Build static site
```

### Running Examples
```bash
# Read from a specific dataset (uses argparse)
python examples/read_dataset.py camels_us
python examples/read_dataset.py camels_aus

# Or use uv run
uv run python examples/read_dataset.py camels_ch

# View all available dataset options
python examples/read_dataset.py --help
```

### Package Installation
```bash
# Install in development mode (editable install)
pip install -e .

# Build distribution packages
python -m build

# Install from local build
pip install dist/hydrodataset-*.whl
```

## Architecture

### Global Configuration and Constants

The package initialization (`hydrodataset/__init__.py`) sets up critical global variables:

**Key Global Variables:**
- `SETTING`: Loaded from `~/hydro_setting.yml`, contains all configuration
- `ROOT_DIR`: Alias for `SETTING["local_data_path"]["datasets-origin"]` - where raw datasets are stored
- `CACHE_DIR`: Alias for `SETTING["local_data_path"]["cache"]` - where processed `.nc` files are cached

**Dataset Constants:**
- `DATASETS`: List of supported dataset categories (CAMELS, Caravan, GRDC, HYSETS, LamaH, MOPEX)
- `CAMELS_REGIONS`: Regional variants of CAMELS (AUS, BR, CL, GB, US)
- Check `__init__.py` for complete list

**Import Pattern:**
```python
from hydrodataset import SETTING, ROOT_DIR, CACHE_DIR
from hydrodataset.camels_us import CamelsUs

# Access configuration
data_path = SETTING["local_data_path"]["datasets-origin"]
cache_path = SETTING["local_data_path"]["cache"]
```

### Core Class Hierarchy

**Base Class:** `HydroDataset` (in `hydrodataset/hydro_dataset.py`)
- Abstract base class defining the unified interface
- All dataset classes inherit from this
- Handles NetCDF caching, variable name standardization, and unit management

**Key Properties (must be implemented by subclasses):**
- `_attributes_cache_filename`: NetCDF filename for static attributes (e.g., "camels_us_attributes.nc")
- `_timeseries_cache_filename`: NetCDF filename for timeseries data (e.g., "camels_us_timeseries.nc")
- `default_t_range`: Default time range for the dataset as `["YYYY-MM-DD", "YYYY-MM-DD"]`

**Key Methods:**
- `read_object_ids()`: Returns basin/station IDs
- `read_attr_xrdataset(gage_id_lst, var_lst, ...)`: Read static attributes as xarray.Dataset
- `read_ts_xrdataset(gage_id_lst, t_range, var_lst, ...)`: Read timeseries as xarray.Dataset
- `cache_attributes_xrdataset()`: Generate and cache attributes NetCDF
- `cache_timeseries_xrdataset()`: Generate and cache timeseries NetCDF
- `read_area(gage_id_lst)`: Convenience method for reading basin areas
- `read_mean_prcp(gage_id_lst)`: Convenience method for reading mean precipitation

**Key Properties:**
- `available_static_features`: List of available static/attribute variables
- `available_dynamic_features`: List of available timeseries variables

### Variable Name Standardization

The library uses `StandardVariable` class (in `hydro_dataset.py`) to define standardized variable names across datasets:

**Static Variables:**
- Defined in `_base_static_definitions` and `_subclass_static_definitions`
- Include name mapping and units: `{"area": {"specific_name": "area_km2", "unit": "km^2"}}`

**Dynamic Variables:**
- Defined in `_dynamic_variable_mapping` (per subclass)
- Support multiple sources: e.g., streamflow from "bom", "gr4j", or "depth_based" in CAMELS-AUS
- Format:
```python
_dynamic_variable_mapping = {
    StandardVariable.STREAMFLOW: {
        "default_source": "bom",
        "sources": {
            "bom": {"specific_name": "q_cms_obs", "unit": "mm^3/s"},
            "gr4j": {"specific_name": "streamflow_mld_inclinfilled", "unit": "ML/day"},
        }
    }
}
```

**Common Standard Variables:**
- `streamflow`, `precipitation`, `temperature_max`, `temperature_min`
- `potential_evapotranspiration`, `evapotranspiration`, `evaporation`
- `solar_radiation`, `wind_speed`, `vapor_pressure`
- See `StandardVariable` class for complete list

### AquaFetch Integration

Each dataset class initializes an AquaFetch instance:
```python
self.aqua_fetch = CAMELS_US(data_path)  # or CAMELS_AUS, etc.
```

The base class methods automatically call:
- `aqua_fetch.stations()` for station IDs
- `aqua_fetch.dynamic_features` for available timeseries variables
- `aqua_fetch.static_features` for available attributes
- `aqua_fetch.fetch_stations_features()` for actual data

### Dataset Migration Status

**Fully Migrated (new architecture with HydroDataset base class):**
- camels_us, camels_aus

**Legacy Datasets (being migrated):**
The following 50+ datasets are supported but being migrated to the unified interface:
- CAMELS variants: camels_br, camels_ch, camels_cl, camels_col, camels_de, camels_dk, camels_es, camels_fi, camels_fr, camels_gb, camels_ind, camels_lux, camels_nz, camels_se, camels_deby
- CAMELSH variants: camelsh, camelsh_kr
- Other datasets: caravan, caravan_dk, grdc_caravan, hysets, lamah_ce, lamah_ice, mopex, bull, estreams, hype, jialing, simbi, waterbenchiowa, hyd_responses, arcade

See `examples/read_dataset.py` `DATASET_MAPPING` for the complete and up-to-date list of all supported datasets.

### Dataset-Specific Notes

**CAMELS-US Special Features:**
- Includes custom logic to read PET and ET from model output files (not available through AquaFetch)
- `read_camels_us_model_output_data()` method reads from `basin_timeseries_v1p2_modelOutput_*` directories
- Requires HUC-02 codes to locate model output files organized by hydrologic unit

## File Structure

- `hydrodataset/hydro_dataset.py`: Base class and core abstractions
- `hydrodataset/camels_*.py`: Individual dataset implementations (e.g., camels_us.py, camels_aus.py)
- `hydrodataset/__init__.py`: Package initialization, loads SETTING from hydro_setting.yml
- `examples/read_dataset.py`: CLI tool demonstrating usage for all datasets
- `tests/`: pytest test suite
- `docs/`: MkDocs documentation

## Implementation Guidelines

### Adding a New Dataset

1. Create `hydrodataset/dataset_name.py` extending `HydroDataset`
2. Implement required properties: `_attributes_cache_filename`, `_timeseries_cache_filename`, `default_t_range`
3. Initialize AquaFetch backend in `__init__`
4. Define `_subclass_static_definitions` for static variable mappings
5. Define `_dynamic_variable_mapping` for timeseries variable mappings
6. Add dataset to `DATASET_MAPPING` in `examples/read_dataset.py`
7. Write tests in `tests/test_dataset_name.py`

### Code Style

- Line length: 88 characters (Black default)
- Python 3.10+ required
- Use type hints where appropriate
- Follow existing patterns in camels_us.py and camels_aus.py for consistency

### NetCDF Caching

**Cache Mechanism:**
- Cache files stored in `CACHE_DIR` (from hydro_setting.yml)
- Separate files for attributes and timeseries (e.g., `camels_us_attributes.nc`, `camels_us_timeseries.nc`)
- Files named according to `_attributes_cache_filename` and `_timeseries_cache_filename` properties

**Data Flow:**
1. **First Access**: If cache file doesn't exist:
   - Call `cache_attributes_xrdataset()` or `cache_timeseries_xrdataset()`
   - Fetch raw data via AquaFetch backend
   - Standardize variable names and units
   - Clean feature names: `_clean_feature_names()` removes units, converts to lowercase, keeps only alphanumeric + underscore
   - Save as NetCDF to cache directory
2. **Subsequent Access**:
   - Fast read from cache using `xarray.open_dataset()`
   - No re-processing needed

**Cache File Management:**
- Delete cache files manually to force regeneration (useful after data updates)
- Cache files can be large (100s of MB to GBs depending on dataset)
- Ensure sufficient disk space in cache directory

### Unit Handling

- Units specified in variable mappings
- Stored as xarray attributes: `ds[var].attrs["units"]`
- Use pint-compatible unit strings (e.g., "mm/day", "km^2", "°C")

## Working with Datasets

### Exploring a Dataset

When working with a new or unfamiliar dataset:

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Initialize dataset
ds = CamelsUs(SETTING["local_data_path"]["datasets-origin"])

# 1. Check available features
print("Static features:", ds.available_static_features)
print("Dynamic features:", ds.available_dynamic_features)

# 2. Get basin/station IDs
basin_ids = ds.read_object_ids()
print(f"Number of basins: {len(basin_ids)}")

# 3. Check default time range
print(f"Default time range: {ds.default_t_range}")

# 4. Read a small sample to understand data structure
sample_attr = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:2],
    var_lst=["area", "p_mean"]
)
print(sample_attr)

sample_ts = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:2],
    t_range=["1990-01-01", "1990-01-07"],
    var_lst=["streamflow", "precipitation"]
)
print(sample_ts)
```

### Understanding Variable Mapping

Each dataset maps its specific variable names to standardized names:

```python
# Check the mapping in the dataset class definition
# Example from camels_aus.py:
_dynamic_variable_mapping = {
    StandardVariable.STREAMFLOW: {
        "default_source": "bom",
        "sources": {
            "bom": {"specific_name": "q_cms_obs", "unit": "mm^3/s"},
            "gr4j": {"specific_name": "streamflow_mld_inclinfilled", "unit": "ML/day"},
        }
    }
}

# This means you can request 'streamflow' and optionally specify source
ds.read_ts_xrdataset(..., var_lst=["streamflow"])  # Uses default source (bom)
ds.read_ts_xrdataset(..., var_lst=[("streamflow", "gr4j")])  # Uses gr4j source
```

## Testing

**Prerequisites:**
- Ensure `~/hydro_setting.yml` is properly configured with valid paths
- Datasets must be downloaded to `datasets-origin` directory before running tests
- Tests will create cache files in the `cache` directory on first run

**Test Structure:**
- Test files follow pytest conventions: `test_*.py` with `test_*()` functions
- Tests verify:
  - Data reading from cache and source
  - Variable name standardization
  - NetCDF caching behavior
  - Consistency between standardized API and raw data

**Running Specific Tests:**
```bash
# Test a specific dataset
pytest tests/test_camels_series.py::test_read_camels_aus_attr_xrdataset -v

# Run all tests for a specific file
pytest tests/test_camels_series.py -v
```

## Dependencies

- **AquaFetch**: Installed from GitHub dev branch (see `pyproject.toml` `[tool.uv.sources]`)
- Key packages: xarray, netCDF4, pandas, numpy, pint (for units)
- Optional: dask for large dataset handling

## Common Issues & Debugging

**Cache Issues:**
- If data appears incorrect, try deleting cache files in `CACHE_DIR` and regenerating
- Cache files named `{dataset}_attributes.nc` and `{dataset}_timeseries.nc`
- First access after cache deletion will be slower (regenerates cache)

**Missing Variables:**
- Check `available_static_features` and `available_dynamic_features` properties
- Use standardized names from `StandardVariable` class, not dataset-specific names
- For dynamic variables with multiple sources (e.g., CAMELS-AUS streamflow), specify source in method call

**Configuration Errors:**
- Ensure `~/hydro_setting.yml` exists in home directory (not project directory)
- Paths can be absolute or relative; use raw strings or forward slashes on Windows
- Both `datasets-origin` and `cache` paths must be writable

## Git Workflow

### Branch Management
- Main branch: `main` (stable releases)
- Development branch: `dev` (active development)
- Feature branches: create from `dev`, merge back to `dev` when ready

### Common Git Operations
```bash
# Check current status
git status

# Create a feature branch
git checkout -b feature/your-feature-name

# Stage and commit changes
git add <files>
git commit -m "Descriptive commit message"

# Push to remote
git push origin <branch-name>

# Update from remote dev branch
git checkout dev
git pull origin dev
```

### Pre-commit Checklist
Before committing, ensure:
1. Code is formatted with black: `black hydrodataset tests`
2. Linting passes: `flake8 hydrodataset tests`
3. Tests pass: `pytest`
4. Documentation builds (if modified): `mkdocs build`
