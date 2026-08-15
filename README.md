# hydrodataset

[![PyPI version](https://img.shields.io/pypi/v/hydrodataset.svg)](https://pypi.python.org/pypi/hydrodataset)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/hydrodataset.svg)](https://anaconda.org/conda-forge/hydrodataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://OuyangWenyu.github.io/hydrodataset)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A Python package for accessing hydrological datasets with a unified API, optimized for deep learning workflows.**

- 🌊 **Unified Interface**: Consistent API across 27 hydrological datasets
- ⚡ **Fast Access**: NetCDF caching locally / Zarr caching on cloud for instant data loading
- 🎯 **Standardized Variables**: Common naming across all datasets
- 🔗 **Built on AquaFetch**: Powered by the comprehensive [AquaFetch](https://github.com/hyex-research/AquaFetch) backend
- 📊 **ML-Ready**: Optimized for integration with [torchhydro](https://github.com/OuyangWenyu/torchhydro)

---


## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Local vs Cloud Data Access](#local-vs-cloud-data-access)
- [Supported Datasets](#supported-datasets)
- [Key Features](#key-features)
- [Project Status](#project-status)
- [Credits](#credits)

## Core Philosophy

This library has been redesigned to serve as a powerful data-adapting layer on top of the [AquaFetch](https://github.com/hyex-research/AquaFetch) package.

While `AquaFetch` handles the complexities of downloading and reading numerous public hydrological datasets, `hydrodataset` takes the next step: it standardizes this data into a clean, consistent format — NetCDF (`.nc`) locally, Zarr on cloud object storage — optimized for seamless integration with hydrological modeling libraries like [torchhydro](https://github.com/OuyangWenyu/torchhydro).

**One unified way to reach any dataset.** Every dataset is addressed by a logical id (`"camels_us"`, `"bull"`, …) resolved through a single chain:

```
~/hydro_setting.yml (storage config) → resolve_data_path / open_dataset → absolute path or s3:// URI
```

You never construct a data path by hand. `open_dataset(dataset_id, source="local"|"cloud")` resolves the id, picks the right reader class, and returns an instantiated dataset — `source` is chosen per call, or defaults to `storage.default_source`. When you need the raw path or a specific class directly, `resolve_data_path(dataset_id)` + the class constructor remain available (see [Quick Start](#quick-start)).

The core workflow is:
1.  **Resolve**: `resolve_data_path` / `open_dataset` turns a dataset id into an absolute local path or an `s3://` URI, using the config in `~/hydro_setting.yml`.
2.  **Standardize**: The `hydrodataset` reader (backed by `AquaFetch`) fetches raw data and exposes it through a consistent, unified interface across all datasets.
3.  **Cache**: On the first run, the data is processed into an `xarray.Dataset` and saved as `.nc` files (timeseries + attributes) in the local cache directory — or as Zarr stores on cloud storage for `source="cloud"`.
4.  **Access**: All subsequent requests read from the fast cache (NetCDF locally / Zarr on cloud), giving you analysis-ready data instantly.

## Installation

We strongly recommend using a virtual environment to manage dependencies.

### Using uv (Recommended)

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable package and environment management:

```bash
# Install uv if you haven't already
pip install uv

# Install hydrodataset with uv
uv pip install hydrodataset
```

For more advanced usage or to work on the project locally:

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/hydrodataset.git
cd hydrodataset

# Create virtual environment and install all dependencies
uv sync --all-extras
```

The `--all-extras` flag installs base dependencies plus all optional dependencies for development and documentation.

### Using pip (Alternative)

If you prefer traditional pip:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install hydrodataset
```

## Quick Start

The primary goal of `hydrodataset` is to provide a simple, unified API for accessing various hydrological datasets. Here's a complete example showing the core workflow:

> **⚠️ Important Note on First-Time Data Download**
>
> ***If you haven't pre-downloaded the datasets, the first access will trigger automatic downloads via AquaFetch, which can take considerable time depending on dataset size:***
>
> - **Small datasets** (< 1GB, e.g., CAMELS-CL, CAMELS-COL): ~10-30 minutes
> - **Medium datasets** (1-5GB, e.g., CAMELS-AUS, CAMELS-BR): ~30 minutes to 1 hour
> - **Large datasets** (10-20GB, e.g., CAMELS-US, LamaH-CE): ~1-3 hours
> - **Very large datasets** (> 30GB, e.g., HYSETS): ~3-6 hours or more
>
> Download times vary based on your internet connection speed and server availability.
>
> ***We strongly recommend downloading datasets manually during off-peak hours if possible.***
>
> After the initial download, all subsequent access will be fast thanks to NetCDF caching (locally) or Zarr caching (on cloud).

### Basic Example

```python
from hydrodataset import open_dataset

# open_dataset reads storage config from ~/hydro_setting.yml.
# Example hydro_setting.yml:
#
#   storage:
#     local:
#       root: D:/data/hydrodatasets
#
# source defaults to storage.default_source ("local" unless configured otherwise).
ds = open_dataset("camels_us")

# 1. Check which features are available
print("Available static features:")
print(ds.available_static_features)

print("Available dynamic features:")
print(ds.available_dynamic_features)

# 2. Get a list of all basin IDs
basin_ids = ds.read_object_ids()

# 3. Read static (attribute) data for a subset of basins
# Note: We use standardized names like 'area' and 'p_mean'
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:2],
    var_lst=["area", "p_mean"]
)
print("Static attribute data:")
print(attr_data)

# 4. Read dynamic (time-series) data for the same basins
# Note: We use standardized names like 'streamflow' and 'precipitation'
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:2],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"]
)
print("Time-series data:")
print(ts_data)
```

### Explicit construction (advanced)

`open_dataset` is the recommended entry point. If you need the resolved path
directly (e.g. to pass it somewhere) or want to instantiate a specific reader
class, use `resolve_data_path` + the class constructor — this is equivalent
and still fully supported:

```python
from hydrodataset import resolve_data_path
from hydrodataset.camels_us import CamelsUs

data_path = resolve_data_path("camels_us")   # absolute local path (or s3:// URI)
ds = CamelsUs(data_path)                     # same object as open_dataset("camels_us")
```

### Standardized Variable Names

A key feature of the new architecture is the use of standardized variable names. This allows you to use the same variable name to fetch the same type of data across different datasets, without needing to know the specific, internal naming scheme of each one.

For example, you can get streamflow from both CAMELS-US and CAMELS-AUS using the same variable name:

```python
# Get streamflow from CAMELS-US
us_ds.read_ts_xrdataset(gage_id_lst=["01013500"], var_lst=["streamflow"], t_range=["1990-01-01", "1995-12-31"])

# Get streamflow from CAMELS-AUS
aus_ds.read_ts_xrdataset(gage_id_lst=["A4260522"], var_lst=["streamflow"], t_range=["1990-01-01", "1995-12-31"])
```

Similarly, you can use `precipitation`, `temperature_max`, etc., across datasets. See [Standard Variables](api/standard_variables.md) for the comprehensive list of standardized names and their coverage across datasets.

## Local vs Cloud Data Access

hydrodataset can read the same datasets from either a local disk or cloud object storage (S3-compatible, e.g. Alibaba Cloud OSS). The backend is chosen per call with `source="local" | "cloud"`; when omitted, `storage.default_source` from `~/hydro_setting.yml` is used (default: `local`).

Both backends share one configuration file, `~/hydro_setting.yml`:

```yaml
storage:
  default_source: local         # local | cloud — used when `source` is omitted
  local:
    root: D:/data/hydrodatasets # absolute local path; must exist
  cache: data/cache             # optional; relative paths resolve against local.root
  s3:
    bucket: hydrodataset        # required for cloud access
    prefix: ""                  # optional prefix inside the bucket
    endpoint_url: https://oss-cn-beijing.aliyuncs.com
    access_key_id: <your-access-key>
    secret_access_key: <your-secret-key>
```

**Local**

- `resolve_data_path("camels_us", source="local")` returns an absolute local path under `storage.local.root`.
- Readers cache analysis-ready data as NetCDF files (`{dataset}_timeseries.nc`, `{dataset}_attributes.nc`) in the cache directory (`storage.cache`, default `~/.cache/hydrodataset`). Missing caches are generated automatically on first read.

**Cloud**

- `resolve_data_path("camels_us", source="cloud")` returns an S3 URI such as `s3://hydrodataset/`.
- Readers access the raw dataset directly on OSS via s3fs and cache analysis-ready data as Zarr stores at `s3://<bucket>/zarr/{dataset}_timeseries.zarr` and `..._attributes.zarr` (with consolidated metadata). Missing Zarr stores are generated automatically on first read — typically on a cloud VM (ECS) using the internal OSS endpoint for bandwidth.

**Recommended usage**: `open_dataset(dataset_id, source=...)` — one call to resolve and construct, `source` picks the backend per call. `resolve_data_path(dataset_id, source=...)` is for when you need the raw path or URI explicitly (e.g. to inspect or pass it on). Both share the same `source` semantics:

```python
from hydrodataset import resolve_data_path, open_dataset

# Local
local_uri = resolve_data_path("camels_us", source="local")
ds = open_dataset("camels_us", source="local")
ts = ds.read_ts_xrdataset(
    gage_id_lst=["01013500"],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"],
)

# Cloud
cloud_uri = resolve_data_path("camels_us", source="cloud")
print(cloud_uri)  # e.g. s3://hydrodataset/
ds_cloud = open_dataset("camels_us", source="cloud")
ts_cloud = ds_cloud.read_ts_xrdataset(
    gage_id_lst=["01013500"],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow"],
)
```

The CLI exposes the same `--source` switch:

```bash
hydrodataset config                                    # show effective config (secrets masked)
hydrodataset resolve camels_us --source local
hydrodataset resolve camels_us --source cloud
hydrodataset info camels_us --source cloud
hydrodataset read-ts bull --source cloud --gages BULL_10004 --vars precipitation -o ts.nc
```

Notes:

- The config file lives in your home directory (`~/hydro_setting.yml`). A project-level `.hydro_setting.yml` in the project root is also supported and overrides user-level settings.
- `storage.s3.*` contains credentials — never commit it to a repository.
- Readers constructed with an `s3://` URI skip local path validation and are treated as cloud readers (`_is_cloud()`).

## Supported Datasets

hydrodataset currently provides unified access to **27 hydrological datasets** across the globe. Below is a summary of all supported datasets:

| Dataset Name | Paper | Temporal Resolution | Data Version | Region | Basins | Time Span | Release Date | Size |
|-------------|-------|---------------------|--------------|---------|---------|-----------|--------------|------|
| BULL | [Paper](https://www.nature.com/articles/s41597-024-03594-5) / [Code](https://github.com/UW-Hydro/VIC/tree/support/VIC.4.2.d) | Daily | [Version 3](https://zenodo.org/records/10844207) (code) / [Version 2](https://zenodo.org/records/10629809) (data) | Spain | 484 | 1951-01-02 to 2021-12-31 | 2024-03-10 | 2.2G |
| CAMELS-AUS | [Paper (V1)](https://essd.copernicus.org/articles/13/3847/2021/) / [Paper (V2)](https://essd.copernicus.org/preprints/essd-2024-263/) | Daily | [Version 1](https://doi.pangaea.de/10.1594/PANGAEA.921850?format=html#download) / [Version 2](https://zenodo.org/records/14289037) | Australia | 561 | 1950-01-01 to 2022-03-31 | 2024-12 | 2.1G |
| CAMELS-BR | [Paper](https://doi.org/10.5194/essd-12-2075-2020) | Daily | [Version 1.2](https://zenodo.org/records/15025488) / [Version 1.1](https://zenodo.org/records/3964745#.YA6rUxZS-Uk) | Brazil | 897 | 1980-01-01 to 2024-10-22 | 2025-03-21 | 1.4G |
| CAMELS-CH | [Paper](https://essd.copernicus.org/articles/15/5755/2023/) | Daily | [Version 0.9](https://doi.org/10.5281/zenodo.7784632) / [Version 0.6](https://zenodo.org/records/7957061) | Switzerland | 331 | 1981-01-01 to 2020-12-31 | 2025-03-14 | 793.1M |
| CAMELS-CL | [Paper](https://hess.copernicus.org/articles/22/5817/2018) | Daily | [Dataset](https://doi.org/10.1594/PANGAEA.894885) | Chile | 516 | 1913-02-15 to 2018-03-09 | 2018-09-28 | 208M |
| CAMELS-COL | [Paper](https://essd.copernicus.org/preprints/essd-2025-200/) | Daily | [Version 2](https://zenodo.org/records/15554735) | Colombia | 347 | 1981-05 to 2022-12 | 2025-05 | 80.9M |
| CAMELS-DE | [Paper](https://essd.copernicus.org/preprints/essd-2024-318/) | Daily | [Version 1.1](https://zenodo.org/records/16755906) / [Version 0.1](https://zenodo.org/records/12733968) | Germany | 1582 | 1951-01-01 to 2020-12-31 | 2025-08-07 | 2.2G |
| CAMELS-DK | [Paper](https://essd.copernicus.org/preprints/essd-2024-292/) | Daily | [Version 6.0](https://dataverse.geus.dk/dataset.xhtml?persistentId=doi%3A10.22008%2FFK2%2FAZXSYP&fileAccess=Public) | Denmark | 304 | 1989-01-02 to 2023-12-31 | 2025-02-14 | 1.41G |
| CAMELS-FI | [Meeting](https://meetingorganizer.copernicus.org/EGU25/EGU25-18296.html?pdf) | Yearly/Daily | [Version 1.0.1](https://zenodo.org/records/16257216) | Finland | 320 | 1961-01-01 to 2023-12-31 | 2025-07 | 382M |
| CAMELS-FR | [Paper](https://essd.copernicus.org/articles/17/1461/2025/essd-17-1461-2025.html) | Daily/Monthly/Yearly | [Version 3.2](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/WH7FJR&version=3.2) / [Version 3](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/WH7FJR) | France | 654 | 1970-01-01 to 2021-12-31 | 2025-08-12 | 364M |
| CAMELS-GB | [Paper](https://essd.copernicus.org/articles/12/2459/2020/) | Daily | [Dataset](https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9) | United Kingdom | 671 | 1970-10-01 to 2015-09-30 | 2025-05 (new data link) | 244M |
| CAMELS-IND | [Paper](https://doi.org/10.5194/essd-17-461-2025) | Daily | [Version 2.2](https://zenodo.org/records/14999580) | India | 472 (242 sufficient flow) | 1980-01-01 to 2020-12-31 | 2025-03-13 | 529.4M |
| CAMELS-LUX | [Paper](https://essd.copernicus.org/preprints/essd-2024-482/) | Hourly/Daily | [Version 1.1](https://zenodo.org/records/14910359) | Luxembourg | 56 | 2004-11-01 to 2021-10-31 | 2024-09-27 | 1.4G |
| CAMELS-PE | [Paper](https://essd.copernicus.org/preprints/essd-2026-386/) | Daily | [Version 1.0.1](https://zenodo.org/records/21195425) | Peru | 136 | 1981-01-01 to 2025-12-31 | 2026-07-04 | 121.4M |
| CAMELS-NZ | [Paper](https://essd.copernicus.org/preprints/essd-2025-244/) | Hourly/Daily | [Version 2](https://figshare.canterbury.ac.nz/articles/dataset/CAMELS-NZ_Hydrometeorological_time_series_and_landscape_attributes_for_Aotearoa_New_Zealand/28827644) / [Version 1](https://figshare.canterbury.ac.nz/articles/dataset/CAMELS-NZ_Hydrometeorological_time_series_and_landscape_attributes_for_Aotearoa_New_Zealand/28827644/1) | New Zealand | 369 | 1972-01-01 to 2024-08-02 | 2025-08-05 | 4.81G |
| CAMELS-SE | [Paper](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/gdj3.239) | Daily | [Version 1](https://snd.se/sv/catalogue/dataset/2023-173/1) | Sweden | 50 | 1961-2020 | 2024-02 | 16.19M |
| CAMELS-US | [Paper](https://hess.copernicus.org/articles/21/5293/2017/) | Daily | [Version 1.2](https://zenodo.org/records/15529996) | United States | 671 | 1980-2014 | 2022-06-24 | 14.6G |
| CAMELSH-KR | - | Hourly | [Version 1](https://zenodo.org/records/15073264) | South Korea | 178 | 2000-2019 | 2025-03-23 | 3.1G |
| CAMELSH | [Paper](https://www.nature.com/articles/s41597-025-05612-6) | Hourly | [Version 6](https://zenodo.org/records/16729675) + [3](https://zenodo.org/records/15070091) + [2](https://zenodo.org/records/15066778) | United States | 9008 | 1980-2024 | 2025-08-14 | 4.2G+3.57G+2.18G |
| Caravan-DK | [Paper](https://essd.copernicus.org/articles/17/1551/2025/essd-17-1551-2025-discussion.html) | Daily | [Version 7](https://zenodo.org/records/15200118) / [Version 5](https://zenodo.org/records/7962379) | Denmark | 308 | 1981-01-02 to 2020-12-31 | 2025-04-11 | 521.6M |
| Caravan | [Paper](https://www.nature.com/articles/s41597-023-01975-w) / [Code](https://github.com/kratzert/Caravan/) | Daily | [Version 0.3](https://zenodo.org/record/7944025) | Global | 16299 | 1950-2023 | 2023-05 | 24.8G |
| EStream | [Paper](https://www.nature.com/articles/s41597-024-03706-1) / [Code](https://github.com/thiagovmdon/EStreams) | Daily (weekly, monthly, yearly available) | [Version 1.3](https://zenodo.org/records/15756335) / [Version 1.1](https://zenodo.org/records/13961394) | Europe | 17130 | 1950-01-01 to 2023-06-30 | 2025-06-30 | 12.3G |
| GRDC-Caravan | [Paper](https://essd.copernicus.org/preprints/essd-2024-427/) | Daily | [Version 0.6](https://zenodo.org/records/15349031) / [Version 0.2](https://zenodo.org/records/10074416) | Global | 5357 | 1950-2023 | 2025-05-06 | 16.4G |
| HYSETS | [Paper](https://www.nature.com/articles/s41597-020-00583-2) / [Code](https://github.com/dankovacek/hysets_validation) | Daily | [Dataset](https://osf.io/rpc3w/files) (dynamic attributes) | North America | 14425 | 1950-01-01 to 2023-12-31 | 2024-09 | 41.9G |
| LamaH-CE | [Paper](https://doi.org/10.5194/essd-13-4529-2021) | Daily/Hourly | [Version 1.0](https://zenodo.org/records/5153305) | Central Europe | 859 | 1981-01-01 to 2019-12-31 | 2021-08-02 | 16.3G |
| LamaH-Ice | [Paper](https://essd.copernicus.org/articles/16/2741/2024/) | Daily/Hourly | [Version 1.5](https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/) / [old version](https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/) | Iceland | 111 | 1950-01-01 to 2021-12-31 | 2025-08-12 | 9.6G |
| Simbi | [Paper](https://essd.copernicus.org/articles/16/2073/2024/) | Daily/Monthly | [Version 6.0](https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/02POK6) | Haiti | 24 | 1920-01-01 to 2005-12-31 | 2024-07-02 | 125M |
>

## Key Features

### 🎯 Unified API Across All Datasets

Access any dataset using the same method calls:
```python
# Same API works for all datasets
ds.read_object_ids()                          # Get basin IDs
ds.read_attr_xrdataset(...)                   # Read attributes
ds.read_ts_xrdataset(...)                     # Read timeseries
```

### ⚡ Fast Caching (NetCDF locally, Zarr on cloud)

First access processes and caches data; all subsequent reads are instant:
- **Local**: NetCDF files `{dataset}_timeseries.nc` / `{dataset}_attributes.nc`
- **Cloud**: Zarr stores `{dataset}_timeseries.zarr` / `{dataset}_attributes.zarr` (with consolidated metadata)
- Configured via `~/hydro_setting.yml` (`storage.cache` locally, `storage.s3` for cloud)

### 🔄 Standardized Variable Names

Use common names across all datasets:
- `streamflow` - River discharge
- `precipitation` - Rainfall
- `temperature_max` / `temperature_min` - Temperature extremes
- `potential_evapotranspiration` - PET
- And many more...

### 📊 xarray Integration

All data returned as `xarray.Dataset` objects:
- Labeled dimensions and coordinates
- Built-in metadata and units
- Easy slicing, selection, and computation
- Compatible with Dask for large datasets

### 🌐 Station Network Connectivity (LamaH-CE)

LamaH-CE dataset supports querying stream network topology between gauging stations:

```python
from hydrodataset import resolve_data_path
from hydrodataset.lamah_ce import LamahCe

ds = LamahCe(resolve_data_path("lamah_ce"))

# Read station connectivity data
stations = ds.read_stations_xrdataset(station_id_lst=["3", "4"])
print(stations)
# Returns: NEXTDOWNID, dist_hdn, elev_diff, strm_slope
```

See [LamaH-CE API documentation](api/lamah_ce.md#station-connectivity-data) for detailed variable descriptions and usage examples.


## Project Status

hydrodataset provides unified access to **27 hydrological datasets**, all implemented on the `HydroDataset` base class with the ADR 0001 path-resolution architecture (`resolve_data_path` → absolute URI → reader). CAMELS-US and CAMELS-AUS serve as the reference implementations; every other supported dataset follows the same pattern.

New datasets and features are added continuously. Please check the [Changelog](changelog.md) for the latest updates.

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template. The data fetching and reading is now powered by [AquaFetch](https://github.com/hyex-research/AquaFetch).
