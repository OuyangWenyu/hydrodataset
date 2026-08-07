# hydrodataset

## Overview

**hydrodataset** is a Python package for accessing hydrological datasets with a unified API. It provides a standardized interface for reading diverse hydrological datasets, serving as a data-adapting layer on top of AquaFetch.

## Key Features

- **Unified API**: Consistent interface across all datasets
- **Standardized Variables**: Common variable names across different datasets
- **NetCDF Caching**: Fast data access through cached `.nc` files
- **Local & Cloud Access**: Read from local disk or S3-compatible object storage (e.g. Alibaba Cloud OSS), with automatic Zarr caching in the cloud
- **Multiple Data Sources**: Support for alternative data sources within datasets
- **Unit Management**: Automatic unit handling with pint integration

## Core Components

### Base Classes

- **[HydroDataset](hydro_dataset.md)**: Abstract base class for all dataset implementations
- **[StandardVariable](standard_variables.md)**: Standardized variable name constants

## Supported Datasets

### CAMELS Series
Continental-scale hydrological datasets from different regions:
- CAMELS-AUS, CAMELS-BR, CAMELS-CH, CAMELS-CL, CAMELS-COL
- CAMELS-DE, CAMELS-DK, CAMELS-FI, CAMELS-FR, CAMELS-GB
- CAMELS-IND, CAMELS-LUX, CAMELS-NZ, CAMELS-SE, CAMELS-US

### CAMELSH Series
Hourly resolution CAMELS datasets:
- CAMELSH (Hourly US data)
- CAMELSH-KR (South Korea)

### Caravan Series
Global datasets from Caravan project:
- Caravan-DK (Denmark)
- GRDC-Caravan (Global Runoff Data Centre)

### LamaH Series
Large-sample hydrological datasets:
- LamaH-CE (Central Europe)
- LamaH-ICE (Iceland)

### Other Datasets
- BULL (Spain)
- E-STREAMS (Europe)
- HYSETS (North America)
- SIMBI (Multiple regions)

## Quick Start

```python
from hydrodataset import open_dataset

# source="local" (default) reads from storage.local.root in ~/hydro_setting.yml;
# source="cloud" reads from storage.s3 (S3/OSS) with Zarr caches on the bucket.
ds = open_dataset("camels_us", source="local")

# Read basin IDs
basin_ids = ds.read_object_ids()

# Read timeseries data
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"]
)

# Read attribute data
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:5],
    var_lst=["area", "p_mean"]
)
```

## Configuration

Create `~/hydro_setting.yml` in your home directory (the `s3` block is only needed for cloud access):

```yaml
storage:
  default_source: local         # local | cloud — used when `source` is omitted
  local:
    root: D:/data/hydrodatasets # absolute local path; must exist
  cache: data/cache             # optional; relative to local.root
  s3:
    bucket: hydrodataset        # required for cloud access
    prefix: ""                  # optional prefix inside the bucket
    endpoint_url: https://oss-cn-beijing.aliyuncs.com
    access_key_id: <your-access-key>
    secret_access_key: <your-secret-key>
```

## Module Reference

::: hydrodataset
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
