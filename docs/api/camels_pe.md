# CAMELS-PE

## Overview

**CAMELS-PE** is the Peru hydrological dataset implementation, providing CAMELS-style daily hydrometeorological time series and catchment attributes for 136 catchments in Peru (Llauca et al., 2026).

## Dataset Information

- **Region**: Peru
- **Module**: `hydrodataset.camels_pe`
- **Class**: `CamelsPe`
- **Backend**: `aqua_fetch.CAMELS_PE` (available since aqua-fetch 1.1.0)
- **Download source**: [Zenodo 21195425](https://zenodo.org/records/21195425) (~121 MB)

## Features

### Static Attributes
Static catchment attributes include:
- Basin area
- Mean precipitation
- Topographic characteristics
- Land cover information
- Soil properties
- Geological and human-intervention attributes
- Gauge metadata (name, region, record period)

### Dynamic Variables
Timeseries variables available:
- Streamflow (observed)
- Precipitation
- Temperature (min, max, mean)
- Potential evapotranspiration
- Solar radiation
- Vapor pressure

## Usage

### Basic Usage

```python
from hydrodataset.camels_pe import CamelsPe
from hydrodataset import resolve_data_path

# Initialize dataset (first access triggers a ~121 MB download)
data_path = resolve_data_path("camels_pe")
ds = CamelsPe(data_path)

# Get basin IDs
basin_ids = ds.read_object_ids()
print(f"Number of basins: {len(basin_ids)}")

# Check available features
print("Static features:", ds.available_static_features)
print("Dynamic features:", ds.available_dynamic_features)

# Read timeseries data
timeseries = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=ds.default_t_range,
    var_lst=["streamflow", "precipitation"]
)
print(timeseries)

# Read attribute data
attributes = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:5],
    var_lst=["area", "p_mean"]
)
print(attributes)
```

### Reading Specific Variables

```python
# Read with specific time range
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:10],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)

# Read basin area
areas = ds.read_area(gage_id_lst=basin_ids[:10])

# Read mean precipitation
mean_precip = ds.read_mean_prcp(gage_id_lst=basin_ids[:10])
```

## Data Sources

The dataset supports multiple data sources for certain variables. Check the class documentation for available sources and use tuple notation to specify:

```python
# Request specific data source
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=[
        ("precipitation", "pisco"),  # Specify PISCO source
        "streamflow"  # Use default source
    ]
)
```

## API Reference

::: hydrodataset.camels_pe.CamelsPe
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - __init__
        - read_object_ids
        - read_ts_xrdataset
        - read_attr_xrdataset
        - read_constant_cols
        - read_relevant_cols
        - read_target_cols
        - available_static_features
        - available_dynamic_features
        - default_t_range
