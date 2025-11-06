# CAMELS-SE

## Overview

**CAMELS-SE** is the Sweden hydrological dataset implementation. Swedish CAMELS dataset for Scandinavian catchments.

## Dataset Information

- **Region**: Sweden
- **Module**: `hydrodataset.camels_se`
- **Class**: `CamelsSe`

## Features

### Static Attributes
Static catchment attributes include:
- Basin area
- Mean precipitation
- Topographic characteristics
- Land cover information
- Soil properties
- Climate indices

### Dynamic Variables
Timeseries variables available (varies by dataset):
- Streamflow
- Precipitation
- Temperature (min, max, mean)
- Potential evapotranspiration
- Solar radiation
- And more...

## Usage

### Basic Usage

```python
from hydrodataset.camels_se import CamelsSe
from hydrodataset import SETTING

# Initialize dataset
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = CamelsSe(data_path)

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
        ("precipitation", "era5land"),  # Specify ERA5-Land source
        "streamflow"  # Use default source
    ]
)
```

## API Reference

::: hydrodataset.camels_se.CamelsSe
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - __init__
        - read_object_ids
        - read_ts_xrdataset
        - read_attr_xrdataset
        - available_static_features
        - available_dynamic_features
        - default_t_range
