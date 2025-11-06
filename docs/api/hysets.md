# HYSETS

## Overview

**HYSETS** is a hydrological dataset for North America. North American hydrological dataset with extensive coverage across Canada and parts of the United States.

## Dataset Information

- **Region**: North America
- **Module**: `hydrodataset.hysets`
- **Class**: `Hysets`

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
Timeseries variables available:
- Streamflow
- Precipitation  
- Temperature (min, max, mean)
- Potential evapotranspiration
- And additional variables depending on the dataset

## Usage

### Basic Usage

```python
from hydrodataset.hysets import Hysets
from hydrodataset import SETTING

# Initialize dataset
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = Hysets(data_path)

# Get basin IDs
basin_ids = ds.read_object_ids()
print(f"Number of basins: {len(basin_ids)}")

# Check available features
print("Static features:", ds.available_static_features)
print("Dynamic features:", ds.available_dynamic_features)

# Check default time range
print(f"Default time range: {ds.default_t_range}")

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
    t_range=["1990-01-01", "2020-12-31"],
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)

# Read basin area
areas = ds.read_area(gage_id_lst=basin_ids[:10])

# Read mean precipitation
mean_precip = ds.read_mean_prcp(gage_id_lst=basin_ids[:10])
```

### Working with Multiple Data Sources

If the dataset provides multiple sources for variables:

```python
# Request specific data source
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=[
        ("precipitation", "era5land"),  # Specify source
        "streamflow"  # Use default source
    ]
)
```

## API Reference

::: hydrodataset.hysets.Hysets
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
