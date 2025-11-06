# Caravan-DK

## Overview

**Caravan-DK** is the Denmark dataset from the Caravan project. Danish subset of the global Caravan dataset, providing standardized hydrological data for Danish catchments.

## Dataset Information

- **Region**: Denmark
- **Project**: Caravan
- **Module**: `hydrodataset.caravan_dk`
- **Class**: `CaravanDK`

## About Caravan

The Caravan project provides a global, standardized dataset of catchment attributes and meteorological forcings for large-sample hydrology. It combines data from multiple sources to create a unified dataset for hydrological modeling and analysis.

### Key Characteristics
- Standardized variable naming across regions
- Quality-controlled data
- Comprehensive catchment attributes
- Multiple meteorological data sources
- Suitable for machine learning applications

## Features

### Static Attributes
Static catchment attributes include:
- Basin area and geometry
- Topographic characteristics
- Land cover information
- Soil properties
- Climate indices
- Human impact indicators

### Dynamic Variables
Timeseries variables available:
- Streamflow (observed)
- Precipitation (multiple sources)
- Temperature (min, max, mean)
- Potential evapotranspiration
- Solar radiation
- Snow water equivalent
- And more...

## Usage

### Basic Usage

```python
from hydrodataset.caravan_dk import CaravanDK
from hydrodataset import SETTING

# Initialize dataset
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = CaravanDK(data_path)

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

### Working with Multiple Data Sources

Caravan datasets often provide multiple precipitation and temperature sources:

```python
# Compare different precipitation products
precip_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:3],
    t_range=["2000-01-01", "2005-12-31"],
    var_lst=[
        ("precipitation", "era5"),
        ("precipitation", "mswep"),
        ("precipitation", "chirps")
    ]
)

# Use specific meteorological forcing
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=["2000-01-01", "2010-12-31"],
    var_lst=[
        "streamflow",
        ("precipitation", "era5land"),
        ("temperature_mean", "era5land")
    ]
)
```

### Reading Specific Variables

```python
# Read with specific time range
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:10],
    t_range=["1990-01-01", "2020-12-31"],
    var_lst=["streamflow", "precipitation", "temperature_mean", "pet"]
)

# Read basin area
areas = ds.read_area(gage_id_lst=basin_ids[:10])

# Read mean precipitation
mean_precip = ds.read_mean_prcp(gage_id_lst=basin_ids[:10])
```

## Data Quality

Caravan datasets undergo quality control:
- Removal of unrealistic values
- Gap filling documentation
- Metadata completeness checks
- Cross-validation with regional datasets

## API Reference

::: hydrodataset.caravan_dk.CaravanDK
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
