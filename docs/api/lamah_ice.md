# LamaH-ICE

## Overview

**LamaH-ICE** is the Iceland large-sample hydrological dataset. Large-sample hydrological dataset for Iceland, featuring volcanic and glacial-influenced catchments with unique characteristics.

## Dataset Information

- **Region**: Iceland
- **Project**: LamaH (Large-sample hydrological data and models)
- **Module**: `hydrodataset.lamah_ice`
- **Class**: `LamahIce`

## About LamaH

LamaH (Large-sample hydrological data and models) provides comprehensive hydrological data for research and modeling:

### Key Features
- High-quality, quality-controlled data
- Extensive catchment attributes
- Multiple temporal resolutions
- Detailed metadata
- Suitable for large-sample hydrology studies

### Research Applications
- Hydrological model development and testing
- Climate change impact studies
- Regionalization studies
- Machine learning applications
- Comparative hydrology

## Features

### Static Attributes
Comprehensive static catchment attributes:
- Basin geometry and area
- Topographic characteristics (elevation, slope)
- Land cover information
- Soil properties and classes
- Geological characteristics
- Climate indices
- Human influence indicators

### Dynamic Variables
Timeseries variables available:
- Streamflow (observed)
- Precipitation
- Temperature (min, max, mean)
- Potential evapotranspiration
- Snow water equivalent
- Solar radiation
- Humidity
- And more...

## Usage

### Basic Usage

```python
from hydrodataset.lamah_ice import LamahIce
from hydrodataset import SETTING

# Initialize dataset
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = LamahIce(data_path)

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

### Advanced Analysis

```python
# Read multiple variables for detailed analysis
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:10],
    t_range=["1990-01-01", "2020-12-31"],
    var_lst=[
        "streamflow",
        "precipitation", 
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "pet",
        "snow_water_equivalent"
    ]
)

# Analyze snow-influenced catchments
import xarray as xr
winter_months = ts_data.sel(time=ts_data.time.dt.month.isin([12, 1, 2]))
mean_swe = winter_months["snow_water_equivalent"].mean(dim="time")
print("Mean winter SWE:", mean_swe)
```

### Reading Specific Variables

```python
# Read with specific time range
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:10],
    t_range=["2000-01-01", "2010-12-31"],
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)

# Read basin area
areas = ds.read_area(gage_id_lst=basin_ids[:10])

# Read mean precipitation
mean_precip = ds.read_mean_prcp(gage_id_lst=basin_ids[:10])
```

## Data Quality and Completeness

LamaH datasets feature:
- Rigorous quality control procedures
- Documentation of data gaps
- Metadata completeness
- Peer-reviewed methodology
- Regular updates

## Regional Characteristics

### LamaH-CE
- Alpine and pre-Alpine catchments
- Snow-influenced hydrology
- Elevation range from lowlands to high mountains
- Mixed land use patterns

### LamaH-ICE  
- Volcanic landscapes
- Glacial-influenced catchments
- Geothermal activity impact
- Unique geological conditions

## API Reference

::: hydrodataset.lamah_ice.LamahIce
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
