# LamaH-CE

## Overview

**LamaH-CE** is the Central Europe large-sample hydrological dataset. Large-sample hydrological dataset for Central Europe, covering diverse Alpine and pre-Alpine catchments with high-quality data.

## Dataset Information

- **Region**: Central Europe
- **Project**: LamaH (Large-sample hydrological data and models)
- **Module**: `hydrodataset.lamah_ce`
- **Class**: `LamahCe`

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
from hydrodataset.lamah_ce import LamahCe
from hydrodataset import SETTING

# Initialize dataset
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = LamahCe(data_path)

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

### Station Connectivity Data

LamaH-CE provides stream network connectivity information between gauging stations. This data is essential for hydrological routing models and understanding the river network topology.

#### Connectivity Variables

The station connectivity data includes the following variables (from `Stream_dist.csv`):

| Variable | Description | Unit |
|----------|-------------|------|
| `NEXTDOWNID` | ID of the next downstream gauge (only one); 0 indicates no downstream gauge | - |
| `dist_hdn` | Horizontal stream length from the actual gauge to the next downstream gauge | km |
| `elev_diff` | Elevation difference from the actual gauge's zero point to the next downstream gauge's zero point | m |
| `strm_slope` | Slope of the actual gauge to the next downstream gauge; fraction of `elev_diff` and `dist_hdn` | m km⁻¹ |

#### Basic Usage

```python
from hydrodataset.lamah_ce import LamahCe
from hydrodataset import SETTING

# Initialize dataset
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = LamahCe(data_path)

# Cache the station connectivity data (only needed once)
# This reads Stream_dist.csv and saves it as NetCDF
ds.cache_stations_xrdataset()

# Read station connectivity for specific stations
stations = ds.read_stations_xrdataset(
    station_id_lst=["3", "4"]
)
print(stations)

# Example output:
# <xarray.Dataset>
# Dimensions:     (ID: 2)
# Coordinates:
#   * ID          (ID) <U3 '3' '4'
# Data variables:
#     NEXTDOWNID  (ID) <U3 '2' '3'
#     dist_hdn    (ID) <U18 '8.9' '12.3'
#     elev_diff   (ID) <U5 '45.0' '32.0'
#     strm_slope  (ID) <U19 '5.06' '2.60'

# Read all station connectivity data (no filter)
all_stations = ds.read_stations_xrdataset()
print(f"Total stations: {all_stations.dims['ID']}")
```

#### Finding Downstream Stations

```python
# Find downstream station for a specific gauge
station_id = "114"
station_data = ds.read_stations_xrdataset(station_id_lst=[station_id])
downstream_id = station_data["NEXTDOWNID"].values[0]
print(f"Downstream station of {station_id}: {downstream_id}")
```

#### Network Traversal Example

```python
def get_downstream_path(ds, start_station_id, max_depth=10):
    """Get the downstream path from a starting station.

    Args:
        ds: LamahCe dataset instance
        start_station_id: Starting station ID
        max_depth: Maximum number of downstream stations to traverse

    Returns:
        List of station IDs from start to outlet
    """
    path = [start_station_id]
    current_id = start_station_id

    for _ in range(max_depth):
        station_data = ds.read_stations_xrdataset(station_id_lst=[current_id])
        next_id = station_data["NEXTDOWNID"].values[0]

        if next_id == "0":  # No downstream gauge (outlet)
            break

        path.append(next_id)
        current_id = next_id

    return path

# Example: trace downstream from station 114
downstream_path = get_downstream_path(ds, "114")
print(f"Downstream path: {' -> '.join(downstream_path)}")
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

::: hydrodataset.lamah_ce.LamahCe
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - __init__
        - read_object_ids
        - read_ts_xrdataset
        - read_attr_xrdataset
        - cache_stations_xrdataset
        - read_stations_xrdataset
        - available_static_features
        - available_dynamic_features
        - default_t_range
