# CAMELSH-KR

## Overview

**CAMELSH-KR** is the South Korea hourly hydrological dataset. Hourly resolution hydrological dataset for South Korean catchments with monsoon climate characteristics.

## Dataset Information

- **Region**: South Korea
- **Temporal Resolution**: Hourly
- **Module**: `hydrodataset.camelsh_kr`
- **Class**: `CamelshKr`

## Key Features

### Hourly Resolution
Unlike daily CAMELS datasets, CAMELSH provides hourly timeseries data, enabling:
- Sub-daily hydrological process analysis
- Flash flood and storm event studies
- High-frequency streamflow dynamics
- Detailed precipitation event analysis

### Static Attributes
Static catchment attributes include:
- Basin area
- Mean precipitation
- Topographic characteristics
- Land cover information
- Soil properties
- Climate indices

### Dynamic Variables
Hourly timeseries variables available:
- Streamflow (hourly)
- Precipitation (hourly)
- Temperature
- Potential evapotranspiration
- Solar radiation
- And more...

## Usage

### Basic Usage

```python
from hydrodataset.camelsh_kr import CamelshKr
from hydrodataset import SETTING

# Initialize dataset
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = CamelshKr(data_path)

# Get basin IDs
basin_ids = ds.read_object_ids()
print(f"Number of basins: {len(basin_ids)}")

# Check available features
print("Static features:", ds.available_static_features)
print("Dynamic features:", ds.available_dynamic_features)

# Read hourly timeseries data
timeseries = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=["2015-01-01", "2015-01-31"],  # One month of hourly data
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

### Analyzing Storm Events

```python
# Read hourly data for a specific storm event
storm_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:3],
    t_range=["2015-06-15 00:00:00", "2015-06-20 23:00:00"],
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)

# Analyze sub-daily patterns
import xarray as xr
hourly_precip = storm_data["precipitation"]
daily_total = hourly_precip.resample(time="1D").sum()
print("Daily precipitation totals:", daily_total)
```

### Reading Specific Variables

```python
# Read with specific time range (note hourly timestamps)
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:10],
    t_range=["2015-01-01 00:00:00", "2015-12-31 23:00:00"],
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)

# Read basin area
areas = ds.read_area(gage_id_lst=basin_ids[:10])

# Read mean precipitation
mean_precip = ds.read_mean_prcp(gage_id_lst=basin_ids[:10])
```

## Data Considerations

### Large Data Volumes
Hourly data results in significantly larger datasets compared to daily data:
- 24x more data points per day
- Larger cache files
- Longer initial cache generation time

### Time Range Selection
When working with hourly data:
```python
# Specify full datetime for hourly data
t_range = ["2015-01-01 00:00:00", "2015-01-31 23:00:00"]

# Or use date strings (defaults to 00:00:00)
t_range = ["2015-01-01", "2015-01-31"]
```

## API Reference

::: hydrodataset.camelsh_kr.CamelshKr
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
