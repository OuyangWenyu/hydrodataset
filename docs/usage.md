# Usage Guide

This guide provides comprehensive examples for using hydrodataset in your projects.

## Basic Setup

### Configuration File

First, ensure you have a `hydro_setting.yml` file in your home directory:

```yaml
local_data_path:
  root: 'D:\data\waterism'                    # Your root data directory
  datasets-origin: 'D:\data\waterism\datasets-origin'  # Raw data location
  cache: 'D:\data\waterism\cache'             # Cache directory for .nc files
```

### Import and Initialize

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Access configured paths
data_path = SETTING["local_data_path"]["datasets-origin"]

# Initialize dataset
ds = CamelsUs(data_path)
```

## Exploring Available Data

### Check Available Features

```python
# List all static (attribute) features
print("Static features:")
print(ds.available_static_features)

# List all dynamic (timeseries) features
print("Dynamic features:")
print(ds.available_dynamic_features)
```

### Get Basin/Station IDs

```python
# Get all available basin IDs
basin_ids = ds.read_object_ids()
print(f"Total basins: {len(basin_ids)}")
print(f"First 5 basins: {basin_ids[:5]}")
```

## Reading Data

### Read Static Attributes

Static attributes are catchment characteristics that don't change over time:

```python
# Read specific attributes for selected basins
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=["01013500", "01022500"],  # Basin IDs
    var_lst=["area", "p_mean", "elev_mean"]  # Attribute names
)

print(attr_data)
# Output is an xarray.Dataset with dimensions [basin, variable]
```

### Read Timeseries Data

Timeseries data includes streamflow, precipitation, temperature, etc.:

```python
# Read timeseries for specific basins and time period
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=["01013500", "01022500"],
    t_range=["1990-01-01", "1995-12-31"],  # Start and end dates
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)

print(ts_data)
# Output is an xarray.Dataset with dimensions [basin, time, variable]
```

## Advanced Usage

### Using Multiple Data Sources

Some datasets provide the same variable from different sources:

```python
from hydrodataset.camels_aus import CamelsAus

ds_aus = CamelsAus(data_path)

# Use default streamflow source (BOM)
ts_data_bom = ds_aus.read_ts_xrdataset(
    gage_id_lst=["A4260522"],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow"]  # Uses default source
)

# Explicitly specify GR4J model output
ts_data_gr4j = ds_aus.read_ts_xrdataset(
    gage_id_lst=["A4260522"],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=[("streamflow", "gr4j")]  # Tuple: (variable, source)
)
```

### Reading All Basins

```python
# Read all available basins
all_basin_ids = ds.read_object_ids()

# Read attributes for all basins
all_attrs = ds.read_attr_xrdataset(
    gage_id_lst=all_basin_ids,
    var_lst=["area", "p_mean"]
)
```

### Selective Basin Filtering

```python
import numpy as np

# Get all basins
all_basins = ds.read_object_ids()

# Read areas for all basins
areas = ds.read_attr_xrdataset(
    gage_id_lst=all_basins,
    var_lst=["area"]
)

# Filter basins by area (e.g., > 1000 km²)
large_basins = all_basins[areas['area'].values > 1000]

# Read timeseries only for large basins
ts_large = ds.read_ts_xrdataset(
    gage_id_lst=large_basins.tolist(),
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["streamflow"]
)
```

## Working with xarray Datasets

hydrodataset returns data as `xarray.Dataset` objects, which provide powerful data manipulation capabilities:

### Basic Operations

```python
# Select specific basin
basin_data = ts_data.sel(basin="01013500")

# Select time range
period_data = ts_data.sel(time=slice("1992-01-01", "1993-12-31"))

# Access specific variable
streamflow = ts_data["streamflow"]

# Convert to numpy array
streamflow_array = streamflow.values

# Convert to pandas DataFrame
streamflow_df = streamflow.to_dataframe()
```

### Computations

```python
# Calculate mean streamflow for each basin
mean_flow = ts_data["streamflow"].mean(dim="time")

# Calculate annual maximum streamflow
annual_max = ts_data["streamflow"].resample(time="1Y").max()

# Correlation between variables
import xarray as xr
correlation = xr.corr(
    ts_data["streamflow"],
    ts_data["precipitation"],
    dim="time"
)
```

## Integration with Deep Learning

### Converting to NumPy/PyTorch

```python
import torch
import numpy as np

# Get timeseries data
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:10],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["streamflow", "precipitation", "temperature_mean"]
)

# Convert to numpy
data_np = ts_data.to_array().values  # Shape: (variables, basins, time)

# Convert to PyTorch tensor
data_tensor = torch.from_numpy(data_np).float()

# Reshape for deep learning (e.g., [batch, time, features])
data_dl = data_tensor.permute(1, 2, 0)  # [basins, time, variables]
```

### Creating Training/Test Splits

```python
# Split by time
train_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=["1990-01-01", "2005-12-31"],  # Training period
    var_lst=["streamflow", "precipitation"]
)

test_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=["2006-01-01", "2010-12-31"],  # Test period
    var_lst=["streamflow", "precipitation"]
)
```

## Working Across Multiple Datasets

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset.camels_aus import CamelsAus

# Initialize multiple datasets
ds_us = CamelsUs(data_path)
ds_aus = CamelsAus(data_path)

# Read the same variables from different datasets using standardized names
us_data = ds_us.read_ts_xrdataset(
    gage_id_lst=["01013500"],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["streamflow", "precipitation"]
)

aus_data = ds_aus.read_ts_xrdataset(
    gage_id_lst=["A4260522"],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["streamflow", "precipitation"]  # Same variable names!
)
```

## Cache Management

### Understanding the Cache

hydrodataset caches processed data as NetCDF files for faster subsequent access:

```python
# First access: slow (processes raw data and creates cache)
ts_data = ds.read_ts_xrdataset(...)  # May take minutes

# Subsequent access: fast (reads from .nc cache)
ts_data = ds.read_ts_xrdataset(...)  # Instant!
```

### Regenerating Cache

If you need to regenerate the cache (e.g., after data updates):

```bash
# Navigate to cache directory (from hydro_setting.yml)
cd ~/data/cache

# Remove cache files for CAMELS-US
rm camels_us_timeseries.nc
rm camels_us_attributes.nc

# Next Python access will regenerate the cache
```

## Best Practices

### 1. Start Small, Scale Up

```python
# Test with few basins first
test_basins = basin_ids[:5]
test_data = ds.read_ts_xrdataset(
    gage_id_lst=test_basins,
    t_range=["2000-01-01", "2000-12-31"],
    var_lst=["streamflow"]
)

# Once working, scale to full dataset
```

### 2. Check Data Availability

```python
# Always check what's available before requesting
print(ds.available_dynamic_features)
print(ds.available_static_features)

# Check default time range
print(ds.default_t_range)
```

### 3. Handle Missing Data

```python
# Check for NaN values
has_nan = ts_data["streamflow"].isnull().any()

# Fill or drop NaN values
filled_data = ts_data.fillna(0)
dropped_data = ts_data.dropna(dim="time")
```

## Next Steps

- Explore the [API Reference](api/hydrodataset.md) for detailed method documentation
- Check the [FAQ](faq.md) for common questions and troubleshooting
- See [examples](https://github.com/OuyangWenyu/hydrodataset/tree/main/examples) in the repository
- Integrate with [torchhydro](https://github.com/OuyangWenyu/torchhydro) for deep learning workflows
