# HydroDataset Base Class

## Overview

`HydroDataset` is the abstract base class that defines the unified interface for all hydrological dataset implementations in hydrodataset. It provides common functionality for data reading, caching, variable standardization, and unit management.

## Key Responsibilities

### Data Reading
- Read basin/station IDs
- Read timeseries data as xarray.Dataset
- Read attribute/static data as xarray.Dataset
- Support for selecting specific variables and time ranges

### NetCDF Caching
- Generate and cache timeseries data in `.nc` format
- Generate and cache attribute data in `.nc` format
- Fast subsequent reads from cached files
- Cache location configured in `hydro_setting.yml`

### Variable Standardization
- Map dataset-specific variable names to standard names
- Support multiple data sources for same variable
- Automatic unit conversion and tracking

### Feature Management
- List available static features (attributes)
- List available dynamic features (timeseries)
- Clean feature names for consistency

## Architecture

### Required Properties (Subclass Implementation)

Subclasses must implement these properties:

- **`_attributes_cache_filename`**: NetCDF filename for cached attributes (e.g., "camels_us_attributes.nc")
- **`_timeseries_cache_filename`**: NetCDF filename for cached timeseries (e.g., "camels_us_timeseries.nc")
- **`default_t_range`**: Default time range as `["YYYY-MM-DD", "YYYY-MM-DD"]`

### Optional Properties (Subclass Override)

- **`_subclass_static_definitions`**: Dictionary mapping standard static variable names to dataset-specific names and units
- **`_dynamic_variable_mapping`**: Dictionary mapping StandardVariable constants to dataset-specific timeseries variables

## Usage Pattern

### Creating a New Dataset Class

```python
from hydrodataset import HydroDataset, StandardVariable
from aqua_fetch import DatasetClass

class MyDataset(HydroDataset):
    def __init__(self, data_path, region=None, download=False):
        super().__init__(data_path)
        self.region = region
        self.download = download
        self.aqua_fetch = DatasetClass(data_path)

    @property
    def _attributes_cache_filename(self):
        return "mydataset_attributes.nc"

    @property
    def _timeseries_cache_filename(self):
        return "mydataset_timeseries.nc"

    @property
    def default_t_range(self):
        return ["1980-01-01", "2020-12-31"]

    # Define static variable mappings
    _subclass_static_definitions = {
        "area": {"specific_name": "area_km2", "unit": "km^2"},
        "p_mean": {"specific_name": "p_mean", "unit": "mm/day"},
    }

    # Define dynamic variable mappings
    _dynamic_variable_mapping = {
        StandardVariable.STREAMFLOW: {
            "default_source": "observations",
            "sources": {
                "observations": {"specific_name": "q_cms", "unit": "m^3/s"},
            },
        },
        StandardVariable.PRECIPITATION: {
            "default_source": "gauge",
            "sources": {
                "gauge": {"specific_name": "precip_mm", "unit": "mm/day"},
                "era5": {"specific_name": "tp", "unit": "mm/day"},
            },
        },
    }
```

### Using a Dataset Instance

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Initialize
ds = CamelsUs(SETTING["local_data_path"]["datasets-origin"])

# Get available features
print(ds.available_static_features)
print(ds.available_dynamic_features)

# Read data
basin_ids = ds.read_object_ids()
timeseries = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"]
)
attributes = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:5],
    var_lst=["area", "p_mean"]
)

# Access convenience methods
areas = ds.read_area(gage_id_lst=basin_ids[:5])
mean_precip = ds.read_mean_prcp(gage_id_lst=basin_ids[:5])
```

## API Reference

::: hydrodataset.hydro_dataset.HydroDataset
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - __init__
        - read_object_ids
        - read_ts_xrdataset
        - read_attr_xrdataset
        - cache_timeseries_xrdataset
        - cache_attributes_xrdataset
        - read_area
        - read_mean_prcp
        - available_static_features
        - available_dynamic_features
        - default_t_range
