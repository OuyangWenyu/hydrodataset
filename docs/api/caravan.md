# Caravan

## Overview

**Caravan** is a global, standardized dataset of catchment attributes and meteorological forcings for large-sample hydrology. It combines data from multiple regional datasets (CAMELS variants, HYSETS, LamaH) to create a unified, quality-controlled dataset suitable for hydrological modeling and machine learning applications.

## Dataset Information

- **Region**: Global (16,299 basins across 7 regions)
- **Project**: Caravan
- **Module**: `hydrodataset.caravan`
- **Class**: `Caravan`
- **Paper**: [Kratzert et al. (2023)](https://www.nature.com/articles/s41597-023-01975-w)
- **Code**: [GitHub Repository](https://github.com/kratzert/Caravan)
- **Data Source**: [Zenodo](https://zenodo.org/record/7944025)

## Supported Regions

The Caravan dataset includes data from the following regions:

| Region Code | Name | Source Dataset | Description |
|-------------|------|----------------|-------------|
| `US` | United States | CAMELS-US | Continental US catchments |
| `AUS` | Australia | CAMELS-AUS | Australian catchments |
| `BR` | Brazil | CAMELS-BR | Brazilian catchments |
| `CL` | Chile | CAMELS-CL | Chilean catchments |
| `GB` | Great Britain | CAMELS-GB | UK catchments |
| `CE` | Central Europe | LamaH-CE | Central European catchments |
| `NA` | North America | HYSETS | Canadian catchments (HYSETS) |
| `Global` | All regions | Combined | All 16,299 basins (default) |

## Key Characteristics

- **Standardization**: Unified variable naming and data format across all regions
- **Quality Control**: Rigorous quality checks and data cleaning
- **Multi-source**: Combines ERA5-Land reanalysis with regional observations
- **Comprehensive**: Includes streamflow, meteorological forcings, and catchment attributes
- **Version**: Currently supports Version 0.3 (Version 1.6 available but not yet integrated)

## Features

### Static Attributes

Caravan provides three types of catchment attributes:

1. **Caravan-specific attributes** (e.g., gauge metadata, climate indices)
2. **HydroATLAS attributes** (e.g., topography, land cover, soil properties)
3. **Other attributes** (e.g., region-specific characteristics)

Key static variables include:
- Basin area (`area`)
- Mean precipitation (`p_mean`)
- Gauge location (latitude, longitude)
- Climate indices (aridity, moisture index, seasonality)
- Topographic characteristics (elevation, slope)
- Land cover fractions (forest, crop, urban, etc.)
- Soil properties (clay, silt, sand content)
- And 200+ HydroATLAS attributes

### Dynamic Variables

Timeseries variables standardized across all regions:

| Standard Variable | Caravan Name | Unit | Source |
|-------------------|--------------|------|--------|
| Streamflow | `streamflow` | mm/day | Regional observations |
| Precipitation | `total_precipitation_sum` | mm/day | ERA5-Land |
| Temperature (max) | `temperature_2m_max` | °C | ERA5-Land |
| Temperature (min) | `temperature_2m_min` | °C | ERA5-Land |
| Solar radiation | `surface_net_solar_radiation_mean` | W/m² | ERA5-Land |
| Snow water equivalent | `snow_depth_water_equivalent_mean` | mm/day | ERA5-Land |
| Potential ET | `potential_evaporation_sum` | mm/day | ERA5-Land |

## Usage

### Basic Usage - Global Dataset

```python
from hydrodataset.caravan import Caravan
from hydrodataset import SETTING

# Initialize with all regions (Global mode)
data_path = SETTING["local_data_path"]["datasets-origin"]
caravan = Caravan(data_path)

# Get all basin IDs
basin_ids = caravan.read_object_ids()
print(f"Total basins: {len(basin_ids)}")  # 16,299 basins

# Check available features
print("Static features:", caravan.available_static_features)
print("Dynamic features:", caravan.available_dynamic_features)

# Read timeseries data
timeseries = caravan.read_ts_xrdataset(
    gage_id_lst=basin_ids[:10],
    t_range=["2000-01-01", "2010-12-31"],
    var_lst=["streamflow", "precipitation", "temperature_max", "temperature_min"]
)
print(timeseries)

# Read attribute data
attributes = caravan.read_attr_xrdataset(
    gage_id_lst=basin_ids[:10],
    var_lst=["area", "p_mean", "gauge_lat", "gauge_lon"]
)
print(attributes)
```

### Region-Specific Usage

```python
# Initialize for a specific region
caravan_us = Caravan(data_path, region="US")
caravan_aus = Caravan(data_path, region="AUS")
caravan_ce = Caravan(data_path, region="CE")

# Get basins for specific region
us_basins = caravan_us.read_object_ids()
print(f"US basins: {len(us_basins)}")

# Read region-specific data
us_data = caravan_us.read_ts_xrdataset(
    gage_id_lst=us_basins[:5],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["streamflow", "precipitation"]
)
```

### Reading Multiple Variables

```python
# Read comprehensive timeseries
ts_data = caravan.read_ts_xrdataset(
    gage_id_lst=basin_ids[:20],
    t_range=["2000-01-01", "2015-12-31"],
    var_lst=[
        "streamflow",
        "precipitation",
        "temperature_max",
        "temperature_min",
        "potential_evapotranspiration",
        "solar_radiation",
        "snow_water_equivalent"
    ]
)

# Read basin properties
basin_attrs = caravan.read_attr_xrdataset(
    gage_id_lst=basin_ids[:20],
    var_lst=[
        "area",
        "p_mean",
        "gauge_lat",
        "gauge_lon",
        "aridity",
        "frac_snow",
        "high_prec_freq",
        "seasonality"
    ]
)
```

### Convenience Methods

```python
# Read basin area
areas = caravan.read_area(gage_id_lst=basin_ids[:10])

# Read mean precipitation
mean_precip = caravan.read_mean_prcp(gage_id_lst=basin_ids[:10])
```

## Data Format and Caching

### NetCDF Cache Structure

Due to the large dataset size (>30GB), Caravan uses a batched caching approach:

**Attributes Cache**: Single file
- File: `caravan_attributes.nc`
- Contains: All static attributes for all basins
- Size: ~100 MB

**Timeseries Cache**: Multiple batch files per region
- Files: `caravan_{region}_timeseries_batch_{start_id}_{end_id}.nc`
- Contains: Timeseries data for batches of ~100 basins
- Total size: ~30 GB (all regions)

### First-Time Setup

```python
# If data not downloaded, Caravan will automatically download from Zenodo
# This may take several hours for the full dataset (~30GB)
caravan = Caravan(data_path)  # Downloads if needed

# To manually cache the data:
caravan.cache_xrdataset(checkregion="all")  # Cache all regions
# or
caravan.cache_xrdataset(checkregion="hysets")  # Cache specific region
```

## Working with Raw Attributes

```python
# Read raw attributes (not standardized variable names)
raw_attrs = caravan.read_attr_xrdataset(
    gage_id_lst=basin_ids[:5],
    var_lst=[
        "ele_mt_sav",  # HydroATLAS: mean elevation
        "slp_dg_sav",  # HydroATLAS: mean slope
        "for_pc_sse",  # HydroATLAS: forest cover
        "crp_pc_sse",  # HydroATLAS: crop cover
    ]
)
```

## Data Quality

### Quality Control Features

- **Streamflow**: Quality flags, outlier detection, gap documentation
- **Meteorological forcings**: ERA5-Land reanalysis (consistent global coverage)
- **Attributes**: Cross-validation with HydroATLAS and regional datasets
- **Standardization**: Unified units and coordinate systems

### Known Limitations

1. **Version**: Currently supports Version 0.3; Version 1.6 available but not yet integrated
2. **Damaged Files**: Some HYSETS files may be corrupted (automatically checked and repaired from CSV)
3. **Download Time**: Full dataset download can take 3-6 hours depending on connection

## Integration with Deep Learning

Caravan is designed for machine learning applications:

```python
import torch
from torch.utils.data import Dataset

class CaravanDataset(Dataset):
    def __init__(self, caravan, basin_ids, t_range):
        # Read timeseries
        self.data = caravan.read_ts_xrdataset(
            gage_id_lst=basin_ids,
            t_range=t_range,
            var_lst=["streamflow", "precipitation", "temperature_max", "temperature_min"]
        )
        # Read static attributes
        self.attrs = caravan.read_attr_xrdataset(
            gage_id_lst=basin_ids,
            var_lst=["area", "p_mean", "aridity"]
        )

    def __len__(self):
        return len(self.data.basin)

    def __getitem__(self, idx):
        basin_id = self.data.basin.values[idx]
        timeseries = self.data.sel(basin=basin_id).to_array().values
        attributes = self.attrs.sel(basin=basin_id).to_array().values
        return torch.tensor(timeseries), torch.tensor(attributes)
```

## Performance Tips

### Memory Management

```python
# For large queries, use parallel reading
ts_data = caravan.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=["2000-01-01", "2010-12-31"],
    var_lst=["streamflow", "precipitation"],
    parallel=True  # Enable parallel reading with dask
)

# Load data lazily (returns dask arrays)
ts_data_lazy = caravan.read_ts_xrdataset(
    gage_id_lst=basin_ids[:100],
    t_range=["1990-01-01", "2020-12-31"],
    var_lst=["streamflow"]
)
# Compute when needed
ts_data_computed = ts_data_lazy.compute()
```

### Region-Specific Processing

```python
# Process one region at a time to reduce memory usage
regions = ["US", "AUS", "BR", "CL", "GB", "CE", "NA"]
for region in regions:
    caravan_region = Caravan(data_path, region=region)
    basin_ids = caravan_region.read_object_ids()

    # Process this region's data
    data = caravan_region.read_ts_xrdataset(
        gage_id_lst=basin_ids,
        t_range=["2000-01-01", "2010-12-31"],
        var_lst=["streamflow", "precipitation"]
    )

    # Your processing code here
    # ...
```

## References

If you use Caravan in your research, please cite:

```bibtex
@article{kratzert2023caravan,
  title={Caravan--a global community dataset for large-sample hydrology},
  author={Kratzert, Frederik and Nearing, Grey and Addor, Nans and Erickson, Tyler and Gauch, Martin and Gilon, Oren and Gudmundsson, Lukas and Hassidim, Avinatan and Klotz, Daniel and Nevo, Sella and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={61},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Additional Resources

- **Caravan Website**: [https://github.com/kratzert/Caravan](https://github.com/kratzert/Caravan)
- **Paper**: [Nature Scientific Data](https://www.nature.com/articles/s41597-023-01975-w)
- **HydroATLAS**: [Technical Documentation](https://data.hydrosheds.org/file/technical-documentation/BasinATLAS_Catalog_v10.pdf)
- **ERA5-Land**: [ECMWF Documentation](https://confluence.ecmwf.int/display/CKB/ERA5-Land)

## API Reference

::: hydrodataset.caravan.Caravan
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members:
        - __init__
        - read_object_ids
        - read_ts_xrdataset
        - read_attr_xrdataset
        - read_area
        - read_mean_prcp
        - available_static_features
        - available_dynamic_features
        - default_t_range
        - cache_xrdataset
        - cache_attributes_xrdataset
        - cache_timeseries_xrdataset
