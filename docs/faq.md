# FAQ

## General Questions

### What is hydrodataset?

hydrodataset is a Python package that provides a unified API for accessing 50+ hydrological datasets. It serves as a data-adapting layer on top of AquaFetch, standardizing diverse datasets into a consistent NetCDF format optimized for deep learning workflows.

### How is hydrodataset different from AquaFetch?

- **AquaFetch**: Handles downloading and reading raw data from public hydrological datasets
- **hydrodataset**: Takes AquaFetch data and standardizes it into a consistent format with unified variable names, NetCDF caching, and ML-ready outputs

Think of AquaFetch as the data fetcher and hydrodataset as the data standardizer.

### Which Python versions are supported?

hydrodataset requires Python 3.10 or higher.

## Installation & Setup

### Where should I create the `hydro_setting.yml` file?

The `hydro_setting.yml` file should be placed in your **home directory** (`~/hydro_setting.yml`):
- **Windows**: `C:\Users\YourUsername\hydro_setting.yml`
- **Linux/Mac**: `/home/username/hydro_setting.yml` or `~/hydro_setting.yml`

### What should be in `hydro_setting.yml`?

```yaml
local_data_path:
  root: 'D:\data\waterism'                    # Your root data directory
  datasets-origin: 'D:\data\waterism\datasets-origin'  # Raw data from AquaFetch
  cache: 'D:\data\waterism\cache'             # NetCDF cache files
```

Adjust paths according to your system and preferences.

### I'm getting an error about missing `hydro_setting.yml`. What should I do?

1. Create the file in your home directory (see above)
2. Ensure the paths in the file exist and are writable
3. Use absolute paths or proper forward slashes on Windows

## Data Access

### How do I know which datasets are available?

Check the [Supported Datasets](https://github.com/OuyangWenyu/hydrodataset#supported-datasets) section in the README or browse the [API documentation](https://OuyangWenyu.github.io/hydrodataset/api/hydrodataset/).

### What are standardized variable names?

Standardized variable names allow you to request the same type of data across different datasets using a common name:
- `streamflow` - works for CAMELS-US, CAMELS-AUS, etc.
- `precipitation` - consistent across all datasets
- `temperature_max` / `temperature_min` - temperature extremes

This eliminates the need to learn each dataset's specific naming conventions.

### How do I see what variables are available for a dataset?

```python
from hydrodataset.camels_us import CamelsUs
ds = CamelsUs(data_path)

print(ds.available_static_features)   # Static attributes
print(ds.available_dynamic_features)  # Timeseries variables
```

## Caching & Performance

### Where are the NetCDF cache files stored?

Cache files are stored in the `cache` directory specified in your `hydro_setting.yml`:
```
{cache_directory}/{dataset}_timeseries.nc
{cache_directory}/{dataset}_attributes.nc
```

### The first data access is slow. Is this normal?

Yes! The first access:
1. Fetches raw data via AquaFetch
2. Standardizes variable names and units
3. Saves to NetCDF cache files

All subsequent reads are instant as they load from the fast `.nc` cache.

### How do I regenerate the cache?

Simply delete the corresponding `.nc` files in your cache directory:
```bash
# Delete cache for CAMELS-US
rm ~/data/cache/camels_us_timeseries.nc
rm ~/data/cache/camels_us_attributes.nc
```

Next access will regenerate the cache.

## Usage & Examples

### How do I read data for specific basins?

```python
ds = CamelsUs(data_path)
basin_ids = ds.read_object_ids()

# Read for first 5 basins
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:5],
    var_lst=["area", "p_mean"]
)
```

### How do I specify a time range?

```python
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=["01013500"],
    t_range=["1990-01-01", "1995-12-31"],  # YYYY-MM-DD format
    var_lst=["streamflow", "precipitation"]
)
```

### Can I use this with deep learning frameworks?

Yes! The data is returned as `xarray.Dataset` objects which can be easily converted to numpy arrays or PyTorch tensors:

```python
import torch

ts_data = ds.read_ts_xrdataset(...)
streamflow_array = ts_data['streamflow'].values  # numpy array
streamflow_tensor = torch.from_numpy(streamflow_array)  # PyTorch tensor
```

For integration with deep learning workflows, check out [torchhydro](https://github.com/OuyangWenyu/torchhydro).

## Troubleshooting

### I'm getting import errors. What should I check?

1. Ensure hydrodataset is installed: `pip install hydrodataset`
2. Check your Python version: `python --version` (must be 3.10+)
3. Try reinstalling: `pip install --upgrade hydrodataset`

### Data is not being cached. What's wrong?

1. Check that the `cache` path in `hydro_setting.yml` exists
2. Verify write permissions for the cache directory
3. Check disk space availability

### I'm getting "FileNotFoundError" when reading data. Help!

1. Ensure raw data is downloaded to the `datasets-origin` directory
2. Some datasets require manual download - check AquaFetch documentation
3. Verify paths in `hydro_setting.yml` are correct

### Where can I get help?

- 📖 Read the [Documentation](https://OuyangWenyu.github.io/hydrodataset)
- 🐛 Check [GitHub Issues](https://github.com/OuyangWenyu/hydrodataset/issues)
- 💬 Open a new issue with your question
