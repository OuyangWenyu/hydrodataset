# hydrodataset

[![image](https://img.shields.io/pypi/v/hydrodataset.svg)](https://pypi.python.org/pypi/hydrodataset)
[![image](https://img.shields.io/conda/vn/conda-forge/hydrodataset.svg)](https://anaconda.org/conda-forge/hydrodataset)

**A Python package for accessing hydrological datasets, with a focus on preparing data for deep learning models.**

-   Free software: MIT license
-   Documentation: https://OuyangWenyu.github.io/hydrodataset

## Core Philosophy

This library has been redesigned to serve as a powerful data-adapting layer on top of the [AquaFetch](https://github.com/hyex-research/AquaFetch) package.

While `AquaFetch` handles the complexities of downloading and reading numerous public hydrological datasets, `hydrodataset` takes the next step: it standardizes this data into a clean, consistent NetCDF (`.nc`) format. This format is specifically optimized for seamless integration with hydrological modeling libraries like [torchhydro](https://github.com/OuyangWenyu/torchhydro).

The core workflow is:
1.  **Fetch**: Use a `hydrodataset` class for a specific dataset (e.g., `CamelsAus`).
2.  **Standardize**: It uses `AquaFetch` as the primary backend for fetching raw data, while maintaining a consistent, unified interface across all datasets.
3.  **Cache**: On the first run, `hydrodataset` processes the data into an `xarray.Dataset` and saves it as `.nc` files for timeseries and attributes separately in a specified local directory set in `hydro_setting.yml` in the user's home directory.
4.  **Access**: All subsequent data requests are read directly from the fast `.nc` cache, giving you analysis-ready data instantly.

## Installation

We strongly recommend using a virtual environment to manage dependencies.

### For Users

To install the package from PyPI, you can use `pip` or any other package manager. Here is an example using Python's built-in `venv`:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install the package
pip install hydrodataset
```

### For Developers

This project uses [uv](https://github.com/astral-sh/uv) for package and environment management. To work on the project locally:

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/hydrodataset.git
cd hydrodataset

# Create a virtual environment and install dependencies using uv
uv sync --all-extras
```
This command will install the base dependencies plus all optional dependencies for development and documentation.

## Core API and Usage

The primary goal of `hydrodataset` is to provide a simple, unified API for accessing various hydrological datasets. The core interface is exposed through the dataset objects. A typical workflow is demonstrated in `examples/read_dataset.py` and summarized below.

First, initialize the dataset class you want to use. Then, you can explore the available data and read it.

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING
import os

# All datasets are expected to be in the directory defined in your hydro_setting.yml
# A example of hydro_setting.yml in Windows is like this:
# local_data_path:
#   root: 'D:\data\waterism' # Update with your root data directory
#   datasets-origin: 'D:\data\waterism\datasets-origin'
#   cache: 'D:\data\waterism\cache'
data_path = SETTING["local_data_path"]["datasets-origin"]

# Initialize the dataset class
ds = CamelsUs(data_path)

# 1. Check which features are available
print("Available static features:")
print(ds.available_static_features)

print("\nAvailable dynamic features:")
print(ds.available_dynamic_features)

# 2. Get a list of all basin IDs
basin_ids = ds.read_object_ids()

# 3. Read static (attribute) data for a subset of basins
# Note: We use standardized names like 'area' and 'p_mean'
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:2],
    var_lst=["area", "p_mean"]
)
print("\nStatic attribute data:")
print(attr_data)

# 4. Read dynamic (time-series) data for the same basins
# Note: We use standardized names like 'streamflow' and 'precipitation'
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:2],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"]
)
print("\nTime-series data:")
print(ts_data)
```

### Standardized Variable Names

A key feature of the new architecture is the use of standardized variable names. This allows you to use the same variable name to fetch the same type of data across different datasets, without needing to know the specific, internal naming scheme of each one.

For example, you can get streamflow from both CAMELS-US and CAMELS-AUS using the same variable name:

```python
# Get streamflow from CAMELS-US
us_ds.read_ts_xrdataset(gage_id_lst=["01013500"], var_lst=["streamflow"], t_range=["1990-01-01", "1995-12-31"])

# Get streamflow from CAMELS-AUS
aus_ds.read_ts_xrdataset(gage_id_lst=["A4260522"], var_lst=["streamflow"], t_range=["1990-01-01", "1995-12-31"])
```

Similarly, you can use `precipitation`, `temperature_max`, etc., across datasets. A comprehensive list of these standardized names and their coverage across all datasets is in progress and will be published soon.

## Project Status & Future Work

The new, unified API architecture is currently in active development.

*   **Current Implementation**: The framework has been fully implemented and tested for the **`camels_us`** and **`camels_aus`** datasets.
*   **In Progress**: We are in the process of migrating all other datasets supported by the library to this new architecture.
*   **Release Schedule**: We plan to release new versions frequently in the short term as more datasets are integrated. Please check back for updates.

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template. The data fetching and reading is now powered by [AquaFetch](https://github.com/hyex-research/AquaFetch).