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

## Usage

### 1. Configure Your Data Path

Before using `hydrodataset`, you need to create a configuration file named `hydro_setting.yml` in your user home directory. You need to edit this file to specify where the datasets should be stored.

-   **Windows**: `C:\Users\<YourUsername>\hydro_setting.yml`
-   **Linux/macOS**: `/home/<YourUsername>/hydro_setting.yml`
-   **MacOS**: `/Users/<YourUsername>/hydro_setting.yml`

You only need to fill in the `root`, `datasets-origin` and `cache` paths. `datasets-origin` will be the base directory where all datasets are stored and `cache` will be the directory where the netcdf files of the datasets are stored.

```yaml
local_data_path:
  root: 'E:\data'
  datasets-origin: 'E:\data\ClassA\1st_origin\hydrodatasets'
  cache: 'E:\data\.cache'
  ... # Other fields can be left empty
```

### 2. Accessing Data

The new architecture simplifies data access significantly. You no longer need to worry about download parameters or complex directory structures; `AquaFetch` handles that automatically.

The first time you run the code for a dataset, it will be downloaded and cached as a NetCDF file. This might take some time. Subsequent runs will be much faster.

Here is a typical example:

```python
from hydrodataset.camels import Camels
from hydrodataset import SETTING
import os

# The data_path is now just the name of the dataset directory
# It will be created inside the `datasets-origin` you defined in hydro_setting.yml
data_path = os.path.join(SETTING["local_data_path"]["datasets-origin"], "camels_us")

# Initialize the dataset class
camels_us = Camels(data_path=data_path)

# Get a list of all basin IDs
basin_ids = camels_us.read_object_ids()
print(f"Found {len(basin_ids)} basins.")

# Read time-series data (e.g., streamflow and precipitation) for the first 5 basins
# Data is returned as an xarray.Dataset
ts_data = camels_us.read_ts_xrdataset(
    gage_id_lst=basin_ids[:5],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "prcp"]
)
print("Time-series data:")
print(ts_data)

# Read static attribute data (e.g., elevation and soil conductivity)
attr_data = camels_us.read_attr_xrdataset(
    gage_id_lst=basin_ids[:5],
    var_lst=["elev_mean", "soil_conductivity"]
)
print("\nStatic attribute data:")
print(attr_data)

```

## Supported Datasets

`hydrodataset` supports a wide range of datasets provided by `AquaFetch`, including:
-   **CAMELS**: The full series (AUS, BR, CH, CL, COL, DE, DK, FI, FR, GB, IND, LUX, NZ, SE, US, US-Hourly, KR-Hourly).
-   **BULL**: The BULL dataset.
-   **Caravan**: The Caravan_DK dataset.
-   **HYSETS**: The HYSETS dataset.
-   **Estreams**: The Estreams dataset.
-   **Hype**: The Hype dataset.
-   **LamaH**: The LamaH-ICE dataset.
-   And many more will be added in the future.

Each dataset has its own class within `hydrodataset` (e.g., `CamelsAus`, `CamelsBr`, `Hysets`).

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template. The data fetching and reading is now powered by [AquaFetch](https://github.com/hyex-research/AquaFetch).