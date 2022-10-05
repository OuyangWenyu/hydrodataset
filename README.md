<!--
 * @Author: Wenyu Ouyang
 * @Date: 2021-12-05 22:13:21
 * @LastEditTime: 2022-10-05 18:03:53
 * @LastEditors: Wenyu Ouyang
 * @Description: README for hydrodataset
 * @FilePath: \hydrodataset\README.md
 * Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
-->
# hydrodataset


[![image](https://img.shields.io/pypi/v/hydrodataset.svg)](https://pypi.python.org/pypi/hydrodataset)
[![image](https://img.shields.io/conda/vn/conda-forge/hydrodataset.svg)](https://anaconda.org/conda-forge/hydrodataset)


**A Python package for downloading and reading hydrological datasets**

-   Free software: MIT license
-   Documentation: https://OuyangWenyu.github.io/hydrodataset

## Installation

It is quite easy to install hydrodataset. We provide a pip package to install:

```Bash
pip install hydrodataset
```

I highly recommend you to install this package in a virtual environment, so that it won't have negative impact on other packages in your base environment.

for example:

```Bash
# xxx is your env's name, such as hydrodataset
conda create -n xxx python=3.10
# activate the env
conda activate xxx
# install hydrodataset
conda install pip
pip install hydrodataset
```

## Usage

### 1. Download datasets

There are many CAMELS datasets, including CAMELS-AUS (Australia), CAMELS-BR (Brazil), CAMELS-CL (Chile), CAMELS-GB (Great Britain), CAMELS-US (United States).

Now we only support auto-downloading for CAMELS-US (later for others), but I highly recommend you to download them manually, as the downloading is not stable sometimes because of unstable web connection to the servers of these datasets all over the world.

the download links:

- [CAMELS-AUS (Australia)](https://doi.pangaea.de/10.1594/PANGAEA.921850)
- [CAMELS-BR (Brazil)](https://zenodo.org/record/3964745#.YNsjKOgzbIU)
- [CAMELS-CL (Chile)](https://doi.pangaea.de/10.1594/PANGAEA.894885)
- [CAMELS-GB (Great Britain)](https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9)
- [CAMELS-US (United States)](https://gdex.ucar.edu/dataset/camels)

put these downloaded files in the directory organized as follows:

```dir
camels/
├─ camels_aus/
│  ├─ 01_id_name_metadata.zip
│  ├─ 02_location_boundary_area.zip
│  ├─ 03_streamflow.zip
│  ├─ 04_attributes.zip
│  ├─ 05_hydrometeorology.zip
├─ camels_br/
│  ├─ 01_CAMELS_BR_attributes.zip
│  ├─ 02_CAMELS_BR_streamflow_m3s.zip
│  ├─ 03_CAMELS_BR_streamflow_mm_selected_catchments.zip
│  ├─ 04_CAMELS_BR_streamflow_simulated.zip
│  ├─ 05_CAMELS_BR_precipitation_chirps.zip
│  ├─ 06_CAMELS_BR_precipitation_mswep.zip
│  ├─ 07_CAMELS_BR_precipitation_cpc.zip
│  ├─ 08_CAMELS_BR_evapotransp_gleam.zip
│  ├─ 09_CAMELS_BR_evapotransp_mgb.zip
│  ├─ 10_CAMELS_BR_potential_evapotransp_gleam.zip
│  ├─ 11_CAMELS_BR_temperature_min_cpc.zip
│  ├─ 12_CAMELS_BR_temperature_mean_cpc.zip
│  ├─ 13_CAMELS_BR_temperature_max_cpc.zip
│  ├─ 14_CAMELS_BR_catchment_boundaries.zip
│  ├─ 15_CAMELS_BR_gauges_location_shapefile.zip
├─ camels_cl/
│  ├─ 10_CAMELScl_tmean_cr2met.zip
│  ├─ 11_CAMELScl_pet_8d_modis.zip
│  ├─ 12_CAMELScl_pet_hargreaves.zip
│  ├─ 13_CAMELScl_swe.zip
│  ├─ 14_CAMELScl_catch_hierarchy.zip
│  ├─ 1_CAMELScl_attributes.zip
│  ├─ 2_CAMELScl_streamflow_m3s.zip
│  ├─ 3_CAMELScl_streamflow_mm.zip
│  ├─ 4_CAMELScl_precip_cr2met.zip
│  ├─ 5_CAMELScl_precip_chirps.zip
│  ├─ 6_CAMELScl_precip_mswep.zip
│  ├─ 7_CAMELScl_precip_tmpa.zip
│  ├─ 8_CAMELScl_tmin_cr2met.zip
│  ├─ 9_CAMELScl_tmax_cr2met.zip
│  ├─ CAMELScl_catchment_boundaries.zip
├─ camels_gb/
│  ├─ 8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip
├─ camels_us/
│  ├─ basin_set_full_res.zip
│  ├─ basin_timeseries_v1p2_metForcing_obsFlow.zip
│  ├─ camels_attributes_v2.0.xlsx
│  ├─ camels_clim.txt
│  ├─ camels_geol.txt
│  ├─ camels_hydro.txt
│  ├─ camels_name.txt
│  ├─ camels_soil.txt
│  ├─ camels_topo.txt
│  ├─ camels_vege.txt
```

### 2. Run the code

First, run the following Python code:

```Python
import hydrodataset
```

then in your home directory, you will find the directory for hydrodataset: 

- Windows: C:\\Users\\xxx\\.hydrodataset (xxx is your username))
- Ubuntu: /home/xxx/.hydrodataset 

In .hydrodataset directory, there is a settings.txt file and only one line is in it. This line means the root directory of hydrodataset, and its 1st sub-directory is camels (shown in the previous directory tree).

Now modify this line to your root directory, for example, I put camels/ to D:\\data\\  , so this line should be D:\\data\\

Then, you can use functions in hydrodataset, examples could be seen here: https://github.com/OuyangWenyu/hydrodataset/blob/main/examples/scripts.py

These functions are about reading attributes/forcing/streamflow data.

**When you first run the code, you should set the parameter "download" to True**:

```Python
import os
from hydrodataset.camels import Camels
camels = Camels(data_path=os.path.join("camels", "camels_us"), download=True, region="US")
```

It will unzip all downloaded files, and take some minutes, please be patient.

**Except for the first run, you should set "download" to False**:

```Python
import os
from hydrodataset.camels import Camels
# default is False
camels = Camels(data_path=os.path.join("camels", "camels_us"), region="US")
```

You can change your data_path to anywhere you put in the the root directory of hydrodataset.

## Features

HydroDataset is designed to help (1) download, (2) read, (3)format and (4) visualize some datasets through a
core language (Python) for watershed hydrological modeling.

**Note**: But now this repository is still developing and only supports quite simple functions such as downloading and reading data for watersheds.

Now the dataset zoo list includes:

| **Number** | **Dataset** | **Description**                                         | **Format**        |
| ---------- | ----------- | ------------------------------------------------------- | ----------------- |
| 1          | **CAMELS**  | CAMELS series datasets including CAMELS-AUS/BR/CL/GB/US | Dataset Directory |

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.

It was inspired by [HydroData](https://github.com/mikejohnson51/HydroData) and used some tools made by [cheginit](https://github.com/cheginit).
