# HydroDatasets

## What's HydroDatasets

HydroDataset is designed to help (1) find, (2) get, (3) visualize, and (4) format disparate earth systems data through a
core language (Python) for watershed hydrological modeling.

It is inspired by https://github.com/mikejohnson51/HydroData and the following related similar packages:

- https://github.com/cheginit/hydrodata
- https://github.com/jsta/nhdR
- https://github.com/lawinslow/hydrolinks
- https://github.com/mbtyers/riverdist
- https://github.com/ropensci/FedData
- https://github.com/usgs-r/nhdplusTools
  ... others -- please suggest additions?

Now this repository is still developing and only supports quite simple functions such as downloading and processing data
for watersheds. In the future, we will upgrade all functions to unify vocabulary built around querying data by a basin
of interest (AOI).

Now the dataset zoo list includes:

|**Number**|**Dataset**|**Description**|**Format**|
|----------|-----------|---------------|-----------|
|1|**CAMELS/MOPEX/LAMAH**|Datasets for large-sample hydrological modeling|Dataset Directory|
|2|**Daymet**|Daymet meteorological forcing data North America|Dataset Directory|
|3|**ECMWF ERA5-Land**|ERA5-LAND dataet|Dataset Directory|
|4|**GHS**|Geospatial attributes and Hydrometeorological forcing for Streamflow modeling|Dataset Directory|
|5|**MODIS ET**|Evapotranspiration data product of MODIS|Dataset Directory|
|6|**NEX-GDDP-CMIP5/6**|NASA Earth Exchange Global Daily Downscaled Climate Projections dataset|Dataset Directory|
|7|**NLDAS**|NLDAS datset|Dataset Directory|
|8|**ChinaHydroMap**|Basic maps for watersheds in China|Vector|

More details are shown in the following sections.

## CAMELS/MOPEX/LAMAH

The CAMELS series data include:

- CAMELS-AUS
  ([CAMELS-AUS: Hydrometeorological time series and landscape attributes for 222 catchments in Australia](https://essd.copernicus.org/preprints/essd-2020-228/))
- CAMELS-BR
  ([CAMELS-BR: Hydrometeorological time series and landscape attributes for 897 catchments in Brazil - link to files](https://doi.org/10.5194/essd-12-2075-2020))
- CAMELS-CL
  ([The CAMELS-CL dataset: catchment attributes and meteorology for large sample studies – Chile dataset](https://doi.org/10.5194/hess-22-5817-2018))
- CAMELS-GB
  ([CAMELS-GB: Hydrometeorological time series and landscape attributes for 671 catchments in Great Britain](https://doi.org/10.5194/essd-2020-49))
- CAMELS-US
  ([The CAMELS data set: catchment attributes and meteorology for large-sample studies](https://doi.org/10.5194/hess-21-5293-2017))
- CAMELS-YR
  ([Catchment attributes and meteorology for large sample study in contiguous China](https://doi.org/10.5194/essd-2021-71))

We also support [CANOPEX](https://doi.org/10.1002/hyp.10880) (Canada's MOPEX dataset)
and [LamaH-CE](https://doi.org/10.5194/essd-13-4529-2021) (similar with CAMELS and it is for Central Europe), because we
use these datasets just like CAMELS, we write similar code in data/data_camels.py.

If you can read Chinese, [this blog](https://github.com/OuyangWenyu/aqualord/blob/master/CAMELS/CAMELS.md) may be a
quick start for CAMELS (CAMELS-US)
and [this](https://github.com/OuyangWenyu/aqualord/blob/master/CAMELS/CAMELS-other.md)
for other CAMELS datasets.

### Download CAMELS/MOPEX/LAMAH datasets

We recommend downloading the datasets manually, the downloading address are as follows:

- [Download CAMELS-AUS](https://doi.pangaea.de/10.1594/PANGAEA.921850)
- [Download CAMELS-BR](https://doi.org/10.5281/zenodo.3709337)
- [Download CAMELS-CL](https://doi.pangaea.de/10.1594/PANGAEA.894885)
- [Download CAMELS-GB](https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9)
- [Download CAMELS-US](https://ral.ucar.edu/solutions/products/camels)
- [Download CAMELS-YR](http://doi.org/10.5281/zenodo.4704017)
- [Download CANOPEX](http://canopex.etsmtl.net/)
- [Download LamaH-CE](https://zenodo.org/record/5153305#.YYdEgGBByUk)

You can also use the following code to download CAMELS-US (notice: the unzipped file is 10+ GB):

```Python
import os
import definitions
from hydrodataset.data.data_camels import Camels

# DATASET_DIR is defined in the definitions.py file
camels_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
camels = Camels(camels_path, download=True)
```

For CAMELS_YR, it is enough to
download [9_Normal_Camels_YR.zip](https://zenodo.org/record/4704017/files/9_Normal_Camels_YR.zip?download=1)

To download CANOPEX, you have to deal with the GFW. In addtion, there is no attributes data in CANOPEX, we choose an
alternative: [attributes data](https://osf.io/7fn4c/) from [HYSETS](https://doi.org/10.1038/s41597-020-00583-2)

After downloading, puteach dataset in one directory, the following file-organization is recommended:

```Directory
camels
│
└── camels_aus
    └── 01_id_name_metadata.zip
    └── 02_location_boundary_area.zip
    └── ...
└── camels_br
    └── ...
└── camels_cl
    └── ...
└── camels_gb
    └── ... 
└── camels_us
    └── ... 
└── camels_yr
    └── ... 
canopex
    └── Boundaries.zip
    └── HYSETS_watershed_properties.txt
    └── ...   
lamah_ce
    └── 2_LamaH-CE_daily.tar.gz
    └── ...   
```

### Process datasets

All methods for processing CAMELS datasets are written in Camels class in hydrobench/data/data_camels.py.

## Daymet

We download and process Daymet data for the 671 basins in [CAMELS(-US)](https://ral.ucar.edu/solutions/products/camels).

### Download Daymet V4 dataset for basins in CAMELS

Use hydrobench/app/download/download_daymet_camels_basin.py to download daymet grid data for the boundaries of basins in
CAMELS.

### Process the raw Daymet V4 data

We provided some scripts to process the Daymet grid data for basins:

- Regrid the raw data to the required resolutions (hydrobench/app/daymet4basins/regrid_daymet_nc.py)
- calculate_basin_mean_forcing_include_pet.py and calculate_basin_mean_values.pyin hydrobench/app/daymet4basins can be
  used for getting basin mean values
- If you want to get P (precipitation), PE (potential evapotranspiration), Q (streamflow) and Basin areas, please use
  hydrobench/app/daymet4basins/pbm_p_pe_q_basin_area.py

## ECMWF ERA5-Land

### Download ERA5-Land data

Although we provide tools to use cds toolbox from ECMWF to retrieve ERA5-land data, it seems it didn't work well (even
when data is MB level). Hence, we recommend a manual way to download the ERA5-land data archive
from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form

### Process the downloaded ERA5-Land data

TODO: Regrid the raw data to the required resolutions (src/regrid.py from https://github.com/pangeo-data/WeatherBench)

## GHS

The dataset's full name is "Geospatial attributes and Hydrometeorological forcings of gages for Streamflow modeling".

"GHS" is an extension for [the CAMELS dataset](https://ral.ucar.edu/solutions/products/camels). It contains geospatial
attributes, hydrometeorological forcings and streamflow data of 9067 gages over the Contiguous United States (CONUS)
in [the GAGES-II dataset](https://water.usgs.gov/GIS/metadata/usgswrd/XML/gagesII_Sept2011.xml).

Now we have not provided an online way to download the data. You can refer to the following paper to learn about how to
get it.

Wenyu Ouyang, Kathryn Lawson, Dapeng Feng, Lei Ye, Chi Zhang, & Chaopeng Shen (2021). Continental-scale streamflow
modeling of basins with reservoirs: Towards a coherent deep-learning-based
strategy. https://doi.org/10.1016/j.jhydrol.2021.126455

## MODIS ET

### Download basin mean ET data from GEE

We provided [Google Earth Engine](https://earthengine.google.com/) scripts to download the PML V2 and MODIS MOD16A2_105
product for given basins:

TODO: provide a link -- [Download basin mean values of ET data]()

### Process ET data to CAMELS format

Use hydrobench\app\modis4basins\trans_modis_et_to_camels_format.py to process the downloaded ET data from GEE to the
format of forcing data in CAMELS

## NEX-GDDP-CMIP5/6

### Download

NEX-GDDP-CMIP5 data for basins could be downloaded from Google Earth Engine. The code
is [here](https://code.earthengine.google.com/5edfca6263bea36f5c093fc6b80a68aa)

For NEX-GDDP-CMIP6, data should be downloaded
from [this website](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6)

### Process

Use hydrodataset/app/climateproj4basins/trans_nexdcp30_to_camels_format.py to process NEX-GDDP-CMIP5 data for basins

We will provide tool for NEX-GDDP-CMIP6 data soon

## NLDAS

### Download basin mean NLDAS data from GEE

The GEE script is [here](https://code.earthengine.google.com/72cb2661f2206b4f986e24af3560c000)

### Download NLDAS grid data from NASA Earth data

Use hydrobench/app/download/download_nldas_hourly.py to download them.

Notice: you should finish some necessary steps (see the comments in hydrobench/nldas4basins/download_nldas.py) before
using the script

### Process NLDAS basin mean forcing

Use hydrobench/app/nldas4basins/trans_nldas_to_camels_format.py to transform the data to the format of forcing data in
CAMELS.

TODO: more processing scripts are needed for NLDAS grid data.

## ChinaHydroMap

We use data from https://github.com/GaryBikini/ChinaAdminDivisonSHP and refer to code
from https://github.com/ytkz11/Visualization-of-China-Sentinel5P-NO2 to get maps for some basins.

More maps could be found from:

- http://gaohr.win/site/blogs/2017/2017-04-18-GIS-basic-data-of-China.html
- https://zhuanlan.zhihu.com/p/25634886

Data could be downloaded from these sources. Unzip and put them in the "hydromap" directory

## How to run the code

Use environment.yml to create conda environment:

```Shell
mamba env create -f environment.yml
conda activate HydroDataset
```

Then, you can try python script in "app" directory
