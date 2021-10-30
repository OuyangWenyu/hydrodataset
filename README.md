# HydroBench

Data downloader and processor for Hydrologic Modeling

## Data source zoo list

- CAMELS
- Daymet
- ECMWF
- MODIS
- MOPEX
- NLDAS

More details are shown in the following sections.

## CAMELS

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

If you can read Chinese, [this blog](https://github.com/OuyangWenyu/aqualord/blob/master/CAMELS/CAMELS.md) may be a
quick start for CAMELS (CAMELS-US)
and [this](https://github.com/OuyangWenyu/aqualord/blob/master/CAMELS/CAMELS-other.md)
for other CAMELS datasets.

### Download CAMELS datasets

We recommend downloading the datasets manually, the downloading address are as follows:

- [Download CAMELS-AUS](https://doi.pangaea.de/10.1594/PANGAEA.921850)
- [Download CAMELS-BR](https://doi.org/10.5281/zenodo.3709337)
- [Download CAMELS-CL](https://doi.pangaea.de/10.1594/PANGAEA.894885)
- [Download CAMELS-GB](https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9)
- [Download CAMELS-US](https://ral.ucar.edu/solutions/products/camels)
- [Download CAMELS-YR](http://doi.org/10.5281/zenodo.4704017)

You can also use the following code to download CAMELS-US:

```Python
import os
import definitions
from hydrobench.data.data_camels import Camels

camels_path = os.path.join(definitions.DATASET_DIR, "camels")
camels = Camels(camels_path, download=True)
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

## ECMWF

### Download ERA5-Land data

Although we provide tools to use cds toolbox from ECMWF to retrieve ERA5-land data, it seems it didn't work well (even
when data is MB level). Hence, we recommend a manual way to download the ERA5-land data archive
from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form

### Process the downloaded ERA5-Land data

TODO: Regrid the raw data to the required resolutions (src/regrid.py from https://github.com/pangeo-data/WeatherBench)

## MODIS

### Download basin mean ET data from GEE

We provided [Google Earth Engine](https://earthengine.google.com/) scripts to download the PML V2 and MODIS MOD16A2_105
product for given basins:

TODO: provide a link -- [Download basin mean values of ET data]()

### Process ET data to CAMELS format

Use hydrobench\app\modis4basins\trans_modis_et_to_camels_format.py to process the downloaded ET data from GEE to the
format of forcing data in CAMELS

## MOPEX

TODO: Now we support [CANOPEX](http://canopex.etsmtl.net/), Canada's MOPEX dataset.

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

## Acknowledgement

- [HyRiver](https://github.com/cheginit/HyRiver)
- [WeatherBench](https://github.com/pangeo-data/WeatherBench)
