# HydroBench

Data downloader and processor for Hydrologic Modeling

## Data source zoo list

- Daymet
- ECMWF
- MODIS
- NLDAS

More details are shown in the following sections.

## Daymet

We download and process Daymet data for the 671 basins in [CAMELS](https://ral.ucar.edu/solutions/products/camels).

If you can read Chinese, [this blog](https://github.com/OuyangWenyu/aqualord/blob/master/CAMELS/CAMELS.md) may be a
quick start for CAMELS.

### Downloading the CAMELS dataset

You can download CAMELS manually from https://ral.ucar.edu/solutions/products/camels ; or you can use the following
code:

```Python
import os
import definitions
from hydrobench.data.data_camels import Camels

camels_path = os.path.join(definitions.DATASET_DIR, "camels")
camels = Camels(camels_path, download=True)
```

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
