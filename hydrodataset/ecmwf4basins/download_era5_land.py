"""use cds toolbox from ECMWF to retrieve ERA5-land data
era5-land data:　https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form

we refer to these repos:
https://cds.climate.copernicus.eu/toolbox/doc/how-to/1_how_to_retrieve_data/1_how_to_retrieve_data.html#ht1
https://github.com/loicduffar/ERA5-tools
https://github.com/coecms/era5

At first, you need to register an account and install the cds api
Please see this tutorial: https://cds.climate.copernicus.eu/api-how-to

Then, login to CDS and Copy a 2 line code, which shows a url and your own uid:API key details as followed:
Go to this page(https://cds.climate.copernicus.eu/api-how-to)
and copy the 2 line code displayed in the black box in the "Install the CDS API key" section.

Paste the 2 line code into a  %USERPROFILE%\\.cdsapirc file, where in your windows environment,
%USERPROFILE% is usually located at C:\\Users\\Username folder (in Windows).

Next, client has to agree to the required terms and conditions.
To access this resource, you need to accept the terms of 'Licence to use Copernicus Products' at
https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products

Then you can use the following code. The original code comes from @author: Loïc Duffar (https://github.com/loicduffar)

"""
from typing import Union, List, Tuple
import numpy as np
import cdsapi
import datetime
import pygeoutils as geoutils
from shapely.geometry import MultiPolygon, Polygon

from hydrodataset.utils import hydro_utils

DEF_CRS = "epsg:4326"


def download_era5(downloaded_file: str, date_range: Union[tuple, list],
                  lat_lon_range: Union[Polygon, MultiPolygon, Tuple[float, float, float, float], List[float]],
                  variables_list: Union[list, str], file_format='grib', crs: str = DEF_CRS):
    """
    Download ERA5 data.
    Notice: it seems that it will cost much time when the time range is over 1 year,
    so I suggest download the data directly from:
     https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form

    Parameters
    ----------
    downloaded_file
        a file's name such as D:\\Download\\Hydro-Climato\\CDS (Climate Data Store)\\ERA5-Land\\ERA5-Land-hourly.nc
    date_range
        TIME PERIOD to extract. For example, ["2000-01-01", "2000-01-02"]
        The first year now cannot be earlier than 1950 (for ERA-land) and the latest year can be the present year
    lat_lon_range
        AREA to extract. The elements are north, west, south and east, respectively.
        For example,
        lat_max, lon_min, lat_min, lon_max  = [41, 120, 40, 121]
    variables_list
        the variables you wanna downloading.
        In era5-land, VARIABLE(S) to extract: single name or list of names among those below
        - total_precipitation, surface_runoff,  runoff, snow_depth_water_equivalent (m)
        - 2m_temperature (K)
        - potential_evaporation, total_evaporation, evaporation_from_open_water_surfaces_excluding_ocean, evaporation_from_bare_soil (m negative)
        variables_list = 'total_precipitation'
        OR
        variables_list = ['total_precipitation',
                      'surface_runoff',
                      'snow_depth_water_equivalent',
                      '2m_temperature',
                      'potential_evaporation', 'total_evaporation', 'evaporation_from_open_water_surfaces_excluding_ocean',
                     ]
    file_format
        'grib' or 'netcdf'
    crs
        the coordination system
    Returns
    -------
    None
    """

    if type(lat_lon_range) is Polygon or type(lat_lon_range) is MultiPolygon:
        _geometry = geoutils.pygeoutils._geo2polygon(lat_lon_range, crs, DEF_CRS)
        bound = _geometry.bounds  # (minx, miny, maxx, maxy)
        lat_lon_range = [bound[3], bound[0], bound[1],
                         bound[2]]  # lat_max(maxy), lon_min(minx), lat_min(miny), lon_max
    lat_max, lon_min, lat_min, lon_max = lat_lon_range

    t_ranges = hydro_utils.t_range_days(date_range)
    years = [str(t_ranges[0].astype(object).year + i) for i in
             range(t_ranges[-1].astype(object).year - t_ranges[0].astype(object).year + 1)]
    months = [str(month).zfill(2) for month in
              np.sort(np.unique([t_range.astype(object).month for t_range in t_ranges]))]
    if len(t_ranges) > 31:
        # in ERA5, we need specify the days, not the range, so if days' number > 31, we need download all days' data
        start_day = 1
        end_day = 31
        days = [str(start_day + i).zfill(2) for i in range(end_day - start_day + 1)]
    else:
        days = [str(day).zfill(2) for day in np.sort([t_range.astype(object).day for t_range in t_ranges])]

    print('Process started. Please wait the ending message ... ')
    start = datetime.datetime.now()  # Start Timer

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-land',
        {
            'year': years,
            'variable': variables_list,
            'month': months,
            'day': days,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [lat_max, lon_min, lat_min, lon_max],
            'format': file_format,
        },
        downloaded_file)

    print('Process completed in ', datetime.datetime.now() - start)
