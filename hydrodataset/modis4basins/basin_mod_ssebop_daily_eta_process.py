import datetime
import os

import numpy as np
from pygeoutils.pygeoutils import _get_transform, _geo2polygon
from rasterio.features import geometry_mask
import xarray as xr
import geopandas as gpd
import pandas as pd


def calculate_tif_data_basin_mean(eta_tif_files: list, camels_shp_file: str):
    """
    Calculate basin mean values for tif data

    Parameters
    ----------
    eta_tif_files
        a list with tif files path
    camels_shp_file
        shapefile of all basins in CAMELS

    Returns
    -------

        basin mean values
    """

    # https://github.com/pydata/xarray/issues/2313
    def preprocess(ds_):
        """add time for each file"""
        # https://stackoverflow.com/questions/37743940/how-to-convert-julian-date-to-standard-date
        time_now = datetime.datetime.strptime(list(ds_.encoding.values())[0].split(os.sep)[-1].split(".")[0][5:],
                                              '%y%j').date()
        times = pd.date_range(time_now, periods=1)
        time_da = xr.DataArray(times, [('time', times)])
        dst = ds_.expand_dims(time=time_da)
        return dst

    basins = gpd.read_file(camels_shp_file)
    xds = xr.open_mfdataset(eta_tif_files, engine="rasterio", preprocess=preprocess, concat_dim='time')
    geo_crs = basins.crs.to_string()
    arr = np.empty((basins.shape[0], len(eta_tif_files)))
    for i in range(basins.shape[0]):
        geometry = basins.geometry[i]
        ds_dims = ("y", "x")
        transform, width, height = _get_transform(xds, ds_dims)
        _geometry = _geo2polygon(geometry, geo_crs, xds.rio.crs)
        _mask = geometry_mask([_geometry], (height, width), transform, invert=True)
        coords = {ds_dims[0]: xds.coords[ds_dims[0]], ds_dims[1]: xds.coords[ds_dims[1]]}
        mask = xr.DataArray(_mask, coords, dims=ds_dims)

        ds_masked = xds.where(mask, drop=True)
        ds_masked.attrs["transform"] = transform
        ds_masked.attrs["bounds"] = _geometry.bounds

        for k in ds_masked.data_vars:
            # only one band now
            arr[i, :] = ds_masked[k].mean(dim=('x', 'y', 'band')).values
    return arr
