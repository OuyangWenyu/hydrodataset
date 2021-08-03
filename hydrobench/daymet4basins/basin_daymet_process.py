from typing import Union, Tuple, List, Optional
import pygeoutils as geoutils
import pygeoogc as ogc
import py3dep
from pydaymet.pydaymet import Daymet, gridded_urls, _check_requirements
from shapely.geometry import MultiPolygon, Polygon, box
import rasterio.transform as rio_transform
import xarray as xr
import numpy as np
import pandas as pd

from hydrobench.pet.pet4daymet import pm_fao56, priestley_taylor

DEF_CRS = "epsg:4326"


def download_daymet_by_geom_bound(
        geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
        dates: Union[Tuple[str, str], Union[int, List[int]]],
        geo_crs: str = DEF_CRS,
        variables: Optional[List[str]] = None,
        region: str = "na",
        time_scale: str = "daily",
        boundary: bool = True
) -> xr.Dataset:
    """
    Get gridded data from the Daymet database at 1-km resolution in the boundary of the "geometry"

    :param geometry:  The geometry of the region of interest.
    :param dates: Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    :param geo_crs: The CRS of the input geometry, defaults to epsg:4326.
    :param variables:
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    :param region:
        Region in the US, defaults to na. Acceptable values are:
        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico
    :param time_scale:
        Data time scale which can be daily, monthly (monthly average),
        or annual (annual average). Defaults to daily.
    :param boundary:
        if boundary is true, we will use the box of bounds as the geometry mask;
        otherwise, return downloaded data acccording to urls directly
    :return: Daily climate data within a geometry
    """

    daymet = Daymet(variables, time_scale=time_scale)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)
    # notice: there is a bug in the geo2polygon function when using geometry.bounds as the parameter, so warning
    if type(geometry) is tuple:
        raise NotImplementedError("Please don't use tuple as the type of geometry when calling geoutils.geo2polygon,"
                                  "because there is a bug here.")
    _geometry = geoutils.geo2polygon(geometry, geo_crs, DEF_CRS)
    urls = gridded_urls(
        daymet.code[time_scale], _geometry.bounds, region, daymet.variables, dates_itr
    )

    clm = xr.open_mfdataset(ogc.async_requests(urls, "binary", max_workers=8))

    for k, v in daymet.units.items():
        if k in clm.variables:
            clm[k].attrs["units"] = v

    clm = clm.drop_vars(["lambert_conformal_conic"])

    crs = " ".join(
        [
            "+proj=lcc",
            "+lat_1=25",
            "+lat_2=60",
            "+lat_0=42.5",
            "+lon_0=-100",
            "+x_0=0",
            "+y_0=0",
            "+ellps=WGS84",
            "+units=km",
            "+no_defs",
        ]
    )
    clm.attrs["crs"] = crs
    clm.attrs["nodatavals"] = (-9999,)
    # we need a transfrom between xarray-dataset's x-y loc and its geo-coord
    xdim, ydim = "x", "y"
    height, width = clm.sizes[ydim], clm.sizes[xdim]

    left, right = clm[xdim].min().item(), clm[xdim].max().item()
    bottom, top = clm[ydim].min().item(), clm[ydim].max().item()

    x_res = abs(left - right) / (width - 1)
    y_res = abs(top - bottom) / (height - 1)

    left -= x_res * 0.5
    right += x_res * 0.5
    top += y_res * 0.5
    bottom -= y_res * 0.5

    clm.attrs["transform"] = rio_transform.from_bounds(left, bottom, right, top, width, height)
    clm.attrs["res"] = (x_res, y_res)
    clm.attrs["bounds"] = (left, bottom, right, top)

    if isinstance(clm, xr.Dataset):
        for v in clm:
            clm[v].attrs["crs"] = crs
            clm[v].attrs["nodatavals"] = (-9999,)
    if boundary:
        # notice: there is a bug in the geo2polygon function when using geometry.bounds as the parameter,
        # so DO NOT use bounds and we use box(*geometry.bounds) here
        return geoutils.xarray_geomask(clm, box(*geometry.bounds), geo_crs)
    else:
        return clm


def calculate_basin_grids_pet(clm_ds: xr.Dataset, pet_method: Union[str, list] = "priestley_taylor") -> xr.Dataset:
    """
    Compute Potential EvapoTranspiration using Daymet dataset.
    Parameters
    ----------
    clm_ds : xarray.DataArray
        The dataset should include the following variables:
            `tmin``, ``tmax``, ``lat``, ``lon``, ``vp``, ``srad``, ``dayl``
    pet_method: now support priestley_taylor and fao56
    Returns
    -------
    xarray.DataArray
        The input dataset with an additional variable called ``pet``.
    """
    if type(pet_method) is str:
        pet_method = [pet_method]
    assert np.sort(pet_method) in np.sort(["priestley_taylor", "pm_fao56"])

    keys = list(clm_ds.keys())
    reqs = ["tmin", "tmax", "lat", "vp", "srad", "dayl"]
    # units: °C, °C, °, Pa, W/m^2, seconds
    _check_requirements(reqs, keys)
    dtype = clm_ds.tmin.dtype
    dates = clm_ds["time"]
    # m -> km
    res = clm_ds.res[0] * 1.0e3
    elev = py3dep.elevation_bygrid(clm_ds.x.values, clm_ds.y.values, clm_ds.crs, res)
    attrs = clm_ds.attrs
    clm_ds = xr.merge([clm_ds, elev])
    clm_ds.attrs = attrs
    clm_ds["elevation"] = clm_ds.elevation.where(
        ~np.isnan(clm_ds.isel(time=0)[keys[0]]), drop=True
    )
    # Pa -> kPa
    clm_ds["vp"] *= 1e-3
    # data -> day of year
    clm_ds["time"] = pd.to_datetime(clm_ds.time.values).dayofyear.astype(dtype)

    t_min = clm_ds["tmin"]
    t_max = clm_ds["tmax"]
    # average over the daylight period of the day, W/m^2 -> average over the day, MJ m-2 day-1
    r_surf = clm_ds["srad"] * clm_ds["dayl"] * 1e-6
    lat = clm_ds.isel(time=0).lat
    # ° -> rad
    phi = lat * np.pi / 180.0
    elevation = clm_ds["elevation"]
    doy = clm_ds["time"]
    e_a = clm_ds["vp"]

    for pet_name in pet_method:
        if pet_name == "pm_fao56":
            clm_ds["pet_fao56"] = pm_fao56(t_min, t_max, r_surf, phi, elevation, doy, e_a=e_a)
            clm_ds["pet_fao56"].attrs["units"] = "mm/day"
        elif pet_name == "priestley_taylor":
            clm_ds["pet_pt"] = priestley_taylor(t_min, t_max, r_surf, phi, elevation, doy, e_a=e_a)
            clm_ds["pet_pt"].attrs["units"] = "mm/day"

    # after calculation, recover the value of time and vp
    clm_ds["time"] = dates
    clm_ds["vp"] *= 1.0e3
    return clm_ds


def calculate_basin_mean(clm_ds: xr.Dataset,
                         geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
                         geo_crs: str = DEF_CRS) -> xr.Dataset:
    """Get gridded data from the Daymet database at 1-km resolution.

        Parameters
        ----------
        clm_ds :
            gridded daymet Dataset of a basin.
        geometry : Polygon, MultiPolygon, or bbox
            The geometry of a basin.
        geo_crs : str, optional
            The CRS of the input geometry, defaults to epsg:4326.

        Returns
        -------
        xarray.Dataset
            Daily mean climate data of the basin
        """
    clm = geoutils.xarray_geomask(clm_ds, geometry, geo_crs)
    ds = xr.Dataset({}, coords={'time': clm.time})
    for k in clm.data_vars:
        ds[k] = clm[k].mean(dim=('x', 'y'))
    return ds
