from typing import Union, Tuple, List, Optional, MutableMapping, Any
import pygeoutils as geoutils
import async_retriever as ar
import py3dep
from pydaymet import InvalidInputRange
from pydaymet.core import Daymet, _check_requirements
from pydaymet.pydaymet import _gridded_urls, _xarray_geomask
from shapely.geometry import MultiPolygon, Polygon
import rasterio.features as rio_features
import io
import xarray as xr
import numpy as np
import pandas as pd
from hydrobench.pet.pet4daymet import priestley_taylor, pm_fao56

DEF_CRS = "epsg:4326"


def download_daymet_by_geom_bound(
        geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
        dates: Union[Tuple[str, str], Union[int, List[int]]],
        crs: str = DEF_CRS,
        variables: Optional[List[str]] = None,
        region: str = "na",
        time_scale: str = "daily",
        boundary: bool = True
) -> xr.Dataset:
    """
    Get gridded data from the Daymet database at 1-km resolution in the boundary of the "geometry"

    if the error occurred: sqlite3.DatabaseError: database disk image is malformed,
    please delete the cache in the current directory of the performing script

    Parameters
    ----------
    geometry
        The geometry of the region of interest.
    dates
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    crs
        The CRS of the input geometry, defaults to epsg:4326.
    variables
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    region
        Region in the US, defaults to na. Acceptable values are:
        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico
    time_scale
        Data time scale which can be daily, monthly (monthly average),
        or annual (annual average). Defaults to daily.
    boundary
        if boundary is true, we will use the box of bounds as the geometry mask;
        otherwise, return downloaded data acccording to urls directly

    Returns
    -------
    xr.Dataset
        Daily climate data within a geometry's boundary

    Raises
    -------
    ValueError
        when downloading failed, raise a ValueError

    """

    daymet = Daymet(variables, time_scale=time_scale, region=region)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)
    # transform the crs
    _geometry = geoutils.pygeoutils._geo2polygon(geometry, crs, DEF_CRS)

    if not _geometry.intersects(daymet.region_bbox[region]):
        raise InvalidInputRange(daymet.invalid_bbox_msg)

    urls, kwds = zip(
        *_gridded_urls(
            daymet.time_codes[time_scale],
            _geometry.bounds,
            daymet.region,
            daymet.variables,
            dates_itr,
        )
    )

    try:
        clm = xr.open_mfdataset(
            (io.BytesIO(r) for r in ar.retrieve(urls, "binary", request_kwds=kwds, max_workers=8)),
            engine="scipy",
            coords="minimal",
        )
    except ValueError:
        msg = (
                "The server did NOT process your request successfully. "
                + "Check your inputs and try again."
        )
        raise ValueError(msg)

    for k, v in daymet.units.items():
        if k in clm.variables:
            clm[k].attrs["units"] = v

    clm = clm.drop_vars(["lambert_conformal_conic"])
    # daymet's crs comes from: https://daymet.ornl.gov/overview
    daymet_crs = " ".join(
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
    clm.attrs["crs"] = daymet_crs
    clm.attrs["nodatavals"] = (0.0,)
    transform, _, _ = geoutils.pygeoutils._get_transform(clm, ("y", "x"))
    clm.attrs["transform"] = transform
    clm.attrs["res"] = (transform.a, transform.e)

    if isinstance(clm, xr.Dataset):
        for v in clm:
            clm[v].attrs["crs"] = crs
            clm[v].attrs["nodatavals"] = (0.0,)
    if boundary:
        return _xarray_geomask(clm, geometry.bounds, crs)
    else:
        return clm


def calculate_basin_grids_pet(clm_ds: xr.Dataset, pet_method: Union[str, list] = "priestley_taylor") -> xr.Dataset:
    """
    Compute Potential EvapoTranspiration using Daymet dataset.

    Parameters
    ----------
    clm_ds
        The dataset should include the following variables:
        `tmin``, ``tmax``, ``lat``, ``lon``, ``vp``, ``srad``, ``dayl``
    pet_method
        now support priestley_taylor and fao56

    Returns
    -------
    xr.Dataset
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
    """
    Get gridded data from the Daymet database at 1-km resolution.

    Parameters
    ----------
    clm_ds
            gridded daymet Dataset of a basin.
    geometry
        The geometry of a basin.
    geo_crs
        The CRS of the input geometry, defaults to epsg:4326.
    Returns
    -------
    xr.Dataset
        Daily mean climate data of the basin

    """

    clm = _xarray_geomask(clm_ds, geometry, geo_crs)
    ds = xr.Dataset({}, coords={'time': clm.time})
    for k in clm.data_vars:
        ds[k] = clm[k].mean(dim=('x', 'y'))
    return ds


def generate_boundary_dataset(clm_ds: xr.Dataset,
                              geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
                              geo_crs: str = DEF_CRS) -> xr.Dataset:
    """
    Generate an xarray dataset in the boundary of geometry, but the boundary belongs to clm_ds's array, not the geometry

    Parameters
    ----------
    clm_ds
        Downloaded gridded daymet Dataset of a basin.
    geometry
        The geometry of a basin.
    geo_crs
        The CRS of the input geometry, defaults to epsg:4326.

    Returns
    -------
    xr.Dataset
        an xarray dataset in the boundary of the geometry

    """

    ds_dims = ("y", "x")
    transform, width, height = geoutils.pygeoutils._get_transform(clm_ds, ds_dims)
    _geometry = geoutils.pygeoutils._geo2polygon(geometry, geo_crs, clm_ds.crs)

    _mask = rio_features.geometry_mask([_geometry], (height, width), transform, invert=True)
    # x - column, y - row
    y_idx, x_idx = np.where(_mask)
    y_idx_min = y_idx.min()
    y_idx_max = y_idx.max()
    x_idx_min = x_idx.min()
    x_idx_max = x_idx.max()
    _mask_bound = np.full(_mask.shape, False)
    _mask_bound[y_idx_min:y_idx_max + 1, x_idx_min:x_idx_max + 1] = True

    coords = {ds_dims[0]: clm_ds.coords[ds_dims[0]], ds_dims[1]: clm_ds.coords[ds_dims[1]]}
    mask_bound = xr.DataArray(_mask_bound, coords, dims=ds_dims)

    ds_bound_masked = clm_ds.where(mask_bound, drop=True)
    ds_bound_masked.attrs["transform"] = transform
    return ds_bound_masked


def resample_nc(clm_ds: xr.Dataset,
                resample_size: Union[int, float]) -> xr.Dataset:
    """
    Resample the dataset to the resample_size

    Because Daymet's resolution is 1km which means each grid is 1km * 1km in a x-y coordinate system,
    we think it's enough to use general regrid methods such as interpolate functions in scipy.

    Parameters
    ----------
    clm_ds
        the original xarray dataset
    resample_size
        the ratio of resampled dataset's resolution to the original dataset's

    Returns
    -------
    xr.Dataset
        the resampled dataset

    """

    if resample_size > 1:
        # coarsen the original values
        ds = clm_ds.coarsen(x=resample_size, boundary="pad").mean().coarsen(y=resample_size, boundary="pad").mean()
    else:
        ydim, xdim = ("y", "x")
        height, width = clm_ds.sizes[ydim], clm_ds.sizes[xdim]
        left, right = clm_ds[xdim].min().item(), clm_ds[xdim].max().item()
        bottom, top = clm_ds[ydim].min().item(), clm_ds[ydim].max().item()

        x_res = abs(left - right) / (width - 1)
        y_res = abs(top - bottom) / (height - 1)
        # interpolate the original values to the new resolution
        x_res_new = x_res * resample_size
        y_res_new = y_res * resample_size
        # the array is in a left-close-right-open range, so  right + x_res
        new_x = np.arange(left, right + x_res, x_res_new)
        # the sequence of y is large -> small, for example, [941, 940, 939, ...]
        new_y = np.arange(bottom, top + y_res, y_res_new)[::-1]
        # we extrapolate some out-range values
        ds = clm_ds.interp(x=new_x, y=new_y, kwargs={"fill_value": "extrapolate"})
    return ds
