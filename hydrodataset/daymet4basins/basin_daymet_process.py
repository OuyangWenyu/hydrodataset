import fnmatch
import os
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
from hydrodataset.pet.pet4daymet import priestley_taylor, pm_fao56
from hydrodataset.utils.hydro_utils import t_range_days

DEF_CRS = "epsg:4326"


def download_daymet_by_geom_bound(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: str = DEF_CRS,
    variables: Optional[List[str]] = None,
    region: str = "na",
    time_scale: str = "daily",
    boundary: bool = True,
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
            (
                io.BytesIO(r)
                for r in ar.retrieve(urls, "binary", request_kwds=kwds, max_workers=8)
            ),
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


def calculate_basin_grids_pet(
    clm_ds: xr.Dataset, pet_method: Union[str, list] = "priestley_taylor"
) -> xr.Dataset:
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
    # km -> m
    res = clm_ds.res[0] * 1.0e3
    elev = py3dep.elevation_bygrid(clm_ds.x.values, clm_ds.y.values, clm_ds.crs, res)
    attrs = clm_ds.attrs
    clm_ds = xr.merge([clm_ds, elev], combine_attrs="override")
    clm_ds.attrs = attrs
    clm_ds["elevation"] = clm_ds.elevation.where(
        ~np.isnan(clm_ds.isel(time=0)[keys[0]]), drop=True
    ).T
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
            clm_ds["pet_fao56"] = pm_fao56(
                t_min, t_max, r_surf, phi, elevation, doy, e_a=e_a
            )
            clm_ds["pet_fao56"].attrs["units"] = "mm/day"
        elif pet_name == "priestley_taylor":
            clm_ds["pet_pt"] = priestley_taylor(
                t_min, t_max, r_surf, phi, elevation, doy, e_a=e_a
            )
            clm_ds["pet_pt"].attrs["units"] = "mm/day"

    # after calculation, recover the value of time and vp
    clm_ds["time"] = dates
    clm_ds["vp"] *= 1.0e3
    return clm_ds


def calculate_basin_mean(
    clm_ds: xr.Dataset,
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    geo_crs: str = DEF_CRS,
) -> xr.Dataset:
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
    ds = xr.Dataset({}, coords={"time": clm.time})
    for k in clm.data_vars:
        ds[k] = clm[k].mean(dim=("x", "y"))
    return ds


def generate_boundary_dataset(
    clm_ds: xr.Dataset,
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    geo_crs: str = DEF_CRS,
) -> xr.Dataset:
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

    _mask = rio_features.geometry_mask(
        [_geometry], (height, width), transform, invert=True
    )
    # x - column, y - row
    y_idx, x_idx = np.where(_mask)
    y_idx_min = y_idx.min()
    y_idx_max = y_idx.max()
    x_idx_min = x_idx.min()
    x_idx_max = x_idx.max()
    _mask_bound = np.full(_mask.shape, False)
    _mask_bound[y_idx_min : y_idx_max + 1, x_idx_min : x_idx_max + 1] = True

    coords = {
        ds_dims[0]: clm_ds.coords[ds_dims[0]],
        ds_dims[1]: clm_ds.coords[ds_dims[1]],
    }
    mask_bound = xr.DataArray(_mask_bound, coords, dims=ds_dims)

    ds_bound_masked = clm_ds.where(mask_bound, drop=True)
    ds_bound_masked.attrs["transform"] = transform
    return ds_bound_masked


def resample_nc(clm_ds: xr.Dataset, resample_size: Union[int, float]) -> xr.Dataset:
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
        ds = (
            clm_ds.coarsen(x=resample_size, boundary="pad")
            .mean()
            .coarsen(y=resample_size, boundary="pad")
            .mean()
        )
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


def trans_daymet_to_camels_format(
    daymet_dir: str, output_dir: str, gage_dict: dict, region: str, year: int
):
    """
    Transform forcing data of daymet downloaded from GEE to the format in CAMELS.

    The GEE code used to generate the original data can be seen here:
    https://code.earthengine.google.com/e910596013b5b90cb9c800d17a54a2b3
    If you can read Chinese, and prefer Python code, you can see here:
    https://github.com/OuyangWenyu/hydroGIS/blob/master/GEE/4-geepy-gallery.ipynb

    Parameters
    ----------
    daymet_dir
        the original data's directory
    output_dir
        the transformed data's directory
    gage_dict
        a dict containing gage's ids and the correspond HUC02 ids
    region
        we named the file downloaded from GEE as daymet_<region>_mean_<year>.csv,
        because we use GEE code to generate data for each year for each shape file (region) containing some basins.
        For example, if we use the basins' shpfile in CAMELS, the region is "camels".
    year
        we use GEE code to generate data for each year, so each year for each region has one data file.
    Returns
    -------
    None
    """

    name_dataset = [
        "gage_id",
        "time_start",
        "dayl",
        "prcp",
        "srad",
        "swe",
        "tmax",
        "tmin",
        "vp",
    ]
    camels_index = [
        "Year",
        "Mnth",
        "Day",
        "Hr",
        "dayl(s)",
        "prcp(mm/day)",
        "srad(W/m2)",
        "swe(mm)",
        "tmax(C)",
        "tmin(C)",
        "vp(Pa)",
    ]

    if "STAID" in gage_dict.keys():
        gage_id_key = "STAID"
    elif "gauge_id" in gage_dict.keys():
        gage_id_key = "gauge_id"
    elif "gage_id" in gage_dict.keys():
        gage_id_key = "gage_id"
    else:
        raise NotImplementedError("No such gage id name")

    if "HUC02" in gage_dict.keys():
        huc02_key = "HUC02"
    elif "huc_02" in gage_dict.keys():
        huc02_key = "huc_02"
    else:
        raise NotImplementedError("No such huc02 id")

    for f_name in os.listdir(daymet_dir):
        if fnmatch.fnmatch(f_name, "daymet_" + region + "_mean_" + str(year) + ".csv"):
            data_file = os.path.join(daymet_dir, f_name)
            # because this func only works for one region and one year, it means it only works for one file once
            # Hence, when we find the file and transform it, just finish
            break
    data_temp = pd.read_csv(data_file, sep=",", dtype={name_dataset[0]: str})
    for i_basin in range(len(gage_dict[gage_id_key])):
        # name csv
        basin_data = data_temp[
            data_temp[name_dataset[0]].values.astype(int)
            == int(gage_dict[gage_id_key][i_basin])
        ]
        if basin_data.shape[0] == 0:
            raise ArithmeticError("chosen basins' number is zero")
        # get Year,Month,Day,Hour info
        csv_date = pd.to_datetime(basin_data[name_dataset[1]])
        # the hour is set to 12, as 12 is the average hour of a day
        year_month_day_hour = pd.DataFrame(
            [[dt.year, dt.month, dt.day, 12] for dt in csv_date],
            columns=camels_index[0:4],
        )
        data_df = pd.DataFrame(basin_data.iloc[:, 2:].values, columns=camels_index[4:])
        # concat
        new_data_df = pd.concat([year_month_day_hour, data_df], axis=1)
        # output the result
        huc_id = gage_dict[huc02_key][i_basin]
        output_huc_dir = os.path.join(output_dir, huc_id)
        if not os.path.isdir(output_huc_dir):
            os.makedirs(output_huc_dir)
        output_file = os.path.join(
            output_huc_dir, gage_dict[gage_id_key][i_basin] + "_lump_daymet_forcing.txt"
        )
        print(
            "output forcing data of", gage_dict[gage_id_key][i_basin], "year", str(year)
        )
        if os.path.isfile(output_file):
            data_old = pd.read_csv(output_file, sep=" ")
            years = np.unique(data_old[camels_index[0]].values)
            if year in years:
                continue
            else:
                os.remove(output_file)
                new_data_df = pd.concat([data_old, new_data_df]).sort_values(
                    by=camels_index[0:3]
                )
        new_data_df.to_csv(
            output_file, header=True, index=False, sep=" ", float_format="%.2f"
        )


def insert_daymet_value_in_leap_year(
    data_dir: str, t_range: list = ["1980-01-01", "2020-01-01"]
):
    """
    interpolation for the 12.31 data in leap year

    Parameters
    ----------
    data_dir
        the transformed but not inserted data's directory
    t_range
        the time range to insert, the default range is ["1980-01-01", "2020-01-01"]

    Returns
    -------
    None
    """

    subdir_str = os.listdir(data_dir)
    col_lst = [
        "dayl(s)",
        "prcp(mm/day)",
        "srad(W/m2)",
        "swe(mm)",
        "tmax(C)",
        "tmin(C)",
        "vp(Pa)",
    ]
    for i in range(len(subdir_str)):
        subdir = os.path.join(data_dir, subdir_str[i])
        path_list = os.listdir(subdir)
        path_list.sort()
        for filename in path_list:
            data_file = os.path.join(subdir, filename)
            is_leap_file_name = data_file[-8:]
            if "leap" in is_leap_file_name:
                continue
            print("reading", data_file)
            data_temp = pd.read_csv(data_file, sep=r"\s+")
            data_temp.rename(columns={"Mnth": "Month"}, inplace=True)
            df_date = data_temp[["Year", "Month", "Day"]]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")
            # daymet file not for leap year, there is no data in 12.31 in leap year
            assert all(x < y for x, y in zip(date, date[1:]))
            t_range_list = t_range_days(t_range)
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            assert date[0] <= t_range_list[0] and date[-1] >= t_range_list[-1]
            nt = t_range_list.size
            out = np.full([nt, 7], np.nan)
            out[ind2, :] = data_temp[col_lst].values[ind1]
            x = pd.DataFrame(out, columns=col_lst)
            x_intepolate = x.interpolate(
                method="linear", limit_direction="forward", axis=0
            )
            csv_date = pd.to_datetime(t_range_list)
            year_month_day_hour = pd.DataFrame(
                [[dt.year, dt.month, dt.day, dt.hour] for dt in csv_date],
                columns=["Year", "Mnth", "Day", "Hr"],
            )
            # concat
            new_data_df = pd.concat([year_month_day_hour, x_intepolate], axis=1)
            output_file = data_file[:-4] + "_leap.txt"
            new_data_df.to_csv(
                output_file, header=True, index=False, sep=" ", float_format="%.2f"
            )
            os.remove(data_file)
