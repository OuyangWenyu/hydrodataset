import os
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio.features as rio_features
import pygeoutils as geoutils
import definitions
from hydrodataset.climateproj4basins.basin_nexdcp30_process import (
    trans_month_nex_dcp30to_camels_format,
)
from hydrodataset.data.data_camels import Camels
from hydrodataset.data.data_gages import Gages
from hydrodataset.daymet4basins.basin_daymet_process import (
    generate_boundary_dataset,
    resample_nc,
    trans_daymet_to_camels_format,
    insert_daymet_value_in_leap_year,
)
from hydrodataset.ecmwf4basins.basin_era5_process import (
    trans_era5_land_to_camels_format,
)
from hydrodataset.modis4basins.basin_mod16a2v105_process import (
    trans_8day_modis16a2v105_to_camels_format,
)
from hydrodataset.modis4basins.basin_mod_ssebop_daily_eta_process import (
    calculate_tif_data_basin_mean,
)
from hydrodataset.modis4basins.basin_pmlv2_process import (
    trans_8day_pmlv2_to_camels_format,
)
from hydrodataset.nldas4basins.basin_nldas_process import (
    trans_daily_nldas_to_camels_format,
)
from hydrodataset.smap4basins.basin_smap_process import trans_nasa_usda_smap_to_camels_format
from hydrodataset.utils.hydro_geo import (
    gage_intersect_time_zone,
    split_shp_to_shps_in_time_zones,
)
from hydrodataset.utils.hydro_utils import serialize_json, unserialize_json_ordered


@pytest.fixture()
def save_dir():
    dir_ = os.path.join(definitions.ROOT_DIR, "test", "test_data")
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    return dir_


@pytest.fixture()
def var():
    return ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]


@pytest.fixture()
def camels():
    camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
    if not os.path.isfile(
            os.path.join(
                camels_dir,
                "camels_attributes_v2.0",
                "camels_attributes_v2.0",
                "camels_name.txt",
            )
    ):
        return Camels(camels_dir, True)
    return Camels(camels_dir, False)


@pytest.fixture()
def gages():
    gages_dir = os.path.join(definitions.DATASET_DIR, "gages")
    if not os.path.isfile(
            os.path.join(
                gages_dir,
                "basinchar_and_report_sept_2011",
                "spreadsheets-in-csv-format",
                "conterm_basinid.txt",
            )
    ):
        return Gages(gages_dir, True)
    return Gages(gages_dir, False)


def test1_trans_to_csv_load_to_gis(save_dir):
    basin_id = "01013500"
    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    daily = xr.open_dataset(read_path)

    arr_lat = daily["lat"].values.flatten()
    arr_lon = daily["lon"].values.flatten()
    arr_data = daily["prcp"].values[0, :, :].flatten()

    arr_all = np.c_[arr_lat, arr_lon, arr_data]
    # remove the rows with nan value
    arr = arr_all[~np.isnan(arr_all).any(axis=1)]
    df = pd.DataFrame(data=arr, columns=["lat", "lon", "prcp"])
    df.to_csv(os.path.join(save_dir, "load_to_qgis.csv"), index=False)
    # after getting the csv file, please use "Layer -> Add Layer -> Add Delimited Text Layer" in QGIS to import it.


def test2_which_basin_boundary_out_of_camels(camels, save_dir):
    basin_id = "01013500"
    camels_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = gpd.read_file(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
    geometry = camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
        ].geometry.item()
    gb = geometry.bounds
    gb_west = gb[0]
    gb_south = gb[1]
    gb_east = gb[2]
    gb_north = gb[3]

    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    daily = xr.open_dataset(read_path)

    arr_lat = daily["lat"].values.flatten()
    arr_lon = daily["lon"].values.flatten()
    arr_data = daily["prcp"].values[0, :, :].flatten()

    arr_all = np.c_[arr_lat, arr_lon, arr_data]
    # remove the rows with nan value
    arr = arr_all[~np.isnan(arr_all).any(axis=1)]
    df = pd.DataFrame(data=arr, columns=["lat", "lon", "prcp"])

    df_east = df["lon"].max()
    df_west = df["lon"].min()
    df_north = df["lat"].max()
    df_south = df["lat"].min()
    # if boundary is in the
    assert not (gb_west > df_west)
    assert not (gb_east < df_east)
    assert gb_north < df_north
    assert not (gb_south > df_south)


def test3_trans_to_rectangle(camels, save_dir):
    basin_id = "01013500"
    camels_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = gpd.read_file(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
    geometry = camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
        ].geometry.item()
    save_path = os.path.join(save_dir, basin_id + "_camels.shp")
    camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
        ].geometry.to_file(save_path)

    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_from_urls.nc")
    ds = xr.open_dataset(read_path)
    ds_dims = ("y", "x")
    transform, width, height = geoutils.pygeoutils._get_transform(ds, ds_dims)
    _geometry = geoutils.pygeoutils._geo2polygon(geometry, "epsg:4326", ds.crs)

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
    _mask_bound[y_idx_min: y_idx_max + 1, x_idx_min: x_idx_max + 1] = True

    coords = {ds_dims[0]: ds.coords[ds_dims[0]], ds_dims[1]: ds.coords[ds_dims[1]]}
    mask = xr.DataArray(_mask, coords, dims=ds_dims)
    mask_bound = xr.DataArray(_mask_bound, coords, dims=ds_dims)

    ds_masked = ds.where(mask, drop=True)
    ds_masked.attrs["transform"] = transform
    ds_masked.attrs["bounds"] = _geometry.bounds

    ds_bound_masked = ds.where(mask_bound, drop=True)
    ds_bound_masked.attrs["transform"] = transform
    ds_bound_masked.attrs["bounds"] = _geometry.bounds

    arr_lat = ds_masked["lat"].values.flatten()
    arr_lon = ds_masked["lon"].values.flatten()
    arr_data = ds_masked["prcp"].values[0, :, :].flatten()

    arr_all = np.c_[arr_lat, arr_lon, arr_data]
    # remove the rows with nan value
    arr = arr_all[~np.isnan(arr_all).any(axis=1)]
    df = pd.DataFrame(data=arr, columns=["lat", "lon", "prcp"])
    df.to_csv(os.path.join(save_dir, "geometry_load_to_qgis.csv"), index=False)

    arr_bound_lat = ds_bound_masked["lat"].values.flatten()
    arr_bound_lon = ds_bound_masked["lon"].values.flatten()
    arr_bound_data = ds_bound_masked["prcp"].values[0, :, :].flatten()

    arr_bound_all = np.c_[arr_bound_lat, arr_bound_lon, arr_bound_data]
    # remove the rows with nan value
    arr_bound = arr_bound_all[~np.isnan(arr_bound_all).any(axis=1)]
    df_bound = pd.DataFrame(data=arr_bound, columns=["lat", "lon", "prcp"])
    df_bound.to_csv(os.path.join(save_dir, "bound_load_to_qgis.csv"), index=False)
    # after getting the csv file, please use "Layer -> Add Layer -> Add Delimited Text Layer" in QGIS to import it.


def test4_read_nc_write_boundary(camels, save_dir):
    basin_id = "01013500"
    camels_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = gpd.read_file(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
    geometry = camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
        ].geometry.item()

    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_from_urls.nc")
    ds = xr.open_dataset(read_path)
    ds_masked = generate_boundary_dataset(ds, geometry)

    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_bound.nc")
    ds_masked.to_netcdf(save_path)


def test_resample_nc(save_dir):
    basin_id = "01013500"
    nc_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_bound.nc")
    ds = xr.open_dataset(nc_path)
    ds_high_res = resample_nc(ds, 0.5)
    ds_low_res = resample_nc(ds, 2)
    # the direction of exploration is to the first row (y-axis in this example), so we chose [0, 1, 0]
    np.testing.assert_array_equal(
        ds_high_res["swe"].values[0, 1, 0], ds["swe"].values[0, 0, 0]
    )
    np.testing.assert_array_equal(
        ds_low_res["swe"].values[0, 0, 0], np.mean(ds["swe"].values[0, 0:2, 0:2])
    )


def test_gee_daymet_to_camels_format(camels):
    """
    the example data comes from the code here:
    https://code.earthengine.google.com/1ffc9a50f7749d7be2f67368f465a993
    """
    daymet_dir = "example_data"
    output_dir = os.path.join("test_data", "daymet")
    gage_dict = camels.camels_sites.to_dict(orient="list")
    region = "camels"
    year = 2000
    trans_daymet_to_camels_format(daymet_dir, output_dir, gage_dict, region, year)
    insert_daymet_value_in_leap_year(output_dir, t_range=["2000-01-01", "2000-01-04"])
    print("Trans finished")


def test_gee_daily_nldas_to_camels_format(camels):
    """
    the example data comes from the code here:
    https://code.earthengine.google.com/f62826e26e52996b63ccb3c0ceea3282
    """
    nldas_dir = "example_data"
    output_dir = os.path.join("test_data", "nldas")
    gage_dict = camels.camels_sites.to_dict(orient="list")
    region = "camels"
    year = 2000
    trans_daily_nldas_to_camels_format(nldas_dir, output_dir, gage_dict, region, year)
    print("Trans finished")


def test_gee_8day_pmlv2_to_camels_format(camels):
    pmlv2_dir = "example_data"
    output_dir = os.path.join("test_data", "pmlv2")
    gage_dict = camels.camels_sites.to_dict(orient="list")
    region = "camels"
    year = 2002
    trans_8day_pmlv2_to_camels_format(pmlv2_dir, output_dir, gage_dict, region, year)


def test_gee_8day_modis16a2v105_to_camels_format(camels):
    modis16a2v105_dir = "example_data"
    output_dir = os.path.join("test_data", "modis16a2v105")
    gage_dict = camels.camels_sites.to_dict(orient="list")
    region = "camels"
    year = 2000
    trans_8day_modis16a2v105_to_camels_format(
        modis16a2v105_dir, output_dir, gage_dict, region, year
    )


def test_read_nldas_nc():
    nc_file = os.path.join("example_data", "NLDAS_FORA0125_H.A19790101.1300.020.nc")
    nc4_file = os.path.join(
        "example_data", "NLDAS_FORA0125_H.A19790101.1300.002.grb.SUB.nc4"
    )
    ds1 = xr.open_dataset(nc_file)
    ds2 = xr.open_dataset(nc4_file)
    # data from v002 and v2.0 are same
    np.testing.assert_array_equal(
        np.nansum(ds1["CAPE"].values), np.nansum(ds2["CAPE"].values)
    )


def test_read_era5_land_nc():
    nc_file = os.path.join(
        "example_data", "ERA5_LAND_20010101_20010102_total_precipitation.nc"
    )
    # nc_file = os.path.join("test_data", "a_test_range.nc")
    ds = xr.open_dataset(nc_file)
    print(ds)


def test_gee_daily_era5_land_to_camels_format():
    era5_land_dir = "example_data"
    output_dir = os.path.join("test_data", "era5_land")
    region = "camels_mr"
    year = 2010
    camels_mr_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_mr")
    camels_mr_streamflow_dir = os.path.join(camels_mr_dir, "streamflow")
    flow_files = os.listdir(camels_mr_streamflow_dir)
    gage_dict = pd.DataFrame({"gage_id": np.sort([i[:-4] for i in flow_files])})
    trans_era5_land_to_camels_format(era5_land_dir, output_dir, gage_dict, region, year)
    print("Trans finished")


# For specific use:
# def test_gee_daily_era5_land_to_camels_format_for_china_basins():
#     era5_land_dir = "D:\\data\\DO_CHINA\\ERA5_LAND"
#     output_dir = "D:\\data\\DO_CHINA\\ERA5_LAND_CAMELS_DO_CHINA"
#     region = "camels_cc_do"
#     camels_cc_do_dir = "D:\\data\\DO_CHINA"
#     sites_file = os.path.join(camels_cc_do_dir, "sites_basins.txt")
#     gage_dict = pd.read_csv(sites_file, sep="\t")
#     year_list = np.arange(1981, 2022)
#     for year in year_list:
#         trans_era5_land_to_camels_format(era5_land_dir, output_dir, gage_dict, region, year)
#     print("Trans finished")
#
#
# def test_gee_daily_smap_to_camels_format_for_china_basins():
#     source_dir = "D:\\data\\SMAP10KM_soil_moisture_29"
#     output_dir = "D:\\data\\smap4camels\\NASA_USDA_SMAP_CAMELS_CC"
#     region = "camels_cc"
#     camels_cc_dir = "D:\\data\\camels\\camels_cc"
#     sites_file = os.path.join(camels_cc_dir, "gage_points.csv")
#     gage_dict = pd.read_csv(sites_file, sep=",")
#     year_list = np.arange(2015, 2022)
#     for year in year_list:
#         trans_nasa_usda_smap_to_camels_format(source_dir, output_dir, gage_dict, region, year)
#     print("Trans finished")


def test_gee_monthly_nexdcp30_history_to_camels_format(gages):
    nex_dir = "example_data"
    output_dir = os.path.join("test_data", "nex_dcp30")
    region = "MxWdShld"
    year = 2005
    gage_dict = gages.sites_in_one_region(region)
    trans_month_nex_dcp30to_camels_format(nex_dir, output_dir, gage_dict, region, year)
    print("Trans finished")


def test_gee_monthly_nexdcp30_rcps_to_camels_format(gages):
    nex_dir = "example_data"
    output_dir = os.path.join("test_data", "nex_dcp30")
    region = "MxWdShld"
    year = 2006
    gage_dict = gages.sites_in_one_region(region)
    trans_month_nex_dcp30to_camels_format(nex_dir, output_dir, gage_dict, region, year)
    print("Trans finished")


def test_time_zone_gages_intersect(gages):
    gages_points_shp_file = gages.data_source_description["GAGES_POINT_SHP_FILE"]
    time_zone_shp_file = os.path.join(
        definitions.DATASET_DIR, "Time_Zones", "Time_Zones.shp"
    )
    if not os.path.isfile(time_zone_shp_file):
        raise FileNotFoundError(
            "Please download time zone file from: https://data-usdot.opendata.arcgis.com/datasets/time-zones"
        )
    gage_tz_dict = gage_intersect_time_zone(gages_points_shp_file, time_zone_shp_file)
    serialize_json(gage_tz_dict, os.path.join("test_data", "gage_tz.json"))


def test_split_shp_to_shps_in_time_zones(camels, save_dir):
    basins_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    gage_tz_dict = unserialize_json_ordered(os.path.join("test_data", "gage_tz.json"))
    split_shp_to_shps_in_time_zones(basins_shp_file, gage_tz_dict, save_dir)


def test_tif_basin_mean(camels):
    eta_tif_files = [
        os.path.join(
            "test_data",
            "det2000001.modisSSEBopETactual",
            "det2000001.modisSSEBopETactual.tif",
        ),
        os.path.join(
            "test_data",
            "det2000002.modisSSEBopETactual",
            "det2000002.modisSSEBopETactual.tif",
        ),
    ]
    if not os.path.isfile(eta_tif_files[0]) or not os.path.isfile(eta_tif_files[1]):
        raise FileNotFoundError(
            "Please download time zone file from: https://earlywarning.usgs.gov/ssebop/modis"
        )
    basins_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    print(calculate_tif_data_basin_mean(eta_tif_files, basins_shp_file))


def test_read_camels_streaflow(camels):
    sites_id = np.array(['01594950', '02112120', '02112360', '02125000', '02342933',
                         '02430615', '02464146', '02464360', '03049000', '03238500',
                         '03338780', '03500240', '03592718', '04127918', '04161580',
                         '04233000', '06154410', '06291500', '09035800', '10336740',
                         '12010000', '12147500', '12383500', '12388400'])
    streamflow = camels.read_target_cols(sites_id, ["2014-04-01", "2021-10-01"], ["usgsFlow"])
    print(streamflow)
