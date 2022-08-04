import os
import pytest
import geopandas as gpd
from pynhd import NLDI
import xarray as xr
import pydaymet as daymet

import definitions
from hydrodataset.climateproj4basins.download_cmip6 import NexGddpCmip6
from hydrodataset.data.data_camels import Camels
from hydrodataset.data.data_gages import read_usgs_daily_flow
from hydrodataset.daymet4basins.basin_daymet_process import (
    download_daymet_by_geom_bound,
    calculate_basin_grids_pet,
    calculate_basin_mean,
)
from hydrodataset.utils.hydro_utils import unserialize_geopandas
from hydrodataset.ecmwf4basins.download_era5_land import download_era5
from hydrodataset.nldas4basins.download_nldas import download_nldas_with_url_lst


@pytest.fixture()
def save_dir():
    save_dir_ = os.path.join(definitions.ROOT_DIR, "test", "test_data")
    if not os.path.isdir(save_dir_):
        os.makedirs(save_dir_)
    return save_dir_


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


def test_read_daymet_1basin_3days(save_dir):
    basin_id = "01013500"
    dates = ("2000-01-01", "2000-01-03")
    geometry = NLDI().get_basins(basin_id).geometry[0]
    # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
    var = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
    daily = daymet.get_bygeom(geometry, dates, variables=var, pet=True)
    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03.nc")
    daily.to_netcdf(save_path)


def test_read_daymet_1basin_in_camels_2days(camels, save_dir):
    basin_id = "01013500"
    dates = ("2000-01-01", "2000-01-02")
    camels_shp_file = camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = gpd.read_file(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
    geometry = camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
    ].geometry[0]
    # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
    var = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
    daily = daymet.get_bygeom(geometry, dates, variables=var, pet=True)
    save_path = os.path.join(save_dir, basin_id + "_in_camels_2000_01_01-02.nc")
    daily.to_netcdf(save_path)


def test_read_daymet_basins_3days(var, save_dir):
    basin_id = ["01013500", "01031500"]
    dates = ("2000-01-01", "2000-01-03")
    basins = NLDI().get_basins(basin_id)
    for i in range(len(basin_id)):
        # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        daily = daymet.get_bygeom(basins.geometry[i], dates, variables=var)
        save_path = os.path.join(save_dir, basin_id[i] + "_2000_01_01-03.nc")
        daily.to_netcdf(save_path)


def test_download_nldi_shpfile(save_dir):
    basin_id = "01013500"
    basin = NLDI().get_basins(basin_id)
    # geometry = basin.geometry[0]
    save_path = os.path.join(save_dir, basin_id + ".shp")
    basin.to_file(save_path)


def test_download_daymet_without_dask(var, save_dir):
    basin_id = "01013500"
    dates = ("2000-01-01", "2000-01-03")
    geometry = NLDI().get_basins(basin_id).geometry[0]
    daily = download_daymet_by_geom_bound(geometry, dates, variables=var)
    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    daily.to_netcdf(save_path)


def test_download_daymet_without_dask_local_shpfile(save_dir, var):
    # Notice the projection, otherwise it will occur many NaN values since the locations are wrong
    basin_id = "01013500"
    dates = ("2000-01-01", "2000-01-03")
    basin_shp_dir = os.path.join(definitions.ROOT_DIR, "test", "test_data")
    basin_shp_file = os.path.join(basin_shp_dir, "01013500.shp")
    if not os.path.isfile(basin_shp_file):
        basin = NLDI().get_basins(basin_id)
        save_path = os.path.join(save_dir, basin_id + ".shp")
        basin.to_file(save_path)
    basins = unserialize_geopandas(basin_shp_file)
    geometry = basins.geometry[0]
    daily = download_daymet_by_geom_bound(geometry, dates, variables=var)
    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask_local_shp.nc")
    daily.to_netcdf(save_path)


def test_equal_local_shp_download_shp_nc(save_dir):
    basin_id = "01013500"
    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    read_path_local = os.path.join(
        save_dir, basin_id + "_2000_01_01-03_nomask_local_shp.nc"
    )
    daily = xr.open_dataset(read_path)
    daily_local = xr.open_dataset(read_path_local)
    print(daily.equals(daily_local))


def test_download_from_url_directly(var, save_dir):
    basin_id = "01013500"
    dates = ("2000-01-01", "2000-01-03")
    geometry = NLDI().get_basins(basin_id).geometry[0]
    daily = download_daymet_by_geom_bound(
        geometry, dates, variables=var, boundary=False
    )
    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_from_urls.nc")
    daily.to_netcdf(save_path)


def test_basin_bound_pet_fao56(save_dir):
    basin_id = "01013500"
    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    daily = xr.open_dataset(read_path)
    include_pet = calculate_basin_grids_pet(daily, pet_method="pm_fao56")
    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_pet_fao56.nc")
    include_pet.to_netcdf(save_path)


def test_basin_bound_pet_pt(save_dir):
    basin_id = "01013500"
    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    daily = xr.open_dataset(read_path)
    include_pet = calculate_basin_grids_pet(daily, pet_method="priestley_taylor")
    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_pet_pt.nc")
    include_pet.to_netcdf(save_path)


def test_basin_mean(save_dir):
    basin_id = "01013500"
    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_pet_pt.nc")
    daily = xr.open_dataset(read_path)
    basins = NLDI().get_basins(basin_id)
    mean_daily = calculate_basin_mean(daily, basins.geometry[0])
    print(mean_daily)


def test_batch_download_nldi_shpfile(save_dir):
    basin_id = ["01013500", "01031500"]
    basins = NLDI().get_basins(basin_id)
    # geometry = basin.geometry[0]
    save_path = os.path.join(save_dir, "two_test_basins.shp")
    basins.to_file(save_path)


def test_batch_download_daymet_without_dask(save_dir, var):
    basins_id = ["01013500", "01031500"]
    dates = ("2000-01-01", "2000-01-03")
    basins = NLDI().get_basins(basins_id)
    save_dir = save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for i in range(len(basins_id)):
        # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        daily = download_daymet_by_geom_bound(basins.geometry[i], dates, variables=var)
        save_path = os.path.join(save_dir, basins_id[i] + "_2000_01_01-03_nomask.nc")
        daily.to_netcdf(save_path)


def test_batch_basins_pet(save_dir):
    basin_id = ["01013500", "01031500"]
    for i in range(len(basin_id)):
        read_path = os.path.join(save_dir, basin_id[i] + "_2000_01_01-03_nomask.nc")
        daily = xr.open_dataset(read_path)
        include_pet = calculate_basin_grids_pet(
            daily, pet_method=["pm_fao56", "priestley_taylor"]
        )
        save_path = os.path.join(save_dir, basin_id[i] + "_2000_01_01-03_pet.nc")
        include_pet.to_netcdf(save_path)


def test_batch_basins_mean(save_dir):
    basins_id = ["01013500", "01031500"]
    basins = NLDI().get_basins(basins_id)
    for i in range(len(basins_id)):
        read_path = os.path.join(save_dir, basins_id[i] + "_2000_01_01-03_pet.nc")
        daily = xr.open_dataset(read_path)
        mean_daily = calculate_basin_mean(daily, basins.geometry[i])
        print(mean_daily)
        save_path = os.path.join(save_dir, basins_id[i] + "_2000_01_01-03_mean.nc")
        mean_daily.to_netcdf(save_path)


def test_download_era5(save_dir):
    downloaded_file = os.path.join(save_dir, "a_test_range.nc")
    date_range = ["2000-01-01", "2000-01-03"]
    lat_lon_range = (31, 108, 30, 109)  # lat_max, lon_min, lat_min, lon_max
    variables_list = "total_precipitation"
    download_era5(
        downloaded_file, date_range, lat_lon_range, variables_list, file_format="netcdf"
    )
    print("Downloading ERA5 hourly data is finished!")


def test_download_nldas_hourly():
    download_lst_dir = os.path.join(definitions.ROOT_DIR, "hydrobench", "nldas4basins")
    save_dir = os.path.join(definitions.DATASET_DIR, "nldas_hourly")
    for file in os.listdir(download_lst_dir):
        if "NLDAS" in file and ".txt" in file:
            url_lst_file = os.path.join(
                definitions.ROOT_DIR, "hydrobench", "nldas4basins", file
            )
            download_nldas_with_url_lst(url_lst_file, save_dir)
    print("Downloading NLDAS hourly data is finished!")


def test_download_usgs_streamflow(camels):
    sites_id = camels.read_object_ids().tolist()
    date_range = ("2020-10-01", "2021-10-01")
    gage_dict = camels.camels_sites
    save_dir = os.path.join("test_data", "camels_streamflow_2021")
    unit = "cfs"
    qobs = read_usgs_daily_flow(sites_id, date_range, gage_dict, save_dir, unit)
    print(qobs)


def test_download_cmip6():
    gcm = "ACCESS-CM2"
    scenario = "ssp585"
    year = 2015
    var = "tasmin"
    north = 51
    west = 234
    east = 294
    south = 23
    cmip6 = NexGddpCmip6()
    save_dir = os.path.join(definitions.DATASET_DIR, "NEX-GDDP-CMIP6")

    cmip6.download_one_nex_gddp_cmip6_file_for_a_region(
        gcm, scenario, year, var, north, east, south, west, save_dir
    )
