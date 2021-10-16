import os
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio.features as rio_features
import pygeoutils as geoutils
import cfgrib
import definitions
from hydrobench.data.data_camels import Camels
from hydrobench.daymet4basins.basin_daymet_process import generate_boundary_dataset, resample_nc, \
    trans_daymet_to_camels_format, insert_daymet_value_in_leap_year
from hydrobench.modis4basins.basin_mod16a2v105_process import trans_8day_modis16a2v105_to_camels_format
from hydrobench.modis4basins.basin_pmlv2_process import trans_8day_pmlv2_to_camels_format
from hydrobench.nldas4basins.basin_nldas_process import trans_daily_nldas_to_camels_format


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        save_dir = os.path.join(definitions.ROOT_DIR, "test", "test_data")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.var = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
        camels_dir = os.path.join(definitions.DATASET_DIR, "camels")
        self.camels = Camels(camels_dir, True)

    def test1_trans_to_csv_load_to_gis(self):
        basin_id = "01013500"
        read_path = os.path.join(self.save_dir, basin_id + "_2000_01_01-03_nomask.nc")
        daily = xr.open_dataset(read_path)

        arr_lat = daily['lat'].values.flatten()
        arr_lon = daily['lon'].values.flatten()
        arr_data = daily['prcp'].values[0, :, :].flatten()

        arr_all = np.c_[arr_lat, arr_lon, arr_data]
        # remove the rows with nan value
        arr = arr_all[~np.isnan(arr_all).any(axis=1)]
        df = pd.DataFrame(data=arr, columns=['lat', 'lon', 'prcp'])
        df.to_csv(os.path.join(self.save_dir, 'load_to_qgis.csv'), index=False)
        # after getting the csv file, please use "Layer -> Add Layer -> Add Delimited Text Layer" in QGIS to import it.

    def test2_which_basin_boundary_out_of_camels(self):
        basin_id = "01013500"
        camels_shp_file = self.camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
        camels_shp = gpd.read_file(camels_shp_file)
        # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
        camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
        geometry = camels_shp_epsg4326[camels_shp_epsg4326["hru_id"] == int(basin_id)].geometry.item()
        gb = geometry.bounds
        gb_west = gb[0]
        gb_south = gb[1]
        gb_east = gb[2]
        gb_north = gb[3]

        read_path = os.path.join(self.save_dir, basin_id + "_2000_01_01-03_nomask.nc")
        daily = xr.open_dataset(read_path)

        arr_lat = daily['lat'].values.flatten()
        arr_lon = daily['lon'].values.flatten()
        arr_data = daily['prcp'].values[0, :, :].flatten()

        arr_all = np.c_[arr_lat, arr_lon, arr_data]
        # remove the rows with nan value
        arr = arr_all[~np.isnan(arr_all).any(axis=1)]
        df = pd.DataFrame(data=arr, columns=['lat', 'lon', 'prcp'])

        df_east = df["lon"].max()
        df_west = df["lon"].min()
        df_north = df["lat"].max()
        df_south = df["lat"].min()
        # if boundary is in the
        print(gb_west > df_west)
        print(gb_east < df_east)
        print(gb_north < df_north)
        print(gb_south > df_south)

    def test3_trans_to_rectangle(self):
        basin_id = "01013500"
        camels_shp_file = self.camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
        camels_shp = gpd.read_file(camels_shp_file)
        # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
        camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
        geometry = camels_shp_epsg4326[camels_shp_epsg4326["hru_id"] == int(basin_id)].geometry.item()
        save_path = os.path.join(self.save_dir, basin_id + "_camels.shp")
        camels_shp_epsg4326[camels_shp_epsg4326["hru_id"] == int(basin_id)].geometry.to_file(save_path)

        read_path = os.path.join(self.save_dir, basin_id + "_2000_01_01-03_from_urls.nc")
        ds = xr.open_dataset(read_path)
        ds_dims = ("y", "x")
        transform, width, height = geoutils.pygeoutils._get_transform(ds, ds_dims)
        _geometry = geoutils.pygeoutils._geo2polygon(geometry, "epsg:4326", ds.crs)

        _mask = rio_features.geometry_mask([_geometry], (height, width), transform, invert=True)
        # x - column, y - row
        y_idx, x_idx = np.where(_mask)
        y_idx_min = y_idx.min()
        y_idx_max = y_idx.max()
        x_idx_min = x_idx.min()
        x_idx_max = x_idx.max()
        _mask_bound = np.full(_mask.shape, False)
        _mask_bound[y_idx_min:y_idx_max + 1, x_idx_min:x_idx_max + 1] = True

        coords = {ds_dims[0]: ds.coords[ds_dims[0]], ds_dims[1]: ds.coords[ds_dims[1]]}
        mask = xr.DataArray(_mask, coords, dims=ds_dims)
        mask_bound = xr.DataArray(_mask_bound, coords, dims=ds_dims)

        ds_masked = ds.where(mask, drop=True)
        ds_masked.attrs["transform"] = transform
        ds_masked.attrs["bounds"] = _geometry.bounds

        ds_bound_masked = ds.where(mask_bound, drop=True)
        ds_bound_masked.attrs["transform"] = transform
        ds_bound_masked.attrs["bounds"] = _geometry.bounds

        arr_lat = ds_masked['lat'].values.flatten()
        arr_lon = ds_masked['lon'].values.flatten()
        arr_data = ds_masked['prcp'].values[0, :, :].flatten()

        arr_all = np.c_[arr_lat, arr_lon, arr_data]
        # remove the rows with nan value
        arr = arr_all[~np.isnan(arr_all).any(axis=1)]
        df = pd.DataFrame(data=arr, columns=['lat', 'lon', 'prcp'])
        df.to_csv(os.path.join(self.save_dir, 'geometry_load_to_qgis.csv'), index=False)

        arr_bound_lat = ds_bound_masked['lat'].values.flatten()
        arr_bound_lon = ds_bound_masked['lon'].values.flatten()
        arr_bound_data = ds_bound_masked['prcp'].values[0, :, :].flatten()

        arr_bound_all = np.c_[arr_bound_lat, arr_bound_lon, arr_bound_data]
        # remove the rows with nan value
        arr_bound = arr_bound_all[~np.isnan(arr_bound_all).any(axis=1)]
        df_bound = pd.DataFrame(data=arr_bound, columns=['lat', 'lon', 'prcp'])
        df_bound.to_csv(os.path.join(self.save_dir, 'bound_load_to_qgis.csv'), index=False)
        # after getting the csv file, please use "Layer -> Add Layer -> Add Delimited Text Layer" in QGIS to import it.

    def test4_read_nc_write_boundary(self):
        basin_id = "01013500"
        camels_shp_file = self.camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
        camels_shp = gpd.read_file(camels_shp_file)
        # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
        camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
        geometry = camels_shp_epsg4326[camels_shp_epsg4326["hru_id"] == int(basin_id)].geometry.item()

        read_path = os.path.join(self.save_dir, basin_id + "_2000_01_01-03_from_urls.nc")
        ds = xr.open_dataset(read_path)
        ds_masked = generate_boundary_dataset(ds, geometry)

        save_path = os.path.join(self.save_dir, basin_id + "_2000_01_01-03_bound.nc")
        ds_masked.to_netcdf(save_path)

    def test_resample_nc(self):
        basin_id = "01013500"
        nc_path = os.path.join(self.save_dir, basin_id + "_2000_01_01-03_bound.nc")
        ds = xr.open_dataset(nc_path)
        ds_high_res = resample_nc(ds, 0.5)
        ds_low_res = resample_nc(ds, 2)
        # the direction of exploration is to the first row (y-axis in this example), so we chose [0, 1, 0]
        self.assertAlmostEqual(ds_high_res["swe"].values[0, 1, 0].item(), ds["swe"].values[0, 0, 0].item())
        self.assertAlmostEqual(ds_low_res["swe"].values[0, 0, 0].item(), np.mean(ds["swe"].values[0, 0:2, 0:2]))

    def test_gee_daymet_to_camels_format(self):
        """
        the example data comes from the code here:
        https://code.earthengine.google.com/1ffc9a50f7749d7be2f67368f465a993
        """
        daymet_dir = "example_data"
        output_dir = os.path.join("test_data", "daymet")
        camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
        gage_dict = camels.camels_sites.to_dict(orient="list")
        region = "camels"
        year = 2000
        trans_daymet_to_camels_format(daymet_dir, output_dir, gage_dict, region, year)
        insert_daymet_value_in_leap_year(output_dir, t_range=["2000-01-01", "2000-01-04"])
        print("Trans finished")

    def test_gee_daily_nldas_to_camels_format(self):
        """
        the example data comes from the code here:
        https://code.earthengine.google.com/f62826e26e52996b63ccb3c0ceea3282
        """
        nldas_dir = "example_data"
        output_dir = os.path.join("test_data", "nldas")
        camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
        gage_dict = camels.camels_sites.to_dict(orient="list")
        region = "camels"
        year = 2000
        trans_daily_nldas_to_camels_format(nldas_dir, output_dir, gage_dict, region, year)
        print("Trans finished")

    def test_gee_8day_pmlv2_to_camels_format(self):
        pmlv2_dir = "example_data"
        output_dir = os.path.join("test_data", "pmlv2")
        camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
        gage_dict = camels.camels_sites.to_dict(orient="list")
        region = "camels"
        year = 2002
        trans_8day_pmlv2_to_camels_format(pmlv2_dir, output_dir, gage_dict, region, year)
        print()

    def test_gee_8day_modis16a2v105_to_camels_format(self):
        modis16a2v105_dir = "example_data"
        output_dir = os.path.join("test_data", "modis16a2v105")
        camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
        gage_dict = camels.camels_sites.to_dict(orient="list")
        region = "camels"
        year = 2000
        trans_8day_modis16a2v105_to_camels_format(modis16a2v105_dir, output_dir, gage_dict, region, year)
        print()

    def test_read_nldas_nc(self):
        nc_file = os.path.join("example_data", "NLDAS_FORA0125_H.A19790101.1300.020.nc")
        nc4_file = os.path.join("example_data", "NLDAS_FORA0125_H.A19790101.1300.002.grb.SUB.nc4")
        ds1 = xr.open_dataset(nc_file)
        ds2 = xr.open_dataset(nc4_file)
        # data from v002 and v2.0 are same
        self.assertEqual(np.nansum(ds1["CAPE"].values), np.nansum(ds2["CAPE"].values))

    def test_read_era5_land_nc(self):
        nc_file = os.path.join("example_data", "ERA5_LAND_20010101_20010102_total_precipitation.nc")
        # nc_file = os.path.join("test_data", "a_test_range.nc")
        ds = xr.open_dataset(nc_file)
        print(ds)


if __name__ == '__main__':
    unittest.main()
