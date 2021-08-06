import os
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio.features as rio_features
import pygeoutils as geoutils

import definitions
from hydrobench.data.data_camels import Camels
from hydrobench.daymet4basins.basin_daymet_process import generate_boundary_dataset, resample_nc


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


if __name__ == '__main__':
    unittest.main()
