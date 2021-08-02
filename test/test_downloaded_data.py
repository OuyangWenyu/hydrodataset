import os
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

import definitions
from hydrobench.data.data_camels import Camels


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
        geometry = camels_shp_epsg4326[camels_shp_epsg4326["hru_id"] == int(basin_id)].geometry[0]
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


if __name__ == '__main__':
    unittest.main()
