import os
import unittest

from pynhd import NLDI
import xarray as xr
import definitions
import pydaymet as daymet

from src.daymet4basins.basin_daymet_process import download_daymet_by_geom_bound, calculate_basin_grids_pet


class TestDaymet4Basin(unittest.TestCase):
    def setUp(self) -> None:
        save_dir = os.path.join(definitions.ROOT_DIR, "data")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.var = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

    def test_read_daymet_1basin_3days(self):
        basin_id = "01013500"
        dates = ("2000-01-01", "2000-01-03")
        geometry = NLDI().get_basins(basin_id).geometry[0]
        # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        var = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
        daily = daymet.get_bygeom(geometry, dates, variables=var, pet=True)
        save_dir = self.save_dir
        save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03.nc")
        daily.to_netcdf(save_path)

    def test_read_daymet_basins_3days(self):
        basin_id = ["01013500", "01031500"]
        dates = ("2000-01-01", "2000-01-03")
        basins = NLDI().get_basins(basin_id)
        for i in range(len(basin_id)):
            # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
            var = self.var
            daily = daymet.get_bygeom(basins.geometry[i], dates, variables=var)
            save_path = os.path.join(self.save_dir, basin_id[i] + "_2000_01_01-03.nc")
            daily.to_netcdf(save_path)

    def test_download_daymet_without_dask(self):
        basin_id = "01013500"
        dates = ("2000-01-01", "2000-01-03")
        geometry = NLDI().get_basins(basin_id).geometry[0]
        var = self.var
        daily = download_daymet_by_geom_bound(geometry, dates, variables=var)
        save_path = os.path.join(self.save_dir, basin_id + "_2000__01_01-03_nomask.nc")
        daily.to_netcdf(save_path)

    def test_basin_bound_pet_fao56(self):
        basin_id = "01013500"
        read_path = os.path.join(self.save_dir, basin_id + "_2000__01_01-03_nomask.nc")
        daily = xr.open_dataset(read_path)
        include_pet = calculate_basin_grids_pet(daily, pet_method="pm_fao56")
        save_path = os.path.join(self.save_dir, basin_id + "_2000__01_01-03_pet_fao56.nc")
        include_pet.to_netcdf(save_path)

    def test_basin_bound_pet_pt(self):
        basin_id = "01013500"
        read_path = os.path.join(self.save_dir, basin_id + "_2000__01_01-03_nomask.nc")
        daily = xr.open_dataset(read_path)
        include_pet = calculate_basin_grids_pet(daily, pet_method="priestley_taylor")
        save_path = os.path.join(self.save_dir, basin_id + "_2000__01_01-03_pet_pt.nc")
        include_pet.to_netcdf(save_path)
