"""
Read nc files as xarray's dataset, then get masked dataset according to the bound of geometry
"""

import argparse
import os
import sys
import xarray as xr
from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrodataset.data.data_camels import Camels
from hydrodataset.utils.hydro_utils import unserialize_geopandas, hydro_logger
from hydrodataset.daymet4basins.basin_daymet_process import generate_boundary_dataset, calculate_basin_grids_pet


def main(args):
    hydro_logger.info("Start Cutting:\n")
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"), download=True)
    camels_shp_file = camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = unserialize_geopandas(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    basins = camels_shp.to_crs(epsg=4326)
    assert (all(x < y for x, y in zip(basins["hru_id"].values, basins["hru_id"].values[1:])))
    basins_id = camels.camels_sites["gauge_id"].values.tolist()
    dir1 = os.path.join(definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_unmask")
    dir2 = os.path.join(definitions.DATASET_DIR, "daymet4basins", "daymet_camels_671_unmask")
    # same name with CAMELS
    save_dir = os.path.join(definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_bound")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.year_range is not None:
        assert int(args.year_range[0]) < int(args.year_range[1])
        years = list(range(int(args.year_range[0]), int(args.year_range[1])))
    else:
        raise NotImplementedError("Please enter the time range (Start year and end year)")
    for i in tqdm(range(len(basins_id))):
        save_one_basin_dir = os.path.join(save_dir, basins_id[i])
        if not os.path.isdir(save_one_basin_dir):
            os.makedirs(save_one_basin_dir)
        for j in tqdm(range(len(years)), leave=False):
            save_path = os.path.join(save_one_basin_dir, basins_id[i] + "_" + str(years[j]) + "_boundary.nc")
            nc_path = os.path.join(dir1, basins_id[i], basins_id[i] + "_" + str(years[j]) + "_nomask.nc")
            if not os.path.isfile(nc_path):
                nc_path = os.path.join(dir2, basins_id[i], basins_id[i] + "_" + str(years[j]) + "_nomask.nc")
                if not os.path.isfile(nc_path):
                    raise FileNotFoundError("This file has not been downloaded.")
            ds = xr.open_dataset(nc_path)
            ds_bound = generate_boundary_dataset(ds, basins.geometry[i])
            ds_bound_with_pet = calculate_basin_grids_pet(ds_bound, ["priestley_taylor", "pm_fao56"])
            ds_bound_with_pet.to_netcdf(save_path)

    hydro_logger.info("\n Finished!")


# python cut_nc_to_bounds_add_pet.py --year_range 1990 2010
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut basin forcings in the boundary.')
    parser.add_argument('--year_range', dest='year_range', help='The start and end years (right open interval)',
                        default=[1990, 1992], nargs='+')
    the_args = parser.parse_args()
    main(the_args)
