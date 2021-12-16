"""
regrid all ncs to the assigned resolution
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
from hydrodataset.daymet4basins.basin_daymet_process import resample_nc


def main(args):
    hydro_logger.info("Start Regriding:\n")
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    camels_shp_file = camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = unserialize_geopandas(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    basins = camels_shp.to_crs(epsg=4326)
    assert (all(x < y for x, y in zip(basins["hru_id"].values, basins["hru_id"].values[1:])))
    basins_id = camels.camels_sites["gauge_id"].values.tolist()

    if args.dir is not None:
        dir_name = args.dir
    else:
        raise NotImplementedError("Please enter the directory of initial data")
    if dir_name == "daymet_camels_671_unmask":
        dir1 = os.path.join(definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_unmask")
        dir2 = os.path.join(definitions.DATASET_DIR, "daymet4basins", "daymet_camels_671_unmask")
    elif dir_name == "daymet_camels_671_bound":
        dir1 = os.path.join(definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_bound")
        dir2 = os.path.join(definitions.DATASET_DIR, "daymet4basins", "daymet_camels_671_bound")
    else:
        raise NotADirectoryError("We don't have such a directory yet")

    # same name with CAMELS
    save_dir = os.path.join(definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_bound_resample")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.year_range is not None:
        assert int(args.year_range[0]) < int(args.year_range[1])
        years = list(range(int(args.year_range[0]), int(args.year_range[1])))
    else:
        raise NotImplementedError("Please enter the time range (Start year and end year)")
    if args.rs is not None:
        resample_size = int(args.rs)
    else:
        raise NotImplementedError("Please enter the resample size")
    for i in tqdm(range(len(basins_id))):
        save_one_basin_dir = os.path.join(save_dir, basins_id[i])
        if not os.path.isdir(save_one_basin_dir):
            os.makedirs(save_one_basin_dir)
        for j in tqdm(range(len(years)), leave=False):
            save_path = os.path.join(save_one_basin_dir,
                                     basins_id[i] + "_" + str(years[j]) + "_resample_" + str(resample_size) + ".nc")

            if dir_name == "daymet_camels_671_unmask":
                file_name = "_nomask.nc"
            elif dir_name == "daymet_camels_671_bound":
                file_name = "_boundary.nc"
            else:
                raise FileNotFoundError("We don't have such a file yet")

            nc_path = os.path.join(dir1, basins_id[i], basins_id[i] + "_" + str(years[j]) + file_name)
            if not os.path.isfile(nc_path):
                nc_path = os.path.join(dir2, basins_id[i], basins_id[i] + "_" + str(years[j]) + file_name)
                if not os.path.isfile(nc_path):
                    raise FileNotFoundError("This file has not been downloaded.")
            ds = xr.open_dataset(nc_path)
            ds_bound = resample_nc(ds, resample_size)
            ds_bound.to_netcdf(save_path)

    hydro_logger.info("\n Finished!")


# python regrid_daymet_nc.py --year_range 1990 2010 --rs 10 --dir daymet_camels_671_bound
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regrid basin forcings.')
    parser.add_argument('--year_range', dest='year_range', help='The start and end years (right open interval)',
                        default=[1990, 1992], nargs='+')
    parser.add_argument('--rs', dest='rs', help='resample size', default=10, type=int)
    parser.add_argument('--dir', dest='dir', help='the directory of original data', default="daymet_camels_671_bound",
                        type=str)
    the_args = parser.parse_args()
    main(the_args)
