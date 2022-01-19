"""
Download daymet grid data for the boundaries of basins in CAMELS
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrodataset.data.data_camels import Camels
from hydrodataset.daymet4basins.basin_daymet_process import (
    download_daymet_by_geom_bound,
)
from hydrodataset.utils.hydro_utils import unserialize_geopandas, hydro_logger


def main(args):
    hydro_logger.info("Start Downloading:\n")
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    camels_shp_file = camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = unserialize_geopandas(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    basins = camels_shp.to_crs(epsg=4326)
    assert all(
        x < y for x, y in zip(basins["hru_id"].values, basins["hru_id"].values[1:])
    )
    basins_id = camels.camels_sites["gauge_id"].values.tolist()
    var = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
    save_dir = os.path.join(
        definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_unmask"
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.year_range is not None:
        assert int(args.year_range[0]) < int(args.year_range[1])
        years = list(range(int(args.year_range[0]), int(args.year_range[1])))
    else:
        raise NotImplementedError(
            "Please enter the time range (Start year and end year)"
        )
    for i in tqdm(range(len(basins_id))):
        save_one_basin_dir = os.path.join(save_dir, basins_id[i])
        if not os.path.isdir(save_one_basin_dir):
            os.makedirs(save_one_basin_dir)
        for j in tqdm(range(len(years)), leave=False):
            dates = (str(years[j]) + "-01-01", str(years[j]) + "-12-31")
            save_path = os.path.join(
                save_one_basin_dir, basins_id[i] + "_" + str(years[j]) + "_nomask.nc"
            )
            if os.path.isfile(save_path):
                hydro_logger.info("This file has been downloaded.")
                continue
            daily = download_daymet_by_geom_bound(
                basins.geometry[i], dates, variables=var, boundary=False
            )
            daily.to_netcdf(save_path)
    hydro_logger.info("\n Finished!")


# python download_daymet_camels_basin.py --year_range 1990 1991
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Daymet within the boundary of each basin in CAMELS"
    )
    parser.add_argument(
        "--year_range",
        dest="year_range",
        help="The start and end years (right open interval)",
        default=[1990, 1991],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
