"""
Download the daymet data for camels-basins outside of the boundary of downloaded xarray dataset using nldi shapfile
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrobench.data.data_camels import Camels
from hydrobench.daymet4basins.basin_daymet_process import download_daymet_by_geom_bound
from hydrobench.utils.hydro_utils import unserialize_geopandas, hydro_logger


def do_we_need_redownload(geometry, basin_id, previous_download_data_dir, download_year):
    """
    See if we need redownload the daymet data for a camels basin

    If the boundary of the downloaded data cover the bound of the camels-basin' geometry,
    we don't need to download the dataset again, and just copy the data files to a given directory

    :param geometry: the camels-basin's geometry (epsg:4326)
    :param basin_id: the id of a basin
    :param previous_download_data_dir: the directory of previously downloaded daymet data
    :param download_year: the downloaded data's year
    :return: if we need download again, return True, otherwise, return False
    """
    gb = geometry.bounds
    gb_west = gb[0]
    gb_south = gb[1]
    gb_east = gb[2]
    gb_north = gb[3]

    read_path = os.path.join(previous_download_data_dir, basin_id, basin_id + "_" + str(download_year) + "_nomask.nc")
    if not os.path.isfile(read_path):
        return True
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

    is_not_need = (gb_west >= df_west) and (gb_east <= df_east) and (gb_north <= df_north) and (gb_south >= df_south)
    return not is_not_need


def main(args):
    hydro_logger.info("Start Downloading:\n")
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
    camels_shp_file = camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = unserialize_geopandas(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    basins = camels_shp.to_crs(epsg=4326)
    assert (all(x < y for x, y in zip(basins["hru_id"].values, basins["hru_id"].values[1:])))
    basins_id = camels.camels_sites["gauge_id"].values.tolist()
    var = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    previous_save_dir = os.path.join(definitions.DATASET_DIR, "daymet4basins", "daymet_camels_671_unmask")
    save_dir = os.path.join(definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_unmask")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.year_range is not None:
        assert int(args.year_range[0]) < int(args.year_range[1])
        years = list(range(int(args.year_range[0]), int(args.year_range[1])))
    else:
        raise NotImplementedError("Please enter the time range (Start year and end year)")
    # it seems lead to a breakdown of the file system to use shutil.copy for too many times,
    # so we don't copy files, just select which parts should be downloaded.
    need_download_index = OrderedDict()
    for i in range(len(basins_id)):
        need_download_year_index_lst = []
        save_one_basin_dir = os.path.join(save_dir, basins_id[i])
        if not os.path.isdir(save_one_basin_dir):
            os.makedirs(save_one_basin_dir)
        for j in range(len(years)):
            save_path = os.path.join(save_one_basin_dir, basins_id[i] + "_" + str(years[j]) + "_nomask.nc")
            if os.path.isfile(save_path) or (
                    not do_we_need_redownload(basins.geometry[i], basins_id[i], previous_save_dir, years[j])):
                print(save_path + " has been downloaded.")
            else:
                need_download_year_index_lst.append(j)
        need_download_index[basins_id[i]] = need_download_year_index_lst
    #  Download data
    for i in tqdm(range(len(basins_id))):
        save_one_basin_dir = os.path.join(save_dir, basins_id[i])
        for j in tqdm(range(len(need_download_index[basins_id[i]])), leave=False):
            dates = (str(years[j]) + "-01-01", str(years[j]) + "-12-31")
            save_path = os.path.join(save_one_basin_dir, basins_id[i] + "_" + str(years[j]) + "_nomask.nc")
            if os.path.isfile(save_path):
                hydro_logger.info(save_path + " has been downloaded.")
            else:
                # download from url directly, no mask of geometry or geometry's boundary
                daily = download_daymet_by_geom_bound(basins.geometry[i], dates, variables=var, boundary=False)
                daily.to_netcdf(save_path)

    hydro_logger.info("\n Finished!")


# python download_daymet_camels_outside_nldi.py --year_range 1990 1991
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Daymet within the boundary of each basin in CAMELS')
    parser.add_argument('--year_range', dest='year_range', help='The start and end years (right open interval)',
                        default=[1990, 1991], nargs='+')
    the_args = parser.parse_args()
    main(the_args)
