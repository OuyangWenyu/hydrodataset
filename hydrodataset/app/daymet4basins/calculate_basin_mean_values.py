"""
calculate basin mean forcings
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrodataset.data.data_camels import Camels
from hydrodataset.utils.hydro_utils import unserialize_geopandas, hydro_logger, t_range_days
from hydrodataset.daymet4basins.basin_daymet_process import calculate_basin_mean


def main(args):
    hydro_logger.info("Start Calculating:\n")
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"), download=True)
    camels_shp_file = camels.dataset_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = unserialize_geopandas(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    basins = camels_shp.to_crs(epsg=4326)
    assert (all(x < y for x, y in zip(basins["hru_id"].values, basins["hru_id"].values[1:])))
    basins_id = camels.camels_sites["gauge_id"].values.tolist()

    camels_index = ['Year', 'Mnth', 'Day', 'Hr', 'dayl(s/day)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(kg/m2)', 'tmax(C)',
                    'tmin(C)', 'vp(Pa)']
    var = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]

    dir1 = os.path.join(definitions.DATASET_DIR, "daymet4camels", "daymet_camels_671_unmask")
    dir2 = os.path.join(definitions.DATASET_DIR, "daymet4basins", "daymet_camels_671_unmask")
    # same name with CAMELS
    save_dir = os.path.join(definitions.DATASET_DIR, "daymet4camels", "basin_mean_forcing", "daymet")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.year_range is not None:
        assert int(args.year_range[0]) < int(args.year_range[1])
        years = list(range(int(args.year_range[0]), int(args.year_range[1])))
    else:
        raise NotImplementedError("Please enter the time range (Start year and end year)")
    for i in tqdm(range(len(basins_id))):
        gage_id_df = camels.camels_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == basins_id[i]]["huc_02"].values[0]
        huc02dir = os.path.join(save_dir, huc)
        if not os.path.isdir(huc02dir):
            os.makedirs(huc02dir)
        save_file_i = os.path.join(huc02dir, basins_id[i] + "_lump_cida_forcing_leap.txt")
        frames_basin = []
        for j in tqdm(range(len(years)), leave=False):
            nc_path = os.path.join(dir1, basins_id[i], basins_id[i] + "_" + str(years[j]) + "_nomask.nc")
            if not os.path.isfile(nc_path):
                nc_path = os.path.join(dir2, basins_id[i], basins_id[i] + "_" + str(years[j]) + "_nomask.nc")
                if not os.path.isfile(nc_path):
                    raise FileNotFoundError("This file has not been downloaded.")
            ds = xr.open_dataset(nc_path)
            daily_mean = calculate_basin_mean(ds, basins.geometry[i])
            df = daily_mean.to_dataframe()
            # interpolation for the 12.31 data in leap year
            t_range = [str(years[j]) + "-01-01", str(years[j] + 1) + "-01-01"]
            t_range_list = t_range_days(t_range)
            [c, ind1, ind2] = np.intersect1d(
                pd.to_datetime([str(dt.year) + "-" + str(dt.month) + "-" + str(dt.day) for dt in df.index]),
                t_range_list, return_indices=True)
            nt = t_range_list.size
            out = np.full([nt, len(var)], np.nan)
            out[ind2, :] = df[var].values[ind1]
            x = pd.DataFrame(out, columns=camels_index[4:])
            csv_date = pd.to_datetime(t_range_list)
            year_month_day_hour = pd.DataFrame(
                [[dt.year, dt.month, dt.day, 12] for dt in csv_date], columns=camels_index[0:4])
            # concat
            new_data_df = pd.concat([year_month_day_hour, x], axis=1)
            frames_basin.append(new_data_df)

        df_i = pd.concat(frames_basin)
        df_i_intepolate = df_i.interpolate(method='linear', limit_direction='forward', axis=0)
        df_i_intepolate.to_csv(save_file_i, header=True, index=False, sep=' ', float_format='%.2f')

    hydro_logger.info("\n Finished!")


# python calculate_basin_mean_values.py --year_range 1990 2010
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate basin mean forcings.')
    parser.add_argument('--year_range', dest='year_range', help='The start and end years (right open interval)',
                        default=[1990, 1992], nargs='+')
    the_args = parser.parse_args()
    main(the_args)
