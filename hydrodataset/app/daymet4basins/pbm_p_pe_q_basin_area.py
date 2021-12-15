"""
Get P (precipitation), PE (potential evapotranspiration), Q (streamflow) and Basin areas for some physics-based models
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrodataset.utils.hydro_utils import hydro_logger, t_range_days
from hydrodataset.data.data_daymet4camels import Daymet4Camels


def main(args):
    hydro_logger.info("Start Reading and Writing:\n")
    camels = Daymet4Camels(os.path.join(definitions.DATASET_DIR, "daymet4camels"),
                           os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    basins_id = camels.camels671_sites["gauge_id"].values.tolist()
    # notice the unit of streamflow is cubic feet per second
    pbm_index = ['date', 'prcp(mm/day)', 'petpt(mm/day)', 'petfao56(mm/day)', 'streamflow(ft3/s)', 'area(km2)']
    # all_var = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp", "pet_pt", "pet_fao56"]
    chosen_var = ["prcp", "petpt", "petfao56"]
    chosen_const = ["area_gages2"]

    # same name with CAMELS
    save_dir = os.path.join(definitions.DATASET_DIR, "daymet4camels", "pbm_daily_dataset")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.year_range is not None:
        assert int(args.year_range[0]) < int(args.year_range[1])
        years = args.year_range
    else:
        raise NotImplementedError("Please enter the time range (Start year and end year)")
    t_range = [str(years[0]) + "-01-01", str(years[-1]) + "-01-01"]
    t_range_list = t_range_days(t_range)
    csv_date = pd.DataFrame(np.array(pd.to_datetime(t_range_list)).T, columns=pbm_index[0:1])

    attrs = camels.read_constant_cols(basins_id, chosen_const)
    c = pd.DataFrame(attrs, columns=pbm_index[-1:])
    attr_df = pd.concat([camels.camels671_sites, c], axis=1)
    save_file_attr = os.path.join(save_dir, "camels_basins_areas.txt")
    attr_df.to_csv(save_file_attr, header=True, index=False)

    mean_forcings = camels.read_relevant_cols(basins_id, t_range, chosen_var, resample=0)
    streamflows = camels.read_target_cols(basins_id, t_range)
    for i in tqdm(range(len(basins_id))):
        x = pd.DataFrame(mean_forcings[i, :, :], columns=pbm_index[1:4])
        y = pd.DataFrame(streamflows[i:(i + 1), :].T, columns=pbm_index[4:5])
        new_data_df = pd.concat([csv_date, x, y], axis=1)
        save_file_i = os.path.join(save_dir, basins_id[i] + "_lump_p_pe_q.txt")
        new_data_df.to_csv(save_file_i, header=True, index=False)

    hydro_logger.info("\n Finished!")


# python pbm_p_pe_q_basin_area.py --year_range 1990 2010
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get basin areas and basin mean values including P, PET, and Q')
    parser.add_argument('--year_range', dest='year_range', help='The start and end years (right open interval)',
                        default=[1990, 1992], nargs='+')
    the_args = parser.parse_args()
    main(the_args)
