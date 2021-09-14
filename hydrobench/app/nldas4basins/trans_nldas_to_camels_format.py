"""
Transform the data format of NLDAS to the camels'
"""
import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrobench.data.data_camels import Camels
from hydrobench.nldas4basins.basin_nldas_process import trans_daily_nldas_to_camels_format


def main(args):
    nldas_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(nldas_dir):
        raise NotADirectoryError("Please download the data manually and unzip it as you wanna!!!")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    region = args.name
    gage_file = args.gage_file
    sta_id_str = args.staid_str
    huc_str = args.huc02_str
    assert int(args.year_range[0]) < int(args.year_range[1])
    years = list(range(int(args.year_range[0]), int(args.year_range[1])))

    if region == "camels":
        camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
        gage_dict = camels.camels_sites.to_dict(orient="list")
    else:
        if not os.path.isfile(gage_file):
            raise FileNotFoundError("Please give it a gage_dict file")
        else:
            gage_dict = pd.read_csv(gage_file, dtype={sta_id_str: str, huc_str: str})
    for i in tqdm(range(len(years)), leave=False):
        trans_daily_nldas_to_camels_format(nldas_dir, output_dir, gage_dict, region, years[i])
    print("Trans finished")


# python trans_nldas_to_camels_format.py --input_dir /mnt/sdc/owen/datasets/NLDAS --output_dir /mnt/sdc/owen/datasets/NLDAS_DO --name do --gage_file /mnt/data/owen411/code/HydroBench/test/example_data/site_nobs_DO.csv --staid_str STAID --huc02_str HUC02 --year_range 1980 2021
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Daymet within the boundary of each basin in CAMELS')
    parser.add_argument('--input_dir', dest='input_dir', help='The directory of downloaded NLDAS data',
                        default="/mnt/sdc/owen/datasets/NLDAS", type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='The directory of transformed data',
                        default="/mnt/sdc/owen/datasets/NLDAS_DO", type=str)
    parser.add_argument('--name', dest='name',
                        help='All files are named nldas_xx_avg/sum_mean_xxx, where xx is the NAME and xxx is the year',
                        default="do", type=str)
    parser.add_argument('--gage_file', dest='gage_file', help='The dict of gages',
                        default="/mnt/data/owen411/code/HydroBench/test/example_data/site_nobs_DO.csv", type=str)
    parser.add_argument('--staid_str', dest='staid_str',
                        help='The name of gage id column, e.g. gauge_id, STAID',
                        default="STAID", type=str)
    parser.add_argument('--huc02_str', dest='huc02_str',
                        help='The name of huc02 column, e.g. huc02, HUC02',
                        default="HUC02", type=str)
    parser.add_argument('--year_range', dest='year_range', help='The start and end years (right open interval)',
                        default=[1980, 1982], nargs='+')
    the_args = parser.parse_args()
    main(the_args)
