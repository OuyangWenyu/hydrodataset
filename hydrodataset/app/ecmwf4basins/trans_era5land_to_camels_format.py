"""
Author: Wenyu Ouyang
Date: 2022-03-19 16:15:00
LastEditTime: 2022-03-19 16:27:48
LastEditors: Wenyu Ouyang
Description: Trans ERA5-LAND data to the format of CAMELS
FilePath: /HydroBench/hydrodataset/app/ecmwf4basins/trans_era5land_to_camels_format.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import os
import sys

from tqdm import tqdm

from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydrodataset.data.data_camels import Camels
from hydrodataset.ecmwf4basins.basin_era5_process import (
    trans_era5_land_to_camels_format,
)


def main(args):
    nldas_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(nldas_dir):
        raise NotADirectoryError(
            "Please download the data manually and unzip it as you wanna!!!"
        )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    region = args.name
    time_zone = args.time_zone
    assert int(args.year_range[0]) < int(args.year_range[1])
    years = list(range(int(args.year_range[0]), int(args.year_range[1])))

    camels = Camels(
        os.path.join(definitions.DATASET_DIR, "camels", "camels_us"), download=False
    )
    gage_dict = camels.camels_sites.to_dict(orient="list")

    for i in tqdm(range(len(years)), leave=False):
        trans_era5_land_to_camels_format(
            nldas_dir, output_dir, gage_dict, region, years[i], time_zone
        )
    print("Trans finished")


# python trans_era5land_to_camels_format.py --input_dir /mnt/sdc/owen/datasets/ERA5_LAND --output_dir /mnt/sdc/owen/datasets/ERA5_LAND_CAMELS --name Camels_Pacific --tz US/Pacific --year_range 1990 2022
# python trans_era5land_to_camels_format.py --input_dir /mnt/sdc/owen/datasets/ERA5_LAND --output_dir /mnt/sdc/owen/datasets/ERA5_LAND_CAMELS --name Camels_Central --tz US/Central --year_range 1990 2022
# python trans_era5land_to_camels_format.py --input_dir /mnt/sdc/owen/datasets/ERA5_LAND --output_dir /mnt/sdc/owen/datasets/ERA5_LAND_CAMELS --name Camels_Eastern --tz US/Eastern --year_range 1990 2022
# python trans_era5land_to_camels_format.py --input_dir /mnt/sdc/owen/datasets/ERA5_LAND --output_dir /mnt/sdc/owen/datasets/ERA5_LAND_CAMELS --name Camels_Mountain --tz US/Mountain --year_range 1990 2022
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trans ERA5-LAND data to CAMELS format"
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        help="The directory of downloaded ERA5-LAND data",
        default="/mnt/sdc/owen/datasets/ERA5_LAND",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="The directory of transformed data",
        default="/mnt/sdc/owen/datasets/ERA5_LAND_CAMELS",
        type=str,
    )
    parser.add_argument(
        "--name",
        dest="name",
        help="All files are named era5_land_xx_avg/sum_mean_xxx, where xx is the NAME and xxx is the year",
        default="Camels_Pacific",
        type=str,
    )
    parser.add_argument(
        "--tz",
        dest="time_zone",
        help="four time zones for US: US/Central, US/Eastern, US/Mountain, US/Pacific",
        default="US/Pacific",
        type=str,
    )
    parser.add_argument(
        "--year_range",
        dest="year_range",
        help="The start and end years (right open interval)",
        default=[1990, 1992],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
