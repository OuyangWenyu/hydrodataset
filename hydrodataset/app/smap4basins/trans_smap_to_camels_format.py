"""
Author: Wenyu Ouyang
Date: 2022-01-18 23:30:01
LastEditTime: 2022-03-19 16:39:25
LastEditors: Wenyu Ouyang
Description: Trans SMAP-USDA 3-day data to CAMELS format
FilePath: /HydroBench/hydrodataset/app/smap4basins/trans_smap_to_camels_format.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import os
import sys
import calendar
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydrodataset.data.data_camels import Camels
from hydrodataset.smap4basins.basin_smap_process import (
    trans_nasa_usda_smap_to_camels_format,
)
from hydrodataset.utils.hydro_utils import unzip_nested_zip


def main(args):
    dataset_name = args.dataset_name
    smap_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(smap_dir):
        raise NotADirectoryError(
            "Please download the data manually and unzip it as you wanna!!!"
        )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    assert int(args.year_range[0]) < int(args.year_range[1])
    years = list(range(int(args.year_range[0]), int(args.year_range[1])))

    region = "camels"
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    gage_dict = camels.camels_sites.to_dict(orient="list")

    for i in tqdm(range(len(years)), leave=False):
        if dataset_name == "NASA_USDA_SMAP":
            trans_nasa_usda_smap_to_camels_format(
                smap_dir, output_dir, gage_dict, region, years[i]
            )
        else:
            raise FileNotFoundError(
                "No such data! Please check if you have chosen correctly. "
                "We only provide PML_V2 and MOD16A2_105 now!"
            )

    print("Trans finished")


# python trans_smap_to_camels_format.py --dataset_name NASA_USDA_SMAP --input_dir /mnt/sdc/owen/datasets/SMAP10KM --output_dir /mnt/sdc/owen/datasets/NASA_USDA_SMAP_CAMELS --year_range 2015 2022
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trans SMAP data of each basin to CAMELS format"
    )
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        help="The downloaded SMAP data",
        default="NASA_USDA_SMAP",
        type=str,
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        help="The directory of downloaded SMAP data",
        default="/mnt/sdc/owen/datasets/SMAP10KM",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="The directory of transformed data",
        default="/mnt/sdc/owen/datasets/NASA_USDA_SMAP_CAMELS",
        type=str,
    )
    parser.add_argument(
        "--year_range",
        dest="year_range",
        help="The start and end years (right open interval)",
        default=[2015, 2022],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
