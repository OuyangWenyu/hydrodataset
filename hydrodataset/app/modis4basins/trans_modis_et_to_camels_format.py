"""
Author: Wenyu Ouyang
Date: 2022-01-25 16:49:00
LastEditTime: 2022-01-27 09:40:16
LastEditors: Wenyu Ouyang
Description: Transform the data format of PMLV2, MOD16A2_105, and MOD16A2_006 to the camels'
FilePath: /HydroBench/hydrodataset/app/modis4basins/trans_modis_et_to_camels_format.py
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
from hydrodataset.modis4basins.basin_pmlv2_process import (
    trans_8day_pmlv2_to_camels_format,
)
from hydrodataset.modis4basins.basin_mod16a2v105_process import (
    trans_8day_modis16a2v105_to_camels_format,
)
from hydrodataset.modis4basins.basin_mod16a2v006_process import (
    trans_8day_modis16a2v006_to_camels_format,
)


def main(args):
    dataset_name = args.dataset_name
    camels_name = args.camels_name
    camels_region = args.camels_region
    modis_et_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(modis_et_dir):
        raise NotADirectoryError(
            "Please download the data manually and unzip it as you wanna!!!"
        )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    assert int(args.year_range[0]) < int(args.year_range[1])
    years = list(range(int(args.year_range[0]), int(args.year_range[1])))

    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", camels_region), region=camels_name)
    gage_dict = camels.camels_sites.to_dict(orient="list")

    for i in tqdm(range(len(years)), leave=False):
        if dataset_name == "PML_V2":
            trans_8day_pmlv2_to_camels_format(
                modis_et_dir, output_dir, gage_dict, camels_region, years[i]
            )
        elif dataset_name == "MOD16A2_105":
            trans_8day_modis16a2v105_to_camels_format(
                modis_et_dir, output_dir, gage_dict, camels_region, years[i]
            )
        elif dataset_name == "MOD16A2_006":
            trans_8day_modis16a2v006_to_camels_format(
                modis_et_dir, output_dir, gage_dict, camels_region, years[i]
            )
        else:
            raise FileNotFoundError(
                "No such data! Please check if you have chosen correctly. "
                "We only provide PML_V2 and MOD16A2_105 now!"
            )

    print("Trans finished")


# python trans_modis_et_to_camels_format.py --dataset_name PML_V2 --input_dir /mnt/sdc/owen/datasets/PML_V2 --output_dir /mnt/sdc/owen/datasets/PML_V2_CAMELS --year_range 2002 2018
# python trans_modis_et_to_camels_format.py --dataset_name MOD16A2_105 --input_dir /mnt/sdc/owen/datasets/MOD16A2_105 --output_dir /mnt/sdc/owen/datasets/MOD16A2_105_CAMELS --year_range 2000 2015
# python trans_modis_et_to_camels_format.py --dataset_name MOD16A2_006 --input_dir /mnt/sdc/owen/datasets/MOD16A2_105 --output_dir /mnt/sdc/owen/datasets/MOD16A2_105_CAMELS --year_range 2000 2015
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trans MODIS ET data of each basin to CAMELS format"
    )
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        help="The downloaded ET data",
        default="MOD16A2_006",
        type=str,
    )
    parser.add_argument(
        "--camels_name",
        dest="camels_name",
        help="name of the region: US/CC/AUS/...",
        default="CC",
        type=str,
    )
    parser.add_argument(
        "--camels_region",
        dest="camels_region",
        help="the region",
        default="camels_cc",
        type=str,
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        help="The directory of downloaded ET data",
        default="D:/data/MOD16A2_006_CC",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="The directory of transformed data",
        default="D:/data/modiset4camels/basin_mean_forcing/MOD16A2_006_CAMELS_CC",
        type=str,
    )
    parser.add_argument(
        "--year_range",
        dest="year_range",
        help="The start and end years (right open interval)",
        default=[2001, 2022],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
