"""
Author: Wenyu Ouyang
Date: 2021-12-16 14:31:21
LastEditTime: 2022-01-19 00:52:18
LastEditors: Wenyu Ouyang
Description: Transform the data format of PMLV2, MOD16A2_105 and SSEBop to the camels'
FilePath: /HydroBench/hydrodataset/app/modis4basins/trans_modis_et_to_camels_format.py
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
from hydrodataset.modis4basins.basin_pmlv2_process import (
    trans_8day_pmlv2_to_camels_format,
)
from hydrodataset.modis4basins.basin_mod16a2v105_process import (
    trans_8day_modis16a2v105_to_camels_format,
)
from hydrodataset.modis4basins.basin_mod_ssebop_daily_eta_process import (
    calculate_tif_data_basin_mean,
)
from hydrodataset.utils.hydro_utils import unzip_nested_zip, serialize_numpy


def main(args):
    dataset_name = args.dataset_name
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

    region = "camels"
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    gage_dict = camels.camels_sites.to_dict(orient="list")

    for i in tqdm(range(len(years)), leave=False):
        if dataset_name == "PML_V2":
            trans_8day_pmlv2_to_camels_format(
                modis_et_dir, output_dir, gage_dict, region, years[i]
            )
        elif dataset_name == "MOD16A2_105":
            trans_8day_modis16a2v105_to_camels_format(
                modis_et_dir, output_dir, gage_dict, region, years[i]
            )
        elif dataset_name == "SSEBop_ETa":
            # TODO: now save 671 basins' ETa data to npy file
            f_name = str(years[i]) + ".zip"
            unzip_dir = os.path.join(modis_et_dir, f_name[0:-4])
            file_name = os.path.join(modis_et_dir, f_name)
            unzip_nested_zip(file_name, unzip_dir)
            if calendar.isleap(years[i]):
                days = 366
            else:
                days = 365
            eta_tif_dirs = [
                "det2000" + str(i).zfill(3) + ".modisSSEBopETactual"
                for i in range(days)
            ]
            eta_tif_files = [
                os.path.join(unzip_dir, eta_dir, eta_dir + ".tif")
                for eta_dir in eta_tif_dirs
            ]
            eta_year_i = calculate_tif_data_basin_mean(
                eta_tif_files, camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
            )
            npy_file = os.path.join(output_dir, str(years[i]) + ".npy")
            serialize_numpy(eta_year_i, npy_file)
        else:
            raise FileNotFoundError(
                "No such data! Please check if you have chosen correctly. "
                "We only provide PML_V2 and MOD16A2_105 now!"
            )

    print("Trans finished")


# python trans_modis_et_to_camels_format.py --dataset_name PML_V2 --input_dir /mnt/sdc/owen/datasets/PML_V2 --output_dir /mnt/sdc/owen/datasets/PML_V2_CAMELS --year_range 2002 2018
# python trans_modis_et_to_camels_format.py --dataset_name MOD16A2_105 --input_dir /mnt/sdc/owen/datasets/MOD16A2_105 --output_dir /mnt/sdc/owen/datasets/MOD16A2_105_CAMELS --year_range 2000 2015
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trans MODIS ET data of each basin to CAMELS format"
    )
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        help="The downloaded ET data",
        default="PML_V2",
        type=str,
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        help="The directory of downloaded ET data",
        default="/mnt/sdc/owen/datasets/PML_V2",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="The directory of transformed data",
        default="/mnt/sdc/owen/datasets/PML_V2_CAMELS",
        type=str,
    )
    parser.add_argument(
        "--year_range",
        dest="year_range",
        help="The start and end years (right open interval)",
        default=[2002, 2005],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
