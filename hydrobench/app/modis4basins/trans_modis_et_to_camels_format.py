"""
Transform the data format of PMLV2 to the camels'
"""
import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrobench.data.data_camels import Camels
from hydrobench.modis4basins.basin_pmlv2_process import trans_8day_pmlv2_to_camels_format
from hydrobench.modis4basins.basin_mod16a2v105_process import trans_8day_modis16a2v105_to_camels_format


def main(args):
    dataset_name = args.dataset_name
    modis_et_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(modis_et_dir):
        raise NotADirectoryError("Please download the data manually and unzip it as you wanna!!!")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    assert int(args.year_range[0]) < int(args.year_range[1])
    years = list(range(int(args.year_range[0]), int(args.year_range[1])))

    region = "camels"
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"), download=True)
    gage_dict = camels.camels_sites.to_dict(orient="list")

    for i in tqdm(range(len(years)), leave=False):
        if dataset_name == "PML_V2":
            trans_8day_pmlv2_to_camels_format(modis_et_dir, output_dir, gage_dict, region, years[i])
        elif dataset_name == "MOD16A2_105":
            trans_8day_modis16a2v105_to_camels_format(modis_et_dir, output_dir, gage_dict, region, years[i])
        else:
            raise FileNotFoundError("No such data! Please check if you have chosen correctly. "
                                    "We only provide PML_V2 and MOD16A2_105 now!")

    print("Trans finished")


# python trans_modis_et_to_camels_format.py --dataset_name PML_V2 --input_dir /mnt/sdc/owen/datasets/PML_V2 --output_dir /mnt/sdc/owen/datasets/PML_V2_CAMELS --year_range 2002 2018
# python trans_modis_et_to_camels_format.py --dataset_name MOD16A2_105 --input_dir /mnt/sdc/owen/datasets/MOD16A2_105 --output_dir /mnt/sdc/owen/datasets/MOD16A2_105_CAMELS --year_range 2000 2015
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trans MODIS ET data of each basin to CAMELS format')
    parser.add_argument('--dataset_name', dest='dataset_name', help='The downloaded ET data',
                        default="PML_V2", type=str)
    parser.add_argument('--input_dir', dest='input_dir', help='The directory of downloaded ET data',
                        default="/mnt/sdc/owen/datasets/PML_V2", type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='The directory of transformed data',
                        default="/mnt/sdc/owen/datasets/PML_V2_CAMELS", type=str)
    parser.add_argument('--year_range', dest='year_range', help='The start and end years (right open interval)',
                        default=[2002, 2005], nargs='+')
    the_args = parser.parse_args()
    main(the_args)
