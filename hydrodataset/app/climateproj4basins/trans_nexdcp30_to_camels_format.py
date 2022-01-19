import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrodataset.climateproj4basins.basin_nexdcp30_process import (
    trans_month_nex_dcp30to_camels_format,
)
from hydrodataset.data.data_gages import Gages


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(
            "Please download the data manually and unzip it as you wanna!!!"
        )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    assert int(args.year_range[0]) < int(args.year_range[1])
    years = list(range(int(args.year_range[0]), int(args.year_range[1])))

    regions = [
        "allref",
        "CntlPlains",
        "EastHghlnds",
        "MxWdShld",
        "NorthEast",
        "SECstPlain",
        "SEPlains",
        "WestMnts",
        "WestPlains",
        "WestXeric",
    ]
    gages = Gages(os.path.join(definitions.DATASET_DIR, "gages"))

    for i in tqdm(range(len(years)), leave=False):
        for region in regions:
            gage_dict = gages.sites_in_one_region(region)
            trans_month_nex_dcp30to_camels_format(
                input_dir, output_dir, gage_dict, region, years[i]
            )

    print("Trans finished")


# python trans_nexdcp30_to_camels_format.py --input_dir D:\\data\\NEX-DCP30 --output_dir D:\\data\\NEX-DCP30-CAMELS --year_range 2004 2008
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trans NEX-DCP30 data of each basin to CAMELS format"
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        help="The directory of downloaded NEX-DCP30 data",
        default="D:\\data\\NEX-DCP30",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="The directory of transformed data",
        default="D:\\data\\NEX-DCP30-CAMELS",
        type=str,
    )
    parser.add_argument(
        "--year_range",
        dest="year_range",
        help="The start and end years (right open interval)",
        default=[2004, 2008],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
