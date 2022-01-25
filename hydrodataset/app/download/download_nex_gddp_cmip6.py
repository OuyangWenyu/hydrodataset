import argparse
import os
import sys

sys.path.append(os.path.join("..", "..", ".."))
import definitions
from hydrodataset.utils.hydro_utils import hydro_logger
from hydrodataset.climateproj4basins.download_cmip6 import NexGddpCmip6


def main(args):
    save_dir = args.save_dir
    gcm = args.gcm
    scenario = args.sce
    north = args.north
    west = args.west
    east = args.east
    south = args.south
    cmip6 = NexGddpCmip6()
    cmip6.download_one_gcm_scenario_nex_gddp_cmip6_for_a_region(
        gcm, scenario, north, east, south, west, save_dir
    )


# python download_nex_gddp_cmip6.py --gcm ACCESS-CM2 --sce ssp585 --north 51 --east 294 --south 23 --west 234 --save_dir /mnt/sdc/owen/datasets/NEX-GDDP-CMIP6
if __name__ == "__main__":
    hydro_logger.info("Download CMIP6 data!")
    parser = argparse.ArgumentParser(description="Download CMIP6 data")
    parser.add_argument(
        "--gcm",
        dest="gcm",
        help="The GCM",
        default="ACCESS-CM2",
        type=str,
    )
    parser.add_argument(
        "--sce",
        dest="sce",
        help="The scenario",
        default="ssp585",
        type=str,
    )
    parser.add_argument(
        "--north",
        dest="north",
        help="The north latitude",
        default=51,
        type=int,
    )
    parser.add_argument(
        "--east",
        dest="east",
        help="The east longitude",
        default=294,
        type=int,
    )
    parser.add_argument(
        "--south",
        dest="south",
        help="The south latitude",
        default=23,
        type=int,
    )
    parser.add_argument(
        "--west",
        dest="west",
        help="The west longitude",
        default=234,
        type=int,
    )
    default_save_dir = os.path.join(definitions.DATASET_DIR, "NEX-GDDP-CMIP6")
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="The directory of downloaded data",
        default=default_save_dir,
        type=str,
    )
    the_args = parser.parse_args()
    main(the_args)
